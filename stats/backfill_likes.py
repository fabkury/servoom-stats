"""One-time backfill of like lists for artworks uploaded in the 12 months before the
pulse log began (design: servoom/docs/community-network/README.md).

Interruptible and resumable at every step:

* The work list (public artworks of the window with at least one like, by GalleryId) is
  fixed on the first run and saved on the ``state-backfill`` branch, so later runs read
  the same list even though the catalog moves on.
* Rows are written in files of CHUNK artworks, ``obs/likes-backfill/<first>-<last>.parquet``,
  each pushed to ``main`` before the checkpoint ``{next_index, ...}`` is force-pushed to
  ``state-backfill``. A kill between the two re-reads at most one chunk. A re-read of a
  full chunk overwrites the same file name; a re-read after a partial flush writes a
  second file overlapping it, so readers deduplicate by (gid, liker), newest ``t`` wins.
* SIGTERM/SIGINT, a wall-clock budget (``MAX_MINUTES``) and the API circuit breaker all
  end the loop at the next artwork; the partial chunk is flushed and checkpointed.
* Once ``finished_at`` is set, every later run exits at once.

Read-only commands only. Rows: (gid, pos, liker, t, auto), ``pos`` 1-based newest-first.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from . import rawrepo
from .accounts import Halted, Pool, flush_issues
from .api import Api, ApiDown
from .util import DAY, is_auto, iso, load_ranges, now, read_parquet, write_parquet

BRANCH = "state-backfill"
CHUNK = int(os.environ.get("BACKFILL_CHUNK", "2000"))
WINDOW_DAYS = int(os.environ.get("BACKFILL_WINDOW_DAYS", "365"))
PAGE = 100
MAX_PAGES = 60                      # 6,000 likes; the window's maximum is about 1,700

_stop: List[str] = []


def _on_signal(signum, frame) -> None:
    _stop.append(signal.Signals(signum).name)
    print(f"[backfill] {signal.Signals(signum).name} received; finishing the current artworks", flush=True)


def work_list(st: Path, t: int) -> pd.DataFrame:
    """The fixed list of (gid, like) to read, created on the first run from the catalog."""
    wl = read_parquet(st / "worklist.parquet")
    if wl is not None:
        return wl
    snap = rawrepo.clone_state("state-snap")
    cat = read_parquet(snap / "catalog.parquet", columns=["gid", "date", "like", "gone"])
    if cat is None:
        raise RuntimeError("no catalog on state-snap; run a snapshot first")
    first_log = int(os.environ.get("BACKFILL_WINDOW_END", "0")) or t
    lo = first_log - WINDOW_DAYS * DAY
    wl = cat[(cat.gone == 0) & (cat.like > 0) & (cat.date >= lo) & (cat.date < first_log)]
    wl = wl[["gid", "like", "date"]].sort_values("gid").reset_index(drop=True).astype("int64")
    write_parquet(wl, st / "worklist.parquet")
    print(f"[backfill] work list: {len(wl)} artworks dated {iso(lo)} to {iso(first_log)}, "
          f"{int(((wl.like + PAGE - 1) // PAGE).sum())} pages expected", flush=True)
    return wl


def read_one(api: Api, gid: int) -> List[Tuple[int, int, int]]:
    """Every (gid, pos, liker) of one artwork; raises ApiDown when the token is refused."""
    out: List[Tuple[int, int, int]] = []
    start = 1
    for _ in range(MAX_PAGES):
        r = api.post("Cloud/GetLikeUserList", {"GalleryId": gid, "StartNum": start, "EndNum": start + PAGE - 1})
        code = r.get("ReturnCode")
        if code == 11:
            raise ApiDown("token refused (ReturnCode 11)")
        if code not in (0, 1):
            raise ApiDown(f"GetLikeUserList answered {code} {r.get('ReturnMessage')}")
        ul = r.get("UserList") or []
        out.extend((gid, start + k, int(u["UserId"])) for k, u in enumerate(ul))
        if len(ul) < PAGE:
            break
        start += PAGE
    return out


def flush(rows: List[Tuple[int, int, int]], gids: List[int], main: Path, t: int, ranges) -> None:
    if not gids:
        return
    df = pd.DataFrame(rows, columns=["gid", "pos", "liker"]).astype("int64")
    df["t"] = t
    df["auto"] = is_auto(df.liker, ranges).astype("int8") if len(df) else pd.Series([], dtype="int8")
    name = f"{gids[0]}-{gids[-1]}.parquet"
    tmp = main / "obs" / "likes-backfill" / (name + ".tmp")
    write_parquet(df, tmp)
    tmp.replace(tmp.with_name(name))             # atomic on the same filesystem
    rawrepo.push_main(f"backfill likes {gids[0]}-{gids[-1]}")


def run() -> None:
    t0 = now()
    budget = float(os.environ.get("MAX_MINUTES", "235")) * 60
    st = rawrepo.clone_state(BRANCH)
    ckf = st / "checkpoint.json"
    ck: Dict = json.loads(ckf.read_text()) if ckf.exists() else {"next_index": 0, "done": 0, "requests": 0, "runs": 0}
    if ck.get("finished_at"):
        print(f"[backfill] finished on {iso(ck['finished_at'])}; nothing to do")
        return
    main = rawrepo.clone_main(["state", "obs/likes-backfill"])
    wl = work_list(st, t0)
    if not ck.get("started_at"):
        ck["started_at"] = t0
        rawrepo.push_state(BRANCH, "backfill: work list")

    api = Api(rps=float(os.environ.get("BACKFILL_RPS", "8")), workers=int(os.environ.get("BACKFILL_WORKERS", "8")))
    try:
        Pool(main).acquire(api, "backfill")
    except Halted as exc:
        print(f"[backfill] halted: {exc}")
        rawrepo.push_main("backfill: halted")
        flush_issues(Path("_issues"))
        return
    rawrepo.push_main("backfill: account state")
    ranges = load_ranges()
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    i, n = int(ck["next_index"]), len(wl)
    ck["runs"] = ck.get("runs", 0) + 1
    print(f"[backfill] run {ck['runs']}: resuming at {i}/{n}, budget {budget/60:.0f} min, {api.interval and 1/api.interval:.0f} rps", flush=True)
    rows: List[Tuple[int, int, int]] = []
    gids: List[int] = []
    reason = "done"
    batch = max(8, api.workers * 4)
    while i < n:
        if _stop:
            reason = _stop[0]
            break
        if now() - t0 > budget:
            reason = "budget"
            break
        todo = [int(g) for g in wl.gid.iloc[i:i + batch]]
        try:
            parts = api.pmap(lambda g: read_one(api, g), todo)
        except ApiDown as exc:
            reason = f"api: {exc}"
            break
        for g, part in zip(todo, parts):
            rows.extend(part)
            gids.append(g)
        i += len(todo)
        if len(gids) >= CHUNK:
            flush(rows, gids, main, now(), ranges)
            ck.update(next_index=i, done=i, requests=ck.get("requests", 0) + api.n_requests, last_t=now())
            api.n_requests = 0
            ckf.write_text(json.dumps(ck))
            rawrepo.push_state(BRANCH, f"backfill checkpoint {i}/{n}")
            print(f"[backfill] {i}/{n} artworks, {len(rows)} rows in this chunk, {(now()-t0)/60:.0f} min", flush=True)
            rows, gids = [], []
    flush(rows, gids, main, now(), ranges)
    ck.update(next_index=i, done=i, requests=ck.get("requests", 0) + api.n_requests, last_t=now(), last_reason=reason)
    if i >= n:
        ck["finished_at"] = now()
    ckf.write_text(json.dumps(ck))
    rawrepo.push_state(BRANCH, f"backfill checkpoint {i}/{n} ({reason})")
    print(f"[backfill] stopped: {reason}; {i}/{n} artworks done, {ck['requests']} requests so far"
          + (", FINISHED" if i >= n else ""), flush=True)
    flush_issues(Path("_issues"))
    if reason.startswith("api"):
        sys.exit(1)


if __name__ == "__main__":
    run()
