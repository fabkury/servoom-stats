"""Daily probe: the slow, whole-gallery readings the hourly pulse can no longer afford.

Since 2026-10-23 Divoom serves one source 30 requests a minute, so the pulse's hour
holds only its own window reads. This job runs once a day on its own runner (its own
budget) and leaves its results on the branch ``state-probe``, which the next pulse
takes over (``pulse.apply_probe``):

* ``lists.json``: every (category, size) list read with a long look-ahead over the
  30-day window, so that lists with window artworks the pulse does not yet know, and
  lists that carry a block of old artworks after the newest ones, are found;
* ``categories.json``: the total items per category (``refresh.poll_categories``);
* ``held.parquet``: the outcome of uploads that were held for review
  (``gid, chk_final, t_res``).

It reuses the snapshot account's token; without one it reads anonymously (shallower
lists, no held follow-up).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

from . import rawrepo, refresh
from .accounts import Halted, Pool
from .api import Api, ApiDown, OutOfTime
from .util import ALL_SIZES, CATEGORIES, DAY, env_flag, now, read_parquet, write_parquet

WINDOW_DAYS = int(os.environ.get("PULSE_DAYS", "30"))
BUDGET = int(os.environ.get("PROBE_BUDGET_SECONDS", str(170 * 60)))
MIN_GAP = 20 * 3600


def survey_lists(api: Api, st: Path, t: int, cutoff: int) -> bool:
    f = st / "lists.json"
    known = json.loads(f.read_text()) if f.exists() else {}
    complete = True
    for cls in CATEGORIES:
        for size in ALL_SIZES:
            key = f"{cls}_{size}"
            try:
                fl = api.crawl_list(cls, size, stop_before=cutoff, old_streak=16)
            except OutOfTime:
                complete = False
                break
            n = sum(1 for x in fl if x["Date"] >= cutoff)
            if n:
                known[key] = {"n": n, "t": t, "block": api.last_gap}
            elif key in known and t - known[key]["t"] > 40 * DAY:
                del known[key]
        if not complete:
            break
    known["_t"] = t
    f.write_text(json.dumps(known))
    return complete


def follow_up_held(api: Api, st: Path, sp: Path, t: int) -> None:
    """Re-read uploads the pulse saw held for review, to record the outcome."""
    w = read_parquet(sp / "walk.parquet")
    if w is None or not len(w):
        return
    done = read_parquet(st / "held.parquet")
    seen = set(done.gid) if done is not None else set()
    todo = w[(w.chk == 1) & w.t_res.isna() & (w.priv == 0) & (w.t > t - 8 * DAY) & ~w.gid.isin(seen)].gid.tolist()[:400]
    rows = []
    for gid, r in zip(todo, api.pmap(api.gallery_info, todo)):
        if r.get("ReturnCode") != 0:
            rows.append((gid, -1, t))
        elif int(r.get("CheckConfirm") or 0) != 1:
            rows.append((gid, int(r["CheckConfirm"]), t))
    new = pd.DataFrame(rows, columns=["gid", "chk_final", "t_res"])
    out = pd.concat(([done] if done is not None else []) + [new], ignore_index=True)
    write_parquet(out[out.t_res > t - 30 * DAY], st / "held.parquet")


def run() -> None:
    t = now()
    st = rawrepo.clone_state("state-probe")
    lf = st / "lists.json"
    last = json.loads(lf.read_text()).get("_t", 0) if lf.exists() else 0
    if t - last < MIN_GAP and not env_flag("FORCE_PROBE"):
        print(f"[probe] the last probe was {(t - last) // 3600} hours ago; nothing to do")
        return
    main = rawrepo.clone_main(["state"])
    sp = rawrepo.clone_state("state-pulse")
    api = Api(workers=4)
    api.set_budget(BUDGET)
    authed = True
    try:
        Pool(main).token_for(api, "snapshot")
    except Halted as exc:
        print(f"[probe] running without a token: {exc}")
        authed = False
    cut = []
    complete = survey_lists(api, st, t, t - WINDOW_DAYS * DAY)
    if not complete:
        cut.append("lists")
    for name, fn in (("categories", lambda: refresh.poll_categories(api, st, t) if authed else None),
                     ("held", lambda: follow_up_held(api, st, sp, t) if authed else None)):
        try:
            fn()
        except OutOfTime:
            cut.append(name)
        except ApiDown as exc:
            print(f"[probe] {name} skipped: {exc}")
    rawrepo.push_state("state-probe", f"probe {time.strftime('%Y-%m-%d %H:%M', time.gmtime(t))}")
    print(f"[probe] done in {now() - t}s, {api.n_served} answers, {api.n_limited} refusals"
          + (f", cut: {', '.join(cut)}" if cut else ""))


if __name__ == "__main__":
    try:
        run()
    except ApiDown as exc:
        print(f"[probe] server unreachable, ending quietly: {exc}")
        sys.exit(0)
