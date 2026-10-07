"""Daily artist like-network: who likes whose work, for the /artists/ pages.

Design: servoom/docs/community-network/README.md. Inputs are the like event log (pulse,
snapshot and backfill files in the raw repository) and the snapshot's state tables. No
Divoom request is made. Output goes to ``data/network/``: a status file, the monthly
history, an index of featured artists and one file per featured artist.

Naming rule: only accounts in ``data/daily/artists/index.json`` (the stats site's top
artists) appear with an id. Every other artist is counted, never listed.

Decided 2026-10-07: no community detection and no map. A first version (Leiden + UMAP)
produced groups that were arbitrary, mostly nameless and not useful to browse.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from . import rawrepo
from .util import CONFIG, DATA, DAY, iso, is_auto, load_ranges, now, read_parquet, write_json

SUB = "network"
WINDOW = 365 * DAY
TOP_N = 25                      # neighbours listed per featured artist


def log(msg: str) -> None:
    print(f"[network] {msg}", flush=True)


def load_events(main: Path, cat: pd.DataFrame, t: int) -> pd.DataFrame:
    """Human likes as (gid, liker, uid, time, approx), one row per (gid, liker)."""
    files = sorted((main / "obs" / "likes-backfill").glob("*.parquet"))
    files += sorted(p for p in (main / "obs").rglob("*likes*.parquet") if "likes-backfill" not in p.as_posix())
    parts = []
    for p in files:
        d = pd.read_parquet(p, columns=["gid", "liker", "t"])
        d["bf"] = "likes-backfill" in p.as_posix()
        parts.append(d)
    if not parts:
        raise RuntimeError("no like files in the raw repository")
    ev = pd.concat(parts, ignore_index=True)
    n_raw = len(ev)
    ev = ev.sort_values("t").drop_duplicates(["gid", "liker"], keep="last")
    ev = ev[~is_auto(ev.liker, load_ranges())]
    ev = ev.merge(cat[["gid", "uid", "date"]], on="gid", how="inner")
    ev = ev[ev.liker != ev.uid]
    # Backfilled likes have no time of their own: date them by the upload.
    ev["time"] = np.where(ev.bf, ev.date, ev.t).astype("int64")
    ev["approx"] = ev.bf.astype("int8")
    ev = ev[ev.time >= t - WINDOW - 31 * DAY]
    log(f"events: {n_raw} rows read, {len(ev)} human likes on known artworks after dedup")
    return ev[["gid", "liker", "uid", "time", "approx"]].reset_index(drop=True)


def directed_pairs(ev: pd.DataFrame) -> pd.DataFrame:
    return ev.groupby(["liker", "uid"], as_index=False).size().rename(columns={"size": "n"})


def undirected(pairs: pd.DataFrame) -> pd.DataFrame:
    p = pairs.assign(a=np.minimum(pairs.liker, pairs.uid), b=np.maximum(pairs.liker, pairs.uid))
    return p.groupby(["a", "b"], as_index=False).agg(n=("n", "sum"), dirs=("n", "size"))


def history(ev: pd.DataFrame, nodes: set, cat: pd.DataFrame, t: int) -> List[Dict]:
    e = ev[ev.liker.isin(nodes) & ev.uid.isin(nodes)].copy()
    e["month"] = pd.to_datetime(e.time, unit="s").dt.strftime("%Y-%m")
    first_month = pd.to_datetime(cat.groupby("uid").date.min(), unit="s").dt.strftime("%Y-%m")
    out = []
    for mth, d in e.groupby("month"):
        if mth < time.strftime("%Y-%m", time.gmtime(t - WINDOW)):
            continue
        und = undirected(directed_pairs(d))
        newcomers = set(first_month[first_month == mth].index)
        out.append({"month": mth, "likes": int(len(d)), "artists": int(pd.concat([d.liker, d.uid]).nunique()),
                    "pairs": int(len(und)), "mutual_share": round(float((und.dirs == 2).mean()), 4) if len(und) else None,
                    "newcomers": len(newcomers), "newcomers_liked": int(d.uid[d.uid.isin(newcomers)].nunique()),
                    "approx_share": round(float(d.approx.mean()), 4)})
    return out


def ego(uid: int, pairs: pd.DataFrame, ev: pd.DataFrame, allev: pd.DataFrame, nodes: set, top: set, cc: Dict) -> Dict:
    """One featured artist's neighbourhood. Named rows only for featured artists; the rest
    are folded into one summary per direction."""
    def side(df: pd.DataFrame, col: str) -> Dict:
        named = df[df[col].isin(top)].sort_values("n", ascending=False).head(TOP_N)
        rest = df[~df[col].isin(top)]
        return {"named": [{"id": int(x), "likes": int(n), "cc": cc.get(int(x), "")} for x, n in zip(named[col], named.n)],
                "others": {"artists": int(len(rest)), "likes": int(rest.n.sum())}}
    out_p = pairs[pairs.liker == uid]; in_p = pairs[pairs.uid == uid]
    mutual = set(out_p.uid) & set(in_p.liker)
    e = ev[(ev.liker == uid) | (ev.uid == uid)].copy()
    e["month"] = pd.to_datetime(e.time, unit="s").dt.strftime("%Y-%m")
    e["given"] = (e.liker == uid).astype(int); e["received"] = (e.uid == uid).astype(int)
    months = e.groupby("month")[["given", "received"]].sum().astype(int).to_dict("index")
    aud = allev[(allev.uid == uid) & ~allev.liker.isin(nodes)]
    return {"id": int(uid), "cc": cc.get(int(uid), ""),
            "out_degree": int(len(out_p)), "in_degree": int(len(in_p)), "mutual": len(mutual),
            "mutual_named": sorted(int(x) for x in mutual if x in top),
            "likes_given": int(out_p.n.sum()), "likes_received": int(in_p.n.sum()),
            "likes_to": side(out_p, "uid"), "likes_from": side(in_p, "liker"),
            "audience": {"likers": int(aud.liker.nunique()), "likes": int(len(aud))},
            "months": [{"month": m, **v} for m, v in sorted(months.items())]}


def run() -> None:
    t = now()
    t0 = time.time()
    snap = rawrepo.clone_state("state-snap")
    main = rawrepo.clone_main(["obs/likes-backfill", "obs/pulse", "obs/snapshot"])
    cat = read_parquet(snap / "catalog.parquet", columns=["gid", "uid", "date", "gone"])
    users = read_parquet(snap / "users.parquet")
    if cat is None or users is None:
        raise RuntimeError("state-snap has no catalog yet")
    cat = cat[cat.gone == 0]
    excluded = set(json.loads((CONFIG / "excluded_artists.json").read_text())["excluded"])
    idx = json.loads((DATA / "daily" / "artists" / "index.json").read_text(encoding="utf-8"))
    top = {int(a["id"]) for a in idx["artists"]} - excluded
    name_of = users.set_index("uid").name.to_dict()
    cc_of = users.set_index("uid").cc.to_dict()

    ev = load_events(main, cat, t)
    ev12 = ev[ev.time >= t - WINDOW]
    uploaders = set(cat.loc[cat.date >= t - WINDOW, "uid"])
    e12 = ev12[ev12.liker.isin(uploaders)]
    pairs = directed_pairs(e12)
    und = undirected(pairs)
    nodes = set(pairs.liker) | set(pairs.uid)
    log(f"network: {len(nodes)} artists, {len(und)} pairs, {int((und.dirs == 2).sum())} mutual, {len(uploaders)} uploaders in the window")

    out_dir = DATA / SUB
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("artists/*.json"):
        old.unlink()
    write_json("history.json", {"schema": 2, "generated": iso(t), "months": history(ev, nodes, cat, t)}, SUB)

    index = []
    for u in sorted(n for n in nodes if n in top):
        e = ego(u, pairs, e12, ev12, nodes, top, cc_of)
        write_json(f"artists/{u}.json", e, SUB)
        index.append({"id": u, "name": name_of.get(u, ""), "cc": e["cc"], "in": e["in_degree"], "out": e["out_degree"],
                      "mutual": e["mutual"], "received": e["likes_received"], "given": e["likes_given"], "audience": e["audience"]["likers"]})
    write_json("artists/index.json", {"schema": 2, "generated": iso(t), "artists": index}, SUB)
    status = {"schema": 2, "generated": iso(t), "artists": len(nodes), "pairs": int(len(und)), "mutual_pairs": int((und.dirs == 2).sum()),
              "uploaders_in_window": len(uploaders), "human_likes_in_window": int(len(ev12)), "artist_likes_in_window": int(len(e12)),
              "approx_share": round(float(e12.approx.mean()), 4) if len(e12) else None,
              "featured_in_network": len(index), "seconds": int(time.time() - t0)}
    write_json("status.json", status, SUB)
    log(f"done in {status['seconds']} s, {len(index)} featured artists")


if __name__ == "__main__":
    run()
