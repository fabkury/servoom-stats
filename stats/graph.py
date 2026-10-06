"""Daily artist graph: communities, map, flows and top-artist neighbourhoods.

Design: servoom/docs/community-network/README.md. Inputs are the like event log (pulse,
snapshot and backfill files in the raw repository) and the snapshot's state tables.
No Divoom request is made. Output goes to ``data/community/`` (public: aggregates, the
id-free map and files for top artists only) and to the raw branch ``state-graph``
(yesterday's memberships and positions, for day-to-day matching).

Naming rule: only accounts in ``data/daily/artists/index.json`` (the stats site's top
artists) appear with an id. Every other artist is a dot carrying community and size
class only. Communities under MIN_COMMUNITY members are merged into community 0.
"""

from __future__ import annotations

import json
import os
import struct
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.optimize import linear_sum_assignment

from . import rawrepo
from .util import (CAT_NAME, CONFIG, DATA, DAY, SIZE_NAME, day_of, iso, is_auto, load_ranges, now, read_parquet, recs,
                   small_cells, write_json, write_parquet)

SUB = "community"
BRANCH = "state-graph"
WINDOW = 365 * DAY
MIN_COMMUNITY = 20
TARGET_MEDIAN = 150
GRID = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0]
MAX_LARGEST = 0.15
EMBED_DIM = 32
SEED = 7
TOP_N = 20                      # neighbours listed per top artist


def log(msg: str) -> None:
    print(f"[graph] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Events
# ---------------------------------------------------------------------------

def load_events(main: Path, cat: pd.DataFrame, t: int) -> pd.DataFrame:
    """Human likes as (gid, liker, uid, time, approx), one row per (gid, liker)."""
    parts: List[pd.DataFrame] = []
    files = sorted((main / "obs" / "likes-backfill").glob("*.parquet"))
    files += sorted(p for p in (main / "obs").rglob("*likes*.parquet") if "likes-backfill" not in p.as_posix())
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


# ---------------------------------------------------------------------------
# 2. Graph
# ---------------------------------------------------------------------------

def directed_pairs(ev: pd.DataFrame) -> pd.DataFrame:
    return ev.groupby(["liker", "uid"], as_index=False).size().rename(columns={"size": "n"})


def build_graph(pairs: pd.DataFrame) -> Tuple[ig.Graph, List[int]]:
    p = pairs.copy()
    p["a"] = np.minimum(p.liker, p.uid)
    p["b"] = np.maximum(p.liker, p.uid)
    p["w"] = np.log1p(p.n)
    u = p.groupby(["a", "b"], as_index=False).agg(w=("w", "sum"), n=("n", "sum"), dirs=("n", "size"))
    nodes = sorted(set(u.a) | set(u.b))
    idx = {x: i for i, x in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), edges=list(zip(u.a.map(idx), u.b.map(idx))))
    g.es["w"] = u.w.values
    g.es["n"] = u.n.values
    g.es["mutual"] = (u.dirs == 2).values
    deg = np.array(g.strength(weights="w"))
    src = np.array([e.source for e in g.es]); dst = np.array([e.target for e in g.es])
    g.es["wn"] = u.w.values / np.sqrt(deg[src] * deg[dst])
    g.vs["uid"] = nodes
    return g, nodes


def leiden(g: ig.Graph, res: float) -> np.ndarray:
    part = la.find_partition(g, la.RBConfigurationVertexPartition, weights="wn", resolution_parameter=res, seed=SEED)
    return np.array(part.membership)


def sizes_of(m: np.ndarray) -> np.ndarray:
    return np.bincount(m)


def choose_resolution(g: ig.Graph) -> Tuple[float, List[Dict]]:
    trials = []
    for res in GRID:
        m = leiden(g, res)
        s = sizes_of(m)
        big = s[s >= MIN_COMMUNITY]
        med = float(np.median(big)) if len(big) else 0.0
        largest = float(s.max() / len(m))
        trials.append({"resolution": res, "communities": int(len(big)), "median": med, "largest_share": round(largest, 4),
                       "covered": round(float(big.sum() / len(m)), 4), "modularity": round(g.modularity(m, weights="wn"), 4)})
        log(f"resolution {res}: {len(big)} communities >= {MIN_COMMUNITY}, median {med:.0f}, "
            f"largest {largest:.2f} of nodes, covered {big.sum()/len(m):.2f}")
    # A low resolution leaves one giant community (75% of artists at 0.5 on 2026-10-06), which
    # the median does not see. Require the largest community to stay under MAX_LARGEST of the
    # nodes, then take the median size closest to the target.
    ok = [x for x in trials if x["median"] > 0 and x["largest_share"] <= MAX_LARGEST]
    if not ok:
        ok = [min(trials, key=lambda x: x["largest_share"])]
    best = min(ok, key=lambda x: abs(np.log(x["median"] / TARGET_MEDIAN)))
    return best["resolution"], trials


def match_labels(nodes: List[int], m: np.ndarray, prev: Optional[pd.DataFrame]) -> Tuple[np.ndarray, Dict]:
    """Relabel raw Leiden labels with stable ids. 0 = small groups; others keep
    yesterday's id when at least half of the new members were in that community."""
    s = sizes_of(m)
    big = np.flatnonzero(s >= MIN_COMMUNITY)
    out = np.zeros(len(m), dtype="int64")
    info = {"kept": 0, "new": 0, "stability": None}
    if prev is None or prev.empty:
        for k, raw in enumerate(big, start=1):
            out[m == raw] = k
        info["new"] = len(big)
        return out, info
    pm = dict(zip(prev.uid, prev.comm))
    old_of = np.array([pm.get(u, -1) for u in nodes])
    old_ids = sorted(set(pm.values()) - {0})
    oi = {c: j for j, c in enumerate(old_ids)}
    overlap = np.zeros((len(big), len(old_ids)))
    for i, raw in enumerate(big):
        members_old = old_of[m == raw]
        for c, cnt in zip(*np.unique(members_old[members_old > 0], return_counts=True)):
            overlap[i, oi[c]] = cnt
    rows, cols = linear_sum_assignment(-overlap)
    assigned: Dict[int, int] = {}
    for i, j in zip(rows, cols):
        if overlap[i, j] >= 0.5 * s[big[i]]:
            assigned[i] = old_ids[j]
    next_id = max(old_ids, default=0) + 1
    for i, raw in enumerate(big):
        if i in assigned:
            out[m == raw] = assigned[i]
            info["kept"] += 1
        else:
            out[m == raw] = next_id
            next_id += 1
            info["new"] += 1
    # stability: among nodes present both days and in a big community both days, share whose
    # community id is unchanged
    both = (old_of > 0) & (out > 0)
    info["stability"] = float((old_of[both] == out[both]).mean()) if both.any() else None
    return out, info


# ---------------------------------------------------------------------------
# 3. Map
# ---------------------------------------------------------------------------

def embed(g: ig.Graph, prev_pos: Optional[pd.DataFrame]) -> np.ndarray:
    import umap  # slow import, kept local
    n = g.vcount()
    src = np.array([e.source for e in g.es]); dst = np.array([e.target for e in g.es])
    A = sp.coo_matrix((np.array(g.es["w"]), (src, dst)), shape=(n, n))
    A = (A + A.T).tocsr()
    d = np.asarray(A.sum(1)).ravel()
    Dm = sp.diags(1 / np.sqrt(np.maximum(d, 1e-9)))
    k = min(EMBED_DIM, n - 2)
    _, vecs = sla.eigsh(Dm @ A @ Dm, k=k, which="LA")
    init: object = "spectral"
    if prev_pos is not None and not prev_pos.empty:
        pp = dict(zip(prev_pos.uid, zip(prev_pos.x, prev_pos.y)))
        uids = g.vs["uid"]
        pos = np.array([pp.get(u, (np.nan, np.nan)) for u in uids], dtype="float64")
        known = ~np.isnan(pos[:, 0])
        if known.sum() >= 0.5 * n:
            # new nodes start at the mean position of their known neighbours, else the centre
            adj = A.tolil()
            centre = np.nanmean(pos, axis=0)
            for i in np.flatnonzero(~known):
                nb = [j for j in adj.rows[i] if known[j]]
                pos[i] = pos[nb].mean(axis=0) if nb else centre
            init = pos + np.random.default_rng(SEED).normal(0, 1e-3, pos.shape)
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=SEED, init=init)
    xy = reducer.fit_transform(vecs)
    xy = (xy - xy.mean(axis=0)) / (xy.std(axis=0).max() + 1e-9)
    return xy.astype("float32")


def size_class(deg: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(np.log2(np.maximum(deg, 1))), 0, 7).astype("uint8")


def write_map(path: Path, nodes: List[int], xy: np.ndarray, comm: np.ndarray, deg: np.ndarray, top: set) -> None:
    order = np.random.default_rng(SEED).permutation(len(nodes))   # file order reveals nothing
    sc = size_class(deg)
    with open(path, "wb") as f:
        f.write(b"SVCM")
        f.write(struct.pack("<II", 1, len(nodes)))
        for i in order:
            uid = nodes[i]
            f.write(struct.pack("<ffHBxI", float(xy[i, 0]), float(xy[i, 1]), int(comm[i]), int(sc[i]),
                                uid if uid in top else 0))


# ---------------------------------------------------------------------------
# 4. Descriptions
# ---------------------------------------------------------------------------

def mix(series: pd.Series, k: int = 5) -> Dict[str, float]:
    c = small_cells(series.value_counts(), k)
    tot = c.sum()
    return {str(a): round(float(b / tot), 4) for a, b in c.items()} if tot else {}


def distinctive(local: Dict[str, float], whole: Dict[str, float], min_share: float = 0.25, ratio: float = 1.5) -> List[str]:
    out = [(k, v / max(whole.get(k, 1e-9), 1e-9)) for k, v in local.items() if k != "other" and v >= min_share]
    return [k for k, r in sorted(out, key=lambda x: -x[1]) if r >= ratio][:2]


def community_profiles(comm: np.ndarray, nodes: List[int], cat12: pd.DataFrame, users: pd.DataFrame,
                       ev12: pd.DataFrame, pairs: pd.DataFrame, top: set) -> Tuple[List[Dict], Dict]:
    node_comm = pd.Series(comm, index=nodes)
    cat12 = cat12[cat12.uid.isin(node_comm.index)].copy()
    cat12["comm"] = cat12.uid.map(node_comm)
    cat12["sz"] = cat12["size"].map(SIZE_NAME).fillna("other")
    cat12["cat"] = cat12.cls.map(CAT_NAME).fillna("other")
    cat12["hour"] = (cat12.date % DAY) // 3600
    u = users.set_index("uid")
    cc = pd.Series([u.cc.get(x, "") or "" for x in nodes], index=nodes).replace("", "other")
    whole = {"cc": mix(cc), "size": mix(cat12.sz), "cat": mix(cat12.cat)}
    hour_all = np.bincount(cat12.hour, minlength=24) / max(1, len(cat12))
    # audience: likers of members who are not graph nodes
    nodeset = set(nodes)
    aud = ev12[~ev12.liker.isin(nodeset) & ev12.uid.isin(nodeset)].copy()
    aud["comm"] = aud.uid.map(node_comm)
    # flows between communities
    pc = pairs.copy()
    pc["ca"] = pc.liker.map(node_comm); pc["cb"] = pc.uid.map(node_comm)
    pc = pc.dropna()
    flow = pc.groupby(["ca", "cb"]).n.sum()
    out = []
    for c in sorted(set(comm)):
        members = [n for n, k in zip(nodes, comm) if k == c]
        if c == 0 or len(members) < MIN_COMMUNITY:
            continue
        mcat = cat12[cat12.comm == c]
        mcc = cc.loc[members]
        prof = {"cc": mix(mcc), "size": mix(mcat.sz), "cat": mix(mcat.cat)}
        desc = {k: distinctive(prof[k], whole[k]) for k in prof}
        hours = np.bincount(mcat.hour, minlength=24) / max(1, len(mcat))
        peak = int(np.argmax(hours - hour_all))
        a = aud[aud.comm == c]
        per_liker = a.groupby("liker").agg(n=("gid", "size"), members=("uid", "nunique"))
        internal = int(flow.get((c, c), 0))
        given = int(flow.xs(c, level="ca").sum()) if c in flow.index.get_level_values(0) else 0
        received = int(flow.xs(c, level="cb").sum()) if c in flow.index.get_level_values(1) else 0
        tops = [int(m) for m in members if m in top]
        out.append({
            "id": int(c), "members": len(members), "top_artists": len(tops), "top_ids": sorted(tops),
            "descriptor": desc, "peak_hour_utc": peak, "hours": [round(float(h), 4) for h in hours],
            "mix": prof, "uploads": int(len(mcat)),
            "likes_internal": internal, "likes_given": given, "likes_received": received,
            "internal_share": round(internal / given, 4) if given else None,
            "audience": {"likers": int(len(per_liker)), "likes": int(len(a)),
                         "repeat_share": round(float((per_liker.n > 1).mean()), 4) if len(per_liker) else None,
                         "multi_member_share": round(float((per_liker.members > 1).mean()), 4) if len(per_liker) else None},
        })
    out.sort(key=lambda x: -x["members"])
    ids = [x["id"] for x in out]
    matrix = [[int(flow.get((a, b), 0)) for b in ids] for a in ids]
    return out, {"ids": ids, "likes": matrix, "whole": whole, "hours": [round(float(h), 4) for h in hour_all]}


def country_flows(pairs: pd.DataFrame, users: pd.DataFrame, k: int = 25) -> Dict:
    u = users.set_index("uid").cc
    p = pairs.copy()
    p["ca"] = p.liker.map(u).fillna("").replace("", "other")
    p["cb"] = p.uid.map(u).fillna("").replace("", "other")
    tot = p.groupby("ca").n.sum().add(p.groupby("cb").n.sum(), fill_value=0).sort_values(ascending=False)
    keep = [c for c in tot.index[:k] if c != "other"]
    p["ca"] = np.where(p.ca.isin(keep), p.ca, "other")
    p["cb"] = np.where(p.cb.isin(keep), p.cb, "other")
    m = p.groupby(["ca", "cb"]).n.sum()
    ids = keep + ["other"]
    return {"ids": ids, "likes": [[int(m.get((a, b), 0)) for b in ids] for a in ids]}


# ---------------------------------------------------------------------------
# 5. History and egos
# ---------------------------------------------------------------------------

def history(ev: pd.DataFrame, node_comm: pd.Series, cat: pd.DataFrame, t: int) -> List[Dict]:
    e = ev[ev.liker.isin(node_comm.index) & ev.uid.isin(node_comm.index)].copy()
    e["month"] = pd.to_datetime(e.time, unit="s").dt.strftime("%Y-%m")
    e["ca"] = e.liker.map(node_comm); e["cb"] = e.uid.map(node_comm)
    first_up = cat.groupby("uid").date.min()
    first_month = pd.to_datetime(first_up, unit="s").dt.strftime("%Y-%m")
    out = []
    for mth, d in e.groupby("month"):
        if mth < time.strftime("%Y-%m", time.gmtime(t - WINDOW)):
            continue
        pairs = d.groupby(["liker", "uid"]).size().reset_index()
        a = np.minimum(pairs.liker, pairs.uid); b = np.maximum(pairs.liker, pairs.uid)
        und = pd.DataFrame({"a": a, "b": b}).groupby(["a", "b"]).size()
        newcomers = set(first_month[first_month == mth].index)
        adopted = set(d.uid[d.uid.isin(newcomers)])
        out.append({"month": mth, "likes": int(len(d)), "artists": int(pd.concat([d.liker, d.uid]).nunique()),
                    "pairs": int(len(und)), "mutual_share": round(float((und == 2).mean()), 4) if len(und) else None,
                    "internal_share": round(float(((d.ca == d.cb) & (d.ca > 0)).mean()), 4),
                    "newcomers": len(newcomers), "newcomers_liked": len(adopted),
                    "approx_share": round(float(d.approx.mean()), 4)})
    return out


def ego(uid: int, pairs: pd.DataFrame, ev: pd.DataFrame, node_comm: pd.Series, top: set, t: int) -> Dict:
    def side(df: pd.DataFrame, col: str) -> List[Dict]:
        s = df.sort_values("n", ascending=False).head(TOP_N)
        return [{"id": int(x) if x in top else None, "comm": int(node_comm.get(x, 0)), "likes": int(n)}
                for x, n in zip(s[col], s.n)]
    out_p = pairs[pairs.liker == uid]; in_p = pairs[pairs.uid == uid]
    mutual = set(out_p.uid) & set(in_p.liker)
    e = ev[(ev.liker == uid) | (ev.uid == uid)].copy()
    e["month"] = pd.to_datetime(e.time, unit="s").dt.strftime("%Y-%m")
    e["given"] = (e.liker == uid).astype(int); e["received"] = (e.uid == uid).astype(int)
    months = e.groupby("month")[["given", "received"]].sum().astype(int).to_dict("index")
    aud = ev[(ev.uid == uid) & ~ev.liker.isin(node_comm.index)]
    return {"id": int(uid), "comm": int(node_comm.get(uid, 0)),
            "out_degree": int(len(out_p)), "in_degree": int(len(in_p)), "mutual": len(mutual),
            "likes_given": int(out_p.n.sum()), "likes_received": int(in_p.n.sum()),
            "likes_to": side(out_p, "uid"), "likes_from": side(in_p, "liker"),
            "comm_of_likers": {str(int(k)): int(v) for k, v in in_p.liker.map(node_comm).fillna(0).value_counts().items()},
            "audience": {"likers": int(aud.liker.nunique()), "likes": int(len(aud))},
            "months": [{"month": m, **v} for m, v in sorted(months.items())]}


# ---------------------------------------------------------------------------
# 6. Run
# ---------------------------------------------------------------------------

def run() -> None:
    t = now()
    t0 = time.time()
    st = rawrepo.clone_state(BRANCH)
    snap = rawrepo.clone_state("state-snap")
    main = rawrepo.clone_main(["obs/likes-backfill", "obs/pulse", "obs/snapshot"])
    cat = read_parquet(snap / "catalog.parquet", columns=["gid", "uid", "date", "size", "cls", "gone"])
    users = read_parquet(snap / "users.parquet")
    if cat is None or users is None:
        raise RuntimeError("state-snap has no catalog yet")
    cat = cat[cat.gone == 0]
    excluded = set(json.loads((CONFIG / "excluded_artists.json").read_text())["excluded"])
    idx = json.loads((DATA / "daily" / "artists" / "index.json").read_text(encoding="utf-8"))
    top = {int(a["id"]) for a in idx["artists"]} - excluded
    cfgf = DATA / SUB / "config.json"
    cfg = json.loads(cfgf.read_text()) if cfgf.exists() else {}

    ev = load_events(main, cat, t)
    ev12 = ev[ev.time >= t - WINDOW]
    uploaders = set(cat.loc[cat.date >= t - WINDOW, "uid"])
    e12 = ev12[ev12.liker.isin(uploaders)]
    pairs = directed_pairs(e12)
    g, nodes = build_graph(pairs)
    log(f"graph: {g.vcount()} artists, {g.ecount()} edges, {int(np.sum(g.es['mutual']))} mutual pairs, "
        f"{len(uploaders)} uploaders in the window")

    trials = None
    if not cfg.get("resolution"):
        cfg["resolution"], trials = choose_resolution(g)
        cfg["resolution_pinned_on"] = day_of(t)
        log(f"resolution pinned at {cfg['resolution']}")
    raw = leiden(g, cfg["resolution"])
    prev = read_parquet(st / "membership.parquet")
    comm, minfo = match_labels(nodes, raw, prev)
    node_comm = pd.Series(comm, index=nodes)
    s = sizes_of(comm)
    log(f"communities: {int((s[1:] > 0).sum())} with >= {MIN_COMMUNITY} members, {int(s[0])} artists in small groups, "
        f"kept {minfo['kept']} ids, {minfo['new']} new, stability {minfo['stability']}")

    xy = embed(g, read_parquet(st / "positions.parquet"))
    log(f"map embedded in {time.time() - t0:.0f} s total so far")
    deg = np.array(g.degree())
    out_dir = DATA / SUB
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("artists/*.json"):
        old.unlink()
    write_map(out_dir / "map.bin", nodes, xy, comm, deg, top)

    cat12 = cat[cat.date >= t - WINDOW]
    comms, flows = community_profiles(comm, nodes, cat12, users, ev12, pairs, top)
    write_json("communities.json", {"schema": 1, "generated": iso(t), "window_days": 365, "min_members": MIN_COMMUNITY,
                                    "whole": flows["whole"], "hours": flows["hours"], "communities": comms}, SUB)
    write_json("flows.json", {"schema": 1, "generated": iso(t), "communities": {"ids": flows["ids"], "likes": flows["likes"]},
                              "countries": country_flows(pairs, users)}, SUB)
    write_json("history.json", {"schema": 1, "generated": iso(t), "months": history(ev, node_comm, cat, t)}, SUB)

    in_graph = [u for u in nodes if u in top]
    name_of = users.set_index("uid").name.to_dict()
    index = []
    for u in in_graph:
        e = ego(u, pairs, e12, node_comm, top, t)
        write_json(f"artists/{u}.json", e, SUB)
        index.append({"id": u, "name": name_of.get(u, ""), "comm": e["comm"], "in": e["in_degree"], "out": e["out_degree"], "mutual": e["mutual"]})
    write_json("artists/index.json", {"schema": 1, "generated": iso(t), "artists": index}, SUB)

    # dots are published without ids; the state branch keeps the mapping for tomorrow
    write_parquet(pd.DataFrame({"uid": nodes, "comm": comm}), st / "membership.parquet")
    write_parquet(pd.DataFrame({"uid": nodes, "x": xy[:, 0], "y": xy[:, 1]}), st / "positions.parquet")
    rawrepo.push_state(BRANCH, f"graph {day_of(t)}")

    status = {"schema": 1, "generated": iso(t), "artists": g.vcount(), "edges": g.ecount(),
              "mutual_pairs": int(np.sum(g.es["mutual"])), "uploaders_in_window": len(uploaders),
              "human_likes_in_window": int(len(ev12)), "artist_likes_in_window": int(len(e12)),
              "approx_share": round(float(e12.approx.mean()), 4) if len(e12) else None,
              "communities": int((s[1:] > 0).sum()), "small_groups_artists": int(s[0]),
              "resolution": cfg["resolution"], "kept_ids": minfo["kept"], "new_ids": minfo["new"],
              "stability": minfo["stability"], "top_artists_in_graph": len(in_graph),
              "modularity": round(g.modularity(raw, weights="wn"), 4), "seconds": int(time.time() - t0)}
    if trials:
        cfg["trials"] = trials
    cfgf.write_text(json.dumps(cfg, indent=1))
    write_json("status.json", status, SUB)
    log(f"done in {status['seconds']} s")


if __name__ == "__main__":
    run()
