"""Hourly pulse: counters for every upload of the past 30 days, the Popular ordering,
and who gave each new like.

Since Divoom's budget of 30 requests a minute (2026-10-23) a pulse reads for at most
BUDGET seconds: the window listings come first, then the feeds, then the re-reads of
omitted artworks, then the like lists. What the budget cuts is left out of the hour and
logged (``skipped`` in the run log), and the health lanes show the hour as degraded for
that reason. Two other jobs have their own runner, and so their own budget: the daily
look-ahead over every list, the category totals and the review follow-up are in
stats/probe.py; the id walk, the comment and sign-up polls and the rebuild of
``data/pulse`` are in stats/refresh.py, which the workflow runs right after this job.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import health, rawrepo
from .accounts import Halted, Pool, flush_issues
from .api import Api, ApiDown, OutOfTime
from .util import (ALL_SIZES, CATEGORIES, DAY, PHOTO, SIZES, add_rollup, env_flag, is_auto, load_ranges,
                   now, read_parquet, rows_frame, tier, write_parquet)

pd.set_option("future.no_silent_downcasting", True)
WINDOW_DAYS = int(os.environ.get("PULSE_DAYS", "30"))
AGE_BANDS = [0, 1, 3, 7, 14, 31]
MIN_GAP = 50 * 60              # a pulse is skipped when the previous one is younger than this
# Reading time per run. The refresh job follows on another runner and must end before
# the next pulse starts (refresh.py), so the hour is shared: 44 minutes here leave it
# about 12 minutes, enough for the id walk, the comments and the sign-up sample.
BUDGET = int(os.environ.get("PULSE_BUDGET_SECONDS", str(44 * 60)))
POPULAR_CATS = [3, 6, 8, 12, 1, 4]
CARRY = ["first_seen", "t_new", "t_rec", "lp", "la", "cmt_ref", "miss"]


def crawl_window(api: Api, st: Path, t: int, cutoff: int, probe: dict):
    """Records of the window from every list known to hold some; (records, complete)."""
    lists_file = st / "lists.json"
    known = json.loads(lists_file.read_text()) if lists_file.exists() else {}
    # The daily probe (stats/probe.py) reads every list with a long look-ahead and
    # reports which hold window artworks, and whether an old block sits in the list.
    for key, v in probe.items():
        if key.startswith("_"):
            continue
        if v.get("n") and v["t"] > known.get(key, {}).get("t", 0):
            known[key] = {"n": v["n"], "t": v["t"], "block": v.get("block", False)}
        elif key in known:
            known[key]["block"] = v.get("block", False)
    known.pop("_probed", None)
    records, complete = [], True
    for cls in CATEGORIES:
        for size in ALL_SIZES:
            key = f"{cls}_{size}"
            if key not in known:
                continue
            # Lists are not strictly newest-first: at times a long run of much older
            # artworks sits right after the newest 60. The hourly read uses a short
            # look-ahead and relies on the daily probe plus refresh_missing().
            try:
                fl = api.crawl_list(cls, size, stop_before=cutoff, old_streak=2)
            except OutOfTime:
                complete = False
                break
            fl = [x for x in fl if x["Date"] >= cutoff]
            if fl:
                known[key] = {"n": len(fl), "t": t, "block": known[key].get("block", False)}
            elif t - known[key]["t"] > 40 * DAY:
                del known[key]
            records.extend(fl)
        if not complete:
            break
    lists_file.write_text(json.dumps(known))
    return records, complete


def apply_probe(st: Path, pr: Path) -> dict:
    """Take over what the daily probe left on its branch: the category totals; returns
    its list survey. (The outcome of held uploads is applied to the walk table by the
    refresh job, which owns it.)"""
    out = {}
    f = pr / "lists.json"
    if f.exists():
        out = json.loads(f.read_text())
    c, mine = pr / "categories.json", st / "categories.json"
    if c.exists():
        newer = not mine.exists() or json.loads(c.read_text())["log"][-1:] != json.loads(mine.read_text())["log"][-1:]
        if newer:
            mine.write_text(c.read_text())
    return out


def refresh_missing(api: Api, cur: pd.DataFrame, prev: pd.DataFrame, t: int, cutoff: int) -> pd.DataFrame:
    """Re-read, one by one, window artworks that the listings did not return this hour.

    A listing can temporarily leave out artworks that still exist. Their counters are
    read with Cloud/GalleryInfo so the hourly series has no holes; what the record
    lacks (category, canvas size, curation flags) is carried over from the last reading.
    """
    miss = prev[~prev.gid.isin(cur.gid) & (prev.date >= cutoff)].sort_values("date", ascending=False).head(400)
    rows = []
    for row, r in zip(miss.to_dict("records"), api.pmap(api.gallery_info, miss.gid.tolist())):
        if r.get("ReturnCode") != 0 or int(r.get("PrivateFlag") or 0) or int(r.get("IsDel") or 0):
            continue
        d = {c: row[c] for c in cur.columns}
        d.update(like=int(r.get("LikeCnt") or 0), watch=int(r.get("WatchCnt") or 0), cmt=int(r.get("CommentCnt") or 0),
                 likeutc=int(r.get("LikeUTC") or 0), cmtutc=int(r.get("CommentUTC") or 0), t=t)
        rows.append(d)
    return pd.concat([cur, pd.DataFrame(rows, columns=cur.columns).astype("int64")], ignore_index=True) if rows else cur


def crawl_feeds(api: Api, cutoff: int):
    """Heads of Recommend and NEW (promotion times) and the Popular ordering."""
    heads, popular = [], []
    for size in SIZES:
        heads += [x for x in api.list_page(18, size, 1) if x["Date"] >= cutoff]
        heads += [x for x in api.list_page(0, size, 1) if x["Date"] >= cutoff]
    jobs = [(c, s, k) for c in (18, 0) for s in SIZES for k in range(3)] + \
           [(c, 127, k) for c in POPULAR_CATS for k in range(2)]

    def one(j):
        c, s, k = j
        try:
            return j, api.list_page(c, s, 1 + 30 * k, sort=1)
        except ApiDown:
            return j, []
    t = now()
    for (c, s, k), fl in api.pmap(one, jobs):
        for i, x in enumerate(fl):
            popular.append({"lst": f"{c}_{s}", "rank": 30 * k + i + 1, "gid": x["GalleryId"], "t": t,
                            "like": x["LikeCnt"], "watch": x["WatchCnt"], "date": x["Date"],
                            "rec": x.get("IsAddRecommend") or 0, "new": x.get("IsAddNew") or 0})
    return heads, pd.DataFrame(popular)


def like_events(api: Api, d: pd.DataFrame, t: int, authed: bool) -> pd.DataFrame:
    """Read the newest likers of every artwork whose like count rose."""
    cols = ["gid", "liker", "t", "t_prev", "auto"]
    todo = d[d.dl > 0].sort_values("dl", ascending=False).head(600)
    if not authed or todo.empty:
        return pd.DataFrame(columns=cols)
    ranges = load_ranges()

    def one(row):
        try:
            ul = api.like_list(int(row.gid), int(min(row.dl, 500)))
        except ApiDown:
            ul = None
        if ul is None:
            return []
        return [(int(row.gid), int(u["UserId"]), t, int(row.t_p)) for u in ul]
    ev = [e for part in api.pmap(one, list(todo.itertuples())) for e in part]
    out = pd.DataFrame(ev, columns=cols[:4])
    out["auto"] = is_auto(out.liker, ranges).astype("int8") if len(out) else []
    return out.astype({"gid": "int64", "liker": "int64", "t": "int64", "t_prev": "int64"})


def run() -> None:
    t = now()
    hour = t // 3600 * 3600
    cutoff = t - WINDOW_DAYS * DAY
    st = rawrepo.clone_state("state-pulse")
    # Several triggers aim at every hour (an outside timer, plus GitHub's own schedule as
    # a fallback) because any one of them can be late or dropped. Only the first does work.
    lp = st / "last_pulse.json"
    age = t - json.loads(lp.read_text())["t"] if lp.exists() else None
    if age is not None and age < MIN_GAP and not env_flag("FORCE_PULSE"):
        print(f"[pulse] the last pulse was {age // 60} minutes ago; nothing to do")
        return
    main = rawrepo.clone_main(["state"])
    api = Api(workers=4)
    api.set_budget(BUDGET)
    pool, authed = Pool(main), True
    try:
        pool.acquire(api, "pulse")
    except (Halted, ApiDown) as exc:        # degraded mode: list heads need no token
        print(f"[pulse] running without a token: {exc}")
        api.auth, authed = {}, False
    probe = apply_probe(st, rawrepo.clone_state("state-probe"))

    skipped = []                            # phases the reading budget cut, in order

    def phase(name, fn, default):
        try:
            return fn()
        except OutOfTime:
            skipped.append(name)
            print(f"[pulse] out of time before {name}")
            return default

    def spent(name) -> None:                # a phase that swallows ApiDown per item
        if api.seconds_left is not None and api.seconds_left <= 0 and name not in skipped:
            skipped.append(name)

    records, complete = crawl_window(api, st, t, cutoff, probe)
    if not complete:
        skipped.append("window")
    heads, popular = phase("feeds", lambda: crawl_feeds(api, cutoff), ([], pd.DataFrame()))
    spent("feeds")
    cur = rows_frame(records + heads, t)
    cur = cur[cur.priv == 0].drop(columns=["priv"])
    prev = read_parquet(st / "window.parquet")
    bootstrap = prev is None
    if bootstrap:
        prev = pd.DataFrame(columns=list(cur.columns) + CARRY)
    n_listed = len(cur)
    if authed and not bootstrap and complete:       # after a cut window, omissions are the cut
        cur = phase("missing", lambda: refresh_missing(api, cur, prev, t, cutoff), cur)
    p = prev[["gid", "like", "watch", "cmt", "new", "rec", "t"] + CARRY].add_suffix("_p").rename(columns={"gid_p": "gid"})
    d = cur.merge(p, on="gid", how="left")
    isnew = d.like_p.isna()
    fresh = isnew & ((t - d.date) < 3 * 3600) & (not bootstrap)     # seen within hours of upload
    for a, b in (("dl", "like"), ("dv", "watch"), ("dc", "cmt")):
        d[a] = np.where(isnew, np.where(fresh, d[b], 0), d[b] - d[f"{b}_p"].fillna(0)).astype("int64")
    d["unl"] = (-d.dl).clip(lower=0)
    d["dl"] = d.dl.clip(lower=0)
    d["dv"] = d.dv.clip(lower=0)
    d["dc"] = d.dc.clip(lower=0)
    d["t_p"] = np.where(isnew, d.date, d.t_p.fillna(0)).astype("int64")
    d["dt"] = np.where(isnew & ~fresh, 0, t - d.t_p)

    ev = like_events(api, d, t, authed)
    spent("likes")
    per = ev.groupby("gid").auto.agg(["sum", "count"]) if len(ev) else pd.DataFrame(columns=["sum", "count"])
    d["dla"] = d.gid.map(per["sum"]).fillna(0).astype("int64")
    d["dlp"] = (d.gid.map(per["count"]).fillna(0) - d.dla).astype("int64")
    d["dlu"] = (d.dl - d.dla - d.dlp).clip(lower=0)

    # carried per-artwork fields
    d["first_seen"] = d.first_seen_p.fillna(t).astype("int64")
    to_new = (d.new == 1) & ((d.new_p == 0) | (isnew & fresh))
    to_rec = (d.rec == 1) & ((d.rec_p == 0) | (isnew & fresh))
    d["t_new"] = np.where(to_new, t, d.t_new_p)
    d["t_rec"] = np.where(to_rec, t, d.t_rec_p)
    d["lp"] = (d.lp_p.fillna(0) + d.dlp).astype("int64")
    d["la"] = (d.la_p.fillna(0) + d.dla).astype("int64")
    d["cmt_ref"] = d.cmt_ref_p.fillna(d.cmt).astype("int64")
    d["miss"] = 0
    d["tier"] = tier(d)
    d["photo"] = (d.cls == PHOTO).astype("int8")
    age_h = ((t - d.date) // 3600).clip(lower=0)
    d["ab"] = pd.cut(age_h / 24, AGE_BANDS, right=False, labels=False).fillna(len(AGE_BANDS) - 2).astype("int8")

    # -- rollups ------------------------------------------------------------
    live = d[d.dt > 0]
    m = ["dl", "dlp", "dla", "dlu", "dv", "dc", "unl", "dt"]
    h = live.assign(hour=hour, n=1).groupby(["hour", "size", "tier", "photo", "ab"], as_index=False)[["n"] + m].sum()
    add_rollup(st / "hourly.parquet", h, ["hour", "size", "tier", "photo", "ab"])
    ups = d if bootstrap else d[isnew]
    if len(ups):
        u = ups.assign(hour=ups.date // 3600 * 3600, n=1).groupby(["hour", "size", "cls"], as_index=False)[["n"]].sum()
        add_rollup(st / "uploads.parquet", u, ["hour", "size", "cls"])
    bucket = np.where(age_h < 168, age_h, 168 + (age_h // 24 - 7))
    a = live.assign(b=bucket[live.index], n=1).groupby(["b", "size", "tier", "photo"], as_index=False)[["n", "dl", "dlp", "dv", "dt"]].sum()
    add_rollup(st / "agecurve.parquet", a, ["b", "size", "tier", "photo"])
    rel = (t - d.t_rec) // 3600
    pr = live[d.t_rec.notna()[live.index] & (rel[live.index] < 336)]
    if len(pr):
        r_ = rel[pr.index].astype("int64")
        pr = pr.assign(rel=np.where(r_ < 72, r_, 72 + (r_ - 72) // 6), n=1)
        add_rollup(st / "promo.parquet", pr.groupby(["rel", "size"], as_index=False)[["n", "dl", "dlp", "dv", "dt"]].sum(), ["rel", "size"])
    trans = []
    for kind, mask in (("new", to_new), ("rec", to_rec)):
        x = d[mask & (not bootstrap)]
        if len(x):
            trans.append(x.assign(kind=kind, t_ev=t, like_at=x.like, watch_at=x.watch)[
                ["gid", "kind", "t_ev", "date", "size", "cls", "like_at", "watch_at", "lp"]])
    if trans:
        old = read_parquet(st / "promo_events.parquet")
        write_parquet(pd.concat(([old] if old is not None else []) + trans, ignore_index=True), st / "promo_events.parquet")

    # popular state: when each artwork was first and last seen near the top
    if len(popular):
        popular = popular.merge(d[["gid", "lp", "la"]], on="gid", how="left")
        pp = read_parquet(st / "popular.parquet")
        g = popular.groupby(["lst", "gid"], as_index=False).agg(
            first_t=("t", "min"), last_t=("t", "max"), n_obs=("t", "size"), best=("rank", "min"),
            like0=("like", "first"), watch0=("watch", "first"), date=("date", "first"),
            rec=("rec", "max"), new=("new", "max"), lp0=("lp", "first"), la0=("la", "first"))
        if pp is not None:
            g = pd.concat([pp, g]).groupby(["lst", "gid"], as_index=False).agg(
                first_t=("first_t", "min"), last_t=("last_t", "max"), n_obs=("n_obs", "sum"), best=("best", "min"),
                like0=("like0", "first"), watch0=("watch0", "first"), date=("date", "first"),
                rec=("rec", "max"), new=("new", "max"), lp0=("lp0", "first"), la0=("la0", "first"))
        write_parquet(g[g.last_t > t - 90 * DAY], st / "popular.parquet")

    # -- raw observations ---------------------------------------------------
    obs = main / "obs" / "pulse" / time.strftime("%Y/%m/%d", time.gmtime(t))
    hh = time.strftime("%H%M", time.gmtime(t))
    changed = d[isnew | (d.dl > 0) | (d.dv > 0) | (d.dc > 0) | to_new | to_rec | (d.unl > 0)]
    write_parquet(changed[["gid", "uid", "date", "cls", "size", "like", "watch", "cmt", "new", "rec", "t"]], obs / f"{hh}-changes.parquet")
    if len(ev):
        write_parquet(ev, obs / f"{hh}-likes.parquet")
    if len(popular):
        write_parquet(popular[["lst", "rank", "gid", "t"]], obs / f"{hh}-popular.parquet")

    # -- window state -------------------------------------------------------
    keep = list(cur.columns) + CARRY
    gone = prev[~prev.gid.isin(d.gid) & (prev.date >= cutoff)].copy()
    if len(gone):
        gone["miss"] = gone.miss.fillna(0) + 1
        gone = gone[gone.miss <= 72]
    win = pd.concat([d[keep], gone[keep]], ignore_index=True)
    write_parquet(win, st / "window.parquet")
    meta = {"t": t, "mode": "normal" if authed else "degraded", "requests": api.n_requests, "served": api.n_served,
            "limited": api.n_limited, "n_window": int(len(d)), "n_listed": int(n_listed), "window_days": WINDOW_DAYS,
            "bootstrap": bootstrap, "skipped": skipped, "seconds_left": None if api.seconds_left is None else int(api.seconds_left)}
    (st / "last_pulse.json").write_text(json.dumps(meta))
    print(f"[pulse] {meta}")

    # one row per run, the evidence the health lanes are built from (stats/health.py)
    health.log_run(st, {"t": t, "hour": hour, "mode": meta["mode"], "requests": api.n_requests, "limited": api.n_limited,
                        "n_window": meta["n_window"], "n_listed": meta["n_listed"], "seconds": now() - t,
                        "skipped": ",".join(skipped), "served": api.n_served})
    rawrepo.push_state("state-pulse", f"pulse {time.strftime('%Y-%m-%d %H:%M', time.gmtime(t))}")
    rawrepo.push_main(f"pulse {time.strftime('%Y-%m-%d %H', time.gmtime(t))}")
    flush_issues(Path("_issues"))
    print(f"[pulse] done in {now() - t}s, {api.summary()}" + (f", cut: {', '.join(skipped)}" if skipped else ""))


if __name__ == "__main__":
    try:
        run()
    except ApiDown as exc:
        print(f"[pulse] server unreachable, ending quietly: {exc}")
        sys.exit(0)
