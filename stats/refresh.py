"""Every-4-hours refresh: a few slow sources, then ``data/pulse/*.json`` for the site."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .api import Api, ApiDown
from .util import (CAT_NAME, CATEGORIES, DATA, DAY, SCHEMA, SIZE_NAME, SIZES, add_rollup, fingerprint, is_auto,
                   iso, load_ranges, read_parquet, recs, script_of, small_cells, write_json, write_parquet)

SUB = "pulse"


# ---------------------------------------------------------------- polling ----
def user_frontier(api: Api, lo: int) -> int:
    """Highest issued account id, by galloping then bisecting from the last known one."""
    def any_near(u: int) -> bool:
        u |= 1                                    # odd ids are ~85% in use
        return any(api.user_info(u + 2 * k).get("ReturnCode") == 0 for k in range(6))
    step, hi = 512, lo + 512
    while any_near(hi):
        lo, step = hi, step * 2
        hi = lo + step
    while hi - lo > 24:
        mid = (lo + hi) // 2
        if any_near(mid):
            lo = mid
        else:
            hi = mid
    return lo


def poll_signups(api: Api, st: Path, t: int) -> None:
    f = st / "signups.json"
    s = json.loads(f.read_text()) if f.exists() else {"frontier": 405024000, "log": []}
    lo = s["frontier"]
    hi = user_frontier(api, lo)
    span = hi - lo
    if span <= 0:
        return
    ids = random.sample(range(lo + 1, hi + 1), min(200, span))
    infos = api.pmap(api.user_info, ids)
    ok = [(u, r) for u, r in zip(ids, infos) if r.get("ReturnCode") == 0]
    cc = pd.Series([r.get("CountryISOCode") or "??" for _, r in ok]).value_counts()
    s["log"].append({"t": t, "lo": lo, "hi": hi, "sampled": len(ids), "exist": len(ok),
                     "cc": {k: int(v) for k, v in cc.items()}})
    s["log"] = s["log"][-2500:]
    s["frontier"] = hi
    f.write_text(json.dumps(s))


def poll_comments(api: Api, st: Path, main: Path, win: pd.DataFrame, t: int) -> pd.DataFrame:
    """Comments written since the last refresh on window artworks. No text is kept."""
    salt_file = main / "state" / "salt.txt"
    salt = salt_file.read_text().strip() if salt_file.exists() else "unsalted"
    lf = st / "last_refresh.json"
    since = json.loads(lf.read_text())["t"] if lf.exists() else t - 4 * 3600
    todo = win[(win.cmt > win.cmt_ref) | (win.cmtutc > since)].gid.head(400).tolist()
    ranges = load_ranges()
    rows = []
    for gid, cl in zip(todo, api.pmap(api.comments, todo)):
        for c in cl:
            d = int(c.get("Date") or 0)
            if d <= since:
                continue
            txt = c.get("Comment") or ""
            rows.append((gid, int(c.get("CommentId") or 0), int(c["UserId"]), d, fingerprint(txt, salt), len(txt),
                         script_of(txt), int(str(c.get("RobertFlag") or "0") != "0")))
    df = pd.DataFrame(rows, columns=["gid", "cid", "uid", "date", "fp", "len", "script", "robot"])
    if len(df):
        df["auto"] = is_auto(df.uid, ranges).astype("int8")
        write_parquet(df, main / "obs" / "pulse" / time.strftime("%Y/%m/%d", time.gmtime(t)) / f"{time.strftime('%H%M', time.gmtime(t))}-comments.parquet")
        g = df.assign(hour=df.date // 3600 * 3600, n=1, people=1 - df.auto.clip(upper=1))
        add_rollup(st / "comments.parquet", g.groupby(["hour", "script"], as_index=False)[["n", "people", "auto", "robot"]].sum(), ["hour", "script"])
        # repeated fingerprints from different accounts hint at a campaign
        rep = df.groupby("fp").uid.nunique()
        (st / "comment_repeats.json").write_text(json.dumps({"t": t, "max_accounts_same_text": int(rep.max())}))
    win.loc[win.gid.isin(todo), "cmt_ref"] = win.cmt
    return df


def poll_categories(api: Api, st: Path, t: int) -> None:
    """Total items per category, by bisecting for the last non-empty page. Once a day."""
    f = st / "categories.json"
    s = json.loads(f.read_text()) if f.exists() else {"log": []}
    if s["log"] and t - s["log"][-1]["t"] < 20 * 3600:
        return
    prev = s["log"][-1]["n"] if s["log"] else {}

    def total(cls: int) -> int:
        guess = int(prev.get(str(cls), 0))
        lo, hi = 0, max(60, guess + 3000)
        while api.list_page(cls, 127, hi, n=1):
            lo, hi = hi, hi * 2
        if guess > lo and api.list_page(cls, 127, guess, n=1):
            lo = guess
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if api.list_page(cls, 127, mid, n=1):
                lo = mid
            else:
                hi = mid
        return lo
    n = {}
    for cls in [0, 18] + CATEGORIES:
        try:
            n[str(cls)] = total(cls)
        except ApiDown:
            pass
    try:
        key = api.post("Cloud/GetMatchInfo", auth=False).get("MatchKey")
    except ApiDown:
        key = None
    s["log"].append({"t": t, "n": n, "match": key})
    s["log"] = s["log"][-800:]
    f.write_text(json.dumps(s))


def follow_up_held(api: Api, st: Path, t: int) -> None:
    """Once a day, re-read uploads that were held for review to record the outcome."""
    w = read_parquet(st / "walk.parquet")
    f = st / "held.json"
    last = json.loads(f.read_text())["t"] if f.exists() else 0
    if w is None or t - last < 20 * 3600:
        return
    todo = w[(w.chk == 1) & w.t_res.isna() & (w.priv == 0) & (w.t > t - 8 * DAY)].gid.tolist()[:400]
    for gid, r in zip(todo, api.pmap(api.gallery_info, todo)):
        if r.get("ReturnCode") != 0:
            w.loc[w.gid == gid, ["chk_final", "t_res"]] = [-1, t]
        elif int(r.get("CheckConfirm") or 0) != 1:
            w.loc[w.gid == gid, ["chk_final", "t_res"]] = [int(r["CheckConfirm"]), t]
    write_parquet(w, st / "walk.parquet")
    f.write_text(json.dumps({"t": t}))


# ------------------------------------------------------------------ build ----
def _people_share(h: pd.DataFrame) -> float:
    known = h.dlp.sum() + h.dla.sum()
    return float(h.dlp.sum() / known) if known > 0 else float("nan")


def build(st: Path, win: pd.DataFrame, t: int, authed: bool) -> None:
    out = {}
    last = json.loads((st / "last_pulse.json").read_text())
    h = read_parquet(st / "hourly.parquet")
    up = read_parquet(st / "uploads.parquet")
    share = _people_share(h[h.hour > t - 7 * DAY]) if h is not None else float("nan")

    # hourly series, last 21 days
    if h is not None:
        g = h[h.hour > t - 21 * DAY].groupby("hour", as_index=False)[["n", "dl", "dlp", "dla", "dlu", "dv", "dc", "dt"]].sum()
        g["hours"] = (g.dt / g.n / 3600).round(2)
        g["people"] = g.dlp + (g.dlu * share if share == share else 0)
        u = up[up.hour > t - 21 * DAY].groupby("hour").n.sum() if up is not None else pd.Series(dtype=float)
        g["uploads"] = g.hour.map(u).fillna(0).astype(int)
        out["hourly"] = recs(g.rename(columns={"hour": "t", "dl": "likes", "dla": "auto", "dv": "views", "dc": "comments"})[
            ["t", "uploads", "likes", "people", "auto", "views", "comments", "hours"]])

        # hour-by-weekday heatmaps (UTC); skip hours that followed a gap
        ok = h.groupby("hour", as_index=False)[["n", "dl", "dlp", "dlu", "dv", "dt"]].sum()
        ok = ok[(ok.dt / ok.n) < 5400]
        ts = pd.to_datetime(ok.hour, unit="s")
        ok = ok.assign(dow=ts.dt.dayofweek, hod=ts.dt.hour, people=ok.dlp + ok.dlu * (share if share == share else 0))
        heat = ok.groupby(["dow", "hod"], as_index=False).agg(likes=("people", "mean"), views=("dv", "mean"), hours=("hour", "size"))
        out["heat"] = recs(heat)
        if up is not None:
            ts = pd.to_datetime(up.hour, unit="s")
            uh = up.assign(dow=ts.dt.dayofweek, hod=ts.dt.hour).groupby(["hour", "dow", "hod"], as_index=False).n.sum()
            out["heat_uploads"] = recs(uh.groupby(["dow", "hod"], as_index=False).agg(uploads=("n", "mean")))

        # by canvas size, last 7 and 30 days
        rows = []
        for days in (7, 30):
            x = h[h.hour > t - days * DAY].groupby("size")[["dl", "dlp", "dla", "dlu", "dv"]].sum()
            ux = up[up.hour > t - days * DAY].groupby("size").n.sum() if up is not None else {}
            for s in SIZES:
                if s in x.index:
                    r = x.loc[s]
                    rows.append({"days": days, "size": SIZE_NAME[s], "uploads": int(ux.get(s, 0)), "likes": int(r.dl),
                                 "people": float(r.dlp + r.dlu * (share if share == share else 0)), "views": int(r.dv)})
        out["sizes"] = rows

    # attention curve by age
    a = read_parquet(st / "agecurve.parquet")
    if a is not None:
        a = a[a.photo == 0]
        rows = []
        for name, grp in (("all", a.groupby("b")), ("tier", a.groupby(["b", "tier"])), ("size", a.groupby(["b", "size"]))):
            g = grp[["n", "dl", "dlp", "dv", "dt"]].sum().reset_index()
            g["hrs"] = g.dt / 3600
            g = g[g.hrs > 0]
            g["likes"] = g.dl / g.hrs
            g["people"] = g.dlp / g.hrs
            g["views"] = g.dv / g.hrs
            g["by"] = name
            g["key"] = g["tier"] if name == "tier" else (g["size"].map(SIZE_NAME) if name == "size" else "all")
            rows += recs(g[["by", "key", "b", "n", "likes", "people", "views"]])
        out["curves"] = rows

    # promotions
    pe = read_parquet(st / "promo_events.parquet")
    pr = read_parquet(st / "promo.parquet")
    promo = {}
    if pe is not None and len(pe):
        pe = pe.assign(day=pd.to_datetime(pe.t_ev, unit="s").dt.strftime("%Y-%m-%d"), delay_h=(pe.t_ev - pe.date) / 3600)
        promo["per_day"] = recs(pe.groupby(["day", "kind"], as_index=False).size().rename(columns={"size": "n"}))
        promo["delay"] = recs(pe.groupby("kind").delay_h.describe(percentiles=[.25, .5, .75, .9]).reset_index()[["kind", "count", "25%", "50%", "75%", "90%"]])
        ts = pd.to_datetime(pe.t_ev, unit="s")
        promo["heat"] = recs(pe.assign(dow=ts.dt.dayofweek, hod=ts.dt.hour).groupby(["kind", "dow", "hod"], as_index=False).size().rename(columns={"size": "n"}))
        rec = pe[pe.kind == "rec"]
        if len(rec):
            age = (rec.t_ev - rec.date).clip(lower=3600) / 3600
            promo["before"] = {"n": int(len(rec)), "likes_per_hour": float((rec.like_at / age).mean()), "views_per_hour": float((rec.watch_at / age).mean())}
    if pr is not None:
        g = pr.groupby("rel", as_index=False)[["n", "dl", "dlp", "dv", "dt"]].sum()
        g["hrs"] = g.dt / 3600
        g = g[g.hrs > 0]
        promo["after"] = recs(g.assign(likes=g.dl / g.hrs, people=g.dlp / g.hrs, views=g.dv / g.hrs)[["rel", "n", "likes", "people", "views"]])
    out["promo"] = promo

    # every upload, listed or not
    w = read_parquet(st / "walk.parquet")
    if w is not None and len(w):
        w = w.assign(day=pd.to_datetime(w.date.where(w.date > 0, w.t), unit="s").dt.strftime("%Y-%m-%d"))
        w["kind"] = np.where(w.priv == 1, "private", np.where(w.isdel == 1, "removed", np.where(w.chk == 1, "held", "public")))
        pv = w.pivot_table(index="day", columns="kind", values="gid", aggfunc="count", fill_value=0).reset_index()
        res = w[w.t_res.notna() & (w.chk == 1)]
        review = {"resolved": int(len(res))}
        if len(res):
            review["approved_share"] = float((res.chk_final == 2).mean())
            review["median_wait_h"] = float(((res.t_res - res.date) / 3600).median())
        out["uploads_all"] = {"days": recs(pv), "review": review}

    # popular tab
    pp = read_parquet(st / "popular.parquet")
    if pp is not None and len(pp):
        pp = pp.assign(dwell_h=(pp.last_t - pp.first_t) / 3600, age_h=(pp.first_t - pp.date) / 3600,
                       tier=np.where(pp.rec == 1, "rec", np.where(pp.new == 1, "new", "none")))
        done = pp[pp.last_t < t - 6 * 3600]
        out["popular"] = {
            "tracked": int(len(pp)),
            "dwell": recs(done.groupby("lst").dwell_h.describe(percentiles=[.5, .9]).reset_index()[["lst", "count", "50%", "90%"]]) if len(done) else [],
            "entry": recs(pp.groupby("lst", as_index=False).agg(n=("gid", "size"), likes=("like0", "median"), views=("watch0", "median"),
                                                               age_days=("age_h", lambda s: s.median() / 24), rec_share=("rec", "mean"))),
            "tiers": recs(pp.groupby("tier", as_index=False).size().rename(columns={"size": "n"})),
        }

    # sign-ups
    f = st / "signups.json"
    if f.exists():
        log = json.loads(f.read_text())["log"]
        rows, cc = [], pd.Series(dtype=float)
        for i, e in enumerate(log):
            if i == 0:
                continue
            dt = e["t"] - log[i - 1]["t"]
            if dt <= 0 or not e["sampled"]:
                continue
            est = (e["hi"] - e["lo"]) * e["exist"] / e["sampled"]
            rows.append({"t": e["t"], "ids": e["hi"] - e["lo"], "accounts": round(est), "per_day": round(est * DAY / dt)})
            if e["t"] > t - 30 * DAY:
                cc = cc.add(pd.Series(e["cc"]), fill_value=0)
        out["signups"] = {"series": rows[-400:], "countries": {k: int(v) for k, v in small_cells(cc.sort_values(ascending=False)).head(25).items()},
                          "frontier": log[-1]["hi"] if log else None}

    # comments
    c = read_parquet(st / "comments.parquet")
    if c is not None and len(c):
        c = c.assign(day=pd.to_datetime(c.hour, unit="s").dt.strftime("%Y-%m-%d"))
        out["comments"] = {"days": recs(c.groupby("day", as_index=False)[["n", "people", "auto", "robot"]].sum()),
                           "scripts": {k: int(v) for k, v in c[c.hour > t - 30 * DAY].groupby("script").n.sum().items()}}

    # categories
    f = st / "categories.json"
    if f.exists():
        log = json.loads(f.read_text())["log"]
        out["categories"] = {"names": {str(k): v for k, v in {**CAT_NAME, 0: "NEW", 18: "Recommend"}.items()},
                             "log": [{"t": e["t"], "n": e["n"], "match": e.get("match")} for e in log[-400:]]}

    # top artists on recent uploads (named: only accounts in the published top-artist set)
    idx = DATA / "daily" / "artists" / "index.json"
    if idx.exists():
        top = {a["id"]: a["name"] for a in json.loads(idx.read_text(encoding="utf-8"))["artists"]}
        rows = []
        for days in (7, 30):
            x = win[(win.date > t - days * DAY) & win.uid.isin(top)]
            g = x.groupby("uid", as_index=False).agg(uploads=("gid", "size"), people=("lp", "sum"), likes=("like", "sum"), views=("watch", "sum"))
            g = g.sort_values("people", ascending=False).head(50)
            g["name"] = g.uid.map(top)
            rows += [{"days": days, **r} for r in recs(g.rename(columns={"uid": "id"}))]
        out["leaders"] = rows

    for k, v in out.items():
        write_json(f"{k}.json", v, SUB)
    write_json("status.json", {
        "schema": SCHEMA, "generated": iso(t), "last_pulse": iso(last["t"]), "mode": "normal" if authed else "degraded",
        "window_days": last["window_days"], "artworks_in_window": last["n_window"],
        "people_share_7d": None if share != share else round(share, 4),
        "automated_ranges": load_ranges(), "requests_last_pulse": last["requests"]}, SUB)


def run(api: Api, st: Path, main: Path, win: pd.DataFrame, t: int, authed: bool) -> None:
    for name, fn in (("signups", lambda: poll_signups(api, st, t)),
                     ("comments", lambda: poll_comments(api, st, main, win, t)),
                     ("categories", lambda: poll_categories(api, st, t) if authed else None),
                     ("held", lambda: follow_up_held(api, st, t) if authed else None)):
        try:
            fn()
        except ApiDown as exc:
            print(f"[refresh] {name} skipped: {exc}")
    write_parquet(win, st / "window.parquet")          # cmt_ref moved
    build(st, win, t, authed)
    (st / "last_refresh.json").write_text(json.dumps({"t": t}))
    print("[refresh] data/pulse rebuilt")
