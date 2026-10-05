"""Builds ``data/daily/*.json`` from the catalog and the snapshot's state tables.

Only aggregates leave this module. The one place accounts are named is the top-artist
set, and any cell counted in accounts needs at least five of them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .util import (CAT_NAME, DATA, DAY, PHOTO, SCHEMA, SIZE_NAME, SIZES, day_of, iso, load_ranges, read_parquet,
                   recs, small_cells, tier, write_json, write_parquet)

SUB = "daily"
SCRIPTS = ["none", "latin", "han", "kana", "hangul", "cyrillic", "arabic", "thai"]
AGE_NAME = ["0-30 days", "1-6 months", "6-12 months", "1-2 years", "2-4 years", "over 4 years"]
# Share of likes from the automated block by upload year, from the like-list sample of
# 2026-10-04 (2,440 artworks). Kept as a fixed baseline for the history page.
OCT2026_AUTO_SHARE = {2017: .056, 2018: .036, 2019: .047, 2020: .047, 2021: .047, 2022: .062, 2023: .084,
                      2024: .080, 2025: .213, 2026: .460}


def month(s) -> pd.Series:
    return pd.to_datetime(s, unit="s").dt.strftime("%Y-%m")


def id_to_date(cat: pd.DataFrame):
    """Approximate sign-up date for an account id, from the first uploads of nearby ids."""
    first = cat[cat.uid % 2 == 1].groupby("uid").date.min()
    b = (first.index // 2000 * 2000)
    mp = first.groupby(b).min().sort_index()
    mp = mp[::-1].cummin()[::-1]
    keys, vals = mp.index.to_numpy(), mp.to_numpy()

    def f(uids):
        u = np.asarray(uids, dtype="int64")
        i = np.searchsorted(keys, u // 2000 * 2000, side="left").clip(0, len(keys) - 1)
        return vals[i]
    return f, keys, vals


def build(st: Path, cat: pd.DataFrame, users: pd.DataFrame, acc, meta: pd.DataFrame, ctx: dict) -> None:
    t, since = ctx["t"], ctx["since"]
    today = day_of(t)
    live = cat[cat.gone == 0]
    live = live.assign(tier=tier(live), photo=(live.cls == PHOTO).astype(int), age=(t - live.date) / DAY)
    year = live[live.age <= 365]
    top = ctx["top"]
    topset = set(top.uid)
    names = users.set_index("uid").name
    warnings = ctx["warnings"]
    out = {}

    def part(name, fn):
        try:
            r = fn()
            if r is not None:
                out[name] = r
        except Exception as exc:
            print(f"[daily] {name} failed: {exc!r}")
            warnings.append(f"build:{name}")

    # ---- today's scalar series, appended to a long table -------------------
    S = []

    def put(metric, value, key=""):
        if value is not None and value == value:
            S.append((today, metric, str(key), float(value)))

    for d in (1, 7, 30, 365):
        put("uploaders", live[live.age <= d].uid.nunique(), d)
    put("uploads_24h", int((live.age <= 1).sum()))
    put("catalog_n", len(live))
    put("catalog_likes", int(live.like.sum()))
    put("catalog_views", int(live.watch.sum()))
    ev = ctx["ev"]
    if acc is not None and len(acc):
        ppl = acc[acc.auto == 0]
        for d in (1, 7, 30):
            put("likers", int((ppl.last_like > t - d * DAY).sum()), d)
        put("likers_new", int((ppl.first_like > since).sum()) if since else None)
        put("auto_accounts_30d", int(((acc.auto == 1) & (acc.last_like > t - 30 * DAY)).sum()))
    if len(ev) and since:
        pe = ev[ev.auto == 0]
        todays = pd.Series(pe.liker.unique())
        prev_last = ctx["prev_last"].reindex(todays).to_numpy()
        put("likers_returning_share", float(np.nanmean(prev_last > since - 7 * DAY)) if len(todays) else None)
        put("likers_uploader_share", float(todays.isin(set(cat.uid)).mean()))
        put("likes_by_uploaders_share", float(pe.liker.isin(set(cat.uid)).mean()))
        per = pe.groupby("liker").size()
        put("likes_top1pct_likers_share", float(per.nlargest(max(1, len(per) // 100)).sum() / per.sum()))
    pairs = read_parquet(st / "pairs.parquet")
    if pairs is not None and len(pairs) > 50:
        s = set(zip(pairs.liker, pairs.owner))
        put("reciprocity_30d", sum((b, a) in s for a, b in s) / len(s))
        put("pairs_30d", len(s))
    cm = ctx["cm"]
    if cm is not None and len(cm):
        put("old_comments", len(cm))
        put("old_comments_auto", int(cm.auto.sum()))
    u = year.groupby("uid").like.sum().sort_values(ascending=False)
    if len(u) > 200:
        put("conc_top1", u.head(len(u) // 100).sum() / max(u.sum(), 1))
        put("conc_top10", u.head(len(u) // 10).sum() / max(u.sum(), 1))
    gone_today = cat[(cat.gone == 1) & (cat.t_gone == t)]
    for k, v in gone_today.gone_kind.value_counts().items():
        put("vanished", int(v), k)
    series = read_parquet(st / "series.parquet")
    new = pd.DataFrame(S, columns=["day", "metric", "key", "value"])
    if not ctx["partial"] or series is None:
        series = new if series is None else pd.concat([series[series.day != today], new], ignore_index=True)
        write_parquet(series, st / "series.parquet")

    def ser(metric):
        x = series[series.metric == metric]
        return recs(x.pivot_table(index="day", columns="key", values="value").reset_index()) if len(x) else []

    # ---- daily flows -------------------------------------------------------
    dly = read_parquet(st / "daily.parquet")

    def flows():
        res = {"uploads": recs(live[live.age <= 400].assign(day=pd.to_datetime(live.date, unit="s").dt.strftime("%Y-%m-%d"))
                               .groupby("day", as_index=False).size().rename(columns={"size": "n"})),
               "uploaders": ser("uploaders"), "likers": ser("likers"), "catalog": {
                   "n": int(len(live)), "likes": int(live.like.sum()), "views": int(live.watch.sum())}}
        if dly is not None and len(dly):
            g = dly.groupby("day", as_index=False)[["n", "dl", "dlp", "dla", "dlu", "dv", "dc", "dt"]].sum()
            known = (g.dlp + g.dla).replace(0, np.nan)
            g["people"] = g.dlp + g.dlu * (g.dlp / known).fillna(0)
            g["hours"] = g.dt / g.n / 3600
            res["flows"] = recs(g.rename(columns={"dl": "likes", "dla": "auto", "dlu": "unattributed", "dv": "views", "dc": "comments"})[
                ["day", "likes", "people", "auto", "unattributed", "views", "comments", "hours"]])
            a = dly.groupby(["day", "ab"], as_index=False)[["dl", "dlp", "dla", "dv"]].sum()
            a["age"] = a.ab.map(lambda i: AGE_NAME[int(i)])
            res["by_age"] = recs(a[["day", "age", "dl", "dlp", "dla", "dv"]])
            res["catalog_by_age"] = {AGE_NAME[int(k)]: int(v) for k, v in pd.cut(live.age, [0, 30, 180, 365, 730, 1460, 1e6], right=False, labels=False).value_counts().items()}
        return res
    part("daily", flows)

    def sizes():
        x = live[(live.date > t - 1100 * DAY) & live["size"].isin(SIZES)].assign(m=lambda d: month(d.date))
        g = x.groupby(["m", "size"], as_index=False).agg(
            n=("gid", "size"), like_med=("like", "median"), watch_med=("watch", "median"), photo=("photo", "mean"),
            ai=("ai", "mean"), layer=("layer", "mean"), music=("music", "mean"), nocopy=("copy", "mean"),
            new=("new", "mean"), rec=("rec", "mean"), uploaders=("uid", "nunique"))
        g["size"] = g["size"].map(SIZE_NAME)
        res = {"cohorts": recs(g)}
        nonp = x[(x.photo == 0) & (x.age > 60) & (x.age <= 365)]
        q = nonp.groupby("size").agg(n=("gid", "size"), like_med=("like", "median"), like_p90=("like", lambda s: s.quantile(.9)),
                                     watch_med=("watch", "median"), new=("new", "mean"), rec=("rec", "mean"),
                                     uploaders=("uid", "nunique")).reset_index()
        q["size"] = q["size"].map(SIZE_NAME)
        res["mature"] = recs(q)
        if dly is not None and len(dly):
            f = dly[dly.day >= day_of(t - 30 * DAY)].groupby("size", as_index=False)[["dl", "dlp", "dla", "dv"]].sum()
            f = f[f["size"].isin(SIZES)]
            f["size"] = f["size"].map(SIZE_NAME)
            res["flows_30d"] = recs(f)
        return res
    part("sizes", sizes)

    def curation():
        x = live[(live.photo == 0) & (live.date > t - 800 * DAY) & live["size"].isin(SIZES)].assign(m=lambda d: month(d.date))
        g = x.groupby(["m", "size"], as_index=False).agg(n=("gid", "size"), new=("new", "mean"), rec=("rec", "mean"))
        g["size"] = g["size"].map(SIZE_NAME)
        c = live[(live.age <= 90) & (live.age > 14)].groupby("cls", as_index=False).agg(n=("gid", "size"), new=("new", "mean"), rec=("rec", "mean"))
        c["category"] = c.cls.map(CAT_NAME)
        res = {"by_month": recs(g), "by_category": recs(c.dropna(subset=["category"])[["category", "n", "new", "rec"]])}
        if dly is not None and len(dly):
            res["per_day"] = recs(dly.groupby("day", as_index=False)[["to_rec", "to_new", "refiled"]].sum())
        return res
    part("curation", curation)

    def survival():
        g = cat[cat.gone == 1]
        res = {"total_gone": int(len(g)), "kinds": {k: int(v) for k, v in g.gone_kind.value_counts().items()}}
        tr = cat[(cat.first_seen - cat.date) < 1.5 * DAY]            # followed since upload
        rows = []
        for d in (1, 7, 30, 90, 365):
            el = tr[(t - tr.date) >= d * DAY]
            if len(el) >= 200:
                goneby = (el.gone == 1) & ((el.t_gone - el.date) <= d * DAY)
                rows.append({"days": d, "n": int(len(el)), "gone": float(goneby.mean()),
                             **{k: float(((el.gone_kind == k) & goneby).mean()) for k in ("removed", "made private", "hidden", "no record")}})
        res["by_age"] = rows
        res["per_day"] = ser("vanished")
        return res
    part("survival", survival)

    def countries():
        c = small_cells(year.drop_duplicates("uid").uid.map(users.set_index("uid").cc).replace("", "??").value_counts())
        return {"uploaders_365d": {k: int(v) for k, v in c.head(40).items()}, "n": int(year.uid.nunique())}
    part("countries", countries)

    def retention():
        f = cat.groupby("uid").date.agg(["min", "count"])
        x = cat.merge(f["min"].rename("first"), left_on="uid", right_index=True)
        again30 = x[(x.date > x["first"]) & (x.date <= x["first"] + 30 * DAY)].uid.unique()
        again90 = x[(x.date > x["first"]) & (x.date <= x["first"] + 90 * DAY)].uid.unique()
        f = f.assign(m=month(f["min"]), a30=f.index.isin(again30), a90=f.index.isin(again90))
        g = f[f["min"] > t - 1500 * DAY].groupby("m", as_index=False).agg(n=("min", "size"), again30=("a30", "mean"), again90=("a90", "mean"))
        return {"cohorts": recs(g), "note": "from artworks still listed"}
    part("retention", retention)

    def funnel():
        if acc is None or not len(acc):
            return None
        f, keys, vals = id_to_date(cat)
        ppl = acc[acc.auto == 0].copy()
        ppl["signup"] = f(ppl.uid)
        ppl = ppl[ppl.signup > t - 200 * DAY]
        fu = cat.groupby("uid").date.min()
        ppl["first_upload"] = ppl.uid.map(fu)
        ppl["week"] = pd.to_datetime(ppl.signup, unit="s").dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")
        g = ppl.groupby("week", as_index=False).agg(likers=("uid", "size"), liked_7d=("first_like", lambda s: 0), uploaded=("first_upload", "count"))
        g["liked_7d"] = ppl[(ppl.first_like - ppl.signup) <= 7 * DAY].groupby("week").size().reindex(g.week).fillna(0).to_numpy()
        # accounts issued per week, from the id span of that week
        wk = pd.to_datetime(pd.Series(vals), unit="s").dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")
        span = pd.Series(keys).groupby(wk.to_numpy()).agg(lambda s: s.max() - s.min() + 2000)
        sig = DATA / "pulse" / "signups.json"
        exist = 0.62
        if sig.exists():
            log = [e for e in json.loads(sig.read_text())["series"] if e["ids"] > 0]
            if log:
                exist = sum(e["accounts"] for e in log) / sum(e["ids"] for e in log)
        g["accounts_est"] = (g.week.map(span).fillna(0) * exist).round()
        return {"weeks": recs(g), "existence_share": round(exist, 3),
                "note": "likers counted since tracking began; sign-up week is approximate"}
    part("funnel", funnel)

    def audience():
        res = {"likers": ser("likers"), "new_likers": ser("likers_new"), "returning": ser("likers_returning_share"),
               "uploader_share": ser("likers_uploader_share"), "likes_by_uploaders": ser("likes_by_uploaders_share"),
               "top1pct": ser("likes_top1pct_likers_share"), "reciprocity": ser("reciprocity_30d"),
               "automated_accounts": ser("auto_accounts_30d")}
        if len(ev):
            per = ev[ev.auto == 0].groupby("liker").size()
            res["likes_per_liker"] = {str(k): int(v) for k, v in pd.cut(per, [0, 1, 2, 5, 10, 50, 1e9], labels=["1", "2", "3-5", "6-10", "11-50", "51+"]).value_counts().sort_index().items()}
        return res
    part("audience", audience)

    def tags():
        if not len(meta):
            return None
        x = meta[meta.tags.map(len) > 0][["gid", "uid", "date", "tags"]].explode("tags")
        x = x.merge(cat[["gid", "like", "watch", "lp"]], on="gid", how="left")
        res = {}
        for d in (7, 30, 90):
            g = x[x.date > t - d * DAY].groupby("tags").agg(uploads=("gid", "nunique"), uploaders=("uid", "nunique"), likes=("like", "sum"), views=("watch", "sum"))
            g = g[g.uploaders >= 5].sort_values("uploads", ascending=False).head(60).reset_index().rename(columns={"tags": "tag"})
            res[f"top_{d}d"] = recs(g)
        a = x[x.date > t - 7 * DAY].groupby("tags").uid.nunique()
        b = x[(x.date <= t - 7 * DAY) & (x.date > t - 35 * DAY)].groupby("tags").uid.nunique() / 4
        r = pd.DataFrame({"now": a, "before": b}).fillna(0)
        r = r[r.now >= 5].assign(gain=lambda d: d.now - d.before).sort_values("gain", ascending=False).head(30).reset_index().rename(columns={"tags": "tag", "index": "tag"})
        res["rising"] = recs(r)
        return res
    part("tags", tags)

    def remix():
        x = live[live.date > t - 1100 * DAY].assign(m=lambda d: month(d.date))
        res = {"by_month": recs(x.groupby("m", as_index=False).agg(n=("gid", "size"), remix=("orig", lambda s: (s > 0).mean())))}
        r = live[live.orig > 0].groupby("orig").size().rename("remixes")
        o = cat[cat.gid.isin(r.index) & cat.uid.isin(topset)][["gid", "uid", "like"]].merge(r, left_on="gid", right_index=True)
        o["artist"] = o.uid.map(names)
        res["most_remixed"] = recs(o.sort_values("remixes", ascending=False).head(30)[["gid", "artist", "remixes", "like"]])
        if len(meta):
            mm = meta.assign(m=month(meta.date), has=meta["at"].map(len) > 0)
            res["mentions_by_month"] = recs(mm.groupby("m", as_index=False).agg(with_tags_or_mentions=("gid", "size"), mentions=("has", "sum")))
            e = meta[meta["at"].map(len) > 0][["uid", "at"]].explode("at")
            e = e[e.uid.isin(topset) & e["at"].isin(topset) & (e.uid != e["at"])]
            g = e.groupby(["uid", "at"]).size().rename("n").reset_index()
            g = g[g.n >= 2].sort_values("n", ascending=False).head(80)
            res["dedications"] = [{"from": names.get(a, ""), "to": names.get(b, ""), "n": int(n)} for a, b, n in g.itertuples(index=False)]
        s = x.assign(script=x.ts.map(lambda i: SCRIPTS[int(i)])).groupby(["m", "script"]).size().rename("n").reset_index()
        res["title_scripts"] = recs(s)
        return res
    part("remix", remix)

    def formats():
        f = read_parquet(st / "files_daily.parquet")
        if f is None or not len(f):
            return None
        f = f[f["size"].isin(SIZES)].assign(size=lambda d: d["size"].map(SIZE_NAME))
        tot = f.groupby("day", as_index=False)[["n", "decoded", "anim", "frames", "layer", "music", "dup_other", "dup_self", "near_other"]].sum()
        return {"per_day": recs(tot), "by_format": recs(f.groupby(["fmt", "size"], as_index=False)[["n", "decoded", "anim", "frames", "colors", "bytes"]].sum()),
                "since": str(f.day.min())}
    part("formats", formats)

    def history():
        x = live[live["size"].isin(SIZES)].assign(m=lambda d: month(d.date))
        up = x.groupby(["m", "size"]).size().rename("n").reset_index()
        up["size"] = up["size"].map(SIZE_NAME)
        f, keys, vals = id_to_date(cat)
        sm = pd.Series(np.full(len(keys), 2000), index=month(pd.Series(vals)).to_numpy()).groupby(level=0).sum() * 0.62
        yr = live.assign(y=pd.to_datetime(live.date, unit="s").dt.year).groupby("y", as_index=False).agg(n=("gid", "size"), likes=("like", "sum"), views=("watch", "sum"))
        yr["auto_share_oct2026"] = yr.y.map(OCT2026_AUTO_SHARE)
        return {"uploads": recs(up), "rec_picks": recs(live[live.rec == 1].assign(m=lambda d: month(d.date)).groupby("m").size().rename("n").reset_index()),
                "signups": [{"m": k, "accounts": int(v)} for k, v in sm.items()], "by_upload_year": recs(yr),
                "measured_from": day_of(read_first_day(st, t))}
    part("history", history)

    def benchmarks():
        x = live[live["size"].isin(SIZES) & (live.age > 7)].copy()
        x["group"] = np.where(x.photo == 1, "photo", x.tier)
        x["age_band"] = np.where(x.age <= 60, "7-60 days", "over 60 days")
        qs = [.1, .25, .5, .75, .9, .99]
        rows = []
        for (s, g, a), d in x.groupby(["size", "group", "age_band"]):
            if len(d) >= 50:
                rows.append({"size": int(s), "group": g, "age": a, "n": int(len(d)),
                             "likes": [float(v) for v in d.like.quantile(qs)], "views": [float(v) for v in d.watch.quantile(qs)]})
        return {"quantiles": qs, "rows": rows, "generated": iso(t)}
    part("benchmarks", benchmarks)

    def artists():
        adir = DATA / SUB / "artists"
        adir.mkdir(parents=True, exist_ok=True)
        hist = read_parquet(st / "artist_daily.parquet")
        latest = hist.sort_values("day").groupby("uid").tail(1).set_index("uid") if hist is not None else pd.DataFrame()
        mine = cat[cat.uid.isin(topset)]
        yr = mine[(t - mine.date) <= 365 * DAY]
        agg = yr.groupby("uid").agg(uploads=("gid", "size"), likes=("like", "sum"), views=("watch", "sum"), picks=("rec", "sum"),
                                    people=("lp", "sum"), auto=("la", "sum"))
        idx = []
        for r in top.itertuples():
            a = agg.loc[r.uid] if r.uid in agg.index else None
            p = latest.loc[r.uid] if r.uid in latest.index else None
            idx.append({"id": int(r.uid), "name": r.name, "cc": r.cc, "level": int(p.level) if p is not None and p.level == p.level else int(r.level),
                        "fans": None if p is None else int(p.fans or 0), "score": None if p is None else int(p.score or 0),
                        "uploads": 0 if a is None else int(a.uploads), "likes": 0 if a is None else int(a.likes),
                        "views": 0 if a is None else int(a.views), "picks": 0 if a is None else int(a.picks),
                        "people": 0 if a is None else int(a.people), "avatar": (adir / f"{r.uid}.webp").exists()})
        write_json("artists/index.json", {"artists": idx, "generated": iso(t)}, SUB)
        mm = mine.assign(m=month(mine.date))
        bym = mm[mm.date > t - 1100 * DAY].groupby(["uid", "m"]).agg(n=("gid", "size"), likes=("like", "sum"), views=("watch", "sum"), picks=("rec", "sum")).reset_index()
        for uid, g in bym.groupby("uid"):
            h = hist[hist.uid == uid] if hist is not None else None
            write_json(f"artists/{int(uid)}.json", {
                "id": int(uid), "months": recs(g.drop(columns=["uid"])),
                "daily": recs(h[["day", "fans", "score", "level", "up_day", "dl_day", "dlp_day", "dv_day"]]) if h is not None and len(h) else []}, SUB)
        for f in adir.glob("*.json"):
            if f.stem != "index" and int(f.stem) not in topset:
                f.unlink()
        return None
    part("artists", artists)

    def score():
        hist = read_parquet(st / "artist_daily.parquet")
        if hist is None or hist.day.nunique() < 15:
            return {"ready": False, "days": 0 if hist is None else int(hist.day.nunique())}
        h = hist.sort_values(["uid", "day"])
        h["dscore"] = h.groupby("uid").score.diff()
        h = h.dropna(subset=["dscore"])
        h = h[(h.dscore >= 0) & (h.dscore < h.dscore.quantile(.995))]
        feats = ["up_day", "dl_day", "dc_day", "rec_day", "new_day"]
        X = np.column_stack([h[f].to_numpy(float) for f in feats] + [np.ones(len(h))])
        coef, *_ = np.linalg.lstsq(X, h.dscore.to_numpy(float), rcond=None)
        pred = X @ coef
        r2 = 1 - ((h.dscore - pred) ** 2).sum() / ((h.dscore - h.dscore.mean()) ** 2).sum()
        return {"ready": True, "n": int(len(h)), "r2": float(r2),
                "points": {k: float(v) for k, v in zip(["per upload", "per like received", "per comment received", "per Recommend pick", "per NEW pick", "per day (baseline)"], coef)}}
    part("score", score)

    def study():
        x = cat[cat.lp21.notna() & cat.lp7.notna() & (cat.cls != PHOTO)].copy()
        if len(x) < 300:
            return {"ready": False, "n": int(len(x))}
        x["later"] = x.lp21 - x.lp7
        x["vb"] = pd.qcut(x.v7.rank(method="first"), 5, labels=False)
        x["tier"] = tier(x)
        rows = []
        for (sz, tr, vb), g in x.groupby(["size", "tier", "vb"]):
            if len(g) >= 30 and g.la7.nunique() > 1:
                lo, hi = g[g.la7 <= g.la7.median()], g[g.la7 > g.la7.median()]
                if len(lo) >= 10 and len(hi) >= 10:
                    rows.append((len(g), hi.later.mean() - lo.later.mean(), lo.later.mean(), hi.la7.mean() - lo.la7.mean()))
        if not rows:
            return {"ready": False, "n": int(len(x))}
        r = np.array(rows)
        w = r[:, 0] / r[:, 0].sum()
        return {"ready": True, "n": int(len(x)), "strata": int(len(r)), "extra_auto_likes": float((r[:, 3] * w).sum()),
                "baseline_later_people_likes": float((r[:, 2] * w).sum()), "difference": float((r[:, 1] * w).sum()),
                "se": float(np.sqrt(((r[:, 1] - (r[:, 1] * w).sum()) ** 2 * w).sum() / max(len(r) - 1, 1)))}
    part("study", study)

    def calendar():
        f = st / "calendar.json"
        return {"events": json.loads(f.read_text())["events"][-300:]} if f.exists() else None
    part("calendar", calendar)

    def methods():
        res = {"ranges": load_ranges(), "people_by_upload_year_oct2026": OCT2026_AUTO_SHARE,
               "old_comments": ser("old_comments"), "old_comments_auto": ser("old_comments_auto"),
               "concentration": {"top1": ser("conc_top1"), "top10": ser("conc_top10")}}
        tr = cat[(cat.lp + cat.la) > 0]
        if len(tr):
            res["tracked_likes"] = {"people": int(tr.lp.sum()), "automated": int(tr.la.sum())}
        return res
    part("methods", methods)

    for k, v in out.items():
        write_json(f"{k}.json", v, SUB)
    write_json("status.json", {"schema": SCHEMA, "generated": iso(t), "snapshot": iso(t), "artworks": int(len(live)),
                               "requests": ctx["requests"], "first_snapshot": ctx["bootstrap"], "partial": ctx["partial"],
                               "top_artists": int(len(top)), "warnings": warnings}, SUB)


def read_first_day(st: Path, t: int) -> int:
    f = st / "first_day.json"
    if f.exists():
        return json.loads(f.read_text())["t"]
    f.write_text(json.dumps({"t": t}))
    return t
