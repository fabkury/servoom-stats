"""The refresh job: the slow hourly reads the pulse cannot afford, and ``data/pulse``.

It runs on its own runner right after the hourly pulse (pulse.yml), so it has its own
budget of 30 requests a minute and its own account (role ``refresh``), and it must end
before the next pulse starts: its reading deadline is MARGIN seconds before the next
``PULSE_MINUTE``. In that time, in this order, it reads

* the id walk: every artwork id issued since the last run, listed or not (the only
  count of private and held uploads);
* new comments on window artworks (no text is kept, only fingerprints);
* a sample of new accounts.

Then, every REFRESH_EVERY (or with ``FORCE_REFRESH=1``), it rebuilds ``data/pulse`` and
the pulse health lanes from the pulse's tables and its own. The rebuild needs no
request, so it happens whether or not the reads fit: the site is never left stale
because the hour was busy. Until 2026-10-27 all of this ran inside the pulse, where the
like lists always used the budget up first.

State lives on the branch ``state-refresh`` (``walk.*``, ``cmt_ref.parquet``,
``signups.json``, ``comments.parquet``, ``last_refresh.json``, ``runs.parquet``); the
pulse's branch is read, never written. The first run seeds its files from the pulse's.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import health, rawrepo
from .accounts import Halted, Pool, flush_issues
from .api import Api, ApiDown, OutOfTime
from .util import (CAT_NAME, CATEGORIES, DATA, DAY, SCHEMA, SIZE_NAME, SIZES, add_rollup, env_flag, fingerprint,
                   is_auto, iso, load_ranges, now, read_parquet, recs, script_of, small_cells, write_json,
                   write_parquet)

SUB = "pulse"
MIN_GAP = 50 * 60                   # a run is skipped when the previous one is younger than this
REFRESH_EVERY = 3.75 * 3600         # data/pulse is rebuilt by the first run after this long
PULSE_MINUTE = int(os.environ.get("PULSE_MINUTE", "17"))        # when the Worker starts the pulse
MARGIN = int(os.environ.get("REFRESH_MARGIN_SECONDS", "150"))    # reads stop this long before it
MIN_BUDGET = 120                    # less than this and the run reads nothing, only builds
# Share of the time still left that a phase may use before the next one starts, so a
# long walk (after a gap, up to 1,200 ids) cannot starve the comments and the sign-ups.
# A phase that ends early hands what it did not use to the next ones.
WALK_SHARE, COMMENTS_SHARE = 0.6, 0.6
BUDGET_CAP = int(os.environ.get("REFRESH_BUDGET_SECONDS", "0")) # tests: a shorter budget
SEEDED = ["walk.json", "walk.parquet", "signups.json", "comments.parquet", "comment_repeats.json",
          "last_refresh.json"]


def reading_budget(t: int) -> int:
    """Seconds until MARGIN before the next pulse starts."""
    nxt = t // 3600 * 3600 + PULSE_MINUTE * 60
    if nxt <= t:
        nxt += 3600
    budget = nxt - t - MARGIN
    if BUDGET_CAP:
        budget = min(budget, BUDGET_CAP)
    return max(budget, 0)


def seed_from_pulse(st: Path, sp: Path) -> None:
    """First run: take over the files the pulse kept until 2026-10-27."""
    for name in SEEDED:
        if not (st / name).exists() and (sp / name).exists():
            shutil.copy(sp / name, st / name)
            print(f"[refresh] seeded {name} from the pulse branch")


# ---------------------------------------------------------------- polling ----
WALK_COLS = ["gid", "t", "date", "priv", "isdel", "chk", "listed"]


def id_walk(api: Api, st: Path, listed: set, t: int, authed: bool, rows: list) -> None:
    """Read every artwork id issued since the last run, listed or not, into ``rows``.
    A cut (OutOfTime) propagates; the rows read so far and the frontier are kept."""
    wf = st / "walk.json"
    state = json.loads(wf.read_text()) if wf.exists() else {}
    if not authed or not listed:
        return
    frontier = state.get("frontier") or max(listed)
    misses, g = 0, frontier + 1
    try:
        while misses < 24 and g - frontier < 1200:
            batch = list(range(g, g + 8))
            for gid, r in zip(batch, api.pmap(api.gallery_info, batch)):
                if r.get("ReturnCode") != 0:
                    misses += 1
                    continue
                misses = 0
                priv = int(r.get("PrivateFlag") or 0)
                date = int(r.get("Date") or 0)
                if priv:                       # private upload: count it, keep nothing else
                    date = date // 3600 * 3600
                rows.append((gid, t, date, priv, int(r.get("IsDel") or 0), int(r.get("CheckConfirm") or 0),
                             int(gid in listed)))
            g += 8
    finally:                                   # a cut walk keeps what it read and resumes next run
        if rows:
            state["frontier"] = max(r[0] for r in rows)
        elif "frontier" not in state:
            state["frontier"] = frontier
        wf.write_text(json.dumps(state))


def apply_held(st: Path, pr: Path) -> None:
    """Take over from the daily probe the outcome of uploads that were held for review."""
    h = read_parquet(pr / "held.parquet")
    w = read_parquet(st / "walk.parquet")
    if h is None or not len(h) or w is None or not len(w):
        return
    res = h.drop_duplicates("gid", keep="last").set_index("gid")
    idx = w.gid.isin(res.index) & w.t_res.isna()
    if idx.any():
        w.loc[idx, "chk_final"] = w.loc[idx, "gid"].map(res.chk_final).to_numpy()
        w.loc[idx, "t_res"] = w.loc[idx, "gid"].map(res.t_res).to_numpy()
        write_parquet(w, st / "walk.parquet")


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
    ids = random.sample(range(lo + 1, hi + 1), min(100, span))
    infos = api.pmap(api.user_info, ids)
    ok = [(u, r) for u, r in zip(ids, infos) if r.get("ReturnCode") == 0]
    cc = pd.Series([r.get("CountryISOCode") or "??" for _, r in ok]).value_counts()
    s["log"].append({"t": t, "lo": lo, "hi": hi, "sampled": len(ids), "exist": len(ok),
                     "cc": {k: int(v) for k, v in cc.items()}})
    s["log"] = s["log"][-2500:]
    s["frontier"] = hi
    f.write_text(json.dumps(s))


def poll_comments(api: Api, st: Path, main: Path, win: pd.DataFrame, t: int, since: int) -> pd.DataFrame:
    """Comments written since ``since`` on window artworks. No text is kept.

    ``cmt_ref.parquet`` remembers, per artwork, the comment count already read, so an
    artwork is re-read when its count rose or when the listing says a comment is newer
    than the last run. (The pulse's window carries a ``cmt_ref`` column from before this
    job existed; it seeds the table and is otherwise unused.)
    """
    salt_file = main / "state" / "salt.txt"
    salt = salt_file.read_text().strip() if salt_file.exists() else "unsalted"
    rf = st / "cmt_ref.parquet"
    ref = read_parquet(rf)
    w = win[["gid", "cmt", "cmtutc"]].copy()
    w["ref"] = w.gid.map(ref.set_index("gid").cmt_ref) if ref is not None else np.nan
    if "cmt_ref" in win.columns:
        w["ref"] = w.ref.fillna(win["cmt_ref"])
    w["ref"] = w.ref.fillna(0)
    todo = w[(w.cmt > w.ref) | (w.cmtutc > since)].gid.head(150).tolist()
    ranges = load_ranges()
    rows, done = [], []
    try:
        for gid, cl in zip(todo, api.pmap(api.comments, todo)):
            done.append(gid)
            for c in cl:
                d = int(c.get("Date") or 0)
                if d <= since:
                    continue
                txt = c.get("Comment") or ""
                rows.append((gid, int(c.get("CommentId") or 0), int(c["UserId"]), d, fingerprint(txt, salt), len(txt),
                             script_of(txt), int(str(c.get("RobertFlag") or "0") != "0")))
    finally:
        # remember the counts of the artworks read, whole or cut; forget artworks out of the window
        read = w[w.gid.isin(done)][["gid", "cmt"]].rename(columns={"cmt": "cmt_ref"})
        keep = ref[~ref.gid.isin(done) & ref.gid.isin(win.gid)] if ref is not None else None
        write_parquet(pd.concat(([keep] if keep is not None else []) + [read], ignore_index=True), rf)
    df = pd.DataFrame(rows, columns=["gid", "cid", "uid", "date", "fp", "len", "script", "robot"])
    if len(df):
        df["auto"] = is_auto(df.uid, ranges).astype("int8")
        write_parquet(df, main / "obs" / "pulse" / time.strftime("%Y/%m/%d", time.gmtime(t)) / f"{time.strftime('%H%M', time.gmtime(t))}-comments.parquet")
        g = df.assign(hour=df.date // 3600 * 3600, n=1, people=1 - df.auto.clip(upper=1))
        add_rollup(st / "comments.parquet", g.groupby(["hour", "script"], as_index=False)[["n", "people", "auto", "robot"]].sum(), ["hour", "script"])
        # repeated fingerprints from different accounts hint at a campaign
        rep = df.groupby("fp").uid.nunique()
        (st / "comment_repeats.json").write_text(json.dumps({"t": t, "max_accounts_same_text": int(rep.max())}))
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


# ------------------------------------------------------------------ build ----
def _people_share(h: pd.DataFrame) -> float:
    known = h.dlp.sum() + h.dla.sum()
    return float(h.dlp.sum() / known) if known > 0 else float("nan")


def build(sp: Path, st: Path, win: pd.DataFrame, t: int) -> None:
    """``data/pulse/*.json`` from the pulse's tables (``sp``) and this job's (``st``)."""
    out = {}
    last = json.loads((sp / "last_pulse.json").read_text())
    authed = last.get("mode", "normal") == "normal"
    h = read_parquet(sp / "hourly.parquet")
    up = read_parquet(sp / "uploads.parquet")
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
    a = read_parquet(sp / "agecurve.parquet")
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
    pe = read_parquet(sp / "promo_events.parquet")
    pr = read_parquet(sp / "promo.parquet")
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
        # hrs lets the page re-bin the hourly buckets with the right weights
        promo["after"] = recs(g.assign(likes=g.dl / g.hrs, people=g.dlp / g.hrs, views=g.dv / g.hrs)[["rel", "n", "hrs", "likes", "people", "views"]])
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
    pp = read_parquet(sp / "popular.parquet")
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
    f = sp / "categories.json"
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



# -------------------------------------------------------------------- job ----
def run() -> None:
    t = now()
    hour = t // 3600 * 3600
    st = rawrepo.clone_state("state-refresh")
    lr = st / "last_run.json"
    age = t - json.loads(lr.read_text())["t"] if lr.exists() else None
    if age is not None and age < MIN_GAP and not env_flag("FORCE_REFRESH"):
        print(f"[refresh] the last run was {age // 60} minutes ago; nothing to do")
        return
    main = rawrepo.clone_main(["state"])
    sp = rawrepo.clone_state("state-pulse")
    pr = rawrepo.clone_state("state-probe")
    seed_from_pulse(st, sp)
    win = read_parquet(sp / "window.parquet")
    if win is None:
        print("[refresh] the pulse has no window yet; nothing to do")
        return
    since = t - int(age) if age is not None else t - 4 * 3600

    budget = reading_budget(t)
    api = Api(workers=4)
    api.set_budget(budget)
    final = api.deadline

    def allot(share: float) -> None:
        """Let the next phase use ``share`` of the time left (all of it with share 1)."""
        left = max(0.0, final - time.monotonic())
        api.deadline = final - (1 - share) * left

    skipped = []
    authed = budget >= MIN_BUDGET
    if authed:
        try:
            Pool(main).acquire(api, "refresh")
        except (Halted, ApiDown) as exc:        # the polls that need no token still run
            print(f"[refresh] running without a token: {exc}")
            api.auth, authed = {}, False
    else:
        print(f"[refresh] only {budget}s before the next pulse; reading nothing this hour")
        skipped.append("time")
    apply_held(st, pr)

    def phase(name, fn):
        try:
            return fn()
        except OutOfTime:
            skipped.append(name)
            print(f"[refresh] out of time during {name}")
        except ApiDown as exc:
            skipped.append(name)
            print(f"[refresh] {name} skipped: {exc}")

    walked: list = []
    allot(WALK_SHARE)
    phase("walk", lambda: id_walk(api, st, set(win.gid), t, authed, walked))
    walk = pd.DataFrame(walked, columns=WALK_COLS)
    if len(walk):
        w = read_parquet(st / "walk.parquet")
        w = pd.concat(([w] if w is not None else []) + [walk.assign(chk_final=np.nan, t_res=np.nan)], ignore_index=True)
        write_parquet(w[w.t > t - 45 * DAY].drop_duplicates("gid"), st / "walk.parquet")
        write_parquet(walk, main / "obs" / "pulse" / time.strftime("%Y/%m/%d", time.gmtime(t)) / f"{time.strftime('%H%M', time.gmtime(t))}-walk.parquet")
    allot(COMMENTS_SHARE)
    phase("comments", lambda: poll_comments(api, st, main, win, t, since))
    allot(1.0)
    phase("signups", lambda: poll_signups(api, st, t))

    lf = st / "last_refresh.json"
    refresh_age = t - json.loads(lf.read_text())["t"] if lf.exists() else None
    refreshed = 0                       # 1 the data was rebuilt, -1 the rebuild failed, 0 not due
    if refresh_age is None or refresh_age > REFRESH_EVERY or env_flag("FORCE_REFRESH"):
        try:
            build(sp, st, win, t)
            lf.write_text(json.dumps({"t": t}))
            refreshed = 1
            print("[refresh] data/pulse rebuilt")
        except Exception as exc:        # a failed rebuild must not lose the reads
            refreshed = -1
            print(f"[refresh] build failed: {exc!r}")
            if env_flag("STRICT"):
                raise
    health.log_run(st, {"t": t, "hour": hour, "mode": "normal" if authed else "degraded", "requests": api.n_requests,
                        "limited": api.n_limited, "served": api.n_served, "seconds": now() - t, "budget": int(budget),
                        "refresh": refreshed, "skipped": ",".join(skipped)})
    if refreshed:
        try:
            health.build_pulse(sp, st, t)
        except Exception as exc:
            print(f"[health] failed: {exc!r}")
            if env_flag("STRICT"):
                raise
    lr.write_text(json.dumps({"t": t}))
    rawrepo.push_state("state-refresh", f"refresh {time.strftime('%Y-%m-%d %H:%M', time.gmtime(t))}")
    rawrepo.push_main(f"refresh {time.strftime('%Y-%m-%d %H', time.gmtime(t))}")
    flush_issues(Path("_issues"))
    print(f"[refresh] done in {now() - t}s, {api.summary()}, budget {budget}s"
          + (f", cut: {', '.join(skipped)}" if skipped else "") + (", data rebuilt" if refreshed == 1 else ""))


if __name__ == "__main__":
    try:
        run()
    except ApiDown as exc:
        print(f"[refresh] server unreachable, ending quietly: {exc}")
        sys.exit(0)
