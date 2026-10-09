"""Daily snapshot: the whole public catalog, who gave each new like, new files, vanished
artworks, top-artist profiles and Divoom's calendar. Then ``data/daily`` is rebuilt.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import daily, files, rawrepo
from .accounts import Halted, Pool, flush_issues, note_issue
from .api import Api, ApiDown
from .util import (ALL_SIZES, CATEGORIES, CONFIG, DATA, DAY, PHOTO, ROW_FIELDS, add_rollup, day_of, env_flag,
                   fingerprint, is_auto, load_ranges, now, read_parquet, script_of, tier, write_parquet)

pd.set_option("future.no_silent_downcasting", True)
SCRIPTS = ["none", "latin", "han", "kana", "hangul", "cyrillic", "arabic", "thai"]
INT = list(ROW_FIELDS) + ["music", "layer", "ts", "cs"]
STATE_COLS = ["first_seen", "miss", "lp", "la", "lp7", "la7", "v7", "lp21", "gone", "gone_kind", "t_gone"]
AGE = [0, 30, 180, 365, 730, 1460, 1e6]


class Collector:
    """Turns records into compact rows while the crawl runs, to keep memory flat."""

    def __init__(self, t: int):
        self.t, self.users, self.fids, self.meta = t, {}, {}, []

    def take(self, x):
        gid, uid, date = x["GalleryId"], x["UserId"], x["Date"]
        self.users[uid] = (x.get("UserName") or "", x.get("CountryISOCode") or "", int(x.get("Level") or 0),
                           1 if x.get("PixelAmbName") else 0, x.get("UserHeaderId") or "")
        if date > self.t - 3 * DAY:
            self.fids[gid] = x.get("FileId")
        if date > self.t - 130 * DAY:
            tags = [str(s)[:40].lower() for s in (x.get("FileTagArray") or [])][:12]
            at = [int(a["AtUserId"]) for a in (x.get("AtList") or []) if str(a.get("AtUserId", "")).isdigit()]
            if tags or at:
                self.meta.append((gid, uid, date, tags, at))
        return tuple(int(x.get(v) or 0) for v in ROW_FIELDS.values()) + (
            1 if x.get("MusicFileId") else 0, 1 if x.get("LayerFileId") else 0,
            SCRIPTS.index(script_of(x.get("FileName"))), SCRIPTS.index(script_of(x.get("Content"))))


def list_plan() -> list:
    """The (category, size) lists a crawl reads, in order."""
    limit = os.environ.get("LIMIT_LISTS")
    return [tuple(map(int, s.split("_"))) for s in limit.split(",")] if limit else \
        [(c, s) for c in CATEGORIES for s in ALL_SIZES]


def crawl(api: Api, col: Collector, lists: list, seed: Optional[pd.DataFrame] = None, start: int = 0,
          save=None) -> pd.DataFrame:
    """Read ``lists[start:]``; ``seed`` holds the raw rows of the lists before ``start``.

    With ``save``, the raw rows read so far are checkpointed every CHECKPOINT_EVERY
    seconds, so a run that dies (the first list alone takes over an hour) can resume.
    """
    frames = [seed] if seed is not None and len(seed) else []
    rows, t0, t_save = [], now(), now()
    for i in range(start, len(lists)):
        cls, size = lists[i]
        rows += api.crawl_list(cls, size, transform=col.take)
        if i % 20 == 0:
            print(f"[snapshot] list {i}/{len(lists)}, {len(rows) + sum(map(len, frames))} rows, "
                  f"{api.n_requests} requests, {now() - t0}s", flush=True)
        if save is not None and i + 1 < len(lists) and now() - t_save >= CHECKPOINT_EVERY:
            if rows:
                frames.append(pd.DataFrame(rows, columns=INT))
                rows = []
            save(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(rows, columns=INT),
                 col, i + 1, len(lists))
            t_save = now()
    if rows:                               # empty frames are left out so the int columns keep their dtype
        frames.append(pd.DataFrame(rows, columns=INT))
    df = (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(rows, columns=INT)).drop_duplicates("gid")
    return df[df.priv == 0].drop(columns=["priv"]).reset_index(drop=True)


CHECKPOINT_MAX_AGE = 6 * 3600
# The catalog is read every two days (decided 2026-10-09 to halve the request volume
# per polling account). The worker still dispatches daily; a run sooner than this
# after the last snapshot exits at once.
SNAPSHOT_GAP = int(os.environ.get("SNAPSHOT_MIN_GAP_HOURS", "36")) * 3600
CHECKPOINT_EVERY = 20 * 60


def save_checkpoint(cur: pd.DataFrame, col: Collector, done: int, total: int, branch: str = "state-crawl") -> None:
    """Keep the crawl in the private repository: the raw rows of the first ``done`` of
    ``total`` lists while it runs, the finished table at the end. A run that dies resumes
    from it instead of reading the catalog again for three hours."""
    try:
        ck = rawrepo.WORK / branch
        if not ck.exists():
            ck = rawrepo.clone_state(branch)
        for f in ck.iterdir():
            if f.name != ".git":
                f.unlink()
        write_parquet(cur, ck / "crawl.parquet")
        write_parquet(pd.DataFrame([(u, *v) for u, v in col.users.items()], columns=["uid", "name", "cc", "level", "amb", "head"]), ck / "users.parquet")
        write_parquet(pd.DataFrame(col.meta, columns=["gid", "uid", "date", "tags", "at"]), ck / "meta.parquet")
        (ck / "fids.json").write_text(json.dumps(col.fids))
        (ck / "info.json").write_text(json.dumps({"t": col.t, "rows": int(len(cur)), "done": done, "total": total}))
        rawrepo.push_state(branch, f"crawl checkpoint {day_of(col.t)}" + ("" if done >= total else f" ({done}/{total} lists)"))
        print(f"[snapshot] crawl checkpoint saved, {len(cur)} rows, {done}/{total} lists")
    except Exception as exc:                 # a checkpoint is a convenience; never fail the run for it
        print(f"[snapshot] could not save the crawl checkpoint: {exc!r}")


def load_checkpoint(now_t: int, last_t: int, total: int, branch: str = "state-crawl"):
    """A saved crawl that is recent and was not yet turned into a snapshot, or None.
    Returns (rows, collector, t, done): the finished table when ``done == total``, else
    the raw rows of the first ``done`` lists."""
    try:
        ck = rawrepo.clone_state(branch)
        info_file = ck / "info.json"
        if not info_file.exists():
            return None
        info = json.loads(info_file.read_text())
        if now_t - info["t"] > CHECKPOINT_MAX_AGE or info["t"] <= last_t:
            return None
        done = info.get("done", total)
        if info.get("total", total) != total:     # the plan changed; a partial crawl is useless
            return None
        cur = pd.read_parquet(ck / "crawl.parquet")
        if len(cur) != info["rows"]:
            return None
        col = Collector(info["t"])
        u = pd.read_parquet(ck / "users.parquet")
        col.users = {r[0]: tuple(r[1:]) for r in u.itertuples(index=False, name=None)}
        col.fids = {int(k): v for k, v in json.loads((ck / "fids.json").read_text()).items()}
        mt = pd.read_parquet(ck / "meta.parquet")
        col.meta = [(g, uid, d, list(tags), [int(x) for x in at]) for g, uid, d, tags, at in mt.itertuples(index=False, name=None)]
        return cur, col, info["t"], min(done, total)
    except Exception as exc:
        print(f"[snapshot] crawl checkpoint not usable: {exc!r}")
        return None


def pulse_events(main: Path, since: int) -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in (main / "obs" / "pulse").rglob("*-likes.parquet")]
    ev = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["gid", "liker", "t", "t_prev", "auto"])
    return ev[ev.t > since]


def fetch_events(api: Api, m: pd.DataFrame, t: int) -> pd.DataFrame:
    todo = m[(m.dl > 0) & (m.date < t - 30 * DAY) & (~m.isnew)].sort_values("dl", ascending=False).head(20000)
    ranges = load_ranges()

    def one(row):
        try:
            ul = api.like_list(int(row.gid), int(min(row.dl, 500)))
        except ApiDown:
            ul = None
        return [(int(row.gid), int(u["UserId"]), t, int(row.t_p)) for u in (ul or [])]
    ev = pd.DataFrame([e for part in api.pmap(one, list(todo.itertuples())) for e in part],
                      columns=["gid", "liker", "t", "t_prev"]).astype("int64")
    ev["auto"] = is_auto(ev.liker, ranges).astype("int8") if len(ev) else []
    return ev


def fetch_comments(api: Api, m: pd.DataFrame, t: int, since: int, salt: str) -> pd.DataFrame:
    todo = m[(m.dc > 0) & (m.date < t - 30 * DAY) & (~m.isnew)].gid.head(3000).tolist()
    rows = []
    for gid, cl in zip(todo, api.pmap(api.comments, todo)):
        for c in cl:
            d = int(c.get("Date") or 0)
            if d > since:
                txt = c.get("Comment") or ""
                rows.append((gid, int(c.get("CommentId") or 0), int(c["UserId"]), d, fingerprint(txt, salt), len(txt),
                             script_of(txt), int(str(c.get("RobertFlag") or "0") != "0")))
    df = pd.DataFrame(rows, columns=["gid", "cid", "uid", "date", "fp", "len", "script", "robot"])
    df["auto"] = is_auto(df.uid).astype("int8") if len(df) else []
    return df


def check_vanished(api: Api, cat: pd.DataFrame, seen: set, t: int) -> pd.DataFrame:
    """Artworks missing from two snapshots in a row: ask what became of them."""
    absent = (cat.gone == 0) & ~cat.gid.isin(seen)
    cat.loc[absent, "miss"] = cat.loc[absent, "miss"] + 1
    cat.loc[~absent & (cat.gone == 0), "miss"] = 0
    todo = cat[absent & (cat.miss >= 2)].gid.head(4000).tolist()
    rows = []
    for gid, r in zip(todo, api.pmap(api.gallery_info, todo)):
        if r.get("ReturnCode") == 11:
            break
        if r.get("ReturnCode") != 0:
            kind = "no record"
        elif int(r.get("PrivateFlag") or 0):
            kind = "made private"
        elif int(r.get("IsDel") or 0):
            kind = "removed"
        elif int(r.get("HideFlag") or 0):
            kind = "hidden"
        else:
            continue          # still public: a listing left it out for now, it is not gone
        rows.append((gid, kind))
    v = pd.DataFrame(rows, columns=["gid", "kind"])
    if len(v):
        idx = cat.gid.isin(v.gid)
        cat.loc[idx, "gone"] = 1
        cat.loc[idx, "t_gone"] = t
        cat.loc[idx, "gone_kind"] = cat.loc[idx, "gid"].map(v.set_index("gid").kind)
    return v


def process_files(api: Api, st: Path, new: pd.DataFrame, fids: dict, t: int) -> pd.DataFrame:
    hashes = read_parquet(st / "hashes.parquet")
    if hashes is None:
        hashes = pd.DataFrame({"gid": pd.Series(dtype="int64"), "uid": pd.Series(dtype="int64"), "size": pd.Series(dtype="int64"),
                               "date": pd.Series(dtype="int64"), "exact": pd.Series(dtype="str"), "dh": pd.Series(dtype="int64")})
    todo = new[new.gid.isin(fids) & ~new.gid.isin(hashes.gid)].sort_values("date").tail(2500)
    if files.PixelBeanDecoder is None:
        print("[snapshot] servoom decoders not importable; files skipped")
        return pd.DataFrame()

    # Download in parallel, decode here: the LZO decoder must stay on the thread that made it.
    rows = list(todo.itertuples())
    blobs = api.pmap(lambda row: api.get_file(fids[row.gid]) if fids.get(row.gid) else None, rows)
    feats = []
    for row, data in zip(rows, blobs):
        f = files.features(data) if data else None
        if f is not None:
            feats.append({"gid": row.gid, "uid": row.uid, "size": row.size, "date": row.date, "ftype": row.ftype,
                          "layer": row.layer, "music": row.music, **f})
    if not feats:
        return pd.DataFrame()
    fd = pd.DataFrame(feats).sort_values("date")
    ok = fd[fd.decoded == 1].copy()
    known_exact = hashes.groupby("exact").uid.first().to_dict()
    hd, hu = hashes.dh.to_numpy(), hashes.uid.to_numpy()
    dup_other, dup_self, near = [], [], []
    for r in ok.itertuples():
        owner = known_exact.get(r.exact)
        dup_other.append(int(owner is not None and owner != r.uid))
        dup_self.append(int(owner is not None and owner == r.uid))
        n = 0
        if owner is None and r.w >= 64 and len(hd):
            close = files.hamming(hd, int(r.dh)) <= 3
            n = int((close & (hu != r.uid)).any())
        near.append(n)
        known_exact.setdefault(r.exact, r.uid)
    ok["dup_other"], ok["dup_self"], ok["near_other"] = dup_other, dup_self, near
    fd = fd.merge(ok[["gid", "dup_other", "dup_self", "near_other"]], on="gid", how="left")
    write_parquet(pd.concat([hashes, ok[["gid", "uid", "size", "date", "exact", "dh"]]], ignore_index=True), st / "hashes.parquet")
    g = fd.assign(day=pd.to_datetime(fd.date, unit="s").dt.strftime("%Y-%m-%d"), n=1, anim=(fd.frames.fillna(1) > 1).astype(int),
                  frames=fd.frames.fillna(0), colors=fd.colors.fillna(0))
    g[["dup_other", "dup_self", "near_other"]] = g[["dup_other", "dup_self", "near_other"]].fillna(0)
    add_rollup(st / "files_daily.parquet", g.groupby(["day", "size", "fmt"], as_index=False)[
        ["n", "decoded", "anim", "frames", "colors", "bytes", "layer", "music", "dup_other", "dup_self", "near_other"]].sum(), ["day", "size", "fmt"])
    return fd


def top_artists(api: Api, cat: pd.DataFrame, users: pd.DataFrame, t: int) -> pd.DataFrame:
    excluded = set(json.loads((CONFIG / "excluded_artists.json").read_text())["excluded"])
    recent = cat[(cat.date > t - 365 * DAY) & (cat.gone == 0)]
    picks = recent[recent.rec == 1].groupby("uid").size()
    ids = set(picks[picks >= 3].index)
    ids |= set(users[(users.amb == 1) & users.uid.isin(recent.uid)].uid)
    try:
        start = 1
        while start < 3000:
            el = api.post("GetExpertListV4", {"StartNum": start, "EndNum": start + 29, "Language": "en"}, auth=False).get("ExpertList") or []
            ids |= {int(e["UserId"]) for e in el}
            start += len(el)
            if len(el) < 30:
                break
    except ApiDown:
        pass
    ids -= excluded
    top = users[users.uid.isin(ids)].copy()
    top["picks"] = top.uid.map(picks).fillna(0).astype(int)
    return top.sort_values("picks", ascending=False).head(1500)


def profiles(api: Api, st: Path, top: pd.DataFrame, cat: pd.DataFrame, m: pd.DataFrame, t: int) -> pd.DataFrame:
    def one(uid):
        try:
            a = api.user_info(uid)
            b = api.post("LookScore", {"TargetUserId": uid}, auth=False)
        except ApiDown:
            return None
        if a.get("ReturnCode") != 0:
            return None
        return {"uid": uid, "fans": a.get("FansCnt"), "score": a.get("Score"), "level": a.get("Level"), "likecnt": a.get("LikeCnt"),
                "pix": b.get("PixelCnt"), "ani": b.get("AniCnt"), "reccnt": b.get("RecommendCnt"), "head": a.get("HeadId") or ""}
    rows = [r for r in api.pmap(one, top.uid.tolist()) if r]
    if not rows:
        return pd.DataFrame()
    p = pd.DataFrame(rows)
    mine = m[m.uid.isin(p.uid)]
    act = mine.groupby("uid").agg(up_day=("isnew", "sum"), dl_day=("dl", "sum"), dlp_day=("dlp", "sum"), dv_day=("dv", "sum"),
                                  dc_day=("dc", "sum"), rec_day=("to_rec", "sum"), new_day=("to_new", "sum"))
    p = p.merge(act, on="uid", how="left").fillna({c: 0 for c in act.columns})
    p["day"] = day_of(t)
    hist = read_parquet(st / "artist_daily.parquet")
    write_parquet(pd.concat(([hist] if hist is not None else []) + [p.drop(columns=["head"])], ignore_index=True), st / "artist_daily.parquet")

    # avatars: rehosted for top artists only, as lossless WebP
    af = st / "avatars.json"
    known = json.loads(af.read_text()) if af.exists() else {}
    out_dir = DATA / "daily" / "artists"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in p.itertuples() if r.head and (known.get(str(r.uid)) != r.head or not (out_dir / f"{r.uid}.webp").exists())]

    todo = todo[:1500]
    for r, data in zip(todo, api.pmap(lambda r: api.get_file(r.head), todo)):
        webp = files.avatar_webp(data) if data else None
        if webp:
            (out_dir / f"{r.uid}.webp").write_bytes(webp)
        known[str(r.uid)] = r.head
    for f in out_dir.glob("*.webp"):                 # an artist who leaves the set loses the avatar
        if int(f.stem) not in set(top.uid):
            f.unlink()
    af.write_text(json.dumps(known))
    return p


def calendar(api: Api, st: Path, t: int) -> None:
    f = st / "calendar.json"
    cal = json.loads(f.read_text()) if f.exists() else {"seen": {}, "events": []}
    found = []
    try:
        k = api.post("Cloud/GetMatchInfo", auth=False).get("MatchKey")
        if k:
            found.append(("contest", k))
        for th in api.post("Discover/GetTheme").get("ThemeList") or []:
            found.append(("theme", str(th.get("Title") or "")[:80]))
        for n in api.post("Discover/GetTopNew").get("NewList") or []:
            found.append(("news", str(n.get("Title") or "")[:80]))
        a = api.post("GetGalleryAdvert").get("AdvertName")
        if a:
            found.append(("advert", str(a)[:80]))
        al = api.post("Discover/GetAlbumListV3", {"StartNum": 1, "EndNum": 30, "FileSize": 127, "FileSort": 0}, auth=False).get("AlbumList") or []
        for x in al:
            found.append(("album", str(x.get("AlbumName") or "")[:80]))
    except ApiDown:
        pass
    first_run = not cal["seen"]
    for kind, label in found:
        key = f"{kind}|{label}"
        if label and key not in cal["seen"]:
            cal["seen"][key] = t
            if not first_run:                     # what already existed on day one has no known start
                cal["events"].append({"date": day_of(t), "kind": kind, "label": label})
    f.write_text(json.dumps(cal))


def detect_automation(ev: pd.DataFrame) -> None:
    """Flag dense runs of account ids among today's likers that are not yet excluded."""
    if not len(ev):
        return
    per = ev[ev.auto == 0].groupby("liker").size()
    if per.empty:
        return
    bins = pd.Series(per.index // 10000 * 10000, index=per.index)
    for b, ids in bins.groupby(bins):
        if len(ids) >= 150:
            parity = pd.Series(ids.index % 2).value_counts(normalize=True).max()
            counts = per[ids.index]
            if parity > 0.95 and counts.std() / max(counts.mean(), 1e-9) < 0.8:
                note_issue(f"Possible automated accounts near id {b}",
                           f"{len(ids)} accounts in the id range {b} to {b + 9999} gave likes in one day, "
                           f"{parity:.0%} of them with the same id parity and with unusually even activity.\n\n"
                           "Nothing was excluded. If this is a new automated range, add it to "
                           "`config/automated_ranges.json`.")


def run() -> None:
    t = now()
    st = rawrepo.clone_state("state-snap")
    days = [time.strftime("%Y/%m/%d", time.gmtime(t - k * DAY)) for k in range(3)]
    main = rawrepo.clone_main(["state"] + [f"obs/pulse/{d}" for d in days])
    lastf = st / "last_snapshot.json"
    last = json.loads(lastf.read_text()) if lastf.exists() else {}
    if t - last.get("t", 0) < SNAPSHOT_GAP and not env_flag("FORCE_SNAPSHOT"):
        print(f"[snapshot] the last snapshot is {(t - last['t']) // 3600} hours old, under "
              f"{SNAPSHOT_GAP // 3600}; nothing to do")
        return
    # Deep pages answer slowly, so many workers are needed to reach the request rate.
    api = Api(rps=float(os.environ.get("SNAPSHOT_RPS", "8")), workers=24)
    try:
        Pool(main).acquire(api, "snapshot")
    except Halted as exc:
        print(f"[snapshot] halted: {exc}")
        rawrepo.push_main("snapshot: halted")
        flush_issues(Path("_issues"))
        return
    rawrepo.push_main("snapshot: account state")       # token and strikes, before the long crawl

    since = last.get("t", 0)
    partial = bool(os.environ.get("LIMIT_LISTS"))
    lists = list_plan()
    saved = None if partial or env_flag("NO_CHECKPOINT") else load_checkpoint(t, since, len(lists))
    if saved and saved[3] >= len(lists):
        cur, col, t, _ = saved         # the snapshot is dated at the time of the crawl it reuses
        print(f"[snapshot] reusing the crawl of {time.strftime('%Y-%m-%d %H:%M', time.gmtime(t))} UTC, {len(cur)} rows")
    else:
        seed, done = None, 0
        if saved:
            seed, col, t, done = saved
            print(f"[snapshot] resuming the crawl of {time.strftime('%Y-%m-%d %H:%M', time.gmtime(t))} UTC "
                  f"at list {done}/{len(lists)}, {len(seed)} rows so far")
        else:
            col = Collector(t)
        cur = crawl(api, col, lists, seed, done, save=None if partial else save_checkpoint)
        if not partial:
            save_checkpoint(cur, col, len(lists), len(lists))
    prev = read_parquet(st / "catalog.parquet")
    bootstrap = prev is None
    if bootstrap:
        prev = pd.DataFrame(columns=list(cur.columns) + ["t"] + STATE_COLS)

    p = prev[["gid", "like", "watch", "cmt", "new", "rec", "cls", "t"] + STATE_COLS].add_suffix("_p").rename(columns={"gid_p": "gid"})
    m = cur.assign(t=t).merge(p, on="gid", how="left")
    m["isnew"] = m.like_p.isna()
    fresh = m.isnew & (m.date > since) & (not bootstrap)
    for a, b in (("dl", "like"), ("dv", "watch"), ("dc", "cmt")):
        m[a] = np.where(m.isnew, np.where(fresh, m[b], 0), m[b] - m[f"{b}_p"].fillna(0)).astype("int64").clip(min=0)
    m["t_p"] = np.where(m.isnew, m.date, m.t_p.fillna(0)).astype("int64")
    m["to_rec"] = ((m.rec == 1) & ((m.rec_p == 0) | fresh)).astype(int)
    m["to_new"] = ((m.new == 1) & ((m.new_p == 0) | fresh)).astype(int)
    m["refiled"] = (~m.isnew & (m.cls != m.cls_p)).astype(int)

    # who gave the new likes
    ev_old = fetch_events(api, m, t) if not bootstrap else pd.DataFrame(columns=["gid", "liker", "t", "t_prev", "auto"])
    ev = pd.concat([pulse_events(main, since), ev_old], ignore_index=True).drop_duplicates(["gid", "liker", "t"])
    per = ev.groupby("gid").auto.agg(["sum", "count"]) if len(ev) else pd.DataFrame(columns=["sum", "count"])
    m["dla"] = np.minimum(m.gid.map(per["sum"]).fillna(0), m.dl).astype("int64")
    m["dlp"] = np.minimum(m.gid.map(per["count"]).fillna(0) - m.gid.map(per["sum"]).fillna(0), m.dl - m.dla).astype("int64")
    m["dlu"] = (m.dl - m.dla - m.dlp).clip(lower=0)

    # catalog state
    m["first_seen"] = m.first_seen_p.fillna(t).astype("int64")
    m["miss"] = 0
    m["lp"] = (m.lp_p.fillna(0) + m.dlp).astype("int64")
    m["la"] = (m.la_p.fillna(0) + m.dla).astype("int64")
    age = (t - m.date) / DAY
    tracked = (m.first_seen - m.date) < 1.5 * DAY
    cross7 = (age >= 7) & m.lp7_p.isna() & tracked & (age < 9)
    m["lp7"] = np.where(cross7, m.lp, m.lp7_p)
    m["la7"] = np.where(cross7, m.la, m.la7_p)
    m["v7"] = np.where(cross7, m.watch, m.v7_p)
    m["lp21"] = np.where((age >= 21) & m.lp21_p.isna() & pd.notna(m.lp7), m.lp, m.lp21_p)
    m["gone"], m["gone_kind"], m["t_gone"] = 0, None, np.nan
    keep = list(cur.columns) + ["t"] + STATE_COLS
    old = prev[~prev.gid.isin(m.gid)]
    cat = pd.concat([m[keep], old[keep]], ignore_index=True)
    cat["gone"] = cat.gone.fillna(0).astype(int)
    cat["miss"] = cat.miss.fillna(0).astype(int)

    users = pd.DataFrame([(u, *v) for u, v in col.users.items()], columns=["uid", "name", "cc", "level", "amb", "head"])
    pu = read_parquet(st / "users.parquet")
    if pu is not None:
        users = pd.concat([users, pu[~pu.uid.isin(users.uid)]], ignore_index=True)

    ctx = {"t": t, "since": since, "bootstrap": bootstrap, "partial": partial, "warnings": []}

    def step(name, fn, default=None):
        try:
            return fn()
        except ApiDown:
            raise
        except Exception as exc:                 # one failed part must not lose the snapshot
            print(f"[snapshot] {name} failed: {exc!r}")
            ctx["warnings"].append(name)
            if env_flag("STRICT"):
                raise
            return default

    salt = (main / "state" / "salt.txt").read_text().strip()
    cm = step("comments", lambda: fetch_comments(api, m, t, since, salt), pd.DataFrame()) if not bootstrap else pd.DataFrame()
    vanished = step("vanished", lambda: check_vanished(api, cat, set(cur.gid), t), pd.DataFrame()) if not (bootstrap or partial) else pd.DataFrame()
    new = m[m.isnew & (m.date > t - 3 * DAY)]
    fd = step("files", lambda: process_files(api, st, new, col.fids, t), pd.DataFrame())
    top = step("top artists", lambda: top_artists(api, cat, users, t), pd.DataFrame(columns=["uid", "name", "cc", "level", "amb", "head", "picks"]))
    prof = step("profiles", lambda: profiles(api, st, top, cat, m, t), pd.DataFrame())
    step("calendar", lambda: calendar(api, st, t))
    step("detection", lambda: detect_automation(ev))

    # account dimension and uploader-to-uploader like pairs (30 days)
    acc = read_parquet(st / "accounts.parquet")
    prev_last = acc.set_index("uid").last_like if acc is not None else pd.Series(dtype="float64")
    if len(ev):
        g = ev.groupby("liker").agg(first_like=("t", "min"), last_like=("t", "max"), n_likes=("t", "size"), auto=("auto", "max")).reset_index().rename(columns={"liker": "uid"})
        acc = g if acc is None else pd.concat([acc, g]).groupby("uid", as_index=False).agg(
            first_like=("first_like", "min"), last_like=("last_like", "max"), n_likes=("n_likes", "sum"), auto=("auto", "max"))
        owner = cat.set_index("gid").uid
        pr = ev[ev.auto == 0].assign(owner=ev.gid.map(owner)).dropna(subset=["owner"])
        pr = pr[pr.liker.isin(set(cat.uid)) & (pr.liker != pr.owner)][["liker", "owner", "t"]].astype("int64")
        pp = read_parquet(st / "pairs.parquet")
        pr = pd.concat(([pp] if pp is not None else []) + [pr]).groupby(["liker", "owner"], as_index=False).t.max()
        write_parquet(pr[pr.t > t - 30 * DAY], st / "pairs.parquet")
    if acc is not None:
        write_parquet(acc, st / "accounts.parquet")

    # recent tags and mentions (kept 130 days for trends)
    meta = pd.DataFrame(col.meta, columns=["gid", "uid", "date", "tags", "at"]).drop_duplicates("gid")
    if partial:
        pm = read_parquet(st / "meta_recent.parquet")
        if pm is not None:
            meta = pd.concat([meta, pm[~pm.gid.isin(meta.gid)]], ignore_index=True)

    # daily rollup of flows
    m["tier"] = tier(m)
    m["photo"] = (m.cls == PHOTO).astype("int8")
    m["ab"] = pd.cut(age.clip(0, 9e5), AGE, right=False, labels=False).astype("int8")   # a few records carry a future date
    live = m[~m.isnew | fresh].assign(n=1, day=day_of(t), dt=lambda x: t - x.t_p)
    if not bootstrap:
        add_rollup(st / "daily.parquet", live.groupby(["day", "size", "tier", "photo", "ab"], as_index=False)[
            ["n", "dl", "dlp", "dla", "dlu", "dv", "dc", "dt", "to_rec", "to_new", "refiled"]].sum(), ["day", "size", "tier", "photo", "ab"])

    # raw observations
    obs = main / "obs" / "snapshot" / time.strftime("%Y/%m/%d", time.gmtime(t))
    changed = m if bootstrap else m[m.isnew | (m.dl > 0) | (m.dv > 0) | (m.dc > 0) | (m.to_rec > 0) | (m.to_new > 0) | (m.refiled > 0)]
    cols = ["gid", "uid", "date", "cls", "size", "ftype", "like", "watch", "cmt", "likeutc", "new", "rec", "t"]
    for i in range(0, len(changed), 500000):
        write_parquet(changed[cols].iloc[i:i + 500000], obs / f"changes-{i // 500000}.parquet")
    for name, df in (("likes", ev_old), ("comments", cm), ("vanished", vanished), ("files", fd), ("profiles", prof)):
        if df is not None and len(df):
            write_parquet(df.drop(columns=[c for c in ("head",) if c in df.columns]), obs / f"{name}.parquet")
    newmeta = meta[meta.gid.isin(m[m.isnew].gid)] if not bootstrap else meta
    if len(newmeta):
        write_parquet(newmeta, obs / "meta.parquet")

    # build the public aggregates, then persist state
    ctx.update(requests=api.n_requests, prev_last=prev_last, ev=ev, cm=cm, top=top, m=m)
    step("build", lambda: daily.build(st, cat, users, acc, meta, ctx))
    write_parquet(cat, st / "catalog.parquet")
    write_parquet(users, st / "users.parquet")
    write_parquet(meta, st / "meta_recent.parquet")
    lastf.write_text(json.dumps({"t": t, "rows": int(len(cur)), "requests": api.n_requests, "warnings": ctx["warnings"]}))
    rawrepo.push_state("state-snap", f"snapshot {day_of(t)}")
    rawrepo.push_main(f"snapshot {day_of(t)}")
    flush_issues(Path("_issues"))
    print(f"[snapshot] done in {now() - t}s, {api.n_requests} requests, {len(cur)} artworks, warnings {ctx['warnings']}")


if __name__ == "__main__":
    try:
        run()
    except ApiDown as exc:
        print(f"[snapshot] server unreachable: {exc}")
        sys.exit(1)
