"""Collection health: for every hour since the pipeline began, whether each collector
ran and how complete its reading was, published as spans the site shades on its charts
and draws on the overview page.

Everything here is derived from what the collectors themselves recorded (the hourly
rollup, the daily rollup, the scalar series and, since 2026-10-15, one row per run in
``runs.parquet`` on each state branch). Nothing is typed in by hand, so a hole in the
data is marked on the site whether or not anyone noticed it.

Levels: ``ok``; ``degraded`` (a reading exists but is incomplete, or averages a longer
period than usual); ``missing`` (no reading). ``why`` names the evidence:

* ``gap``: no reading in the period (the next reading, if any, averages over it);
* ``spread``: one reading covers several hours or days, so its figures are averages;
* ``partial``: a listing returned far fewer artworks than usual;
* ``no_token``: collected without an account, so likes are not attributed to people;
* ``budget``: the hour's reading time ran out before some phases (named in ``parts``);
* ``reach``: only part of the catalog was readable (``reach`` holds the share);
* ``skipped``: parts of the snapshot failed (``parts`` names them).

Output: ``data/health/pulse.json`` (lanes ``pulse`` and ``refresh``, written by the
4-hour refresh) and ``data/health/snapshot.json`` (lanes ``snapshot`` and ``catalog``,
written by the snapshot). A span is ``{a, b, level, why, ...}`` with ``a``/``b`` in
unix seconds; the spans of a lane are contiguous and ordered.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .util import CONFIG, DAY, SCHEMA, day_of, iso, read_parquet, write_json, write_parquet

SUB = "health"
HOUR = 3600
SPREAD_HOURS = 1.6         # observations spaced wider than this: the reading averages several hours
PARTIAL_SHARE = 0.6        # fewer artworks than this share of the usual count: an incomplete listing
REFRESH_EVERY = 4 * HOUR   # the refresh cadence (REFRESH_EVERY in pulse.py)
REFRESH_TOL = 6 * HOUR     # a longer wait between refreshes leaves a hole in the 4-hourly series
SNAPSHOT_TOL = 3 * DAY     # snapshots come every two days; longer, and one reading is spread too thin
REACH = 0.9                # share of the catalog a snapshot must reach to count as whole


def log_run(st: Path, row: Dict) -> None:
    """Append one run record to the job's state branch (``runs.parquet``)."""
    f = st / "runs.parquet"
    old = read_parquet(f)
    new = pd.DataFrame([row])
    write_parquet(new if old is None else pd.concat([old, new], ignore_index=True), f)


def merge(spans: List[Dict]) -> List[Dict]:
    """Join consecutive spans of equal level and reason; a numeric extra keeps its maximum."""
    out: List[Dict] = []
    for s in spans:
        last = out[-1] if out else None
        if last and last["b"] == s["a"] and last["level"] == s["level"] and last["why"] == s["why"] \
                and last.get("parts") == s.get("parts"):
            last["b"] = s["b"]
            for k in ("hours", "days"):
                if k in s:
                    last[k] = max(last.get(k, 0), s[k])
            for k in ("share", "reach"):
                if k in s:
                    last[k] = min(last.get(k, 1), s[k])
        else:
            out.append(dict(s))
    return out


def timeline(times: List[int], unit: int, tol: int) -> List[Dict]:
    """Spans from the times a collector ran: ``ok`` between runs closer than ``tol``;
    otherwise ``missing`` until one ``unit`` before the next run, which is ``spread``."""
    ts = sorted({int(x) for x in times})
    spans = []
    for a, b in zip(ts, ts[1:]):
        if b - a <= tol:
            spans.append({"a": a, "b": b, "level": "ok", "why": ""})
        else:
            spans.append({"a": a, "b": b - unit, "level": "missing", "why": "gap"})
            spans.append({"a": b - unit, "b": b, "level": "degraded", "why": "spread", "hours": round((b - a) / HOUR, 1)})
    return merge(spans)


# ---- hourly job -------------------------------------------------------------
def pulse_lane(st: Path) -> List[Dict]:
    """One span per hour slot from the first reading to the last, from the hourly rollup."""
    h = read_parquet(st / "hourly.parquet", columns=["hour", "n", "dt"])
    if h is None or not len(h):
        return []
    g = h.groupby("hour")[["n", "dt"]].sum().sort_index()
    g["spacing"] = g.dt / g.n / HOUR
    # the usual size of a reading: the median of the previous week of readings
    usual = g.n.rolling(168, min_periods=1).median().shift(1).fillna(g.n.median())
    runs = read_parquet(st / "runs.parquet")
    no_token = set(runs[runs["mode"] == "degraded"].hour.astype(int)) if runs is not None and "mode" in runs else set()
    cut: Dict[int, str] = {}
    if runs is not None and "skipped" in runs:
        for r in runs[runs.skipped.fillna("") != ""].itertuples():
            cut[int(r.hour)] = str(r.skipped)
    spans = []
    for hr in range(int(g.index.min()), int(g.index.max()) + HOUR, HOUR):
        s = {"a": hr, "b": hr + HOUR}
        if hr not in g.index:
            s.update(level="missing", why="gap")
        else:
            r = g.loc[hr]
            if r.n < PARTIAL_SHARE * usual.loc[hr]:
                s.update(level="degraded", why="partial", share=round(float(r.n / usual.loc[hr]), 2))
            elif hr in no_token:
                s.update(level="degraded", why="no_token")
            elif hr in cut:
                s.update(level="degraded", why="budget", parts=cut[hr])
            elif r.spacing > SPREAD_HOURS:
                s.update(level="degraded", why="spread", hours=round(float(r.spacing), 1))
            else:
                s.update(level="ok", why="")
        spans.append(s)
    return merge(spans)


def refresh_lane(st: Path) -> List[Dict]:
    """Spans from the times the 4-hour refresh ran: the run log, plus the seed of refreshes
    before the log existed (taken from the data commits of this repository)."""
    times: List[int] = []
    seed = CONFIG / "health_seed.json"
    if seed.exists():
        times += json.loads(seed.read_text())["refresh"]
    runs = read_parquet(st / "runs.parquet")
    if runs is not None and "refresh" in runs:
        times += runs[runs.refresh == 1].t.astype(int).tolist()
    return timeline(times, REFRESH_EVERY, REFRESH_TOL)


def build_pulse(st: Path, t: int) -> None:
    lanes = {"pulse": pulse_lane(st), "refresh": refresh_lane(st)}
    write_json("pulse.json", {"schema": SCHEMA, "generated": iso(t), "lanes": lanes}, SUB)
    print(f"[health] pulse lanes: {', '.join(f'{k} {len(v)} spans' for k, v in lanes.items())}")


# ---- snapshot job -----------------------------------------------------------
def readings(st: Path) -> pd.DataFrame:
    """One row per snapshot: ``t``, ``since``, ``days`` covered, ``reach`` (share of the
    catalog observed), ``partial``, ``warnings``. Exact times come from the run log;
    older readings are placed from the chain of ``span_days`` in the scalar series."""
    series = read_parquet(st / "series.parquet")
    if series is None or not len(series):
        return pd.DataFrame(columns=["t", "since", "days", "reach", "partial", "warnings"])
    sp = series[series.metric == "span_days"].set_index("day").value
    cat = series[series.metric == "catalog_n"].set_index("day").value
    daily = read_parquet(st / "daily.parquet", columns=["day", "n"])
    seen = daily.groupby("day").n.sum() if daily is not None else pd.Series(dtype="float64")
    runs = read_parquet(st / "runs.parquet")
    exact: Dict[str, Dict] = {}
    if runs is not None and len(runs):
        for r in runs[runs.partial == 0].itertuples():
            exact[day_of(r.t)] = {"t": int(r.t), "reach": float(r.rows) / max(float(r.catalog), 1),
                                  "warnings": str(r.warnings or "")}
    days = sorted(cat.index)
    rows: List[Dict] = []
    t_next: Optional[float] = None
    for day in reversed(days):
        if day in exact:
            t = exact[day]["t"]
        elif t_next is not None and rows[-1]["days"] == rows[-1]["days"]:
            t = t_next - rows[-1]["days"] * DAY
        else:
            last = st / "last_snapshot.json"
            t = json.loads(last.read_text())["t"] if last.exists() else None
        if t is None:
            break
        span = float(sp.get(day, float("nan")))
        reach = exact[day]["reach"] if day in exact else (float(seen.get(day, 0)) / float(cat[day]) if day in seen.index else 1.0)
        rows.append({"day": day, "t": int(t), "days": span, "reach": round(min(reach, 1.0), 4),
                     "partial": 0, "warnings": exact.get(day, {}).get("warnings", "")})
        t_next = t
    df = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)
    df["since"] = df.t.shift(1)
    return df


def snapshot_lanes(st: Path) -> Dict[str, List[Dict]]:
    rd = readings(st)
    snap, catalog = [], []
    for r in rd.itertuples():
        if r.since != r.since:                                   # the first reading: nothing before it
            continue
        a, b = int(r.since), int(r.t)
        s = {"a": a, "b": b, "level": "ok", "why": ""}
        if r.warnings:
            s.update(level="degraded", why="skipped", parts=r.warnings)
        elif r.days == r.days and r.days * DAY > SNAPSHOT_TOL:
            s.update(level="degraded", why="spread", days=round(float(r.days), 1))
        snap.append(s)
        c = dict(s)
        if r.reach < REACH:
            c = {"a": a, "b": b, "level": "degraded", "why": "reach", "reach": float(r.reach)}
        catalog.append(c)
    return {"snapshot": merge(snap), "catalog": merge(catalog)}


def build_snapshot(st: Path, t: int) -> None:
    lanes = snapshot_lanes(st)
    rd = readings(st)
    out = {"schema": SCHEMA, "generated": iso(t), "lanes": lanes,
           "readings": [{"t": int(r.t), "reach": float(r.reach), "days": None if r.days != r.days else round(float(r.days), 2)}
                        for r in rd.itertuples()]}
    write_json("snapshot.json", out, SUB)
    print(f"[health] snapshot lanes: {', '.join(f'{k} {len(v)} spans' for k, v in lanes.items())}")
