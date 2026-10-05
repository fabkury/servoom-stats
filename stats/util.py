"""Shared constants and small helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CONFIG = ROOT / "config"
SCHEMA = 1

SIZES = [1, 2, 4, 16, 32]                 # canvas-size bits shown on the site
ALL_SIZES = [1, 2, 4, 8, 16, 32, 64]      # every bit that is crawled
SIZE_NAME = {1: "16", 2: "32", 4: "64", 8: "Planet", 16: "128", 32: "256", 64: "round"}
CATEGORIES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 19, 30, 40]
CAT_NAME = {1: "Default", 2: "LED text", 3: "Character", 4: "Emoji", 5: "Daily", 6: "Nature",
            7: "Icon", 8: "Pattern", 9: "Creative", 10: "old 10", 11: "old 11", 12: "Photo",
            13: "old 13", 15: "Gadget", 16: "Business", 17: "Season", 19: "Planet",
            30: "Pixel Match", 40: "AI"}
PHOTO = 12
DAY = 86400

# Fields kept from an artwork record (numbers and flags only).
ROW_FIELDS = {
    "gid": "GalleryId", "uid": "UserId", "date": "Date", "cls": "Classify", "size": "FileSize",
    "ftype": "FileType", "like": "LikeCnt", "watch": "WatchCnt", "cmt": "CommentCnt",
    "likeutc": "LikeUTC", "cmtutc": "CommentUTC", "new": "IsAddNew", "rec": "IsAddRecommend",
    "copy": "CopyrightFlag", "ai": "AIFlag", "orig": "OriginalGalleryId", "priv": "PrivateFlag",
}


def now() -> int:
    return int(time.time())


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def to_row(x: Dict, t: int) -> Dict:
    r = {k: int(x.get(v) or 0) for k, v in ROW_FIELDS.items()}
    r["music"] = 1 if x.get("MusicFileId") else 0
    r["layer"] = 1 if x.get("LayerFileId") else 0
    r["t"] = t
    return r


def rows_frame(records: List[Dict], t: int) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=list(ROW_FIELDS) + ["music", "layer", "t"]).astype("int64")
    df = pd.DataFrame([to_row(x, t) for x in records]).astype("int64")
    return df.drop_duplicates("gid", keep="first")


def tier(df: pd.DataFrame) -> np.ndarray:
    return np.where(df["rec"] == 1, "rec", np.where(df["new"] == 1, "new", "none"))


def load_ranges() -> List[List[int]]:
    """Automated account ranges: [lo, hi, parity] with parity 0 even, 1 odd, -1 any."""
    return json.loads((CONFIG / "automated_ranges.json").read_text())["ranges"]


def is_auto(uids, ranges=None) -> np.ndarray:
    u = np.asarray(uids, dtype="int64")
    out = np.zeros(len(u), dtype=bool)
    for lo, hi, par in (ranges if ranges is not None else load_ranges()):
        m = (u >= lo) & (u <= hi)
        if par in (0, 1):
            m &= (u % 2 == par)
        out |= m
    return out


def script_of(s: Optional[str]) -> str:
    """Writing system of a text, as a coarse stand-in for language."""
    for ch in s or "":
        n = unicodedata.name(ch, "")
        if "CJK" in n:
            return "han"
        if "HIRAGANA" in n or "KATAKANA" in n:
            return "kana"
        if "HANGUL" in n:
            return "hangul"
        if "CYRILLIC" in n:
            return "cyrillic"
        if "ARABIC" in n:
            return "arabic"
        if "THAI" in n:
            return "thai"
    return "latin" if re.search("[A-Za-z]", s or "") else "none"


def fingerprint(text: str, salt: str) -> str:
    norm = re.sub(r"\s+", " ", (text or "").strip().lower())
    return hashlib.sha256((salt + norm).encode("utf-8")).hexdigest()[:16]


def read_parquet(path: Path, columns=None) -> Optional[pd.DataFrame]:
    return pd.read_parquet(path, columns=columns) if Path(path).exists() else None


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="zstd", index=False)


def add_rollup(path: Path, new: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    """Add ``new`` into an additive rollup table keyed by ``keys``."""
    old = read_parquet(path)
    df = new if old is None else pd.concat([old, new], ignore_index=True)
    df = df.groupby(keys, as_index=False, observed=True).sum(numeric_only=True)
    write_parquet(df, path)
    return df


def write_json(name: str, obj, sub: str) -> None:
    p = DATA / sub / name
    p.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else round(float(o), 4)
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        raise TypeError(type(o))
    p.write_text(json.dumps(obj, default=default, ensure_ascii=False, separators=(",", ":")),
                 encoding="utf-8")


def recs(df: pd.DataFrame) -> List[Dict]:
    """DataFrame to JSON-ready records with NaN as null and floats rounded."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].round(4)
    return json.loads(out.to_json(orient="records"))


def iso(t: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def day_of(t) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(int(t)))


def small_cells(counts: pd.Series, k: int = 5) -> pd.Series:
    """Merge cells built from fewer than ``k`` accounts into 'other'."""
    big = counts[counts >= k]
    rest = counts[counts < k].sum()
    if rest >= k:
        big = pd.concat([big, pd.Series({"other": rest})])
    return big
