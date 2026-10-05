"""Artwork files: decode once, keep derived features and hashes, discard the pixels."""

from __future__ import annotations

import hashlib
import io
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

try:                                    # the decoders live in the servoom package
    from servoom import PixelBeanDecoder
except Exception as exc:                # pragma: no cover - reported once by the caller
    print(f"[files] cannot import the servoom decoders: {exc!r}")
    PixelBeanDecoder = None


def decode(data: bytes) -> Optional[List[np.ndarray]]:
    if PixelBeanDecoder is None or not data:
        return None
    try:
        bean = PixelBeanDecoder.decode_stream(io.BytesIO(data))
        frames = bean.frames_data if bean is not None else None
        return [np.asarray(f, dtype=np.uint8) for f in frames] if frames else None
    except Exception:
        return None


def dhash(frame: np.ndarray) -> int:
    """64-bit difference hash: 9x8 grey thumbnail, compare horizontal neighbours."""
    g = np.asarray(Image.fromarray(frame).convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v - (1 << 64) if v >= (1 << 63) else v          # fit a signed 64-bit column


def features(data: bytes) -> Optional[Dict]:
    """Features of one artwork file, or None when it cannot be decoded."""
    out: Dict = {"fmt": data[0] if data else -1, "bytes": len(data)}
    frames = decode(data)
    if not frames:
        return {**out, "decoded": 0}
    f0 = frames[0]
    sample = frames[:: max(1, len(frames) // 8)][:8]
    colors = len(np.unique(np.concatenate([f.reshape(-1, 3) for f in sample]), axis=0))
    h = hashlib.sha1()
    for f in frames[:120]:
        h.update(f.tobytes())
    out.update(decoded=1, w=int(f0.shape[1]), h=int(f0.shape[0]), frames=len(frames), colors=int(colors),
               black=float((f0.sum(axis=2) == 0).mean()), exact=h.hexdigest()[:16],
               dh=dhash(f0), dh_mid=dhash(frames[len(frames) // 2]))
    return out


def hamming(a: np.ndarray, b: int) -> np.ndarray:
    x = np.bitwise_xor(a.astype(np.uint64), np.uint64(b & 0xFFFFFFFFFFFFFFFF))
    return np.array([bin(int(v)).count("1") for v in x]) if len(x) < 64 else _popcount(x)


def _popcount(x: np.ndarray) -> np.ndarray:
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def avatar_webp(data: bytes) -> Optional[bytes]:
    """An avatar file as lossless WebP (first frame)."""
    frames = decode(data)
    if not frames:
        return None
    buf = io.BytesIO()
    Image.fromarray(frames[0]).save(buf, "WEBP", lossless=True, quality=100, method=6)
    return buf.getvalue()
