"""Energy-based voice activity detection, one channel at a time.

Deliberately simple and dependency-free: an adaptive threshold above each
channel's own noise floor, then hangover smoothing. Good enough to measure
turn-taking on clean per-speaker channels; swap in Silero/WebRTC VAD via
`detect_segments` if you need more.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FRAME_S = 0.020
HOP_S = 0.010

Segment = tuple[float, float]


@dataclass
class ChannelVAD:
    db: np.ndarray            # per-frame RMS level, dBFS
    segments: list[Segment]   # speech segments in seconds
    noise_floor_db: float
    threshold_db: float


def frame_db(x: np.ndarray, sr: int) -> np.ndarray:
    """Per-frame RMS in dBFS using a cumulative sum (O(n), long files are fine)."""
    f, h = int(FRAME_S * sr), int(HOP_S * sr)
    if len(x) < f:
        return np.array([-120.0])
    cs = np.concatenate([[0.0], np.cumsum(np.asarray(x, np.float64) ** 2)])
    starts = np.arange(0, len(x) - f + 1, h)
    energy = (cs[starts + f] - cs[starts]) / f
    return 10.0 * np.log10(energy + 1e-12)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of True runs."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def detect_segments(
    db: np.ndarray,
    margin_db: float = 12.0,
    abs_min_db: float = -55.0,
    min_speech_s: float = 0.12,
    min_gap_s: float = 0.25,
) -> ChannelVAD:
    floor = float(np.percentile(db, 10))
    thr = max(floor + margin_db, abs_min_db)
    segs: list[list[float]] = []
    for i, j in _runs(db > thr):
        s, e = i * HOP_S, (j - 1) * HOP_S + FRAME_S
        if segs and s - segs[-1][1] < min_gap_s:
            segs[-1][1] = e
        else:
            segs.append([s, e])
    out = [(round(s, 3), round(e, 3)) for s, e in segs if e - s >= min_speech_s]
    return ChannelVAD(db=db, segments=out, noise_floor_db=floor, threshold_db=thr)


def to_mask(segments: list[Segment], n_frames: int, offset_s: float = 0.0) -> np.ndarray:
    """Rasterise segments onto the 10 ms frame grid."""
    m = np.zeros(n_frames, dtype=bool)
    for s, e in segments:
        i = max(0, int(round((s + offset_s) / HOP_S)))
        j = min(n_frames, int(round((e + offset_s) / HOP_S)))
        if j > i:
            m[i:j] = True
    return m
