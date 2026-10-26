"""WAV I/O with the standard library only (no libsndfile needed)."""
from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Audio:
    samples: np.ndarray  # shape (n_frames, n_channels), float32 in [-1, 1]
    sr: int
    bit_depth: int
    path: str

    @property
    def channels(self) -> int:
        return self.samples.shape[1]

    @property
    def duration(self) -> float:
        return self.samples.shape[0] / self.sr


def read_wav(path: str | Path) -> Audio:
    """Read 8/16/24/32-bit PCM WAV. Raises wave.Error for unsupported files."""
    with wave.open(str(path), "rb") as w:
        ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)

    if sw == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v & 0x800000, v - 0x1000000, v)
        x = v.astype(np.float32) / 8388608.0
    elif sw == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
    else:  # pragma: no cover
        raise wave.Error(f"unsupported sample width: {sw} bytes")

    return Audio(x.reshape(-1, ch), sr, sw * 8, str(path))


def write_wav(path: str | Path, samples: np.ndarray, sr: int) -> None:
    """Write float samples (n,) or (n, ch) as 16-bit PCM."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    pcm = np.clip(np.round(x * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(x.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
