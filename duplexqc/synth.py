"""Generate a demo delivery: synthetic two-speaker conversations, some with
realistic defects injected. No real voices, no licensing questions, no
downloads — `duplex-qc synth demo/` makes the whole thing reproducibly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import signal

from .audio import write_wav

VOWELS = [(730, 1090), (270, 2290), (530, 1840), (570, 840), (300, 870), (660, 1720), (490, 1350)]
WORDS = ("so the data we need is mostly about how people actually talk when they "
         "are not reading a script and that means overlap pauses and little reactions").split()
BACKCHANNELS = ["mm-hmm", "yeah", "right", "uh-huh", "okay", "oh nice"]


def _speech(dur: float, sr: int, f0: float, rng: np.random.Generator) -> np.ndarray:
    """Speech-like signal: harmonic source + moving formants + syllable envelope."""
    n = int(dur * sr)
    if n <= 0:
        return np.zeros(0)
    t = np.arange(n) / sr
    f0c = f0 * (1 + 0.08 * np.sin(2 * np.pi * rng.uniform(0.2, 0.7) * t + rng.uniform(0, 6.28)))
    phase = 2 * np.pi * np.cumsum(f0c) / sr
    src = np.zeros(n)
    for k in range(1, 30):
        if k * f0 * 1.1 > sr / 2 * 0.9:
            break
        src += np.sin(k * phase) / k
    out = np.zeros(n)
    pos = 0
    while pos < n:
        L = min(n - pos, int(rng.uniform(0.16, 0.30) * sr))
        chunk = src[pos: pos + L]
        f1, f2 = VOWELS[rng.integers(len(VOWELS))]
        y = np.zeros(L)
        for fc, gain in ((f1, 1.0), (f2, 0.6)):
            lo, hi = fc * 0.8 / (sr / 2), min(fc * 1.2 / (sr / 2), 0.99)
            sos = signal.butter(2, [lo, hi], btype="band", output="sos")
            y += gain * signal.sosfilt(sos, chunk)
        env = 0.2 + 0.8 * np.sin(np.linspace(0, np.pi, L)) ** 2
        burst = int(0.025 * sr)
        if L > 2 * burst:  # consonant-ish noise onset
            y[:burst] += 0.3 * rng.standard_normal(burst) * np.hanning(burst)
        out[pos: pos + L] = y * env
        pos += L
    edge = min(int(0.02 * sr), n // 2)
    if edge:
        out[:edge] *= np.linspace(0, 1, edge)
        out[-edge:] *= np.linspace(1, 0, edge)
    return out / (np.sqrt(np.mean(out ** 2)) + 1e-12)


def _schedule(total: float, rng: np.random.Generator, style: str) -> list[tuple[int, float, float, str]]:
    """Turn plan: (speaker, start, end, kind). style: natural | stitched | monologue."""
    ev, t, spk = [], rng.uniform(0.4, 1.2), 0
    while t < total - 1.0:
        if style == "monologue":
            dur = rng.uniform(7, 12) if spk == 0 else rng.uniform(0.6, 1.1)
        else:
            dur = float(np.clip(rng.lognormal(np.log(3.0), 0.5), 0.8, 9.0))
        end = min(t + dur, total - 0.4)
        ev.append((spk, t, end, "turn"))
        if style == "natural" and end - t > 2.5 and rng.random() < 0.55:
            s = rng.uniform(t + 0.8, end - 1.0)
            ev.append((1 - spk, s, s + rng.uniform(0.3, 0.6), "bc"))
        fto = rng.uniform(0.35, 0.8) if style == "stitched" else float(np.clip(rng.normal(0.2, 0.45), -1.2, 1.8))
        t, spk = end + fto, 1 - spk
    return ev


def make_conversation(seed: int, sr: int = 16000, total: float = 60.0, style: str = "natural",
                      noise_dbfs: float = -68.0) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    f0s = (rng.uniform(100, 140), rng.uniform(180, 230))
    level = 10 ** (rng.uniform(-24, -20) / 20)
    x = np.zeros((int(total * sr), 2))
    segs = []
    for spk, s, e, kind in _schedule(total, rng, style):
        i = int(s * sr)
        y = _speech(e - s, sr, f0s[spk], rng) * level * (0.7 if kind == "bc" else 1.0)
        x[i: i + len(y), spk] += y[: len(x) - i]
        if kind == "bc":
            text = str(rng.choice(BACKCHANNELS))
        else:
            n_words = max(1, int(round((e - s) * rng.uniform(2.3, 3.1))))
            text = " ".join(rng.choice(WORDS, n_words))
        segs.append({"speaker": "AB"[spk], "start": round(s, 3), "end": round(e, 3), "text": text})
    x += 10 ** (noise_dbfs / 20) * rng.standard_normal(x.shape)
    segs.sort(key=lambda d: d["start"])
    return x, {"channels": {"A": 0, "B": 1}, "segments": segs}


FIXTURES = {
    "conv_001": dict(kind="clean"),
    "conv_002": dict(kind="clean"),
    "conv_003": dict(kind="clean"),
    "conv_004": dict(kind="clean"),
    "conv_005": dict(kind="clean"),
    "conv_006": dict(kind="clipped"),
    "conv_007": dict(kind="bleed"),
    "conv_008": dict(kind="dead_channel"),
    "conv_009": dict(kind="stitched"),
    "conv_010": dict(kind="monologue"),
    "conv_011": dict(kind="transcript_drift"),
    "conv_012": dict(kind="long_silence"),
    "conv_013": dict(kind="noisy"),
    "conv_014": dict(kind="wrong_sample_rate"),
    "conv_015": dict(kind="mono_mixdown"),
}


def build(kind: str, seed: int, total: float = 60.0) -> tuple[np.ndarray, int, dict | None]:
    sr = 8000 if kind == "wrong_sample_rate" else 16000
    style = {"stitched": "stitched", "monologue": "monologue"}.get(kind, "natural")
    noise = -30.0 if kind == "noisy" else -68.0
    x, tr = make_conversation(seed, sr, total, style, noise)
    if kind == "clipped":
        x[:, 0] = np.clip(x[:, 0] * 7.0, -1, 1)
    elif kind == "bleed":
        d = int(0.004 * sr)
        x[d:, 1] += 0.12 * x[:-d, 0]          # A audible on B's mic at about -18 dB
    elif kind == "dead_channel":
        x[:, 1] = 10 ** (-68 / 20) * np.random.default_rng(seed).standard_normal(len(x))
        tr["segments"] = [sg for sg in tr["segments"] if sg["speaker"] == "A"]
    elif kind == "transcript_drift":
        for sg in tr["segments"]:
            sg["start"], sg["end"] = round(sg["start"] + 1.2, 3), round(sg["end"] + 1.2, 3)
    elif kind == "long_silence":
        cut, gap = len(x) // 2, int(14 * sr)
        x = np.concatenate([x[:cut], 10 ** (-68 / 20) * np.random.default_rng(seed).standard_normal((gap, 2)), x[cut:]])
        mid = cut / sr
        for sg in tr["segments"]:
            if sg["start"] >= mid:
                sg["start"], sg["end"] = round(sg["start"] + 14, 3), round(sg["end"] + 14, 3)
    elif kind == "mono_mixdown":
        x = x.mean(axis=1, keepdims=True)
    return x, sr, tr


def write_demo(out: str | Path, total: float = 60.0) -> list[Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (cid, cfg) in enumerate(FIXTURES.items()):
        x, sr, tr = build(cfg["kind"], seed=1000 + i, total=total)
        p = out / f"{cid}.wav"
        write_wav(p, x, sr)
        if tr is not None:
            p.with_suffix(".json").write_text(json.dumps(tr, indent=1))
        paths.append(p)
    (out / "_fixture_key.json").write_text(json.dumps({k: v["kind"] for k, v in FIXTURES.items()}, indent=1))
    return paths
