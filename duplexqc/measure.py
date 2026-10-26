"""Measure a conversation. No pass/fail judgement here — that's spec.py's job."""
from __future__ import annotations

import json
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal

from .audio import Audio, read_wav
from .vad import HOP_S, ChannelVAD, Segment, detect_segments, frame_db, to_mask

BACKCHANNEL_MAX_S = 1.0
BLEED_COMPENSATE_DB = -40.0


@dataclass
class ChannelStats:
    peak_dbfs: float
    clipping_pct: float
    dc_offset: float
    speech_level_dbfs: float | None
    noise_floor_dbfs: float
    snr_db: float | None
    speech_ratio: float
    speech_s: float


@dataclass
class Bleed:
    leakage_db: float     # level of channel `src` found inside channel `dst`
    lag_ms: float
    compensated: bool


@dataclass
class TurnTaking:
    talk_balance: float
    overlap_ratio: float
    longest_mutual_silence_s: float
    turns: int
    turns_per_min: float
    ftos: list[float]                 # floor-transfer offsets (s); negative = overlap
    median_fto_s: float | None
    overlapped_transition_pct: float
    interruptions: int
    backchannels: dict[str, int]


@dataclass
class TranscriptStats:
    n_segments: int
    schema_errors: list[str]
    alignment: float          # share of transcript time that lines up with detected speech
    speech_coverage: float    # share of detected speech that has a transcript
    best_offset_s: float      # shift that would best align transcript to audio
    aligned_at_best: float
    words_per_s: float | None
    empty_segments: int
    segments: dict[str, list[Segment]] = field(default_factory=dict)


@dataclass
class Measurement:
    id: str
    path: str
    error: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    bit_depth: int | None = None
    duration_s: float | None = None
    channel_stats: dict[str, ChannelStats] = field(default_factory=dict)
    bleed: dict[str, Bleed] = field(default_factory=dict)
    vad_segments: dict[str, list[Segment]] = field(default_factory=dict)
    turn_taking: TurnTaking | None = None
    transcript: TranscriptStats | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------- signal

def _channel_stats(x: np.ndarray, vad: ChannelVAD, duration: float) -> ChannelStats:
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    speech_mask = vad.db > vad.threshold_db
    if speech_mask.any():
        level = float(10 * np.log10(np.mean(10 ** (vad.db[speech_mask] / 10)) + 1e-12))
        snr = level - vad.noise_floor_db
    else:
        level, snr = None, None
    speech_s = sum(e - s for s, e in vad.segments)
    return ChannelStats(
        peak_dbfs=round(20 * np.log10(peak + 1e-12), 2),
        clipping_pct=round(100.0 * float(np.mean(np.abs(x) >= 0.999)), 4),
        dc_offset=round(float(np.mean(x)), 5),
        speech_level_dbfs=None if level is None else round(level, 2),
        noise_floor_dbfs=round(vad.noise_floor_db, 2),
        snr_db=None if snr is None else round(snr, 2),
        speech_ratio=round(speech_s / duration, 4) if duration else 0.0,
        speech_s=round(speech_s, 2),
    )


def estimate_leakage(src: np.ndarray, dst: np.ndarray, sr: int,
                     max_lag_s: float = 0.03, window_s: float = 300.0,
                     block_s: float = 1.0) -> tuple[float | None, int]:
    """How loudly speaker `src` shows up on channel `dst`.

    Separate close-talk mics should be near-independent; a gain above roughly
    -30 dB means the far speaker is audible on the near channel (mixer
    routing, speakerphone, open-back headphones...).

    Whole-file correlation is symmetric, so it can't tell A->B bleed from
    B->A. Instead the gain is measured block by block, only in the blocks
    where `src` is loudest: real bleed gives the same gain in every one of
    them, while the mirror direction averages to ~0.
    Returns (gain or None if `src` has too little speech, lag_samples).
    """
    n = min(len(src), len(dst), int(window_s * sr))
    s = np.asarray(src[:n], np.float64) - np.mean(src[:n])
    d = np.asarray(dst[:n], np.float64) - np.mean(dst[:n])

    # global lag from full cross-correlation
    c = signal.correlate(d, s, mode="full", method="fft")
    lags = signal.correlation_lags(len(d), len(s), mode="full")
    keep = np.abs(lags) <= int(max_lag_s * sr)
    lag = int(lags[keep][int(np.argmax(np.abs(c[keep])))])
    s = _shift(s, lag)

    B = int(block_s * sr)
    nb = n // B
    if nb < 4:
        return None, lag
    sb, db_ = s[: nb * B].reshape(nb, B), d[: nb * B].reshape(nb, B)
    e_src = np.sum(sb ** 2, axis=1)
    e_db = 10 * np.log10(e_src / B + 1e-12)
    active = (e_db >= np.percentile(e_db, 75)) & (e_db >= np.percentile(e_db, 10) + 15)
    if active.sum() < 3:
        return None, lag
    g = np.sum(sb[active] * db_[active], axis=1) / e_src[active]
    return float(np.median(g)), lag


def _shift(x: np.ndarray, lag: int) -> np.ndarray:
    if lag == 0:
        return x
    out = np.zeros_like(x)
    if lag > 0:
        out[lag:] = x[:-lag]
    else:
        out[:lag] = x[-lag:]
    return out


# --------------------------------------------------------------- turn-taking

def _contained(seg: Segment, others: list[Segment], tol: float = 0.3) -> bool:
    return any(o[0] - tol <= seg[0] and seg[1] <= o[1] + tol for o in others)


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def turn_taking(segs: dict[str, list[Segment]], duration: float) -> TurnTaking:
    a, b = segs["A"], segs["B"]
    n = int(np.ceil(duration / HOP_S)) + 1
    ma, mb = to_mask(a, n), to_mask(b, n)
    union, both = ma | mb, ma & mb
    sa, sb = float(ma.sum() * HOP_S), float(mb.sum() * HOP_S)

    # mutual silence, only between first and last speech
    idx = np.flatnonzero(union)
    longest_sil = 0.0
    if len(idx):
        longest_sil = float(_longest_run(~union[idx[0]: idx[-1] + 1]) * HOP_S)

    # backchannels: short segments that sit inside the other speaker's segment
    bc = {"A": 0, "B": 0}
    main: list[tuple[float, float, str]] = []
    for spk, mine, other in (("A", a, b), ("B", b, a)):
        for sg in mine:
            if sg[1] - sg[0] < BACKCHANNEL_MAX_S and _contained(sg, other):
                bc[spk] += 1
            else:
                main.append((sg[0], sg[1], spk))
    main.sort()

    # floor transfers
    ftos: list[float] = []
    turns = 0
    holder, floor_end = None, 0.0
    for s, e, spk in main:
        if holder is None:
            holder, floor_end, turns = spk, e, 1
        elif spk == holder:
            floor_end = max(floor_end, e)
        elif e <= floor_end:
            continue  # embedded interjection, floor doesn't change
        else:
            ftos.append(round(s - floor_end, 3))
            holder, floor_end = spk, e
            turns += 1

    f = np.array(ftos) if ftos else np.array([])
    return TurnTaking(
        talk_balance=round(min(sa, sb) / max(sa, sb), 3) if max(sa, sb) > 0 else 0.0,
        overlap_ratio=round(float(both.sum() / union.sum()), 4) if union.any() else 0.0,
        longest_mutual_silence_s=round(longest_sil, 2),
        turns=turns,
        turns_per_min=round(turns / (duration / 60), 2) if duration else 0.0,
        ftos=ftos,
        median_fto_s=round(float(np.median(f)), 3) if len(f) else None,
        overlapped_transition_pct=round(100 * float(np.mean(f < 0)), 1) if len(f) else 0.0,
        interruptions=int(np.sum(f < -0.5)) if len(f) else 0,
        backchannels=bc,
    )


# ---------------------------------------------------------------- transcript

def load_transcript(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def transcript_stats(tr: dict, vad_segs: dict[str, list[Segment]], duration: float) -> TranscriptStats:
    errors: list[str] = []
    by_spk: dict[str, list[Segment]] = {"A": [], "B": []}
    words, word_time, empty = 0, 0.0, 0
    for i, sg in enumerate(tr.get("segments", [])):
        try:
            spk, s, e, text = sg["speaker"], float(sg["start"]), float(sg["end"]), str(sg.get("text", ""))
        except (KeyError, TypeError, ValueError):
            errors.append(f"segment {i}: missing speaker/start/end")
            continue
        if spk not in by_spk:
            errors.append(f"segment {i}: unknown speaker {spk!r}")
            continue
        if e <= s:
            errors.append(f"segment {i}: end <= start")
            continue
        if s < -0.05 or e > duration + 0.05:
            errors.append(f"segment {i}: outside audio bounds")
        if not text.strip():
            empty += 1
        by_spk[spk].append((s, e))
        words += len(text.split())
        word_time += e - s

    n = int(np.ceil(duration / HOP_S)) + 1
    v = {k: to_mask(vad_segs[k], n) for k in ("A", "B")}

    def score(offset: float) -> tuple[float, float]:
        hit = tr_total = sp_total = 0
        for k in ("A", "B"):
            t = to_mask(by_spk[k], n, offset)
            hit += int((t & v[k]).sum())
            tr_total += int(t.sum())
            sp_total += int(v[k].sum())
        return (hit / tr_total if tr_total else 0.0, hit / sp_total if sp_total else 0.0)

    align, cover = score(0.0)
    best_off, best_align = 0.0, align
    for off in np.arange(-3.0, 3.0001, 0.05):
        al, _ = score(float(off))
        if al > best_align + 1e-9:
            best_off, best_align = float(off), al

    return TranscriptStats(
        n_segments=sum(len(x) for x in by_spk.values()),
        schema_errors=errors,
        alignment=round(align, 3),
        speech_coverage=round(cover, 3),
        best_offset_s=round(best_off, 2),
        aligned_at_best=round(best_align, 3),
        words_per_s=round(words / word_time, 2) if word_time else None,
        empty_segments=empty,
        segments=by_spk,
    )


# ----------------------------------------------------------------- top level

def measure(path: str | Path) -> Measurement:
    path = Path(path)
    m = Measurement(id=path.stem, path=str(path))
    try:
        audio: Audio = read_wav(path)
    except (wave.Error, EOFError, ValueError) as exc:
        m.error = f"unreadable WAV: {exc}"
        return m

    m.sample_rate, m.channels, m.bit_depth = audio.sr, audio.channels, audio.bit_depth
    m.duration_s = round(audio.duration, 3)
    names = ["A", "B"][: audio.channels] if audio.channels <= 2 else [f"ch{i}" for i in range(audio.channels)]
    x = audio.samples

    vads: dict[str, ChannelVAD] = {}
    if audio.channels == 2:
        # bleed in both directions, measured on the raw channels
        cleaned = {}
        for src, dst, si, di in (("A", "B", 0, 1), ("B", "A", 1, 0)):
            g, lag = estimate_leakage(x[:, si], x[:, di], audio.sr)
            if g is None:
                continue
            db = round(float(20 * np.log10(abs(g) + 1e-9)), 1)
            comp = db > BLEED_COMPENSATE_DB
            m.bleed[f"{src}->{dst}"] = Bleed(db, round(1000 * lag / audio.sr, 2), comp)
            if comp:
                cleaned[dst] = x[:, di] - g * _shift(x[:, si].astype(np.float64), lag)
        for name, i in (("A", 0), ("B", 1)):
            ch = cleaned.get(name, x[:, i])
            vads[name] = detect_segments(frame_db(ch, audio.sr))
    else:
        for i, name in enumerate(names):
            vads[name] = detect_segments(frame_db(x[:, i], audio.sr))

    for i, name in enumerate(names):
        m.channel_stats[name] = _channel_stats(x[:, i], vads[name], audio.duration)
        m.vad_segments[name] = vads[name].segments

    if audio.channels == 2:
        m.turn_taking = turn_taking(m.vad_segments, audio.duration)
        tr = load_transcript(path.with_suffix(".json"))
        if tr is not None:
            m.transcript = transcript_stats(tr, m.vad_segments, audio.duration)
    return m
