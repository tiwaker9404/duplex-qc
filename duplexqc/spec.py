"""Turn a customer's data spec into pass / warn / fail findings."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from .measure import Measurement

DEFAULT_SPEC: dict = {
    "name": "default",
    "description": "",
    "format": {
        "sample_rates": [16000, 24000, 44100, 48000],
        "channels": 2,
        "min_bit_depth": 16,
        "duration_s": {"min": 30, "max": 7200},
    },
    "signal": {
        "max_clipping_pct": 0.1,
        "max_dc_offset": 0.01,
        "speech_level_dbfs": {"min": -35, "max": -8},
        "min_snr_db": 20,
        "max_bleed_db": -30,
    },
    "turn_taking": {
        "min_speech_ratio": 0.15,
        "min_talk_balance": 0.35,
        "overlap_ratio": {"min": 0.03, "max": 0.35},
        "max_mutual_silence_s": 8,
        "min_turns_per_min": 4,
    },
    "transcript": {
        "required": False,
        "min_alignment": 0.80,
        "max_offset_s": 0.25,
        "min_speech_coverage": 0.85,
        "words_per_s": {"min": 1.0, "max": 5.5},
    },
    "severity": {},  # check id -> "warn" to downgrade a failure
}

# What to tell ops / the vendor when a check fails.
FIXES: dict[str, str] = {
    "unreadable": "Re-export as PCM WAV; the file header is corrupt or the codec is unsupported.",
    "sample_rate": "Re-export at the contracted sample rate. Do not upsample low-rate source — re-record.",
    "channels": "Deliver one speaker per channel (stereo). Mono mixes can't train full-duplex models.",
    "bit_depth": "Export at 16-bit PCM or higher.",
    "duration": "Trim or merge sessions to the contracted length range.",
    "clipping": "Lower input gain on that speaker's mic and re-record; clipped audio can't be repaired.",
    "dc_offset": "Apply a high-pass filter (~20 Hz) in post; check the interface for a grounding fault.",
    "speech_level": "Normalize per channel to the target speech level.",
    "snr": "Background noise is too high: move to a quieter room or use close-talk headsets.",
    "bleed": "The other speaker is audible on this channel: check routing, use closed headphones.",
    "dead_channel": "Dead or muted mic on this channel: check the recording chain and re-record.",
    "speech_ratio": "Channel is near-silent: dead mic, muted input, or the speaker barely talks.",
    "talk_balance": "One speaker dominates: brief participants to converse, not interview.",
    "overlap": "Too little overlap usually means stitched or turn-by-turn recordings, which "
               "don't teach backchannels or interruptions.",
    "mutual_silence": "Long dead air: trim it or flag the session for re-recording.",
    "turn_rate": "Too few speaker changes for conversational data.",
    "transcript_schema": "Fix malformed transcript segments (speaker/start/end).",
    "transcript_offset": "Timestamps are shifted by a constant amount: apply the global offset "
                         "shown and re-verify, no need to re-transcribe.",
    "transcript_alignment": "Transcript timing doesn't match the audio: re-run forced alignment.",
    "transcript_coverage": "Speech present in the audio has no transcript: send back for completion.",
    "speaking_rate": "Words per second is implausible: check for truncated or padded text.",
    "transcript_missing": "Transcript sidecar is required by this spec but missing.",
}


DEAD_RATIO = 0.02  # below this share of speech, a channel is treated as dead


@dataclass
class Finding:
    check: str
    category: str
    status: str            # pass | warn | fail | skip
    value: object
    expected: str
    message: str
    fix: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_spec(path: str | Path | None) -> dict:
    if path is None:
        return copy.deepcopy(DEFAULT_SPEC)
    return _merge(DEFAULT_SPEC, yaml.safe_load(Path(path).read_text()) or {})


class _Eval:
    def __init__(self, spec: dict):
        self.spec, self.findings = spec, []

    def add(self, check, category, ok, value, expected, message, skip=False):
        if skip:
            status = "skip"
        elif ok:
            status = "pass"
        else:
            status = self.spec["severity"].get(check, "fail")
        self.findings.append(Finding(check, category, status, value, expected, message,
                                     "" if status in ("pass", "skip") else FIXES.get(check, "")))


def evaluate(m: Measurement, spec: dict) -> list[Finding]:
    ev = _Eval(spec)
    if m.error:
        ev.add("unreadable", "format", False, None, "readable PCM WAV", m.error)
        return ev.findings

    f, s, t, tr = spec["format"], spec["signal"], spec["turn_taking"], spec["transcript"]

    # ---- format
    ev.add("sample_rate", "format", m.sample_rate in f["sample_rates"], m.sample_rate,
           f"one of {f['sample_rates']}", f"{m.sample_rate} Hz")
    ev.add("channels", "format", m.channels == f["channels"], m.channels,
           f"{f['channels']}", f"{m.channels} channel(s)")
    ev.add("bit_depth", "format", m.bit_depth >= f["min_bit_depth"], m.bit_depth,
           f">= {f['min_bit_depth']}", f"{m.bit_depth}-bit")
    d = f["duration_s"]
    ev.add("duration", "format", d["min"] <= m.duration_s <= d["max"], m.duration_s,
           f"{d['min']}-{d['max']} s", f"{m.duration_s:.1f} s")

    # ---- signal, per channel
    for name, cs in m.channel_stats.items():
        ev.add("clipping", "signal", cs.clipping_pct <= s["max_clipping_pct"], cs.clipping_pct,
               f"<= {s['max_clipping_pct']}%", f"{name}: {cs.clipping_pct:.3f}% samples clipped")
        ev.add("dc_offset", "signal", abs(cs.dc_offset) <= s["max_dc_offset"], cs.dc_offset,
               f"|x| <= {s['max_dc_offset']}", f"{name}: DC offset {cs.dc_offset:+.4f}")
        lv = s["speech_level_dbfs"]
        if cs.speech_ratio < DEAD_RATIO:
            ev.add("dead_channel", "signal", False, cs.speech_ratio, "speech on every channel",
                   f"{name}: no usable speech ({cs.speech_ratio:.1%} of session), "
                   f"level {cs.noise_floor_dbfs:.0f} dBFS")
            continue
        else:
            ev.add("speech_level", "signal", lv["min"] <= cs.speech_level_dbfs <= lv["max"],
                   cs.speech_level_dbfs, f"{lv['min']} to {lv['max']} dBFS",
                   f"{name}: speech at {cs.speech_level_dbfs:.1f} dBFS")
        ev.add("snr", "signal", cs.snr_db is not None and cs.snr_db >= s["min_snr_db"], cs.snr_db,
               f">= {s['min_snr_db']} dB",
               f"{name}: SNR {'n/a' if cs.snr_db is None else f'{cs.snr_db:.1f} dB'}")

    two = m.channels == 2
    # ---- channel separation
    for direction, b in m.bleed.items():
        src, dst = direction.split("->")
        ev.add("bleed", "channel", b.leakage_db <= s["max_bleed_db"], b.leakage_db,
               f"<= {s['max_bleed_db']} dB",
               f"speaker {src} leaks into channel {dst} at {b.leakage_db:.1f} dB "
               f"(lag {b.lag_ms:.1f} ms)" + (" — compensated before VAD" if b.compensated else ""))
    if not two:
        ev.add("bleed", "channel", False, None, "2 channels", "needs two channels", skip=True)

    # Root cause first: on a noisy channel the VAD can't find speech reliably,
    # so turn-taking and transcript checks would only add symptom noise.
    noisy = [n for n, cs in m.channel_stats.items()
             if cs.snr_db is not None and cs.snr_db < s["min_snr_db"]]
    if two and noisy:
        msg = f"not scored: SNR too low on {', '.join(noisy)} for reliable speech detection"
        ev.add("turn_taking", "turn_taking", False, None, "", msg, skip=True)
        if m.transcript is not None:
            ev.add("transcript", "transcript", False, None, "", msg, skip=True)
        return ev.findings

    dead = [n for n, cs in m.channel_stats.items() if cs.speech_ratio < DEAD_RATIO]
    # ---- turn-taking
    if two and dead:
        ev.add("turn_taking", "turn_taking", False, None, "",
               f"not scored: channel {', '.join(dead)} has no speech", skip=True)
    elif two and m.turn_taking:
        tt = m.turn_taking
        for name, cs in m.channel_stats.items():
            ev.add("speech_ratio", "turn_taking", cs.speech_ratio >= t["min_speech_ratio"],
                   cs.speech_ratio, f">= {t['min_speech_ratio']:.0%}",
                   f"{name} speaks {cs.speech_ratio:.0%} of the session")
        ev.add("talk_balance", "turn_taking", tt.talk_balance >= t["min_talk_balance"],
               tt.talk_balance, f">= {t['min_talk_balance']}", f"balance {tt.talk_balance:.2f}")
        o = t["overlap_ratio"]
        ev.add("overlap", "turn_taking", o["min"] <= tt.overlap_ratio <= o["max"], tt.overlap_ratio,
               f"{o['min']:.0%}-{o['max']:.0%}", f"{tt.overlap_ratio:.1%} of speech is overlapped")
        ev.add("mutual_silence", "turn_taking",
               tt.longest_mutual_silence_s <= t["max_mutual_silence_s"], tt.longest_mutual_silence_s,
               f"<= {t['max_mutual_silence_s']} s", f"longest silence {tt.longest_mutual_silence_s:.1f} s")
        ev.add("turn_rate", "turn_taking", tt.turns_per_min >= t["min_turns_per_min"], tt.turns_per_min,
               f">= {t['min_turns_per_min']}/min", f"{tt.turns_per_min:.1f} turns/min")
    else:
        ev.add("turn_taking", "turn_taking", False, None, "2 channels",
               "turn-taking needs one speaker per channel", skip=True)

    # ---- transcript
    if m.transcript is None:
        if tr["required"] and two:
            ev.add("transcript_missing", "transcript", False, None, "sidecar .json", "no transcript found")
        return ev.findings
    x = m.transcript
    ev.add("transcript_schema", "transcript", not x.schema_errors, len(x.schema_errors), "0 errors",
           "; ".join(x.schema_errors[:3]) or f"{x.n_segments} segments")
    offset_bad = abs(x.best_offset_s) > tr["max_offset_s"] and x.aligned_at_best - x.alignment > 0.05
    ev.add("transcript_offset", "transcript", not offset_bad, x.best_offset_s,
           f"|offset| <= {tr['max_offset_s']} s",
           f"best alignment at {x.best_offset_s:+.2f} s shift "
           f"({x.alignment:.0%} -> {x.aligned_at_best:.0%})")
    if offset_bad:
        why = f"explained by the {x.best_offset_s:+.2f} s offset ({x.aligned_at_best:.0%} aligned once shifted)"
        ev.add("transcript_alignment", "transcript", False, x.alignment, "", why, skip=True)
        ev.add("transcript_coverage", "transcript", False, x.speech_coverage, "", why, skip=True)
    else:
        ev.add("transcript_alignment", "transcript", x.alignment >= tr["min_alignment"], x.alignment,
               f">= {tr['min_alignment']:.0%}", f"{x.alignment:.0%} of transcript time lines up with speech")
        ev.add("transcript_coverage", "transcript", x.speech_coverage >= tr["min_speech_coverage"],
               x.speech_coverage, f">= {tr['min_speech_coverage']:.0%}",
               f"{x.speech_coverage:.0%} of detected speech is transcribed")
    w = tr["words_per_s"]
    ev.add("speaking_rate", "transcript",
           x.words_per_s is not None and w["min"] <= x.words_per_s <= w["max"], x.words_per_s,
           f"{w['min']}-{w['max']} words/s", f"{x.words_per_s} words/s")
    return ev.findings


def verdict(findings: list[Finding]) -> str:
    st = {f.status for f in findings}
    return "fail" if "fail" in st else "warn" if "warn" in st else "pass"
