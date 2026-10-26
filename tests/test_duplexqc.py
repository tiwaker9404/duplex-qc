"""Each injected defect must be caught by the right check, and clean files must pass.

Fixtures are synthesized at test time (30 s each) so nothing binary lives in the repo.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from duplexqc.audio import read_wav, write_wav
from duplexqc.cli import main
from duplexqc.measure import estimate_leakage, measure, turn_taking
from duplexqc.spec import evaluate, load_spec, verdict
from duplexqc.synth import build

SPEC = load_spec(Path(__file__).parent.parent / "specs" / "full_duplex_v1.yaml")
SPEC["format"]["duration_s"]["min"] = 10


def _write(tmp: Path, kind: str, seed: int = 7, seconds: float = 30.0) -> Path:
    x, sr, tr = build(kind, seed=seed, total=seconds)
    p = tmp / f"{kind}.wav"
    write_wav(p, x, sr)
    if tr is not None:
        p.with_suffix(".json").write_text(json.dumps(tr))
    return p


def _failed(path: Path) -> set[str]:
    return {f.check for f in evaluate(measure(path), SPEC) if f.status == "fail"}


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_clean_conversations_pass(tmp_path, seed):
    p = _write(tmp_path, "clean", seed)
    fs = evaluate(measure(p), SPEC)
    assert verdict(fs) == "pass", [f.message for f in fs if f.status == "fail"]


@pytest.mark.parametrize("kind,expected", [
    ("clipped", {"clipping"}),
    ("bleed", {"bleed"}),
    ("dead_channel", {"dead_channel"}),
    ("stitched", {"overlap"}),
    ("monologue", {"talk_balance"}),
    ("transcript_drift", {"transcript_offset"}),
    ("long_silence", {"mutual_silence"}),
    ("noisy", {"snr"}),
    ("wrong_sample_rate", {"sample_rate"}),
    ("mono_mixdown", {"channels"}),
])
def test_defects_are_caught(tmp_path, kind, expected):
    assert expected <= _failed(_write(tmp_path, kind))


def test_root_causes_suppress_symptoms(tmp_path):
    # a dead mic should not also be reported as bad balance/overlap/turn rate
    assert _failed(_write(tmp_path, "dead_channel")) == {"dead_channel"}
    # a constant transcript shift is one problem, not three
    failed = _failed(_write(tmp_path, "transcript_drift"))
    assert "transcript_alignment" not in failed and "transcript_coverage" not in failed


def test_transcript_offset_is_recovered(tmp_path):
    m = measure(_write(tmp_path, "transcript_drift"))
    assert m.transcript.best_offset_s == pytest.approx(-1.2, abs=0.1)


def test_bleed_direction_and_compensation(tmp_path):
    m = measure(_write(tmp_path, "bleed"))
    assert m.bleed["A->B"].leakage_db == pytest.approx(-18.4, abs=1.5)
    assert m.bleed["B->A"].leakage_db < -40          # mirror direction is not reported
    assert m.turn_taking.overlap_ratio < 0.2          # VAD wasn't fooled by the leak


def test_leakage_on_independent_noise_is_low():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(16000 * 20) * np.repeat(rng.random(20) > 0.5, 16000)
    b = rng.standard_normal(16000 * 20) * 0.5
    g, _ = estimate_leakage(a, b, 16000)
    assert 20 * np.log10(abs(g) + 1e-9) < -30


def test_turn_taking_math():
    segs = {"A": [(0.0, 3.0), (6.0, 9.0), (4.0, 4.4)], "B": [(2.5, 5.5), (1.0, 1.5)]}
    tt = turn_taking({"A": sorted(segs["A"]), "B": sorted(segs["B"])}, 10.0)
    # B's (1.0,1.5) sits inside A's turn -> backchannel; A's (4.0,4.4) inside B's -> backchannel
    assert tt.backchannels == {"A": 1, "B": 1}
    assert tt.ftos == [-0.5, 0.5]  # B takes floor 0.5 s early, A returns 0.5 s after B stops
    assert tt.turns == 3


def test_wav_roundtrip(tmp_path):
    x = np.stack([np.sin(np.linspace(0, 100, 8000)), np.zeros(8000)], axis=1) * 0.5
    write_wav(tmp_path / "t.wav", x, 16000)
    a = read_wav(tmp_path / "t.wav")
    assert a.channels == 2 and a.sr == 16000 and np.allclose(a.samples, x, atol=1e-4)


def test_unreadable_file(tmp_path):
    p = tmp_path / "broken.wav"
    p.write_bytes(b"not a wav")
    assert _failed(p) == {"unreadable"}


def test_cli_scan_writes_outputs(tmp_path):
    (tmp_path / "in").mkdir()
    for k in ("clean", "stitched"):
        _write(tmp_path / "in", k)
    out = tmp_path / "out"
    rc = main(["scan", str(tmp_path / "in"), "--spec", str(Path(__file__).parent.parent / "specs" / "full_duplex_v1.yaml"),
               "--out", str(out), "--fail-under", "0.9"])
    assert rc == 1  # 1 of 2 accepted is under 90%
    for f in ("report.html", "summary.json", "results.jsonl", "accepted.jsonl", "rejected.csv"):
        assert (out / f).exists()
    s = json.loads((out / "summary.json").read_text())
    assert s["conversations"] == 2
