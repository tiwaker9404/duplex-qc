# duplex-qc

**A QA gate for two-channel conversational speech datasets.** Point it at a delivery folder and a
customer's spec; get back which conversations are usable, why the rest aren't, and what to tell
the recording team.

Built for the kind of data full-duplex voice models train on: two people, one close-talk mic each,
talking naturally — with overlaps, backchannels ("mm-hmm") and interruptions.

```
$ duplex-qc synth demo/
$ duplex-qc scan demo/ --spec specs/full_duplex_v1.yaml --out qc_out/
5/15 accepted (33%) · 5.0/15.2 min usable · spec full-duplex-conversational-v1
  ✗ overlap                2 conversation(s)
  ✗ clipping               1 conversation(s)
  ✗ bleed                  1 conversation(s)
  ✗ dead_channel           1 conversation(s)
  ...
report: qc_out/report.html
```

Runs locally on CPU in seconds. No API keys, no model downloads, three dependencies
(numpy, scipy, pyyaml).

## Why this exists

Generic audio QA checks levels and sample rates. That's necessary, but the defects that actually
ruin a conversational dataset are structural:

| Defect | What it looks like | How duplex-qc catches it |
|---|---|---|
| **Mic bleed** | Speaker A is audible on B's channel, so a model learns B "said" A's words | Block-wise least-squares leakage estimate with direction; leak is subtracted before VAD so turn metrics stay correct |
| **Stitched turns** | Sessions recorded turn-by-turn and concatenated — zero overlap, no backchannels | Overlap ratio + floor-transfer-offset (FTO) distribution |
| **Interview, not conversation** | One speaker talks 90% of the time | Talk balance, per-speaker speech ratio |
| **Dead or muted channel** | One side is silent | Speech-ratio floor, reported as one root cause |
| **Transcript drift** | Timestamps shifted by a constant | Searches ±3 s for the shift that best aligns transcript to speech and reports it, so ops can fix it with one offset instead of re-transcribing |
| Clipping, noise, wrong format | The usual | Per-channel clipping %, SNR, speech level, DC offset, format |

It reports **root causes, not symptoms**: a dead mic is one finding, not six; a noisy channel
suppresses turn-taking checks that the VAD can't score reliably; a constant transcript shift
explains the alignment and coverage failures it causes.

## Specs are files

Every lab defines "good data" differently. The contract lives in YAML both sides can read and
version ([`specs/full_duplex_v1.yaml`](specs/full_duplex_v1.yaml)):

```yaml
turn_taking:
  min_talk_balance: 0.35
  overlap_ratio: {min: 0.03, max: 0.35}
signal:
  max_bleed_db: -30
severity:
  dc_offset: warn      # accept with a warning instead of rejecting
```

## Outputs

| File | For |
|---|---|
| `report.html` | Self-contained delivery report: acceptance, rejection reasons with fixes, FTO histogram, per-conversation speech timelines |
| `accepted.jsonl` | Manifest of usable audio + transcript paths, ready for a training pipeline |
| `rejected.csv` | One row per rejected file with failed checks and the suggested fix — hand to ops/vendors |
| `summary.json`, `results.jsonl` | Machine-readable metrics for dashboards or diffing deliveries |

`--fail-under 0.9` exits non-zero when acceptance drops below 90%, so it works as a CI gate on
incoming batches.

## Input format

- `*.wav` PCM (8/16/24/32-bit), one speaker per channel: channel 0 = speaker A, channel 1 = B
- Optional transcript sidecar with the same name, `conv_001.json`:

```json
{"segments": [{"speaker": "A", "start": 1.20, "end": 3.85, "text": "so what did you think"}]}
```

## How it measures

- **VAD:** per-channel frame energy with an adaptive threshold above each channel's own noise
  floor, plus hangover smoothing. Simple and transparent; swap in Silero or WebRTC VAD in
  `vad.py` for noisier data.
- **Bleed:** whole-file correlation can't tell A→B from B→A, so the gain is estimated per 1-second
  block, only in blocks where the source speaker is loudest. Real bleed gives a consistent gain;
  the mirror direction averages out near zero.
- **Turn-taking:** short segments contained in the other speaker's turn are backchannels;
  remaining segments define floor transfers. FTO = next speaker's start − current speaker's end
  (negative = overlapping hand-off). Natural conversation clusters around +200 ms.
- **Transcript:** alignment = share of transcript time with detected speech; coverage = share of
  detected speech that's transcribed; offset = shift maximizing alignment.

## Demo data and accuracy

`duplex-qc synth` generates 15 synthetic conversations (speech-like harmonic signals with moving
formants, realistic turn schedules) — 5 clean and 10 with one injected defect each. No real
voices, nothing to license, fully reproducible. `_fixture_key.json` is the answer key.

On a 100-file sweep (20 seeds each, 60 s files): clean files accepted 20/20; stitched, bleed and
transcript-drift defects caught 20/20; monologue-style imbalance caught on talk balance 18/20.

## Limits

- Energy VAD is fine for close-talk channels, weaker in far-field or very noisy audio.
- Leakage assumes roughly linear bleed at a small fixed lag (≤30 ms); reverberant rooms need more.
- Speaker-to-channel assignment is assumed, not verified by voice.
- Synthetic fixtures prove the checks fire correctly; thresholds should be tuned on real deliveries.

## Develop

```
pip install -e ".[dev]"
pytest -q
```

MIT licensed.
