"""Self-contained HTML delivery report (no JS frameworks, no CDN)."""
from __future__ import annotations

import html
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .scan import Result

E = html.escape

CSS = """
:root{--bg:#f7f6f2;--panel:#fff;--ink:#1b1b19;--muted:#6b6a64;--line:#e4e2db;
--a:#2f6f8f;--b:#c27a2c;--ov:#c2410c;--pass:#2f7d4f;--warn:#a16207;--fail:#b42318;--skip:#8a8984;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--panel:#1d1d1b;--ink:#ecebe6;--muted:#9c9b94;
--line:#2e2d2a;--a:#6aaecf;--b:#e0a35c;--ov:#f07a4a;--pass:#5bbf84;--warn:#e0b44c;--fail:#f07067;--skip:#77766f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 var(--sans)}
main{max-width:1080px;margin:0 auto;padding:40px 20px 80px}
h1{font-size:28px;margin:0 0 4px;letter-spacing:-.01em}h2{font-size:17px;margin:40px 0 12px}
.sub{color:var(--muted);margin:0}.sub code{font-family:var(--mono);font-size:13px}
.verdict{font-size:20px;margin:24px 0 0}.verdict b{font-variant-numeric:tabular-nums}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-top:20px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi .v{font:600 24px/1.2 var(--mono)}.kpi .l{color:var(--muted);font-size:13px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px}
.rej{display:grid;grid-template-columns:170px 1fr;gap:6px 16px;align-items:start}
.rej .bar{height:10px;background:var(--fail);border-radius:3px;margin-top:6px}
.rej .name{font-family:var(--mono);font-size:13px}.rej .fix{color:var(--muted);font-size:13px;margin:2px 0 10px}
.note{color:var(--muted);font-size:13px;margin:10px 0 0}
.conv{background:var(--panel);border:1px solid var(--line);border-radius:10px;margin-bottom:8px}
.conv summary{list-style:none;cursor:pointer;display:grid;
grid-template-columns:110px 74px 70px 1fr;gap:12px;align-items:center;padding:12px 16px}
.conv summary::-webkit-details-marker{display:none}
.conv summary .id{font-family:var(--mono);font-weight:600}.conv summary .why{color:var(--muted);font-size:13px;
white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.num{font-family:var(--mono);font-size:13px;color:var(--muted)}
.pill{font:600 11px/1 var(--sans);text-transform:uppercase;letter-spacing:.06em;padding:5px 8px;border-radius:99px;
text-align:center;color:#fff}.pill.pass{background:var(--pass)}.pill.warn{background:var(--warn)}
.pill.fail{background:var(--fail)}.pill.skip{background:var(--skip)}
.body{padding:4px 16px 16px;border-top:1px solid var(--line)}
.stats{display:flex;flex-wrap:wrap;gap:6px 22px;font-size:13px;margin:12px 0}.stats b{font-family:var(--mono)}
table{width:100%;border-collapse:collapse;font-size:13px}td{padding:6px 8px;border-top:1px solid var(--line);vertical-align:top}
td.c{font-family:var(--mono);white-space:nowrap}td.f{color:var(--muted)}
svg text{fill:var(--muted);font:11px var(--mono)}
.legend{display:flex;gap:16px;font-size:12px;color:var(--muted);margin:6px 0}
.legend i{display:inline-block;width:12px;height:8px;border-radius:2px;margin-right:5px;vertical-align:middle}
footer{margin-top:48px;color:var(--muted);font-size:12px}
@media (max-width:640px){.conv summary{grid-template-columns:1fr auto;}.conv summary .why,.conv summary .num{display:none}
.rej{grid-template-columns:1fr}}
"""


def _timeline(meas: dict) -> str:
    dur = meas.get("duration_s") or 0
    segs = meas.get("vad_segments", {})
    if dur <= 0 or set(segs) != {"A", "B"}:
        return ""
    W, x0 = 1000, 26
    sx = (W - x0 - 4) / dur

    def rects(items, y, h, color, op=1.0):
        return "".join(
            f'<rect x="{x0 + s * sx:.1f}" y="{y}" width="{max(1.0, (e - s) * sx):.1f}" height="{h}" '
            f'fill="var({color})" opacity="{op}" rx="1.5"/>' for s, e in items)

    a, b = segs["A"], segs["B"]
    ov = [(max(s1, s2), min(e1, e2)) for s1, e1 in a for s2, e2 in b if min(e1, e2) > max(s1, s2)]
    tr = (meas.get("transcript") or {}).get("segments") or {}
    step = next(st for st in (5, 10, 30, 60, 300, 600, 1800) if dur / st <= 14)
    ticks = "".join(
        f'<line x1="{x0 + t * sx:.1f}" x2="{x0 + t * sx:.1f}" y1="84" y2="88" stroke="var(--line)"/>'
        f'<text x="{x0 + t * sx:.1f}" y="100" text-anchor="{"end" if t > dur - step / 2 else "middle"}">'
        f'{int(t // 60)}:{int(t % 60):02d}</text>'
        for t in np.arange(0, dur + 1e-6, step))
    return (
        f'<svg viewBox="0 0 {W} 104" width="100%" role="img" aria-label="speech timeline">'
        f'<text x="4" y="30">A</text><text x="4" y="68">B</text>'
        + rects(a, 18, 16, "--a") + rects(tr.get("A", []), 36, 3, "--a", .45)
        + rects(ov, 43, 6, "--ov")
        + rects(b, 54, 16, "--b") + rects(tr.get("B", []), 72, 3, "--b", .45)
        + f'<line x1="{x0}" x2="{W - 4}" y1="84" y2="84" stroke="var(--line)"/>' + ticks + "</svg>"
        '<div class="legend"><span><i style="background:var(--a)"></i>A speech</span>'
        '<span><i style="background:var(--b)"></i>B speech</span>'
        '<span><i style="background:var(--ov)"></i>overlap</span>'
        '<span><i style="background:var(--muted);height:3px"></i>transcript segments</span></div>'
    )


def _fto_hist(ftos: list[float]) -> str:
    if len(ftos) < 5:
        return '<p class="note">Not enough accepted transitions to plot.</p>'
    lo, hi, w = -1.5, 2.0, 0.1
    edges = np.arange(lo, hi + w / 2, w)
    counts, _ = np.histogram(np.clip(ftos, lo, hi - 1e-6), bins=edges)
    W, H, x0, y0 = 1000, 190, 36, 160
    bw = (W - x0 - 10) / len(counts)
    mx = counts.max() or 1
    bars = "".join(
        f'<rect x="{x0 + i * bw + 1:.1f}" y="{y0 - c / mx * 140:.1f}" width="{bw - 2:.1f}" '
        f'height="{c / mx * 140:.1f}" fill="var({"--ov" if edges[i] < 0 else "--a"})" rx="1.5">'
        f'<title>{edges[i]:+.1f} to {edges[i] + w:+.1f} s: {c}</title></rect>' for i, c in enumerate(counts))
    zx = x0 + (0 - lo) / w * bw
    labels = "".join(
        f'<text x="{x0 + (v - lo) / w * bw:.1f}" y="178" text-anchor="middle">{v:+.1f}s</text>'
        for v in (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0))
    return (f'<svg viewBox="0 0 {W} {H + 4}" width="100%" role="img" aria-label="floor transfer offsets">'
            f'{bars}<line x1="{zx:.1f}" x2="{zx:.1f}" y1="12" y2="{y0}" stroke="var(--muted)" '
            f'stroke-dasharray="3 3"/><line x1="{x0}" x2="{W - 10}" y1="{y0}" y2="{y0}" '
            f'stroke="var(--line)"/>{labels}</svg>')


def _conv(r: "Result") -> str:
    m, fs = r.measurement, r.findings
    fails = [f for f in fs if f["status"] in ("fail", "warn")]
    why = "; ".join(dict.fromkeys(f["message"] for f in fails)) or "meets spec"
    tt = m.get("turn_taking") or {}
    stats = []
    if m.get("sample_rate"):
        stats.append(f"<span><b>{m['sample_rate']}</b> Hz · <b>{m['channels']}</b> ch · "
                     f"<b>{m['bit_depth']}</b>-bit</span>")
    if tt:
        mf = tt.get("median_fto_s")
        stats += [f"<span>overlap <b>{tt['overlap_ratio']:.1%}</b></span>",
                  f"<span>balance <b>{tt['talk_balance']:.2f}</b></span>",
                  f"<span>median FTO <b>{'n/a' if mf is None else f'{mf:+.2f}s'}</b></span>",
                  f"<span>turns/min <b>{tt['turns_per_min']:.1f}</b></span>",
                  f"<span>backchannels <b>{sum(tt['backchannels'].values())}</b></span>"]
    rows = "".join(
        f'<tr><td><span class="pill {f["status"]}">{f["status"]}</span></td><td class="c">{E(f["check"])}</td>'
        f'<td>{E(f["message"])}{"<br><span class=note>expected " + E(f["expected"]) + "</span>" if f["expected"] and f["status"] != "pass" else ""}</td>'
        f'<td class="f">{E(f["fix"])}</td></tr>'
        for f in sorted(fs, key=lambda f: {"fail": 0, "warn": 1, "skip": 2, "pass": 3}[f["status"]]))
    dur = f"{r.duration_s / 60:.1f} min" if r.duration_s else "—"
    return (f'<details class="conv"><summary><span class="id">{E(r.id)}</span>'
            f'<span class="pill {r.verdict}">{"accept" if r.verdict != "fail" else "reject"}</span>'
            f'<span class="num">{dur}</span><span class="why">{E(why)}</span></summary>'
            f'<div class="body"><div class="stats">{"".join(stats)}</div>{_timeline(m)}'
            f'<table>{rows}</table></div></details>')


def render_html(results: list["Result"], s: dict) -> str:
    tt = s["accepted_turn_taking"]
    fto, ov = tt["fto_s"], tt["overlap_ratio"]
    rej = s["rejections_by_check"]
    mx = max((x["conversations"] for x in rej), default=1)
    rej_html = "".join(
        f'<div class="name">{E(x["check"])} · {x["conversations"]}</div>'
        f'<div><div class="bar" style="width:{100 * x["conversations"] / mx:.0f}%"></div>'
        f'<div class="fix">{E(x["fix"])}</div></div>' for x in rej) or "<p>No rejections.</p>"
    accepted = [r for r in results if r.verdict != "fail"]
    ftos = [x for r in accepted for x in (r.measurement.get("turn_taking") or {}).get("ftos", [])]
    order = {"fail": 0, "warn": 1, "pass": 2}
    convs = "".join(_conv(r) for r in sorted(results, key=lambda r: (order[r.verdict], r.id)))
    kpi = lambda v, l: f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>'
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>duplex-qc report</title>
<style>{CSS}</style></head><body><main>
<h1>Delivery QA report</h1>
<p class="sub">Spec <code>{E(s['spec'])}</code> · source <code>{E(s['source'])}</code> · {E(s['generated_at'])}</p>
<p class="sub">{E(s['spec_description'])}</p>
<p class="verdict"><b>{s['passed'] + s['warned']}</b> of <b>{s['conversations']}</b> conversations accepted ·
<b>{s['hours_accepted'] * 60:.1f}</b> of <b>{s['hours_delivered'] * 60:.1f}</b> minutes usable</p>
<div class="kpis">
{kpi(f"{s['acceptance_rate']:.0%}", "acceptance rate")}
{kpi(s['rejected'], "conversations rejected")}
{kpi("—" if not ov else f"{ov['median']:.1%}", "median overlap (accepted)")}
{kpi("—" if not fto else f"{fto['median'] * 1000:+.0f} ms", "median floor-transfer offset")}
{kpi(tt['transitions'], "speaker transitions measured")}
</div>
<h2>Why files were rejected</h2>
<div class="panel rej">{rej_html}</div>
<p class="note">Counts are conversations failing each check; one file can fail several. Symptoms of a
root cause (e.g. turn-taking on a noisy channel) are skipped rather than double-counted.</p>
<h2>Turn-taking in accepted conversations</h2>
<div class="panel">{_fto_hist(ftos)}
<p class="note">Floor-transfer offset = next speaker's start minus current speaker's end. Orange bars are
overlapping hand-offs. Natural conversation clusters around +200 ms with a real share of overlaps;
stitched or turn-by-turn recordings show no mass left of zero.</p></div>
<h2>Conversations</h2>
{convs}
<footer>Generated by duplex-qc. Energy-based VAD; metrics are estimates intended for triage, not ground truth.</footer>
</main></body></html>"""
