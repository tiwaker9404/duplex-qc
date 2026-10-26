"""Scan a delivery folder and write the accept/reject outputs."""
from __future__ import annotations

import csv
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .measure import measure
from .spec import FIXES, evaluate, verdict


@dataclass
class Result:
    id: str
    path: str
    verdict: str
    duration_s: float
    measurement: dict
    findings: list[dict]


def check_one(path: str, spec: dict) -> Result:
    m = measure(path)
    fs = evaluate(m, spec)
    return Result(m.id, m.path, verdict(fs), m.duration_s or 0.0, m.to_dict(), [f.to_dict() for f in fs])


def scan(folder: str | Path, spec: dict, workers: int = 1) -> list[Result]:
    paths = sorted(str(p) for p in Path(folder).rglob("*.wav"))
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            return list(ex.map(check_one, paths, [spec] * len(paths)))
    return [check_one(p, spec) for p in paths]


def summarize(results: list[Result], spec: dict, source: str) -> dict:
    accepted = [r for r in results if r.verdict != "fail"]
    fail_by_check: Counter = Counter()
    for r in results:
        for c in {f["check"] for f in r.findings if f["status"] == "fail"}:
            fail_by_check[c] += 1

    tts = [r.measurement["turn_taking"] for r in accepted if r.measurement.get("turn_taking")]
    ftos = [x for t in tts for x in t["ftos"]]
    total_s = sum(r.duration_s for r in results)
    acc_s = sum(r.duration_s for r in accepted)

    def stat(vals):
        return None if not vals else {"median": round(float(np.median(vals)), 3),
                                      "p10": round(float(np.percentile(vals, 10)), 3),
                                      "p90": round(float(np.percentile(vals, 90)), 3)}

    return {
        "source": source,
        "spec": spec["name"],
        "spec_description": spec.get("description", ""),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "conversations": len(results),
        "passed": sum(r.verdict == "pass" for r in results),
        "warned": sum(r.verdict == "warn" for r in results),
        "rejected": sum(r.verdict == "fail" for r in results),
        "acceptance_rate": round(len(accepted) / len(results), 3) if results else 0.0,
        "hours_delivered": round(total_s / 3600, 3),
        "hours_accepted": round(acc_s / 3600, 3),
        "rejections_by_check": [
            {"check": c, "conversations": n, "fix": FIXES.get(c, "")} for c, n in fail_by_check.most_common()
        ],
        "accepted_turn_taking": {
            "overlap_ratio": stat([t["overlap_ratio"] for t in tts]),
            "talk_balance": stat([t["talk_balance"] for t in tts]),
            "fto_s": stat(ftos),
            "turns_per_min": stat([t["turns_per_min"] for t in tts]),
            "transitions": len(ftos),
        },
    }


def write_outputs(results: list[Result], summary: dict, out: str | Path) -> dict[str, Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "results": out / "results.jsonl",
        "summary": out / "summary.json",
        "accepted": out / "accepted.jsonl",
        "rejected": out / "rejected.csv",
    }
    with files["results"].open("w") as f:
        for r in results:
            f.write(json.dumps({"id": r.id, "path": r.path, "verdict": r.verdict,
                                "findings": r.findings, "measurement": r.measurement}) + "\n")
    files["summary"].write_text(json.dumps(summary, indent=2))
    with files["accepted"].open("w") as f:
        for r in results:
            if r.verdict != "fail":
                tr = Path(r.path).with_suffix(".json")
                f.write(json.dumps({"id": r.id, "audio": r.path, "duration_s": r.duration_s,
                                    "transcript": str(tr) if tr.exists() else None,
                                    "verdict": r.verdict}) + "\n")
    with files["rejected"].open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "failed_checks", "details", "suggested_fix"])
        for r in results:
            fails = [x for x in r.findings if x["status"] == "fail"]
            if fails:
                w.writerow([r.id, ";".join(dict.fromkeys(x["check"] for x in fails)),
                            " | ".join(x["message"] for x in fails), fails[0]["fix"]])
    from .report import render_html
    files["report"] = out / "report.html"
    files["report"].write_text(render_html(results, summary))
    return files
