"""duplex-qc command line."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .scan import check_one, scan, summarize, write_outputs
from .spec import load_spec

ICON = {"pass": "✓", "warn": "!", "fail": "✗", "skip": "·"}


def _scan(a) -> int:
    spec = load_spec(a.spec)
    results = scan(a.folder, spec, a.workers)
    if not results:
        print(f"no .wav files under {a.folder}", file=sys.stderr)
        return 2
    summary = summarize(results, spec, str(a.folder))
    files = write_outputs(results, summary, a.out)
    print(f"{summary['passed'] + summary['warned']}/{summary['conversations']} accepted "
          f"({summary['acceptance_rate']:.0%}) · {summary['hours_accepted'] * 60:.1f}/"
          f"{summary['hours_delivered'] * 60:.1f} min usable · spec {spec['name']}")
    for x in summary["rejections_by_check"]:
        print(f"  ✗ {x['check']:<22} {x['conversations']} conversation(s)")
    print(f"report: {files['report']}")
    if a.fail_under is not None and summary["acceptance_rate"] < a.fail_under:
        print(f"acceptance {summary['acceptance_rate']:.0%} is below --fail-under {a.fail_under:.0%}",
              file=sys.stderr)
        return 1
    return 0


def _inspect(a) -> int:
    r = check_one(a.file, load_spec(a.spec))
    print(f"{r.id}: {r.verdict.upper()}")
    for f in r.findings:
        if a.all or f["status"] != "pass":
            print(f"  {ICON[f['status']]} {f['check']:<22} {f['message']}")
            if f["fix"]:
                print(f"    → {f['fix']}")
    return 0 if r.verdict != "fail" else 1


def _synth(a) -> int:
    from .synth import write_demo
    paths = write_demo(a.out, a.seconds)
    print(f"wrote {len(paths)} conversations to {a.out} (answer key: _fixture_key.json)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="duplex-qc",
                                description="QA gate for two-channel conversational speech datasets.")
    p.add_argument("--version", action="version", version=f"duplex-qc {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="check every .wav in a folder against a spec")
    s.add_argument("folder", type=Path)
    s.add_argument("--spec", type=Path, help="YAML spec (defaults built in)")
    s.add_argument("--out", type=Path, default=Path("qc_out"))
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--fail-under", type=float, help="exit 1 if acceptance rate is below this (0-1)")
    s.set_defaults(fn=_scan)

    i = sub.add_parser("inspect", help="check one file and print findings")
    i.add_argument("file", type=Path)
    i.add_argument("--spec", type=Path)
    i.add_argument("--all", action="store_true", help="show passing checks too")
    i.set_defaults(fn=_inspect)

    g = sub.add_parser("synth", help="generate the synthetic demo delivery")
    g.add_argument("out", type=Path)
    g.add_argument("--seconds", type=float, default=60.0)
    g.set_defaults(fn=_synth)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
