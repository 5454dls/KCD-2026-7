"""Fail if the public site would expose credentialed MIMIC-IV patient-level data.

Run in CI before every Pages deploy and locally before committing site/data.
Rules:
- no *.sqlite / *_private.csv / *EVALUATION_ONLY* anywhere in the repository tree
- site/data/cases/* is allowed only when manifest.source is mimic-iv-demo or synthetic
- summary files built from full mimic-iv must carry min_cell_suppression
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ["*.sqlite", "*.sqlite3", "*_private.csv", "*EVALUATION_ONLY*", "*.duckdb"]


def main() -> int:
    problems = []
    for pattern in FORBIDDEN:
        for p in ROOT.rglob(pattern):
            if ".git" not in p.parts:
                problems.append(f"forbidden file in repo tree: {p.relative_to(ROOT)}")
    data = ROOT / "site" / "data"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8")) if (data / "manifest.json").exists() else {}
    cases = list((data / "cases").glob("*.json")) if (data / "cases").exists() else []
    if cases and manifest.get("source") not in {"mimic-iv-demo", "synthetic"}:
        problems.append(f"patient-level case frames present with source={manifest.get('source')!r}")
    for f in data.glob("summary*.json"):
        s = json.loads(f.read_text(encoding="utf-8"))
        if s.get("source") == "mimic-iv" and not s.get("min_cell_suppression"):
            problems.append(f"{f.name}: full MIMIC-IV aggregates without small-cell suppression")
    for f in data.glob("results*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        meta = r.get("meta", {})
        if meta.get("source") == "mimic-iv" and not meta.get("min_cell_suppression"):
            problems.append(f"{f.name}: full MIMIC-IV results without small-cell suppression")
        if f.name == "results_full.json" and meta.get("source") != "mimic-iv":
            problems.append(f"{f.name}: expected source mimic-iv, found {meta.get('source')!r}")
    for p in problems:
        print("DATA GUARD:", p)
    print("public data check:", "FAILED" if problems else "ok", f"(source={manifest.get('source')}, cases={len(cases)})")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
