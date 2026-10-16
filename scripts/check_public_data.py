"""Fail if the public site would expose credentialed MIMIC-IV patient-level data."""
from __future__ import annotations

import fnmatch
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ["*.sqlite", "*.sqlite3", "*_private.csv", "*EVALUATION_ONLY*", "*.duckdb"]
LOCAL_ONLY = {".git", "run_all", "data", "models", "m4_data"}
DEEP_LOCAL = {".git", "run_all", "kcd_liver_v2_outputs", "specialist_blinded_packets"}


def committable_files() -> list[Path]:
    """Files git could commit (tracked + untracked not ignored); without git, skip LOCAL_ONLY folders."""
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT,
                             capture_output=True, check=True).stdout.decode("utf-8")
        return [Path(x) for x in out.split("\0") if x and (ROOT / x).exists()]
    except (OSError, subprocess.CalledProcessError):
        files = []
        for p in ROOT.rglob("*"):
            rel = p.relative_to(ROOT)
            if p.is_file() and rel.parts[0] not in LOCAL_ONLY and not set(rel.parts) & DEEP_LOCAL:
                files.append(rel)
        return files


def main() -> int:
    problems = []
    for rel in committable_files():
        if any(fnmatch.fnmatch(rel.name, pattern) for pattern in FORBIDDEN):
            problems.append(f"forbidden file in repo tree: {rel}")
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
