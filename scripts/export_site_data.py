"""Export notebook outputs (replay.sqlite + aggregate CSVs) to static JSON for GitHub Pages.

Data governance (the repository is public):
- Patient-level replay frames are exported ONLY for ``--source mimic-iv-demo``
  (MIMIC-IV Clinical Database Demo, ODbL v1.0) or ``--source synthetic``.
- For the credentialed full MIMIC-IV use ``--source mimic-iv --aggregates-only``.
  Only cohort-level tables are written, and counts below ``--min-cell`` are
  suppressed. Never commit replay.sqlite or *_private.csv from the full data.

Usage:
  python scripts/export_site_data.py --out-dir kcd_liver_v2_outputs --source mimic-iv-demo
  python scripts/export_site_data.py --out-dir kcd_liver_v2_outputs --source mimic-iv --aggregates-only \
      --summary-name summary_full.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kcd_liver.evaluation import adoption_table, wilson  # noqa: E402
from kcd_liver.site_aggregates import suppress  # noqa: E402

PUBLIC_CASE_SOURCES = {"mimic-iv-demo", "synthetic"}

# CSV -> summary key. Missing files stay null and the site shows "미집계".
AGGREGATE_FILES = {
    "cohort_flow": "04_cohort_flow.csv",
    "event_groups": "15_event_groups.csv",
    "alert_aggregation": "18_alert_aggregation.csv",
    "threshold_sensitivity": "33_threshold_sensitivity.csv",
    "cause_validation": "12_cause_validation.csv",
    "stress_before_after_specialist": "24_stress_before_after.csv",
    "agent2_complete_case_scores": "21_agent2_complete_case_scores.csv",
    "agent3_complete_case_scores": "22_agent3_complete_case_scores.csv",
    "paired_test": "35_paired_mcnemar.csv",
}
COUNT_COLUMNS = {"n", "k", "n_patients_flagged", "n_known", "n_all", "complete_n", "all_n",
                 "base_only", "new_only", "discordant", "support_items", "verified"}


def clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [clean(x) for x in v]
    return v


def num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if not math.isfinite(f):
        return None
    return int(f) if f.is_integer() and "." not in str(v) else f


def read_csv(path: Path, min_cell: int | None) -> list[dict] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = [{k: num(v) if v != "" else None for k, v in r.items()} for r in csv.DictReader(fh)]
    if not rows:
        return None  # intentionally blank tables stay "not aggregated"
    if min_cell:
        for r in rows:
            for k in list(r):
                if k in COUNT_COLUMNS and isinstance(r[k], (int, float)) and 0 < r[k] < min_cell:
                    r[k] = None
                    r.setdefault("_suppressed", []).append(k)
    return rows


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def export_cases(db_path: Path, site_data: Path) -> list[dict]:
    con = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    cols = [r[1] for r in con.execute("PRAGMA table_info(cases)")]
    extra = ", cohort_group" if "cohort_group" in cols else ""
    cases = []
    (site_data / "cases").mkdir(parents=True, exist_ok=True)
    for row in con.execute(f"SELECT case_key, max_hour{extra} FROM cases ORDER BY case_key").fetchall():
        key, max_hour = row[0], int(row[1])
        group = row[2] if extra else None
        frames = [{"hour": h, "labs": json.loads(l), "agent1": json.loads(a1),
                   "agent2": json.loads(a2), "agent3": json.loads(a3)}
                  for h, l, a1, a2, a3 in con.execute(
                      "SELECT hour, labs_json, agent1_json, agent2_json, agent3_json FROM frames "
                      "WHERE case_key=? ORDER BY hour", (key,))]
        stress = [{"input_condition": c, "guarded": bool(g), "answer": json.loads(r)}
                  for c, g, r in con.execute(
                      "SELECT condition, guarded, result_json FROM stress WHERE case_key=?", (key,))]
        for s in stress:  # failures are kept, not dropped
            if isinstance(s["answer"], dict) and "error" in s["answer"] and "candidates" not in s["answer"]:
                s["error"] = s["answer"].get("error")
        obs = con.execute("SELECT outcome_json FROM observed_outcomes WHERE case_key=?", (key,)).fetchone()
        doc = {"case_key": key, "max_hour": max_hour, "group": group, "frames": frames,
               "stress": stress, "observed": json.loads(obs[0]) if obs else None}
        (site_data / "cases" / f"{key}.json").write_text(
            json.dumps(clean(doc), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        cases.append({"case_key": key, "max_hour": max_hour, "group": group})
    con.close()
    return cases


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, type=Path, help="notebook OUT folder")
    ap.add_argument("--site-data", type=Path, default=ROOT / "site" / "data")
    ap.add_argument("--source", required=True, choices=["mimic-iv-demo", "synthetic", "mimic-iv"])
    ap.add_argument("--aggregates-only", action="store_true")
    ap.add_argument("--min-cell", type=int, default=10, help="suppress counts below this (aggregates)")
    ap.add_argument("--summary-name", default="summary.json")
    ap.add_argument("--injected-cause", default="drug_related",
                    help="cause code injected by the misleading-input experiment")
    args = ap.parse_args(argv)

    if args.source not in PUBLIC_CASE_SOURCES and not args.aggregates_only:
        ap.error("Full MIMIC-IV is credentialed data: patient-level frames cannot go to a public site. "
                 "Use --aggregates-only, or run the notebook on mimic-iv-demo.")
    args.site_data.mkdir(parents=True, exist_ok=True)
    min_cell = args.min_cell if args.source == "mimic-iv" else None

    summary = {"source": args.source, "min_cell_suppression": min_cell}
    for key, name in AGGREGATE_FILES.items():
        summary[key] = read_csv(args.out_dir / name, min_cell)
    if summary["cause_validation"]:
        for r in summary["cause_validation"]:
            if r.get("k") is not None and r.get("n"):
                r["ci_low"], r["ci_high"] = wilson(int(r["k"]), int(r["n"]))
    stress_file = args.out_dir / "17_input_stress.jsonl"
    if stress_file.exists():
        rows = [json.loads(x) for x in stress_file.read_text(encoding="utf-8").splitlines() if x.strip()]
        table = adoption_table(rows, args.injected_cause)
        if min_cell:
            for r in table:
                for k in ("n", "adopted", "errors", "abstained"):
                    if 0 < r[k] < min_cell:
                        r[k] = None
                        r["adoption_rate"] = r["ci_low"] = r["ci_high"] = None
        summary["stress_adoption"] = table
        summary["injected_cause"] = args.injected_cause
    if summary.get("paired_test"):
        summary["paired_test"] = summary["paired_test"][0]
    # Figure aggregates computed in the notebook from in-memory tables take precedence.
    agg_path = args.out_dir / "40_site_aggregates.json"
    if agg_path.exists():
        agg = json.loads(agg_path.read_text(encoding="utf-8"))
        if agg.get("source") and agg["source"] != args.source:
            ap.error(f"40_site_aggregates.json was built for source={agg['source']}, not {args.source}")
        summary.update({k: v for k, v in agg.items() if v is not None})
    summary = suppress(summary, min_cell)
    (args.site_data / args.summary_name).write_text(
        json.dumps(clean(summary), ensure_ascii=False, indent=1), encoding="utf-8")

    manifest_path = args.site_data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    config_path = args.out_dir / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    public_config = {k: config.get(k) for k in
                     ["seed", "max_patients", "threshold", "LLM_enabled", "cluster_k", "backend",
                      "m4_reviewed_commit", "python", "pandas", "installed_versions"] if k in config}
    if not args.aggregates_only:
        db = args.out_dir / "replay.sqlite"
        if not db.exists():
            ap.error(f"{db} not found: run the notebook's replay export cell first")
        manifest.update({"source": args.source, "cases": export_cases(db, args.site_data),
                         "config": public_config})
    manifest.setdefault("summaries", {})[args.summary_name] = {
        "source": args.source, "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    manifest.update({"built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "git_commit": git_commit()})
    manifest_path.write_text(json.dumps(clean(manifest), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {args.site_data} (source={args.source}, cases={len(manifest.get('cases', []))})")


if __name__ == "__main__":
    main()
