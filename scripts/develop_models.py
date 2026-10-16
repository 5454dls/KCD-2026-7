"""Develop and evaluate the three agents on FULL MIMIC-IV, then write aggregate result tables.

Run AFTER the notebook has been executed on the full data (DATA_SOURCE='mimic-iv',
MAX_PATIENTS=None) so that its OUT folder holds the private patient-level files.

    python scripts/develop_models.py \
        --out-dir notebooks/kcd_liver_v2_outputs \
        --mimic-root "\\\\10.7.12.146\\데이터 및 소스\\MIMIC\\MIMIC4-V3.1" \
        --model-dir models

Writes (all aggregate, counts < 10 suppressed):
    <out-dir>/results_full.json          -> copy to site/data/results_full.json
    <out-dir>/results_tables/*.csv       -> paper tables (Table 1, Table 2, ...)
Writes locally only (do NOT commit; trained on credentialed data):
    models/agent3_gbm.joblib, models/agent2_gbm.joblib

The 100 public MIMIC-IV Demo patients (reference/mimic_iv_demo_subject_ids.csv)
are removed from training and testing, so the public replay applies the models
to patients they have never seen.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kcd_liver.develop import develop  # noqa: E402


def read_hosp(root: Path, name: str, usecols=None):
    for ext in (".csv.gz", ".csv"):
        p = root / "hosp" / f"{name}{ext}"
        if p.exists():
            return pd.read_csv(p, usecols=usecols, low_memory=False)
    print(f"warning: hosp/{name} not found under {root}; related tables will be empty")
    return None


def flat_tables(R: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    t1 = []
    for r in R.get("table1") or []:
        row = {"variable": r["variable"]}
        for g in ["overall", "event", "no_event", "development", "evaluation"]:
            v = r.get(g)
            if r["kind"] == "median_iqr" and v:
                row[g] = f"{v[0]:.1f} [{v[1]:.1f}-{v[2]:.1f}]"
            elif r["kind"] == "n_pct" and v:
                row[g] = "<10" if v[0] is None else (f"{v[0]:,} ({v[1] * 100:.1f}%)" if v[1] is not None else f"{v[0]:,}")
            else:
                row[g] = v
        row["SMD (사건 vs 비사건)"] = r.get("smd_event")
        row["SMD (개발 vs 평가)"] = r.get("smd_split")
        t1.append(row)
    pd.DataFrame(t1).to_csv(out / "table1_baseline.csv", index=False, encoding="utf-8-sig")
    rows = []
    for m in (R.get("agent3_performance") or {}).get("models", []):
        rows.append({"agent": "Agent 3 (매시간, 24h 이상 발생)", "model": m["model"], "n_landmarks": m.get("n"), "patients": m.get("patients"),
                     "AUROC": m.get("auroc"), "AUROC_CI": m.get("auroc_ci"), "AUPRC": m.get("auprc"), "AUPRC_CI": m.get("auprc_ci"),
                     "Brier": m.get("brier"), "cal_intercept": m.get("cal_intercept"), "cal_slope": m.get("cal_slope"),
                     "top10_PPV": m.get("top10_ppv"), "top10_sensitivity": m.get("top10_sensitivity")})
    for m in (R.get("agent2_performance") or {}).get("models", []):
        for c in m["per_class"]:
            rows.append({"agent": "Agent 2 (알림 시, 48h 첫 사건)", "model": f"{m['model']} · {c['class']}", "n_landmarks": c.get("n"),
                         "AUROC": c.get("auroc"), "AUROC_CI": c.get("auroc_ci"), "AUPRC": c.get("auprc"), "Brier": c.get("brier"),
                         "cal_intercept": c.get("cal_intercept"), "cal_slope": c.get("cal_slope")})
    for m in (R.get("agent1_performance") or {}).get("models", []):
        rows.append({"agent": "Agent 1 (알림 시, 원인 후보, ICD 대리 기준)", "model": m["model"], "n_landmarks": m.get("n"),
                     "Top1": (m.get("top1") or {}).get("rate"), "Top1_CI": (m.get("top1") or {}).get("ci"),
                     "Top3": (m.get("top3") or {}).get("rate"), "Top3_CI": (m.get("top3") or {}).get("ci"), "abstain": m.get("abstain_rate")})
    pd.DataFrame(rows).to_csv(out / "table2_performance.csv", index=False, encoding="utf-8-sig")
    sh = {k: v for k, v in R.items() if k.startswith("shadow_")}
    (out / "shadow_before_after.json").write_text(json.dumps(sh, ensure_ascii=False, indent=1), encoding="utf-8")
    # paper tables the team asked for: flowchart, 2/3/4/5x basic counts, Agent 3 by target, death model, preprocessing
    for name, key in [("flowchart.csv", "flow"), ("basic_counts_by_threshold.csv", "event_thresholds"),
                      ("agent3_by_threshold.csv", "agent3_by_threshold")]:
        if R.get(key):
            pd.DataFrame(R[key]).to_csv(out / name, index=False, encoding="utf-8-sig")
    if R.get("death_1y_performance"):
        pd.DataFrame([{k: v for k, v in R["death_1y_performance"].items() if not isinstance(v, (list, dict))}]).to_csv(
            out / "death_1y_performance.csv", index=False, encoding="utf-8-sig")
    pre = (R.get("preprocessing") or {}).get("variables")
    if pre:
        pd.DataFrame(pre).to_csv(out / "preprocessing_variables.csv", index=False, encoding="utf-8-sig")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--mimic-root", required=True, type=Path, help="folder that contains hosp/ (admissions, diagnoses_icd)")
    ap.add_argument("--exclude-subjects", type=Path, default=ROOT / "reference" / "mimic_iv_demo_subject_ids.csv")
    ap.add_argument("--source", default="mimic-iv", choices=["mimic-iv", "mimic-iv-demo", "synthetic"])
    ap.add_argument("--min-cell", type=int, default=10)
    ap.add_argument("--model-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--step-hours", type=int, default=1, help="landmark spacing; 1 = hourly like the live agent")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--results", type=Path)
    a = ap.parse_args(argv)
    t = time.time()
    adm = read_hosp(a.mimic_root, "admissions", ["subject_id", "hadm_id", "dischtime"])
    dx = read_hosp(a.mimic_root, "diagnoses_icd", ["subject_id", "hadm_id", "icd_code", "icd_version"])
    excl = pd.read_csv(a.exclude_subjects).subject_id.tolist() if a.exclude_subjects and a.exclude_subjects.exists() else []
    if a.source != "mimic-iv":
        excl = []  # the demo run cannot exclude itself
    R = develop(a.out_dir, adm, dx, excl, source=a.source, min_cell=a.min_cell if a.source == "mimic-iv" else None,
                step_hours=a.step_hours, boot=a.boot, model_dir=a.model_dir)
    from kcd_liver.develop import develop_v2
    R = develop_v2(R, a.out_dir, adm, dx, excl, model_dir=a.model_dir, min_cell=a.min_cell if a.source == "mimic-iv" else None,
                   boot=a.boot)
    R["meta"]["runtime_minutes"] = round((time.time() - t) / 60, 1)
    res = a.results or a.out_dir / ("results_full.json" if a.source == "mimic-iv" else f"results_{a.source}.json")
    res.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
    flat_tables(R, a.out_dir / "results_tables")
    print(f"done in {R['meta']['runtime_minutes']} min -> {res}")
    print("Check the JSON (aggregate only) before sharing; keep model files local.")


if __name__ == "__main__":
    main()
