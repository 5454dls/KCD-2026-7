"""External validation CLI: frozen models -> results_external.json (aggregate only).

python scripts/external_validate.py --out-dir <notebook outputs on the external data> --model-dir models --data-root <MIMIC-layout root>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kcd_liver.external import validate  # noqa: E402
from scripts.develop_models import read_hosp  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--site", default="external")
    ap.add_argument("--min-cell", type=int, default=10)
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args(argv)
    adm = read_hosp(a.data_root, "admissions", ["subject_id", "hadm_id", "dischtime"])
    dx = read_hosp(a.data_root, "diagnoses_icd", ["subject_id", "hadm_id", "icd_code", "icd_version"])
    R = validate(a.out_dir, a.model_dir, adm, dx, boot=a.boot, min_cell=a.min_cell, site=a.site)
    p = a.out_dir / "results_external.json"
    p.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
