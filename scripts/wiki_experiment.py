"""Run the wiki before/after experiment on evaluation events (aggregate output only).

LLM_BASE_URL=... LLM_MODEL=... python scripts/wiki_experiment.py --out-dir <notebook outputs> --wiki wiki_kb --mimic-root <root> --n 200
Full MIMIC-IV snapshots may only go to an approved, no-retention endpoint (PhysioNet credentialed data use).
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kcd_liver.develop import history_flags, icd_proxy, load_private, snapshot_payload, _suppress_counts  # noqa: E402
from kcd_liver.wiki_experiment import POISON_PAGE, run  # noqa: E402
from kcd_liver.wiki_rag import WikiRetriever, build_index  # noqa: E402
from scripts.develop_models import read_hosp  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--wiki", type=Path, required=True)
    ap.add_argument("--mimic-root", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--min-cell", type=int, default=10)
    a = ap.parse_args(argv)
    D = load_private(a.out_dir)
    ev = D["events"][D["events"].split.eq("evaluation")].sample(frac=1, random_state=20261016).head(a.n)
    adm = read_hosp(a.mimic_root, "admissions", ["subject_id", "hadm_id", "dischtime"])
    dx = read_hosp(a.mimic_root, "diagnoses_icd", ["subject_id", "hadm_id", "icd_code", "icd_version"])
    hist = history_flags(D["cohort"], adm, dx).set_index("stay_id").to_dict("index")
    lab_g = {k: g for k, g in D["lab"][D["lab"].stay_id.isin(set(ev.stay_id))].groupby("stay_id")}
    ctx_g = {k: g for k, g in D["context"][D["context"].stay_id.isin(set(ev.stay_id))].groupby("stay_id")}
    cases = [{"payload": snapshot_payload(e.stay_id, e.t0, lab_g, ctx_g, hist.get(e.stay_id)), "hadm_id": int(e.hadm_id)} for e in ev.itertuples()]
    proxy = icd_proxy(ev, dx)
    if not (a.wiki / "index" / "chunks.jsonl").exists():
        build_index(a.wiki)
    retr = WikiRetriever(a.wiki / "index")
    tmp = Path(tempfile.mkdtemp())
    shutil.copytree(a.wiki / "wiki", tmp / "wiki")
    (tmp / "wiki" / "concepts" / "poison-ceftriaxone.md").write_text(POISON_PAGE, encoding="utf-8")
    build_index(tmp, tmp / "index", with_embeddings=retr.mode == "hybrid")
    pois = WikiRetriever(tmp / "index")
    R = run(cases, proxy, retriever=retr, poisoned_retriever=pois, repeats=a.repeats)
    R["meta"] = {"retrieval_mode": retr.mode, "wiki_sha256": retr.manifest["wiki_sha256"], "n_cases": len(cases), "min_cell_suppression": a.min_cell}
    R = _suppress_counts(R, a.min_cell)
    p = a.out_dir / "results_wiki_experiment.json"
    p.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
