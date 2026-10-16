"""External validation: apply the four frozen models to another hospital's data (MIMIC-IV layout), no retraining.

1 Agent 3   hourly while normal   24 h abnormality risk            agent3_gbm.joblib
2 Agent 2   first abnormal time   4-class prognosis (recovery x ICU death)  prognosis_gbm.joblib
3 Agent 1   first abnormal time   10 causes 0/1 (rule code) + supplementary clustering  rule_agent.py, 08c_cluster_models.joblib
4 Trajectory every hour           24 h quantile trajectory of ALT/AST/ALP/TBIL      trajectory_qgbm.joblib

Every patient of the external site is an evaluation patient. Output is aggregate only (small cells suppressed).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .develop import (PROGNOSIS, TRAJ_HORIZONS, TRAJ_QUANTILES, _json_safe, _noncrossing, _suppress_counts, agent3_labels,
                      agent3_landmarks, binary_metrics, build_features, curves, evaluate_flags, history_flags, load_private,
                      trajectory_rows)
from .rule_agent import VERSION as RULE_VERSION


def _X(bundle, X):
    missing = [c for c in bundle["features"] if c not in X]
    return X.reindex(columns=bundle["features"]).astype(float), missing


def validate(out_dir, model_dir, admissions=None, diagnoses=None, boot=200, min_cell=10, site="external", log=print):
    import joblib
    D = load_private(out_dir)
    cohort, lab, events, context, outcomes = D["cohort"], D["lab"], D["events"], D["context"], D["outcomes"]
    cohort["split"] = "evaluation"
    events["split"] = "evaluation"
    hist = history_flags(cohort, admissions, diagnoses)
    M = Path(model_dir)
    load = lambda n: joblib.load(M / n) if (M / n).exists() else None
    R = {"meta": {"site": site, "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "mode": "external_validation",
                  "patients": int(cohort.subject_id.nunique()), "icu_stays": int(len(cohort)), "events": int(len(events)),
                  "min_cell_suppression": min_cell, "rule_model": RULE_VERSION,
                  "models": {n: (load(n) or {}).get("trained_on") for n in
                             ["agent3_gbm.joblib", "prognosis_gbm.joblib", "trajectory_qgbm.joblib", "agent2_gbm.joblib"]}}}

    # 1 Agent 3: hourly 24 h abnormality risk while all three triggers are normal
    b = load("agent3_gbm.joblib")
    if b:
        log("external: agent 3")
        L = agent3_landmarks(cohort, lab, events, step_hours=1)
        L["label"] = agent3_labels(L, lab) if len(L) else []
        K = L[L.eligible & L.label.isin(["abnormal_observed", "no_abnormality_observed"])]
        if len(K) and K.label.nunique() == 2:
            X, miss = _X(b, build_features(K[["stay_id", "t"]], lab, context, cohort, hist))
            p = b["model"].predict_proba(X)[:, 1]
            y = K.label.eq("abnormal_observed").astype(int).to_numpy()
            R["agent3"] = {**binary_metrics(y, p, K.subject_id.to_numpy(), boot), "curves": curves(y, p),
                           "missing_features": miss, "landmarks_total": int(len(L)), "eligible": int(L.eligible.sum())}

    # 4 trajectory: every 4 h across the ICU stay, regardless of state
    tb = load("trajectory_qgbm.joblib")
    if tb:
        log("external: trajectory")
        Lt, targets = trajectory_rows(cohort, lab, step_hours=4)
        if len(Lt):
            Xt, miss = _X(tb, build_features(Lt[["stay_id", "t"]], lab, context, cohort, hist))
            perf = []
            for a, qs in tb["models"].items():
                last = Xt[f"{a}_latest"].to_numpy(float) if f"{a}_latest" in Xt else np.full(len(Xt), np.nan)
                for h in TRAJ_HORIZONS:
                    y = targets.get((a, h))
                    if y is None:
                        continue
                    keep = ~np.isnan(y) & ~np.isnan(last)
                    if keep.sum() < 3:
                        continue
                    XA = Xt[keep].assign(horizon_h=float(h))
                    P = _noncrossing({q: qs[q].predict(XA) for q in TRAJ_QUANTILES})
                    yy, lastv = y[keep], np.log2(np.clip(last[keep], 1e-3, None))
                    perf.append({"analyte": a, "horizon_h": h, "n": int(keep.sum()),
                                 "mae_model": float(np.mean(np.abs(P[0.5] - yy))),
                                 "mae_persistence": float(np.mean(np.abs(lastv - yy))),
                                 "coverage_80": float(np.mean((yy >= P[0.1]) & (yy <= P[0.9])))})
            R["trajectory"] = {"performance": perf, "landmarks": int(len(Lt)), "missing_features": miss}

    # 2 prognosis at the first abnormal time
    pb = load("prognosis_gbm.joblib")
    if pb and len(outcomes):
        log("external: prognosis")
        ev2 = events.merge(outcomes[["case_id", "observed_path"]], on="case_id", how="inner")
        ev2 = ev2[ev2.observed_path.isin(PROGNOSIS)]
        if len(ev2):
            X2, miss = _X(pb, build_features(ev2.rename(columns={"t0": "t"})[["stay_id", "t"]], lab, context, cohort, hist))
            P = pb["model"].predict_proba(X2)
            cls = list(pb["classes"])
            y = ev2.observed_path.to_numpy()
            res = {"n": int(len(ev2)), "per_class": [], "missing_features": miss,
                   "label_counts": ev2.observed_path.value_counts().to_dict()}
            for c in PROGNOSIS:
                yy = (y == c).astype(int)
                pp = P[:, cls.index(c)] if c in cls else np.zeros(len(yy))
                res["per_class"].append({"class": c, **binary_metrics(yy, pp, np.arange(len(yy)), boot)})
            pred = np.array(cls)[P.argmax(1)]
            res["accuracy"] = float(np.mean(pred == y))
            res["confusion"] = {f"{a}|{b}": int(((y == a) & (pred == b)).sum()) for a in PROGNOSIS for b in PROGNOSIS}
            R["prognosis"] = res

    # 3 causes: rule code (no training) vs discharge-ICD proxy; supplementary clustering transported
    if len(events):
        log("external: agent 1 flags")
        R["agent1_flags"] = evaluate_flags(events, lab, context, hist, diagnoses)
    cm = load("cluster_models.joblib") or load("08c_cluster_models.joblib")
    fp = Path(out_dir) / "08_features.csv"
    if cm and fp.exists():
        from .pattern import assign_clusters
        F = pd.read_csv(fp)
        cl = {}
        for k in ("liver_pattern_gmm", "clinical_phenotype_gmm"):
            m = cm.get(k)
            if m:
                lab_k = assign_clusters(m, F)
                cl[k] = {"n_clusters": m["k"], "stable_in_development": m["stable"],
                         "sizes": [int(s) for s in np.bincount(lab_k, minlength=m["k"])]}
                if "liver_pattern" in F:
                    t = pd.crosstab(pd.Series(lab_k, name="cluster"), F.liver_pattern)
                    cl[k]["crosstab_pattern"] = [{"cluster": int(a), "pattern": str(p), "n": int(v)} for a, r in t.iterrows() for p, v in r.items()]
        R["clustering_transport"] = cl
    fa = Path(out_dir) / "02c_value_format_audit.csv"
    if fa.exists():
        R["value_format_audit"] = pd.read_csv(fa).to_dict("records")
    return _suppress_counts(_json_safe(R), min_cell)
