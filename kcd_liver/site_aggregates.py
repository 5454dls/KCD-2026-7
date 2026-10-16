"""Cohort-level aggregates for the paper-style figures (notebook + GitHub Pages).

Input: the DataFrames the notebook already holds in memory. Output: a JSON-safe
dict with NO patient-level rows. For the credentialed full MIMIC-IV, call with
``min_cell=10`` so that small counts are suppressed before anything is written.

Every figure on the site and every PNG from ``kcd_liver.figures`` reads only
this dict, so the notebook and the web page always show the same numbers.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .evaluation import adoption_table, wilson

TRIGGERS = ["ALT", "ALP", "TBIL"]
RATIO_BINS = [2, 3, 5, 10, 20, 50, math.inf]
COUNT_KEYS = {"n", "k", "count", "n_patients_flagged", "unit_invalid", "missing_uln", "adopted",
              "errors", "abstained", "n_known", "n_all", "at_risk", "events"}


def _records(df: pd.DataFrame | None) -> list[dict] | None:
    if df is None or len(df) == 0:
        return None
    out = df.replace([np.inf, -np.inf], np.nan).astype(object).where(pd.notna(df), None)
    return [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()} for r in out.to_dict("records")]


def aalen_johansen(times: Iterable[float], events: Iterable[str], horizon: float = 48,
                   causes=("resolution", "death", "hospital_discharge"), grid_step: float = 1.0) -> dict:
    """Cumulative incidence per competing cause on a regular grid (same estimator as the notebook's competing_cif)."""
    d = pd.DataFrame({"time": list(times), "event": list(events)}).dropna(subset=["time"])
    grid = np.arange(0, horizon + grid_step / 2, grid_step)
    surv, cif = 1.0, {c: 0.0 for c in causes}
    km_res = 1.0  # naive: competing events treated as censoring -> 1 - KM for resolution
    event_times = sorted(d.loc[(d.time <= horizon) & d.event.isin(causes), "time"].unique())
    steps = []
    for t in event_times:
        risk = int((d.time >= t).sum())
        at = d[d.time.eq(t)]
        counts = {c: int(at.event.eq(c).sum()) for c in causes}
        total = sum(counts.values())
        if risk:
            for c, n in counts.items():
                cif[c] += surv * n / risk
            surv *= 1 - total / risk
            km_res *= 1 - counts["resolution"] / risk
        steps.append((t, dict(cif), 1 - km_res))
    curves = {c: [] for c in causes}
    naive = []
    j, cur, cur_naive = 0, {c: 0.0 for c in causes}, 0.0
    for g in grid:
        while j < len(steps) and steps[j][0] <= g:
            cur, cur_naive = steps[j][1], steps[j][2]
            j += 1
        for c in causes:
            curves[c].append(round(cur[c], 4))
        naive.append(round(cur_naive, 4))
    at_risk = [int((d.time >= g).sum()) for g in grid]
    return {"grid_hours": grid.tolist(), "cif": curves, "naive_resolution": naive,
            "at_risk": at_risk, "n": int(len(d))}


def auroc(y: Iterable[int], score: Iterable[float]) -> float | None:
    """Rank-based AUROC (Mann-Whitney), ties averaged. No sklearn dependency."""
    y = np.asarray(list(y), dtype=int)
    s = pd.Series(list(score), dtype=float)
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    if not pos or not neg:
        return None
    ranks = s.rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def average_precision(y, score) -> float | None:
    y = np.asarray(list(y), dtype=int)
    s = np.asarray(list(score), dtype=float)
    if y.sum() == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def calibration(y, risk, bins=(0, .05, .1, .2, .3, .5, 1.0001)) -> list[dict]:
    d = pd.DataFrame({"y": list(y), "p": list(risk)})
    d["bin"] = pd.cut(d.p, bins=list(bins), right=False)
    rows = []
    for b, g in d.groupby("bin", observed=True):
        k, n = int(g.y.sum()), int(len(g))
        lo, hi = wilson(k, n)
        rows.append({"bin": f"{b.left:.2f}-{min(b.right, 1):.2f}", "mean_predicted": float(g.p.mean()),
                     "observed": k / n if n else None, "ci_low": lo, "ci_high": hi, "n": n, "k": k})
    return rows


def suppress(obj, min_cell: int | None):
    """Replace counts 1..min_cell-1 with None (and mark) recursively."""
    if not min_cell:
        return obj
    if isinstance(obj, list):
        return [suppress(x, min_cell) for x in obj]
    if isinstance(obj, dict):
        out, flagged = {}, []
        for k, v in obj.items():
            if k in COUNT_KEYS and isinstance(v, list):
                out[k] = [None if isinstance(x, (int, float)) and 0 < x < min_cell else x for x in v]
            elif k in COUNT_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v < min_cell:
                out[k] = None
                flagged.append(k)
            else:
                out[k] = suppress(v, min_cell)
        if flagged:
            out["_suppressed"] = flagged
            for rate in ("rate", "observed", "adoption_rate", "ci_low", "ci_high"):
                if rate in out:
                    out[rate] = None
        return out
    return obj


def build_site_aggregates(*, source: str, flow: pd.DataFrame | None = None, qc: pd.DataFrame | None = None,
                          events: pd.DataFrame | None = None, features: pd.DataFrame | None = None,
                          cluster_cols: list[str] | None = None, outcomes: pd.DataFrame | None = None,
                          risk_reference: pd.DataFrame | None = None, cause_validation: pd.DataFrame | None = None,
                          paired_test: Mapping | None = None, grounding: pd.DataFrame | None = None,
                          stress_rows: list[Mapping] | None = None, injected_cause: str = "drug_related",
                          agent3_predictions: pd.DataFrame | None = None, threshold: pd.DataFrame | None = None,
                          alert_aggregation: pd.DataFrame | None = None, measurement: pd.DataFrame | None = None,
                          horizon: float = 48, min_cell: int | None = None) -> dict:
    agg: dict = {"source": source, "min_cell_suppression": min_cell, "horizon_hours": horizon}
    agg["cohort_flow"] = _records(flow)
    if qc is not None and len(qc):
        q = qc.reset_index() if "analyte" not in qc.columns else qc.copy()
        q = q[["analyte", "n", "unit_invalid", "missing_uln"]].copy()
        agg["lab_qc"] = _records(q.sort_values("n", ascending=False))
    if events is not None and len(events):
        agg["event_groups"] = _records(events.groupby(["split", "group"]).size().rename("n").reset_index())
        e = events.copy()
        e["bin"] = pd.cut(e.trigger_ratio.astype(float), bins=RATIO_BINS, right=False)
        hist = e.groupby(["trigger_analyte", "bin"], observed=False).size().rename("count").reset_index()
        hist["bin"] = hist["bin"].map(lambda b: f"{b.left:g}-{b.right:g}" if math.isfinite(b.right) else f"≥{b.left:g}")
        agg["trigger_ratio_hist"] = _records(hist.rename(columns={"trigger_analyte": "analyte"}))
    if features is not None and len(features) and "cluster" in features and (features.cluster >= 0).any():
        cols = [c for c in (cluster_cols or []) if c in features]
        dev = features[features.split.eq("development") & features.cluster.ge(0)]
        prof = dev.groupby("cluster")[cols].median()
        prof.insert(0, "n", dev.groupby("cluster").size())
        z = (prof[cols] - prof[cols].mean()) / prof[cols].std(ddof=0).replace(0, np.nan)
        agg["cluster_profile"] = {"features": cols, "rows": _records(prof.reset_index()),
                                  "z": _records(z.reset_index())}
    if outcomes is not None and len(outcomes):
        agg["outcome_paths"] = _records(outcomes.observed_path.value_counts().rename_axis("path").rename("n").reset_index())
        agg["first_event_48h"] = _records(outcomes.first_event_48h.value_counts().rename_axis("event").rename("n").reset_index())
    if risk_reference is not None and len(risk_reference) and "follow_hours" in risk_reference:
        agg["cif_development"] = aalen_johansen(risk_reference.follow_hours, risk_reference.first_event, horizon)
    agg["cause_validation"] = _records(cause_validation)
    agg["paired_test"] = dict(paired_test) if paired_test else None
    if grounding is not None and len(grounding):
        g = grounding.groupby("condition").agg(n=("case_id", "size"), support_items=("support_items", "sum"),
                                               verified=("verified", "sum")).reset_index()
        g["rate"] = g.verified / g.support_items.replace(0, np.nan)
        agg["grounding"] = _records(g)
    if stress_rows:
        agg["stress_adoption"] = adoption_table(stress_rows, injected_cause)
        agg["injected_cause"] = injected_cause
    if agent3_predictions is not None and len(agent3_predictions):
        known = agent3_predictions[agent3_predictions.truth.isin(["abnormal_observed", "no_abnormality_observed"])]
        y = known.truth.eq("abnormal_observed").astype(int)
        agg["agent3"] = {"n_all": int(len(agent3_predictions)), "n_known": int(len(known)),
                         "truth_counts": _records(agent3_predictions.truth.value_counts().rename_axis("truth").rename("n").reset_index()),
                         "auroc": auroc(y, known.risk) if len(known) else None,
                         "auprc": average_precision(y, known.risk) if len(known) else None,
                         "prevalence": float(y.mean()) if len(known) else None,
                         "brier": float(np.mean((known.risk - y) ** 2)) if len(known) else None,
                         "calibration": calibration(y, known.risk) if len(known) else None}
    agg["threshold_sensitivity"] = _records(threshold)
    agg["alert_aggregation"] = _records(alert_aggregation)
    agg["measurement_coverage"] = _records(measurement)
    return suppress(agg, min_cell)
