"""Model development and evaluation on the notebook's private outputs (full MIMIC-IV).

Inputs are the patient-level files the notebook already writes to its OUT folder
(01_cohort_private.csv, 03_labs_private.csv, 05_events_private.csv,
07_context_private.csv, 10_outcomes_EVALUATION_ONLY.csv) plus admissions and
diagnoses_icd from the MIMIC-IV hosp module. Everything here is vectorised
(merge_asof), so the hourly landmark design scales to the full cohort.

Output: ONE aggregate results dictionary (tables, metrics with 95% CI, curves)
that holds no patient-level rows. With ``min_cell`` set, counts below it are
suppressed before writing. Trained models are saved separately and must stay
local (they were fit on credentialed data).

Design (pre-specified, frozen before looking at evaluation patients):
- Split: the notebook's patient-level development/evaluation split. Subjects in
  ``exclude_subjects`` (the public demo patients) are removed from BOTH sets so
  that the public replay never shows in-sample predictions.
- Agent 3 (runs every hour): currently normal landmarks (ALT, ALP, TBIL all
  measured within 24 h and all < 2xULN, before the stay's first event) ->
  observed 2xULN abnormality within the next 24 h. Unknown follow-up is
  excluded from training and reported, never forced negative.
- Agent 2 (runs once at the first alert): first event within 48 h among
  resolution / death before resolution / discharge before resolution / none.
- Agent 1 (runs once at the first alert): rule baseline (and LLM outputs when
  present) scored against discharge-ICD proxy categories of the same admission.
  The proxy is an evaluation-only reference, never an agent input, and it is
  not the specialist reference standard.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import mcnemar_exact, wilson
from .rule_agent import exposures, VERSION as RULE_VERSION, rule_agent1
from .site_aggregates import aalen_johansen, auroc as _auroc, average_precision as _ap

TRIGGERS = ["ALT", "ALP", "TBIL"]
RATIO_FEATS = ["ALT", "AST", "ALP", "TBIL"]
VALUE_FEATS = ["INR", "ALBUMIN", "LACTATE", "CREAT", "BUN", "PLATELET", "WBC", "SODIUM", "GLUCOSE", "BICARB", "HEMOGLOBIN"]
VITALS = ["MAP", "HR", "RR", "SPO2", "TEMP_C"]
PRESSORS = "norepinephrine|epinephrine|vasopressin|phenylephrine|dopamine"
# Agent 3 target (team decision): any of ALT/ALP/TBIL >= 5x ULN within 24 h; eligibility stays all three < 2x
A3_TARGET = 5.0
THRESHOLDS = [2.0, 3.0, 4.0, 5.0]
HISTORY = ["history_liver", "history_viral_hepatitis", "history_heart_failure", "history_ckd", "history_diabetes", "history_cancer"]
A2_CLASSES = ["resolution", "death_before_resolution", "discharge_before_resolution", "no_event_observed"]
SEED = 20261016

# ICD proxy for Agent 1 evaluation (current admission discharge codes, evaluation only)
# Discharge-ICD proxy categories (NOT a reference standard; assigned after discharge, never an input)
ICD_PROXY = {
    "hypoxic_hepatitis": {9: r"^(570|7855[0-9]?)", 10: r"^(K720|K762|K763|R57)"},
    "congestive_hepatopathy": {9: r"^(5730|428)", 10: r"^(K761|I50)"},
    "sepsis_cholestasis": {9: r"^(038|99591|99592|78552)", 10: r"^(A40|A41|R652)"},
    "dili": {9: r"^(5733|E93[0-9]|E94[0-9])", 10: r"^(K71|T3[6-9]|T4[0-9]|T50)"},
    "pn_associated": None,  # no specific ICD code: proxy not available
    "biliary_obstruction_cholangitis": {9: r"^(5761|5762|574[3-9])", 10: r"^(K830|K831|K80[3-7])"},
    "cholecystitis": {9: r"^(5750|5751|574[01]|574[3-4]|574[6-8])", 10: r"^(K81|K80[0146])"},
    "hemolysis": {9: r"^(282|283)", 10: r"^(D5[5-9])"},
    "viral_hepatitis": {9: r"^070", 10: r"^(B1[5-9]|B251|B0081)"},
    "others": {9: r"^(571|72888)", 10: r"^(K70|K74|K75|K760|M6282)"},
}
HIST_RULES = {
    "history_liver": {9: r"^(570|571|572|573|070)", 10: r"^(B18|K70|K71|K72|K73|K74|K76)"},
    "history_viral_hepatitis": {9: r"^070", 10: r"^(B1[5-9])"},
    "history_heart_failure": {9: r"^428", 10: r"^(I50|I43|I110|I130|I132)"},
    "history_ckd": {9: r"^(585|586|V56)", 10: r"^(N18|N19|Z992|Z940)"},
    "history_diabetes": {9: r"^250", 10: r"^E1[0-4]"},
    "history_cancer": {9: r"^(1[4-9][0-9]|20[0-8])", 10: r"^C"},
}


# ------------------------------------------------------------------ loading
def load_private(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    rd = lambda n, **k: pd.read_csv(out_dir / n, low_memory=False, **k)
    cohort = rd("01_cohort_private.csv")
    for c in ["intime", "outtime", "admittime", "dischtime", "deathtime", "dod", "monitor_end", "obs_end"]:
        if c in cohort:
            cohort[c] = pd.to_datetime(cohort[c], errors="coerce")
    lab = rd("03_labs_private.csv", usecols=lambda c: c in {"stay_id", "analyte", "value", "ratio", "charttime", "available_at", "valid_value"})
    for c in ["charttime", "available_at"]:
        lab[c] = pd.to_datetime(lab[c], errors="coerce")
    lab = lab[lab.valid_value.astype(str).str.lower().eq("true")] if lab.valid_value.dtype == object else lab[lab.valid_value]
    events = rd("05_events_private.csv")
    events["t0"] = pd.to_datetime(events.t0)
    ctx_path = out_dir / "07_context_private.csv"
    context = rd("07_context_private.csv") if ctx_path.exists() and ctx_path.stat().st_size > 10 else pd.DataFrame(
        columns=["stay_id", "charttime", "available_at", "kind", "label", "value", "unit"])
    for c in ["charttime", "available_at"]:
        context[c] = pd.to_datetime(context[c], errors="coerce")
    outcomes = rd("10_outcomes_EVALUATION_ONLY.csv") if (out_dir / "10_outcomes_EVALUATION_ONLY.csv").exists() else pd.DataFrame()
    for c in ["resolution_time", "end_time"]:
        if c in outcomes:
            outcomes[c] = pd.to_datetime(outcomes[c], errors="coerce")
    for d in (lab, context, events, cohort, outcomes):
        for c in d.columns:
            if pd.api.types.is_datetime64_any_dtype(d[c]):
                d[c] = d[c].astype("datetime64[ns]")
    for d in (lab, context, events, cohort):
        if "stay_id" in d:
            d.dropna(subset=["stay_id"], inplace=True)
            d["stay_id"] = d["stay_id"].astype("int64")
    return {"cohort": cohort, "lab": lab, "events": events, "context": context, "outcomes": outcomes}


def history_flags(cohort: pd.DataFrame, admissions: pd.DataFrame | None, diagnoses: pd.DataFrame | None) -> pd.DataFrame:
    """Coded history from completed PREVIOUS admissions only (discharged before ICU intime)."""
    base = cohort[["stay_id", "subject_id", "intime"]].copy()
    if admissions is None or diagnoses is None or not len(diagnoses):
        for h in HISTORY:
            base[h] = np.nan
        return base.drop(columns=["subject_id", "intime"])
    adm = admissions[["subject_id", "hadm_id", "dischtime"]].copy()
    adm["dischtime"] = pd.to_datetime(adm.dischtime, errors="coerce")
    dx = diagnoses[["subject_id", "hadm_id", "icd_code", "icd_version"]].merge(adm, on=["subject_id", "hadm_id"], how="inner")
    dx = dx.merge(base, on="subject_id")
    dx = dx[dx.dischtime < dx.intime]
    code = dx.icd_code.astype(str).str.replace(".", "", regex=False)
    for h, rule in HIST_RULES.items():
        dx[h] = ((dx.icd_version == 9) & code.str.match(rule[9])) | ((dx.icd_version == 10) & code.str.match(rule[10]))
    flags = dx.groupby("stay_id")[list(HIST_RULES)].any().astype(float)
    known = set(dx.stay_id)
    out = base[["stay_id"]].merge(flags, on="stay_id", how="left")
    # no completed previous admission with codes -> unknown, not absent
    for h in HISTORY:
        out.loc[~out.stay_id.isin(known), h] = np.nan
    return out


# ------------------------------------------------------------------ features
def _asof(rows: pd.DataFrame, ev: pd.DataFrame, time_col: str, take: list[str], prefix: str, at: str = "t") -> pd.DataFrame:
    if ev.empty:
        for c in take:  # an absent analyte: keep time columns datetime so age arithmetic still works
            rows[prefix + c] = pd.NaT if c in ("charttime", "available_at") else np.nan
        return rows
    right = ev[["stay_id", time_col] + [c for c in take if c != time_col]].dropna(subset=[time_col]).sort_values(time_col)
    right = right.rename(columns={c: prefix + c for c in right.columns if c != "stay_id"})
    left = rows.copy()
    # merge_asof needs identical key dtypes (notebook frames may carry object ids or us-resolution times)
    for d, col in ((left, at), (right, prefix + time_col)):
        d["stay_id"] = pd.to_numeric(d["stay_id"], errors="coerce").astype("float64")
        d[col] = pd.to_datetime(d[col], errors="coerce").astype("datetime64[ns]")
    left, right = left.dropna(subset=["stay_id", at]).sort_values(at), right.dropna(subset=["stay_id"])
    m = pd.merge_asof(left, right, left_on=at, right_on=prefix + time_col, by="stay_id", direction="backward")
    m["stay_id"] = m["stay_id"].astype("int64")
    return m


def build_features(rows: pd.DataFrame, lab: pd.DataFrame, context: pd.DataFrame, cohort: pd.DataFrame,
                   hist: pd.DataFrame | None = None, time_col: str = "available_at") -> pd.DataFrame:
    """Point-in-time features for (stay_id, t) rows.

    ``time_col='available_at'`` is the correct design (result reported by t).
    ``time_col='charttime'`` deliberately leaks results drawn but not yet
    reported; it exists only for the leakage before/after experiment.
    """
    lab = lab[["stay_id", "analyte", "value", "ratio", "charttime", "available_at"]].copy()
    context = context.copy()
    for d in (lab, context):
        d["stay_id"] = pd.to_numeric(d["stay_id"], errors="coerce")
        d.dropna(subset=["stay_id"], inplace=True)
        d["stay_id"] = d["stay_id"].astype("int64")
        for c in ("charttime", "available_at"):
            if c in d:
                d[c] = pd.to_datetime(d[c], errors="coerce").astype("datetime64[ns]")
    X = rows[["stay_id", "t"]].copy()
    X["stay_id"] = X["stay_id"].astype("int64")
    X["_rid"] = np.arange(len(X))
    X["t"] = pd.to_datetime(X.t).astype("datetime64[ns]")
    for a in RATIO_FEATS + VALUE_FEATS:
        e = lab[lab.analyte.eq(a)].copy()
        e = e.sort_values(["stay_id", time_col])
        e["k"] = e.groupby("stay_id").cumcount() + 1
        col = "ratio" if a in RATIO_FEATS else "value"
        e["v"] = e[col]
        X = _asof(X, e, time_col, ["v", "charttime", "k"], f"{a}__")
        X[f"{a}__age_h"] = (X.t - X[f"{a}__charttime"]).dt.total_seconds() / 3600
        stale = X[f"{a}__age_h"] > 24
        latest = X[f"{a}__v"].where(~stale)
        X["_t24"] = X.t - pd.Timedelta(hours=24)
        prev = _asof(X[["_rid", "stay_id", "_t24"]].rename(columns={"_t24": "t"}), e, time_col, ["v", "k"], "p__")
        prev = prev.set_index("_rid").reindex(X._rid)
        X[f"{a}_latest"] = latest.values
        X[f"{a}_age_h"] = X[f"{a}__age_h"].where(~stale)
        X[f"{a}_n24"] = (X[f"{a}__k"].fillna(0) - prev["p__k"].fillna(0).values).clip(lower=0)
        X[f"{a}_delta24"] = (X[f"{a}__v"] - prev["p__v"].values).where(~stale)
        X = X.drop(columns=[c for c in X.columns if c.startswith(f"{a}__")] + ["_t24"])
    v = context[context.kind.eq("vital")].copy()
    v["v"] = pd.to_numeric(v.value, errors="coerce")
    for name in VITALS:
        e = v[v.label.eq(name)]
        X = _asof(X, e, time_col if time_col in e else "available_at", ["v", "charttime"], f"{name}__")
        age = (X.t - X[f"{name}__charttime"]).dt.total_seconds() / 3600
        X[f"{name}_latest"] = X[f"{name}__v"].where(age <= 6)
        X = X.drop(columns=[c for c in X.columns if c.startswith(f"{name}__")])
    p = context[context.kind.eq("infusion_record") & context.label.astype(str).str.contains(PRESSORS, case=False, na=False)]
    X = _asof(X, p.assign(seen=1.0), "available_at", ["seen", "charttime"], "pr__")
    X["pressor_6h"] = (((X.t - X["pr__charttime"]).dt.total_seconds() / 3600) <= 6).astype(float)
    X = X.drop(columns=[c for c in X.columns if c.startswith("pr__")])
    cc = ["stay_id", "age", "gender", "intime"] + (["admittime"] if "admittime" in cohort else [])
    c = cohort[cc].drop_duplicates("stay_id").copy()
    c["stay_id"] = c["stay_id"].astype("int64")
    c["intime"] = pd.to_datetime(c["intime"]).astype("datetime64[ns]")
    # pre-ICU ward measurements of the same admission: last and highest x ULN before ICU admission (constant per stay)
    pre = lab[lab.analyte.isin(RATIO_FEATS)].merge(c[["stay_id", "intime"]], on="stay_id")
    pre = pre[(pre.charttime < pre.intime) & (pre.available_at <= pre.intime) & pre.ratio.notna()].sort_values("charttime")
    for a in RATIO_FEATS:
        g = pre[pre.analyte.eq(a)].groupby("stay_id").ratio
        c[f"{a}_preicu_last"] = c.stay_id.map(g.last())
        c[f"{a}_preicu_max"] = c.stay_id.map(g.max())
    if "admittime" in c:
        c["hours_admit_to_icu"] = (c.intime - pd.to_datetime(c.admittime)).dt.total_seconds() / 3600
        c = c.drop(columns=["admittime"])
    X = X.merge(c, on="stay_id", how="left")
    X["female"] = (X.gender == "F").astype(float)
    X["hours_since_icu"] = (X.t - X.intime).dt.total_seconds() / 3600
    X = X.drop(columns=["gender", "intime"])
    if hist is not None:
        X = X.merge(hist, on="stay_id", how="left")
    X = X.sort_values("_rid").drop(columns="_rid").reset_index(drop=True)
    return X


def feature_columns(X: pd.DataFrame, observed_in=None) -> list[str]:
    """Model inputs. With observed_in (development mask), drop columns never observed there:
    HistGradientBoosting cannot bin a feature with zero observed values."""
    cols = [c for c in X.columns if c not in {"stay_id", "t", "subject_id", "split", "y", "label", "case_id"}]
    if observed_in is not None:
        m = np.asarray(observed_in, bool)
        cols = [c for c in cols if pd.to_numeric(X.loc[m, c], errors="coerce").notna().any()]
    return cols


# ------------------------------------------------------------------ labels
def agent3_landmarks(cohort, lab, events, step_hours=1, time_col="available_at"):
    """Hourly landmarks inside the ICU monitoring window, before the first event."""
    c = cohort[["stay_id", "subject_id", "split", "intime", "monitor_end", "obs_end"]].dropna(subset=["intime", "monitor_end"])
    first = events.groupby("stay_id").t0.min()
    rows = []
    for r in c.itertuples():
        end = min(r.monitor_end, first.get(r.stay_id, r.monitor_end))
        n = int((end - r.intime).total_seconds() // 3600 // step_hours)
        if n < 1:
            continue
        ts = r.intime + pd.to_timedelta(np.arange(1, n + 1) * step_hours, unit="h")
        rows.append(pd.DataFrame({"stay_id": r.stay_id, "subject_id": r.subject_id, "split": r.split, "t": ts, "obs_end": r.obs_end}))
    L = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["stay_id", "subject_id", "split", "t", "obs_end"])
    L["t"] = pd.to_datetime(L.t).astype("datetime64[ns]")
    # eligibility: all three triggers fresh (<=24 h since draw) and below threshold
    E = L[["stay_id", "t"]].copy()
    E["_i"] = np.arange(len(E))
    for a in TRIGGERS:
        e = lab[lab.analyte.eq(a)]
        E = _asof(E, e, time_col, ["ratio", "charttime"], f"{a}_")
        E[f"{a}_fresh"] = ((E.t - E[f"{a}_charttime"]).dt.total_seconds() / 3600 <= 24) & E[f"{a}_ratio"].notna()
    E = E.sort_values("_i")
    fresh = np.logical_and.reduce([E[f"{a}_fresh"].values for a in TRIGGERS])
    below = np.logical_and.reduce([(E[f"{a}_ratio"] < 2).fillna(False).values for a in TRIGGERS])
    L["measured_all3"] = fresh
    L["eligible"] = fresh & below
    any_fresh = np.logical_or.reduce([E[f"{a}_fresh"].values for a in TRIGGERS])
    any_high = np.logical_or.reduce([(E[f"{a}_ratio"] >= 2).fillna(False).values & E[f"{a}_fresh"].values for a in TRIGGERS])
    L["naive_normal"] = ~any_high  # "no abnormal value shown" ignores whether anything was measured
    L["unmeasured_any"] = ~fresh
    L["no_trigger_at_all"] = ~any_fresh
    return L


def agent3_labels(L: pd.DataFrame, lab: pd.DataFrame, horizon=24, end_window=6, threshold=None) -> pd.Series:
    """abnormal_observed / no_abnormality_observed / terminal / insufficient_measurement (notebook semantics).

    threshold: x ULN that counts as the predicted abnormality (any of ALT/ALP/TBIL). Default A3_TARGET (team: 5x).
    """
    threshold = A3_TARGET if threshold is None else threshold
    t = pd.to_datetime(L.t)
    h = t + pd.Timedelta(hours=horizon)
    obs_end = pd.to_datetime(L.obs_end)
    ab = lab[lab.analyte.isin(TRIGGERS) & (lab.ratio >= threshold)][["stay_id", "available_at"]].dropna().sort_values("available_at")
    Q = pd.DataFrame({"stay_id": L.stay_id.values, "t": t.values, "_i": np.arange(len(L))}).sort_values("t")
    m = pd.merge_asof(Q, ab.rename(columns={"available_at": "next_ab"}), left_on="t", right_on="next_ab", by="stay_id",
                      direction="forward", allow_exact_matches=False).sort_values("_i")
    next_ab = pd.to_datetime(m.next_ab.values)
    lim = np.minimum(h.values, obs_end.values)
    abnormal = (next_ab <= lim) & pd.notna(next_ab)
    terminal = obs_end.values <= h.values
    measured_end = np.ones(len(L), dtype=bool)
    for a in TRIGGERS:
        e = lab[lab.analyte.eq(a)][["stay_id", "available_at", "charttime"]].dropna().sort_values("available_at")
        Qa = pd.DataFrame({"stay_id": L.stay_id.values, "h": h.values, "_i": np.arange(len(L))}).sort_values("h")
        ma = pd.merge_asof(Qa, e, left_on="h", right_on="available_at", by="stay_id", direction="backward").sort_values("_i")
        ok = (pd.to_datetime(ma.available_at.values) > t.values) & (pd.to_datetime(ma.charttime.values) >= (h - pd.Timedelta(hours=end_window)).values)
        measured_end &= np.asarray(ok, dtype=bool)
    lab_ = np.where(abnormal, "abnormal_observed", np.where(terminal, "death_or_icu_discharge_before_observation_end",
                    np.where(measured_end, "no_abnormality_observed", "insufficient_measurement")))
    return pd.Series(lab_, index=L.index)


def agent2_follow(events: pd.DataFrame, outcomes: pd.DataFrame, horizon=48) -> pd.DataFrame:
    o = events[["case_id", "t0"]].merge(outcomes, on="case_id", how="inner")
    rt = (o.resolution_time - o.t0).dt.total_seconds() / 3600
    et = (o.end_time - o.t0).dt.total_seconds() / 3600
    ev = np.where(rt.notna() & (rt <= horizon) & (o.resolution_time < o.end_time), "resolution",
                  np.where(et <= horizon, np.where(o.terminal.eq("death"), "death", "icu_discharge"), "censor"))
    tm = np.where(ev == "resolution", rt, np.where(ev == "censor", horizon, et)).astype(float)
    return pd.DataFrame({"case_id": o.case_id, "follow_hours": np.clip(tm, 0, horizon), "first_event": ev})


# ------------------------------------------------------------------ models
def preprocessing_report(X: pd.DataFrame, cols, dev_mask, lr_pipeline=None) -> dict:
    """Per-variable missingness (development vs evaluation) and the imputation/scaling learned on development only."""
    dev_mask = np.asarray(dev_mask, bool)
    A = X[cols].apply(pd.to_numeric, errors="coerce")
    rows = []
    imp = sc = None
    if lr_pipeline is not None:
        steps = dict(lr_pipeline.named_steps)
        imp, sc = steps.get("simpleimputer"), steps.get("standardscaler")
    for i, c in enumerate(cols):
        d, e = A.loc[dev_mask, c], A.loc[~dev_mask, c]
        rows.append({"variable": c, "group": group_of(c), "missing_dev": float(d.isna().mean()), "missing_eval": float(e.isna().mean()),
                     "median_dev": None if d.dropna().empty else float(d.median()),
                     "q1_dev": None if d.dropna().empty else float(d.quantile(.25)), "q3_dev": None if d.dropna().empty else float(d.quantile(.75)),
                     "impute_value": float(imp.statistics_[i]) if imp is not None and i < len(imp.statistics_) else None,
                     "scale_mean": float(sc.mean_[i]) if sc is not None and i < len(sc.mean_) else None,
                     "scale_sd": float(sc.scale_[i]) if sc is not None and i < len(sc.scale_) else None})
    return {"rules": ["모든 값은 그 시각까지 보고된 결과만 사용 (max(charttime, storetime))",
                      "단위 통일(µmol/L→mg/dL 등), 생리적 범위 밖 값 제거, 간 수치는 행별 참고범위 상한으로 나눈 ULN 배수",
                      "최근값은 검사 24시간·활력징후 6시간 안의 값만 사용, 그보다 오래되면 결측",
                      "그래디언트 부스팅: 결측을 그대로 두고 분기에서 처리, 정규화 불필요",
                      "로지스틱 회귀: 개발 집단 중앙값으로 대체 + 변수별 결측 지시자, 개발 집단 평균·표준편차로 표준화 (평가 집단은 같은 값 적용)",
                      "평가 집단 정보는 어떤 전처리 값에도 쓰지 않음"],
            "variables": rows}


def threshold_table(cohort: pd.DataFrame, lab: pd.DataFrame, thresholds=None) -> list[dict]:
    """First event at 2/3/4/5x ULN (any of ALT/ALP/TBIL) during ICU monitoring, with recovery (all three < 2x,
    measured within 24 h, before ICU discharge) and death within 1 year (MIMIC-IV deathtime or dod)."""
    thresholds = THRESHOLDS if thresholds is None else thresholds
    c = cohort.set_index("stay_id")
    trig = lab[lab.analyte.isin(TRIGGERS) & lab.ratio.notna()].sort_values("charttime")
    by = {k: g for k, g in trig.groupby("stay_id")}
    out = []
    for k in thresholds:
        n = rec = died = unass = icu_death = 0
        for s_, g in by.items():
            if s_ not in c.index:
                continue
            r = c.loc[s_]
            hit = g[(g.ratio >= k) & (g.available_at >= r.intime) & (g.available_at <= r.monitor_end)]
            if hit.empty:
                continue
            t0 = hit.available_at.min()
            n += 1
            after = g[(g.charttime > t0) & (g.available_at <= r.monitor_end)]
            unass += after.empty  # no ALT/ALP/TBIL after the event before ICU discharge
            icu_death += bool(r.get("icu_death")) if "icu_death" in c else 0
            last = {}
            for row in after.itertuples():
                last[row.analyte] = (row.charttime, row.ratio)
                if len(last) == len(TRIGGERS) and all(row.charttime - tt <= pd.Timedelta(hours=24) and v < 2 for tt, v in last.values()):
                    rec += 1
                    break
            year = t0 + pd.Timedelta(days=365)
            dt, dod = r.get("deathtime"), r.get("dod")
            died += bool((pd.notna(dt) and pd.Timestamp(dt) <= year) or (pd.notna(dod) and pd.Timestamp(dod) <= year.normalize()))
        out.append({"threshold_x_uln": k, "patients_with_event": n, "recovered_before_icu_discharge": rec,
                    "unrecovered_before_icu_discharge": n - rec - unass, "recovery_unassessable": unass,
                    "icu_death": icu_death, "alive_icu_discharge": n - icu_death, "died_within_1y": died,
                    "recovered_rate": rec / n if n else None, "died_1y_rate": died / n if n else None})
    return out


def _models(kind="binary"):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    lr = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), StandardScaler(),
                       LogisticRegression(max_iter=3000, C=0.5))
    gbm = lambda lr_, leaves: HistGradientBoostingClassifier(learning_rate=lr_, max_leaf_nodes=leaves, max_iter=400,
                                                            early_stopping="auto", validation_fraction=0.15, l2_regularization=1.0,
                                                            random_state=SEED)
    return lr, gbm


def _grouped_cv_choice(X, y, groups, gbm, grid, metric, folds=3):
    from sklearn.model_selection import GroupKFold
    res = []
    if len(np.unique(groups)) < folds or len(np.unique(y)) < 2:
        return grid[0], [{"learning_rate": g[0], "max_leaf_nodes": g[1], "cv_score": None} for g in grid]
    for g in grid:
        scores = []
        for tr, va in GroupKFold(n_splits=folds).split(X, y, groups):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                continue
            m = gbm(*g).fit(X.iloc[tr], y[tr])
            scores.append(metric(y[va], m.predict_proba(X.iloc[va])[:, 1]))
        res.append({"learning_rate": g[0], "max_leaf_nodes": g[1], "cv_score": float(np.mean(scores)) if scores else None})
    best = max(res, key=lambda r: -1 if r["cv_score"] is None else r["cv_score"])
    return (best["learning_rate"], best["max_leaf_nodes"]), res


def _calib(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, int)
    if len(np.unique(y)) < 2:
        return None, None
    from sklearn.linear_model import LogisticRegression
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    slope = LogisticRegression(C=1e6, max_iter=2000).fit(lp, y).coef_[0][0]
    # calibration-in-the-large: intercept with slope fixed at 1 (one Newton loop)
    a = 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(lp[:, 0] + a)))
        g, hss = (y - q).sum(), (q * (1 - q)).sum()
        if hss <= 0:
            break
        a += g / hss
    return float(a), float(slope)


def binary_metrics(y, p, groups, boot=200, seed=SEED):
    y, p, groups = np.asarray(y, int), np.asarray(p, float), np.asarray(groups)
    out = {"n": int(len(y)), "events": int(y.sum()), "patients": int(len(np.unique(groups))),
           "prevalence": float(y.mean()) if len(y) else None,
           "auroc": _auroc(y, p), "auprc": _ap(y, p), "brier": float(np.mean((p - y) ** 2)) if len(y) else None}
    out["cal_intercept"], out["cal_slope"] = _calib(y, p)
    # patient-cluster bootstrap: hourly rows of one patient are not independent
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by = pd.Series(np.arange(len(y))).groupby(groups).apply(lambda s: s.values).to_dict()
    stats = {"auroc": [], "auprc": [], "brier": []}
    for _ in range(boot if len(ug) > 5 else 0):
        pick = np.concatenate([idx_by[g] for g in rng.choice(ug, len(ug), replace=True)])
        yy, pp = y[pick], p[pick]
        if len(np.unique(yy)) < 2:
            continue
        stats["auroc"].append(_auroc(yy, pp))
        stats["auprc"].append(_ap(yy, pp))
        stats["brier"].append(float(np.mean((pp - yy) ** 2)))
    for k, v in stats.items():
        v = [x for x in v if x is not None]
        out[k + "_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) >= 20 else None
    # top-decile alert burden
    if len(y) >= 10:
        cut = np.quantile(p, .9)
        top = p >= cut
        out["top10_ppv"] = float(y[top].mean()) if top.any() else None
        out["top10_sensitivity"] = float(y[top].sum() / max(1, y.sum()))
    return out


def curves(y, p, n=60):
    y, p = np.asarray(y, int), np.asarray(p, float)
    if len(np.unique(y)) < 2:
        return None
    order = np.argsort(-p)
    ys = y[order]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    P, N = ys.sum(), len(ys) - ys.sum()
    pick = np.unique(np.linspace(0, len(ys) - 1, min(n, len(ys))).astype(int))
    roc = [{"fpr": 0.0, "tpr": 0.0}] + [{"fpr": float(fp[i] / N), "tpr": float(tp[i] / P)} for i in pick]
    pr = [{"recall": float(tp[i] / P), "precision": float(tp[i] / (i + 1))} for i in pick]
    q = pd.qcut(pd.Series(p), q=min(10, max(2, len(np.unique(p)))), duplicates="drop")
    cal = []
    for b, g in pd.DataFrame({"y": y, "p": p, "b": q}).groupby("b", observed=True):
        k, m = int(g.y.sum()), int(len(g))
        lo, hi = wilson(k, m)
        cal.append({"mean_predicted": float(g.p.mean()), "observed": k / m, "ci_low": lo, "ci_high": hi, "n": m, "k": k})
    return {"roc": roc, "pr": pr, "calibration": cal}


# ------------------------------------------------------------------ agent 1 evaluation
def icd_proxy(events, diagnoses):
    if diagnoses is None or not len(diagnoses):
        return {}
    dx = diagnoses[diagnoses.hadm_id.isin(set(events.hadm_id))].copy()
    code = dx.icd_code.astype(str).str.replace(".", "", regex=False)
    cats = {}
    for cat, rule in ICD_PROXY.items():
        if rule is None:
            continue
        hit = ((dx.icd_version == 9) & code.str.match(rule[9])) | ((dx.icd_version == 10) & code.str.match(rule[10]))
        for h in dx.loc[hit, "hadm_id"].unique():
            cats.setdefault(int(h), set()).add(cat)
    return cats


def snapshot_payload(stay_id, t0, lab_g, ctx_g, hist_row, lookback=48):
    t0 = pd.Timestamp(t0)
    start = t0 - pd.Timedelta(hours=lookback)
    x = lab_g.get(stay_id)
    labs = []
    if x is not None:
        x = x[(x.available_at <= t0) & (x.charttime <= t0) & (x.charttime >= start)].sort_values(["charttime", "available_at"])
        for r in x.drop_duplicates("analyte", keep="last").itertuples():
            labs.append({"analyte": r.analyte, "value": float(r.value), "ratio": None if pd.isna(r.ratio) else float(r.ratio),
                         "measured_at": r.charttime.isoformat(), "age_hours": (t0 - r.charttime).total_seconds() / 3600})
    ctx = []
    y = ctx_g.get(stay_id)
    if y is not None:
        y = y[(y.available_at <= t0) & (y.charttime <= t0) & (y.charttime >= start)].sort_values("available_at").tail(150)
        ctx = [{"kind": r.kind, "label": r.label, "value": str(r.value), "measured_at": r.charttime.isoformat()} for r in y.itertuples()]
    hist = {k: bool(v) for k, v in (hist_row or {}).items() if k in HISTORY and pd.notna(v)}
    out = {"as_of": t0.isoformat(), "labs": labs, "context": ctx, "historical_codes": hist,
           "exposures": exposures(admission_context(ctx_g.get(stay_id), t0), t0)}
    hb = x[x.analyte.eq("HEMOGLOBIN")] if x is not None else None
    if hb is not None and len(hb) >= 2:
        out["hemoglobin_drop_48h"] = float(hb.value.max() - hb.sort_values("charttime").value.iloc[-1])
    return out


def admission_context(y, t0):
    """All context rows of the stay available by t0 (for PN days and serology, beyond the 48 h window)."""
    if y is None or not len(y):
        return []
    y = y[y.available_at <= t0]
    y = y[y.kind.isin(["infusion_record", "emar_record", "serology_record"])]
    return [{"kind": r.kind, "label": r.label, "value": r.value, "charttime": r.charttime, "available_at": r.available_at}
            for r in y.itertuples()]


def evaluate_agent1(events, lab, context, hist, diagnoses, llm_rows=None):
    proxy = icd_proxy(events, diagnoses)
    ev = events[events.split.eq("evaluation")]
    lab_g = {k: g for k, g in lab[lab.stay_id.isin(set(ev.stay_id))].groupby("stay_id")}
    ctx_g = {k: g for k, g in context[context.stay_id.isin(set(ev.stay_id))].groupby("stay_id")}
    hist_i = hist.set_index("stay_id").to_dict("index") if hist is not None else {}
    rows = []
    for e in ev.itertuples():
        ans = rule_agent1(snapshot_payload(e.stay_id, e.t0, lab_g, ctx_g, hist_i.get(e.stay_id)))
        codes = [c["cause_code"] for c in ans["candidates"]]
        ref = proxy.get(int(e.hadm_id), set())
        rows.append({"case_id": e.case_id, "group": e.group, "has_proxy": bool(ref), "abstain": ans["abstain"],
                     "rule_top1": bool(codes[:1] and codes[0] in ref), "rule_top3": bool(set(codes[:3]) & ref)})
    R = pd.DataFrame(rows)
    out = {"n_eval_events": int(len(R)), "n_with_proxy": int(R.has_proxy.sum()) if len(R) else 0, "models": []}
    if not len(R):
        return out
    K = R[R.has_proxy]

    def summ(name, t1, t3, ab):
        r = {"model": name, "n": int(len(t1))}
        for k, s in (("top1", t1), ("top3", t3)):
            kk = int(np.sum(s))
            lo, hi = wilson(kk, len(s))
            r[k] = {"k": kk, "n": int(len(s)), "rate": kk / len(s) if len(s) else None, "ci": [lo, hi]}
        r["abstain_rate"] = float(np.mean(ab)) if len(ab) else None
        return r
    out["models"].append(summ(RULE_VERSION, K.rule_top1.values, K.rule_top3.values, K.abstain.values))
    if llm_rows:
        L = {r["case_id"]: r for r in llm_rows if r.get("condition") == "base" and r.get("backend", "llm") == "llm"}
        both = K[K.case_id.isin(L)]
        if len(both):
            ref_of = {e.case_id: proxy.get(int(e.hadm_id), set()) for e in ev.itertuples()}
            t1, t3, ab = [], [], []
            for cid in both.case_id:
                a = L[cid].get("answer", {})
                codes = [c.get("cause_code") for c in a.get("candidates", [])]
                t1.append(bool(codes[:1] and codes[0] in ref_of[cid]))
                t3.append(bool(set(codes[:3]) & ref_of[cid]))
                ab.append(bool(a.get("abstain")))
            out["models"].append(summ("LLM (" + str(next(iter(L.values())).get("model", "")) + ")", np.array(t1), np.array(t3), np.array(ab)))
            out["paired_top1_rule_vs_llm"] = mcnemar_exact(list(both.rule_top1.values), t1)
    out["proxy_distribution"] = [{"category": c, "n": int(sum(c in s for s in proxy.values()))} for c in ICD_PROXY]
    return out


# ------------------------------------------------------------------ table 1
def _med(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return None if s.empty else [float(s.median()), float(s.quantile(.25)), float(s.quantile(.75))]


def _smd(a, b, binary=False):
    a, b = pd.to_numeric(a, errors="coerce").dropna(), pd.to_numeric(b, errors="coerce").dropna()
    if a.empty or b.empty:
        return None
    if binary:
        p1, p2 = a.mean(), b.mean()
        d = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    else:
        p1, p2 = a.mean(), b.mean()
        d = math.sqrt((a.var() + b.var()) / 2)
    return None if not d else float(abs(p1 - p2) / d)


def table1(cohort, events, hist):
    c = cohort.copy()
    c["event"] = c.stay_id.isin(set(events.stay_id))
    c["icu_los_days"] = (c.outtime - c.intime).dt.total_seconds() / 86400
    c["hospital_death"] = pd.to_numeric(c.get("hospital_expire_flag"), errors="coerce")
    c["female"] = (c.gender == "F").astype(float)
    if hist is not None:
        c = c.merge(hist[["stay_id", "history_liver", "history_heart_failure"]], on="stay_id", how="left")
    groups = {"overall": c, "event": c[c.event], "no_event": c[~c.event], "development": c[c.split.eq("development")],
              "evaluation": c[c.split.eq("evaluation")]}
    rows = [{"variable": "n", "kind": "count", **{g: int(len(d)) for g, d in groups.items()}}]
    for var, label in [("age", "나이 (세)"), ("icu_los_days", "ICU 재원 (일)")]:
        rows.append({"variable": label, "kind": "median_iqr", **{g: _med(d[var]) for g, d in groups.items()},
                     "smd_event": _smd(groups["event"][var], groups["no_event"][var]),
                     "smd_split": _smd(groups["development"][var], groups["evaluation"][var])})
    for var, label in [("female", "여성"), ("hospital_death", "병원 내 사망"), ("history_liver", "이전 입원 간질환 코드"),
                       ("history_heart_failure", "이전 입원 심부전 코드")]:
        if var not in c:
            continue
        rows.append({"variable": label, "kind": "n_pct",
                     **{g: [int(pd.to_numeric(d[var], errors="coerce").fillna(0).sum()), float(pd.to_numeric(d[var], errors="coerce").fillna(0).mean()) if len(d) else None] for g, d in groups.items()},
                     "smd_event": _smd(groups["event"][var].fillna(0), groups["no_event"][var].fillna(0), True),
                     "smd_split": _smd(groups["development"][var].fillna(0), groups["evaluation"][var].fillna(0), True)})
    for col, label in [("first_careunit", "첫 ICU 유형"), ("admission_type", "입원 유형")]:
        if col not in c or c[col].isna().all():
            continue
        for lev in c[col].value_counts().head(5).index:
            ind = (c[col] == lev).astype(float)
            rows.append({"variable": f"{label}: {lev}", "kind": "n_pct",
                         **{g: [int(ind[d.index].sum()), float(ind[d.index].mean()) if len(d) else None] for g, d in groups.items()},
                         "smd_event": _smd(ind[groups["event"].index], ind[groups["no_event"].index], True),
                         "smd_split": _smd(ind[groups["development"].index], ind[groups["evaluation"].index], True)})
    ev = events.copy()
    for gname, lab_ in [("initial_abnormal", "사건 집단: 입실 초기 이상"), ("incident", "사건 집단: 관찰 중 신규"), ("onset_unknown", "사건 집단: 시점 불명")]:
        n_ev = int(ev.group.eq(gname).sum())
        rows.append({"variable": lab_, "kind": "n_pct", "overall": [n_ev, n_ev / max(1, len(ev))],
                     "event": [n_ev, n_ev / max(1, len(ev))],
                     "development": [int((ev.group.eq(gname) & ev.split.eq("development")).sum()), None],
                     "evaluation": [int((ev.group.eq(gname) & ev.split.eq("evaluation")).sum()), None]})
    return rows


# ------------------------------------------------------------------ main entry
def develop(out_dir, admissions=None, diagnoses=None, exclude_subjects=None, source="mimic-iv", min_cell=10,
            step_hours=1, train_step_hours=None, boot=200, model_dir=None, max_train_rows=600_000, log=print):
    from sklearn.base import clone
    D = load_private(out_dir)
    cohort, lab, events, context, outcomes = D["cohort"], D["lab"], D["events"], D["context"], D["outcomes"]
    excluded = set(exclude_subjects or [])
    n_excl = int(cohort.subject_id.isin(excluded).sum())
    cohort = cohort[~cohort.subject_id.isin(excluded)].copy()
    keep = set(cohort.stay_id)
    lab, context = lab[lab.stay_id.isin(keep)], context[context.stay_id.isin(keep)]
    events = events[events.stay_id.isin(keep)].copy()
    hist = history_flags(cohort, admissions, diagnoses)
    R = {"meta": {"source": source, "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "min_cell_suppression": min_cell, "excluded_demo_subjects_in_cohort": n_excl, "seed": SEED,
                  "landmark_step_hours": step_hours, "rule_model": RULE_VERSION}}
    R["table1"] = table1(cohort, events, hist)
    flow_path = Path(out_dir) / "04_cohort_flow.csv"
    R["flow"] = pd.read_csv(flow_path).to_dict("records") if flow_path.exists() else None

    # ---------------- Agent 3: hourly landmarks
    log("agent3: building landmarks")
    L = agent3_landmarks(cohort, lab, events, step_hours=step_hours)
    L["label"] = agent3_labels(L, lab) if len(L) else []
    elig = L[L.eligible].copy()
    known = elig[elig.label.isin(["abnormal_observed", "no_abnormality_observed"])].copy()
    known["y"] = known.label.eq("abnormal_observed").astype(int)
    dev3 = {"landmarks_total": int(len(L)), "patients_total": int(L.subject_id.nunique()) if len(L) else 0,
            "measured_all3": int(L.measured_all3.sum()), "eligible": int(len(elig)),
            "label_counts": elig.label.value_counts().to_dict(),
            "known": int(len(known)), "known_patients": int(known.subject_id.nunique()),
            "train_rows": int(known.split.eq("development").sum()), "test_rows": int(known.split.eq("evaluation").sum()),
            "train_patients": int(known[known.split.eq("development")].subject_id.nunique()),
            "test_patients": int(known[known.split.eq("evaluation")].subject_id.nunique())}
    R["agent3_development"] = dev3
    # shadow: missing-as-normal
    if len(L):
        nn = L[L.naive_normal]
        R["shadow_missing_as_normal"] = {
            "naive_normal_hours": int(len(nn)), "truly_eligible_hours": int(nn.eligible.sum()),
            "unmeasured_share_of_naive_normal": float(nn.unmeasured_any.mean()) if len(nn) else None,
            "no_trigger_measured_share": float(nn.no_trigger_at_all.mean()) if len(nn) else None,
            "patients": int(L.subject_id.nunique())}
    perf3 = {"models": [], "curves": {}}
    best = None
    if len(known) and known.y.nunique() == 2 and known.split.eq("evaluation").any():
        log(f"agent3: features for {len(known):,} known landmarks")
        X = build_features(known[["stay_id", "t"]], lab, context, cohort, hist)
        cols = feature_columns(X, known.split.eq("development").values)
        Xa = X[cols].astype(float)
        tr = known.split.eq("development").values
        te = ~tr
        if tr.sum() > max_train_rows:  # cap rows, sampled by patient-stratified random rows
            pick = np.random.default_rng(SEED).choice(np.where(tr)[0], max_train_rows, replace=False)
            tr = np.zeros(len(known), bool)
            tr[pick] = True
        y = known.y.values
        g = known.subject_id.values
        lr, gbm = _models()
        grid = [(0.05, 15), (0.05, 31), (0.1, 15), (0.1, 31)]
        best, cvres = _grouped_cv_choice(Xa[tr].reset_index(drop=True), y[tr], g[tr], gbm, grid, lambda a, b: _ap(a, b) or 0)
        dev3["gbm_grid"] = cvres
        dev3["gbm_selected"] = {"learning_rate": best[0], "max_leaf_nodes": best[1]}
        dev3["n_features"] = len(cols)
        dev3["features"] = cols
        fitted = {}
        for name, m in [("로지스틱 회귀", clone(lr)), ("그래디언트 부스팅", gbm(*best))]:
            m.fit(Xa[tr], y[tr])
            p = m.predict_proba(Xa[te])[:, 1]
            perf3["models"].append({"model": name, **binary_metrics(y[te], p, g[te], boot)})
            perf3["curves"][name] = curves(y[te], p)
            fitted[name] = m
        R["preprocessing"] = preprocessing_report(X, cols, tr, fitted.get("로지스틱 회귀"))
        prev = y[tr].mean()
        perf3["models"].append({"model": "기준선 (개발 집단 발생률)", **binary_metrics(y[te], np.full(te.sum(), prev), g[te], 0)})
        # one predeclared landmark per evaluation patient (first eligible after 24 h, else first)
        Ke = known[te].copy()
        Ke["p"] = fitted["그래디언트 부스팅"].predict_proba(Xa[te])[:, 1]
        Ke["h"] = (Ke.t - Ke.merge(cohort[["stay_id", "intime"]], on="stay_id", how="left").intime.values).dt.total_seconds() / 3600
        Ke["after24"] = Ke.h >= 24
        one = Ke.sort_values(["subject_id", "after24", "t"], ascending=[True, False, True]).drop_duplicates("subject_id")
        if len(one) and one.y.nunique() == 2:
            perf3["one_landmark_per_patient_gbm"] = binary_metrics(one.y.values, one.p.values, np.arange(len(one)), boot)
        # shadow: row-level random split (patients leak across train/test)
        rng = np.random.default_rng(SEED)
        rows_tr = rng.random(len(known)) < .7
        m_row = gbm(*best).fit(Xa[rows_tr], y[rows_tr])
        p_row = m_row.predict_proba(Xa[~rows_tr])[:, 1]
        shadow = {"after_patient_split": perf3["models"][1]["auroc"], "before_row_split": _auroc(y[~rows_tr], p_row),
                  "after_auprc": perf3["models"][1]["auprc"], "before_auprc": _ap(y[~rows_tr], p_row)}
        # shadow: results drawn but not yet reported used as features (charttime instead of availability)
        Xl = build_features(known[["stay_id", "t"]], lab, context, cohort, hist, time_col="charttime")[cols].astype(float)
        m_l = gbm(*best).fit(Xl[tr], y[tr])
        shadow["before_unreported_results"] = _auroc(y[te], m_l.predict_proba(Xl[te])[:, 1])
        shadow["unreported_share_of_features"] = float((Xl[te].notna() & Xa[te].isna()).values.mean())
        R["shadow_leakage"] = shadow
        if model_dir:
            import joblib
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": fitted["그래디언트 부스팅"], "features": cols, "kind": "agent3_gbm", "seed": SEED,
                         "trained_on": source, "built_at": R["meta"]["built_at"]}, Path(model_dir) / "agent3_gbm.joblib")
    R["agent3_performance"] = perf3
    R["agent3_target"] = {"threshold_x_uln": A3_TARGET, "definition": "any of ALT/ALP/TBIL >= threshold within 24 h, from hours when all three < 2x"}
    # sensitivity: the same model trained for 2/3/4/5x targets (fixed hyperparameters from the main grid search)
    if len(known) and best is not None:
        rows3 = []
        for k in THRESHOLDS:
            lk = agent3_labels(elig, lab, threshold=k)
            Kk = elig.assign(label=lk.values)
            Kk = Kk[Kk.label.isin(["abnormal_observed", "no_abnormality_observed"])]
            yk = Kk.label.eq("abnormal_observed").astype(int).to_numpy()
            trk = Kk.split.eq("development").to_numpy()
            row = {"threshold_x_uln": k, "rows": int(len(Kk)), "events": int(yk.sum()), "prevalence": float(yk.mean()) if len(yk) else None,
                   "patients": int(Kk.subject_id.nunique())}
            if len(np.unique(yk[trk])) == 2 and len(np.unique(yk[~trk])) == 2:
                log(f"agent3: target {k:g}x ULN")
                Xk = build_features(Kk[["stay_id", "t"]], lab, context, cohort, hist)[cols].astype(float)
                mk = gbm(*best).fit(Xk[trk], yk[trk])
                pk = mk.predict_proba(Xk[~trk])[:, 1]
                row.update(binary_metrics(yk[~trk], pk, Kk.subject_id.to_numpy()[~trk], max(20, boot // 4)))
                if model_dir:
                    import joblib
                    joblib.dump({"model": mk, "features": cols, "kind": f"agent3_gbm_{k:g}x", "seed": SEED, "trained_on": source},
                                Path(model_dir) / f"agent3_gbm_{k:g}x.joblib")
            rows3.append(row)
        R["agent3_by_threshold"] = rows3
    R["event_thresholds"] = threshold_table(cohort, lab)

    # ---------------- Agent 2: at the first alert
    perf2 = {"models": []}
    if len(outcomes):
        ev2 = events.merge(outcomes[["case_id", "first_event_48h"]], on="case_id", how="inner")
        R["agent2_development"] = {"events": int(len(ev2)), "label_counts": ev2.first_event_48h.value_counts().to_dict(),
                                   "excluded_insufficient": int(ev2.first_event_48h.eq("insufficient_measurement").sum())}
        K2 = ev2[ev2.first_event_48h.isin(A2_CLASSES)].copy()
        fol = agent2_follow(events, outcomes)
        # shadow: competing events ignored vs kept (evaluation events)
        fe = fol.merge(events[["case_id", "split"]], on="case_id")
        if len(fe):
            for sp in ["development", "evaluation"]:
                f_ = fe[fe.split.eq(sp)]
                if len(f_):
                    aj = aalen_johansen(f_.follow_hours, f_.first_event, 48)
                    R.setdefault("shadow_competing", {})[sp] = {"n": int(len(f_)), "aj_resolution_48h": aj["cif"]["resolution"][-1],
                                                                "naive_resolution_48h": aj["naive_resolution"][-1],
                                                                "aj_death_48h": aj["cif"]["death"][-1],
                                                                "aj_discharge_48h": aj["cif"]["icu_discharge"][-1], "curve": aj}
        if len(K2) and K2.split.eq("evaluation").any() and K2[K2.split.eq("development")].first_event_48h.nunique() >= 2:
            X2 = build_features(K2.rename(columns={"t0": "t"})[["stay_id", "t"]], lab, context, cohort, hist)
            cols2 = feature_columns(X2, K2.split.eq("development").values)
            Xa2 = X2[cols2].astype(float)
            tr2 = K2.split.eq("development").values
            y2 = K2.first_event_48h.values
            lr, gbm = _models()
            for name, m in [("다항 로지스틱 회귀", clone(lr)), ("그래디언트 부스팅", gbm(0.05, 15))]:
                m.fit(Xa2[tr2], y2[tr2])
                P = m.predict_proba(Xa2[~tr2])
                cls = list(m.classes_)
                res = {"model": name, "n_test": int((~tr2).sum()), "per_class": []}
                for c in A2_CLASSES:
                    yy = (y2[~tr2] == c).astype(int)
                    pp = P[:, cls.index(c)] if c in cls else np.zeros(len(yy))
                    res["per_class"].append({"class": c, **binary_metrics(yy, pp, np.arange(len(yy)), boot)})
                onehot = np.stack([(y2[~tr2] == c).astype(float) for c in cls], 1)
                res["multiclass_brier"] = float(np.mean(np.sum((P - onehot) ** 2, 1)))
                perf2["models"].append(res)
                if name == "그래디언트 부스팅" and model_dir:
                    import joblib
                    Path(model_dir).mkdir(parents=True, exist_ok=True)
                    joblib.dump({"model": m, "features": cols2, "kind": "agent2_gbm", "classes": cls, "seed": SEED,
                                 "trained_on": source}, Path(model_dir) / "agent2_gbm.joblib")
    R["agent2_performance"] = perf2

    # ---------------- Agent 1
    pred_path = Path(out_dir) / "09_agent_predictions.jsonl"
    llm_rows = [json.loads(x) for x in pred_path.read_text(encoding="utf-8").splitlines() if x.strip()] if pred_path.exists() else None
    R["agent1_performance"] = evaluate_agent1(events, lab, context, hist, diagnoses, llm_rows)

    # ---------------- other shadows from notebook outputs
    for key, name in [("alert_aggregation", "18_alert_aggregation.csv"), ("threshold_sensitivity", "33_threshold_sensitivity.csv"),
                      ("value_format_audit", "02c_value_format_audit.csv")]:
        p = Path(out_dir) / name
        R[key] = pd.read_csv(p).to_dict("records") if p.exists() else None
    sp = Path(out_dir) / "17_input_stress.jsonl"
    if sp.exists():
        from .evaluation import adoption_table
        rows = [json.loads(x) for x in sp.read_text(encoding="utf-8").splitlines() if x.strip()]
        R["shadow_input"] = adoption_table(rows, rows[0].get("injected_cause", "drug_related") if rows else "drug_related")
    return _suppress_counts(_json_safe(R), min_cell)


NOT_COUNTS = {"seed", "landmark_step_hours", "max_leaf_nodes", "n_features", "cutoff_fold_uln", "min_cell_suppression",
              # settings and ranks, not patient counts
              "horizon_h", "horizon_hours", "horizons", "step_hours", "hour", "rank", "k_selected", "n_clusters", "n_boot", "stable_threshold",
              "threshold_x_uln"}


def _suppress_counts(o, min_cell, key=None):
    """Suppress every small count (1..min_cell-1), including Table 1 [count, pct] pairs and label tallies."""
    if not min_cell:
        return o
    small = lambda v: isinstance(v, int) and not isinstance(v, bool) and 0 < v < min_cell
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            if k in NOT_COUNTS:
                out[k] = v
            elif small(v):
                out[k] = None
                for r in ("rate", "ci", "observed", "ci_low", "ci_high", "adoption_rate"):
                    if r in o:
                        out[r] = None
            else:
                out[k] = _suppress_counts(v, min_cell, k)
            if k in ("k", "n", "events", "adopted") and small(v):
                out["_suppressed"] = True
        if out.get("_suppressed"):
            for r in ("rate", "ci", "observed", "ci_low", "ci_high", "adoption_rate", "prevalence"):
                if r in out:
                    out[r] = None
        return out
    if isinstance(o, list):
        if len(o) == 2 and small(o[0]) and (o[1] is None or isinstance(o[1], float)):
            return [None, None]
        return [_suppress_counts(v, min_cell, key) for v in o]
    return o


def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return o


# ================================================================== v2: explanations, trajectories, prognosis
FEATURE_GROUPS = {
    "liver_level": "간 수치 (현재)", "liver_change": "간 수치 변화", "measurement": "측정 패턴",
    "perfusion": "관류·쇼크", "inflammation": "염증·감염", "renal_metabolic": "신장·대사",
    "synthetic": "합성 기능·혈액", "respiratory": "호흡", "background": "환자 배경·병력",
}
TRAJ_ANALYTES = ["ALT", "AST", "ALP", "TBIL"]
TRAJ_HORIZONS = [6, 12, 18, 24]
TRAJ_QUANTILES = [0.1, 0.5, 0.9]
PROGNOSIS = ["recovered_alive_1y", "recovered_died_1y",
             "unrecovered_alive_1y", "unrecovered_died_1y"]  # recovery-assessable events; unassessable_* kept for death only


def group_of(col: str) -> str:
    base = col.split("_")[0]
    if col.endswith("_n24") or col.endswith("_age_h"):
        return "measurement"
    if base in RATIO_FEATS:
        return "liver_change" if col.endswith("_delta24") else "liver_level"
    if base in ("MAP", "HR", "LACTATE") or col == "pressor_6h":
        return "perfusion"
    if base in ("WBC", "TEMP", "PLATELET"):
        return "inflammation"
    if base in ("CREAT", "BUN", "SODIUM", "GLUCOSE", "BICARB"):
        return "renal_metabolic"
    if base in ("INR", "ALBUMIN", "HEMOGLOBIN"):
        return "synthetic"
    if base in ("RR", "SPO2"):
        return "respiratory"
    return "background"


def group_shapley(predict, X: pd.DataFrame, background: pd.DataFrame, cols: list[str], max_rows=None) -> np.ndarray:
    """Exact Shapley values over clinical feature GROUPS (interventional, averaged over a background sample).

    With 9 groups there are 2^9 coalitions, so no sampling approximation is needed.
    Values are in the model's output units (probability) and sum to f(x) - mean f(background).
    Grouping keeps correlated measurements of one physiology together, so credit is not split arbitrarily
    between, e.g., ALT level and ALT change.
    """
    import itertools
    groups = list(FEATURE_GROUPS)
    gidx = {g: [i for i, c in enumerate(cols) if group_of(c) == g] for g in groups}
    groups = [g for g in groups if gidx[g]]
    G = len(groups)
    Xv = X[cols].to_numpy(float)
    Bv = background[cols].to_numpy(float)
    K = len(Bv)
    masks = np.array(list(itertools.product([0, 1], repeat=G)), dtype=bool)  # 2^G x G
    from math import factorial
    out = np.zeros((len(Xv), G))
    for r in range(len(Xv)):
        # rows: coalition x background
        M = np.repeat(Bv[None, :, :], len(masks), axis=0).copy()  # C x K x p
        for gi, g in enumerate(groups):
            on = masks[:, gi]
            M[np.ix_(on, np.arange(K), gidx[g])] = Xv[r, gidx[g]]
        v = predict(M.reshape(-1, Xv.shape[1])).reshape(len(masks), K).mean(1)
        size = masks.sum(1)
        index = {tuple(m): i for i, m in enumerate(masks)}
        for gi in range(G):
            phi = 0.0
            for i, m in enumerate(masks):
                if m[gi]:
                    continue
                m2 = m.copy()
                m2[gi] = True
                s = size[i]
                w = factorial(s) * factorial(G - s - 1) / factorial(G)
                phi += w * (v[index[tuple(m2)]] - v[i])
            out[r, gi] = phi
    return out, groups


def synthetic_background(X: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """n reference rows; row i takes the ((i+0.5)/n) percentile of every variable independently.

    No row equals a real patient. Binary variables keep 0/1 values; never-observed variables stay missing.
    """
    qs = [(i + .5) / n for i in range(n)]
    A = X.apply(pd.to_numeric, errors="coerce")
    return pd.DataFrame({c: (A[c].quantile(qs, interpolation="nearest").to_numpy() if A[c].notna().any() else [np.nan] * n)
                         for c in A.columns})


def explain_rows(bundle, X: pd.DataFrame, background: pd.DataFrame | None = None) -> list[dict]:
    """Per-row group Shapley for an Agent 3 bundle (dict with model, features, background)."""
    cols = bundle["features"]
    bg = background if background is not None else bundle.get("background")
    predict = lambda A: bundle["model"].predict_proba(pd.DataFrame(A, columns=cols))[:, 1]
    phi, groups = group_shapley(predict, X, bg, cols)
    base = float(predict(bg[cols].to_numpy(float)).mean())
    return [{"base": base, "groups": [{"group": g, "label": FEATURE_GROUPS[g], "value": float(v)} for g, v in zip(groups, row)]}
            for row in phi]


def trajectory_rows(cohort, lab, step_hours=4, horizons=TRAJ_HORIZONS, window=3, max_landmarks=300_000):
    """Landmarks every ``step_hours`` during the ICU stay with the observed log2(value/ULN) near t+h."""
    c = cohort[["stay_id", "subject_id", "split", "intime", "monitor_end"]].dropna()
    rows = []
    for r in c.itertuples():
        n = int((r.monitor_end - r.intime).total_seconds() // 3600 // step_hours)
        if n >= 1:
            rows.append(pd.DataFrame({"stay_id": r.stay_id, "subject_id": r.subject_id, "split": r.split,
                                      "t": r.intime + pd.to_timedelta(np.arange(1, n + 1) * step_hours, unit="h")}))
    L = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["stay_id", "subject_id", "split", "t"])
    L["t"] = pd.to_datetime(L.t).astype("datetime64[ns]")
    if len(L) > max_landmarks:
        L = L.sample(max_landmarks, random_state=SEED).sort_values(["stay_id", "t"]).reset_index(drop=True)
    targets = {}
    for a in TRAJ_ANALYTES:
        e = lab[lab.analyte.eq(a) & lab.ratio.gt(0)][["stay_id", "charttime", "ratio"]].dropna().sort_values("charttime")
        e = e.assign(y=np.log2(e.ratio))
        for h in horizons:
            Q = L[["stay_id", "t"]].assign(_i=np.arange(len(L)), th=L.t + pd.Timedelta(hours=h)).sort_values("th")
            m = pd.merge_asof(Q, e[["stay_id", "charttime", "y"]], left_on="th", right_on="charttime", by="stay_id",
                              direction="nearest", tolerance=pd.Timedelta(hours=window)).sort_values("_i")
            ok = m.charttime.values > L.t.values  # the target must be drawn after t
            targets[(a, h)] = np.where(ok, m.y.values, np.nan)
    return L, targets


def fit_trajectory(L, targets, X, cols, max_rows=400_000, boot_rows=None):
    """Quantile gradient boosting per analyte with horizon as a feature; persistence baseline for comparison."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    models, perf = {}, []
    tr = L.split.eq("development").values
    for a in TRAJ_ANALYTES:
        last = X[f"{a}_latest"].to_numpy(float)
        Xs, ys, hs, sp, lastv = [], [], [], [], []
        for h in TRAJ_HORIZONS:
            y = targets[(a, h)]
            keep = ~np.isnan(y) & ~np.isnan(last)  # forecast only analytes measured in the last 24 h
            Xs.append(X.loc[keep, cols].assign(horizon_h=float(h)))
            ys.append(y[keep]); hs.append(np.full(keep.sum(), h)); sp.append(tr[keep]); lastv.append(np.log2(np.clip(last[keep], 1e-3, None)))
        if not sum(len(y) for y in ys):
            continue
        XA = pd.concat(Xs, ignore_index=True)
        yA, hA, trA, lastA = np.concatenate(ys), np.concatenate(hs), np.concatenate(sp), np.concatenate(lastv)
        idx_tr = np.where(trA)[0]
        if len(idx_tr) > max_rows:
            idx_tr = np.random.default_rng(SEED).choice(idx_tr, max_rows, replace=False)
        if len(idx_tr) < 30 or (~trA).sum() < 5:
            continue
        qs = {}
        for q in TRAJ_QUANTILES:
            m = HistGradientBoostingRegressor(loss="quantile", quantile=q, learning_rate=0.05, max_leaf_nodes=31,
                                              max_iter=300, early_stopping="auto", random_state=SEED)
            m.fit(XA.iloc[idx_tr], yA[idx_tr])
            qs[q] = m
        models[a] = qs
        te = ~trA
        P = _noncrossing({q: qs[q].predict(XA[te]) for q in TRAJ_QUANTILES})
        for h in TRAJ_HORIZONS:
            sel = hA[te] == h
            if sel.sum() < 3:
                continue
            yy = yA[te][sel]
            perf.append({"analyte": a, "horizon_h": h, "n": int(sel.sum()),
                         "mae_model": float(np.mean(np.abs(P[0.5][sel] - yy))),
                         "mae_persistence": float(np.mean(np.abs(lastA[te][sel] - yy))),
                         "coverage_80": float(np.mean((yy >= P[0.1][sel]) & (yy <= P[0.9][sel])))})
    return models, perf


def _noncrossing(P: dict) -> dict:
    """Monotone rearrangement: sort each row's quantile predictions so q10 <= q50 <= q90 (Chernozhukov et al. 2010)."""
    qs = sorted(P)
    M = np.sort(np.stack([P[q] for q in qs], 1), axis=1)
    return {q: M[:, i] for i, q in enumerate(qs)}


def forecast_rows(traj_bundle, X: pd.DataFrame) -> list[dict]:
    """Per-row 24 h trajectory: for each analyte and horizon the 10/50/90% quantiles in x ULN."""
    cols = traj_bundle["features"]
    out = [dict() for _ in range(len(X))]
    for a, qs in traj_bundle["models"].items():
        has = X[f"{a}_latest"].notna().to_numpy()
        for h in TRAJ_HORIZONS:
            XA = X[cols].assign(horizon_h=float(h))
            pred = {q: 2 ** v for q, v in _noncrossing({q: qs[q].predict(XA) for q in TRAJ_QUANTILES}).items()}
            for i in range(len(X)):
                if has[i]:
                    out[i].setdefault(a, []).append({"h": h, "q10": float(pred[0.1][i]), "q50": float(pred[0.5][i]),
                                                     "q90": float(pred[0.9][i])})
    return out


def evaluate_flags(events, lab, context, hist, diagnoses, llm_rows=None):
    """Per-cause 0/1 agreement with discharge-ICD proxy categories (evaluation events only)."""
    from .rule_agent import CAUSE_FLAGS
    proxy = icd_proxy(events, diagnoses)
    ev = events[events.split.eq("evaluation")]
    lab_g = {k: g for k, g in lab[lab.stay_id.isin(set(ev.stay_id))].groupby("stay_id")}
    ctx_g = {k: g for k, g in context[context.stay_id.isin(set(ev.stay_id))].groupby("stay_id")}
    hist_i = hist.set_index("stay_id").to_dict("index") if hist is not None else {}
    pred, ref = [], []
    for e in ev.itertuples():
        a = rule_agent1(snapshot_payload(e.stay_id, e.t0, lab_g, ctx_g, hist_i.get(e.stay_id)))
        pred.append({c: a["flags"][c]["present"] for c in CAUSE_FLAGS})
        ref.append({c: int(c in proxy.get(int(e.hadm_id), set())) for c in CAUSE_FLAGS})
    rows = []
    for c in CAUSE_FLAGS:
        p = np.array([r[c] for r in pred], int)
        y = np.array([r[c] for r in ref], int)
        tp, fp, fn, tn = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum()), int(((p == 0) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
        sens, spec = wilson(tp, tp + fn), wilson(tn, tn + fp)
        from .rule_agent import NOT_ASSESSABLE
        rows.append({"cause": c, "assessable": c not in NOT_ASSESSABLE, "proxy_available": ICD_PROXY.get(c) is not None,
                     "flagged": int(p.sum()), "proxy_positive": int(y.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "sensitivity": tp / (tp + fn) if tp + fn else None, "sensitivity_ci": list(sens) if tp + fn else None,
                     "specificity": tn / (tn + fp) if tn + fp else None, "specificity_ci": list(spec) if tn + fp else None})
    n = len(pred)
    multi = int(sum(sum(r.values()) >= 2 for r in pred))
    return {"model": RULE_VERSION, "n_events": n, "per_cause": rows, "events_with_multiple_flags": multi,
            "events_with_no_flag": int(sum(sum(r.values()) == 0 for r in pred))}


def develop_v2(R, out_dir, admissions, diagnoses, exclude_subjects=None, model_dir=None, min_cell=10, boot=200,
               traj_step_hours=4, log=print):
    """Adds trajectories, group-Shapley explanations, 4-class prognosis and 0/1 cause flags to results ``R``."""
    import joblib
    D = load_private(out_dir)
    cohort, lab, events, context, outcomes = D["cohort"], D["lab"], D["events"], D["context"], D["outcomes"]
    excluded = set(exclude_subjects or [])
    cohort = cohort[~cohort.subject_id.isin(excluded)].copy()
    keep = set(cohort.stay_id)
    lab, context, events = lab[lab.stay_id.isin(keep)], context[context.stay_id.isin(keep)], events[events.stay_id.isin(keep)].copy()
    hist = history_flags(cohort, admissions, diagnoses)
    if model_dir:
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    # ---- Agent 3 explanations (global group Shapley on evaluation landmarks)
    a3p = Path(model_dir) / "agent3_gbm.joblib" if model_dir else None
    if a3p and a3p.exists():
        log("v2: group Shapley for Agent 3")
        b = joblib.load(a3p)
        L = agent3_landmarks(cohort, lab, events, step_hours=1)
        L["label"] = agent3_labels(L, lab) if len(L) else []
        K = L[L.eligible & L.label.isin(["abnormal_observed", "no_abnormality_observed"])]
        if len(K):
            X = build_features(K[["stay_id", "t"]], lab, context, cohort, hist)
            dev, te = K.split.eq("development").values, K.split.eq("evaluation").values
            rng = np.random.default_rng(SEED)
            # synthetic reference rows (per-variable percentiles), so the shareable model file holds no real patient row
            bg = synthetic_background(X.loc[dev, b["features"]] if dev.any() else X[b["features"]], n=10)
            b["background"] = bg.reset_index(drop=True)
            joblib.dump(b, a3p)
            pick = np.where(te)[0]
            pick = rng.choice(pick, min(300, len(pick)), replace=False) if len(pick) else pick
            if len(pick):
                ex = explain_rows(b, X.iloc[pick].reset_index(drop=True))
                G = pd.DataFrame([{g["group"]: g["value"] for g in e["groups"]} for e in ex])
                imp = [{"group": g, "label": FEATURE_GROUPS[g], "mean_abs": float(G[g].abs().mean()), "mean": float(G[g].mean())}
                       for g in G.columns]
                imp.sort(key=lambda r: -r["mean_abs"])
                R["agent3_explanation"] = {"method": "exact group Shapley (9 clinical groups, interventional, 10 background rows)",
                                           "n_rows": int(len(pick)), "importance": imp}
                tot = sum(r["mean_abs"] for r in imp) or 1
                m = next((r for r in imp if r["group"] == "measurement"), None)
                R["shadow_shortcut"] = {"measurement_share": m["mean_abs"] / tot if m else None,
                                        "rank": [r["group"] for r in imp].index("measurement") + 1 if m else None,
                                        "groups": len(imp)}

    # ---- 24 h trajectories
    log("v2: trajectory landmarks")
    Lt, targets = trajectory_rows(cohort, lab, step_hours=traj_step_hours)
    if len(Lt):
        Xt = build_features(Lt[["stay_id", "t"]], lab, context, cohort, hist)
        cols_t = feature_columns(Xt, Lt.split.eq("development").values)
        models, perf = fit_trajectory(Lt, targets, Xt, cols_t)
        R["trajectory_performance"] = perf
        R["trajectory_development"] = {"landmarks": int(len(Lt)), "step_hours": traj_step_hours, "horizons": TRAJ_HORIZONS,
                                       "analytes": TRAJ_ANALYTES, "quantiles": TRAJ_QUANTILES,
                                       "targets_available": {f"{a}_{h}": int(np.isfinite(targets[(a, h)]).sum()) for a, h in targets}}
        if model_dir and models:
            joblib.dump({"models": models, "features": cols_t, "kind": "trajectory_qgbm", "trained_on": R["meta"]["source"]},
                        Path(model_dir) / "trajectory_qgbm.joblib")

    # ---- Agent 2: 4-class prognosis (resolution x death)
    if len(outcomes):
        ev_all = events.merge(outcomes[["case_id", "observed_path"] + (["died_1y"] if "died_1y" in outcomes else [])], on="case_id", how="inner")
        ev2 = ev_all[ev_all.observed_path.isin(PROGNOSIS)]  # recovery assessable
        R["prognosis_development"] = {"events": int(len(ev_all)), "assessable": int(len(ev2)),
                                      "label_counts": ev_all.observed_path.value_counts().to_dict(),
                                      "definition": "recovery = all three < 2x (24 h) before ICU discharge; unassessable = no trigger lab after the event; death = within 1 year"}
        death_bundle = None
        if "died_1y" in ev_all and ev_all.split.eq("evaluation").any() and ev_all[ev_all.split.eq("development")].died_1y.astype(bool).nunique() == 2:
            # 1-year death on ALL events, including those whose recovery cannot be assessed
            Xd = build_features(ev_all.rename(columns={"t0": "t"})[["stay_id", "t"]], lab, context, cohort, hist)
            trd = ev_all.split.eq("development").to_numpy()
            cold = feature_columns(Xd, trd)
            yd = ev_all.died_1y.astype(str).str.lower().isin(["true", "1"]).astype(int).to_numpy()
            _, gbm_d = _models()
            md = gbm_d(0.05, 15).fit(Xd[cold].astype(float)[trd], yd[trd])
            pdx = md.predict_proba(Xd[cold].astype(float)[~trd])[:, 1]
            R["death_1y_performance"] = {"n_events": int(len(ev_all)), **binary_metrics(yd[~trd], pdx, ev_all.subject_id.to_numpy()[~trd], boot)}
            death_bundle = {"death_model": md, "death_features": cold}
        perf = {"models": []}
        if len(ev2) and ev2.split.eq("evaluation").any() and ev2[ev2.split.eq("development")].observed_path.nunique() >= 2:
            from sklearn.base import clone
            X2 = build_features(ev2.rename(columns={"t0": "t"})[["stay_id", "t"]], lab, context, cohort, hist)
            cols2 = feature_columns(X2, ev2.split.eq("development").values)
            Xa = X2[cols2].astype(float)
            tr = ev2.split.eq("development").values
            y = ev2.observed_path.values
            lr, gbm = _models()
            for name, m in [("다항 로지스틱 회귀", clone(lr)), ("그래디언트 부스팅", gbm(0.05, 15))]:
                m.fit(Xa[tr], y[tr])
                P = m.predict_proba(Xa[~tr])
                cls = list(m.classes_)
                res = {"model": name, "n_test": int((~tr).sum()), "per_class": []}
                for c in PROGNOSIS:
                    yy = (y[~tr] == c).astype(int)
                    pp = P[:, cls.index(c)] if c in cls else np.zeros(len(yy))
                    res["per_class"].append({"class": c, **binary_metrics(yy, pp, np.arange(len(yy)), boot)})
                pred = np.array(cls)[P.argmax(1)]
                res["accuracy"] = float(np.mean(pred == y[~tr]))
                res["confusion"] = {f"{a}|{b}": int(((y[~tr] == a) & (pred == b)).sum()) for a in PROGNOSIS for b in PROGNOSIS}
                perf["models"].append(res)
                if name == "그래디언트 부스팅" and model_dir:
                    joblib.dump({"model": m, "features": cols2, "kind": "prognosis_gbm", "classes": cls,
                                 "trained_on": R["meta"]["source"], **(death_bundle or {})}, Path(model_dir) / "prognosis_gbm.joblib")
        R["prognosis_performance"] = perf

    # ---- Agent 1: 0/1 flags
    R["agent1_flags"] = evaluate_flags(events, lab, context, hist, diagnoses)

    # ---- shadow: one wrongly entered value (age 19 -> 190, decimal/unit slips) before vs after a plausibility guard
    try:
        from .data_errors import experiment
        a3b = joblib.load(a3p) if a3p and a3p.exists() else None
        tp = Path(model_dir) / "trajectory_qgbm.joblib" if model_dir else None
        trb = joblib.load(tp) if tp and tp.exists() else None
        L = agent3_landmarks(cohort, lab, events, step_hours=1)
        L = L[L.split.eq("evaluation") & L.eligible]
        # pre-declared sample: one random eligible hour per evaluation patient, at most 2,000 patients
        rows = L.sample(frac=1, random_state=SEED).groupby("stay_id").head(1).head(2000)
        ev_eval = events[events.split.eq("evaluation")]
        ev_lab = lab[lab.stay_id.isin(set(cohort.loc[cohort.split.eq("evaluation"), "stay_id"]))]
        if len(rows):
            log(f"v2: data-entry error shadow on {len(rows):,} evaluation patients")
            R["shadow_data_error"] = experiment(rows, ev_lab, context, cohort, hist, a3b, trb, rule_agent1, snapshot_payload,
                                                events=ev_eval, log=log)
    except Exception as ex:  # reported, never silently dropped
        R["shadow_data_error"] = {"status": "failed", "error": f"{type(ex).__name__}: {ex}"}
    return _suppress_counts(_json_safe(R), min_cell)
