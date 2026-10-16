"""Shadow experiment: what happens when a single value is entered wrongly (age 19 -> 190, decimal or unit slips).

Before = the current pipeline (only the static bounds the notebook already applies).
After  = plus a point-in-time plausibility guard:
  R1 hard physiological range (impossible values are rejected),
  R2 implausible jump against the same patient's previous value within 48 h
     (ALT/AST: accepted when the partner enzyme from the same draw also rose; true shock liver moves both),
  R3 discordance: ALT/AST >= 20x ULN with the partner enzyme from the same draw < 3x ULN.
Quarantined values are treated as not measured and shown as "확인 필요" with the original value.
The guard's own cost is measured on unmodified data: real values it would hold, and true events it would delay.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# what the notebook already rejects (cell "Plausibility bounds"); the realistic "before"
CURRENT_BOUNDS = {"ALBUMIN": (.1, 10), "SODIUM": (80, 200), "CREAT": (.05, 150), "PLATELET": (0, 2000),
                  "WBC": (0, 500), "GLUCOSE": (1, 2000)}
HARD_RANGE = {"ALT": (1, 20000), "AST": (1, 30000), "ALP": (5, 5000), "TBIL": (.05, 60), "CREAT": (.1, 25),
              "LACTATE": (.1, 30), "ALBUMIN": (.5, 7), "INR": (.5, 20), "PLATELET": (1, 2000), "WBC": (.05, 500),
              "SODIUM": (90, 190), "GLUCOSE": (10, 2000), "BUN": (1, 300), "HEMOGLOBIN": (2, 25), "BICARB": (2, 60)}
VITAL_RANGE = {"TEMP_C": (25, 45), "MAP": (15, 250), "HR": (10, 300), "RR": (2, 80), "SPO2": (40, 100)}
AGE_RANGE = (18, 120)
JUMP_UP = {"ALT": 20, "AST": 20, "TBIL": 8, "CREAT": 5, "ALBUMIN": 2.5, "INR": 6}
PARTNER = {"ALT": "AST", "AST": "ALT"}
JUMP_WINDOW_H = 48

SCENARIOS = [
    {"key": "age_x10", "label": "나이 자릿수 추가 (19 → 190)", "kind": "age", "factor": 10},
    {"key": "alt_x100", "label": "ALT 소수점 밀림 (45 → 4500)", "kind": "lab", "analyte": "ALT", "factor": 100},
    {"key": "tbil_umol", "label": "빌리루빈을 µmol/L로 입력 (×17.1)", "kind": "lab", "analyte": "TBIL", "factor": 17.1},
    {"key": "creat_umol", "label": "크레아티닌을 µmol/L로 입력 (×88.4)", "kind": "lab", "analyte": "CREAT", "factor": 88.4},
    {"key": "lactate_x10", "label": "젖산 소수점 밀림 (2.1 → 21)", "kind": "lab", "analyte": "LACTATE", "factor": 10},
    {"key": "temp_f", "label": "체온 °F를 °C 칸에 (37 → 98.6)", "kind": "vital", "label_name": "TEMP_C"},
    {"key": "temp_malformed", "label": "체온 형식 오류 (37.5 → '37..5')", "kind": "malformed_vital", "label_name": "TEMP_C"},
    {"key": "alt_malformed", "label": "ALT 형식 오류 (45 → '4..5')", "kind": "malformed", "analyte": "ALT"},
]


def classify_text_value(text) -> str:
    """Why a numeric field has no number: censored ('<5'), de-identified ('___'), malformed number ('37..5'), other text."""
    t = str(text or "").strip()
    if not t or t.lower() == "nan":
        return "empty"
    if re.fullmatch(r"_+\.?", t):
        return "deidentified"
    if re.match(r"^[<>]=?\s*\d", t):
        return "censored"
    if re.search(r"\d", t) and re.fullmatch(r"[\d\s.,+-]+\*?\.?", t):
        return "malformed_number"
    return "text"


def malformed(v: float) -> str:
    """How a slip typically looks: 37.5 -> '37..5', 45 -> '4..5'."""
    s = f"{v:g}"
    return s.replace(".", "..", 1) if "." in s else (s[:-1] + ".." + s[-1] if len(s) > 1 else s + "..")


def apply_current_bounds(lab: pd.DataFrame) -> pd.DataFrame:
    keep = pd.Series(True, index=lab.index)
    for a, (lo, hi) in CURRENT_BOUNDS.items():
        m = lab.analyte.eq(a)
        keep &= ~m | lab.value.between(lo, hi)
    return lab[keep]


def guard_lab(lab: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (lab without quarantined rows, quarantined rows with reason). Uses only the patient's past."""
    d = lab.copy()
    d["_reason"] = ""
    for a, (lo, hi) in HARD_RANGE.items():
        m = d.analyte.eq(a) & ~d.value.between(lo, hi)
        d.loc[m, "_reason"] = "range"
    d = d.sort_values(["stay_id", "analyte", "charttime"])
    g = d.groupby(["stay_id", "analyte"], sort=False)
    prev_v, prev_t = g.value.shift(1), g.charttime.shift(1)
    recent = (d.charttime - prev_t).dt.total_seconds().le(JUMP_WINDOW_H * 3600)
    d["_jump"] = np.where(recent & prev_v.gt(0), d.value / prev_v, np.nan)
    for a, k in JUMP_UP.items():
        m = d.analyte.eq(a) & d._reason.eq("") & (d._jump >= k)
        if a == "ALBUMIN":
            m |= d.analyte.eq(a) & d._reason.eq("") & (d._jump <= 1 / k)
        d.loc[m, "_reason"] = "jump"
    # ALT/AST jump confirmed when the partner from the same draw (within 1 h) also rose >= 5x
    for a, b in PARTNER.items():
        cand = d[d.analyte.eq(a) & d._reason.eq("jump")]
        if cand.empty:
            continue
        part = d[d.analyte.eq(b)][["stay_id", "charttime", "_jump"]].dropna(subset=["charttime"]).sort_values("charttime")
        c2 = cand[["stay_id", "charttime"]].reset_index().sort_values("charttime")
        m = pd.merge_asof(c2, part.rename(columns={"_jump": "pj"}), on="charttime", by="stay_id",
                          direction="nearest", tolerance=pd.Timedelta(hours=1))
        ok = m.loc[m.pj.ge(5), "index"]
        d.loc[ok, "_reason"] = ""
        d.loc[m.loc[~m.pj.ge(5), "index"], "_reason"] = "jump_unconfirmed"
    # R3 discordance: ALT/AST >= 20x ULN while the partner from the same draw is < 3x ULN (shock liver raises both)
    if "ratio" in d:
        for a, b in PARTNER.items():
            cand = d[d.analyte.eq(a) & d._reason.eq("") & d.ratio.ge(20)]
            if cand.empty:
                continue
            part = d[d.analyte.eq(b) & d.ratio.notna()][["stay_id", "charttime", "ratio"]].sort_values("charttime")
            c2 = cand[["stay_id", "charttime"]].reset_index().sort_values("charttime")
            m = pd.merge_asof(c2, part.rename(columns={"ratio": "pr"}), on="charttime", by="stay_id",
                              direction="nearest", tolerance=pd.Timedelta(hours=1))
            d.loc[m.loc[m.pr.lt(3), "index"], "_reason"] = "discordant_partner"
    q = d[d._reason.ne("")]
    clean = d[d._reason.eq("")].drop(columns=["_reason", "_jump"]).sort_index()
    return clean, q.drop(columns=["_jump"]).rename(columns={"_reason": "reason"})


def guard_context(context: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    v = pd.to_numeric(context.value, errors="coerce")
    bad = pd.Series(False, index=context.index)
    for name, (lo, hi) in VITAL_RANGE.items():
        bad |= context.kind.eq("vital") & context.label.eq(name) & v.notna() & ~v.between(lo, hi)
    return context[~bad], int(bad.sum())


def guard_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    c = cohort.copy()
    c.loc[~c.age.between(*AGE_RANGE), "age"] = np.nan
    return c


def corrupt(sc: dict, rows: pd.DataFrame, lab, context, cohort, lookback_h=24):
    """Corrupt the latest value available at t for each (stay_id, t) row. Returns copies and an applicable mask."""
    lab, context, cohort = lab, context, cohort
    applicable = np.zeros(len(rows), bool)
    orig, new = [np.nan] * len(rows), [np.nan] * len(rows)
    if sc["kind"] == "age":
        cohort = cohort.copy()
        idx = cohort.stay_id.isin(set(rows.stay_id))
        cohort.loc[idx, "age"] = cohort.loc[idx, "age"] * sc["factor"]
        ages = cohort.set_index("stay_id").age
        for i, s in enumerate(rows.stay_id):
            applicable[i] = True
            new[i] = float(ages.get(s, np.nan))
            orig[i] = new[i] / sc["factor"]
        return lab, context, cohort, applicable, orig, new
    on_lab = sc["kind"] in ("lab", "malformed")
    src = lab if on_lab else context
    is_target = src.analyte.eq(sc["analyte"]) if on_lab else (src.kind.eq("vital") & src.label.eq(sc["label_name"]))
    tgt = src[is_target & src.stay_id.isin(set(rows.stay_id))]
    by = {k: g.sort_values("available_at") for k, g in tgt.groupby("stay_id")}
    src = src.copy()
    if sc["kind"] == "vital":
        src["value"] = src["value"].astype(object)
    drop = []
    for i, (s, t) in enumerate(zip(rows.stay_id, pd.to_datetime(rows.t))):
        g = by.get(s)
        if g is None:
            continue
        g = g[(g.available_at <= t) & (g.charttime >= t - pd.Timedelta(hours=lookback_h if on_lab else 6))]
        if g.empty:
            continue
        j = g.index[-1]
        v = float(pd.to_numeric(src.at[j, "value"], errors="coerce"))
        if not np.isfinite(v):
            continue
        if sc["kind"].startswith("malformed"):
            # the pipeline coerces '37..5' to NaN and drops the row without any warning: simulate exactly that
            drop.append(j)
            applicable[i], orig[i], new[i] = True, v, malformed(v)
            continue
        nv = v * sc["factor"] if sc["kind"] == "lab" else v * 9 / 5 + 32
        src.at[j, "value"] = nv
        if sc["kind"] == "lab" and "ratio" in src and pd.notna(src.at[j, "ratio"]):
            src.at[j, "ratio"] = float(src.at[j, "ratio"]) * sc["factor"]
        applicable[i], orig[i], new[i] = True, v, nv
    if drop:
        src = src.drop(index=drop)
    if on_lab:
        return src, context, cohort, applicable, orig, new
    return lab, src, cohort, applicable, orig, new


def _flags(rule, snap_fn, rows, lab, context, hist):
    lab_g = {k: g for k, g in lab[lab.stay_id.isin(set(rows.stay_id))].groupby("stay_id")}
    ctx_g = {k: g for k, g in context[context.stay_id.isin(set(rows.stay_id))].groupby("stay_id")}
    hist_i = hist.set_index("stay_id").to_dict("index") if hist is not None and len(hist) else {}
    out = []
    for s, t in zip(rows.stay_id, rows.t):
        a = rule(snap_fn(s, t, lab_g, ctx_g, hist_i.get(s)))
        out.append(sorted(c for c, v in a["flags"].items() if v["present"]))
    return out


def experiment(rows: pd.DataFrame, lab, context, cohort, hist, a3=None, traj=None, rule=None, snap_fn=None,
               scenarios=SCENARIOS, high_risk_quantile=0.9, max_rule_rows=500, per_row=False, events=None, log=print):
    """Run every scenario on the same rows under four conditions; return a JSON-ready summary."""
    from .develop import build_features, RATIO_FEATS
    def _ids(d):  # notebook frames can carry stay_id as object; merge_asof needs one integer type
        d = d.copy()
        d["stay_id"] = pd.to_numeric(d["stay_id"], errors="coerce")
        d = d.dropna(subset=["stay_id"])
        d["stay_id"] = d["stay_id"].astype("int64")
        for c in ("charttime", "available_at", "t", "intime"):
            if c in d:
                d[c] = pd.to_datetime(d[c], errors="coerce").astype("datetime64[ns]")
        return d
    rows, lab, context, cohort = _ids(rows), _ids(lab), _ids(context), _ids(cohort)
    if hist is not None and len(hist):
        hist = _ids(hist)
    rows = rows[["stay_id", "t"]].reset_index(drop=True)
    cost_lab = apply_current_bounds(lab)  # guard cost is measured on all supplied records
    keep = set(rows.stay_id)
    lab, context, cohort = lab[lab.stay_id.isin(keep)], context[context.stay_id.isin(keep)], cohort[cohort.stay_id.isin(keep)]
    lab_now = apply_current_bounds(lab)
    feats = lambda L, C, H: build_features(rows, L, C, H, hist)

    def risk(X):
        return a3["model"].predict_proba(X[a3["features"]].astype(float))[:, 1] if a3 else None

    def fc24(X, a):
        if not traj or a not in traj["models"]:
            return None
        from .develop import forecast_rows
        f = forecast_rows(traj, X)
        return np.array([next((p["q50"] for p in r.get(a, []) if p["h"] == 24), np.nan) for r in f])

    X0 = feats(lab_now, context, cohort)
    p0 = risk(X0)
    hi = float(np.quantile(p0, high_risk_quantile)) if p0 is not None and len(p0) else None
    rr = rows.head(max_rule_rows)
    fl0 = _flags(rule, snap_fn, rr, lab_now, context, hist) if rule else None
    res = {"design": {"rows": int(len(rows)), "patients": int(rows.stay_id.nunique()),
                      "high_risk_cutoff": hi, "high_risk_definition": f"top {int((1 - high_risk_quantile) * 100)}% of clean risk",
                      "guard": "R1 physiological range + R2 jump vs own previous value within 48 h + R3 ALT/AST partner discordance",
                      "rule_rows": int(len(rr))}, "scenarios": []}
    lab_g0, _ = guard_lab(lab_now)
    ctx_g0, nctx0 = guard_context(context)
    # cost of the guard on unmodified data (all supplied records)
    _, q0 = guard_lab(cost_lab)
    cost = {"lab_values": int(len(cost_lab)), "quarantined": int(len(q0)),
            "quarantined_share": float(len(q0) / len(cost_lab)) if len(cost_lab) else None,
            "by_analyte": [{"analyte": a, "n": int(n), "reason": r} for (a, r), n in q0.groupby(["analyte", "reason"]).size().items()],
            "vitals_rejected_in_sample": nctx0}
    X0g = feats(lab_g0, ctx_g0, guard_cohort(cohort))
    p0g = risk(X0g)
    if p0 is not None:
        cost["clean_median_abs_dp"] = float(np.nanmedian(np.abs(p0g - p0)))
    if events is not None and len(events):
        ev = events[events.stay_id.isin(set(cost_lab.stay_id))]
        qq = q0[["stay_id", "analyte", "charttime"]]
        m = ev.merge(qq, left_on=["stay_id", "trigger_analyte"], right_on=["stay_id", "analyte"], how="left")
        held = m[(m.charttime <= m.t0) & (m.charttime >= m.t0 - pd.Timedelta(hours=1))].drop_duplicates("case_id")
        cost["true_events"] = int(len(ev))
        cost["true_events_trigger_held"] = int(len(held))
    res["guard_cost_on_real_data"] = cost
    rows_out = [dict(stay_id=int(s), t=str(t)) for s, t in zip(rows.stay_id, rows.t)] if per_row else None
    for sc in scenarios:
        log(f"data errors: {sc['key']}")
        L1, C1, H1, app, orig, new = corrupt(sc, rows, lab, context, cohort)
        L1b = apply_current_bounds(L1)
        X1 = feats(L1b, C1, H1)
        L1g, q1 = guard_lab(L1b)
        C1g, _ = guard_context(C1)
        H1g = guard_cohort(H1)
        X1g = feats(L1g, C1g, H1g)
        n = int(app.sum())
        r = {"key": sc["key"], "label": sc["label"], "applicable": n}
        if n == 0:
            res["scenarios"].append(r)
            continue
        # was the corrupted value removed? (before = current bounds only; after = guard)
        if sc["kind"] == "age":
            caught_before = np.zeros(n, bool)
            caught_after = ~H1.set_index("stay_id").age.reindex(rows.stay_id[app]).between(*AGE_RANGE).to_numpy()
        elif sc["kind"] == "lab":
            a = sc["analyte"]
            kept_b = set(zip(L1b.stay_id, L1b.charttime, L1b.analyte, L1b.value.round(6)))
            kept_a = set(zip(L1g.stay_id, L1g.charttime, L1g.analyte, L1g.value.round(6)))
            tgt = L1[L1.analyte.eq(a)]
            key = {(s, round(v, 6)) for s, v in zip(rows.stay_id[app], np.array(new)[app])}
            corr = tgt[[(s, round(v, 6)) in key for s, v in zip(tgt.stay_id, tgt.value)]]
            kb = [(s, c, a, round(v, 6)) in kept_b for s, c, v in zip(corr.stay_id, corr.charttime, corr.value)]
            ka = [(s, c, a, round(v, 6)) in kept_a for s, c, v in zip(corr.stay_id, corr.charttime, corr.value)]
            caught_before, caught_after = ~np.array(kb, bool), ~np.array(ka, bool)
        elif sc["kind"].startswith("malformed"):
            # before: silently dropped (no record of the error); after: the format audit flags it for re-entry
            caught_before = np.zeros(n, bool)
            caught_after = np.array([classify_text_value(v) == "malformed_number" for v in np.array(new, dtype=object)[app]])
        else:
            caught_before = np.zeros(n, bool)
            caught_after = np.array([not (VITAL_RANGE["TEMP_C"][0] <= v <= VITAL_RANGE["TEMP_C"][1]) for v in np.array(new)[app]])
        r["caught_before"] = int(caught_before.sum())
        r["caught_after"] = int(caught_after.sum())
        r["catch_rate_before"] = float(caught_before.mean()) if len(caught_before) else None
        r["catch_rate_after"] = float(caught_after.mean()) if len(caught_after) else None
        flag_store = {}
        for tag, X in (("before", X1), ("after", X1g)):
            out = {}
            if p0 is not None:
                p = risk(X)
                base = p0g if tag == "after" else p0
                dp = np.abs(p - base)[app]
                out.update(median_abs_dp=float(np.median(dp)), p90_abs_dp=float(np.quantile(dp, .9)),
                           new_high_risk=int(((p >= hi) & (base < hi))[app].sum()),
                           lost_high_risk=int(((p < hi) & (base >= hi))[app].sum()))
            if sc["kind"].startswith("malformed"):
                a = sc.get("analyte") or sc["label_name"]
                ref = X0g if tag == "after" else X0
                lost = X[f"{a}_latest"].isna() & ref[f"{a}_latest"].notna()
                stale = X[f"{a}_latest"].notna() & (X[f"{a}_latest"] != ref[f"{a}_latest"])
                out["became_unmeasured"] = int(lost[app].sum())
                out["older_value_used"] = int(stale[app].sum())
                out["warning_shown"] = int(caught_after.sum()) if tag == "after" else 0
            if sc["kind"] == "lab" and sc["analyte"] in RATIO_FEATS:
                a = sc["analyte"]
                ref = X0g if tag == "after" else X0
                fake = (X[f"{a}_latest"] >= 2) & ~(ref[f"{a}_latest"] >= 2)
                out["fake_liver_event"] = int(fake[app].sum())
                f_base, f_new = fc24(ref, a), fc24(X, a)
                if f_base is not None:
                    d = np.abs(np.log2(f_new) - np.log2(f_base))[app]
                    out["traj24_median_abs_log2_change"] = float(np.nanmedian(d)) if np.isfinite(d).any() else None
            if rule and sc["kind"] != "age":
                Lr = L1b if tag == "before" else L1g
                Cr = C1 if tag == "before" else C1g
                base_flags = fl0 if tag == "before" else _flags(rule, snap_fn, rr, lab_g0, ctx_g0, hist)
                fl = _flags(rule, snap_fn, rr, Lr, Cr, hist)
                flag_store[tag], flag_store["clean_" + tag] = fl, base_flags
                appr = app[:len(rr)]
                ch = [a != b for a, b, k in zip(fl, base_flags, appr) if k]
                out["agent1_flag_changed"] = int(sum(ch))
                out["agent1_rows"] = int(appr.sum())
            r[tag] = out
        if per_row:
            p1, p1g = risk(X1), risk(X1g)
            for i in np.where(app)[0]:
                rows_out[i].setdefault("scenarios", {})[sc["key"]] = {
                    "original": orig[i], "entered": new[i],
                    "risk_clean": None if p0 is None else float(p0[i]),
                    "risk_before": None if p1 is None else float(p1[i]),
                    "risk_after": None if p1g is None else float(p1g[i]),
                    **({"flags_clean": flag_store["clean_before"][i], "flags_before": flag_store["before"][i],
                        "flags_after": flag_store["after"][i]} if flag_store and i < len(rr) else {})}
        res["scenarios"].append(r)
    if per_row:
        res["rows"] = rows_out
    return res
