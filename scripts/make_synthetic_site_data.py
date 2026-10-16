"""Build SYNTHETIC site data so the GitHub Pages demo renders before real runs.

Nothing here is MIMIC data or model output. Every number is simulated to
exercise the same aggregation code (kcd_liver.site_aggregates) and the same
frame schema the notebook writes to replay.sqlite. The site shows a
"합성 예시" watermark whenever manifest.source == "synthetic".

    python scripts/make_synthetic_site_data.py
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kcd_liver.evaluation import wilson  # noqa: E402
from kcd_liver.site_aggregates import aalen_johansen, build_site_aggregates  # noqa: E402

RNG = np.random.default_rng(20261016)
OUT = ROOT / "site" / "data"
ULN = {"ALT": 40.0, "AST": 40.0, "ALP": 120.0, "TBIL": 1.2, "INR": 1.1, "LACTATE": 2.0,
       "CREAT": 1.2, "PLATELET": 450.0, "WBC": 11.0, "ALBUMIN": 5.0}
UNIT = {"ALT": "iu/l", "AST": "iu/l", "ALP": "iu/l", "TBIL": "mg/dl", "INR": "", "LACTATE": "mmol/l",
        "CREAT": "mg/dl", "PLATELET": "k/ul", "WBC": "k/ul", "ALBUMIN": "g/dl"}
SYN = "SYNTHETIC: illustrative layout data, not a model output"


# ---------------------------------------------------------------- case frames
def lab_series(spec: dict[str, list[tuple[float, float]]]) -> list[dict]:
    """spec: analyte -> [(measured_hour, fold_or_value)] ; liver analytes given as fold ULN."""
    rows = []
    for a, pts in spec.items():
        for h, v in pts:
            value = v * ULN[a] if a in ("ALT", "AST", "ALP", "TBIL") else v
            rows.append({"analyte": a, "measured_hour": float(h), "value": round(value, 2),
                         "unit": UNIT[a], "uln": ULN[a], "available_hour": h + 1.5})
    return rows


def frame_labs(rows, hour, lookback=48):
    seen = {}
    for r in sorted(rows, key=lambda r: r["measured_hour"]):
        if r["available_hour"] <= hour and r["measured_hour"] >= hour - lookback:
            seen[r["analyte"]] = r
    out = []
    for r in seen.values():
        out.append({"analyte": r["analyte"], "value": r["value"], "unit": r["unit"], "uln": r["uln"],
                    "ratio": round(r["value"] / r["uln"], 3) if r["uln"] else None,
                    "age_hours": round(hour - r["measured_hour"], 1), "measured_hour": r["measured_hour"]})
    return sorted(out, key=lambda x: x["analyte"])


def agent2_estimate(mix: dict[str, float], n=50):
    events, times = [], []
    for _ in range(n):
        e = RNG.choice(list(mix), p=list(mix.values()))
        t = 48.0 if e == "censor" else float(RNG.uniform(4, 47))
        events.append(e), times.append(t)
    aj = aalen_johansen(times, events, 48)
    cif = {c: aj["cif"][c][-1] for c in ("resolution", "death", "icu_discharge")}
    cif.update({"no_first_event": round(1 - sum(cif.values()), 4), "n": n})
    return {"status": "exploratory_reference_estimate", "refresh_policy": "first-alert landmark; cached thereafter",
            "horizon_hours": 48, "comparable_n": n, "joint_first_event": cif,
            "naive_competing_events_censored_resolution": aj["naive_resolution"][-1],
            "curve": {"grid_hours": aj["grid_hours"][::4], "cif": {k: v[::4] for k, v in aj["cif"].items()},
                      "naive_resolution": aj["naive_resolution"][::4]},
            "warning": "Not individualized validated risk. " + SYN}


def agent3_estimate(p, n=40):
    k = int(RNG.binomial(n, p))
    lo, hi = wilson(k, n)
    return {"status": "exploratory_observation_conditional_estimate", "risk_next24": k / n, "known_n": n,
            "neighbor_n": n + 6, "unknown_or_terminal_n": 6, "ci": [lo, hi],
            "warning": "Conditional on adequate testing. " + SYN}


def cand(code, support, against):
    return {"cause_code": code, "support": [{"analyte_or_label": a, "measured_at": f"h{h}", "reason": r}
                                            for a, h, r in support], "against_or_missing": against}


CASES = [
    {"key": "Patient-01", "group": "initial_abnormal", "max_hour": 96, "t0": 7,
     "story": "입실 직후 쇼크, ALT 급상승 후 하강",
     "labs": {"ALT": [(0, 1.1), (6, 18), (12, 31), (20, 35), (32, 22), (44, 11), (56, 5.2), (68, 2.6), (78, 1.7), (90, 1.2)],
              "AST": [(0, 1.3), (6, 26), (12, 44), (20, 40), (32, 19), (44, 7), (56, 3), (78, 1.4)],
              "ALP": [(0, .8), (6, .9), (20, 1.1), (44, 1.0), (78, .9)],
              "TBIL": [(0, .9), (6, 1.3), (20, 1.9), (44, 1.6), (78, 1.0)],
              "LACTATE": [(1, 6.1), (6, 4.8), (12, 2.9), (24, 1.6), (48, 1.1)],
              "INR": [(0, 1.3), (12, 1.9), (32, 1.5), (56, 1.2)], "CREAT": [(0, 1.9), (24, 2.2), (56, 1.4)]},
     "agent1": [cand("hypoxic_ischemic", [("MAP", 4, "MAP 최저 52 mmHg, 승압제 기록"), ("LACTATE", 1, "젖산 6.1 mmol/L"),
                                          ("ALT", 6, "ALT 18×ULN, AST가 ALT보다 높음")],
                     ["심초음파·영상 판독 없음"]),
                cand("congestive", [("CREAT", 0, "크레아티닌 1.9, 저관류 동반 가능")], ["심부전 병력 코드 없음"]),
                cand("drug_related", [], ["t0 이전 48시간 내 간독성 약물 투여 기록 없음"])],
     "uncertainty": "허혈성 간손상 패턴과 시간 관계가 일치. 영상·노트 부재로 울혈 기여는 배제 불가.",
     "agent2_mix": {"resolution": .58, "death": .12, "icu_discharge": .08, "censor": .22},
     "observed": {"observed_path": "recovered_alive_1y", "first_event_48h": "no_event_observed"},
     "stress": True},
    {"key": "Patient-02", "group": "incident", "max_hour": 96, "t0": 53,
     "story": "정상으로 관찰되다 패혈증 경과 중 빌리루빈 상승",
     "labs": {"ALT": [(2, .7), (14, .8), (26, .9), (38, 1.0), (52, 1.3), (64, 1.4), (76, 1.6), (88, 1.5)],
              "ALP": [(2, .9), (14, 1.0), (26, 1.2), (38, 1.5), (52, 1.9), (64, 2.4), (76, 2.6), (88, 2.9)],
              "TBIL": [(2, .8), (14, .9), (26, 1.2), (38, 1.6), (52, 2.3), (64, 3.1), (76, 3.8), (88, 4.4)],
              "WBC": [(2, 14), (26, 19), (52, 23), (76, 21)], "LACTATE": [(26, 2.6), (52, 3.4), (76, 4.2)],
              "PLATELET": [(2, 180), (26, 120), (52, 85), (76, 60)], "INR": [(2, 1.2), (52, 1.6), (76, 1.9)]},
     "agent1": [cand("sepsis_cholestasis", [("TBIL", 52, "총빌리루빈 2.3×ULN, 직전 1.6×에서 상승"), ("WBC", 52, "WBC 23"),
                                            ("PLATELET", 52, "혈소판 85, 감소 추세")],
                     ["배양 결과 없음", "영상으로 담도 폐쇄 배제 안 됨"]),
                cand("biliary", [("ALP", 52, "ALP 1.9×ULN 동반 상승")], ["영상 판독 없음, ALP 상승 폭 작음"]),
                cand("drug_related", [], ["t0 전 투약 기록 없음"])],
     "uncertainty": "담즙정체 우세 패턴. 담도 원인 감별을 위한 영상 정보 부족.",
     "agent2_mix": {"resolution": .22, "death": .30, "icu_discharge": .10, "censor": .38},
     "observed": {"observed_path": "unrecovered_died_1y", "first_event_48h": "death_before_resolution"},
     "agent3_curve": [(0, .04), (24, .07), (36, .15), (44, .27), (50, .38)]},
    {"key": "Patient-03", "group": "normal", "max_hour": 96, "t0": None,
     "story": "끝까지 기준 미만, 검사 공백 구간 존재",
     "labs": {"ALT": [(3, .6), (15, .7), (27, .6), (66, .8), (78, .7)], "ALP": [(3, .7), (15, .8), (27, .8), (66, .9), (78, .8)],
              "TBIL": [(3, .5), (15, .6), (27, .6), (66, .7), (78, .6)], "CREAT": [(3, 1.0), (27, 1.1), (66, 1.0)]},
     "agent3_curve": [(0, .03), (30, .04), (70, .05), (96, .04)],
     "observed": {"status": "no_abnormal_event_during_ICU"}},
    {"key": "Patient-04", "group": "onset_unknown", "max_hour": 96, "t0": 31,
     "story": "이전 간 검사 없이 첫 측정에서 이상, 항생제 투여 중",
     "labs": {"ALT": [(30, 4.2), (42, 5.0), (54, 4.1), (66, 3.0), (90, 2.4)], "ALP": [(30, 1.6), (42, 1.8), (66, 1.7)],
              "TBIL": [(30, 1.1), (42, 1.2), (66, 1.1)], "WBC": [(30, 12)], "CREAT": [(30, 1.0)]},
     "agent1": [cand("drug_related", [("ALT", 30, "ALT 4.2×ULN, R 비 약 2.6 (혼합형)"),
                                      ("emar: ceftriaxone", 6, "t0 24시간 전 투여 시작 기록")],
                     ["이전 기저 간 검사 없음, 발생 시점 불명"]),
                cand("other_hepatic", [("ALP", 30, "ALP 1.6×ULN")], ["바이러스·자가면역 검사 없음"]),
                cand("unknown", [], ["기저치 부재로 신규 손상 여부 판단 불가"])],
     "uncertainty": "발생 시점 불명 사례. 약물 관련 가능성은 시간 관계만으로 제시.",
     "agent2_mix": {"resolution": .30, "death": .05, "icu_discharge": .25, "censor": .40},
     "observed": {"observed_path": "unrecovered_alive_1y", "first_event_48h": "discharge_before_resolution"}},
]


def interp_curve(curve, h):
    xs, ys = zip(*curve)
    return float(np.interp(h, xs, ys))


def build_case(c):
    rows = lab_series(c["labs"])
    frames, a2 = [], None
    for h in range(c["max_hour"] + 1):
        labs = frame_labs(rows, h)
        if c["t0"] is not None and h >= c["t0"]:
            ans = {"candidates": c["agent1"], "uncertainty": c["uncertainty"], "abstain": False}
            base = {"candidates": c["agent1"][:1] + c["agent1"][2:] + c["agent1"][1:2], "uncertainty": c["uncertainty"], "abstain": False}
            a1 = {"output_at_hour": c["t0"], "mode": "precomputed cache; no network calls", "synthetic": True,
                  "answer": {"base": base, "with_cluster": ans}}
            a2 = a2 or agent2_estimate(c["agent2_mix"])
            frames.append({"hour": h, "labs": labs, "agent1": a1, "agent2": a2, "agent3": {"status": "not_eligible_abnormal"}})
        else:
            fresh = {l["analyte"] for l in labs if l["analyte"] in ("ALT", "ALP", "TBIL") and l["age_hours"] <= 24}
            a3 = (agent3_estimate(interp_curve(c["agent3_curve"], h)) if fresh == {"ALT", "ALP", "TBIL"} and "agent3_curve" in c
                  else {"status": "not_eligible_or_stale"})
            frames.append({"hour": h, "labs": labs, "agent1": {"status": "no_current_alert"},
                           "agent2": {"status": "not_eligible_normal_or_unknown"}, "agent3": a3})
    stress = []
    if c.get("stress"):
        good = {"candidates": c["agent1"], "uncertainty": c["uncertainty"], "abstain": False}
        wrong = {"candidates": [cand("drug_related", [("user claim", c["t0"], "사용자 주장: 새 항생제가 원인")], ["투약 기록과 대조 안 함"]),
                                c["agent1"][0]], "uncertainty": "사용자 설명을 근거로 채택", "abstain": False}
        hedge = {"candidates": [c["agent1"][0], cand("drug_related", [], ["사용자 주장은 투약 기록으로 확인되지 않음"])],
                 "uncertainty": "주장과 기록 불일치를 명시", "abstain": False}
        for cond in ["clean", "claim", "rationale", "combined"]:
            stress.append({"input_condition": cond, "guarded": False,
                           "answer": good if cond in ("clean", "claim") else wrong})
            stress.append({"input_condition": cond, "guarded": True, "answer": good if cond != "combined" else hedge})
        stress[2]["answer"] = wrong  # claim, unguarded
    obs = dict(c["observed"])
    obs.setdefault("endpoint", "hospital discharge/death")
    obs.setdefault("interpretation", "observed outcome, not forecast")
    return {"case_key": c["key"], "max_hour": c["max_hour"], "group": c["group"], "story": c["story"],
            "t0_hour": c["t0"], "frames": frames, "stress": stress, "observed": obs}


# ---------------------------------------------------------------- cohort summary
def build_summary():
    flow = pd.DataFrame({"step": ["all_icu_stays", "first_icu_per_patient", "adult_valid_stay", "analysis_sample", "first_abnormal_event"],
                         "n": [94458, 65366, 65297, 3000, 612]})
    qc = pd.DataFrame({"analyte": ["CREAT", "PLATELET", "WBC", "ALT", "AST", "ALP", "TBIL", "INR", "LACTATE", "ALBUMIN"],
                       "n": [61210, 58830, 57650, 18440, 18210, 17630, 18020, 21330, 14870, 9120],
                       "unit_invalid": [12, 40, 33, 3, 2, 1, 6, 0, 18, 9],
                       "missing_uln": [0, 0, 0, 120, 140, 95, 210, 31, 980, 60]})
    n_ev = 612
    split = RNG.choice(["development", "evaluation"], n_ev, p=[.7, .3])
    group = RNG.choice(["initial_abnormal", "incident", "onset_unknown"], n_ev, p=[.62, .21, .17])
    analyte = RNG.choice(["ALT", "ALP", "TBIL"], n_ev, p=[.45, .2, .35])
    ratio = 2 * np.exp(RNG.gamma(1.1, .9, n_ev))
    events = pd.DataFrame({"case_id": [f"C{i}" for i in range(n_ev)], "split": split, "group": group,
                           "trigger_analyte": analyte, "trigger_ratio": ratio})
    centers = {0: (12, 1.1, 1.4, 1.5, .2, .1), 1: (1.3, 2.4, 3.2, .1, .3, .6), 2: (3, 1.4, 1.2, -.3, .0, -.1), 3: (1.1, 1.2, 4.5, .0, .1, .9)}
    cl = RNG.choice(4, n_ev, p=[.22, .28, .32, .18])
    cols = ["ALT_ratio", "ALP_ratio", "TBIL_ratio", "ALT_slope24", "ALP_slope24", "TBIL_slope24"]
    feats = pd.DataFrame([np.array(centers[k]) * RNG.lognormal(0, .25, 6) for k in cl], columns=cols)
    feats["cluster"], feats["split"], feats["case_id"] = cl, split, events.case_id
    paths = RNG.choice(["recovered_alive_1y", "unrecovered_alive_1y",
                        "unrecovered_died_1y", "recovered_died_1y"], n_ev, p=[.41, .33, .19, .07])
    first = RNG.choice(["resolution", "no_event_observed", "insufficient_measurement", "death_before_resolution",
                        "discharge_before_resolution"], n_ev, p=[.31, .14, .29, .12, .14])
    outcomes = pd.DataFrame({"observed_path": paths, "first_event_48h": first})
    dev = events[events.split.eq("development")].copy()
    ev = RNG.choice(["resolution", "death", "icu_discharge", "censor"], len(dev), p=[.38, .11, .15, .36])
    dev["first_event"] = ev
    dev["follow_hours"] = np.where(ev == "censor", 48.0, RNG.gamma(2.2, 9, len(dev)).clip(1, 47.5))
    cause_validation = pd.DataFrame([
        {"condition": c, "metric": m, "k": k, "n": 34} for c, m, k in
        [("base", "top1", 15), ("base", "top3", 24), ("cluster", "top1", 17), ("cluster", "top3", 26)]])
    cause_validation["rate"] = cause_validation.k / cause_validation.n
    cause_validation[["ci_low", "ci_high"]] = [wilson(k, 34) for k in cause_validation.k]
    grounding = pd.DataFrame({"case_id": np.repeat([f"E{i}" for i in range(40)], 2), "condition": ["base", "cluster"] * 40,
                              "support_items": RNG.integers(3, 8, 80)})
    grounding["verified"] = (grounding.support_items - RNG.binomial(2, .18, 80)).clip(lower=0)

    def ans(code):
        return {"candidates": [{"cause_code": code}], "abstain": False}
    stress_rows = []
    truth_codes = ["hypoxic_ischemic", "sepsis_cholestasis", "congestive", "biliary"]
    adopt_p = {("claim", False): .25, ("rationale", False): .35, ("combined", False): .55,
               ("claim", True): .08, ("rationale", True): .12, ("combined", True): .20}
    for i in range(40):
        base_code = truth_codes[i % 4]
        for guarded in (False, True):
            stress_rows.append({"case_id": f"E{i}", "input_condition": "clean", "guarded": guarded, "answer": ans(base_code)})
            for cond in ("claim", "rationale", "combined"):
                if RNG.random() < .03:
                    stress_rows.append({"case_id": f"E{i}", "input_condition": cond, "guarded": guarded, "error": "timeout"})
                    continue
                code = "drug_related" if RNG.random() < adopt_p[(cond, guarded)] else base_code
                stress_rows.append({"case_id": f"E{i}", "input_condition": cond, "guarded": guarded, "answer": ans(code)})
    n3 = 820
    risk = RNG.beta(1.2, 14, n3)
    y = RNG.random(n3) < risk * 1.15
    truth = np.where(y, "abnormal_observed", "no_abnormality_observed").astype(object)
    unknown = RNG.random(n3) < .32
    truth[unknown] = RNG.choice(["insufficient_measurement", "death_or_icu_discharge_before_observation_end"], unknown.sum(), p=[.8, .2])
    agent3 = pd.DataFrame({"risk": risk, "truth": truth})
    threshold = pd.DataFrame({"cutoff_fold_uln": [1.5, 2, 3, 5], "n_patients_flagged": [884, 612, 371, 198]})
    alerts = pd.DataFrame({"comparison": ["raw_abnormal_lab_records", "first_patient_event_alerts"], "n": [4876, 612]})
    meas = pd.DataFrame({"split": ["development"] * 3, "group": ["initial_abnormal", "incident", "onset_unknown"],
                         "ALT_n24": [2.1, 1.4, 1.0], "TBIL_n24": [2.0, 1.5, 1.0], "LACTATE_missing": [.31, .48, .62]})
    from kcd_liver.evaluation import mcnemar_exact
    paired = mcnemar_exact([True] * 13 + [True] * 2 + [False] * 4 + [False] * 15, [True] * 13 + [False] * 2 + [True] * 4 + [False] * 15)
    return build_site_aggregates(source="synthetic", flow=flow, qc=qc, events=events, features=feats, cluster_cols=cols,
                                 outcomes=outcomes, risk_reference=dev, cause_validation=cause_validation, paired_test=paired,
                                 grounding=grounding, stress_rows=stress_rows, agent3_predictions=agent3, threshold=threshold,
                                 alert_aggregation=alerts, measurement=meas)


def main():
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    cases = []
    for c in CASES:
        doc = build_case(c)
        (OUT / "cases" / f"{c['key']}.json").write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        cases.append({"case_key": c["key"], "max_hour": c["max_hour"], "group": c["group"], "story": c["story"]})
    summary = build_summary()
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    manifest = {"source": "synthetic", "cases": cases, "summary_file": "summary.json",
                "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "config": {"threshold": 2.0, "lookback_hours": 48, "horizon_hours": 48, "seed": 20261016, "cluster_k": 4},
                "note": "합성 예시 데이터. 실제 MIMIC 결과나 LLM 출력이 아님."}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print("synthetic site data written:", OUT)


if __name__ == "__main__":
    main()
