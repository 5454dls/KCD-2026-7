"""Deterministic rule baseline for Agent 1 (possible causes of a liver-test abnormality).

Purpose: a transparent comparator for the LLM agent and a working Agent 1 when
no approved LLM endpoint is configured. It reads ONLY the as-of snapshot built
by the notebook and cites only observations present in it, so every support
item is grounded by construction. It is not a diagnostic tool.

Pattern logic follows the ACG 2017 guideline on abnormal liver chemistries
(R ratio = (ALT/ULN)/(ALP/ULN): >=5 hepatocellular, <=2 cholestatic, otherwise
mixed) and the candidate categories of the project plan. Weights are a
pre-specified project choice, frozen before evaluation, not fitted to outcomes.
"""
from __future__ import annotations

# The eight causes shown as independent 0/1 flags (several may apply at once).
CAUSE_FLAGS = ["hypoxic_ischemic", "sepsis_cholestasis", "drug_related", "biliary", "congestive", "baseline_liver",
               "other_hepatic", "nonhepatic"]
FLAG_THRESHOLD = 1.0  # pre-specified: a category is flagged when its evidence score reaches this

HEPATOTOXIC = ["acetaminophen", "ceftriaxone", "amoxicillin", "clavulanate", "piperacillin", "isoniazid", "rifampin",
               "valpro", "phenytoin", "amiodarone", "azithromycin", "fluconazole", "voriconazole", "ketoconazole",
               "methotrexate", "atorvastatin", "simvastatin", "allopurinol", "carbamazepine", "nitrofurantoin",
               "sulfamethoxazole", "trimethoprim", "oxacillin", "nafcillin", "tmp", "linezolid", "propofol"]
PRESSORS = ["norepinephrine", "epinephrine", "vasopressin", "phenylephrine", "dopamine"]
VERSION = "rule_baseline_v1.2"  # v1.2 adds 0/1 flags for 8 causes; frozen before the full MIMIC-IV evaluation


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _sup(rec, reason):
    return {"analyte_or_label": rec.get("analyte") or rec.get("label"), "measured_at": rec.get("measured_at"), "reason": reason}


def rule_agent1(payload: dict) -> dict:
    labs = {r["analyte"]: r for r in payload.get("labs", []) if r.get("analyte")}
    ratio = lambda a: _num((labs.get(a) or {}).get("ratio"))
    val = lambda a: _num((labs.get(a) or {}).get("value"))
    ctx = payload.get("context", [])
    hist = payload.get("historical_codes", {}) or {}

    vit = lambda name: [c for c in ctx if c.get("kind") == "vital" and c.get("label") == name and _num(c.get("value")) is not None]
    maps = vit("MAP")
    low_map = min(maps, key=lambda c: _num(c["value"])) if maps else None
    temps = vit("TEMP_C")
    hot = max(temps, key=lambda c: _num(c["value"])) if temps else None
    pressor = [c for c in ctx if c.get("kind") == "infusion_record" and any(p in str(c.get("label", "")).lower() for p in PRESSORS)]
    meds = [c for c in ctx if c.get("kind") == "emar_record" and "administ" in str(c.get("value", "")).lower()
            and any(h in str(c.get("label", "")).lower() for h in HEPATOTOXIC)]

    alt, ast, alp, tb = ratio("ALT"), ratio("AST"), ratio("ALP"), ratio("TBIL")
    r_ratio = alt / alp if alt is not None and alp else None
    pattern = None if r_ratio is None else "간세포형" if r_ratio >= 5 else "담즙정체형" if r_ratio <= 2 else "혼합형"
    S = {}

    def add(code, score, support=None, against=None):
        s = S.setdefault(code, {"score": 0.0, "support": [], "against": []})
        s["score"] += score
        if support:
            s["support"].append(support)
        if against:
            s["against"].append(against)

    # hypoxic / ischaemic: very high aminotransferases with hypoperfusion evidence
    top_amino = max([(a, ratio(a)) for a in ("ALT", "AST") if ratio(a) is not None], key=lambda x: x[1], default=None)
    if top_amino and top_amino[1] >= 5:
        # aminotransferases alone are not ischaemia: >=10x counts as one unit, 5-10x as half; perfusion evidence adds
        add("hypoxic_ischemic", 1 if top_amino[1] >= 10 else .5, _sup(labs[top_amino[0]], f"{top_amino[0]} {top_amino[1]:.1f}×ULN"))
        if low_map is not None and _num(low_map["value"]) < 65:
            add("hypoxic_ischemic", 1, _sup(low_map, f"MAP 최저 {_num(low_map['value']):.0f} mmHg"))
        elif maps:
            add("hypoxic_ischemic", 0, against="48시간 내 MAP 65 mmHg 미만 기록 없음")
        else:
            add("hypoxic_ischemic", 0, against="혈압 기록 없음")
        lac = val("LACTATE")
        if lac is not None and lac >= 2:
            add("hypoxic_ischemic", 1 if lac >= 4 else .5, _sup(labs["LACTATE"], f"젖산 {lac:.1f} mmol/L"))
        if pressor:
            add("hypoxic_ischemic", 1, _sup(pressor[-1], "승압제 주입 기록"))
    # sepsis-associated cholestasis
    if (tb is not None and tb >= 2) or (alp is not None and alp >= 2 and (r_ratio or 0) <= 2):
        lead = "TBIL" if (tb or 0) >= (alp or 0) else "ALP"
        add("sepsis_cholestasis", 1.5, _sup(labs[lead], f"{lead} {ratio(lead):.1f}×ULN, 담즙정체 우세"))
        w = val("WBC")
        if w is not None and (w > 12 or w < 4):
            add("sepsis_cholestasis", 1, _sup(labs["WBC"], f"WBC {w:.1f}"))
        if hot is not None and _num(hot["value"]) >= 38.3:
            add("sepsis_cholestasis", .5, _sup(hot, f"체온 {_num(hot['value']):.1f}°C"))
        plt = val("PLATELET")
        if plt is not None and plt < 100:
            add("sepsis_cholestasis", .5, _sup(labs["PLATELET"], f"혈소판 {plt:.0f}"))
        add("sepsis_cholestasis", 0, against="배양·감염 진단 기록이 입력에 없음")
    # biliary obstruction pattern
    if alp is not None and alp >= 2 and tb is not None and tb >= 2 and (r_ratio or 0) <= 2:
        add("biliary", 1.2, _sup(labs["ALP"], f"ALP {alp:.1f}× + TBIL {tb:.1f}×ULN 동반"), "영상 판독이 입력에 없음")
    # drug-related: recorded administration of a commonly implicated drug in the window
    seen = {}
    for m in meds:  # one citation per drug, latest record
        seen[str(m.get("label")).lower()] = m
    for m in list(seen.values())[-2:]:
        add("drug_related", .9, _sup(m, f"{m.get('label')} 투여 기록"))
    if "drug_related" in S:
        add("drug_related", 0, against="48시간 창 밖의 투약과 다른 원인 배제 여부는 확인 불가")
    # baseline liver disease and synthetic function
    if hist.get("history_liver"):
        add("baseline_liver", 1.5, {"analyte_or_label": "history_liver", "measured_at": None, "reason": "이전 입원 간질환 코드"})
    for a, cond, txt in [("INR", lambda v: v >= 1.5, "INR {:.1f}"), ("ALBUMIN", lambda v: v < 3, "알부민 {:.1f}")]:
        v = val(a)
        if v is not None and cond(v) and "baseline_liver" in S:
            add("baseline_liver", .5, _sup(labs[a], txt.format(v)))
    # congestive: only when heart-failure history supports it
    if hist.get("history_heart_failure") and ((tb or 0) >= 2 or (alp or 0) >= 2 or (alt or 0) >= 2):
        add("congestive", 1.2, {"analyte_or_label": "history_heart_failure", "measured_at": None, "reason": "이전 입원 심부전 코드"},
            "심초음파·중심정맥압 정보 없음")
    # non-hepatic sources
    ck = ratio("CK")
    # Muscle injury raises AST (and ALT); AST >= ALT with CK >= 5xULN points to a non-hepatic source.
    if ast is not None and alt is not None and ck is not None and ck >= 5 and ast >= .8 * alt:
        add("nonhepatic", 2 if ck >= 20 else 1.5, _sup(labs["CK"], f"CK {ck:.1f}×ULN, AST {ast:.1f}× ≥ ALT {alt:.1f}× (근육 기원 가능)"),
            "근육 손상과 간손상이 동시에 있을 수 있음")
    if tb is not None and tb >= 2 and (alt or 0) < 1.5 and (alp or 0) < 1.5:
        add("nonhepatic", 1, _sup(labs["TBIL"], "빌리루빈 단독 상승 (용혈·포합 장애 감별)"))
    # other hepatocellular causes (viral, autoimmune, ...): hepatocellular pattern without the explanations above
    explained = any(S.get(c, {}).get("score", 0) >= FLAG_THRESHOLD for c in ("hypoxic_ischemic", "drug_related", "nonhepatic"))
    if r_ratio is not None and r_ratio >= 5 and (alt or 0) >= 3 and not explained:
        add("other_hepatic", 1.0, _sup(labs["ALT"], f"간세포형 (R {r_ratio:.1f}), 저관류·약물·근육으로 설명되지 않음"),
            "바이러스·자가면역 검사는 입력에 없음")

    ranked = sorted(((c, v) for c, v in S.items() if v["score"] > 0), key=lambda x: -x[1]["score"])[:3]
    missing = [a for a in ("ALT", "ALP", "TBIL") if a not in labs]
    abstain = not ranked or ranked[0][1]["score"] < 1.5
    if abstain and len(ranked) < 3:
        ranked.append(("unknown", {"support": [], "against": ["근거가 약하거나 서로 충돌해 판단 보류"]}))
    unc = [f"R 비 {r_ratio:.1f} ({pattern})" if r_ratio is not None else "R 비 계산 불가 (ALT·ALP 동시 측정 없음)"]
    if missing:
        unc.append("최근 측정 없음: " + ", ".join(missing))
    unc.append("규칙 기반 기준 모델. 노트·영상·배양 정보는 사용하지 않음.")
    flags = {c: {"present": int(S.get(c, {}).get("score", 0) >= FLAG_THRESHOLD),
                 "score": round(float(S.get(c, {}).get("score", 0)), 2),
                 "support": S.get(c, {}).get("support", []), "against": S.get(c, {}).get("against", [])} for c in CAUSE_FLAGS}
    return {"candidates": [{"cause_code": c, "support": v["support"], "against_or_missing": v["against"]} for c, v in ranked],
            "flags": flags, "uncertainty": " · ".join(unc), "abstain": abstain or not any(f["present"] for f in flags.values()),
            "model": VERSION}
