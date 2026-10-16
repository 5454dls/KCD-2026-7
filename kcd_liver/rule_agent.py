"""Deterministic rule baseline for Agent 1 (possible causes of a liver-test abnormality)."""
from __future__ import annotations

import re

CAUSE_FLAGS = ["hypoxic_hepatitis", "congestive_hepatopathy", "sepsis_cholestasis", "dili", "pn_associated",
               "biliary_obstruction_cholangitis", "cholecystitis", "hemolysis", "viral_hepatitis", "others"]
NOT_ASSESSABLE = {"cholecystitis": "국소 소견(Murphy 징후)과 영상이 구조화 자료에 없어 판단 불가 (TG18)"}
FLAG_THRESHOLD = 1.0

HEPATOTOXIC = ["acetaminophen", "ceftriaxone", "amoxicillin", "clavulanate", "piperacillin", "isoniazid", "rifampin",
               "valpro", "phenytoin", "amiodarone", "azithromycin", "fluconazole", "voriconazole", "ketoconazole",
               "methotrexate", "atorvastatin", "simvastatin", "allopurinol", "carbamazepine", "nitrofurantoin",
               "sulfamethoxazole", "trimethoprim", "oxacillin", "nafcillin", "tmp", "linezolid", "propofol"]
PRESSORS = ["norepinephrine", "epinephrine", "vasopressin", "phenylephrine", "dopamine"]
PN_PATTERN = r"parenteral|\btpn\b|\bppn\b|clinimix|kabiven|intralipid|smoflipid|lipid emulsion|amino acid"
SERO_ACTIVE = r"surface antigen|hbsag|igm|viral load|rna|dna|e antigen"
SERO_EXPOSURE = r"hepatitis c virus antibody|hcv ab|anti-hcv|core antibody"
VERSION = "rule_baseline_v2.1"


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _sup(rec, reason):
    return {"analyte_or_label": rec.get("analyte") or rec.get("label"), "measured_at": rec.get("measured_at"), "reason": reason}


def serology_positive(value) -> bool:
    v = str(value or "").strip().lower()
    if not v or re.search(r"non[- ]?reactive|negative|not detected|undetected|<\s*\d|neg\b", v):
        return False
    n = _num(v)
    return bool(re.search(r"pos|reactive|detected", v)) or (n is not None and n > 0)


def exposures(ctx_rows, as_of) -> dict:
    """Admission-level exposures known at as_of (beyond the 48 h snapshot window)."""
    import pandas as pd
    t = pd.Timestamp(as_of)
    pn, sero = [], []
    for r in ctx_rows:
        when = pd.Timestamp(r.get("available_at") or r.get("measured_at") or r.get("charttime"))
        if pd.isna(when) or when > t:
            continue
        lab = str(r.get("label", ""))
        if r.get("kind") in ("infusion_record", "emar_record") and re.search(PN_PATTERN, lab, re.I):
            if r.get("kind") == "emar_record" and "administ" not in str(r.get("value", "")).lower():
                continue
            pn.append(pd.Timestamp(r.get("charttime") or r.get("measured_at")))
        if r.get("kind") == "serology_record":
            sero.append({"label": lab, "value": str(r.get("value")), "measured_at": pd.Timestamp(r.get("charttime") or r.get("measured_at")).isoformat()})
    out = {"serology": sero[-20:]}
    if pn:
        first = min(pn)
        out.update(pn_first_at=first.isoformat(), pn_days=round((t - first).total_seconds() / 86400, 2),
                   pn_label="parenteral nutrition record")
    return out


def rule_agent1(payload: dict) -> dict:
    labs = {r["analyte"]: r for r in payload.get("labs", []) if r.get("analyte")}
    ratio = lambda a: _num((labs.get(a) or {}).get("ratio"))
    val = lambda a: _num((labs.get(a) or {}).get("value"))
    ctx = payload.get("context", [])
    hist = payload.get("historical_codes", {}) or {}
    ex = payload.get("exposures", {}) or {}

    vit = lambda name: [c for c in ctx if c.get("kind") == "vital" and c.get("label") == name and _num(c.get("value")) is not None]
    maps = vit("MAP")
    low_map = min(maps, key=lambda c: _num(c["value"])) if maps else None
    temps = vit("TEMP_C")
    hot = max(temps, key=lambda c: _num(c["value"])) if temps else None
    pressor = [c for c in ctx if c.get("kind") == "infusion_record" and any(p in str(c.get("label", "")).lower() for p in PRESSORS)]
    meds = [c for c in ctx if c.get("kind") == "emar_record" and "administ" in str(c.get("value", "")).lower()
            and any(h in str(c.get("label", "")).lower() for h in HEPATOTOXIC)]

    alt, ast, alp, tb, ggt = ratio("ALT"), ratio("AST"), ratio("ALP"), ratio("TBIL"), ratio("GGT")
    r_ratio = alt / alp if alt is not None and alp else None
    pattern = None if r_ratio is None else "간세포형" if r_ratio >= 5 else "담즙정체형" if r_ratio <= 2 else "혼합형"
    cholestatic = (tb is not None and tb >= 2) or (alp is not None and alp >= 2 and (r_ratio or 0) <= 2)
    S = {}

    def add(code, score, support=None, against=None):
        s = S.setdefault(code, {"score": 0.0, "support": [], "against": []})
        s["score"] += score
        if support:
            s["support"].append(support)
        if against:
            s["against"].append(against)

    w = val("WBC")
    infl = []
    if w is not None and (w > 10 or w < 4):
        infl.append(_sup(labs["WBC"], f"WBC {w:.1f}"))
    if hot is not None and _num(hot["value"]) > 38:
        infl.append(_sup(hot, f"체온 {_num(hot['value']):.1f}°C"))

    top_amino = max([(a, ratio(a)) for a in ("ALT", "AST") if ratio(a) is not None], key=lambda x: x[1], default=None)
    if top_amino and top_amino[1] >= 5:
        add("hypoxic_hepatitis", 1 if top_amino[1] >= 10 else .5, _sup(labs[top_amino[0]], f"{top_amino[0]} {top_amino[1]:.1f}×ULN"))
        if low_map is not None and _num(low_map["value"]) < 65:
            add("hypoxic_hepatitis", 1, _sup(low_map, f"MAP 최저 {_num(low_map['value']):.0f} mmHg"))
        else:
            add("hypoxic_hepatitis", 0, against="48시간 내 MAP 65 mmHg 미만 기록 없음" if maps else "혈압 기록 없음")
        lac = val("LACTATE")
        if lac is not None and lac >= 2:
            add("hypoxic_hepatitis", 1 if lac >= 4 else .5, _sup(labs["LACTATE"], f"젖산 {lac:.1f} mmol/L"))
        if pressor:
            add("hypoxic_hepatitis", 1, _sup(pressor[-1], "승압제 주입 기록"))
    liver_up = (tb or 0) >= 2 or (alp or 0) >= 2 or (alt or 0) >= 2
    if liver_up and hist.get("history_heart_failure"):
        add("congestive_hepatopathy", 1.2, {"analyte_or_label": "history_heart_failure", "measured_at": None,
                                            "reason": "이전 입원 심부전 코드"}, "심초음파·중심정맥압 정보 없음")
    bnp = val("NTPROBNP")
    if liver_up and bnp is not None:
        age = _num((payload.get("demographics") or {}).get("age"))
        cut = 900 if age is None else 450 if age < 50 else 900 if age <= 75 else 1800
        if bnp >= cut:
            add("congestive_hepatopathy", 1.0, _sup(labs["NTPROBNP"], f"NT-proBNP {bnp:.0f} pg/mL (기준 {cut})"))
        else:
            add("congestive_hepatopathy", 0, against=f"NT-proBNP {bnp:.0f} pg/mL, 기준 {cut} 미만")
    if cholestatic:
        lead = "TBIL" if (tb or 0) >= (alp or 0) else "ALP"
        add("sepsis_cholestasis", .5, _sup(labs[lead], f"{lead} {ratio(lead):.1f}×ULN, 담즙정체 우세"))
        for s in infl:
            add("sepsis_cholestasis", .5, s)
        plt = val("PLATELET")
        if plt is not None and plt < 100:
            add("sepsis_cholestasis", .5, _sup(labs["PLATELET"], f"혈소판 {plt:.0f}"))
        if pressor:
            add("sepsis_cholestasis", .5, _sup(pressor[-1], "승압제 (패혈성 쇼크 가능)"))
        add("sepsis_cholestasis", 0, against="배양·감염 진단 기록이 입력에 없음")
    seen = {}
    for m in meds:
        seen[str(m.get("label")).lower()] = m
    for m in list(seen.values())[-2:]:
        add("dili", .9, _sup(m, f"{m.get('label')} 투여 기록"))
    if "dili" in S:
        add("dili", 0, against="48시간 창 밖의 투약·잠복기·RUCAM은 확인 불가")
    pnd = _num(ex.get("pn_days"))
    if pnd is not None:
        rec = {"analyte_or_label": ex.get("pn_label", "parenteral nutrition record"), "measured_at": ex.get("pn_first_at")}
        add("pn_associated", 1 if pnd >= 7 else .5 if pnd >= 3 else .25, dict(rec, reason=f"정맥영양 노출 {pnd:.1f}일"))
        if cholestatic or (r_ratio is not None and r_ratio < 5):
            add("pn_associated", .5, {"analyte_or_label": "TBIL" if "TBIL" in labs else "ALP",
                                      "measured_at": (labs.get("TBIL") or labs.get("ALP") or {}).get("measured_at"),
                                      "reason": "담즙정체·혼합 패턴"})
        add("pn_associated", 0, against="정맥영양 조성·열량은 입력에 없음")
    tval = val("TBIL")
    jaundice = tval is not None and tval >= 2
    abn = [(a, ratio(a)) for a in ("ALP", "GGT", "AST", "ALT") if ratio(a) is not None and ratio(a) > 1.5]
    obstructive = alp is not None and alp >= 2 and tb is not None and tb >= 2 and (r_ratio or 0) <= 2
    if (jaundice or abn) and (infl or obstructive):
        b = labs["TBIL"] if jaundice else labs[abn[0][0]]
        add("biliary_obstruction_cholangitis", .5, _sup(b, f"TG18 B 담즙정체: TBIL {tval:.1f} mg/dL" if jaundice else f"TG18 B: {abn[0][0]} {abn[0][1]:.1f}×ULN"))
        if infl:
            add("biliary_obstruction_cholangitis", .5, infl[0])
        if obstructive:
            add("biliary_obstruction_cholangitis", .5, _sup(labs["ALP"], f"ALP {alp:.1f}× + TBIL {tb:.1f}×ULN 폐쇄 패턴"))
        dbil, tbv = val("DBIL"), tval
        if dbil is not None and tbv and dbil / tbv >= .5:
            add("biliary_obstruction_cholangitis", .25, _sup(labs["DBIL"], f"직접 빌리루빈 {dbil / tbv:.0%}"))
        add("biliary_obstruction_cholangitis", 0, against="TG18 C 영상(담관 확장)이 입력에 없음: '의심' 수준까지만")
    dbil = val("DBIL")
    if tval is not None and tval >= 1.5:
        if dbil is not None and dbil / tval < .3:
            add("hemolysis", 1, _sup(labs["DBIL"], f"간접 빌리루빈 우세 (직접 {dbil / tval:.0%})"))
        elif dbil is not None:
            add("hemolysis", 0, against=f"직접 빌리루빈 {dbil / tval:.0%}: 포합형 우세")
        hp = val("HAPTO")
        if hp is not None and hp < 30:
            add("hemolysis", 1, _sup(labs["HAPTO"], f"합토글로빈 {hp:.0f} mg/dL"))
        ldh = ratio("LDH")
        if ldh is not None and ldh >= 1.5:
            add("hemolysis", .5, _sup(labs["LDH"], f"LDH {ldh:.1f}×ULN"))
        hb_drop = _num(payload.get("hemoglobin_drop_48h"))
        if hb_drop is not None and hb_drop >= 2:
            add("hemolysis", .5, {"analyte_or_label": "HEMOGLOBIN", "measured_at": (labs.get("HEMOGLOBIN") or {}).get("measured_at"),
                                  "reason": f"48시간 내 혈색소 {hb_drop:.1f} g/dL 감소"})
        if "hemolysis" in S and "HAPTO" not in labs:
            add("hemolysis", 0, against="합토글로빈 측정 없음")
    for s in ex.get("serology", []):
        if serology_positive(s.get("value")):
            active = bool(re.search(SERO_ACTIVE, s["label"], re.I)) and not re.search(r"surface antibody|anti-hbs", s["label"], re.I)
            weak = bool(re.search(SERO_EXPOSURE, s["label"], re.I))
            if active or weak:
                add("viral_hepatitis", 1.5 if active else .5,
                    {"analyte_or_label": s["label"], "measured_at": s.get("measured_at"), "reason": f"{s['label']} 양성 ({s['value']})"})
    if hist.get("history_viral_hepatitis"):
        add("viral_hepatitis", .5, {"analyte_or_label": "history_viral_hepatitis", "measured_at": None, "reason": "이전 입원 만성 바이러스 간염 코드"})
    if "viral_hepatitis" in S and (alt or 0) >= 10:
        add("viral_hepatitis", .5, _sup(labs["ALT"], f"ALT {alt:.1f}×ULN"))
    ck = ratio("CK")
    if ast is not None and alt is not None and ck is not None and ck >= 5 and ast >= .8 * alt:
        add("others", 1.5, _sup(labs["CK"], f"근육 기원 가능: CK {ck:.1f}×ULN, AST {ast:.1f}× ≥ ALT {alt:.1f}×"))
    if hist.get("history_liver"):
        add("others", 1, {"analyte_or_label": "history_liver", "measured_at": None, "reason": "기저 간질환 코드 (이전 입원)"})
    explained = any(S.get(c, {}).get("score", 0) >= FLAG_THRESHOLD for c in ("hypoxic_hepatitis", "dili", "viral_hepatitis", "others"))
    if r_ratio is not None and r_ratio >= 5 and (alt or 0) >= 3 and not explained:
        add("others", 1, _sup(labs["ALT"], f"설명되지 않는 간세포형 (R {r_ratio:.1f}): 자가면역·기타 감별"))

    ranked = sorted(((c, v) for c, v in S.items() if v["score"] > 0), key=lambda x: -x[1]["score"])[:3]
    missing = [a for a in ("ALT", "ALP", "TBIL") if a not in labs]
    abstain = not ranked or ranked[0][1]["score"] < 1.5
    if abstain and len(ranked) < 3:
        ranked.append(("unknown", {"support": [], "against": ["근거가 약하거나 서로 충돌해 판단 보류"]}))
    unc = [f"R 비 {r_ratio:.1f} ({pattern})" if r_ratio is not None else "R 비 계산 불가 (ALT·ALP 동시 측정 없음)"]
    if missing:
        unc.append("최근 측정 없음: " + ", ".join(missing))
    unc.append("규칙 기반 기준 모델. 노트·영상·배양 정보는 사용하지 않음.")
    flags = {}
    for c in CAUSE_FLAGS:
        s = S.get(c, {})
        flags[c] = {"present": int(s.get("score", 0) >= FLAG_THRESHOLD) if c not in NOT_ASSESSABLE else 0,
                    "assessable": c not in NOT_ASSESSABLE, "score": round(float(s.get("score", 0)), 2),
                    "support": s.get("support", []), "against": s.get("against", []) + ([NOT_ASSESSABLE[c]] if c in NOT_ASSESSABLE else [])}
    return {"candidates": [{"cause_code": c, "support": v["support"], "against_or_missing": v["against"]} for c, v in ranked],
            "flags": flags, "uncertainty": " · ".join(unc), "abstain": abstain or not any(f["present"] for f in flags.values()),
            "pattern": pattern, "model": VERSION}
