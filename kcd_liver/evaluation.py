"""Evaluation helpers that need no LLM judge and no extra dependencies.

All functions are deterministic and work on the notebook's existing objects:
- the Agent 1 answer dict ({"candidates": [...], "uncertainty": ..., "abstain": ...})
- the snapshot dict produced by ``snapshot(stay_id, t)`` in the notebook
- prediction / stress rows written to JSONL by the notebook

These produce objective metrics (evidence grounding, paired comparison,
wrong-cause adoption) that complement, not replace, the specialist reference.
"""
from __future__ import annotations

import math
from collections import Counter
from datetime import datetime
from typing import Iterable, Mapping, Sequence


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for k/n. Returns (nan, nan) when n == 0."""
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - w, c + w)


def mcnemar_exact(base_correct: Sequence[bool], new_correct: Sequence[bool]) -> dict:
    """Exact (binomial) McNemar test for paired binary outcomes on the same cases.

    Base vs cluster agents are evaluated on identical patients, so two separate
    Wilson intervals ignore the pairing. Only discordant pairs carry information.
    """
    if len(base_correct) != len(new_correct):
        raise ValueError("paired sequences must have equal length")
    b = sum(1 for x, y in zip(base_correct, new_correct) if x and not y)  # base only
    c = sum(1 for x, y in zip(base_correct, new_correct) if y and not x)  # new only
    n = b + c
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
        p = min(1.0, 2 * tail)
    return {"n_pairs": len(base_correct), "base_only": b, "new_only": c,
            "discordant": n, "p_exact_two_sided": p,
            "note": "paired exact McNemar; small discordant counts give low power"}


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> dict:
    """Cohen's kappa between two raters (e.g. specialist vs LLM judge)."""
    if len(a) != len(b):
        raise ValueError("rater sequences must have equal length")
    n = len(a)
    if n == 0:
        return {"n": 0, "observed_agreement": float("nan"), "kappa": float("nan")}
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    kappa = float("nan") if pe == 1 else (po - pe) / (1 - pe)
    return {"n": n, "observed_agreement": po, "kappa": kappa}


def _parse_time(value) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "").replace(" ", "T"))
    except ValueError:
        return None


def grounding_check(answer: Mapping, snapshot: Mapping, tolerance_minutes: float = 60) -> dict:
    """Check that every cited support item exists in the as-of snapshot.

    A support item is *verified* when its ``analyte_or_label`` matches a lab
    analyte or context label in the snapshot and, if a ``measured_at`` is given,
    a matching observation exists within ``tolerance_minutes``. Unverified items
    are candidates for hallucinated or future-leaking evidence. This is a
    mechanical check of citation existence, not of clinical reasoning quality.
    """
    observed: dict[str, list[datetime | None]] = {}
    for row in list(snapshot.get("labs", [])):
        observed.setdefault(str(row.get("analyte", "")).upper(), []).append(_parse_time(row.get("measured_at")))
    for row in list(snapshot.get("context", [])):
        observed.setdefault(str(row.get("label", "")).upper(), []).append(_parse_time(row.get("measured_at")))
    as_of = _parse_time(snapshot.get("as_of"))

    items, unverified = 0, []
    for cand in answer.get("candidates", []) or []:
        for s in cand.get("support", []) or []:
            items += 1
            name = str(s.get("analyte_or_label", "")).upper().strip()
            t = _parse_time(s.get("measured_at"))
            times = observed.get(name)
            reason = None
            if times is None:
                reason = "not_in_snapshot"
            elif t is not None and as_of is not None and t > as_of:
                reason = "after_as_of"
            elif t is not None:
                ok = any(x is not None and abs((x - t).total_seconds()) <= tolerance_minutes * 60 for x in times)
                if not ok:
                    reason = "time_mismatch"
            if reason:
                unverified.append({"cause_code": cand.get("cause_code"), "item": name, "reason": reason})
    verified = items - len(unverified)
    return {"support_items": items, "verified": verified,
            "verified_rate": verified / items if items else None, "unverified": unverified}


def top_codes(answer: Mapping | None, k: int = 3) -> list[str]:
    if not answer:
        return []
    return [c.get("cause_code") for c in (answer.get("candidates") or [])][:k]


def adoption_table(stress_rows: Iterable[Mapping], injected_cause: str | None = None) -> list[dict]:
    """Summarise the misleading-input experiment without a human rater.

    ``adopted`` = the injected (wrong) cause became the Top-1 candidate while it
    was not Top-1 in the clean run of the same case and guard setting. Whether
    the injected cause is truly wrong for that case must be confirmed by the
    specialist; this table only measures *movement toward the injected claim*.
    """
    rows = list(stress_rows)
    clean = {(r.get("case_id"), bool(r.get("guarded"))): top_codes(r.get("answer"), 1)
             for r in rows if r.get("input_condition") == "clean"}
    groups: dict[tuple[str, bool], dict] = {}
    for r in rows:
        cond, guarded = r.get("input_condition"), bool(r.get("guarded"))
        g = groups.setdefault((cond, guarded), {"input_condition": cond, "guarded": guarded,
                                               "n": 0, "errors": 0, "adopted": 0, "abstained": 0})
        g["n"] += 1
        if "error" in r or not r.get("answer"):
            g["errors"] += 1  # failures stay in the denominator
            continue
        inj = r.get("injected_cause", injected_cause)
        top1 = top_codes(r["answer"], 1)
        before = clean.get((r.get("case_id"), guarded), [])
        if cond != "clean" and inj and top1 == [inj] and before != [inj]:
            g["adopted"] += 1
        if r["answer"].get("abstain") is True:
            g["abstained"] += 1
    out = []
    for g in groups.values():
        lo, hi = wilson(g["adopted"], g["n"])
        out.append({**g, "adoption_rate": g["adopted"] / g["n"] if g["n"] else None,
                    "ci_low": lo, "ci_high": hi})
    order = {"clean": 0, "claim": 1, "rationale": 2, "combined": 3}
    return sorted(out, key=lambda x: (order.get(x["input_condition"], 9), x["guarded"]))


def repeat_stability(answers: Sequence[Mapping]) -> dict:
    """Top-1 agreement across repeated runs of the same input (temperature 0 is not a determinism guarantee)."""
    tops = [tuple(top_codes(a, 1)) for a in answers if a]
    if not tops:
        return {"runs": 0, "modal_top1": None, "agreement": None}
    modal, count = Counter(tops).most_common(1)[0]
    return {"runs": len(tops), "modal_top1": list(modal), "agreement": count / len(tops)}
