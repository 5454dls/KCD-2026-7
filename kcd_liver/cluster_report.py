"""Improved liver-pattern clustering at the first event: model selection, profiles, outcomes, added value, figures."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .pattern import _apply_params, _prep_params, r_pattern

FEATS = ["ALT_ratio", "AST_ratio", "ALP_ratio", "TBIL_ratio"]
PROFILE = ["ALT_ratio", "AST_ratio", "ALP_ratio", "TBIL_ratio", "R_ratio"]
LABEL = {"ALT_ratio": "ALT", "AST_ratio": "AST", "ALP_ratio": "ALP", "TBIL_ratio": "TBIL", "R_ratio": "R"}
PATTERN_KO = {"hepatocellular": "간세포형", "mixed": "혼합형", "cholestatic": "담즙정체형",
              "isolated_bilirubin": "빌리루빈 단독", "unclassifiable": "분류 불가"}
STABLE_ARI = 0.75
BIC_TOLERANCE = 0.01
INK, MUTED, GRID = "#1f2a30", "#5a6b70", "#d3dedb"
REC, DIED = "#2a78d6", "#eb6834"


def wilson(k, n, z=1.959964):
    if not n:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - w, c + w)


def _fit(k, X, cov, seed):
    from sklearn.mixture import GaussianMixture
    return GaussianMixture(k, covariance_type=cov, n_init=3, random_state=seed, reg_covar=1e-4).fit(X)


def select_model(X, ks=range(2, 13), covs=("diag", "full"), n_boot=30, seed=0, log=print):
    """BIC and bootstrap ARI for every (covariance, K); pick the smallest stable K within 1% of the best stable BIC."""
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(seed)
    grid = []
    for cov in covs:
        for k in ks:
            if k >= len(X) // 10:
                break
            m = _fit(k, X, cov, seed)
            ref = m.predict(X)
            aris = [adjusted_rand_score(ref, _fit(k, X[idx], cov, seed + b + 1).predict(X))
                    for b, idx in enumerate(rng.integers(0, len(X), (n_boot, len(X))))]
            q = np.quantile(aris, [.25, .5, .75])
            grid.append({"covariance": cov, "k": int(k), "bic": float(m.bic(X)), "ari_q25": float(q[0]),
                         "ari_median": float(q[1]), "ari_q75": float(q[2]), "smallest": int(np.bincount(ref, minlength=k).min())})
            log(f"cluster grid {cov} K={k}: BIC {grid[-1]['bic']:.0f}, ARI {q[1]:.2f}")
    stable = [g for g in grid if g["ari_median"] >= STABLE_ARI and g["smallest"] >= 30]
    pool = stable or grid
    best = min(g["bic"] for g in pool)
    near = [g for g in pool if g["bic"] <= best + BIC_TOLERANCE * abs(best)]
    pick = min(near, key=lambda g: (g["k"], g["bic"]))
    return grid, {**pick, "stable": bool(stable), "rule": f"stable (ARI median >= {STABLE_ARI}, smallest cluster >= 30), "
                  f"then the smallest K whose BIC is within {BIC_TOLERANCE:.0%} of the best stable BIC"}


def _name(prof, dom):
    if all(prof.get(c) is None for c in PROFILE):
        return f"{PATTERN_KO.get(dom, dom)} · 소수"
    top = max(["ALT_ratio", "AST_ratio", "ALP_ratio", "TBIL_ratio"], key=lambda c: prof.get(c) or 0)
    v = prof.get(top) or 0
    return f"{PATTERN_KO.get(dom, dom)} · {LABEL[top]} {v:.0f}×" if v >= 10 else f"{PATTERN_KO.get(dom, dom)} · {LABEL[top]} {v:.1f}×"


def _rate(k, n, min_cell):
    small = lambda v: 0 < v < min_cell
    if n == 0 or small(n) or small(k) or small(n - k):
        return {"n": None if small(n) else int(n), "k": None, "rate": None, "ci": None}
    return {"n": int(n), "k": int(k), "rate": k / n, "ci": list(wilson(k, n))}


def outcome_table(groups, outcomes, min_cell):
    rows = []
    for g, d in outcomes.groupby(groups, sort=True):
        assess = d[~d.observed_path.str.startswith("unassessable")]
        rows.append({"group": g, "n": int(len(d)) if len(d) >= min_cell else None,
                     "icu_death": _rate(int(d.died_icu.sum()), len(d), min_cell),
                     "recovered": _rate(int(assess.observed_path.str.startswith("recovered").sum()), len(assess), min_cell)})
    return rows


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(set(y)) == 2 else None


def added_value(dev, ev, target, n_boot=500, seed=0):
    """Evaluation AUROC: R pattern; R pattern + the four log2 x ULN values (severity); R pattern + cluster; all three."""
    from sklearn.linear_model import LogisticRegression
    if dev[target].nunique() < 2 or ev[target].nunique() < 2:
        return None
    sev = [c for c in FEATS if c in dev]

    def design(d, use_cluster, use_sev, cols=None):
        parts = [pd.get_dummies(d["liver_pattern"].astype(str), prefix="p")]
        if use_cluster:
            parts.append(pd.get_dummies(d["cluster"].astype(str), prefix="c"))
        if use_sev:
            v = d[sev].apply(pd.to_numeric, errors="coerce").apply(lambda x: np.log2(x.clip(lower=.05)))
            parts.append(v.fillna(v.median() if cols is None else 0))
        X = pd.concat(parts, axis=1).astype(float)
        return X if cols is None else X.reindex(columns=cols, fill_value=0)
    out = {}
    for name, uc, us in [("pattern", False, False), ("pattern_severity", False, True), ("pattern_cluster", True, False), ("pattern_severity_cluster", True, True)]:
        Xd = design(dev, uc, us)
        Xe = design(ev, uc, us, Xd.columns)
        out[name] = LogisticRegression(max_iter=2000).fit(Xd, dev[target]).predict_proba(Xe)[:, 1]
    y = ev[target].to_numpy()
    rng = np.random.default_rng(seed)
    idx = [i for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot)) if len(set(y[i])) == 2]

    def diff(a, b):
        d = [_auc(y[i], out[a][i]) - _auc(y[i], out[b][i]) for i in idx]
        return {"difference": _auc(y, out[a]) - _auc(y, out[b]), "ci": [float(q) for q in np.quantile(d, [.025, .975])] if d else None}
    return {"n": int(len(y)), "events": int(y.sum()), **{f"auroc_{k}": _auc(y, v) for k, v in out.items()},
            "cluster_vs_pattern": diff("pattern_cluster", "pattern"), "cluster_vs_severity": diff("pattern_cluster", "pattern_severity"),
            "cluster_added_to_severity": diff("pattern_severity_cluster", "pattern_severity")}


def run(features, outcomes, extra=None, exclude_cases=(), min_cell=10, seed=20261016, n_boot=30, ks=range(2, 13), log=print):
    """features: 08_features.csv (one row per first event); outcomes: 10_outcomes_EVALUATION_ONLY.csv; extra: case_id + more columns."""
    f = features[~features.case_id.isin(set(exclude_cases))].copy()
    if extra is not None and len(extra):
        f = f.merge(extra.drop_duplicates("case_id"), on="case_id", how="left", suffixes=("", "_extra"))
    feats = [c for c in FEATS if c in f and pd.to_numeric(f[c], errors="coerce").notna().mean() >= .5]
    f["liver_pattern"] = [r_pattern(a, b, c, d) for a, b, c, d in
                          zip(f.get("ALT_ratio"), f.get("ALP_ratio"), f.get("TBIL_ratio"), f.get("R_ratio", pd.Series(np.nan, index=f.index)))]
    dev, ev = f[f.split.eq("development")].copy(), f[f.split.eq("evaluation")].copy()
    prep = _prep_params(dev, feats)
    prep["log_cols"] = list(feats)
    Xl = dev[feats].apply(pd.to_numeric, errors="coerce").apply(lambda v: np.log2(v.clip(lower=.05)))
    prep.update(median=Xl.median().to_dict(), mean=Xl.fillna(Xl.median()).mean().to_dict(),
                sd=Xl.fillna(Xl.median()).std(ddof=0).replace(0, 1).to_dict())
    Xd, Xe = _apply_params(dev, prep), _apply_params(ev, prep)
    grid, pick = select_model(Xd, ks=ks, seed=seed, n_boot=n_boot, log=log)
    gmm = _fit(pick["k"], Xd, pick["covariance"], seed)
    dev["cluster"], ev["cluster"] = gmm.predict(Xd), gmm.predict(Xe)
    num = lambda d, c: pd.to_numeric(d[c], errors="coerce")
    profiles = []
    for k in range(pick["k"]):
        d = dev[dev.cluster.eq(k)]
        prof = {c: float(num(d, c).median()) if c in d and num(d, c).notna().any() else None for c in PROFILE}
        iqr = {c: [float(num(d, c).quantile(.25)), float(num(d, c).quantile(.75))] if c in d and num(d, c).notna().any() else None for c in PROFILE}
        share = d.liver_pattern.value_counts(normalize=True)
        dom = share.index[0] if len(share) else "unclassifiable"
        if len(d) < min_cell:
            prof, iqr = {c: None for c in PROFILE}, {c: None for c in PROFILE}
        profiles.append({"cluster": k, "n_dev": int(len(d)) if len(d) >= min_cell else None, "n_eval": int(ev.cluster.eq(k).sum()) if ev.cluster.eq(k).sum() >= min_cell else None,
                         "median": prof, "iqr": iqr, "main_pattern": dom, "main_pattern_share": round(float(share.iloc[0]), 2) if len(share) else None,
                         "name": _name(prof, dom)})
    order = sorted(range(pick["k"]), key=lambda k: (list(PATTERN_KO).index(profiles[k]["main_pattern"]), -(max(v or 0 for v in profiles[k]["median"].values()))))
    relabel = {old: new for new, old in enumerate(order)}
    for d in (dev, ev):
        d["cluster"] = d.cluster.map(relabel)
    profiles = sorted(({**p, "cluster": relabel[p["cluster"]]} for p in profiles), key=lambda p: p["cluster"])
    oc = outcomes[["case_id", "observed_path", "died_icu"]].copy()
    oc["died_icu"] = oc.died_icu.astype(str).str.lower().isin(["true", "1"])
    res = {}
    for name, d in [("development", dev), ("evaluation", ev)]:
        m = d.merge(oc, on="case_id", how="inner")
        m["recovered"] = m.observed_path.str.startswith("recovered").astype(int)
        res[name] = {"by_cluster": outcome_table(m.cluster, m, min_cell), "by_pattern": outcome_table(m.liver_pattern, m, min_cell), "frame": m}
    dm, em = res["development"].pop("frame"), res["evaluation"].pop("frame")
    av = {"icu_death": added_value(dm, em, "died_icu", seed=seed)}
    da, ea = dm[~dm.observed_path.str.startswith("unassessable")], em[~em.observed_path.str.startswith("unassessable")]
    av["recovered"] = added_value(da, ea, "recovered", seed=seed)
    dens = _density(dev, min_cell)
    log("cluster characterisation over all curated variables")
    char = characterize(dev, min_cell)
    causes = cause_table(pd.concat([dev, ev]), min_cell)
    return {"causes": causes, "characterization": char, "method": "GaussianMixture on log2 x ULN of ALT, AST, ALP, TBIL (R left out: log R = log ALT - log ALP)",
            "features": feats, "n_dev": int(len(dev)), "n_eval": int(len(ev)), "min_cell_suppression": min_cell,
            "grid": grid, "selected": pick, "profiles": profiles, "outcomes": res, "added_value": av,
            "density_alt_alp": dens, "ellipses_alt_alp": _ellipses(gmm, prep, feats, relabel)}


VAR_KO = {"age": "나이", "female": "여성", "hours_since_icu": "ICU 입실 후 시간", "pressor_record6": "승압제 기록 수 (6h)",
          "urine_recorded_sum6": "소변량 (6h)", "history_liver": "이전 간질환", "history_viral_hepatitis": "이전 바이러스 간염",
          "history_heart_failure": "이전 심부전", "history_ckd": "이전 만성 신질환", "history_diabetes": "이전 당뇨",
          "history_cancer": "이전 암", "group_initial_abnormal": "입실 초기 이상", "group_incident": "관찰 중 신규",
          "group_onset_unknown": "시점 불명", "ALBUMIN": "알부민", "BICARB": "중탄산", "BUN": "BUN", "CK": "CK", "CREAT": "크레아티닌",
          "DBIL": "직접 빌리루빈", "FIBRINOGEN": "피브리노겐", "FIO2": "FiO2", "GGT": "GGT", "GLUCOSE": "혈당", "HEMOGLOBIN": "혈색소",
          "HR": "심박", "INR": "INR", "LACTATE": "젖산", "LDH": "LDH", "MAP": "평균동맥압", "DBP": "이완기압", "SBP": "수축기압",
          "PEEP": "PEEP", "PH": "pH", "PLATELET": "혈소판", "PO2_UNSPECIFIED": "PO2", "POTASSIUM": "칼륨", "PT": "PT", "PTT": "PTT",
          "RR": "호흡수", "SODIUM": "나트륨", "SPO2": "SpO2", "TEMP_C": "체온", "WBC": "백혈구", "ALT": "ALT", "AST": "AST", "ALP": "ALP",
          "TBIL": "총빌리루빈", "NTPROBNP": "NT-proBNP", "HAPTO": "합토글로빈"}
CAUSE_KO = {"hypoxic_hepatitis": "저산소성 간염", "congestive_hepatopathy": "울혈성 간병증", "sepsis_cholestasis": "패혈증 담즙정체",
            "dili": "DILI", "pn_associated": "정맥영양 관련", "biliary_obstruction_cholangitis": "담도폐쇄·담관염",
            "cholecystitis": "담낭염", "hemolysis": "용혈", "viral_hepatitis": "바이러스 간염", "others": "기타"}
SKEWED = {"ALT", "AST", "ALP", "TBIL", "DBIL", "GGT", "LDH", "CK", "CREAT", "BUN", "LACTATE", "INR", "PT", "PTT", "WBC",
          "NTPROBNP", "GLUCOSE", "FIBRINOGEN", "urine_recorded_sum6", "hours_since_icu"}


def var_label(c):
    if c.startswith("flag_"):
        return "규칙 판정: " + CAUSE_KO.get(c[5:], c[5:])
    for suf, ko in [("_latest", ""), ("_delta24", " 24h 변화"), ("_slope24", " 24h 기울기")]:
        if c.endswith(suf):
            b = c[: -len(suf)]
            return VAR_KO.get(b, b) + ko
    return VAR_KO.get(c, c)


def candidate_vars(d):
    """Curated variables: context, history, every analyte's latest value, liver 24 h changes, Agent 1 rule flags."""
    base = [c for c in ["age", "female", "hours_since_icu", "pressor_record6", "urine_recorded_sum6"] if c in d]
    hist = [c for c in d if c.startswith("history_") and c != "history_available"]
    labs = [c for c in d if c.endswith("_latest")]
    change = [f"{a}_delta24" for a in ["ALT", "AST", "ALP", "TBIL"] if f"{a}_delta24" in d]
    flags = [c for c in d if c.startswith("flag_")]
    groups = [c for c in d if c.startswith("group_")]
    return base + groups + hist + labs + change + flags


def _binary(x):
    v = x.dropna().unique()
    return len(v) <= 2 and set(np.round(v.astype(float), 6)) <= {0.0, 1.0}


def characterize(dev, min_cell, top=6):
    """Per-cluster summary of every curated variable and its standardized difference against the other clusters."""
    if "group" in dev and dev.group.notna().any():
        for g in ["initial_abnormal", "incident", "onset_unknown"]:
            dev[f"group_{g}"] = dev.group.eq(g).astype(float)
    rows, ks = [], sorted(dev.cluster.unique())
    for c in candidate_vars(dev):
        x = pd.to_numeric(dev[c], errors="coerce")
        if x.notna().mean() < .3 or x.nunique() < 2:
            continue
        b = _binary(x)
        base = c.split("_latest")[0].split("_delta24")[0]
        t = np.log2(x.clip(lower=1e-3)) if (not b and base in SKEWED and c.endswith("_latest")) else x
        for k in ks:
            m = dev.cluster.eq(k)
            a, r = t[m].dropna(), t[~m].dropna()
            if len(a) < min_cell or len(r) < min_cell:
                continue
            sd = math.sqrt((a.var() + r.var()) / 2) or 1.0
            row = {"cluster": int(k), "variable": c, "label": var_label(c), "binary": b, "observed": float(x[m].notna().mean()),
                   "smd": float((a.mean() - r.mean()) / sd)}
            if b:
                cnt = int(x[m].sum())
                row.update(share=float(cnt / x[m].notna().sum()) if cnt >= min_cell and x[m].notna().sum() - cnt >= min_cell else None,
                           share_rest=float(x[~m].mean()))
                if row["share"] is None:
                    row["smd"] = None
            else:
                row.update(median=float(x[m].median()), q1=float(x[m].quantile(.25)), q3=float(x[m].quantile(.75)), median_rest=float(x[~m].median()))
            rows.append(row)
    tops = {}
    for k in ks:
        cand = [r for r in rows if r["cluster"] == k and r["smd"] is not None and r["observed"] >= .5 and not r["variable"].endswith("_ratio")]
        tops[int(k)] = [r["variable"] for r in sorted(cand, key=lambda r: -abs(r["smd"]))[:top]]
    return {"variables": rows, "top_by_cluster": tops,
            "note": "SMD = (cluster mean - other clusters mean) / pooled SD; skewed labs on log2 scale. Binary shares with counts < min_cell are hidden."}


CAUSE_ORDER = list(CAUSE_KO)


def cause_table(d, min_cell):
    """Per cluster: share of events with each of the 10 rule flags (Agent 1) and each discharge-ICD proxy, all events."""
    rows = []
    for k, g in d.groupby("cluster"):
        for c in CAUSE_ORDER:
            for src, col in [("rule", f"flag_{c}"), ("icd", f"icd_{c}")]:
                if col not in g or g[col].notna().sum() == 0:
                    continue
                x = pd.to_numeric(g[col], errors="coerce").dropna()
                cnt, nn = int(x.sum()), int(len(x))
                ok = nn >= min_cell and (cnt == 0 or (cnt >= min_cell and nn - cnt >= min_cell))
                rows.append({"cluster": int(k), "cause": c, "source": src, "n": nn if nn >= min_cell else None,
                             "k": cnt if ok else None, "share": cnt / nn if ok and nn else None})
    return {"rows": rows, "note": "rule = Agent 1 rule model at the first event (point-in-time); icd = same-admission discharge codes "
                                  "(no timestamp, may predate or follow the ICU stay)"}


def _density(dev, min_cell, bins=24):
    a = np.log2(pd.to_numeric(dev.get("ALT_ratio"), errors="coerce").clip(lower=.05))
    b = np.log2(pd.to_numeric(dev.get("ALP_ratio"), errors="coerce").clip(lower=.05))
    ok = a.notna() & b.notna()
    H, xe, ye = np.histogram2d(a[ok], b[ok], bins=bins, range=[[-3, 9], [-3, 7]])
    H = np.where(H < min_cell, 0, H)
    return {"x_edges": xe.tolist(), "y_edges": ye.tolist(), "counts": H.astype(int).tolist(), "note": f"cells < {min_cell} set to 0"}


def _ellipses(gmm, prep, feats, relabel):
    if not {"ALT_ratio", "ALP_ratio"} <= set(feats):
        return []
    i, j = feats.index("ALT_ratio"), feats.index("ALP_ratio")
    sd = np.array([prep["sd"][c] for c in feats])
    mu0 = np.array([prep["mean"][c] for c in feats])
    out = []
    for k in range(gmm.n_components):
        cov = gmm.covariances_[k]
        cov = np.diag(cov) if cov.ndim == 1 else cov
        S = cov[np.ix_([i, j], [i, j])] * np.outer(sd[[i, j]], sd[[i, j]])
        mu = gmm.means_[k][[i, j]] * sd[[i, j]] + mu0[[i, j]]
        out.append({"cluster": relabel[k], "mean": mu.tolist(), "cov": S.tolist(), "weight": float(gmm.weights_[k])})
    return sorted(out, key=lambda e: e["cluster"])


# ---------------------------------------------------------------- figures (aggregate inputs only)
def _style():
    import matplotlib
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    ko = [f for f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"] if f in have]
    matplotlib.rcParams.update({"font.family": ko + ["DejaVu Sans"], "axes.unicode_minus": False, "font.size": 9,
                                "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                                "grid.linewidth": .6, "axes.axisbelow": True, "axes.titlesize": 10, "axes.titleweight": "bold",
                                "axes.titlelocation": "left", "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK,
                                "xtick.color": MUTED, "ytick.color": MUTED, "savefig.dpi": 200, "savefig.bbox": "tight"})


def fig_selection(R, ax):
    sel = R["selected"]
    for cov, col, lab in [("diag", REC, "대각 공분산"), ("full", DIED, "완전 공분산")]:
        g = [r for r in R["grid"] if r["covariance"] == cov]
        if not g:
            continue
        ax.plot([r["k"] for r in g], [r["bic"] for r in g], color=col, lw=2, label=lab)
        for r in g:
            ax.scatter(r["k"], r["bic"], s=36, zorder=3, color=col if r["ari_median"] >= STABLE_ARI else "white", edgecolor=col, linewidth=1.5)
    ax.scatter(sel["k"], sel["bic"], s=180, marker="*", color=INK, zorder=4, label=f"선택: K={sel['k']} ({'완전' if sel['covariance'] == 'full' else '대각'})")
    ax.set_xlabel("군집 수 K")
    ax.set_ylabel("BIC (낮을수록 좋음)")
    ax.set_title("A. 모델 선택: 채운 점 = 부트스트랩 ARI ≥ 0.75")
    ax.legend(frameon=False, fontsize=8)


def fig_profiles(R, ax):
    P = R["profiles"]
    cols = PROFILE
    M = np.array([[math.log2(max(p["median"][c] or .05, .05)) for c in cols] for p in P])
    im = ax.imshow(M, cmap="Blues", vmin=-1, vmax=max(4, np.nanmax(M)), aspect="auto")
    for i, p in enumerate(P):
        for j, c in enumerate(cols):
            v = p["median"][c]
            ax.text(j, i, "–" if v is None else (f"{v:.0f}" if v >= 10 else f"{v:.1f}"), ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 2.5 else INK)
    ax.set_xticks(range(len(cols)), [LABEL[c] + (" 비" if c == "R_ratio" else " ×ULN") for c in cols])
    ax.set_yticks(range(len(P)), [f"C{p['cluster']} {p['name']}  (n={p['n_dev'] or '<10'})" for p in P])
    ax.grid(False)
    ax.set_title("B. 군집별 중앙값 (개발 집단)")
    cb = ax.figure.colorbar(im, ax=ax, fraction=.04, pad=.02)
    cb.set_label("log2 중앙값")


def fig_map(R, ax):
    from matplotlib.patches import Ellipse
    D = R["density_alt_alp"]
    H = np.array(D["counts"], dtype=float).T
    H[H == 0] = np.nan
    ax.pcolormesh(D["x_edges"], D["y_edges"], H, cmap="Greys", alpha=.55, shading="flat")
    for e in R["ellipses_alt_alp"]:
        vals, vecs = np.linalg.eigh(np.array(e["cov"]))
        ang = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
        w, h = 2 * 1.177 * np.sqrt(np.maximum(vals[::-1], 1e-6))
        ax.add_patch(Ellipse(e["mean"], w, h, angle=ang, fill=False, lw=1.6, color=REC))
        ax.text(*e["mean"], f"C{e['cluster']}", ha="center", va="center", fontsize=8, fontweight="bold", color=INK,
                bbox=dict(boxstyle="round,pad=.15", fc="white", ec="none", alpha=.8))
    ax.axvline(1, color=MUTED, lw=.8, ls="--")
    ax.axhline(1, color=MUTED, lw=.8, ls="--")
    ax.plot([-3, 9], [-3 - math.log2(5), 9 - math.log2(5)], color=MUTED, lw=.8, ls=":")
    ax.plot([-3, 9], [-3 - 1, 9 - 1], color=MUTED, lw=.8, ls=":")
    ax.set_xlim(-3, 9)
    ax.set_ylim(-3, 7)
    ax.set_xlabel("log2 ALT ×ULN  (점선 = 2×ULN)")
    ax.set_ylabel("log2 ALP ×ULN")
    ax.set_title("C. ALT·ALP 평면의 군집 (회색 = 사건 밀도, 타원 = 군집 50% 영역, 점선 사선 = R 5와 2)")


def fig_outcomes(R, ax, split="evaluation"):
    rows = R["outcomes"][split]["by_cluster"]
    names = {p["cluster"]: f"C{p['cluster']} {p['name']}" for p in R["profiles"]}
    y = np.arange(len(rows))[::-1]
    for yy, r in zip(y, rows):
        for key, col, off in [("icu_death", DIED, .15), ("recovered", REC, -.15)]:
            v = r[key]
            if v["rate"] is None:
                ax.text(.01, yy + off, "가림", fontsize=7, color=MUTED, va="center")
                continue
            ax.plot(v["ci"], [yy + off] * 2, color=col, lw=2, solid_capstyle="round")
            ax.scatter(v["rate"], yy + off, s=40, color=col, zorder=3, edgecolor="white", linewidth=1)
    ax.set_yticks(y, [f"{names.get(r['group'], r['group'])}  (n={r['n'] or '<10'})" for r in rows])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.scatter([], [], color=DIED, label="ICU 사망")
    ax.scatter([], [], color=REC, label="ICU 퇴실 시 회복 (판정 가능 사건 중)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title(f"D. 군집별 예후 ({'평가' if split == 'evaluation' else '개발'} 집단, 95% Wilson CI)")


def fig_added(R, ax):
    av = R["added_value"]
    models = [("auroc_pattern", "R 비 패턴", GRID), ("auroc_pattern_severity", "패턴 + 수치 크기", "#9fb7d6"),
              ("auroc_pattern_cluster", "패턴 + 군집", REC), ("auroc_pattern_severity_cluster", "패턴 + 수치 + 군집", INK)]
    keys = [(k, lab) for k, lab in [("icu_death", "ICU 사망"), ("recovered", "ICU 퇴실 시 회복")] if av.get(k)]
    w = .2
    for j, (m, lab, col) in enumerate(models):
        ax.bar(np.arange(len(keys)) + (j - 1.5) * w, [av[k][m] for k, _ in keys], w * .92, color=col, edgecolor=MUTED if j == 0 else col, label=lab)
    for i, (k, _) in enumerate(keys):
        d = av[k]["cluster_added_to_severity"]
        ci = d["ci"]
        ax.text(i, max(av[k][m] for m, _, _ in models) + .02, f"수치에 군집을 더하면 {d['difference']:+.3f}" + (f"\n({ci[0]:+.3f} ~ {ci[1]:+.3f})" if ci else ""),
                ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(keys)), [lab for _, lab in keys])
    ax.set_ylim(.5, 1)
    ax.set_ylabel("평가 집단 AUROC")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=2)
    ax.set_title("E. 군집이 예후 정보를 더하는가 (수치 크기와 공정 비교)")


def fig_characterize(R, ax, max_vars=30):
    C = R["characterization"]
    order = []
    for k, vs in C["top_by_cluster"].items():
        for v in vs[:4]:
            if v not in order:
                order.append(v)
    order = order[:max_vars]
    ks = [p["cluster"] for p in R["profiles"]]
    look = {(r["cluster"], r["variable"]): r for r in C["variables"]}
    M = np.array([[(look.get((k, v)) or {}).get("smd") if (look.get((k, v)) or {}).get("smd") is not None else np.nan for k in ks] for v in order])
    im = ax.imshow(np.clip(M, -1.5, 1.5), cmap="PuOr_r", vmin=-1.5, vmax=1.5, aspect="auto")
    for i in range(len(order)):
        for j in range(len(ks)):
            if not np.isnan(M[i, j]) and abs(M[i, j]) >= .5:
                ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=7, color="white" if abs(M[i, j]) > 1.1 else INK)
    ax.set_xticks(range(len(ks)), [f"C{k}" for k in ks])
    ax.set_yticks(range(len(order)), [var_label(v) for v in order])
    ax.grid(False)
    ax.set_title("F. 군집을 가르는 변수 (군집 대 나머지 표준화 차이, |SMD| ≥ 0.5 표시)")
    cb = ax.figure.colorbar(im, ax=ax, fraction=.035, pad=.02)
    cb.set_label("SMD (주황 = 더 높음, 보라 = 더 낮음)")


def fig_causes(R, ax):
    rows = (R.get("causes") or {}).get("rows") or []
    ks = [p["cluster"] for p in R["profiles"]]
    srcs = [s for s in ["rule", "icd"] if any(r["source"] == s for r in rows)]
    if not srcs:
        ax.axis("off")
        ax.text(.5, .5, "원인 판정이 입력되지 않았습니다", ha="center", va="center", color=MUTED)
        return
    look = {(r["cluster"], r["cause"], r["source"]): r for r in rows}
    cols = [(c, s) for s in srcs for c in CAUSE_ORDER]
    M = np.array([[(look.get((k, c, s)) or {}).get("share") for c, s in cols] for k in ks], dtype=float)
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=max(.5, np.nanmax(M) if np.isfinite(M).any() else .5), aspect="auto")
    for i in range(len(ks)):
        for j, (c, s) in enumerate(cols):
            r = look.get((ks[i], c, s))
            t = "–" if not r else ("<10" if r["share"] is None else f"{r['share'] * 100:.0f}")
            ax.text(j, i, t, ha="center", va="center", fontsize=7, color="white" if (r and r["share"] and r["share"] > .35) else INK)
    ax.set_xticks(range(len(cols)), [CAUSE_KO[c] for c, _ in cols], rotation=60, ha="right", fontsize=8)
    names = {p["cluster"]: f"C{p['cluster']} {p['name']}" for p in R["profiles"]}
    ax.set_yticks(range(len(ks)), [names[k] for k in ks])
    if len(srcs) == 2:
        ax.axvline(len(CAUSE_ORDER) - .5, color=INK, lw=2)
        ax.text(len(CAUSE_ORDER) / 2 - .5, -.65, "규칙 판정 (첫 사건 시점)", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(len(CAUSE_ORDER) * 1.5 - .5, -.65, "퇴원 진단 코드 (시각 없음)", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.grid(False)
    ax.set_title("G. 군집별 원인 10개 해당 비율 (%)", pad=30)
    cb = ax.figure.colorbar(im, ax=ax, fraction=.025, pad=.01)
    cb.set_label("해당 비율")


def render(R, out_dir):
    import matplotlib.pyplot as plt
    _style()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for name, fn, size in [("c1_selection", fig_selection, (6, 3.6)), ("c2_profiles", fig_profiles, (7.5, .45 * len(R["profiles"]) + 1.4)),
                           ("c3_map", fig_map, (6.4, 5)), ("c4_outcomes", fig_outcomes, (7.5, .5 * len(R["profiles"]) + 1.4)),
                           ("c5_added_value", fig_added, (5.5, 3.6)),
                           ("c6_characterization", fig_characterize, (.55 * len(R["profiles"]) + 4.5, 8.5)),
                           ("c7_causes", fig_causes, (13, .55 * len(R["profiles"]) + 3))]:
        fig, ax = plt.subplots(figsize=size)
        fn(R, ax)
        fig.text(.995, .995, f"n<{R['min_cell_suppression']} 칸 가림 · 개발 {R['n_dev']:,}건, 평가 {R['n_eval']:,}건", ha="right", va="bottom", fontsize=7, color=MUTED)
        p = out / f"{name}.png"
        fig.savefig(p)
        plt.close(fig)
        files.append(p)
    return files


def save(R, path):
    Path(path).write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
