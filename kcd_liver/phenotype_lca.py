"""Clinical phenotypes at the first event by latent class analysis on clinically banded indicators ("not measured" is a level)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pattern import r_pattern

MISSING = "미측정"


def _band(x, cuts, labels, missing_level=True):
    """missing_level=False: not measured stays missing and is skipped in the likelihood (routine tests)."""
    x = pd.to_numeric(x, errors="coerce")
    out = pd.Series(MISSING if missing_level else np.nan, index=x.index, dtype=object)
    ok = x.notna()
    out[ok] = pd.cut(x[ok], [-np.inf, *cuts, np.inf], labels=labels, right=False).astype(str)
    return out


def _flag(x, unknown=None):
    x = pd.to_numeric(x, errors="coerce")
    out = pd.Series("아니오", index=x.index, dtype=object)
    out[x.fillna(0) > 0] = "예"
    if unknown is not None:
        out[unknown] = np.nan
    return out


def indicators(f: pd.DataFrame) -> pd.DataFrame:
    """Banded indicators from 08_features columns. Cut points are clinical conventions fixed before fitting."""
    g = lambda c: f[c] if c in f else pd.Series(np.nan, index=f.index)
    peak = pd.concat([pd.to_numeric(g(c), errors="coerce") for c in ["ALT_ratio", "AST_ratio", "ALP_ratio", "TBIL_ratio"]], axis=1).max(axis=1)
    pat = pd.Series([r_pattern(a, b, c, d) for a, b, c, d in zip(g("ALT_ratio"), g("ALP_ratio"), g("TBIL_ratio"), g("R_ratio"))], index=f.index)
    ast_alt = pd.to_numeric(g("AST_ratio"), errors="coerce") / pd.to_numeric(g("ALT_ratio"), errors="coerce").clip(lower=.05)
    hist_unknown = pd.to_numeric(g("history_available"), errors="coerce").fillna(0).eq(0) if "history_available" in f else None
    shock = pd.Series("없음", index=f.index, dtype=object)
    shock[pd.to_numeric(g("MAP_min24"), errors="coerce") < 65] = "저혈압"
    shock[pd.to_numeric(g("pressor_record6"), errors="coerce").fillna(0) > 0] = "승압제"
    temp = _band(g("TEMP_C_max24"), [36, 38.3], ["<36", "36-38.3", "≥38.3"], False)
    X = pd.DataFrame({
        "간 패턴": pat.map({"hepatocellular": "간세포형", "mixed": "혼합형", "cholestatic": "담즙정체형",
                          "isolated_bilirubin": "빌리루빈 단독"}).fillna("분류 불가"),
        "최고 상승 (×ULN)": _band(peak, [5, 10, 20], ["2-5", "5-10", "10-20", "≥20"]),
        "AST/ALT": _band(ast_alt, [1, 2], ["<1", "1-2", "≥2"], False),
        "총빌리루빈 (×ULN)": _band(g("TBIL_ratio"), [1, 2, 5], ["<1", "1-2", "2-5", "≥5"], False),
        "쇼크": shock,
        "젖산 (mmol/L)": _band(g("LACTATE_latest"), [2, 4], ["<2", "2-4", "≥4"]),
        "크레아티닌 (mg/dL)": _band(g("CREAT_latest"), [1.5, 3], ["<1.5", "1.5-3", "≥3"], False),
        "INR": _band(g("INR_latest"), [1.5, 2.5], ["<1.5", "1.5-2.5", "≥2.5"], False),
        "혈소판 (×10³/µL)": _band(g("PLATELET_latest"), [50, 150], ["<50", "50-150", "≥150"], False),
        "백혈구 (×10³/µL)": _band(g("WBC_latest"), [4, 12], ["<4", "4-12", ">12"], False),
        "최고 체온 (°C)": temp,
        "LDH (×ULN)": _band(g("LDH_ratio"), [2, 5], ["<2", "2-5", "≥5"]),
        "이전 간질환": _flag(g("history_liver"), hist_unknown),
        "이전 심부전": _flag(g("history_heart_failure"), hist_unknown),
    }, index=f.index)
    if "group" in f:
        X["사건 시점"] = f["group"].map({"initial_abnormal": "입실 초기", "incident": "관찰 중 신규", "onset_unknown": "시점 불명"}).fillna("시점 불명")
    return X


class LCA:
    """Polytomous latent class model fitted by EM (several random starts, the best log-likelihood kept)."""

    def __init__(self, k, n_init=10, max_iter=500, tol=1e-6, seed=0, alpha=1.0):
        self.k, self.n_init, self.max_iter, self.tol, self.seed, self.alpha = k, n_init, max_iter, tol, seed, alpha

    def _encode(self, X):
        out = []
        for c in self.cols_:
            lv, v = self.levels_[c], X[c]
            s = np.array(["" if pd.isna(x) else str(x) for x in v], dtype=str)
            lv = lv.astype(str)
            idx = np.searchsorted(lv, s).clip(0, max(len(lv) - 1, 0))
            out.append(np.where(v.notna().to_numpy() & (lv[idx] == s), idx, -1) if len(lv) else np.full(len(s), -1))
        return out

    def fit(self, X: pd.DataFrame):
        self.cols_ = [c for c in X.columns if X[c].notna().any() and X[c].dropna().astype(str).nunique() > 1]
        self.levels_ = {c: np.array(sorted(X[c].dropna().astype(str).unique())) for c in self.cols_}
        codes = self._encode(X)
        n, rng, best = len(X), np.random.default_rng(self.seed), None
        for _ in range(self.n_init):
            R = rng.dirichlet(np.ones(self.k), n)
            ll_old = -np.inf
            for it in range(self.max_iter):
                pi = R.mean(0)
                theta = [(np.stack([np.bincount(cd[cd >= 0], weights=R[cd >= 0, j], minlength=len(self.levels_[c])) for j in range(self.k)]) + self.alpha)
                         for cd, c in zip(codes, self.cols_)]
                theta = [t / t.sum(1, keepdims=True) for t in theta]
                L = np.log(pi)[None, :] + sum(np.where(cd[:, None] >= 0, np.log(t[:, cd.clip(0)]).T, 0.0) for t, cd in zip(theta, codes))
                m = L.max(1, keepdims=True)
                ll = float((m[:, 0] + np.log(np.exp(L - m).sum(1))).sum())
                R = np.exp(L - m)
                R /= R.sum(1, keepdims=True)
                if ll - ll_old < self.tol * abs(ll):
                    break
                ll_old = ll
            if best is None or ll > best[0]:
                best = (ll, pi, theta, it)
        self.loglik_, self.pi_, self.theta_, self.n_iter_ = best
        self.n_params_ = (self.k - 1) + self.k * sum(len(self.levels_[c]) - 1 for c in self.cols_)
        self.bic_ = -2 * self.loglik_ + self.n_params_ * np.log(n)
        return self

    def predict_proba(self, X):
        codes = self._encode(X)
        L = np.log(self.pi_)[None, :] + sum(np.where(cd[:, None] >= 0, np.log(t[:, cd.clip(0)]).T, 0.0) for t, cd in zip(self.theta_, codes))
        L -= L.max(1, keepdims=True)
        P = np.exp(L)
        return P / P.sum(1, keepdims=True)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def entropy(self, X):
        """Relative entropy (1 = perfectly separated classes)."""
        P = self.predict_proba(X).clip(1e-12)
        return float(1 - (-(P * np.log(P)).sum()) / (len(P) * np.log(self.k))) if self.k > 1 else 1.0


def _match(a, b):
    """Map classes of model b onto model a by profile similarity over the union of levels (Hungarian)."""
    from scipy.optimize import linear_sum_assignment

    def vec(m, c, lv):
        j = m.cols_.index(c)
        idx = {l: i for i, l in enumerate(m.levels_[c])}
        return np.stack([m.theta_[j][:, idx[l]] if l in idx else np.zeros(m.k) for l in lv], axis=1)
    va, vb = [], []
    for c in a.cols_:
        lv = sorted(set(a.levels_[c]) | set(b.levels_.get(c, [])))
        va.append(vec(a, c, lv))
        vb.append(vec(b, c, lv) if c in b.cols_ else np.zeros((b.k, len(lv))))
    va, vb = np.concatenate(va, 1), np.concatenate(vb, 1)
    r, c = linear_sum_assignment(((va[:, None, :] - vb[None, :, :]) ** 2).sum(-1))
    return dict(zip(c, r))


def select(Xd, ks=range(2, 9), n_boot=20, seed=0, min_share=.03, log=print):
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(seed)
    grid = []
    for k in ks:
        m = LCA(k, seed=seed).fit(Xd)
        ref = m.predict(Xd)
        aris = [adjusted_rand_score(ref, LCA(k, n_init=4, seed=seed + b + 1).fit(Xd.iloc[idx]).predict(Xd))
                for b, idx in enumerate(rng.integers(0, len(Xd), (n_boot, len(Xd))))]
        share = np.bincount(ref, minlength=k).min() / len(Xd)
        grid.append({"k": int(k), "bic": float(m.bic_), "loglik": float(m.loglik_), "entropy": m.entropy(Xd),
                     "ari_median": float(np.median(aris)), "ari_q25": float(np.quantile(aris, .25)), "smallest_share": float(share)})
        log(f"LCA K={k}: BIC {m.bic_:.0f}, entropy {grid[-1]['entropy']:.2f}, ARI {grid[-1]['ari_median']:.2f}, smallest {share:.1%}")
    ok = [g for g in grid if g["ari_median"] >= .75 and g["smallest_share"] >= min_share]
    pool = ok or grid
    best = min(g["bic"] for g in pool)
    pick = min((g for g in pool if g["bic"] <= best + .01 * abs(best)), key=lambda g: g["k"])
    return grid, {**pick, "stable": bool(ok),
                  "rule": f"bootstrap ARI median >= 0.75 and smallest class >= {min_share:.0%}, then the smallest K within 1% of the best BIC"}


def run_lca(dev_f, ev_f, min_cell=10, seed=20261016, ks=range(2, 9), n_boot=20, log=print):
    Xd, Xe = indicators(dev_f), indicators(ev_f)
    grid, pick = select(Xd, ks=ks, n_boot=n_boot, seed=seed, log=log)
    m = LCA(pick["k"], seed=seed).fit(Xd)
    cd, ce = m.predict(Xd), m.predict(Xe)
    # replication: refit on the evaluation set alone and compare with the development model's assignment
    from sklearn.metrics import adjusted_rand_score
    me = LCA(pick["k"], seed=seed + 7).fit(Xe)
    mp = _match(m, me)
    ce_refit = np.array([mp.get(c, c) for c in me.predict(Xe)])
    repl = {"ari_dev_model_vs_eval_refit": float(adjusted_rand_score(ce, ce_refit)), "agreement": float((ce == ce_refit).mean())}
    order = np.argsort(-np.bincount(cd, minlength=pick["k"]))
    rl = {int(o): i for i, o in enumerate(order)}
    cd, ce = np.array([rl[c] for c in cd]), np.array([rl[c] for c in ce])
    prof = []
    for new, old in enumerate(order):
        n_dev = int((cd == new).sum())
        rows = []
        for j, c in enumerate(m.cols_):
            for li, lev in enumerate(m.levels_[c]):
                p = float(m.theta_[j][old, li])
                obs = Xd[c].dropna().astype(str)
                overall = float((obs == lev).mean()) if len(obs) else 0.0
                rows.append({"indicator": c, "level": str(lev), "prob": p, "overall": overall})
        prof.append({"class": new, "n_dev": n_dev if n_dev >= min_cell else None, "n_eval": int((ce == new).sum()) if (ce == new).sum() >= min_cell else None,
                     "share": float(m.pi_[old]), "items": rows})
    for p in prof:
        p["name"] = _name(p)
    return {"method": "latent class analysis (EM, categorical indicators, 'not measured' as a level)", "indicators": m.cols_,
            "grid": grid, "selected": pick, "entropy": m.entropy(Xd), "replication": repl, "profiles": prof}, cd, ce


KEY = [("쇼크", "승압제"), ("젖산 (mmol/L)", "≥4"), ("최고 상승 (×ULN)", "≥20"), ("간 패턴", "담즙정체형"), ("간 패턴", "빌리루빈 단독"),
       ("총빌리루빈 (×ULN)", "≥5"), ("이전 간질환", "예"), ("혈소판 (×10³/µL)", "<50"), ("이전 심부전", "예"), ("크레아티닌 (mg/dL)", "≥3"),
       ("INR", "≥2.5"), ("최고 체온 (°C)", "≥38.3"), ("간 패턴", "간세포형")]


def _name(p):
    """Two most over-represented key features (probability / overall)."""
    look = {(r["indicator"], r["level"]): r for r in p["items"]}
    sc = sorted(((look[k]["prob"] / max(look[k]["overall"], 1e-3), k) for k in KEY if k in look and look[k]["prob"] >= .3), reverse=True)
    if not sc:
        return "특징 적음"
    return " · ".join(f"{k[0].split(' (')[0]} {k[1]}" for _, k in sc[:2])
