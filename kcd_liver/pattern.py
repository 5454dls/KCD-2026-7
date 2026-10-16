"""Liver injury pattern (fixed categories) and an exploratory clustering check.

The cause agent receives the ACG 2017 R-ratio pattern, a fixed and clinically defined
category, instead of an unsupervised cluster id. Unsupervised clustering is kept only as a
supplementary analysis: Gaussian mixture, K chosen by BIC over 2..8,
run twice: liver pattern only (ULN ratios + R) and broad clinical phenotype (about 30 variables),
bootstrap adjusted Rand index (ARI) for stability. It is reported as "not used" when unstable.
A cluster is a biochemical pattern, never a cause.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

PATTERNS = ["hepatocellular", "mixed", "cholestatic", "isolated_bilirubin", "unclassifiable"]
STABLE_ARI = 0.75  # operational threshold, stated with the result


def r_pattern(alt_ratio, alp_ratio, tbil_ratio=None, r_ratio=None, abnormal=2.0):
    """ACG 2017: R = (ALT/ULN)/(ALP/ULN). R>=5 hepatocellular, R<=2 cholestatic, else mixed."""
    f = lambda v: None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
    alt, alp, tb, r = f(alt_ratio), f(alp_ratio), f(tbil_ratio), f(r_ratio)
    if r is None and alt is not None and alp:
        r = alt / alp
    enzyme_abnormal = (alt is not None and alt >= abnormal) or (alp is not None and alp >= abnormal)
    if enzyme_abnormal and r is not None:
        return "hepatocellular" if r >= 5 else "cholestatic" if r <= 2 else "mixed"
    if tb is not None and tb >= abnormal:
        return "isolated_bilirubin"
    return "unclassifiable"


def pattern_frame(features: pd.DataFrame) -> pd.Series:
    g = lambda c: features[c] if c in features else pd.Series(np.nan, index=features.index)
    return pd.Series([r_pattern(a, b, c, d) for a, b, c, d in zip(g("ALT_ratio"), g("ALP_ratio"), g("TBIL_ratio"), g("R_ratio"))],
                     index=features.index, name="liver_pattern")


def _matrix(df: pd.DataFrame, cols):
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    for c in cols:
        v = X[c].dropna()
        # log2 for strictly positive, non-binary columns (ULN ratios, lab values); slopes and flags stay linear
        if len(v) and (v > 0).all() and v.nunique() > 2:
            X[c] = np.log2(X[c].clip(lower=0.05))
    X = X.fillna(X.median())
    sd = X.std(ddof=0).replace(0, 1)
    return ((X - X.mean()) / sd).to_numpy()


def gmm_explore(df: pd.DataFrame, cols, ks=range(2, 9), n_boot=50, seed=0, min_n=30, min_observed=0.5):
    """Supplementary: BIC-selected Gaussian mixture plus bootstrap ARI stability."""
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture
    # keep variables observed in at least half of the rows; median imputation beyond that would dominate
    cols = [c for c in cols if c in df and pd.to_numeric(df[c], errors="coerce").notna().mean() >= min_observed]
    out = {"method": "GaussianMixture(diag), log2 for positive labs, K by BIC, bootstrap ARI",
           "n": int(len(df)), "features": cols, "stable_threshold": STABLE_ARI}
    if len(df) < min_n or not cols:
        return {**out, "status": "insufficient", "used": False}, None
    X = _matrix(df, cols)
    fit = lambda k, A, s: GaussianMixture(k, covariance_type="diag", n_init=3, random_state=s).fit(A)
    bic = []
    for k in ks:
        if k >= len(X):
            break
        bic.append({"k": int(k), "bic": float(fit(k, X, seed).bic(X))})
    k_best = min(bic, key=lambda r: r["bic"])["k"]
    ref = fit(k_best, X, seed).predict(X)
    rng = np.random.default_rng(seed)
    aris = []
    for b in range(n_boot):
        idx = rng.integers(0, len(X), len(X))
        aris.append(adjusted_rand_score(ref, fit(k_best, X[idx], seed + b + 1).predict(X)))
    q = np.quantile(aris, [.25, .5, .75])
    stable = bool(q[1] >= STABLE_ARI)
    out.update(status="ok", bic=bic, k_selected=int(k_best), ari_q25=float(q[0]), ari_median=float(q[1]),
               ari_q75=float(q[2]), n_boot=n_boot, used=stable,
               sizes=[int(s) for s in np.bincount(ref, minlength=k_best)])
    return out, ref
