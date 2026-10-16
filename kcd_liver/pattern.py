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


def _prep_params(df: pd.DataFrame, cols):
    """Preprocessing learned on development data only (log2 choice, median, mean, SD)."""
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    logc = [c for c in cols if len(X[c].dropna()) and (X[c].dropna() > 0).all() and X[c].nunique() > 2]
    for c in logc:
        X[c] = np.log2(X[c].clip(lower=0.05))
    med = X.median()
    X = X.fillna(med)
    return {"cols": list(cols), "log_cols": logc, "median": med.to_dict(), "mean": X.mean().to_dict(),
            "sd": X.std(ddof=0).replace(0, 1).to_dict()}


def _apply_params(df: pd.DataFrame, P: dict) -> np.ndarray:
    X = df.reindex(columns=P["cols"]).apply(pd.to_numeric, errors="coerce")
    for c in P["log_cols"]:
        X[c] = np.log2(X[c].clip(lower=0.05))
    X = X.fillna(pd.Series(P["median"]))
    return ((X - pd.Series(P["mean"])) / pd.Series(P["sd"])).to_numpy()


def _matrix(df: pd.DataFrame, cols):
    return _apply_params(df, _prep_params(df, cols))


def assign_clusters(model: dict, df: pd.DataFrame) -> np.ndarray:
    """Apply a saved clustering model (from gmm_explore(return_model=True)) to new data, e.g. another hospital."""
    return model["gmm"].predict(_apply_params(df, model["prep"]))


def gmm_explore(df: pd.DataFrame, cols, ks=range(2, 9), n_boot=50, seed=0, min_n=30, min_observed=0.5, return_model=False):
    """Supplementary: BIC-selected Gaussian mixture plus bootstrap ARI stability."""
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture
    # keep variables observed in at least half of the rows; median imputation beyond that would dominate
    cols = [c for c in cols if c in df and pd.to_numeric(df[c], errors="coerce").notna().mean() >= min_observed]
    out = {"method": "GaussianMixture(diag), log2 for positive labs, K by BIC, bootstrap ARI",
           "n": int(len(df)), "features": cols, "stable_threshold": STABLE_ARI}
    if len(df) < min_n or not cols:
        return ({**out, "status": "insufficient", "used": False}, None) + ((None,) if return_model else ())
    prep = _prep_params(df, cols)
    X = _apply_params(df, prep)
    fit = lambda k, A, s: GaussianMixture(k, covariance_type="diag", n_init=3, random_state=s).fit(A)
    bic = []
    for k in ks:
        if k >= len(X):
            break
        bic.append({"k": int(k), "bic": float(fit(k, X, seed).bic(X))})
    k_best = min(bic, key=lambda r: r["bic"])["k"]
    ref_model = fit(k_best, X, seed)
    ref = ref_model.predict(X)
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
    if return_model:
        return out, ref, {"gmm": ref_model, "prep": prep, "k": int(k_best), "stable": stable, "kind": "gmm_cluster"}
    return out, ref
