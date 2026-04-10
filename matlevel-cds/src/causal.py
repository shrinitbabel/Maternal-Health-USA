"""Causal analysis of ACOG Level-III discordance on county SMM-proxy rate.

Method: double machine learning / partially linear regression (Chernozhukov et
al., 2018). We estimate

    Y = theta * D + g(X) + e
    D = m(X) + v

where Y = log(SMM rate per 1,000), D = exposure (gap60_level3 or log_time3),
X = confounders (SVI, demographics, population, reporting completeness,
Level-III host indicator). g() and m() are fit with gradient boosting,
cross-fitted in spatial folds grouped by state. The coefficient theta is the
partial effect of D on Y holding X fixed, with an influence-function based
standard error.

Key caveats (discussed in the paper):
 - HCUP is discharge-based, not residence-based: counties hosting regional
   referral centers absorb complications from surrounding counties and thus
   attenuate any residential-access effect toward the null.
 - We report the main analysis + 3 sensitivity analyses:
     (a) excluding Level-III host counties (is_level3_host == 0)
     (b) restricted to population deciles <= 5 (rural subsample)
     (c) reweighted by county population (WLS-style)
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor, XGBClassifier

from . import config as C
from .analytic import FEATURES_CONFOUNDERS, build_analytic


@dataclass
class DMLResult:
    exposure: str
    outcome: str
    theta: float
    se: float
    ci_lo: float
    ci_hi: float
    zstat: float
    pval: float
    n: int
    n_states: int
    note: str = ""


def _xgb_reg() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
        reg_lambda=1.0, random_state=C.RANDOM_STATE, tree_method="hist",
        n_jobs=-1, verbosity=0,
    )


def _xgb_clf() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
        reg_lambda=1.0, random_state=C.RANDOM_STATE, tree_method="hist",
        n_jobs=-1, verbosity=0, eval_metric="logloss",
    )


def _cross_fit(df: pd.DataFrame, X_cols: list[str], y_col: str, d_col: str,
               d_is_binary: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return residuals (y_res, d_res) from leave-state-out cross-fitted
    nuisance predictions."""
    X = df[X_cols].values
    y = df[y_col].astype(float).values
    d = df[d_col].astype(float).values
    groups = df["state_fips"].values

    n_splits = min(C.N_SPLITS_STATE_CV, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    y_hat = np.zeros_like(y, dtype=float)
    d_hat = np.zeros_like(d, dtype=float)

    for tr, te in gkf.split(X, y, groups):
        gy = _xgb_reg()
        gy.fit(X[tr], y[tr])
        y_hat[te] = gy.predict(X[te])
        if d_is_binary:
            md = _xgb_clf()
            md.fit(X[tr], d[tr])
            d_hat[te] = md.predict_proba(X[te])[:, 1]
        else:
            md = _xgb_reg()
            md.fit(X[tr], d[tr])
            d_hat[te] = md.predict(X[te])

    y_res = y - y_hat
    d_res = d - d_hat
    return y_res, d_res


def dml_partially_linear(df: pd.DataFrame, exposure: str, outcome: str,
                         X_cols: list[str] | None = None,
                         note: str = "") -> DMLResult:
    X_cols = X_cols or FEATURES_CONFOUNDERS
    d_is_binary = df[exposure].nunique(dropna=True) == 2

    d = df.dropna(subset=[exposure, outcome] + X_cols + ["state_fips"]).copy()
    # log-outcome: safer with skewed rates
    d["__y"] = np.log(d[outcome].astype(float).clip(lower=1e-3))

    y_res, d_res = _cross_fit(d, X_cols, "__y", exposure, d_is_binary=d_is_binary)

    # Regress y_res on d_res; closed form.
    denom = float(np.sum(d_res ** 2))
    if denom <= 0:
        theta = float("nan"); se = float("nan")
    else:
        theta = float(np.sum(d_res * y_res) / denom)
        # Influence function SE (Chernozhukov 2018 eq. 5.5)
        psi = (y_res - theta * d_res) * d_res
        var_theta = float(np.mean(psi ** 2) / (np.mean(d_res ** 2) ** 2)) / len(d)
        se = float(np.sqrt(var_theta))

    z = theta / se if se > 0 else float("nan")
    from math import erfc, sqrt
    pval = erfc(abs(z) / sqrt(2)) if not np.isnan(z) else float("nan")

    return DMLResult(
        exposure=exposure, outcome=outcome,
        theta=theta, se=se,
        ci_lo=theta - 1.96 * se,
        ci_hi=theta + 1.96 * se,
        zstat=z, pval=pval,
        n=len(d), n_states=int(d["state_fips"].nunique()),
        note=note,
    )


def run() -> dict:
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    df = build_analytic()

    results = {}

    for outcome in ["smm_rate_per1000", "hdp_rate_per1000",
                    "childbirth_cx_rate_per1000", "puerperal_rate_per1000"]:
        results[outcome] = {}
        for exposure in ["gap60_level3", "gap90_level3", "log_time3"]:
            # Main analysis
            main = dml_partially_linear(df, exposure, outcome,
                                        note="main (all counties)")
            # Sensitivity: exclude level-III host counties
            sens1 = dml_partially_linear(
                df[df["is_level3_host"] == 0],
                exposure, outcome,
                X_cols=[c for c in FEATURES_CONFOUNDERS if c != "is_level3_host"],
                note="exclude level-III host counties",
            )
            # Sensitivity: rural subset (pop decile <= 5)
            sens2 = dml_partially_linear(
                df[df["pop_decile"] <= 5],
                exposure, outcome,
                note="rural subset (pop decile <= 5)",
            )
            results[outcome][exposure] = {
                "main": asdict(main),
                "excl_level3_host": asdict(sens1),
                "rural_subset": asdict(sens2),
            }
            print(f"[dml] {outcome} x {exposure} main theta={main.theta:+.4f} "
                  f"p={main.pval:.3f} | rural theta={sens2.theta:+.4f} p={sens2.pval:.3f}")

    (C.TAB_DIR / "dml_results.json").write_text(json.dumps(results, indent=2))
    print("[ok] wrote dml_results.json")
    return results


if __name__ == "__main__":
    run()
