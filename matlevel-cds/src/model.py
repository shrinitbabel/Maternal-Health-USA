"""Models for MatLevel-CDS.

Two complementary estimators:

1. **Negative Binomial GLM** (statsmodels) — interpretable, publishable
   association estimates of ACOG Level-III discordance on the SMM-proxy count,
   adjusted for SVI themes, demographics, urbanicity, reporting completeness,
   and an indicator for Level-III host counties (to remove the referral-center
   paradox). Offset = log(denominator) so the coefficient is a rate ratio.

2. **XGBoost regressor** — non-linear baseline for predicting the county-level
   SMM-proxy rate with spatial (leave-state-out) cross-validation. Serves as
   the engine behind the clinical scoring tool.
"""
from __future__ import annotations
import json
import pickle
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from . import config as C
from .analytic import (
    EXPOSURE_BINARY,
    EXPOSURE_CONTINUOUS,
    FEATURES_CONFOUNDERS,
    OFFSET,
    TARGET_COUNT,
    TARGET_PRIMARY,
    TARGETS_SECONDARY,
    build_analytic,
)


# -------------------------- Negative Binomial GLM ---------------------------

NB_PRIMARY_EXPOSURE = "gap60_level3"   # headline: >60 min from Level III


def fit_nb_glm(df: pd.DataFrame, exposure: str = NB_PRIMARY_EXPOSURE,
               outcome_count: str = TARGET_COUNT,
               offset_col: str = OFFSET) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit negative-binomial GLM:
        smm_count ~ exposure + confounders, offset = log(denom)
    """
    data = df[[outcome_count, offset_col, exposure] + FEATURES_CONFOUNDERS].dropna()
    formula = f"{outcome_count} ~ {exposure} + " + " + ".join(FEATURES_CONFOUNDERS)
    offset = np.log(data[offset_col].astype(float).values)
    model = smf.glm(
        formula=formula,
        data=data,
        family=sm.families.NegativeBinomial(alpha=1.0),
        offset=offset,
    )
    return model.fit(maxiter=200)


def nb_rate_ratio_table(res) -> pd.DataFrame:
    """Exponentiate NB-GLM coefficients into rate ratios with 95% CIs."""
    params = res.params
    conf = res.conf_int()
    out = pd.DataFrame({
        "term": params.index,
        "beta": params.values,
        "RR": np.exp(params.values),
        "RR_lo": np.exp(conf[0].values),
        "RR_hi": np.exp(conf[1].values),
        "pval": res.pvalues.values,
    })
    return out.round(4)


# ---------------------------- XGBoost predictor -----------------------------

@dataclass
class SpatialCVResult:
    target: str
    fold_r2: list
    fold_mae: list
    fold_rmse: list
    mean_r2: float
    mean_mae: float
    mean_rmse: float
    n_folds: int
    n_train_mean: float
    n_test_mean: float


def _xgb() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        min_child_weight=5,
        random_state=C.RANDOM_STATE,
        tree_method="hist",
        n_jobs=-1,
    )


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = (FEATURES_CONFOUNDERS + EXPOSURE_CONTINUOUS + EXPOSURE_BINARY
            + ["time3_imputed", "time2_imputed", "time1_imputed", "time0_imputed",
               "effective_access_tier_30min", "pop_decile"])
    # unique, preserve order
    seen = set()
    out = [c for c in cols if not (c in seen or seen.add(c))]
    return df[out].copy()


def spatial_cv_xgb(df: pd.DataFrame, target: str) -> tuple[SpatialCVResult, XGBRegressor]:
    """Leave-state-out cross-validation for the XGBoost regressor."""
    d = df.dropna(subset=[target, "state_fips"]).copy()
    X = feature_matrix(d)
    y = d[target].astype(float).values
    groups = d["state_fips"].values

    n_splits = min(C.N_SPLITS_STATE_CV, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    fold_r2, fold_mae, fold_rmse = [], [], []
    n_tr, n_te = [], []
    for tr_idx, te_idx in gkf.split(X, y, groups):
        mdl = _xgb()
        mdl.fit(X.iloc[tr_idx], y[tr_idx])
        pred = mdl.predict(X.iloc[te_idx])
        fold_r2.append(r2_score(y[te_idx], pred))
        fold_mae.append(mean_absolute_error(y[te_idx], pred))
        fold_rmse.append(float(np.sqrt(mean_squared_error(y[te_idx], pred))))
        n_tr.append(len(tr_idx))
        n_te.append(len(te_idx))

    # Final model fit on all data (for CDS scoring + SHAP)
    final = _xgb()
    final.fit(X, y)

    res = SpatialCVResult(
        target=target,
        fold_r2=[round(x, 4) for x in fold_r2],
        fold_mae=[round(x, 4) for x in fold_mae],
        fold_rmse=[round(x, 4) for x in fold_rmse],
        mean_r2=float(np.mean(fold_r2)),
        mean_mae=float(np.mean(fold_mae)),
        mean_rmse=float(np.mean(fold_rmse)),
        n_folds=n_splits,
        n_train_mean=float(np.mean(n_tr)),
        n_test_mean=float(np.mean(n_te)),
    )
    return res, final


# ------------------------- top-level orchestration -------------------------

def run() -> dict:
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = build_analytic()
    summary: dict = {}

    # --- Negative Binomial GLMs -------------------------------------------
    glm_tables = {}
    for exposure in ["gap30_level3", "gap60_level3", "gap90_level3", "log_time3"]:
        print(f"[nb-glm] outcome={TARGET_COUNT}  exposure={exposure}")
        res = fit_nb_glm(df, exposure=exposure)
        tbl = nb_rate_ratio_table(res)
        tbl.to_csv(C.TAB_DIR / f"nbglm_{exposure}.csv", index=False)
        glm_tables[exposure] = tbl
        # Summarize headline row (the exposure coefficient itself)
        row = tbl[tbl["term"] == exposure]
        if not row.empty:
            r = row.iloc[0]
            summary[f"RR_{exposure}"] = {
                "RR": float(r["RR"]),
                "CI95": [float(r["RR_lo"]), float(r["RR_hi"])],
                "pval": float(r["pval"]),
            }

    # --- XGBoost predictors: primary + secondaries ------------------------
    xgb_results = {}
    xgb_models = {}
    for target in [TARGET_PRIMARY] + TARGETS_SECONDARY:
        print(f"[xgb]  spatial-cv target={target}")
        try:
            res, mdl = spatial_cv_xgb(df, target)
        except Exception as e:
            print(f"  !! failed: {e}")
            continue
        xgb_results[target] = asdict(res)
        xgb_models[target] = mdl

    (C.TAB_DIR / "xgb_spatial_cv.json").write_text(json.dumps(xgb_results, indent=2))
    with open(C.MODEL_DIR / "xgb_models.pkl", "wb") as f:
        pickle.dump({
            "models": xgb_models,
            "feature_cols": list(feature_matrix(df).columns),
        }, f)

    (C.TAB_DIR / "nb_summary.json").write_text(json.dumps(summary, indent=2))
    print("[ok] NB summary ->", summary)
    return {"nb": summary, "xgb": xgb_results}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
