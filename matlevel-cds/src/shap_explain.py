"""Per-patient SHAP explanations for the puerperal-rate XGBoost model.

Uses shap.TreeExplainer because the underlying model is XGBoost. TreeExplainer
is exact and returns in milliseconds, unlike KernelExplainer which takes
minutes for the same input. This is the difference between a working clinical
tool and a spinner that never completes.
"""
from __future__ import annotations
from typing import Callable, Optional
import pickle

import numpy as np
import shap

from . import config as C
from .analytic import build_analytic
from .model import feature_matrix

_TARGET = "puerperal_rate_per1000"


def _log(msg: str, sink: Optional[Callable[[str], None]] = None) -> None:
    print(msg)
    if sink is not None:
        try:
            sink(msg)
        except Exception:
            pass


def _load_model():
    with open(C.MODEL_DIR / "xgb_models.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle["models"][_TARGET], bundle["feature_cols"]


def explain_patient(fips: str,
                    log: Optional[Callable[[str], None]] = None) -> dict:
    """Compute exact TreeExplainer SHAP values for the given county.

    Returns a dict with the ordered feature contributions, the patient's
    feature values, the cohort baseline prediction, and the patient's
    predicted puerperal complication rate.
    """
    _log(f"[shap] loading model for {fips}", log)
    model, feat_cols = _load_model()

    _log("[shap] loading analytic cohort", log)
    df = build_analytic()
    row = df[df["fips"] == str(fips).zfill(5)]
    if row.empty:
        raise KeyError(f"FIPS {fips} not in cohort")

    _log("[shap] building patient feature vector", log)
    X_patient = (
        feature_matrix(row)
        .astype("float64")
        .fillna(0.0)[feat_cols]
    )

    _log("[shap] initializing TreeExplainer (fast, exact for XGBoost)", log)
    explainer = shap.TreeExplainer(model)

    _log("[shap] computing SHAP values", log)
    shap_values = explainer.shap_values(X_patient.values)
    shap_row = np.asarray(shap_values).reshape(-1)

    baseline = float(explainer.expected_value)
    prediction = float(model.predict(X_patient.values)[0])
    _log(f"[shap] baseline={baseline:.2f} prediction={prediction:.2f}", log)

    items = []
    for name, val, contrib in zip(feat_cols, X_patient.values[0], shap_row):
        items.append({
            "feature": name,
            "value": float(val),
            "shap": float(contrib),
        })
    items.sort(key=lambda d: abs(d["shap"]), reverse=True)
    _log("[shap] done", log)

    return {
        "fips": str(fips).zfill(5),
        "county": str(row.iloc[0]["county"]),
        "state_fips": str(row.iloc[0]["state_fips"]),
        "target": _TARGET,
        "baseline_prediction": round(baseline, 3),
        "patient_prediction": round(prediction, 3),
        "delta_from_baseline": round(prediction - baseline, 3),
        "features": items,
    }
