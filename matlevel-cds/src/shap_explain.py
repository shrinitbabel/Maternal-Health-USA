"""Per-patient SHAP explanations for the puerperal-rate XGBoost model.

Takes the patient's county FIPS and risk profile, turns it into the 
exact feature vector the XGBoost model consumes, and streams progress 
while computing KernelExplainer SHAP values against a background 
sample of the cohort.

The tool is a decision aid; the SHAP values explain the county 
complication risk the model predicts, which is the context score 
shown to the clinician alongside the ACOG rule engine output.
"""
from __future__ import annotations
from typing import Callable, Optional
import pickle

import numpy as np
import pandas as pd
import shap

from . import config as C
from .analytic import build_analytic
from .model import feature_matrix

_TARGET = "puerperal_rate_per1000"
_N_BG = 40
_N_SAMPLES = 120

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
    """Compute SHAP values for the county feature vector of the given FIPS.

    Returns a dict with feature names, the patient's feature values, the 
    SHAP contribution of each feature to the predicted puerperal rate, 
    and the model's baseline (background expectation) prediction.
    """
    _log(f"[shap] loading model + cohort for {fips}", log)
    model, feat_cols = _load_model()
    df = build_analytic()

    row = df[df["fips"] == str(fips).zfill(5)]
    if row.empty:
        raise KeyError(f"FIPS {fips} not in cohort")

    _log("[shap] building patient feature vector", log)
    X_all = feature_matrix(df).astype("float64").fillna(0.0)
    X_all = X_all[feat_cols]
    X_patient = feature_matrix(row).astype("float64").fillna(0.0)[feat_cols]

    _log(f"[shap] sampling {_N_BG} background rows for KernelExplainer", log)
    bg = shap.sample(X_all, _N_BG, random_state=C.RANDOM_STATE)

    _log("[shap] initializing KernelExplainer (this is the slow step)", log)
    explainer = shap.KernelExplainer(model.predict, bg)

    _log(f"[shap] computing SHAP values with nsamples={_N_SAMPLES}", log)
    shap_values = explainer.shap_values(X_patient.values, nsamples=_N_SAMPLES)
    shap_row = np.asarray(shap_values).reshape(-1)
    _log("[shap] done computing", log)

    baseline = float(np.mean(explainer.expected_value))
    prediction = float(model.predict(X_patient.values)[0])

    items = []
    for name, val, contrib in zip(feat_cols, X_patient.values[0], shap_row):
        items.append({
            "feature": name,
            "value": float(val),
            "shap": float(contrib),
        })
    items.sort(key=lambda d: abs(d["shap"]), reverse=True)

    _log("[shap] sending result", log)
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
