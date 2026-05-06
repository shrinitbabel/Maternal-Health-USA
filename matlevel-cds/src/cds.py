"""MatLevel-CDS: clinical decision support scoring library.

Operationalizes ACOG/SMFM 'Levels of Maternal Care' (Committee Opinion 726,
reaffirmed 2023) into a point-of-care tool. Inputs at the first prenatal
visit are a county FIPS code and a structured set of risk factors. Outputs
are:
  - the ACOG-recommended level of care for this patient
  - actual travel time from her county to the nearest facility at each level
  - 'concordance gap' = recommended level vs. closest available within 30 min
  - a risk-adjusted complication score from the pre-trained XGBoost model
  - an action bundle (referral / telehealth MFM / social work / relocation)

This module contains only the rules + scoring; no web server.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from functools import lru_cache
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config as C
from .analytic import build_analytic
from .model import feature_matrix


# -------------------------- ACOG risk-factor rules ---------------------------

class ACOGLevel(IntEnum):
    """ACOG/SMFM Levels of Maternal Care.

    Level 0: Accredited birth center / home birth (low-risk)
    Level I:  Basic care       (uncomplicated term pregnancies)
    Level II: Specialty care    (moderate risk, e.g. preterm 32-36w,
                                 mild preeclampsia)
    Level III: Subspecialty     (severe maternal/fetal conditions,
                                  <32w preterm, severe preeclampsia)
    Level IV: Regional perinatal center
              (highest acuity: placenta accreta spectrum, ECMO, cardiac surgery)
    """
    NONE = 0
    LEVEL_I = 1
    LEVEL_II = 2
    LEVEL_III = 3
    LEVEL_IV = 4


@dataclass
class RiskFactors:
    """Structured, binary/ordinal risk factors collected at intake.
    Defaults = None/False -> healthy baseline. Fields below reflect ACOG
    COG-726 risk stratification (simplified for the CDS)."""
    # Medical / chronic
    chronic_hypertension: bool = False
    pregestational_diabetes: bool = False       # Type 1 or pre-existing type 2
    renal_disease: bool = False
    cardiac_disease_class_I_II: bool = False
    cardiac_disease_class_III_IV: bool = False  # Level IV indication
    severe_pulmonary_disease: bool = False
    hemoglobinopathy_severe: bool = False       # e.g. sickle cell with crisis hx
    autoimmune_active: bool = False             # SLE/APS active
    seizure_disorder_uncontrolled: bool = False

    # Obstetric / current pregnancy
    multiple_gestation: bool = False            # twins etc.
    higher_order_multiples: bool = False        # triplets+ -> Level III
    placenta_previa: bool = False
    placenta_accreta_spectrum: bool = False     # Level IV indication
    severe_preeclampsia: bool = False
    eclampsia_hx: bool = False
    preterm_labor_current: bool = False
    anticipated_delivery_lt_32w: bool = False   # Level III/IV
    fetal_growth_restriction_severe: bool = False
    fetal_anomaly_needing_surgery: bool = False # Level III/IV
    prior_severe_maternal_morbidity: bool = False

    # Historical
    prior_cesarean_trial_of_labor: bool = False
    prior_stillbirth: bool = False
    prior_preeclampsia: bool = False

    # Patient-stated
    bmi: Optional[float] = None
    age_years: Optional[float] = None
    parity: Optional[int] = None


def recommend_level(rf: RiskFactors) -> tuple[ACOGLevel, list[str]]:
    """Return the minimum-recommended ACOG level of care and the rule hits."""
    hits: list[str] = []
    level = ACOGLevel.LEVEL_I

    # ---- Level IV triggers (highest acuity) --------------------------------
    lvl4_rules = [
        ("cardiac_disease_class_III_IV", "NYHA III/IV cardiac disease"),
        ("placenta_accreta_spectrum", "Placenta accreta spectrum"),
        ("fetal_anomaly_needing_surgery", "Fetal anomaly requiring in-utero/neonatal surgery"),
    ]
    for attr, label in lvl4_rules:
        if getattr(rf, attr):
            hits.append(label)
            level = max(level, ACOGLevel.LEVEL_IV)

    # ---- Level III triggers ------------------------------------------------
    lvl3_rules = [
        ("anticipated_delivery_lt_32w", "Anticipated delivery <32 weeks"),
        ("severe_preeclampsia", "Severe preeclampsia"),
        ("eclampsia_hx", "Prior eclampsia"),
        ("higher_order_multiples", "Triplets or higher-order multiples"),
        ("placenta_previa", "Placenta previa"),
        ("severe_pulmonary_disease", "Severe pulmonary disease"),
        ("renal_disease", "Significant renal disease"),
        ("hemoglobinopathy_severe", "Severe hemoglobinopathy"),
        ("autoimmune_active", "Active autoimmune disease (SLE/APS)"),
        ("prior_severe_maternal_morbidity", "Prior severe maternal morbidity"),
        ("fetal_growth_restriction_severe", "Severe fetal growth restriction"),
    ]
    for attr, label in lvl3_rules:
        if getattr(rf, attr):
            hits.append(label)
            level = max(level, ACOGLevel.LEVEL_III)

    # ---- Level II triggers -------------------------------------------------
    lvl2_rules = [
        ("chronic_hypertension", "Chronic hypertension"),
        ("pregestational_diabetes", "Pregestational diabetes"),
        ("cardiac_disease_class_I_II", "NYHA I/II cardiac disease"),
        ("multiple_gestation", "Multiple gestation (twins)"),
        ("preterm_labor_current", "Current preterm labor"),
        ("prior_cesarean_trial_of_labor", "Prior cesarean, TOLAC candidate"),
        ("seizure_disorder_uncontrolled", "Poorly controlled seizure disorder"),
        ("prior_preeclampsia", "History of preeclampsia"),
        ("prior_stillbirth", "Prior stillbirth"),
    ]
    for attr, label in lvl2_rules:
        if getattr(rf, attr):
            hits.append(label)
            level = max(level, ACOGLevel.LEVEL_II)

    # ---- Soft Level II uplift from BMI/age --------------------------------
    if rf.bmi is not None and rf.bmi >= 40:
        hits.append(f"Class-III obesity (BMI {rf.bmi:.1f})")
        level = max(level, ACOGLevel.LEVEL_II)
    if rf.age_years is not None and rf.age_years >= 40:
        hits.append(f"Advanced maternal age >=40 (age {rf.age_years:.0f})")
        level = max(level, ACOGLevel.LEVEL_II)

    return level, hits


# --------------------- County lookup from the linked dataset -----------------

@lru_cache(maxsize=1)
def _county_lookup() -> pd.DataFrame:
    df = build_analytic()
    # Keep a lean reference frame for the CDS.
    cols = [
        "fips", "county", "state_fips", "time0_imputed", "time1_imputed",
        "time2_imputed", "time3_imputed",
        "rpl_themes", "rpl_theme1", "rpl_theme2", "rpl_theme3", "rpl_theme4",
        "pct_black", "pct_aian", "pct_minority", "svi_quartile", "pop_decile",
        "effective_access_tier_30min", "maternity_desert",
        "is_level3_host",
    ]
    return df[cols].set_index("fips")


def _ordinal_time_to_level(row: pd.Series) -> dict:
    """Return the travel time (min) to each ACOG level for this county.
    Level IV cannot be distinguished from Level III in the source data
    (AHA survey does not separate them), so Level IV = Level III time."""
    return {
        "LEVEL_I": float(row["time1_imputed"]),
        "LEVEL_II": float(row["time2_imputed"]),
        "LEVEL_III": float(row["time3_imputed"]),
        "LEVEL_IV": float(row["time3_imputed"]),
    }


def _closest_within(times: dict, minutes: float) -> Optional[str]:
    reachable = [(lvl, t) for lvl, t in times.items() if t <= minutes]
    if not reachable:
        return None
    # Pick highest level reachable.
    reachable.sort(key=lambda x: ACOGLevel[x[0]].value, reverse=True)
    return reachable[0][0]


# ---------------- Risk-adjusted complication score ---------------------------

@lru_cache(maxsize=1)
def _scoring_model():
    """Legacy XGBoost loader. Retained ONLY so manuscript-evidence modules
    (src.evaluate, src.gnn_eval) can train and benchmark the XGBoost
    baseline. NOT called by the production /recommend or /access path.

    If outputs/models/xgb_models.pkl is absent, returns ({}, []) — production
    must not depend on this anymore."""
    p = C.MODEL_DIR / "xgb_models.pkl"
    if not p.exists():
        return {}, []
    try:
        with open(p, "rb") as f:
            bundle = pickle.load(f)
        return bundle["models"], bundle["feature_cols"]
    except Exception:
        return {}, []


@lru_cache(maxsize=1)
def _gnn_predictions() -> dict:
    """Return the cached GraphSAGE per-county predictions, or {} if absent.

    GraphSAGE is the **primary** scorer for all six CDS outcomes (SMM, HDP,
    childbirth, placenta, puerperal, DM). It wins or ties XGBoost on
    calibration, ECE, and stability across the head-to-head; on the two
    residential-access-driven outcomes (puerperal, HDP) it also wins R^2,
    Pearson, MAE, and AUROC. XGBoost predictions are kept as a fallback in
    case the GNN cache fails to load. See src.gnn_deploy for training and
    src.gnn_eval for the apples-to-apples comparison.
    """
    import json
    p = C.MODEL_DIR / "gnn_predictions.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _county_complication_score(fips: str) -> dict:
    """Return GraphSAGE-predicted complication rates for the county.

    GraphSAGE (8-NN spatial graph, full cohort training) is the production
    scorer for all six CDS outcomes. XGBoost has been retired from the request
    path; if you need to inspect the legacy XGBoost model, use src.gnn_eval
    which still trains it as a manuscript head-to-head baseline. If the GNN
    cache is missing for some target, that target is simply omitted from the
    response — this lets the UI render gracefully.
    """
    return dict(_gnn_predictions().get("predictions", {})
                .get(str(fips).zfill(5), {}))


# -------------------------- Top-level CDS function --------------------------

@dataclass
class CDSRecommendation:
    fips: str
    county: str
    state_fips: str
    patient_risk_factors: list[str]
    recommended_level: str
    recommended_level_int: int
    travel_times_min: dict
    closest_within_30min: Optional[str]
    closest_within_60min: Optional[str]
    concordance: str         # "concordant", "gap", "severe_gap"
    gap_minutes: Optional[float]
    svi_percentile: Optional[float]
    svi_quartile: Optional[int]
    maternity_desert: bool
    is_level3_host: bool
    county_model_scores: dict = field(default_factory=dict)
    action_bundle: list[str] = field(default_factory=list)
    disclaimer: str = (
        "This tool is a decision aid operationalizing ACOG/SMFM Levels of "
        "Maternal Care at the county level and is not a substitute for "
        "clinical judgement. Travel times are county-centroid estimates from "
        "the Maternal Health Equity Linked Datasets (PRJ-6250)."
    )


def _build_action_bundle(level: ACOGLevel, gap_min: Optional[float],
                         svi_q: Optional[int], desert: bool) -> list[str]:
    actions: list[str] = []
    if level >= ACOGLevel.LEVEL_III:
        actions.append("Refer to Maternal-Fetal Medicine at a Level III/IV "
                       "center for co-management")
        actions.append("Coordinate delivery-hospital assignment by 28 weeks "
                       "(earlier if anticipated preterm)")
        if gap_min is not None and gap_min > 60:
            actions.append(f"Patient is ~{gap_min:.0f} min from the nearest "
                           "Level III facility: arrange antepartum transfer "
                           "plan, lodging near delivery hospital, or "
                           "telehealth MFM co-management")
    elif level == ACOGLevel.LEVEL_II:
        actions.append("Deliver at a Level II facility or higher; confirm "
                       "local hospital capability (24/7 OB anesthesia, "
                       "blood bank, NICU availability)")
    else:
        actions.append("Routine prenatal care; escalate if risk factors "
                       "develop")

    if svi_q == 4:
        actions.append("High social vulnerability quartile: engage social "
                       "work early for transportation, housing, and "
                       "insurance barriers")
    if desert:
        actions.append("County meets 'maternity care desert' criteria: "
                       "counsel patient on early transfer of care and "
                       "emergency transport pathway")
    actions.append("Counsel on recognition and immediate presentation for "
                   "postpartum warning signs (this is the outcome most "
                   "sensitive to geographic access in our national analysis)")
    return actions


def recommend(fips: str, risk: RiskFactors) -> CDSRecommendation:
    fips = str(fips).zfill(5)
    lookup = _county_lookup()
    if fips not in lookup.index:
        raise KeyError(f"FIPS {fips} not found in cohort")
    row = lookup.loc[fips]

    level, hits = recommend_level(risk)
    times = _ordinal_time_to_level(row)

    within30 = _closest_within(times, 30)
    within60 = _closest_within(times, 60)

    # Concordance: is the recommended level reachable within 30 min?
    recommended_time = times.get(level.name, float("nan"))
    if np.isnan(recommended_time):
        concordance = "unknown"
        gap = None
    elif recommended_time <= 30:
        concordance = "concordant"
        gap = recommended_time - 30 if recommended_time > 30 else 0.0
    elif recommended_time <= 60:
        concordance = "gap"
        gap = recommended_time - 30
    else:
        concordance = "severe_gap"
        gap = recommended_time - 30

    svi_q = int(row["svi_quartile"]) if pd.notna(row["svi_quartile"]) else None
    svi_pc = float(row["rpl_themes"]) if pd.notna(row["rpl_themes"]) else None
    desert = bool(row["maternity_desert"])
    actions = _build_action_bundle(level, gap, svi_q, desert)

    try:
        scores = _county_complication_score(fips)
    except Exception:
        scores = {}

    return CDSRecommendation(
        fips=fips,
        county=str(row["county"]),
        state_fips=str(row["state_fips"]),
        patient_risk_factors=hits,
        recommended_level=level.name,
        recommended_level_int=int(level.value),
        travel_times_min={k: round(v, 1) for k, v in times.items()},
        closest_within_30min=within30,
        closest_within_60min=within60,
        concordance=concordance,
        gap_minutes=round(gap, 1) if gap is not None else None,
        svi_percentile=round(svi_pc, 3) if svi_pc is not None else None,
        svi_quartile=svi_q,
        maternity_desert=desert,
        is_level3_host=bool(row["is_level3_host"]),
        county_model_scores={k: round(v, 2) for k, v in scores.items()},
        action_bundle=actions,
    )


# --------------------------- Quick demo harness -----------------------------

def _demo():
    import json
    lookup = _county_lookup()
    # Pick three illustrative counties: one urban, one rural, one AIAN-heavy.
    picks = []
    urban = lookup.sort_values("time3_imputed").head(1).index[0]
    rural = lookup.sort_values("time3_imputed", ascending=False).head(1).index[0]
    med = lookup.iloc[(lookup["time3_imputed"] - lookup["time3_imputed"].median()).abs().argsort()[:1]].index[0]
    picks = [("Urban (closest to Level III)", urban),
             ("Rural (farthest from Level III)", rural),
             ("Median-access county", med)]

    high_risk = RiskFactors(
        chronic_hypertension=True,
        pregestational_diabetes=True,
        prior_severe_maternal_morbidity=True,
        bmi=41.0, age_years=38,
    )
    low_risk = RiskFactors(age_years=27, bmi=26)

    for label, fips in picks:
        for name, rf in [("HIGH-risk patient", high_risk),
                         ("LOW-risk patient", low_risk)]:
            r = recommend(fips, rf)
            print(f"\n=== {label} | FIPS {fips} | {r.county} | {name} ===")
            print(json.dumps(asdict(r), indent=2, default=str))


if __name__ == "__main__":
    _demo()
