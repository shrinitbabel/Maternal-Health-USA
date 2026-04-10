"""Pydantic request/response schemas for the MatLevel-CDS API."""
from __future__ import annotations
from typing import Optional

from pydantic import BaseModel, Field


class RiskFactorsIn(BaseModel):
    """Structured risk factors collected at the first prenatal visit.
    Defaults = healthy baseline; send only the ones that are positive."""

    # Medical / chronic
    chronic_hypertension: bool = False
    pregestational_diabetes: bool = False
    renal_disease: bool = False
    cardiac_disease_class_I_II: bool = False
    cardiac_disease_class_III_IV: bool = False
    severe_pulmonary_disease: bool = False
    hemoglobinopathy_severe: bool = False
    autoimmune_active: bool = False
    seizure_disorder_uncontrolled: bool = False

    # Obstetric / current pregnancy
    multiple_gestation: bool = False
    higher_order_multiples: bool = False
    placenta_previa: bool = False
    placenta_accreta_spectrum: bool = False
    severe_preeclampsia: bool = False
    eclampsia_hx: bool = False
    preterm_labor_current: bool = False
    anticipated_delivery_lt_32w: bool = False
    fetal_growth_restriction_severe: bool = False
    fetal_anomaly_needing_surgery: bool = False
    prior_severe_maternal_morbidity: bool = False

    # Historical
    prior_cesarean_trial_of_labor: bool = False
    prior_stillbirth: bool = False
    prior_preeclampsia: bool = False

    # Patient-stated
    bmi: Optional[float] = Field(default=None, ge=10, le=80)
    age_years: Optional[float] = Field(default=None, ge=10, le=60)
    parity: Optional[int] = Field(default=None, ge=0, le=20)


class RecommendRequest(BaseModel):
    fips: str = Field(..., min_length=4, max_length=5,
                      description="5-digit county FIPS code")
    risk: RiskFactorsIn = Field(default_factory=RiskFactorsIn)


class CountyAccessOut(BaseModel):
    fips: str
    county: str
    state_fips: str
    time_to_level_1_min: Optional[float]
    time_to_level_2_min: Optional[float]
    time_to_level_3_min: Optional[float]
    svi_percentile: Optional[float]
    svi_quartile: Optional[int]
    maternity_desert: bool
    is_level3_host: bool
    gap30_level3: bool
    gap60_level3: bool
    gap90_level3: bool
    county_model_scores: dict = Field(default_factory=dict)


class RecommendResponse(BaseModel):
    fips: str
    county: str
    state_fips: str
    patient_risk_factors: list[str]
    recommended_level: str
    recommended_level_int: int
    travel_times_min: dict
    closest_within_30min: Optional[str]
    closest_within_60min: Optional[str]
    concordance: str
    gap_minutes: Optional[float]
    svi_percentile: Optional[float]
    svi_quartile: Optional[int]
    maternity_desert: bool
    is_level3_host: bool
    county_model_scores: dict
    action_bundle: list[str]
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    version: str
    counties_loaded: int
    model_loaded: bool
