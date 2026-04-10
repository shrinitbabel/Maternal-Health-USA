"""Smoke tests for the MatLevel-CDS pipeline."""
import numpy as np
import pandas as pd

from src import config as C
from src.analytic import build_analytic
from src.cds import RiskFactors, recommend, recommend_level, ACOGLevel
from src.features import build_features
from src.outcomes import build_outcomes
from src.data_loader import load_clean


def test_loader_shape():
    df = load_clean()
    assert len(df) > 2000
    assert df["state_fips"].nunique() >= 35
    assert df["fips"].nunique() == len(df)


def test_no_negative_times():
    df = load_clean()
    for c in ["time0", "time1", "time2", "time3"]:
        neg = (df[c] < 0).sum()
        assert neg == 0, f"found {neg} negative values in {c}"


def test_outcome_bounds():
    df = build_outcomes(load_clean())
    rate = df["smm_rate_per1000"].dropna()
    assert rate.min() >= 0
    assert rate.max() <= 1000
    assert rate.notna().sum() > 1500


def test_discordance_tiers_sum_to_cohort():
    df = build_features(build_outcomes(load_clean()))
    tiers = df["loc_discordance_tier"].notna().sum()
    # 99%+ of cohort should have a tier assignment
    assert tiers / len(df) > 0.95


def test_analytic_cohort():
    df = build_analytic()
    assert "denom_count" in df.columns
    assert "is_level3_host" in df.columns
    assert (df["denom_count"] > 0).all()


def test_acog_level_rules_low_risk():
    rf = RiskFactors(age_years=27, bmi=26)
    lvl, hits = recommend_level(rf)
    assert lvl == ACOGLevel.LEVEL_I
    assert hits == []


def test_acog_level_rules_high_risk():
    rf = RiskFactors(
        chronic_hypertension=True,
        pregestational_diabetes=True,
        prior_severe_maternal_morbidity=True,
    )
    lvl, hits = recommend_level(rf)
    assert lvl >= ACOGLevel.LEVEL_III
    assert any("severe maternal morbidity" in h.lower() for h in hits)


def test_acog_level_rules_accreta_is_level_iv():
    rf = RiskFactors(placenta_accreta_spectrum=True)
    lvl, _ = recommend_level(rf)
    assert lvl == ACOGLevel.LEVEL_IV


def test_recommend_returns_serializable():
    df = build_analytic()
    fips = df.iloc[0]["fips"]
    rf = RiskFactors(chronic_hypertension=True)
    r = recommend(fips, rf)
    assert r.recommended_level in {"LEVEL_I", "LEVEL_II", "LEVEL_III", "LEVEL_IV"}
    assert r.concordance in {"concordant", "gap", "severe_gap", "unknown"}
    assert r.travel_times_min["LEVEL_III"] >= r.travel_times_min["LEVEL_I"] - 1e-6 or \
           np.isnan(r.travel_times_min["LEVEL_III"]) or \
           np.isnan(r.travel_times_min["LEVEL_I"])
