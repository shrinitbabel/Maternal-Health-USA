"""Assemble the final analytic frame for modeling and CDS."""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C
from .data_loader import load_clean
from .features import build_features
from .outcomes import build_outcomes


# PRG categories that HCUP reported for each county: we use this as a
# 'reporting completeness' indicator and to flag referral-center counties.
def _reporting_completeness(df: pd.DataFrame) -> pd.Series:
    return df[C.ALL_PRG].notna().sum(axis=1) / len(C.ALL_PRG)


def _is_level3_host(df: pd.DataFrame) -> pd.Series:
    """A county is considered to host a Level III OB facility if the travel
    time to Level III is <=5 minutes (county centroid is effectively on site).
    These are the referral centers that receive high-risk transfers."""
    t3 = df["time3_imputed"]
    return ((t3.notna()) & (t3 <= 5)).astype("int8")


def build_analytic() -> pd.DataFrame:
    df = load_clean()
    df = build_outcomes(df)
    df = build_features(df)
    df["reporting_completeness"] = _reporting_completeness(df)
    df["is_level3_host"] = _is_level3_host(df)

    # Log population (for the rural/urban covariate)
    df["log_pop"] = np.log1p(df[C.POP_TOTAL])

    # % minority: prefer CDC SVI's own definition (all persons except
    # white non-Hispanic) which is already Hispanic-adjusted.
    df["pct_minority"] = df["ep_minrty"].clip(lower=0, upper=100)

    # Filter to the primary analytic cohort: must have denominator and SVI
    cohort = df[df["denom_count"].notna() & df["rpl_themes"].notna()].copy()
    cohort = cohort.reset_index(drop=True)
    return cohort


# ---- Column groupings for modeling -------------------------------------------------
FEATURES_CONFOUNDERS = [
    "log_pop",
    "rpl_theme1",        # SVI socioeconomic
    "rpl_theme2",        # SVI household composition/disability
    "rpl_theme3",        # SVI minority/language
    "rpl_theme4",        # SVI housing/transportation
    "pct_black",
    "pct_aian",
    "pct_asian",
    "reporting_completeness",
    "is_level3_host",
]

EXPOSURE_CONTINUOUS = ["log_time3"]
EXPOSURE_BINARY = ["gap30_level3", "gap60_level3", "gap90_level3"]
EXPOSURE_ORDINAL = ["loc_discordance_tier"]

TARGET_PRIMARY = "smm_rate_per1000"
TARGET_COUNT = "smm_proxy_count"
OFFSET = "denom_count"

TARGETS_SECONDARY = [
    "hdp_rate_per1000",
    "childbirth_cx_rate_per1000",
    "placenta_rate_per1000",
    "puerperal_rate_per1000",
    "dm_rate_per1000",
]


if __name__ == "__main__":
    d = build_analytic()
    print("analytic cohort shape:", d.shape)
    print("states:", d["state_fips"].nunique())
    print("level3 host counties:", int(d["is_level3_host"].sum()))
    print("reporting completeness summary:")
    print(d["reporting_completeness"].describe().round(3))
