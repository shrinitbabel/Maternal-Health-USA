"""Outcome construction from HCUP PRG (pregnancy-related CCSR) categories.

Rationale
---------
HCUP county-level PRG variables are 3-year aggregate discharge *counts* (2017-2019)
with cell suppression where N < 11. No individual SMM flag is available, so we build
a **SMM-proxy composite** from PRG categories that map onto CDC's 21 SMM indicators:

  - prg018  Placental disorders            -> hemorrhage / abruption / accreta
  - prg019  Diabetes in pregnancy          -> DKA / severe diabetic complications
  - prg020  Hypertensive disorders          -> severe preeclampsia / eclampsia
  - prg023  Complications during childbirth -> intrapartum SMM indicators
  - prg027  Complications during puerperium -> sepsis / VTE / DIC / AFE

Denominator
-----------
There is no clean 'pregnancies' denominator; PRG codes are discharge counts and a
single pregnancy can contribute to multiple PRG categories. We use two denominators:

  (1) **SUM_PRG_ANY**: total reported PRG discharges across all categories
      -> gives "share of pregnancy-related discharges that are complication-coded"
  (2) **Women 15-44 (from p1_* adjusted via share of population in age groups)**:
      not available cleanly -> we fall back to county total population (POP_TOTAL)
      for per-capita rate sensitivity analyses only.

Primary outcome used downstream:
  smm_rate = 1000 * SUM_SMM_PROXY / SUM_PRG_ANY   (per 1,000 pregnancy discharges)

Secondary outcomes (individual):
  hdp_rate      = 1000 * prg020 / SUM_PRG_ANY     (hypertensive disorders of pregnancy)
  childbirth_cx = 1000 * prg023 / SUM_PRG_ANY     (childbirth complications)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C


def _safe_rowsum(df: pd.DataFrame, cols: list[str], min_non_null: int = 1) -> pd.Series:
    """Row-wise sum of count columns, treating NaN (HCUP suppressed) as unknown.
    Returns NaN if fewer than `min_non_null` values are observed in the row."""
    present = df[cols].notna().sum(axis=1)
    raw_sum = df[cols].sum(axis=1, min_count=min_non_null)
    raw_sum[present < min_non_null] = np.nan
    return raw_sum


def build_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Denominator: sum of all PRG discharge counts (require >=3 of 15 categories).
    df["sum_prg_any"] = _safe_rowsum(df, C.ALL_PRG, min_non_null=3)

    # Numerator: sum of SMM-proxy categories (require >=2 of 5).
    df["sum_smm_proxy"] = _safe_rowsum(df, C.SMM_PROXY_PRG, min_non_null=2)

    # Primary outcome: SMM-proxy rate per 1,000 pregnancy-related discharges.
    df["smm_rate_per1000"] = 1000.0 * df["sum_smm_proxy"] / df["sum_prg_any"]
    df.loc[df["sum_prg_any"] <= 0, "smm_rate_per1000"] = np.nan

    # Individual secondary outcomes (use only rows where that PRG is non-missing
    # and denominator is present).
    for prg, name in [
        ("prg020", "hdp_rate_per1000"),            # Hypertensive disorders
        ("prg023", "childbirth_cx_rate_per1000"),  # Childbirth complications
        ("prg018", "placenta_rate_per1000"),       # Placental disorders
        ("prg027", "puerperal_rate_per1000"),      # Puerperal complications
        ("prg019", "dm_rate_per1000"),             # Diabetes in pregnancy
    ]:
        rate = 1000.0 * df[prg] / df["sum_prg_any"]
        rate.loc[df[prg].isna() | (df["sum_prg_any"] <= 0)] = np.nan
        df[name] = rate

    # Raw count versions (for Poisson / NegBin GLMs with offset(log(denom))).
    df["smm_proxy_count"] = df["sum_smm_proxy"]
    df["denom_count"] = df["sum_prg_any"]

    return df


if __name__ == "__main__":
    from .data_loader import load_clean
    d = build_outcomes(load_clean())
    print(d[["fips", "county", "smm_rate_per1000", "hdp_rate_per1000",
             "childbirth_cx_rate_per1000", "denom_count"]].describe())
