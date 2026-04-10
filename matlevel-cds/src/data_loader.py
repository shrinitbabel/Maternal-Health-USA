"""Load and clean the Maternal Health Equity Linked Dataset (PRJ-6250 V2)."""
from __future__ import annotations
import warnings

import numpy as np
import pandas as pd

from . import config as C


def _strip_totals_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the 'US Total' and per-state 'State Total' summary rows that ship
    in the CSV. State-total rows use a synthetic state_fips of '00'."""
    county = df["county"].astype(str).str.strip().str.lower()
    mask_total_rows = county.isin({"us total", "state total"}) | county.str.contains("total", na=False)
    df = df.loc[~mask_total_rows].copy()
    return df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c == "county":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _fix_fips(df: pd.DataFrame) -> pd.DataFrame:
    df["fips"] = df["fips"].astype("Int64").astype(str).str.zfill(5)
    df["state_fips"] = df["fips"].str[:2]
    df = df.loc[df["state_fips"] != "00"].copy()
    return df


def _fix_travel_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """HCUP -1 sentinels -> NaN; impute with state median, fallback national median.
    Build a non-routable flag so models can learn from the suppression pattern."""
    new_cols = {}
    for col in C.TIME_COLS:
        nonrout = (df[col] == C.TIME_MISSING_SENTINEL).astype("int8")
        new_cols[f"{col}_nonroutable"] = nonrout
        clean = df[col].where(df[col] != C.TIME_MISSING_SENTINEL, np.nan)
        state_med = clean.groupby(df["state_fips"]).transform("median")
        imputed = clean.fillna(state_med).fillna(clean.median())
        new_cols[f"{col}_imputed"] = imputed
        df[col] = clean
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def load_raw() -> pd.DataFrame:
    # Row 1 of the CSV is a human-readable label row; row 2 is "US Total".
    df = pd.read_csv(C.RAW_CSV, low_memory=False, skiprows=[1])
    df.columns = [c.strip().lower() for c in df.columns]
    df = _strip_totals_rows(df)
    df = _coerce_numeric(df)
    df = _fix_fips(df)
    df = _fix_travel_and_impute(df)
    return df


def add_race_pcts(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw Decennial race counts into % of county population."""
    pop = df[C.POP_TOTAL].replace(0, np.nan)
    new_cols = {k: (df[v] / pop) * 100.0 for k, v in C.RACE_VARS.items()}
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def select_analytic(df: pd.DataFrame) -> pd.DataFrame:
    cols = (
        ["fips", "county", "state_fips", C.POP_TOTAL]
        + C.TIME_COLS
        + [f"{c}_imputed" for c in C.TIME_COLS]
        + [f"{c}_nonroutable" for c in C.TIME_COLS]
        + C.ALL_PRG
        + C.SVI_THEMES
        + [C.SVI_OVERALL]
        + C.SVI_EP
        + list(C.RACE_VARS.keys())
    )
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()


def load_clean() -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        df = load_raw()
        df = add_race_pcts(df)
        return select_analytic(df)


if __name__ == "__main__":
    d = load_clean()
    print(d.shape)
    print(d.head())
