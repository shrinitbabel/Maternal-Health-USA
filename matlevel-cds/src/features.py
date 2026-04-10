"""Feature engineering for MatLevel-CDS.

Key constructs
--------------
1. **ACOG Level of Care (LoC) discordance** — the exposure of interest.
   For every county we compute three 'discordance' flags:
       gap30_level3  = travel time to Level-III OB > 30 min (ACOG 'timely')
       gap60_level3  = travel time to Level-III OB > 60 min (trauma 'golden hour')
       gap90_level3  = travel time to Level-III OB > 90 min (rural threshold)
   plus an ordinal `loc_discordance_tier` 0/1/2/3.

2. **Effective maternal care access tier** — the lowest ACOG level of care reachable
   within 30 minutes (0 = none, 1 = basic, 2 = complicated, 3 = serious).

3. **Rurality proxy** — county total population deciles (1=most rural, 10=most urban).

4. **SVI quartile** — derived from CDC rpl_themes overall percentile.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C


def _tier_from_minutes(minutes: float) -> int:
    if pd.isna(minutes):
        return np.nan  # type: ignore[return-value]
    if minutes <= 30:
        return 0
    if minutes <= 60:
        return 1
    if minutes <= 90:
        return 2
    return 3


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Level-III discordance (the primary exposure) -----------------------
    t3 = df["time3_imputed"]
    df["gap30_level3"] = (t3 > 30).astype("int8")
    df["gap60_level3"] = (t3 > 60).astype("int8")
    df["gap90_level3"] = (t3 > 90).astype("int8")
    df["loc_discordance_tier"] = t3.apply(_tier_from_minutes).astype("Int64")

    # Log-transform travel times (for GLMs; +1 to handle zeros).
    for c in C.TIME_COLS:
        df[f"log_{c}"] = np.log1p(df[f"{c}_imputed"])

    # --- Effective access tier: highest-level-of-care within 30 min ---------
    def _effective_tier(row) -> int:
        for col, lvl in [("time3_imputed", 3), ("time2_imputed", 2),
                         ("time1_imputed", 1), ("time0_imputed", 0)]:
            v = row[col]
            if pd.notna(v) and v <= 30:
                return lvl
        return -1  # none within 30
    df["effective_access_tier_30min"] = df.apply(_effective_tier, axis=1).astype("Int8")

    # --- Rurality proxy: population deciles ---------------------------------
    df["pop_decile"] = pd.qcut(df[C.POP_TOTAL].rank(method="first"),
                               q=10, labels=False, duplicates="drop").astype("Int8")

    # --- SVI quartile from overall percentile -------------------------------
    svi = df[C.SVI_OVERALL]
    df["svi_quartile"] = pd.qcut(svi, q=4, labels=[1, 2, 3, 4], duplicates="drop")
    df["svi_quartile"] = df["svi_quartile"].astype("Int64")

    # --- "Maternity care desert" binary (>90 min to Level III OR >60 min to any OB) ---
    df["maternity_desert"] = (
        (df["time3_imputed"] > 90) | (df["time1_imputed"] > 60)
    ).astype("int8")

    return df


if __name__ == "__main__":
    from .data_loader import load_clean
    from .outcomes import build_outcomes
    d = build_features(build_outcomes(load_clean()))
    print(d[["fips", "county", "time3_imputed", "loc_discordance_tier",
             "effective_access_tier_30min", "svi_quartile",
             "maternity_desert"]].head())
    print("discordance tier dist:")
    print(d["loc_discordance_tier"].value_counts().sort_index())
    print("maternity desert share:", d["maternity_desert"].mean().round(3))
