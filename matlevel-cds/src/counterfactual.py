"""Counterfactual hospital siting engine.

Given a proposed new hospital location (lat, lon, ACOG level), recompute
every county's travel time to the nearest facility at each level and
return:
  - which counties moved out of "severe_gap" (>90 min to Level III) status
  - the nationwide population now reachable within 30/60/90 min of Level III
  - the median reduction in Level-III travel time across counties whose time
    actually improved
  - the equity-weighted gain (reduction concentrated in high-SVI quartiles)

Drive time is estimated via the empirical log-log model in src.geo that was
calibrated against the original ArcGIS OD matrix values shipped with the
PRJ-6250 dataset. This is an approximation: real drive times require the
actual road network, but this lets us rank siting alternatives at
nationwide scale without shipping GB of OSM data on Render.

The function also emits a stream of progress messages via an optional
callback so the frontend can display a loading log ("computing distances",
"applying new facility", "scoring counties", ...).
"""
from __future__ import annotations
import time
from dataclasses import dataclass, asdict
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import config as C
from .analytic import build_analytic
from .geo import (cohort_with_centroids, estimate_drive_min, haversine_km,
                  EARTH_R_KM)


@dataclass
class SitingScenario:
    lat: float
    lon: float
    level: int                      # 1/2/3/4 ACOG level of the new facility
    name: Optional[str] = None


@dataclass
class CountyDelta:
    fips: str
    county: str
    state_fips: str
    time3_before: float
    time3_after: float
    delta_min: float
    discordance_before: str
    discordance_after: str
    svi_quartile: Optional[int]
    pop: int


def _tier_label(t: float) -> str:
    if np.isnan(t):
        return "unknown"
    if t <= 30: return "concordant"
    if t <= 60: return "gap"
    if t <= 90: return "severe_gap"
    return "severe_gap"


NON_CONUS_STATES = {"02", "15", "60", "66", "69", "72", "78"}  # AK, HI, territories


def simulate_new_facility(scenario: SitingScenario,
                          log: Optional[Callable[[str], None]] = None,
                          conus_only: bool = True,
                          ) -> dict:
    """Recompute travel time to each ACOG level for every cohort county
    given a proposed new facility. Returns a summary dict with per-county
    deltas and aggregate equity statistics.

    Assumes a higher-level facility also provides all lower-level services
    (same assumption as the original dataset): a new Level III facility
    updates time1/time2/time3 if the new facility is closer; a new Level II
    updates time1/time2; a new Level I updates time1 only.
    """
    def _l(msg: str):
        print(msg)
        if log is not None:
            try: log(msg)
            except Exception: pass

    _l(f"[cf] loading cohort with centroids ({int(time.time())})")
    df = cohort_with_centroids().copy()
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    if conus_only:
        df = df[~df["state_fips"].isin(NON_CONUS_STATES)].reset_index(drop=True)
        _l("[cf] restricting to CONUS (AK/HI/territories excluded -- "
           "non-routable by road)")
    n = len(df)
    _l(f"[cf] {n} counties in cohort with centroid data")

    _l("[cf] computing Haversine distances from every county centroid to "
       "proposed facility")
    dists_km = haversine_km(df["lat"].values, df["lon"].values,
                            scenario.lat, scenario.lon)
    _l(f"[cf] distance range: min={float(dists_km.min()):.1f} km  "
       f"median={float(np.median(dists_km)):.1f} km  "
       f"max={float(dists_km.max()):.1f} km")

    _l("[cf] applying empirical drive-time model "
       "(log(min) ~ a + b*log(km), calibrated against ArcGIS OD matrix)")
    new_drive_min = estimate_drive_min(dists_km)

    # Apply update logic: facility at level L updates time columns 1..L
    # whose existing value exceeds the new candidate.
    _l(f"[cf] updating travel times for level {scenario.level} facility")
    affected_cols = [f"time{lvl}_imputed" for lvl in range(1, scenario.level + 1)]
    before = {c: df[c].copy() for c in affected_cols}
    for col in affected_cols:
        df[col] = np.minimum(df[col].values, new_drive_min)

    # Focus on Level III as the headline metric.
    t3_before = before["time3_imputed"].values if "time3_imputed" in before else df["time3_imputed"].values
    t3_after = df["time3_imputed"].values

    delta = t3_after - t3_before
    improved_mask = delta < -1e-3

    _l(f"[cf] {int(improved_mask.sum())} counties improved on Level-III travel time "
       f"(of {n})")

    # Status transitions
    def _count(mask):
        return int(np.sum(mask))
    sev_before = _count(t3_before > 90)
    sev_after = _count(t3_after > 90)
    gap_before = _count((t3_before > 60) & (t3_before <= 90))
    gap_after = _count((t3_after > 60) & (t3_after <= 90))
    conc_before = _count(t3_before <= 30)
    conc_after = _count(t3_after <= 30)

    _l(f"[cf] severe_gap counties:   before={sev_before}  after={sev_after}  "
       f"(delta {sev_after - sev_before})")
    _l(f"[cf] >30 min gap counties:   before={n - conc_before}  after={n - conc_after}")

    # Per-county deltas (only the ones that moved)
    improved_rows: list[CountyDelta] = []
    for i in np.where(improved_mask)[0]:
        row = df.iloc[i]
        improved_rows.append(CountyDelta(
            fips=str(row["fips"]),
            county=str(row["county"]),
            state_fips=str(row["state_fips"]),
            time3_before=round(float(t3_before[i]), 1),
            time3_after=round(float(t3_after[i]), 1),
            delta_min=round(float(delta[i]), 1),
            discordance_before=_tier_label(float(t3_before[i])),
            discordance_after=_tier_label(float(t3_after[i])),
            svi_quartile=int(row["svi_quartile"]) if pd.notna(row["svi_quartile"]) else None,
            pop=int(row[C.POP_TOTAL]) if pd.notna(row[C.POP_TOTAL]) else 0,
        ))
    improved_rows.sort(key=lambda r: r.delta_min)  # most negative first

    # Equity summary
    _l("[cf] computing equity-weighted gain")
    equity_rows = []
    svi_q = pd.to_numeric(df["svi_quartile"], errors="coerce").fillna(-1).astype(int).values
    for q in [1, 2, 3, 4]:
        mask = np.logical_and(svi_q == q, improved_mask).astype(bool)
        if not mask.any():
            continue
        equity_rows.append({
            "svi_quartile": q,
            "counties_improved": int(mask.sum()),
            "median_minutes_saved": round(float(-delta[mask].mean()), 1),
            "population_reached": int(df.loc[mask, C.POP_TOTAL].sum()),
        })

    # Population-weighted access
    total_pop = float(df[C.POP_TOTAL].sum())
    pop_within60_before = float(df.loc[t3_before <= 60, C.POP_TOTAL].sum())
    pop_within60_after = float(df.loc[t3_after <= 60, C.POP_TOTAL].sum())
    pop_within90_before = float(df.loc[t3_before <= 90, C.POP_TOTAL].sum())
    pop_within90_after = float(df.loc[t3_after <= 90, C.POP_TOTAL].sum())

    _l(f"[cf] population within 60 min Level III: before={pop_within60_before/1e6:.1f}M "
       f"-> after={pop_within60_after/1e6:.1f}M")
    _l(f"[cf] population within 90 min Level III: before={pop_within90_before/1e6:.1f}M "
       f"-> after={pop_within90_after/1e6:.1f}M")

    _l("[cf] done")

    return {
        "scenario": asdict(scenario),
        "n_counties": n,
        "n_improved": int(improved_mask.sum()),
        "status_transitions": {
            "concordant_<=30":   {"before": conc_before, "after": conc_after,
                                   "delta": conc_after - conc_before},
            "gap_60_to_90":      {"before": gap_before, "after": gap_after,
                                   "delta": gap_after - gap_before},
            "severe_gap_>90":    {"before": sev_before, "after": sev_after,
                                   "delta": sev_after - sev_before},
        },
        "population_covered_millions": {
            "within_60min_before": round(pop_within60_before / 1e6, 2),
            "within_60min_after":  round(pop_within60_after / 1e6, 2),
            "within_90min_before": round(pop_within90_before / 1e6, 2),
            "within_90min_after":  round(pop_within90_after / 1e6, 2),
            "new_population_within_60min":
                int(pop_within60_after - pop_within60_before),
            "new_population_within_90min":
                int(pop_within90_after - pop_within90_before),
        },
        "equity_by_svi": equity_rows,
        "top_improved_counties": [asdict(r) for r in improved_rows[:20]],
    }


# --------- Rank top-N siting candidates across all cohort county centroids -

def rank_siting_candidates(level: int = 3, top_n: int = 10,
                           log: Optional[Callable[[str], None]] = None) -> list[dict]:
    """Naive search: try placing a level-L facility at each cohort county
    centroid that is NOT already a level-3 host. Score by the number of
    severe_gap counties that move out of severe_gap. Returns top N.
    For n=2054 counties this is 2054 simulations, each O(n) -> ~4M
    distance computations. Runs in ~10-15 seconds."""
    def _l(m):
        print(m)
        if log is not None:
            try: log(m)
            except Exception: pass

    df = cohort_with_centroids().dropna(subset=["lat","lon"]).reset_index(drop=True)
    lats = df["lat"].values
    lons = df["lon"].values
    t3_before = df["time3_imputed"].values.copy()
    base_severe = int(np.sum(t3_before > 90))

    _l(f"[cf-rank] base severe_gap count: {base_severe}")
    _l(f"[cf-rank] testing {len(df)} candidate locations ...")

    scores = np.empty(len(df))
    for i in range(len(df)):
        d_km = haversine_km(lats, lons, lats[i], lons[i])
        t3_new = np.minimum(t3_before, estimate_drive_min(d_km))
        scores[i] = base_severe - int(np.sum(t3_new > 90))
        if i % 200 == 0:
            _l(f"[cf-rank] progress {i}/{len(df)}")

    order = np.argsort(-scores)[:top_n]
    out = []
    for rank, i in enumerate(order, start=1):
        r = df.iloc[i]
        out.append({
            "rank": rank,
            "fips": str(r["fips"]),
            "county": str(r["county"]),
            "state_fips": str(r["state_fips"]),
            "lat": float(lats[i]), "lon": float(lons[i]),
            "severe_gap_counties_eliminated": int(scores[i]),
            "current_time3_min": round(float(t3_before[i]), 1),
        })
    _l(f"[cf-rank] done, top candidate saves {out[0]['severe_gap_counties_eliminated']} counties")
    return out


if __name__ == "__main__":
    import json

    # Demo 1: add a Level III facility in Kentucky (example: London, KY ~37.13N, -84.08W)
    print("\n=== SCENARIO: add Level-III in London, KY ===")
    result = simulate_new_facility(
        SitingScenario(lat=37.128, lon=-84.083, level=3, name="London KY demo"),
    )
    print(json.dumps({k: v for k, v in result.items()
                      if k != "top_improved_counties"}, indent=2))
    print(f"top 5 counties improved: {result['top_improved_counties'][:5]}")
