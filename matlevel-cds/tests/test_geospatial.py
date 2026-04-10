"""Smoke tests for the new geospatial + evaluation modules."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.counterfactual import NON_CONUS_STATES, SitingScenario, simulate_new_facility
from src.geo import (county_centroids, cohort_with_centroids, drive_time_model,
                     estimate_drive_min, haversine_km, knn_adjacency)


def test_centroids_loaded():
    df = county_centroids()
    assert len(df) >= 3200
    assert {"county", "lat", "lon"}.issubset(df.columns)
    assert df["lat"].between(-90, 90).all()
    assert df["lon"].between(-180, 180).all()


def test_cohort_joined():
    df = cohort_with_centroids()
    assert "lat" in df.columns
    # Most cohort counties should have centroids
    assert df["lat"].notna().mean() > 0.98


def test_haversine_known_distance():
    # NYC to LA ~ 3944 km
    km = haversine_km(40.7128, -74.0060, 34.0522, -118.2437)
    assert 3800 < float(km) < 4100


def test_drive_time_model_positive():
    m = drive_time_model()
    assert m["n"] > 1000
    for km in [1, 10, 100]:
        mins = float(estimate_drive_min(km))
        assert mins > 0


def test_knn_adjacency_shape():
    fips_order, edges = knn_adjacency(k=8)
    assert edges.shape[0] == 2
    # 8-NN symmetric directed -> ~16 edges per node
    assert edges.shape[1] == len(fips_order) * 8 * 2


def test_counterfactual_returns_expected_shape():
    scen = SitingScenario(lat=37.128, lon=-84.083, level=3, name="test")
    res = simulate_new_facility(scen)
    assert "n_counties" in res
    assert res["n_counties"] > 2000
    assert "status_transitions" in res
    assert "population_covered_millions" in res
    assert isinstance(res["top_improved_counties"], list)


def test_metrics_json_well_formed():
    p = C.TAB_DIR / "full_metrics.json"
    if not p.exists():
        return  # run evaluate first
    d = json.loads(p.read_text())
    assert "regression" in d and "binary_top_quartile_puerperal" in d
    for tgt, r in d["regression"].items():
        assert r["xgb"]["R2_boot_CI95"][0] <= r["xgb"]["R2_boot_CI95"][1]
    b = d["binary_top_quartile_puerperal"]
    assert 0 <= b["AUROC"] <= 1
    assert 0 <= b["AUPRC"] <= 1
