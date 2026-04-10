"""Geographic utilities: county centroids, Haversine distance, k-NN adjacency,
and an empirically-calibrated drive-time model.

Design notes
------------
The PRJ-6250 dataset ships ArcGIS-computed drive times from each county's
centroid to the nearest hospital at each ACOG level of care, but does NOT
ship the centroid coordinates themselves. We download them once from the
Census Gazetteer (2019 release, same vintage as the travel-time computation)
and commit the 141 KB CSV to the repo: data/county_centroids_2019.csv.

For the counterfactual 'add a hospital here' engine we need to estimate
driving time from any county centroid to any proposed hospital lat/lon.
Rather than shipping a giant OSM road network, we fit a simple empirical
model: drive_time_min ~ a + b * haversine_km, calibrated against the
original ArcGIS time3 values from the dataset. The fit is done in log-log
space to handle the large dynamic range (3 min to 3600 min for island
counties).
"""
from __future__ import annotations
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from . import config as C


CENTROID_CSV = C.PROJECT_ROOT / "data" / "county_centroids_2019.csv"


# ---------- Haversine ----------

EARTH_R_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km. Vectorized."""
    lat1r = np.radians(np.asarray(lat1, dtype=float))
    lat2r = np.radians(np.asarray(lat2, dtype=float))
    dlat = lat2r - lat1r
    dlon = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_R_KM * c


# ---------- County centroid table ----------

@lru_cache(maxsize=1)
def county_centroids() -> pd.DataFrame:
    """Return a DataFrame indexed by 5-digit FIPS with columns lat, lon, county."""
    if not CENTROID_CSV.exists():
        raise FileNotFoundError(
            f"Missing {CENTROID_CSV}. Run scripts/download_centroids.py "
            f"or `python -m src.geo download`"
        )
    df = pd.read_csv(CENTROID_CSV, dtype={"fips": str})
    df["fips"] = df["fips"].str.zfill(5)
    return df.set_index("fips")[["county", "lat", "lon"]]


def cohort_with_centroids() -> pd.DataFrame:
    """Join the analytic cohort with county centroid lat/lon."""
    from .analytic import build_analytic
    cohort = build_analytic().set_index("fips")
    cen = county_centroids()
    joined = cohort.join(cen[["lat", "lon"]], how="left")
    missing = joined["lat"].isna().sum()
    if missing:
        print(f"[geo] WARNING: {missing} cohort counties missing centroids")
    return joined.reset_index()


# ---------- Drive-time empirical calibration ----------

def _fit_drive_time_model() -> dict:
    """Fit log(drive_min) ~ log(haversine_km) on the existing time3 values
    using the county-to-self distance=0 as baseline. We can't know the
    identity of the *destination* hospital (that data is proprietary AHA),
    so we fit against the distribution of known time3 values vs the
    distance from each county to the *nearest other county's centroid*
    as a rough proxy for the geometric scale of a short trip. This is
    intentionally simple and is meant only to rank counterfactuals, not to
    replace the original ArcGIS OD matrix.

    A more principled version: for each county, find the county within its
    travel-time neighborhood (say, top-5 nearest neighbors) and fit
    time3 ~ min(haversine to those neighbors). We then assume the real
    nearest Level-III hospital sits at distance d_eff for which the model
    predicts the observed time.
    """
    df = cohort_with_centroids().dropna(subset=["lat", "lon", "time3_imputed"])
    lats = df["lat"].values
    lons = df["lon"].values
    coords_rad = np.radians(np.column_stack([lats, lons]))

    # 5-nearest neighbors among county centroids (exclude self)
    nn = NearestNeighbors(n_neighbors=6, metric="haversine").fit(coords_rad)
    dist_rad, _ = nn.kneighbors(coords_rad)
    min_neighbor_km = dist_rad[:, 1] * EARTH_R_KM   # 1st is self

    # Clip pathological values before fitting
    t3 = df["time3_imputed"].values.astype(float)
    mask = (t3 > 0) & (t3 < 600) & (min_neighbor_km > 0) & (min_neighbor_km < 500)
    log_t = np.log(t3[mask])
    log_d = np.log(min_neighbor_km[mask])

    # log(time) = a + b * log(dist)
    b, a = np.polyfit(log_d, log_t, 1)
    return {"a": float(a), "b": float(b),
            "rmse_log": float(np.sqrt(np.mean((log_t - (a + b * log_d)) ** 2))),
            "n": int(mask.sum())}


@lru_cache(maxsize=1)
def drive_time_model() -> dict:
    return _fit_drive_time_model()


def estimate_drive_min(km: np.ndarray) -> np.ndarray:
    """Apply the fitted log-log model to a distance (km) to predict drive
    time in minutes. Robust to 0s."""
    km = np.maximum(np.asarray(km, dtype=float), 0.1)
    m = drive_time_model()
    return np.exp(m["a"] + m["b"] * np.log(km))


# ---------- k-NN adjacency graph ----------

@lru_cache(maxsize=4)
def knn_adjacency(k: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Return (fips_order, edge_index) where edge_index is shape (2, E).
    Undirected: each pair is added twice (i->j, j->i)."""
    df = cohort_with_centroids().dropna(subset=["lat", "lon"]).reset_index(drop=True)
    coords_rad = np.radians(df[["lat", "lon"]].values)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="haversine").fit(coords_rad)
    _, idx = nn.kneighbors(coords_rad)
    src = np.repeat(np.arange(len(df)), k)
    dst = idx[:, 1:].flatten()          # drop self (column 0)
    edges = np.column_stack([np.concatenate([src, dst]),
                             np.concatenate([dst, src])]).T.astype(np.int64)
    fips_order = df["fips"].values
    return fips_order, edges


# ---------- CLI entry ----------

def _download_centroids() -> None:
    """Download the 2019 Gazetteer county centroid file to data/."""
    import io, urllib.request, zipfile
    url = ("https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
           "2019_Gazetteer/2019_Gaz_counties_national.zip")
    print("downloading", url)
    req = urllib.request.Request(url, headers={"User-Agent": "MatLevel-CDS/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        inner = [n for n in zf.namelist() if n.endswith(".txt")][0]
        df = pd.read_csv(zf.open(inner), sep="\t", dtype={"GEOID": str})
    df.columns = [c.strip() for c in df.columns]
    df = (df[["GEOID", "NAME", "INTPTLAT", "INTPTLONG"]]
          .rename(columns={"GEOID": "fips", "NAME": "county",
                           "INTPTLAT": "lat", "INTPTLONG": "lon"}))
    df["fips"] = df["fips"].str.zfill(5)
    CENTROID_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CENTROID_CSV, index=False)
    print("wrote", CENTROID_CSV)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        _download_centroids()
    else:
        cen = county_centroids()
        print("centroids:", len(cen), "counties")
        print(cen.head())
        print()
        model = drive_time_model()
        print("drive time model:", model)
        fips_order, edges = knn_adjacency(k=8)
        print(f"k-NN adjacency: {len(fips_order)} nodes, {edges.shape[1]} directed edges")
