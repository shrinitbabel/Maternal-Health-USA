"""MatLevel-CDS FastAPI backend.

Deployed to Render. The Vercel frontend on the user's personal site consumes
these endpoints via CORS.

Routes
------
GET  /                      ->  service banner
GET  /health                ->  liveness + readiness (for Render health check)
GET  /counties              ->  list of (fips, county, state_fips) tuples
GET  /counties/{state}      ->  counties in a given 2-digit state FIPS
GET  /access/{fips}         ->  county access + SVI context, no patient data
POST /recommend             ->  full CDS recommendation (patient risk + county)
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

# ---- make the sibling ``src`` package importable regardless of CWD ---------
_THIS = Path(__file__).resolve()
_PROJECT = _THIS.parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    CountyAccessOut,
    HealthResponse,
    RecommendRequest,
    RecommendResponse,
    RiskFactorsIn,
)
from src.analytic import build_analytic
from src.cds import RiskFactors, _county_lookup, recommend

APP_VERSION = "0.1.0"

app = FastAPI(
    title="MatLevel-CDS API",
    description=(
        "Clinical decision support operationalizing ACOG/SMFM Levels of "
        "Maternal Care through county-level geographic access modeling. "
        "Backed by the Maternal Health Equity Linked Datasets (PRJ-6250)."
    ),
    version=APP_VERSION,
)

# CORS: allow Vercel frontends + local dev.
# When the frontend domain is known, pin it here (Render env var recommended).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --- startup: warm the county lookup + models -------------------------------
_STATE: dict = {"ready": False, "n_counties": 0, "model_loaded": False}


@app.on_event("startup")
def _warm_caches() -> None:
    try:
        df = _county_lookup()
        _STATE["n_counties"] = int(len(df))
        # Best-effort warm for the scoring model; recommend() still works
        # without it (the CDS score bundle becomes empty).
        try:
            from src.cds import _scoring_model
            _scoring_model()
            _STATE["model_loaded"] = True
        except Exception as e:
            print(f"[warn] scoring model not loaded: {e}")
            _STATE["model_loaded"] = False
        _STATE["ready"] = True
    except Exception as e:
        print(f"[error] startup failed: {e}")
        _STATE["ready"] = False


# ------------------------------ endpoints -----------------------------------

@app.get("/")
def root() -> dict:
    return {
        "service": "MatLevel-CDS",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _STATE["ready"] else "starting",
        version=APP_VERSION,
        counties_loaded=_STATE["n_counties"],
        model_loaded=_STATE["model_loaded"],
    )


@app.get("/counties")
def counties(state: Optional[str] = None) -> list[dict]:
    df = _county_lookup().reset_index()
    if state:
        df = df[df["state_fips"] == str(state).zfill(2)]
    return df[["fips", "county", "state_fips"]].to_dict(orient="records")


@app.get("/counties/{state}")
def counties_in_state(state: str) -> list[dict]:
    return counties(state=state)


@app.get("/access/{fips}", response_model=CountyAccessOut)
def access(fips: str) -> CountyAccessOut:
    fips = fips.zfill(5)
    df = _county_lookup()
    if fips not in df.index:
        raise HTTPException(404, f"FIPS {fips} not found")
    row = df.loc[fips]

    # Fetch model scores via recommend() with an empty risk profile.
    r = recommend(fips, RiskFactors())
    return CountyAccessOut(
        fips=fips,
        county=str(row["county"]),
        state_fips=str(row["state_fips"]),
        time_to_level_1_min=_none_if_nan(row["time1_imputed"]),
        time_to_level_2_min=_none_if_nan(row["time2_imputed"]),
        time_to_level_3_min=_none_if_nan(row["time3_imputed"]),
        svi_percentile=_none_if_nan(row["rpl_themes"]),
        svi_quartile=int(row["svi_quartile"]) if _not_nan(row["svi_quartile"]) else None,
        maternity_desert=bool(row["maternity_desert"]),
        is_level3_host=bool(row["is_level3_host"]),
        gap30_level3=bool(row["time3_imputed"] > 30) if _not_nan(row["time3_imputed"]) else False,
        gap60_level3=bool(row["time3_imputed"] > 60) if _not_nan(row["time3_imputed"]) else False,
        gap90_level3=bool(row["time3_imputed"] > 90) if _not_nan(row["time3_imputed"]) else False,
        county_model_scores=r.county_model_scores,
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(req: RecommendRequest) -> RecommendResponse:
    rf = RiskFactors(**req.risk.model_dump())
    try:
        rec = recommend(req.fips, rf)
    except KeyError:
        raise HTTPException(404, f"FIPS {req.fips} not found in cohort")
    return RecommendResponse(**rec.__dict__)


# ----------------------------- small helpers --------------------------------

def _not_nan(x) -> bool:
    try:
        return x == x and x is not None  # NaN != NaN
    except Exception:
        return False


def _none_if_nan(x) -> Optional[float]:
    return float(x) if _not_nan(x) else None
