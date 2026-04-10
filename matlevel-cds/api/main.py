"""MatLevel-CDS FastAPI backend.

Deployed to Render. The Vercel frontend on the user's personal site consumes
these endpoints via CORS.

Routes
------
GET  /                          ->  service banner
GET  /health                    ->  liveness + readiness (Render health check)
GET  /counties                  ->  list of (fips, county, state_fips) tuples
GET  /counties/{state}          ->  counties in a given 2-digit state FIPS
GET  /access/{fips}             ->  county access + SVI context, no patient data
POST /recommend                 ->  full CDS recommendation
GET  /metrics                   ->  the full model-performance JSON
GET  /metrics/stream            ->  Server-Sent-Events stream of evaluation
POST /counterfactual            ->  single-shot new-facility simulation
POST /counterfactual/stream     ->  SSE-streamed new-facility simulation
GET  /counterfactual/rank       ->  top-N siting candidates (slow; streams)
GET  /gnn/compare               ->  cached GNN-vs-XGBoost comparison
GET  /map/centroids             ->  list of county centroids for the map UI
"""
from __future__ import annotations
import asyncio
import json
import sys
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import AsyncIterator, Optional

_THIS = Path(__file__).resolve()
_PROJECT = _THIS.parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from api.schemas import (
    CountyAccessOut,
    CounterfactualRequest,
    GNNComparisonResponse,
    HealthResponse,
    MetricsResponse,
    RecommendRequest,
    RecommendResponse,
    RiskFactorsIn,
)
from src import config as C
from src.analytic import build_analytic
from src.cds import RiskFactors, _county_lookup, recommend
from src.counterfactual import (NON_CONUS_STATES, SitingScenario,
                                rank_siting_candidates, simulate_new_facility)
from src.geo import county_centroids

APP_VERSION = "0.2.0"

app = FastAPI(
    title="MatLevel-CDS API",
    description=(
        "Clinical decision support operationalizing ACOG/SMFM Levels of "
        "Maternal Care through county-level geographic access modeling. "
        "Backed by the Maternal Health Equity Linked Datasets (PRJ-6250)."
    ),
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_STATE: dict = {"ready": False, "n_counties": 0, "model_loaded": False}


@app.on_event("startup")
def _warm_caches() -> None:
    try:
        df = _county_lookup()
        _STATE["n_counties"] = int(len(df))
        try:
            from src.cds import _scoring_model
            _scoring_model()
            _STATE["model_loaded"] = True
        except Exception as e:
            print(f"[warn] scoring model not loaded: {e}")
            _STATE["model_loaded"] = False
        try:
            county_centroids()
            from src.geo import drive_time_model
            drive_time_model()
            print("[ok] geo caches warmed")
        except Exception as e:
            print(f"[warn] geo not available: {e}")
        _STATE["ready"] = True
    except Exception as e:
        print(f"[error] startup failed: {e}")
        _STATE["ready"] = False


# ------------------------------ small helpers -------------------------------

def _not_nan(x) -> bool:
    try:
        return x == x and x is not None
    except Exception:
        return False


def _none_if_nan(x) -> Optional[float]:
    return float(x) if _not_nan(x) else None


def _sse_event(event: str, data: dict | str) -> dict:
    """Format a Server-Sent-Events payload for sse_starlette."""
    if isinstance(data, (dict, list)):
        data = json.dumps(data, default=float)
    return {"event": event, "data": data}


def _run_with_stream(target_callable, log_queue: Queue, *args, **kwargs):
    """Run a blocking function in a background thread while pushing progress
    messages into `log_queue`. Puts the final result into the queue with the
    sentinel key '__done__'."""
    def sink(msg: str):
        log_queue.put({"event": "log", "msg": msg})

    def worker():
        try:
            result = target_callable(*args, log=sink, **kwargs)
            log_queue.put({"event": "done", "result": result})
        except Exception as e:
            log_queue.put({"event": "error", "error": str(e)})
        finally:
            log_queue.put(None)  # sentinel

    t = Thread(target=worker, daemon=True)
    t.start()
    return t


async def _drain_queue(q: Queue) -> AsyncIterator[dict]:
    """Async drain of a thread-safe queue -- yields SSE events until sentinel.
    Real messages pass through; empty ticks are ignored so the frontend only
    sees meaningful progress lines."""
    loop = asyncio.get_event_loop()
    while True:
        item = await loop.run_in_executor(None, _blocking_get, q)
        if item is None:
            break
        if item is _HEARTBEAT:
            # keep the connection alive without spamming the client
            yield {"event": "ping", "data": ""}
            continue
        if item.get("event") == "log":
            yield _sse_event("log", {"msg": item["msg"]})
        elif item.get("event") == "done":
            yield _sse_event("done", item["result"])
            break
        elif item.get("event") == "error":
            yield _sse_event("error", {"error": item["error"]})
            break


_HEARTBEAT = object()


def _blocking_get(q: Queue):
    try:
        return q.get(timeout=0.5)
    except Empty:
        return _HEARTBEAT


# ------------------------------- endpoints ----------------------------------

@app.get("/")
def root() -> dict:
    return {
        "service": "MatLevel-CDS",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "routes": [
            "/health", "/counties", "/counties/{state}", "/access/{fips}",
            "/recommend", "/metrics", "/metrics/stream",
            "/counterfactual", "/counterfactual/stream",
            "/counterfactual/rank", "/gnn/compare", "/map/centroids",
        ],
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


# ---- metrics -------------------------------------------------------------

@app.get("/metrics")
def get_metrics() -> dict:
    p = C.TAB_DIR / "full_metrics.json"
    if not p.exists():
        raise HTTPException(404,
            "No cached metrics. Run `python -m src.evaluate` locally and "
            "commit outputs/tables/full_metrics.json.")
    return json.loads(p.read_text())


@app.get("/metrics/stream")
async def metrics_stream() -> EventSourceResponse:
    """Re-run the evaluate module live and stream progress. On Render this
    takes roughly 60-90 seconds. Useful for showing a 'loading' UI."""
    from src.evaluate import run as evaluate_run

    q: Queue = Queue()
    _run_with_stream(evaluate_run, q)
    return EventSourceResponse(_drain_queue(q))


# ---- counterfactual ------------------------------------------------------

@app.post("/counterfactual")
def counterfactual(req: CounterfactualRequest) -> dict:
    scenario = SitingScenario(lat=req.lat, lon=req.lon, level=req.level, name=req.name)
    return simulate_new_facility(scenario, conus_only=req.conus_only)


@app.post("/counterfactual/stream")
async def counterfactual_stream(req: CounterfactualRequest) -> EventSourceResponse:
    """Stream progress messages while the counterfactual runs."""
    scenario = SitingScenario(lat=req.lat, lon=req.lon, level=req.level, name=req.name)
    q: Queue = Queue()

    def run_it(log=None):
        return simulate_new_facility(scenario, log=log, conus_only=req.conus_only)

    _run_with_stream(run_it, q)
    return EventSourceResponse(_drain_queue(q))


@app.get("/counterfactual/rank")
async def counterfactual_rank(level: int = 3, top_n: int = 10) -> EventSourceResponse:
    """Rank top-N siting candidates. Slow (~15s on Render) -- streams."""
    q: Queue = Queue()

    def run_it(log=None):
        return rank_siting_candidates(level=level, top_n=top_n, log=log)

    _run_with_stream(run_it, q)
    return EventSourceResponse(_drain_queue(q))


# ---- GNN comparison ------------------------------------------------------

@app.get("/gnn/compare", response_model=GNNComparisonResponse)
def gnn_compare() -> GNNComparisonResponse:
    p = C.TAB_DIR / "gnn_vs_xgb.json"
    if not p.exists():
        raise HTTPException(404,
            "No cached GNN comparison. Run `python -m src.gnn` locally "
            "and commit outputs/tables/gnn_vs_xgb.json.")
    return GNNComparisonResponse(results=json.loads(p.read_text()))


# ---- map support ---------------------------------------------------------

@app.get("/map/centroids")
def map_centroids() -> list[dict]:
    """Return county centroids joined with the access + SVI context.
    This is what the Vercel map needs to build its choropleth."""
    df = _county_lookup().reset_index()
    cen = county_centroids()
    joined = df.merge(cen[["lat", "lon"]],
                      left_on="fips", right_index=True, how="left")
    out = []
    for _, r in joined.iterrows():
        out.append({
            "fips": str(r["fips"]),
            "county": str(r["county"]),
            "state_fips": str(r["state_fips"]),
            "lat": _none_if_nan(r.get("lat")),
            "lon": _none_if_nan(r.get("lon")),
            "time3_min": _none_if_nan(r.get("time3_imputed")),
            "svi_percentile": _none_if_nan(r.get("rpl_themes")),
            "svi_quartile": int(r["svi_quartile"]) if _not_nan(r.get("svi_quartile")) else None,
            "maternity_desert": bool(r.get("maternity_desert", False)),
            "is_level3_host": bool(r.get("is_level3_host", False)),
        })
    return out
