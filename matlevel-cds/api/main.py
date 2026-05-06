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
from src.shap_explain import explain_patient


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

_STATE: dict = {"ready": False, "n_counties": 0, "model_loaded": False,
                "gnn_predictions_loaded": 0}


@app.on_event("startup")
def _warm_caches() -> None:
    try:
        df = _county_lookup()
        _STATE["n_counties"] = int(len(df))
        try:
            from src.cds import _gnn_predictions
            gp = _gnn_predictions()
            _STATE["gnn_predictions_loaded"] = len(gp.get("predictions", {}))
            _STATE["model_loaded"] = _STATE["gnn_predictions_loaded"] > 0
            targets = gp.get("_meta", {}).get("deploy_targets", [])
            print(f"[ok] GraphSAGE primary scorer warmed for "
                  f"{_STATE['gnn_predictions_loaded']} counties "
                  f"x {len(targets)} outcomes ({', '.join(targets) or 'none'})")
        except Exception as e:
            print(f"[warn] GraphSAGE cache not loaded: {e}")
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
        gnn_predictions_loaded=_STATE["gnn_predictions_loaded"],
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

# ---- graph endpoints (GraphSAGE-aware UI) --------------------------------

from functools import lru_cache as _lru


@_lru(maxsize=1)
def _graph_cache() -> dict:
    """Pre-bake the full 8-NN county graph: nodes (with lat/lon, SVI, GNN
    predictions, access flags) and edges. Computed once at startup so the
    network endpoint is O(1) after warm-up.

    Edges are de-duplicated to undirected (i<j) to halve payload size.
    The frontend can render with deck.gl (geographic) or three.js (3D)."""
    from src.geo import knn_adjacency
    from src.cds import _gnn_predictions

    df = _county_lookup().reset_index()
    cen = county_centroids().reset_index()
    cen = cen.rename(columns={"index": "fips"}) if "index" in cen.columns else cen
    joined = df.merge(cen[["fips", "lat", "lon"]], on="fips", how="left")

    fips_order, edges = knn_adjacency(k=8)
    fips_to_pos = {f: i for i, f in enumerate(fips_order)}
    by_fips = joined.set_index("fips")

    gnn_preds = _gnn_predictions().get("predictions", {})

    nodes = []
    for fips in fips_order:
        if fips not in by_fips.index:
            continue
        r = by_fips.loc[fips]
        nodes.append({
            "fips": str(fips),
            "county": str(r["county"]),
            "state_fips": str(r["state_fips"]),
            "lat": _none_if_nan(r.get("lat")),
            "lon": _none_if_nan(r.get("lon")),
            "time3_min": _none_if_nan(r.get("time3_imputed")),
            "svi_percentile": _none_if_nan(r.get("rpl_themes")),
            "svi_quartile": int(r["svi_quartile"]) if _not_nan(r.get("svi_quartile")) else None,
            "maternity_desert": bool(r.get("maternity_desert", False)),
            "is_level3_host": bool(r.get("is_level3_host", False)),
            "gnn_scores": gnn_preds.get(str(fips), {}),
        })

    # De-duplicate to undirected edges (i<j only).
    src, dst = edges[0], edges[1]
    seen = set()
    uniq_edges = []
    for s, d in zip(src.tolist(), dst.tolist()):
        a, b = (s, d) if s < d else (d, s)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        uniq_edges.append({
            "source_fips": str(fips_order[a]),
            "target_fips": str(fips_order[b]),
        })

    return {
        "_meta": {
            "n_nodes": len(nodes),
            "n_edges": len(uniq_edges),
            "k_neighbors": 8,
            "graph_construction": "8-NN by haversine on Census 2019 "
                                   "Gazetteer county centroids",
        },
        "nodes": nodes,
        "edges": uniq_edges,
        "_fips_order": fips_order.tolist(),
        "_edge_array": edges.tolist(),
    }


@app.get("/graph/network")
def graph_network() -> dict:
    """Full 8-NN county graph for the GraphSAGE-aware frontend.

    Response shape:
      {
        "_meta": {n_nodes, n_edges, k_neighbors, graph_construction},
        "nodes": [{fips, county, state_fips, lat, lon, time3_min,
                   svi_percentile, svi_quartile, maternity_desert,
                   is_level3_host, gnn_scores}, ...],
        "edges": [{source_fips, target_fips}, ...]   # undirected, deduped
      }

    Cached at startup. Suitable for deck.gl ScatterplotLayer + LineLayer
    (geographic) or a force-directed three.js scene (3D).
    """
    g = _graph_cache()
    return {"_meta": g["_meta"], "nodes": g["nodes"], "edges": g["edges"]}


@app.get("/graph/neighbors/{fips}")
def graph_neighbors(fips: str) -> dict:
    """Return the queried county and its 8 nearest neighbors with their
    GraphSAGE-predicted rates and access metadata. The frontend uses this
    for a 'compare to your neighbors' panel on the recommend page."""
    fips = str(fips).zfill(5)
    g = _graph_cache()
    fips_order = g["_fips_order"]
    if fips not in fips_order:
        raise HTTPException(404, f"FIPS {fips} not in cohort")
    pos = fips_order.index(fips)

    src, dst = g["_edge_array"][0], g["_edge_array"][1]
    neighbor_positions = [d for s, d in zip(src, dst) if s == pos]
    neighbor_fips = sorted({fips_order[p] for p in neighbor_positions})

    by_fips = {n["fips"]: n for n in g["nodes"]}
    self_node = by_fips.get(fips, {})
    neighbors = [by_fips[f] for f in neighbor_fips if f in by_fips]

    return {
        "self": self_node,
        "neighbors": neighbors,
        "n_neighbors": len(neighbors),
        "_meta": {
            "k_neighbors": 8,
            "graph_construction": g["_meta"]["graph_construction"],
        },
    }


# ------------------------------- SHAP explain --------------------------------
from pydantic import BaseModel

class ShapRequest(BaseModel):
    fips: str

@app.post("/shap")
def shap_endpoint(req: ShapRequest) -> dict:
    try:
        return explain_patient(req.fips)
    except KeyError:
        raise HTTPException(404, f"FIPS {req.fips} not found")

@app.post("/shap/stream")
async def shap_stream(req: ShapRequest) -> EventSourceResponse:
    fips = req.fips
    q: Queue = Queue()

    def run_it(log=None):
        return explain_patient(fips, log=log)

    _run_with_stream(run_it, q)
    return EventSourceResponse(_drain_queue(q))
