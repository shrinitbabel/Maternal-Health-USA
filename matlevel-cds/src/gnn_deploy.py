"""Train GraphSAGE on the full cohort and cache per-county predictions for
deployment.

Why precompute? The deployed CDS scorer is point-evaluable per FIPS and called
on every /recommend, /access, and /shap request. Running graph-wide message
passing per request would require torch on Render and recomputing the whole
graph each call. Since cohort county features don't change at inference time,
we precompute predictions once and load them as a dict at startup.

For each target in DEPLOY_TARGETS:
  1. Train CountyGraphSAGE on ALL valid cohort counties (no holdout, no
     leave-state-out). The head-to-head in src.gnn_eval already validated
     spatial CV performance; production training maximizes data use.
  2. Run inference on the full graph.
  3. Save predictions to outputs/models/gnn_predictions.json keyed by FIPS.

**GraphSAGE is the primary scorer for all six CDS outcomes.** Even where it
loses to XGBoost on headline R^2 (e.g. SMM composite, where the outcome is
discharge-capture-confounded), it consistently wins on calibration, ECE, and
stability — the metrics that matter for clinical decision support. XGBoost
predictions stay cached as a fallback if the GNN cache fails to load.
"""
from __future__ import annotations
import json
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from . import config as C
from .geo import cohort_with_centroids, knn_adjacency
from .gnn import CountyGraphSAGE, _HAS_PYG
from .model import feature_matrix


DEPLOY_TARGETS = [
    "puerperal_rate_per1000",
    "hdp_rate_per1000",
    "smm_rate_per1000",
    "childbirth_cx_rate_per1000",
    "placenta_rate_per1000",
    "dm_rate_per1000",
]
PRED_JSON = C.MODEL_DIR / "gnn_predictions.json"


def _log(msg: str, sink: Optional[Callable[[str], None]] = None) -> None:
    print(msg)
    if sink is not None:
        try:
            sink(msg)
        except Exception:
            pass


def train_model(target: str,
                epochs: int = 400,
                lr: float = 1e-3,
                log: Optional[Callable[[str], None]] = None):
    """Fit GraphSAGE on ALL valid cohort nodes (no holdout) and return a
    bundle of (model, x_t, edge_t, fips_order, feature_names, y_mean, y_std)
    that downstream tools (deploy cache, SHAP-for-GNN) can both reuse."""
    if not _HAS_PYG:
        raise RuntimeError("torch_geometric not available")

    df = cohort_with_centroids().reset_index(drop=True)
    fips_order, edges = knn_adjacency(k=8)
    assert np.array_equal(df["fips"].values, fips_order), "fips order mismatch"

    feat_df = feature_matrix(df).astype("float64").fillna(0.0)
    feature_names = list(feat_df.columns)
    X_raw = feat_df.values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw)

    y = df[target].astype(float).values
    mask = ~np.isnan(y)

    y_mean = float(np.nanmean(y))
    y_std = float(np.nanstd(y)) or 1.0
    y_std_arr = (y - y_mean) / y_std

    x_t = torch.tensor(X_std, dtype=torch.float32)
    y_t = torch.tensor(np.nan_to_num(y_std_arr, nan=0.0), dtype=torch.float32)
    edge_t = torch.tensor(edges, dtype=torch.long)
    train_nodes = torch.tensor(np.where(mask)[0], dtype=torch.long)

    torch.manual_seed(C.RANDOM_STATE)
    model = CountyGraphSAGE(in_dim=x_t.shape[1], hidden=64, out_dim=32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    _log(f"[deploy] target={target}  n_train={len(train_nodes)}  "
         f"n_edges={edges.shape[1]}  epochs={epochs}", log)

    best_loss = float("inf")
    best_state = None
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(x_t, edge_t)
        loss = F.mse_loss(pred[train_nodes], y_t[train_nodes])
        loss.backward()
        opt.step()
        if loss.item() < best_loss - 1e-5:
            best_loss = loss.item()
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % 50 == 0 or ep == epochs - 1:
            _log(f"  epoch {ep:3d}  train_mse={loss.item():.4f}", log)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return {
        "model": model,
        "x": x_t,
        "edge_index": edge_t,
        "fips_order": fips_order,
        "feature_names": feature_names,
        "y_mean": y_mean,
        "y_std": y_std,
        "valid_mask": mask,
        "X_raw": X_raw,
    }


def train_full_cohort(target: str,
                      epochs: int = 400,
                      lr: float = 1e-3,
                      log: Optional[Callable[[str], None]] = None) -> dict:
    """Train + return {fips: predicted_rate} for every cohort county."""
    bundle = train_model(target, epochs=epochs, lr=lr, log=log)
    with torch.no_grad():
        pred_std = bundle["model"](bundle["x"], bundle["edge_index"]).cpu().numpy()
    pred = pred_std * bundle["y_std"] + bundle["y_mean"]
    pred = np.maximum(pred, 0.0)
    return {fips: float(p) for fips, p in zip(bundle["fips_order"], pred)}


def run(log: Optional[Callable[[str], None]] = None) -> dict:
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    by_fips: dict[str, dict[str, float]] = {}
    for tgt in DEPLOY_TARGETS:
        preds = train_full_cohort(tgt, log=log)
        for fips, v in preds.items():
            by_fips.setdefault(fips, {})[tgt] = round(v, 4)

    out = {
        "_meta": {
            "deploy_targets": DEPLOY_TARGETS,
            "n_counties": len(by_fips),
            "architecture": "SAGEConv(F->64)->ReLU->SAGEConv(64->32)->Linear(32->1)",
            "k_neighbors": 8,
            "trained_on": "full cohort, no holdout",
            "rationale": "GraphSAGE is the primary CDS scorer. It wins or "
                         "ties XGBoost on calibration, ECE, and stability "
                         "across all six targets, and wins the headline "
                         "regression metrics on the residential-access-driven "
                         "outcomes (puerperal, HDP). XGBoost remains cached "
                         "as a fallback only.",
        },
        "predictions": by_fips,
    }
    PRED_JSON.write_text(json.dumps(out, indent=2, default=float))
    _log(f"[ok] wrote {PRED_JSON.name} for {len(by_fips)} counties "
         f"x {len(DEPLOY_TARGETS)} targets", log)
    return out


if __name__ == "__main__":
    run()
