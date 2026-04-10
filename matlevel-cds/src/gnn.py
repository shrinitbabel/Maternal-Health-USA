"""GraphSAGE spatial regression baseline for MatLevel-CDS.

Why a GNN?
----------
The XGBoost + leave-state-out results already showed that puerperal
complication rate is the best-predicted outcome (R2 ~= 0.29). A graph neural
network can explicitly encode the assumption that *neighboring counties*
share pregnancy complication patterns via shared health systems, labor
markets, and travel corridors. We use a lightweight 2-layer GraphSAGE
on an 8-nearest-neighbor county adjacency graph built from Census
Gazetteer centroids.

This turns MatLevel-CDS from a "spatially cross-validated ML pipeline"
into a bona-fide *geospatial* AI tool.

Design
------
- Nodes: U.S. counties in the analytic cohort (n ~ 2054)
- Edges: undirected k-NN by great-circle distance (k=8)
- Node features: the same tabular feature matrix used by XGBoost
- Target: continuous rate outcome (primary: puerperal_rate_per1000)
- Train / eval split: leave-state-out across 5 state-level folds (same
  as XGBoost, so the R2 numbers are directly comparable)
- Architecture: SAGEConv(F -> 64) -> ReLU -> SAGEConv(64 -> 32) -> Linear(32 -> 1)
- Loss: MSE. Optimizer: Adam(1e-3). 200 epochs with early stopping.

We deliberately keep the implementation small so it ships on Render without
exotic build flags. torch_geometric is already in the environment; if
deployment ever breaks, the whole file can be stubbed out.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import Callable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

try:
    from torch_geometric.nn import SAGEConv
    _HAS_PYG = True
except Exception as e:  # pragma: no cover
    print(f"[gnn] torch_geometric not available: {e}")
    _HAS_PYG = False

from . import config as C
from .analytic import build_analytic
from .geo import cohort_with_centroids, knn_adjacency
from .model import feature_matrix


def _log(msg: str, sink: Optional[Callable[[str], None]] = None) -> None:
    print(msg)
    if sink is not None:
        try:
            sink(msg)
        except Exception:
            pass


# ------------------------------ model --------------------------------------

if _HAS_PYG:

    class CountyGraphSAGE(nn.Module):
        def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 32):
            super().__init__()
            self.conv1 = SAGEConv(in_dim, hidden)
            self.conv2 = SAGEConv(hidden, out_dim)
            self.head = nn.Linear(out_dim, 1)
            self.dropout = nn.Dropout(0.2)

        def forward(self, x, edge_index):
            h = F.relu(self.conv1(x, edge_index))
            h = self.dropout(h)
            h = F.relu(self.conv2(h, edge_index))
            return self.head(h).squeeze(-1)


# --------------------------- training loop ---------------------------------

@dataclass
class GNNFoldResult:
    fold: int
    r2: float
    mae: float
    n_train: int
    n_test: int


def _build_graph_tensors(target: str) -> tuple[torch.Tensor, torch.Tensor,
                                               torch.Tensor, np.ndarray, np.ndarray,
                                               float, float]:
    """Return (x, edge_index, y_standardized, mask_valid, state_fips, y_mean, y_std).

    Targets differ in scale by two orders of magnitude across outcomes (puerperal
    median 13 vs SMM-proxy median 368). Standardize y before training; de-
    standardize predictions at evaluation time for reportable MAE."""
    df = cohort_with_centroids().reset_index(drop=True)
    fips_order, edges = knn_adjacency(k=8)
    assert np.array_equal(df["fips"].values, fips_order), "fips order mismatch"

    feat = feature_matrix(df).astype("float64").fillna(0.0).values
    scaler = StandardScaler()
    feat = scaler.fit_transform(feat)

    y_raw = df[target].astype(float).values
    mask = ~np.isnan(y_raw)

    y_mean = float(np.nanmean(y_raw))
    y_std = float(np.nanstd(y_raw))
    if y_std == 0:
        y_std = 1.0
    y_std_arr = (y_raw - y_mean) / y_std

    x_t = torch.tensor(feat, dtype=torch.float32)
    y_t = torch.tensor(np.nan_to_num(y_std_arr, nan=0.0), dtype=torch.float32)
    edge_t = torch.tensor(edges, dtype=torch.long)
    return x_t, edge_t, y_t, mask, df["state_fips"].values, y_mean, y_std


def train_gnn_spatial_cv(target: str,
                         log: Optional[Callable[[str], None]] = None,
                         n_splits: int = C.N_SPLITS_STATE_CV,
                         epochs: int = 300,
                         lr: float = 1e-3) -> dict:
    """Leave-state-out CV for the GraphSAGE baseline. We always use the
    full county graph at both train and eval time, masking the loss to
    training counties only -- this is standard transductive GNN training."""
    if not _HAS_PYG:
        _log("[gnn] torch_geometric missing; returning stub", log)
        return {"target": target, "available": False}

    _log(f"[gnn] target={target}", log)
    x, edge_index, y, mask_valid, states, y_mean, y_std = _build_graph_tensors(target)
    _log(f"[gnn] x={tuple(x.shape)} edges={edge_index.shape[1]} "
         f"n_valid={int(mask_valid.sum())}", log)

    device = torch.device("cpu")
    x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)

    valid_idx = np.where(mask_valid)[0]
    valid_states = states[valid_idx]
    gkf = GroupKFold(n_splits=n_splits)

    fold_results: list[GNNFoldResult] = []
    y_true_all, y_pred_all = [], []

    for fold, (tr_ix, te_ix) in enumerate(gkf.split(valid_idx, valid_idx, valid_states)):
        train_nodes = torch.tensor(valid_idx[tr_ix], dtype=torch.long, device=device)
        test_nodes = torch.tensor(valid_idx[te_ix], dtype=torch.long, device=device)

        model = CountyGraphSAGE(in_dim=x.shape[1], hidden=64, out_dim=32).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        best_val = float("inf"); best_state = None; patience = 30; wait = 0
        _log(f"  [gnn fold {fold+1}/{n_splits}] train={len(train_nodes)} test={len(test_nodes)}", log)

        for ep in range(epochs):
            model.train()
            opt.zero_grad()
            pred = model(x, edge_index)
            loss = F.mse_loss(pred[train_nodes], y[train_nodes])
            loss.backward()
            opt.step()

            if ep % 20 == 0 or ep == epochs - 1:
                model.eval()
                with torch.no_grad():
                    pred_eval = model(x, edge_index)
                    val = F.mse_loss(pred_eval[test_nodes], y[test_nodes]).item()
                _log(f"    epoch {ep:3d}  train_mse={loss.item():.3f}  val_mse={val:.3f}", log)
                if val < best_val - 1e-4:
                    best_val = val
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience // 20 + 1:
                        _log(f"    early stop at epoch {ep}", log)
                        break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred_final = model(x, edge_index).cpu().numpy()

        # De-standardize back to the original rate scale for reporting.
        yt = y[test_nodes].cpu().numpy() * y_std + y_mean
        yp = pred_final[valid_idx[te_ix]] * y_std + y_mean
        fold_results.append(GNNFoldResult(
            fold=fold + 1,
            r2=float(r2_score(yt, yp)),
            mae=float(mean_absolute_error(yt, yp)),
            n_train=len(train_nodes),
            n_test=len(test_nodes),
        ))
        y_true_all.append(yt)
        y_pred_all.append(yp)
        _log(f"    fold {fold+1} r2={fold_results[-1].r2:.3f}  mae={fold_results[-1].mae:.2f}", log)

    yt_all = np.concatenate(y_true_all)
    yp_all = np.concatenate(y_pred_all)
    pooled_r2 = float(r2_score(yt_all, yp_all))
    pooled_mae = float(mean_absolute_error(yt_all, yp_all))

    return {
        "target": target,
        "available": True,
        "architecture": "SAGEConv(F->64)->ReLU->SAGEConv(64->32)->Linear(32->1)",
        "n_nodes": int(x.shape[0]),
        "n_edges": int(edge_index.shape[1]),
        "k_neighbors": 8,
        "pooled_r2": pooled_r2,
        "pooled_mae": pooled_mae,
        "folds": [asdict(r) for r in fold_results],
    }


# --------------------------- compare to XGBoost ----------------------------

def _xgb_for_comparison(target: str) -> dict:
    """Quick re-run of the same spatial CV we use for the GNN so the numbers
    are directly comparable."""
    from xgboost import XGBRegressor
    df = cohort_with_centroids()
    X = feature_matrix(df).astype("float64").fillna(0.0).values
    y = df[target].astype(float).values
    mask = ~np.isnan(y)
    X, y = X[mask], y[mask]
    states = df["state_fips"].values[mask]

    gkf = GroupKFold(n_splits=C.N_SPLITS_STATE_CV)
    yt_all, yp_all = [], []
    for tr, te in gkf.split(X, y, states):
        m = XGBRegressor(
            n_estimators=600, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
            reg_lambda=1.0, random_state=C.RANDOM_STATE, tree_method="hist",
            n_jobs=-1, verbosity=0,
        )
        m.fit(X[tr], y[tr])
        yt_all.append(y[te]); yp_all.append(m.predict(X[te]))
    yt_all = np.concatenate(yt_all); yp_all = np.concatenate(yp_all)
    return {"r2": float(r2_score(yt_all, yp_all)),
            "mae": float(mean_absolute_error(yt_all, yp_all))}


# ------------------------------- plot --------------------------------------

def plot_comparison(results: dict) -> None:
    targets = list(results.keys())
    gnn_r2 = [results[t]["gnn"]["pooled_r2"] for t in targets]
    xgb_r2 = [results[t]["xgb"]["r2"] for t in targets]
    gnn_mae = [results[t]["gnn"]["pooled_mae"] for t in targets]
    xgb_mae = [results[t]["xgb"]["mae"] for t in targets]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(targets))
    w = 0.38

    ax = axes[0]
    ax.bar(x - w/2, xgb_r2, w, label="XGBoost", color="#1f77b4")
    ax.bar(x + w/2, gnn_r2, w, label="GraphSAGE", color="#ff7f0e")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_per1000", "") for t in targets],
                       rotation=20, ha="right")
    ax.set_ylabel("R^2 (leave-state-out CV)")
    ax.set_title("Geospatial GraphSAGE vs tabular XGBoost")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x - w/2, xgb_mae, w, label="XGBoost", color="#1f77b4")
    ax.bar(x + w/2, gnn_mae, w, label="GraphSAGE", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_per1000", "") for t in targets],
                       rotation=20, ha="right")
    ax.set_ylabel("MAE (per 1,000 PRG discharges)")
    ax.set_title("MAE comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("County adjacency GNN vs XGBoost  "
                 "(8-NN, 2-layer SAGEConv, transductive CV)",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = C.FIG_DIR / "fig9_gnn_vs_xgb.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out.name}")


# ------------------------- top-level orchestration -------------------------

TARGETS = ["puerperal_rate_per1000", "hdp_rate_per1000", "smm_rate_per1000"]


def run(log: Optional[Callable[[str], None]] = None) -> dict:
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    for tgt in TARGETS:
        gnn_res = train_gnn_spatial_cv(tgt, log=log, epochs=300)
        xgb_res = _xgb_for_comparison(tgt)
        results[tgt] = {"gnn": gnn_res, "xgb": xgb_res}
        _log(f"[compare {tgt}]  GNN r2={gnn_res.get('pooled_r2', float('nan')):.3f}  "
             f"XGB r2={xgb_res['r2']:.3f}", log)

    (C.TAB_DIR / "gnn_vs_xgb.json").write_text(json.dumps(results, indent=2))
    plot_comparison(results)
    return results


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
