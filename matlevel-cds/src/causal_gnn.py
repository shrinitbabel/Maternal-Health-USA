"""DoubleML / partially-linear causal model with **GraphSAGE** as the
cross-fitted nuisance learner.

This replaces the XGBoost-based nuisance learners in src.causal. With XGBoost
retired from the project's primary methods, the causal estimates are now
derived under the same model class used by the production CDS scorer.

Estimator (Chernozhukov et al. 2018, partially-linear model)
------------------------------------------------------------
    Y = theta * D + g(X) + eps
    D = m(X) + v

where Y = log(outcome rate per 1,000), D = exposure (log_time3 or a binary
gap indicator), X = confounders (SVI themes 1-4, demographics, log_pop,
reporting_completeness, is_level3_host).

We cross-fit g and m using GraphSAGE on the 8-NN county adjacency graph,
with leave-state-out folds. Inside each fold:
  - Train GraphSAGE on the FULL graph but mask the loss to training nodes
    (transductive training, identical to what gnn_eval and gnn_deploy use).
  - Predict on the held-out state's nodes -> y_hat[te], d_hat[te]
After cross-fitting, theta is recovered via the orthogonalized regression
of y_res on d_res, with influence-function standard errors (eq. 5.5 in
Chernozhukov 2018).

Sensitivity analyses (matching src.causal):
  - main (all counties)
  - excl_level3_host  (drops counties hosting a Level III facility; the
                       referral-capture-paradox specification)
  - rural_subset      (pop_decile <= 5)

Output: outputs/tables/dml_results.json (overwrites the XGB version).
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from . import config as C
from .analytic import FEATURES_CONFOUNDERS, build_analytic
from .geo import cohort_with_centroids, knn_adjacency
from .gnn import CountyGraphSAGE, _HAS_PYG


def _log(msg: str, sink: Optional[Callable[[str], None]] = None) -> None:
    print(msg)
    if sink is not None:
        try:
            sink(msg)
        except Exception:
            pass


@dataclass
class DMLResult:
    exposure: str
    outcome: str
    theta: float
    se: float
    ci_lo: float
    ci_hi: float
    zstat: float
    pval: float
    n: int
    n_states: int
    note: str = ""
    learner: str = "GraphSAGE (8-NN)"


# --------------------------- per-fold trainer ---------------------------

def _train_one(x: torch.Tensor, edge_index: torch.Tensor,
               y_std: torch.Tensor, train_nodes: torch.Tensor,
               seed: int, epochs: int = 250, lr: float = 1e-3) -> np.ndarray:
    """Train GraphSAGE with loss masked to train_nodes; return predictions
    on the FULL graph (caller picks out test indices)."""
    torch.manual_seed(seed)
    model = CountyGraphSAGE(in_dim=x.shape[1], hidden=64, out_dim=32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_loss = float("inf")
    best_state = None
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(x, edge_index)
        loss = F.mse_loss(pred[train_nodes], y_std[train_nodes])
        loss.backward()
        opt.step()
        if loss.item() < best_loss - 1e-5:
            best_loss = loss.item()
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(x, edge_index).cpu().numpy()


def _gnn_cross_fit(df_cohort: pd.DataFrame,
                   X_cols: list[str], y_col: str, d_col: str,
                   epochs: int = 250,
                   log: Optional[Callable[[str], None]] = None
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leave-state-out cross-fitted GraphSAGE residuals (y_res, d_res) and
    the row mask used to filter the cohort.

    df_cohort must already be in cohort_with_centroids row order so the
    edge_index aligns with row positions."""
    if not _HAS_PYG:
        raise RuntimeError("torch_geometric not available")

    fips_order, edges = knn_adjacency(k=8)
    assert np.array_equal(df_cohort["fips"].values, fips_order), \
        "df_cohort must be in cohort_with_centroids row order"

    X_raw = df_cohort[X_cols].astype("float64").fillna(0.0).values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw)

    y = df_cohort[y_col].astype(float).values
    d = df_cohort[d_col].astype(float).values
    states = df_cohort["state_fips"].values

    valid_mask = ~(np.isnan(y) | np.isnan(d))
    valid_idx = np.where(valid_mask)[0]
    valid_states = states[valid_idx]

    # Standardize y and d for stable GNN training; de-standardize on output.
    y_mean = float(np.nanmean(y)); y_sd = float(np.nanstd(y)) or 1.0
    d_mean = float(np.nanmean(d)); d_sd = float(np.nanstd(d)) or 1.0
    y_arr = (y - y_mean) / y_sd
    d_arr = (d - d_mean) / d_sd

    x_t = torch.tensor(X_std, dtype=torch.float32)
    edge_t = torch.tensor(edges, dtype=torch.long)
    y_t = torch.tensor(np.nan_to_num(y_arr, nan=0.0), dtype=torch.float32)
    d_t = torch.tensor(np.nan_to_num(d_arr, nan=0.0), dtype=torch.float32)

    n_splits = min(C.N_SPLITS_STATE_CV, len(np.unique(valid_states)))
    gkf = GroupKFold(n_splits=n_splits)

    y_hat = np.zeros_like(y, dtype=float)
    d_hat = np.zeros_like(d, dtype=float)

    for fold, (tr_pos, te_pos) in enumerate(gkf.split(valid_idx, valid_idx, valid_states)):
        train_global = valid_idx[tr_pos]
        test_global = valid_idx[te_pos]
        train_nodes = torch.tensor(train_global, dtype=torch.long)

        _log(f"    [dml-gnn fold {fold+1}/{n_splits}] train={len(train_global)} "
             f"test={len(test_global)}", log)

        # g(X) for outcome
        pred_y = _train_one(x_t, edge_t, y_t, train_nodes,
                            seed=C.RANDOM_STATE + fold, epochs=epochs)
        y_hat[test_global] = pred_y[test_global] * y_sd + y_mean

        # m(X) for treatment
        pred_d = _train_one(x_t, edge_t, d_t, train_nodes,
                            seed=C.RANDOM_STATE + 1000 + fold, epochs=epochs)
        d_hat[test_global] = pred_d[test_global] * d_sd + d_mean

    y_res = (y - y_hat)
    d_res = (d - d_hat)
    return y_res, d_res, valid_mask


# ------------------------- partial-linear estimator ---------------------

def _theta_from_residuals(y_res: np.ndarray, d_res: np.ndarray,
                          mask: np.ndarray) -> tuple[float, float, int]:
    yr, dr = y_res[mask], d_res[mask]
    n = len(yr)
    denom = float(np.sum(dr ** 2))
    if denom <= 0:
        return float("nan"), float("nan"), n
    theta = float(np.sum(dr * yr) / denom)
    psi = (yr - theta * dr) * dr
    var_theta = float(np.mean(psi ** 2) / (np.mean(dr ** 2) ** 2)) / n
    return theta, float(np.sqrt(var_theta)), n


def dml_partially_linear(df: pd.DataFrame, exposure: str, outcome: str,
                         X_cols: list[str] | None = None,
                         keep_mask: np.ndarray | None = None,
                         note: str = "",
                         log: Optional[Callable[[str], None]] = None) -> DMLResult:
    """df must be in cohort_with_centroids row order. keep_mask (optional)
    selects a sensitivity subsample (e.g. excl_level3_host) without changing
    the graph: we still train on the full adjacency, just restrict the rows
    used in the final theta calculation to keep_mask."""
    X_cols = X_cols or FEATURES_CONFOUNDERS
    d = df.copy()
    d["__y"] = np.log(d[outcome].astype(float).clip(lower=1e-3))

    y_res, d_res, valid_mask = _gnn_cross_fit(
        d, X_cols, "__y", exposure, log=log)

    final_mask = valid_mask
    if keep_mask is not None:
        final_mask = final_mask & keep_mask

    theta, se, n = _theta_from_residuals(y_res, d_res, final_mask)
    z = theta / se if se and se > 0 else float("nan")
    from math import erfc, sqrt
    pval = erfc(abs(z) / sqrt(2)) if not np.isnan(z) else float("nan")

    n_states = int(d.loc[final_mask, "state_fips"].nunique())
    return DMLResult(
        exposure=exposure, outcome=outcome,
        theta=theta, se=se,
        ci_lo=theta - 1.96 * se if se else float("nan"),
        ci_hi=theta + 1.96 * se if se else float("nan"),
        zstat=z, pval=pval, n=n, n_states=n_states, note=note,
    )


# ------------------------------ orchestration ---------------------------

# The four headline outcomes from the manuscript causal-analysis section.
# (placenta, DM are still served by GraphSAGE for predictive scoring; we don't
# run causal estimation on them because their HCUP cell-suppression rates are
# too high (~43-48%) for stable spatial cross-fitting on rural subsamples.)
OUTCOMES = [
    "smm_rate_per1000",
    "hdp_rate_per1000",
    "childbirth_cx_rate_per1000",
    "puerperal_rate_per1000",
]
EXPOSURE_PRIMARY = "log_time3"


def run(log: Optional[Callable[[str], None]] = None) -> dict:
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)

    cohort = cohort_with_centroids().reset_index(drop=True)
    keep_excl = (cohort["is_level3_host"] == 0).values
    keep_rural = (cohort["pop_decile"] <= 5).values

    results: dict = {}
    for outcome in OUTCOMES:
        results[outcome] = {EXPOSURE_PRIMARY: {}}
        _log(f"\n[dml-gnn] {outcome} x {EXPOSURE_PRIMARY}", log)

        main = dml_partially_linear(
            cohort, EXPOSURE_PRIMARY, outcome,
            note="main (all counties)", log=log,
        )
        sens1 = dml_partially_linear(
            cohort, EXPOSURE_PRIMARY, outcome,
            X_cols=[c for c in FEATURES_CONFOUNDERS if c != "is_level3_host"],
            keep_mask=keep_excl,
            note="exclude level-III host counties", log=log,
        )
        sens2 = dml_partially_linear(
            cohort, EXPOSURE_PRIMARY, outcome,
            keep_mask=keep_rural,
            note="rural subset (pop decile <= 5)", log=log,
        )

        results[outcome][EXPOSURE_PRIMARY] = {
            "main": asdict(main),
            "excl_level3_host": asdict(sens1),
            "rural_subset": asdict(sens2),
        }
        _log(f"[dml-gnn] {outcome:<32s} main theta={main.theta:+.4f} "
             f"p={main.pval:.3f} | rural theta={sens2.theta:+.4f} "
             f"p={sens2.pval:.3f}", log)

    out = C.TAB_DIR / "dml_results.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    _log(f"[ok] wrote {out.name} (GraphSAGE-nuisance DoubleML, "
         f"{len(OUTCOMES)} outcomes)", log)
    return results


if __name__ == "__main__":
    run()
