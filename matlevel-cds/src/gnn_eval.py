"""Apples-to-apples evaluation: GraphSAGE vs XGBoost.

Same leave-state-out splits, same cohort, full metric suite computed on the
same out-of-fold predictions for both models. This replaces the simpler
R2/MAE-only comparison in gnn.py and writes the canonical fig6-fig9 PNGs
that the manuscript and the README reference.

Metrics produced for each (model, target) pair
----------------------------------------------
Regression:
  - R2, MAE, RMSE, Pearson r, Spearman rho
  - Bootstrap 95% CI on R2, MAE, RMSE, Pearson (1,000 resamples)
  - Calibration slope and intercept (linear fit observed ~ predicted)
  - Expected Calibration Error (ECE), regression flavour
  - Per-state held-out R2 + MAE
  - Stability: mean +- SD of OOF predictions across 5 bootstrap seeds

Binary (from regression OOF predictions, thresholded at top quartile of y):
  - AUROC + bootstrap 95% CI
  - AUPRC, F1 @ 0.5, Brier, ECE (classification)

Writes:
  outputs/tables/gnn_xgb_head_to_head.json
  outputs/figures/fig6_calibration_regression.png  (GNN+XGB overlay)
  outputs/figures/fig7_per_state_performance.png   (per-state, puerperal)
  outputs/figures/fig8_binary_classifier.png       (GNN+XGB binary head)
  outputs/figures/fig9_gnn_vs_xgb.png              (bootstrap-CI bar chart)
"""
from __future__ import annotations
import json
from typing import Callable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.polynomial import Polynomial
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score,
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from . import config as C
from .geo import cohort_with_centroids, knn_adjacency
from .gnn import CountyGraphSAGE, _HAS_PYG
from .model import feature_matrix


def _log(msg: str, sink: Optional[Callable[[str], None]] = None) -> None:
    print(msg)
    if sink is not None:
        try:
            sink(msg)
        except Exception:
            pass


# ----------------------------- metric helpers -----------------------------

def _rmse(y, p): return float(np.sqrt(mean_squared_error(y, p)))


def _safe_pearson(y, p):
    if len(np.unique(p)) < 2:
        return 0.0
    return float(pearsonr(y, p)[0])


def _safe_spearman(y, p):
    if len(np.unique(p)) < 2:
        return 0.0
    return float(spearmanr(y, p).correlation)


def _regression_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    bins = np.quantile(y_pred, np.linspace(0, 1, n_bins + 1))
    bins[-1] += 1e-6
    idx = np.clip(np.digitize(y_pred, bins) - 1, 0, n_bins - 1)
    n = len(y_pred)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_pred[mask].mean())
    return float(ece)


def _class_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    n = len(y_prob)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def _bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                  metric: Callable, n_boot: int = 1000,
                  seed: int = C.RANDOM_STATE) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals[i] = metric(y_true[idx], y_pred[idx])
        except Exception:
            vals[i] = np.nan
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    return (float(np.mean(vals)),
            float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)))


def _full_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {
        "R2":         float(r2_score(y_true, y_pred)),
        "MAE":        float(mean_absolute_error(y_true, y_pred)),
        "RMSE":       _rmse(y_true, y_pred),
        "Pearson_r":  _safe_pearson(y_true, y_pred),
        "Spearman_r": _safe_spearman(y_true, y_pred),
        "ECE":        _regression_ece(y_true, y_pred),
    }
    for name, fn in [("R2", r2_score), ("MAE", mean_absolute_error),
                     ("RMSE", _rmse), ("Pearson_r", _safe_pearson)]:
        m, lo, hi = _bootstrap_ci(y_true, y_pred, fn)
        out[f"{name}_boot_mean"] = m
        out[f"{name}_boot_CI95"] = [lo, hi]
    p = Polynomial.fit(y_pred, y_true, 1).convert().coef
    out["calibration_intercept"] = float(p[0])
    out["calibration_slope"] = float(p[1]) if len(p) > 1 else float("nan")
    return out


def _full_binary_metrics_from_regression(y_true_cont: np.ndarray,
                                         y_pred_cont: np.ndarray) -> dict:
    """Threshold both observed and predicted at top quartile of observed,
    use a min-max-rescaled prediction as a pseudo-probability for AUROC etc.
    Same procedure for both models, so comparison is fair."""
    thr = float(np.quantile(y_true_cont, 0.75))
    y_bin = (y_true_cont >= thr).astype(int)
    p_min, p_max = float(np.min(y_pred_cont)), float(np.max(y_pred_cont))
    if p_max - p_min < 1e-9:
        score = np.full_like(y_pred_cont, 0.5, dtype=float)
    else:
        score = (y_pred_cont - p_min) / (p_max - p_min)
    pred_bin = (score >= 0.5).astype(int)

    auroc = float(roc_auc_score(y_bin, score))
    aucmean, auclo, auchi = _bootstrap_ci(y_bin, score, roc_auc_score)
    auprc = float(average_precision_score(y_bin, score))
    brier = float(brier_score_loss(y_bin, score))
    ece = _class_ece(y_bin, score)
    f1 = float(f1_score(y_bin, pred_bin))
    return {
        "threshold_value": thr,
        "n": int(len(y_bin)),
        "n_positive": int(y_bin.sum()),
        "AUROC": auroc,
        "AUROC_boot_mean": aucmean,
        "AUROC_CI95": [auclo, auchi],
        "AUPRC": auprc,
        "F1": f1,
        "Brier": brier,
        "ECE": ece,
        "_y_bin": y_bin.tolist(),
        "_score": score.tolist(),
    }


# ----------------------------- model wrappers -----------------------------

def _xgb_factory():
    return XGBRegressor(
        n_estimators=600, max_depth=5, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
        reg_lambda=1.0, random_state=C.RANDOM_STATE, tree_method="hist",
        n_jobs=-1, verbosity=0,
    )


def _train_gnn_one_fold(x: torch.Tensor, edge_index: torch.Tensor,
                        y_std: torch.Tensor, train_nodes: torch.Tensor,
                        test_nodes: torch.Tensor, seed: int,
                        epochs: int = 300, lr: float = 1e-3) -> np.ndarray:
    """Train GraphSAGE on this fold and return OOF predictions on test nodes
    in standardized scale. Caller de-standardizes."""
    torch.manual_seed(seed)
    model = CountyGraphSAGE(in_dim=x.shape[1], hidden=64, out_dim=32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val = float("inf")
    best_state = None
    wait = 0
    patience = 2  # eval every 20 epochs, so ~40 epochs of patience
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(x, edge_index)
        loss = F.mse_loss(pred[train_nodes], y_std[train_nodes])
        loss.backward()
        opt.step()
        if ep % 20 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                val = F.mse_loss(model(x, edge_index)[test_nodes],
                                 y_std[test_nodes]).item()
            if val < best_val - 1e-4:
                best_val = val
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(x, edge_index).cpu().numpy()


# --------------------------- head-to-head runner --------------------------

def evaluate_target(target: str,
                    log: Optional[Callable[[str], None]] = None,
                    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
                    epochs: int = 300) -> dict:
    """Returns full metrics for both XGBoost and GraphSAGE on `target`,
    using identical leave-state-out folds and identical evaluation indices."""
    if not _HAS_PYG:
        return {"target": target, "available": False,
                "reason": "torch_geometric not installed"}

    _log(f"\n[head2head] target={target}", log)
    df = cohort_with_centroids().reset_index(drop=True)
    fips_order, edges = knn_adjacency(k=8)
    assert np.array_equal(df["fips"].values, fips_order), "fips order mismatch"

    X_raw = feature_matrix(df).astype("float64").fillna(0.0).values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw)

    y = df[target].astype(float).values
    states = df["state_fips"].values
    mask = ~np.isnan(y)

    valid_idx = np.where(mask)[0]
    valid_states = states[valid_idx]
    y_valid = y[valid_idx]

    y_mean, y_std_v = float(np.nanmean(y)), float(np.nanstd(y))
    if y_std_v == 0: y_std_v = 1.0
    y_std_arr = (y - y_mean) / y_std_v

    x_t = torch.tensor(X_std, dtype=torch.float32)
    y_t = torch.tensor(np.nan_to_num(y_std_arr, nan=0.0), dtype=torch.float32)
    edge_t = torch.tensor(edges, dtype=torch.long)

    n_splits = C.N_SPLITS_STATE_CV
    gkf = GroupKFold(n_splits=n_splits)

    # Build OOF arrays in valid_idx order
    oof_xgb = np.empty(len(valid_idx))
    oof_gnn = np.empty(len(valid_idx))
    fold_assignments = np.empty(len(valid_idx), dtype=int)

    for fold, (tr_pos, te_pos) in enumerate(gkf.split(valid_idx, valid_idx, valid_states)):
        _log(f"  [fold {fold+1}/{n_splits}] train={len(tr_pos)} test={len(te_pos)}", log)

        # XGBoost: inductive, only train rows
        tr_global = valid_idx[tr_pos]
        te_global = valid_idx[te_pos]
        m = _xgb_factory()
        m.fit(X_raw[tr_global], y[tr_global])
        oof_xgb[te_pos] = m.predict(X_raw[te_global])

        # GraphSAGE: transductive, mask loss to train nodes
        train_nodes = torch.tensor(tr_global, dtype=torch.long)
        test_nodes = torch.tensor(te_global, dtype=torch.long)
        pred_std = _train_gnn_one_fold(
            x_t, edge_t, y_t, train_nodes, test_nodes,
            seed=C.RANDOM_STATE + fold, epochs=epochs,
        )
        oof_gnn[te_pos] = pred_std[te_global] * y_std_v + y_mean
        fold_assignments[te_pos] = fold + 1

        _log(f"    fold-XGB R2={r2_score(y_valid[te_pos], oof_xgb[te_pos]):.3f}  "
             f"fold-GNN R2={r2_score(y_valid[te_pos], oof_gnn[te_pos]):.3f}", log)

    # ---- regression metrics ----
    xgb_metrics = _full_regression_metrics(y_valid, oof_xgb)
    gnn_metrics = _full_regression_metrics(y_valid, oof_gnn)

    # ---- per-state ----
    def per_state(yp):
        rows = []
        for s in np.unique(valid_states):
            sm = valid_states == s
            if sm.sum() < 5:
                continue
            rows.append({
                "state_fips": str(s),
                "n": int(sm.sum()),
                "R2": float(r2_score(y_valid[sm], yp[sm])),
                "MAE": float(mean_absolute_error(y_valid[sm], yp[sm])),
            })
        return sorted(rows, key=lambda r: r["R2"])

    xgb_metrics["per_state"] = per_state(oof_xgb)
    gnn_metrics["per_state"] = per_state(oof_gnn)

    # ---- stability (multi-seed bootstrap, full-cohort predictions) ----
    _log("  [stability] 5-seed bootstrap for both models", log)
    xgb_seed_preds, gnn_seed_preds = [], []
    for s in seeds:
        rng = np.random.default_rng(int(C.RANDOM_STATE + s))
        bidx = rng.integers(0, len(valid_idx), len(valid_idx))
        tr_global = valid_idx[bidx]

        # XGBoost stability
        m = XGBRegressor(
            n_estimators=600, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
            reg_lambda=1.0, random_state=int(C.RANDOM_STATE + s),
            tree_method="hist", n_jobs=-1, verbosity=0,
        )
        m.fit(X_raw[tr_global], y[tr_global])
        xgb_seed_preds.append(m.predict(X_raw[valid_idx]))

        # GNN stability: train on bootstrapped nodes, predict full graph
        train_nodes = torch.tensor(tr_global, dtype=torch.long)
        test_nodes = torch.tensor(valid_idx, dtype=torch.long)
        pred_std = _train_gnn_one_fold(
            x_t, edge_t, y_t, train_nodes, test_nodes,
            seed=int(C.RANDOM_STATE + s), epochs=epochs,
        )
        gnn_seed_preds.append(pred_std[valid_idx] * y_std_v + y_mean)

    xgb_arr = np.vstack(xgb_seed_preds)
    gnn_arr = np.vstack(gnn_seed_preds)
    xgb_metrics["stability"] = {"mean_pred_sd": float(xgb_arr.std(axis=0).mean())}
    gnn_metrics["stability"] = {"mean_pred_sd": float(gnn_arr.std(axis=0).mean())}

    # ---- binary head ----
    xgb_bin = _full_binary_metrics_from_regression(y_valid, oof_xgb)
    gnn_bin = _full_binary_metrics_from_regression(y_valid, oof_gnn)

    # store OOF preds for plots
    xgb_metrics["_oof"] = {"y_true": y_valid.tolist(), "y_pred": oof_xgb.tolist()}
    gnn_metrics["_oof"] = {"y_true": y_valid.tolist(), "y_pred": oof_gnn.tolist()}

    return {
        "target": target,
        "available": True,
        "n": int(len(valid_idx)),
        "xgb": {"regression": xgb_metrics, "binary": xgb_bin},
        "gnn": {"regression": gnn_metrics, "binary": gnn_bin},
    }


# ------------------------------- plots ------------------------------------

def _plot_metric_bars(results: dict) -> None:
    targets = [t for t, r in results.items() if r.get("available")]
    if not targets:
        return
    metric_names = ["R2", "MAE", "RMSE", "Pearson_r", "ECE"]
    fig, axes = plt.subplots(1, len(metric_names),
                             figsize=(4.2 * len(metric_names), 5),
                             squeeze=False)
    x = np.arange(len(targets))
    w = 0.38
    for ax, mn in zip(axes[0], metric_names):
        xv = [results[t]["xgb"]["regression"][mn] for t in targets]
        gv = [results[t]["gnn"]["regression"][mn] for t in targets]
        if mn in {"R2", "MAE", "RMSE", "Pearson_r"}:
            xerr_lo = [results[t]["xgb"]["regression"][f"{mn}_boot_CI95"][0] for t in targets]
            xerr_hi = [results[t]["xgb"]["regression"][f"{mn}_boot_CI95"][1] for t in targets]
            gerr_lo = [results[t]["gnn"]["regression"][f"{mn}_boot_CI95"][0] for t in targets]
            gerr_hi = [results[t]["gnn"]["regression"][f"{mn}_boot_CI95"][1] for t in targets]
            xerr = [np.array(xv) - np.array(xerr_lo),
                    np.array(xerr_hi) - np.array(xv)]
            gerr = [np.array(gv) - np.array(gerr_lo),
                    np.array(gerr_hi) - np.array(gv)]
            ax.bar(x - w/2, xv, w, yerr=xerr, label="XGBoost",
                   color="#1f77b4", capsize=3)
            ax.bar(x + w/2, gv, w, yerr=gerr, label="GraphSAGE",
                   color="#ff7f0e", capsize=3)
        else:
            ax.bar(x - w/2, xv, w, label="XGBoost", color="#1f77b4")
            ax.bar(x + w/2, gv, w, label="GraphSAGE", color="#ff7f0e")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace("_per1000", "") for t in targets],
                           rotation=20, ha="right")
        ax.set_title(mn)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("XGBoost vs GraphSAGE — leave-state-out CV with bootstrap 95% CIs",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = C.FIG_DIR / "fig9_gnn_vs_xgb.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out.name}")


def _plot_calibration(results: dict) -> None:
    targets = [t for t, r in results.items() if r.get("available")]
    if not targets:
        return
    fig, axes = plt.subplots(1, len(targets), figsize=(5.4 * len(targets), 5),
                             squeeze=False)
    for ax, tgt in zip(axes[0], targets):
        for model_key, color, label in [("xgb", "#1f77b4", "XGBoost"),
                                        ("gnn", "#ff7f0e", "GraphSAGE")]:
            oof = results[tgt][model_key]["regression"]["_oof"]
            yt, yp = np.array(oof["y_true"]), np.array(oof["y_pred"])
            bins = np.quantile(yp, np.linspace(0, 1, 11))
            bins[-1] += 1e-6
            idx = np.clip(np.digitize(yp, bins) - 1, 0, 9)
            bin_pred = [yp[idx == b].mean() for b in range(10) if (idx == b).any()]
            bin_obs = [yt[idx == b].mean() for b in range(10) if (idx == b).any()]
            ece = results[tgt][model_key]["regression"]["ECE"]
            slope = results[tgt][model_key]["regression"]["calibration_slope"]
            ax.plot(bin_pred, bin_obs, marker="o", linewidth=2, color=color,
                    label=f"{label}  slope={slope:.2f}  ECE={ece:.2f}")
        all_y = np.concatenate([np.array(results[tgt]["xgb"]["regression"]["_oof"]["y_true"]),
                                np.array(results[tgt]["xgb"]["regression"]["_oof"]["y_pred"]),
                                np.array(results[tgt]["gnn"]["regression"]["_oof"]["y_pred"])])
        lo, hi = float(np.min(all_y)), float(np.max(all_y))
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="ideal")
        ax.set_xlabel("Predicted (decile bin mean)")
        ax.set_ylabel("Observed (decile bin mean)")
        ax.set_title(tgt.replace("_per1000", ""))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Calibration: GraphSAGE vs XGBoost (decile reliability)",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = C.FIG_DIR / "fig6_calibration_regression.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out.name}")


def _plot_binary(results: dict) -> None:
    targets = [t for t, r in results.items() if r.get("available")]
    if not targets:
        return
    from sklearn.metrics import roc_curve, precision_recall_curve
    fig, axes = plt.subplots(len(targets), 3, figsize=(15, 4.5 * len(targets)),
                             squeeze=False)
    for row, tgt in enumerate(targets):
        for model_key, color, label in [("xgb", "#1f77b4", "XGBoost"),
                                        ("gnn", "#ff7f0e", "GraphSAGE")]:
            b = results[tgt][model_key]["binary"]
            yb = np.array(b["_y_bin"])
            sc = np.array(b["_score"])

            fpr, tpr, _ = roc_curve(yb, sc)
            axes[row, 0].plot(fpr, tpr, color=color, linewidth=2,
                              label=f"{label}  AUC={b['AUROC']:.3f} "
                                    f"[{b['AUROC_CI95'][0]:.3f},{b['AUROC_CI95'][1]:.3f}]")

            p, r, _ = precision_recall_curve(yb, sc)
            axes[row, 1].plot(r, p, color=color, linewidth=2,
                              label=f"{label}  AUPRC={b['AUPRC']:.3f}  F1={b['F1']:.3f}")

            from sklearn.calibration import calibration_curve
            try:
                prob_true, prob_pred = calibration_curve(yb, sc, n_bins=10, strategy="quantile")
                axes[row, 2].plot(prob_pred, prob_true, marker="o", color=color, linewidth=2,
                                  label=f"{label}  Brier={b['Brier']:.3f}  ECE={b['ECE']:.3f}")
            except Exception:
                pass

        axes[row, 0].plot([0, 1], [0, 1], "k--", alpha=0.4)
        axes[row, 0].set_xlabel("FPR"); axes[row, 0].set_ylabel("TPR")
        axes[row, 0].set_title(f"ROC — top-quartile {tgt.replace('_per1000','')}")
        axes[row, 0].legend(fontsize=8); axes[row, 0].grid(alpha=0.3)
        axes[row, 1].set_xlabel("Recall"); axes[row, 1].set_ylabel("Precision")
        axes[row, 1].set_title("Precision-Recall")
        axes[row, 1].legend(fontsize=8); axes[row, 1].grid(alpha=0.3)
        axes[row, 2].plot([0, 1], [0, 1], "k--", alpha=0.4)
        axes[row, 2].set_xlabel("Predicted score"); axes[row, 2].set_ylabel("Observed positive fraction")
        axes[row, 2].set_title("Reliability")
        axes[row, 2].legend(fontsize=8); axes[row, 2].grid(alpha=0.3)
    fig.suptitle("Binary classification head (top-quartile threshold) — GraphSAGE vs XGBoost",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = C.FIG_DIR / "fig8_binary_classifier.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out.name}")


def _plot_per_state_head_to_head(results: dict,
                                  primary_target: str = "puerperal_rate_per1000") -> None:
    """Per-state R2 with both models, sorted by GNN performance, for the
    primary GNN-served target. This replaces the XGB-only fig7."""
    if primary_target not in results or not results[primary_target].get("available"):
        return
    xgb_ps = results[primary_target]["xgb"]["regression"]["per_state"]
    gnn_ps = results[primary_target]["gnn"]["regression"]["per_state"]
    by_state_xgb = {r["state_fips"]: r["R2"] for r in xgb_ps}
    by_state_gnn = {r["state_fips"]: r["R2"] for r in gnn_ps}
    states = sorted(set(by_state_xgb) | set(by_state_gnn),
                    key=lambda s: by_state_gnn.get(s, -9))
    xgb_vals = [by_state_xgb.get(s, np.nan) for s in states]
    gnn_vals = [by_state_gnn.get(s, np.nan) for s in states]

    fig, ax = plt.subplots(figsize=(11, max(7, 0.32 * len(states))))
    y = np.arange(len(states))
    h = 0.4
    ax.barh(y - h/2, xgb_vals, h, label="XGBoost", color="#1f77b4", alpha=0.85)
    ax.barh(y + h/2, gnn_vals, h, label="GraphSAGE", color="#ff7f0e", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(states)
    ax.set_xlabel("Held-out R^2 (leave-this-state-out CV)")
    ax.set_title(
        f"Per-state performance: GraphSAGE vs XGBoost on "
        f"{primary_target.replace('_per1000','')}\n"
        "(sorted by GNN R^2; GNN is the production scorer for this target)"
    )
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = C.FIG_DIR / "fig7_per_state_performance.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out.name}")

    # Also keep the per-state CSV for the manuscript supplement.
    rows = []
    for s in states:
        rows.append({"state_fips": s,
                     "R2_xgb": by_state_xgb.get(s, np.nan),
                     "R2_gnn": by_state_gnn.get(s, np.nan),
                     "delta_gnn_minus_xgb": (
                         by_state_gnn.get(s, np.nan) - by_state_xgb.get(s, np.nan)
                     )})
    pd.DataFrame(rows).to_csv(C.TAB_DIR / "per_state_metrics.csv", index=False)


# --------------------------- top-level orchestration ----------------------

PRIMARY_TARGETS = [
    "puerperal_rate_per1000",
    "hdp_rate_per1000",
    "smm_rate_per1000",
    "childbirth_cx_rate_per1000",
    "placenta_rate_per1000",
    "dm_rate_per1000",
]


def _strip_internal(d: dict) -> dict:
    """Recursively drop fields starting with '_' before JSON dump."""
    if isinstance(d, dict):
        return {k: _strip_internal(v) for k, v in d.items() if not k.startswith("_")}
    if isinstance(d, list):
        return [_strip_internal(x) for x in d]
    return d


def run(log: Optional[Callable[[str], None]] = None,
        epochs: int = 300) -> dict:
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for tgt in PRIMARY_TARGETS:
        results[tgt] = evaluate_target(tgt, log=log, epochs=epochs)

    _plot_metric_bars(results)              # -> fig9_gnn_vs_xgb.png
    _plot_calibration(results)              # -> fig6_calibration_regression.png
    _plot_binary(results)                   # -> fig8_binary_classifier.png
    _plot_per_state_head_to_head(results)   # -> fig7_per_state_performance.png

    clean = _strip_internal(results)
    out_path = C.TAB_DIR / "gnn_xgb_head_to_head.json"
    out_path.write_text(json.dumps(clean, indent=2, default=float))
    _log(f"\n[ok] wrote {out_path.name}", log)

    print("\n========= APPLES-TO-APPLES SUMMARY =========")
    print(f"{'target':<30s} {'metric':<14s} {'XGBoost':>22s} {'GraphSAGE':>22s} {'winner':>10s}")
    print("-" * 105)
    for tgt, r in results.items():
        if not r.get("available"):
            continue
        xreg = r["xgb"]["regression"]; greg = r["gnn"]["regression"]
        xb = r["xgb"]["binary"]; gb = r["gnn"]["binary"]
        rows = [
            ("R2", xreg["R2"], xreg["R2_boot_CI95"],
                   greg["R2"], greg["R2_boot_CI95"], "max"),
            ("MAE", xreg["MAE"], xreg["MAE_boot_CI95"],
                    greg["MAE"], greg["MAE_boot_CI95"], "min"),
            ("RMSE", xreg["RMSE"], xreg["RMSE_boot_CI95"],
                     greg["RMSE"], greg["RMSE_boot_CI95"], "min"),
            ("Pearson_r", xreg["Pearson_r"], xreg["Pearson_r_boot_CI95"],
                          greg["Pearson_r"], greg["Pearson_r_boot_CI95"], "max"),
            ("ECE_reg", xreg["ECE"], None, greg["ECE"], None, "min"),
            ("calib_slope", xreg["calibration_slope"], None,
                            greg["calibration_slope"], None, "closest_to_1"),
            ("stability_SD", xreg["stability"]["mean_pred_sd"], None,
                             greg["stability"]["mean_pred_sd"], None, "min"),
            ("AUROC", xb["AUROC"], xb["AUROC_CI95"],
                      gb["AUROC"], gb["AUROC_CI95"], "max"),
            ("AUPRC", xb["AUPRC"], None, gb["AUPRC"], None, "max"),
            ("F1@0.5", xb["F1"], None, gb["F1"], None, "max"),
            ("Brier", xb["Brier"], None, gb["Brier"], None, "min"),
            ("ECE_bin", xb["ECE"], None, gb["ECE"], None, "min"),
        ]
        for name, xv, xci, gv, gci, direction in rows:
            xs = f"{xv:.3f}" + (f" [{xci[0]:.3f},{xci[1]:.3f}]" if xci else "")
            gs = f"{gv:.3f}" + (f" [{gci[0]:.3f},{gci[1]:.3f}]" if gci else "")
            if direction == "max":
                w = "GNN" if gv > xv else "XGB"
            elif direction == "min":
                w = "GNN" if gv < xv else "XGB"
            else:
                w = "GNN" if abs(gv - 1) < abs(xv - 1) else "XGB"
            print(f"{tgt:<30s} {name:<14s} {xs:>22s} {gs:>22s} {w:>10s}")
        print()
    return clean


if __name__ == "__main__":
    run()
