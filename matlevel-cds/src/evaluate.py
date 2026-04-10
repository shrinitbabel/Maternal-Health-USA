"""Comprehensive performance evaluation for MatLevel-CDS.

Goes well beyond the simple R2/MAE reported in `src.model`. Produces:

  Regression head (continuous outcome = rate per 1,000 PRG discharges)
  ----------------------------------------------------------------------
  - Leave-state-out CV R2, MAE, RMSE, Pearson r, Spearman rho
  - Bootstrapped 95% CI on each metric (1,000 resamples of fold-level preds)
  - Baseline comparisons: mean-predictor, linear Ridge
  - Calibration curve (predicted decile vs observed mean) + slope/intercept
  - Expected Calibration Error (ECE) for regression (binned reliability)
  - Per-state held-out R2 + MAE (which states are hardest?)
  - Stability: mean +/- SD of predictions across 5 bootstrap seeds

  Binary head (outcome = top-quartile of puerperal rate -> "high burden")
  ----------------------------------------------------------------------
  - AUROC, AUPRC, F1 at threshold=0.5, precision, recall, accuracy
  - Brier score + ECE (classification)
  - Confusion matrix
  - Calibration curve

Writes:
  outputs/tables/full_metrics.json
  outputs/tables/per_state_metrics.csv
  outputs/figures/fig6_calibration_regression.png
  outputs/figures/fig7_per_state_performance.png
  outputs/figures/fig8_binary_classifier.png
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import Callable, Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.calibration import calibration_curve
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier, XGBRegressor

from . import config as C
from .analytic import build_analytic
from .model import feature_matrix


# ----------------- utilities -----------------

def _log(msg: str, sink: Optional[Callable[[str], None]] = None) -> None:
    print(msg)
    if sink is not None:
        try:
            sink(msg)
        except Exception:
            pass


def _bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                  metric: Callable[[np.ndarray, np.ndarray], float],
                  n_boot: int = 1000, seed: int = C.RANDOM_STATE) -> tuple[float, float, float]:
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
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _safe_pearson(y_true, y_pred):
    if len(np.unique(y_pred)) < 2:
        return 0.0
    return float(pearsonr(y_true, y_pred)[0])


def _safe_spearman(y_true, y_pred):
    if len(np.unique(y_pred)) < 2:
        return 0.0
    return float(spearmanr(y_true, y_pred).correlation)


def _rmse(y, p): return float(np.sqrt(mean_squared_error(y, p)))


def _regression_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """Binned reliability error for regression: mean |observed - predicted|
    across equal-width prediction bins."""
    bins = np.quantile(y_pred, np.linspace(0, 1, n_bins + 1))
    bins[-1] += 1e-6
    idx = np.digitize(y_pred, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    n = len(y_pred)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        gap = abs(y_true[mask].mean() - y_pred[mask].mean())
        ece += (mask.sum() / n) * gap
    return float(ece)


def _class_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Classic classification ECE: mean |accuracy - confidence| across
    equal-width probability bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ece = 0.0
    n = len(y_prob)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def _xgb_reg() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=600, max_depth=5, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
        reg_lambda=1.0, random_state=C.RANDOM_STATE, tree_method="hist",
        n_jobs=-1, verbosity=0,
    )


def _xgb_clf() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=600, max_depth=5, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
        reg_lambda=1.0, random_state=C.RANDOM_STATE, tree_method="hist",
        n_jobs=-1, verbosity=0, eval_metric="logloss",
    )


# --------------------------- regression evaluation ---------------------------

def _spatial_cv_preds(df: pd.DataFrame, target: str,
                      model_factory: Callable[[], object],
                      n_splits: Optional[int] = None,
                      log: Optional[Callable[[str], None]] = None
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    d = df.dropna(subset=[target, "state_fips"]).copy()
    X = feature_matrix(d).astype("float64").fillna(0.0).values
    y = d[target].astype(float).values
    groups = d["state_fips"].values

    n_splits = n_splits or min(C.N_SPLITS_STATE_CV, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    y_true_all = np.empty(0)
    y_pred_all = np.empty(0)
    state_all = np.empty(0, dtype=object)
    fold_metrics = []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        _log(f"    [cv fold {fold+1}/{n_splits}] train={len(tr)} test={len(te)}", log)
        model = model_factory()
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        fold_metrics.append({
            "fold": fold + 1,
            "r2": float(r2_score(y[te], pred)),
            "mae": float(mean_absolute_error(y[te], pred)),
            "rmse": _rmse(y[te], pred),
        })
        y_true_all = np.concatenate([y_true_all, y[te]])
        y_pred_all = np.concatenate([y_pred_all, pred])
        state_all = np.concatenate([state_all, groups[te]])

    return y_true_all, y_pred_all, state_all, fold_metrics


def evaluate_regression(df: pd.DataFrame, target: str,
                        log: Optional[Callable[[str], None]] = None) -> dict:
    """Full regression metrics for a single target: XGBoost, Ridge, mean baseline."""
    _log(f"  [regression] target={target}", log)
    res: dict = {"target": target}

    # XGBoost spatial CV
    yt, yp, st, folds = _spatial_cv_preds(df, target, _xgb_reg, log=log)
    res["n"] = int(len(yt))
    res["xgb"] = {
        "R2":    float(r2_score(yt, yp)),
        "MAE":   float(mean_absolute_error(yt, yp)),
        "RMSE":  _rmse(yt, yp),
        "Pearson_r":  _safe_pearson(yt, yp),
        "Spearman_r": _safe_spearman(yt, yp),
        "ECE":   _regression_ece(yt, yp),
        "fold_metrics": folds,
    }
    # Bootstrap CIs for the headline metrics
    for name, fn in [("R2", r2_score), ("MAE", mean_absolute_error), ("RMSE", _rmse)]:
        mean, lo, hi = _bootstrap_ci(yt, yp, fn)
        res["xgb"][f"{name}_boot_mean"] = mean
        res["xgb"][f"{name}_boot_CI95"] = [lo, hi]

    # Calibration slope/intercept (linear fit of observed ~ predicted)
    from numpy.polynomial import Polynomial
    p = Polynomial.fit(yp, yt, 1).convert().coef
    res["xgb"]["calibration_intercept"] = float(p[0])
    res["xgb"]["calibration_slope"] = float(p[1]) if len(p) > 1 else float("nan")

    # Baselines -------------------------------------------------------------
    # (a) Mean predictor
    _log("    [baseline] mean predictor", log)
    yt_m, yp_m, _, _ = _spatial_cv_preds(
        df, target, lambda: _MeanRegressor(), log=log
    )
    res["baseline_mean"] = {"R2": float(r2_score(yt_m, yp_m)),
                            "MAE": float(mean_absolute_error(yt_m, yp_m)),
                            "RMSE": _rmse(yt_m, yp_m)}

    # (b) Ridge regression
    _log("    [baseline] Ridge linear regression", log)
    yt_r, yp_r, _, _ = _spatial_cv_preds(
        df, target, lambda: _RidgeRegressor(), log=log
    )
    res["baseline_ridge"] = {"R2": float(r2_score(yt_r, yp_r)),
                             "MAE": float(mean_absolute_error(yt_r, yp_r)),
                             "RMSE": _rmse(yt_r, yp_r)}

    # Per-state breakdown
    per_state_rows = []
    for state, mask in pd.Series(st).groupby(st).groups.items():
        idx = np.array(list(mask))
        if len(idx) < 5:
            continue
        per_state_rows.append({
            "state_fips": state, "n": int(len(idx)),
            "R2": float(r2_score(yt[idx], yp[idx])),
            "MAE": float(mean_absolute_error(yt[idx], yp[idx])),
        })
    res["per_state"] = sorted(per_state_rows, key=lambda x: x["R2"])

    # Stability: bootstrap 5 seeds to get SD of predictions -----------------
    _log("    [stability] 5-seed bootstrap", log)
    seeds = [C.RANDOM_STATE + i for i in range(5)]
    preds_stack = []
    d = df.dropna(subset=[target, "state_fips"]).copy()
    X = feature_matrix(d).astype("float64").fillna(0.0).values
    y = d[target].astype(float).values
    groups = d["state_fips"].values
    for sd in seeds:
        rng = np.random.default_rng(sd)
        idx = rng.integers(0, len(y), len(y))
        mdl = XGBRegressor(
            n_estimators=600, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
            reg_lambda=1.0, random_state=int(sd), tree_method="hist",
            n_jobs=-1, verbosity=0,
        )
        mdl.fit(X[idx], y[idx])
        preds_stack.append(mdl.predict(X))
    preds_arr = np.vstack(preds_stack)
    res["stability"] = {
        "mean_pred_sd": float(preds_arr.std(axis=0).mean()),
        "median_pred_sd": float(np.median(preds_arr.std(axis=0))),
    }

    # store raw predictions for plotting
    res["_preds"] = {"y_true": yt.tolist(), "y_pred": yp.tolist(),
                     "state": st.tolist()}
    return res


class _MeanRegressor:
    def fit(self, X, y): self.mean_ = float(np.mean(y)); return self
    def predict(self, X): return np.full(len(X), self.mean_)


class _RidgeRegressor:
    def fit(self, X, y):
        self.mdl_ = Ridge(alpha=1.0, random_state=C.RANDOM_STATE)
        self.mdl_.fit(np.nan_to_num(X), y)
        return self
    def predict(self, X):
        return self.mdl_.predict(np.nan_to_num(X))


# ---------------------------- binary evaluation -----------------------------

def evaluate_binary(df: pd.DataFrame, target: str,
                    log: Optional[Callable[[str], None]] = None) -> dict:
    """Cast the continuous rate into a binary "high-burden" label at the
    top quartile and train an XGBClassifier on the same features."""
    _log(f"  [binary] target={target} (top-quartile classifier)", log)
    d = df.dropna(subset=[target, "state_fips"]).copy()
    X = feature_matrix(d).astype("float64").fillna(0.0).values
    y_cont = d[target].astype(float).values
    thr = float(np.quantile(y_cont, 0.75))
    y = (y_cont >= thr).astype(int)
    groups = d["state_fips"].values

    n_splits = min(C.N_SPLITS_STATE_CV, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    y_true_all, y_prob_all = np.empty(0, dtype=int), np.empty(0)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        _log(f"    [bin fold {fold+1}/{n_splits}] pos={y[tr].sum()}/{len(tr)}", log)
        mdl = _xgb_clf()
        mdl.fit(X[tr], y[tr])
        prob = mdl.predict_proba(X[te])[:, 1]
        y_true_all = np.concatenate([y_true_all, y[te]])
        y_prob_all = np.concatenate([y_prob_all, prob])

    y_pred = (y_prob_all >= 0.5).astype(int)
    auroc = float(roc_auc_score(y_true_all, y_prob_all))
    auprc = float(average_precision_score(y_true_all, y_prob_all))
    brier = float(brier_score_loss(y_true_all, y_prob_all))
    ece = _class_ece(y_true_all, y_prob_all)
    f1 = float(f1_score(y_true_all, y_pred))
    prec = float(precision_score(y_true_all, y_pred, zero_division=0))
    rec = float(recall_score(y_true_all, y_pred))
    acc = float(accuracy_score(y_true_all, y_pred))
    cm = confusion_matrix(y_true_all, y_pred).tolist()

    # bootstrap CI for AUROC
    aucmean, auclo, auchi = _bootstrap_ci(y_true_all, y_prob_all, roc_auc_score)

    return {
        "target": target,
        "threshold_value": thr,
        "n": int(len(y_true_all)),
        "n_positive": int(y_true_all.sum()),
        "AUROC": auroc,
        "AUROC_boot_mean": aucmean,
        "AUROC_CI95": [auclo, auchi],
        "AUPRC": auprc,
        "Brier": brier,
        "ECE": ece,
        "F1": f1,
        "precision": prec,
        "recall": rec,
        "accuracy": acc,
        "confusion_matrix": cm,
        "_y_true": y_true_all.tolist(),
        "_y_prob": y_prob_all.tolist(),
    }


# ------------------------------- plotting -----------------------------------

def _plot_calibration_regression(res_by_target: dict, targets: list[str]) -> None:
    fig, axes = plt.subplots(1, len(targets), figsize=(5.2 * len(targets), 5),
                             squeeze=False)
    for ax, tgt in zip(axes[0], targets):
        r = res_by_target[tgt]
        yt = np.array(r["_preds"]["y_true"])
        yp = np.array(r["_preds"]["y_pred"])
        bins = np.quantile(yp, np.linspace(0, 1, 11))
        bins[-1] += 1e-6
        idx = np.digitize(yp, bins) - 1
        idx = np.clip(idx, 0, 9)
        bin_pred = [yp[idx == b].mean() for b in range(10) if (idx == b).any()]
        bin_obs = [yt[idx == b].mean() for b in range(10) if (idx == b).any()]
        ax.plot(bin_pred, bin_obs, marker="o", linewidth=2, color="#1f77b4")
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.6, label="ideal")
        ax.set_xlabel("Predicted (bin mean)")
        ax.set_ylabel("Observed (bin mean)")
        ax.set_title(f"{tgt}\nR2={r['xgb']['R2']:.3f}  ECE={r['xgb']['ECE']:.2f}")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Regression calibration (XGBoost, leave-state-out CV)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = C.FIG_DIR / "fig6_calibration_regression.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out.name}")


def _plot_per_state(res_by_target: dict, target: str) -> None:
    rows = res_by_target[target]["per_state"]
    if not rows:
        return
    df = pd.DataFrame(rows)
    df = df.sort_values("R2")
    fig, ax = plt.subplots(figsize=(10, max(6, 0.3 * len(df))))
    colors = ["#d62728" if r < 0 else "#2ca02c" for r in df["R2"]]
    ax.barh(df["state_fips"], df["R2"], color=colors, alpha=0.8)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Held-out R^2 (leave-this-state-out CV)")
    ax.set_title(f"Per-state XGBoost performance -> {target}\n"
                 f"(green = model generalizes; red = state is an outlier)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = C.FIG_DIR / "fig7_per_state_performance.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out.name}")
    df.to_csv(C.TAB_DIR / "per_state_metrics.csv", index=False)


def _plot_binary(binary: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    y_true = np.array(binary["_y_true"])
    y_prob = np.array(binary["_y_prob"])

    # ROC
    from sklearn.metrics import roc_curve, precision_recall_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[0].plot(fpr, tpr, linewidth=2, color="#1f77b4")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title(f"ROC  AUC={binary['AUROC']:.3f}\n"
                      f"95% CI [{binary['AUROC_CI95'][0]:.3f}, {binary['AUROC_CI95'][1]:.3f}]")
    axes[0].grid(alpha=0.3)

    # PR
    p, r, _ = precision_recall_curve(y_true, y_prob)
    axes[1].plot(r, p, linewidth=2, color="#ff7f0e")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall  AUPRC={binary['AUPRC']:.3f}\n"
                      f"F1={binary['F1']:.3f}  Brier={binary['Brier']:.3f}")
    axes[1].grid(alpha=0.3)

    # Calibration
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    axes[2].plot(prob_pred, prob_true, marker="o", linewidth=2, color="#2ca02c")
    axes[2].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[2].set_xlabel("Predicted probability")
    axes[2].set_ylabel("Observed fraction positive")
    axes[2].set_title(f"Calibration  ECE={binary['ECE']:.3f}")
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        f"Binary classifier: top-quartile {binary['target']}  "
        f"(positives: {binary['n_positive']}/{binary['n']})",
        fontweight="bold"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = C.FIG_DIR / "fig8_binary_classifier.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out.name}")


# --------------------------- top-level orchestration -----------------------

PRIMARY_TARGETS = [
    "puerperal_rate_per1000",
    "hdp_rate_per1000",
    "smm_rate_per1000",
]


def run(log: Optional[Callable[[str], None]] = None) -> dict:
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    _log("[evaluate] loading analytic cohort...", log)
    df = build_analytic()
    _log(f"[evaluate] n={len(df)} counties, {df['state_fips'].nunique()} states", log)

    res_by_target = {}
    for tgt in PRIMARY_TARGETS:
        res_by_target[tgt] = evaluate_regression(df, tgt, log=log)

    _log("[evaluate] binary classification for puerperal_rate_per1000", log)
    binary = evaluate_binary(df, "puerperal_rate_per1000", log=log)

    # Plots
    _plot_calibration_regression(res_by_target, PRIMARY_TARGETS)
    _plot_per_state(res_by_target, "puerperal_rate_per1000")
    _plot_binary(binary)

    # Write JSON (strip internal prediction arrays)
    clean = {k: {**v, "_preds": None} for k, v in res_by_target.items()}
    clean_bin = {k: v for k, v in binary.items() if not k.startswith("_")}
    out = {"regression": clean, "binary_top_quartile_puerperal": clean_bin}
    (C.TAB_DIR / "full_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    _log("[evaluate] wrote outputs/tables/full_metrics.json", log)

    # Compact console summary
    print("\n======== REGRESSION SUMMARY ========")
    for tgt, r in res_by_target.items():
        xg = r["xgb"]
        print(f"{tgt}")
        print(f"  XGBoost:   R2={xg['R2']:.3f}  MAE={xg['MAE']:.2f}  RMSE={xg['RMSE']:.2f}  "
              f"Pearson={xg['Pearson_r']:.3f}  Spearman={xg['Spearman_r']:.3f}")
        print(f"    R2 bootstrap 95% CI: [{xg['R2_boot_CI95'][0]:.3f}, {xg['R2_boot_CI95'][1]:.3f}]")
        print(f"    calibration slope={xg['calibration_slope']:.3f}  intercept={xg['calibration_intercept']:.2f}  ECE={xg['ECE']:.2f}")
        print(f"    stability (mean pred SD across 5 seeds): {r['stability']['mean_pred_sd']:.2f}")
        print(f"  Ridge:     R2={r['baseline_ridge']['R2']:.3f}  MAE={r['baseline_ridge']['MAE']:.2f}")
        print(f"  Mean base: R2={r['baseline_mean']['R2']:.3f}  MAE={r['baseline_mean']['MAE']:.2f}")
    print("\n======== BINARY (top-quartile puerperal) ========")
    b = binary
    print(f"  AUROC={b['AUROC']:.3f} [{b['AUROC_CI95'][0]:.3f}, {b['AUROC_CI95'][1]:.3f}]  "
          f"AUPRC={b['AUPRC']:.3f}  F1={b['F1']:.3f}  Brier={b['Brier']:.3f}  ECE={b['ECE']:.3f}")
    return out


if __name__ == "__main__":
    run()
