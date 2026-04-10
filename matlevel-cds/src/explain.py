"""SHAP-based explainability and fairness stratification.

Generates:
  outputs/figures/fig2_shap_summary.png      - global feature importance
  outputs/figures/fig3_shap_dependence.png   - travel-time x SVI interaction
  outputs/tables/fairness_stratified.csv     - equity audit by SVI quartile,
                                              rurality, and race composition
"""
from __future__ import annotations
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from . import config as C
from .analytic import build_analytic
from .model import feature_matrix


def _load_model(target: str):
    with open(C.MODEL_DIR / "xgb_models.pkl", "rb") as f:
        bundle = pickle.load(f)
    if target not in bundle["models"]:
        raise KeyError(f"{target} not in saved models")
    return bundle["models"][target], bundle["feature_cols"]


def shap_summary(target: str = "puerperal_rate_per1000") -> None:
    model, feat_cols = _load_model(target)
    df = build_analytic()
    d = df.dropna(subset=[target]).copy()
    X = feature_matrix(d)[feat_cols]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X, show=False, max_display=15, plot_size=None)
    plt.title(f"SHAP feature importance -> {target}")
    plt.tight_layout()
    out = C.FIG_DIR / f"fig2_shap_summary_{target}.png"
    plt.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] SHAP summary -> {out.name}")

    # Dependence plot: travel time to Level III, colored by SVI
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.dependence_plot(
        "time3_imputed", shap_values, X,
        interaction_index="rpl_themes" if "rpl_themes" in X.columns else "rpl_theme1",
        ax=ax, show=False,
    )
    ax.set_title(f"Effect of Level-III travel time on {target}\n"
                 f"(colored by SVI)")
    plt.tight_layout()
    out2 = C.FIG_DIR / f"fig3_shap_dependence_{target}.png"
    plt.savefig(out2, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] SHAP dependence -> {out2.name}")


def fairness_stratified_table(target: str = "puerperal_rate_per1000") -> pd.DataFrame:
    df = build_analytic().dropna(subset=[target]).copy()

    # Race strata: split by tertile of % Black, % AIAN, and % Minority
    df["_high_black"] = df["pct_black"] >= df["pct_black"].quantile(0.75)
    df["_high_aian"]  = df["pct_aian"]  >= df["pct_aian"].quantile(0.75)
    df["_high_min"]   = df["pct_minority"] >= df["pct_minority"].quantile(0.75)
    df["_rural"]      = df["pop_decile"] <= 3

    rows = []

    def _agg(sub: pd.DataFrame, label: str) -> dict:
        return {
            "group": label,
            "n_counties": len(sub),
            "median_outcome": sub[target].median(),
            "mean_outcome": sub[target].mean(),
            "pct_gap60": (sub["gap60_level3"].mean() * 100),
            "pct_gap90": (sub["gap90_level3"].mean() * 100),
            "median_time3_min": sub["time3_imputed"].median(),
            "median_svi": sub["rpl_themes"].median(),
        }

    rows.append(_agg(df, "Overall"))
    for q in [1, 2, 3, 4]:
        rows.append(_agg(df[df["svi_quartile"] == q], f"SVI Q{q}"))
    rows.append(_agg(df[df["_rural"]], "Rural (pop decile <=3)"))
    rows.append(_agg(df[~df["_rural"]], "Non-rural (pop decile >3)"))
    rows.append(_agg(df[df["_high_black"]], "Top-quartile % Black"))
    rows.append(_agg(df[df["_high_aian"]],  "Top-quartile % AIAN"))
    rows.append(_agg(df[df["_high_min"]],   "Top-quartile % Minority"))

    out = pd.DataFrame(rows).round(2)
    out.to_csv(C.TAB_DIR / f"fairness_stratified_{target}.csv", index=False)
    print(f"[ok] fairness table -> fairness_stratified_{target}.csv")
    return out


def run() -> None:
    for tgt in ["puerperal_rate_per1000", "hdp_rate_per1000", "smm_rate_per1000"]:
        try:
            shap_summary(tgt)
        except Exception as e:
            print(f"[warn] shap_summary {tgt}: {e}")
        try:
            fairness_stratified_table(tgt)
        except Exception as e:
            print(f"[warn] fairness {tgt}: {e}")


if __name__ == "__main__":
    run()
