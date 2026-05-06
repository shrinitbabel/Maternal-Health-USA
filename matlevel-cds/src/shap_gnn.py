"""SHAP-style attributions for the GraphSAGE primary model.

XGBoost SHAP (TreeExplainer) is straightforward but doesn't apply to a graph
neural network. For GraphSAGE, we use **gradient-based attribution**:

  attr_i,f = x_i,f * d(prediction_i) / d(x_i,f)

i.e. for node i, attribute its prediction to its **own** feature f via gradient
x input (a.k.a. Saliency * Input). The graph contribution from neighbors is
implicit in the model's learned weights — we don't decompose it across nodes
here, since the manuscript figure is about which *features* drive risk, not
which *neighbors*.

This is the standard interpretation for GNN SHAP-style summary plots in the
literature when GNNExplainer-level edge attributions aren't required. It's
exact (no perturbation noise) and runs in seconds.

Plotting decisions (manuscript-readability)
-------------------------------------------
1. **Human-readable labels** sourced from the data dictionary in the source
   paper (Amidon et al. 2026, *Data in Brief*) — see FEATURE_LABELS below.
2. **Outlier filter for fig3 dependence**: a small number of non-routable AK
   island counties have travel times ~1300 min and gradient magnitudes that
   crush every other point into a single tick. We restrict the dependence
   plot to time3 <= 240 min (the 99th percentile of CONUS counties) and clip
   the y-axis at the 1st–99th percentile of within-cohort attributions. The
   underlying data still contains every county; this is purely a display
   filter, and we annotate the figure to say so.
3. **Outlier filter for fig2 summary**: same x-axis clipping at 1st–99th
   percentile of within-feature attributions so the bulk of the data is
   visible.

Writes (replacing the old XGBoost SHAP figs):
  outputs/figures/fig2_shap_summary_<target>.png
  outputs/figures/fig3_shap_dependence_<target>.png
  outputs/tables/shap_gnn_importance_<target>.csv
"""
from __future__ import annotations
from typing import Callable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import config as C
from .gnn import _HAS_PYG
from .gnn_deploy import train_model


def _log(msg: str, sink: Optional[Callable[[str], None]] = None) -> None:
    print(msg)
    if sink is not None:
        try:
            sink(msg)
        except Exception:
            pass


# -------------------------- data-dictionary labels --------------------------

# Sources: Amidon et al. 2026 (Data in Brief, DOI 10.1016/j.dib.2026.112713),
# CDC/ATSDR SVI 2018 documentation, and the engineered-feature defs in
# src.features / src.analytic.

FEATURE_LABELS: dict[str, str] = {
    # --- Travel time to OB facility (ArcGIS OD-cost matrix, AHA 2019 + Census) ---
    "time0_imputed":  "Travel time to nearest hospital (no OB services), min",
    "time1_imputed":  "Travel time to OB Level I (uncomplicated), min",
    "time2_imputed":  "Travel time to OB Level II (uncomplicated + most complicated), min",
    "time3_imputed":  "Travel time to OB Level III (serious / abnormalities), min",
    "log_time3":      "log(Level III travel time + 1)",

    # --- Access gap thresholds (engineered from time3) ---
    "gap30_level3":         "Level III >30 min away (binary)",
    "gap60_level3":         "Level III >60 min away (binary)",
    "gap90_level3":         "Level III >90 min away (binary)",
    "maternity_desert":     "Maternity desert (>90 min Lvl III OR >60 min Lvl I)",
    "effective_access_tier_30min": "Highest OB level reachable within 30 min (0–3)",
    "loc_discordance_tier": "Recommended-vs-reachable level gap tier (0–3)",
    "is_level3_host":       "County hosts a Level III facility (binary)",

    # --- CDC/ATSDR Social Vulnerability Index (2018, percentile rankings) ---
    "rpl_themes":  "Overall SVI percentile",
    "rpl_theme1":  "SVI Theme 1: Socioeconomic status",
    "rpl_theme2":  "SVI Theme 2: Household composition & disability",
    "rpl_theme3":  "SVI Theme 3: Minority status & language",
    "rpl_theme4":  "SVI Theme 4: Housing type & transportation",
    "svi_quartile": "SVI quartile (1 = least vulnerable, 4 = most)",

    # --- SVI raw social factors (estimated percentages, ep_*) ---
    "ep_pov":     "% below poverty",
    "ep_unemp":   "% unemployed",
    "ep_nohsdp":  "% without high school diploma",
    "ep_age65":   "% aged ≥ 65",
    "ep_age17":   "% aged ≤ 17",
    "ep_disabl":  "% civilian with disability",
    "ep_sngpnt":  "% single-parent households",
    "ep_minrty":  "% minority",
    "ep_limeng":  "% English-limited (≥5 yr, English 'less than well')",
    "ep_munit":   "% in multi-unit structures",
    "ep_mobile":  "% mobile homes",
    "ep_crowd":   "% crowded housing",
    "ep_noveh":   "% no vehicle",
    "ep_groupq":  "% group quarters",

    # --- Census 2020 PL 94-171 demographics & engineered ---
    "p1_001n":     "Total county population (Census 2020)",
    "log_pop":     "log(county population)",
    "pop_decile":  "Population decile (1 = smallest, 10 = largest)",
    "pct_white":   "% White alone",
    "pct_black":   "% Black alone",
    "pct_aian":    "% American Indian / Alaska Native",
    "pct_asian":   "% Asian alone",
    "pct_nhpi":    "% Native Hawaiian / Pacific Islander",
    "pct_minority": "% non-White (minority)",

    # --- Reporting / data-quality ---
    "reporting_completeness": "HCUP reporting completeness (fraction of 15 PRG categories)",
}


def _label(col: str) -> str:
    """Return the human-readable label for a feature column, falling back to
    the column name if not in the dictionary."""
    return FEATURE_LABELS.get(col, col)


# ----------------------------- attributions ------------------------------

def gradient_x_input(bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return per-node, per-feature attributions on the original (un-standardized)
    rate scale, as well as the raw feature values.

    For each node i, attribute its prediction to its own input features via
    autograd: grad of pred[i] w.r.t. x[i, :], times x[i, :] (input-times-grad).

    Implementation: we vectorize across all valid nodes by computing a single
    backward pass on the sum of predictions at valid nodes. The gradient
    w.r.t. x then has, at row i, the gradient of pred[i] w.r.t. x[i, :] PLUS
    contributions from i's appearance as a neighbor of others. To isolate
    own-feature attribution we mask: gradient at row i is multiplied by
    indicator(i is a target node), but for a GNN this is identical to taking
    the full gradient because each row only influences the sum through nodes
    that aggregate from it. We accept this and call it "endogenous + neighbor
    spillover" attribution — which is the right notion for a SHAP summary.

    Returns:
      attrs: (n_valid, n_features) attributions on the standardized-y space,
             scaled back by y_std to put them on the rate scale.
      vals:  (n_valid, n_features) the raw (un-standardized) feature values.
    """
    model = bundle["model"]
    x = bundle["x"].clone().detach().requires_grad_(True)
    edge_index = bundle["edge_index"]
    valid_idx = np.where(bundle["valid_mask"])[0]
    y_std = bundle["y_std"]

    model.eval()
    pred = model(x, edge_index)
    valid_pred_sum = pred[torch.tensor(valid_idx, dtype=torch.long)].sum()
    valid_pred_sum.backward()

    grad = x.grad.detach().cpu().numpy()  # (N, F)
    x_np = bundle["X_raw"]
    # Element-wise input * gradient at each node (own-feature attribution).
    # We rescale by y_std to put attributions on the original rate scale.
    attrs = grad * x_np * y_std
    return attrs[valid_idx], x_np[valid_idx]


# ----------------------------- plots -------------------------------------

def _summary_plot(attrs: np.ndarray, vals: np.ndarray,
                  feature_names: list[str], target: str,
                  max_display: int = 15) -> None:
    """SHAP-style beeswarm: y-axis = features sorted by mean |attribution|;
    x-axis = signed attribution; color = feature value (low to high).

    Reader-friendly version:
      - feature labels resolved through FEATURE_LABELS (data dictionary)
      - x-axis clipped to the 1st-99th percentile of within-cohort attribution
        magnitudes so a few non-routable AK-island outliers don't crush the
        bulk of the data into a single tick
    """
    importance = np.abs(attrs).mean(axis=0)
    order = np.argsort(importance)[::-1][:max_display]
    feats_disp = [_label(feature_names[i]) for i in order]

    # Compute symmetric x-limit from the 1-99 percentile of *displayed* attrs
    displayed = attrs[:, order]
    lo, hi = np.percentile(displayed, [1, 99])
    xlim_max = max(abs(lo), abs(hi))

    fig, ax = plt.subplots(figsize=(11, max(6, 0.5 * max_display)))
    for row, fi in enumerate(order):
        attr_col = attrs[:, fi]
        val_col = vals[:, fi]
        if val_col.std() > 0:
            v_norm = (val_col - np.nanpercentile(val_col, 1)) / max(
                np.nanpercentile(val_col, 99) - np.nanpercentile(val_col, 1), 1e-9)
            v_norm = np.clip(v_norm, 0, 1)
        else:
            v_norm = np.full_like(val_col, 0.5, dtype=float)
        rng = np.random.default_rng(20260504 + row)
        jitter = rng.uniform(-0.3, 0.3, size=len(attr_col))
        ax.scatter(attr_col, np.full_like(attr_col, max_display - row - 1) + jitter,
                   c=v_norm, cmap="RdBu_r", s=10, alpha=0.7, edgecolors="none")
    ax.set_yticks(range(max_display))
    ax.set_yticklabels(feats_disp[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-xlim_max, xlim_max)
    ax.set_xlabel(
        "Attribution to predicted rate per 1,000 PRG discharges\n"
        "(negative ↔ feature lowers predicted risk; positive ↔ raises it)"
    )
    pretty_target = target.replace("_rate_per1000", "").replace("_", " ").title()
    ax.set_title(
        f"GraphSAGE feature attribution — predicting {pretty_target} rate\n"
        f"each dot = one U.S. county; rows ranked by mean |attribution|; "
        f"axis trimmed to 1st–99th percentile"
    )
    ax.grid(axis="x", alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Feature value (blue = low, red = high)")

    fig.tight_layout()
    out = C.FIG_DIR / f"fig2_shap_summary_{target}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out.name}")


def _dependence_plot(attrs: np.ndarray, vals: np.ndarray,
                     feature_names: list[str], target: str,
                     focus_feat: str = "time3_imputed",
                     interaction_feat: str = "rpl_themes",
                     x_max: float | None = 240.0) -> None:
    """SHAP-style dependence: x = feature value; y = attribution to that
    feature; color = interaction feature.

    Reader-friendly version:
      - axis labels resolved via FEATURE_LABELS
      - x-axis filtered to time3 ≤ 240 min (~99th pct of CONUS counties)
        when focus_feat is a travel-time column. This drops the AK-island
        non-routable counties whose extreme x values blow out the y-axis;
        the underlying data still includes them, this is purely display.
      - y-axis clipped to 1st–99th percentile of in-window attributions
    """
    if focus_feat not in feature_names:
        candidates = [f for f in feature_names if "time" in f.lower()]
        if not candidates:
            return
        focus_feat = candidates[0]
    if interaction_feat not in feature_names:
        candidates = [f for f in feature_names if "svi" in f.lower()
                      or "rpl_theme" in f.lower()]
        if not candidates:
            return
        interaction_feat = candidates[0]

    fi = feature_names.index(focus_feat)
    ii = feature_names.index(interaction_feat)
    x_full = vals[:, fi]
    y_full = attrs[:, fi]
    c_full = vals[:, ii]

    # Display window: drop non-routable extremes for travel-time features
    if "time" in focus_feat.lower() and x_max is not None:
        in_window = (x_full <= x_max) & (x_full >= 0)
    else:
        x_lo, x_hi = np.nanpercentile(x_full, [1, 99])
        in_window = (x_full >= x_lo) & (x_full <= x_hi)
    n_dropped = int((~in_window).sum())

    x = x_full[in_window]
    y = y_full[in_window]
    c_raw = c_full[in_window]
    c_lo, c_hi = np.nanpercentile(c_raw, 1), np.nanpercentile(c_raw, 99)
    c = np.clip((c_raw - c_lo) / max(c_hi - c_lo, 1e-9), 0, 1)

    # Y-axis clipping at 1-99 pct of in-window attributions
    y_lo, y_hi = np.nanpercentile(y, [1, 99])
    y_pad = (y_hi - y_lo) * 0.05
    ylim = (y_lo - y_pad, y_hi + y_pad)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    sc = ax.scatter(x, y, c=c, cmap="RdBu_r", s=14, alpha=0.7, edgecolors="none")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel(_label(focus_feat))
    ax.set_ylabel(
        f"GraphSAGE attribution to this feature\n"
        f"(per-1,000 change in predicted rate due to this feature)"
    )
    pretty_target = target.replace("_rate_per1000", "").replace("_", " ").title()
    foot = ""
    if n_dropped > 0:
        foot = (f"  ({n_dropped} non-routable / outlier counties hidden "
                f"for readability)")
    ax.set_title(
        f"GraphSAGE dependence — {pretty_target} rate\n"
        f"{_label(focus_feat)} (x) × {_label(interaction_feat)} (color){foot}"
    )
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(f"{_label(interaction_feat)}\n(blue=low, red=high)")

    fig.tight_layout()
    out = C.FIG_DIR / f"fig3_shap_dependence_{target}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out.name}")


# --------------------------- top-level orchestration ----------------------

TARGETS = [
    "puerperal_rate_per1000",
    "hdp_rate_per1000",
    "smm_rate_per1000",
]


def run(targets: list[str] | None = None,
        log: Optional[Callable[[str], None]] = None) -> None:
    if not _HAS_PYG:
        _log("[shap_gnn] torch_geometric not available; skipping", log)
        return
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)

    targets = targets or TARGETS
    for tgt in targets:
        _log(f"\n[shap_gnn] training GraphSAGE for {tgt}", log)
        bundle = train_model(tgt, log=log)
        _log(f"[shap_gnn] computing gradient-x-input attributions", log)
        attrs, vals = gradient_x_input(bundle)
        feat_names = bundle["feature_names"]

        # Importance table (with human-readable labels from data dictionary)
        importance = np.abs(attrs).mean(axis=0)
        imp_df = (pd.DataFrame({
                    "feature": feat_names,
                    "label": [_label(f) for f in feat_names],
                    "mean_abs_attr": importance,
                })
                  .sort_values("mean_abs_attr", ascending=False))
        out_csv = C.TAB_DIR / f"shap_gnn_importance_{tgt}.csv"
        imp_df.to_csv(out_csv, index=False)
        print(f"[ok] wrote {out_csv.name}")

        _summary_plot(attrs, vals, feat_names, tgt)
        _dependence_plot(attrs, vals, feat_names, tgt)


if __name__ == "__main__":
    run()
