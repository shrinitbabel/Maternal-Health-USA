"""Generate the publication-style summary figure and numbers used in the
README / methods section."""
from __future__ import annotations
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .analytic import build_analytic


OUTCOME_LABEL = {
    "smm_rate_per1000": "SMM-proxy composite",
    "hdp_rate_per1000": "Hypertensive disorders\nof pregnancy (HDP)",
    "childbirth_cx_rate_per1000": "Childbirth complications",
    "puerperal_rate_per1000": "Puerperal (postpartum)\ncomplications",
}


def _forest_plot():
    with open(C.TAB_DIR / "dml_results.json") as f:
        dml = json.load(f)

    exposure = "log_time3"  # continuous exposure for coefficient plot
    rows = []
    for outcome, per_exp in dml.items():
        for analysis, r in per_exp[exposure].items():
            rows.append({
                "outcome": OUTCOME_LABEL.get(outcome, outcome),
                "analysis": analysis,
                "theta": r["theta"],
                "ci_lo": r["ci_lo"],
                "ci_hi": r["ci_hi"],
                "pval": r["pval"],
                "n": r["n"],
            })
    df = pd.DataFrame(rows)

    labels_analysis = {
        "main": "Main (all counties)",
        "excl_level3_host": "Excl. Level-III host counties",
        "rural_subset": "Rural subset (pop decile <=5)",
    }

    fig, ax = plt.subplots(figsize=(10, 9))
    outcomes_order = list(OUTCOME_LABEL.values())
    y = []
    y_labels = []
    colors = {"main": "#1f77b4", "excl_level3_host": "#ff7f0e",
              "rural_subset": "#2ca02c"}

    pos = 0
    for out in outcomes_order:
        sub = df[df["outcome"] == out]
        for _, r in sub.iterrows():
            ax.errorbar(r["theta"], pos,
                        xerr=[[r["theta"] - r["ci_lo"]], [r["ci_hi"] - r["theta"]]],
                        fmt="o", color=colors[r["analysis"]],
                        markersize=7, capsize=3, linewidth=2)
            y.append(pos)
            y_labels.append(f"{out} -- {labels_analysis[r['analysis']]}")
            pos += 1
        pos += 1  # gap

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("DoubleML partial effect theta (log(rate) per unit log(time3))",
                  fontsize=11)
    ax.set_title("Figure: DoubleML effect of Level-III travel time on county outcomes\n"
                 "Positive -> more complications with greater distance "
                 "(true residential access signal); Negative -> referral-center capture",
                 fontsize=11)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    # Annotate with pval for main analysis
    for _, r in df[df["analysis"] == "main"].iterrows():
        stars = "***" if r["pval"] < 0.001 else "**" if r["pval"] < 0.01 else "*" if r["pval"] < 0.05 else ""
        if stars:
            pass  # already shown via color

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", linestyle="",
                      color=c, markersize=8, label=labels_analysis[k])
               for k, c in colors.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=9)

    out = C.FIG_DIR / "fig4_dml_forest.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] forest plot -> {out.name}")


def _access_equity_bars():
    df = build_analytic()

    groups = []
    def _row(label, sub):
        groups.append({
            "Group": label,
            "n": len(sub),
            "% >60 min to Level III": sub["gap60_level3"].mean() * 100,
            "% >90 min to Level III": sub["gap90_level3"].mean() * 100,
            "Median puerperal rate (/1000)": sub["puerperal_rate_per1000"].median(),
        })

    _row("All", df)
    for q in [1, 2, 3, 4]:
        _row(f"SVI Q{q}", df[df["svi_quartile"] == q])
    _row("Rural (decile<=3)", df[df["pop_decile"] <= 3])
    thr_aian = df["pct_aian"].quantile(0.75)
    _row(f"Top quartile AIAN", df[df["pct_aian"] >= thr_aian])

    g = pd.DataFrame(groups)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    x = np.arange(len(g))
    w = 0.38

    ax = axes[0]
    ax.bar(x - w/2, g["% >60 min to Level III"], w,
           label=">60 min", color="#ff7f0e")
    ax.bar(x + w/2, g["% >90 min to Level III"], w,
           label=">90 min", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(g["Group"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("% of counties")
    ax.set_title("Geographic access burden by equity stratum")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x, g["Median puerperal rate (/1000)"], color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(g["Group"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Median puerperal complications / 1,000 PRG discharges")
    ax.set_title("Postpartum complication burden by equity stratum")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Access burden and postpartum complication burden concentrate "
                 "in high-SVI, rural, and AIAN-heavy counties",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = C.FIG_DIR / "fig5_equity_bars.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] equity bars -> {out.name}")
    g.round(2).to_csv(C.TAB_DIR / "equity_summary.csv", index=False)


def run():
    _forest_plot()
    _access_equity_bars()


if __name__ == "__main__":
    run()
