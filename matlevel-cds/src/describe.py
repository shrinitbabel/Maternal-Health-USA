"""Descriptive statistics + disparity tables for the MatLevel-CDS paper.

Generates all Table 1 / headline figures. Writes to outputs/tables and
outputs/figures.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .analytic import build_analytic, TARGET_PRIMARY


def _ensure_dirs():
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)


def table1(df: pd.DataFrame) -> pd.DataFrame:
    """Classic Table 1 — cohort summary."""
    def _fmt(s: pd.Series, pct: bool = False) -> str:
        if pct:
            return f"{s.mean():.1f}%"
        return f"{s.median():.1f} ({s.quantile(0.25):.1f}-{s.quantile(0.75):.1f})"

    rows = []
    rows.append(("Counties in cohort", f"{len(df):,}"))
    rows.append(("States represented", f"{df['state_fips'].nunique()}"))
    rows.append(("Total population (M)",
                 f"{df[C.POP_TOTAL].sum()/1e6:.1f}"))
    rows.append(("", ""))
    rows.append(("-- Travel time to OB care, median (IQR), minutes --", ""))
    rows.append(("  Level 0 (no maternal services)",
                 _fmt(df["time0_imputed"])))
    rows.append(("  Level I (basic OB)",   _fmt(df["time1_imputed"])))
    rows.append(("  Level II (complicated OB)", _fmt(df["time2_imputed"])))
    rows.append(("  Level III (serious illness/abnormalities)",
                 _fmt(df["time3_imputed"])))
    rows.append(("", ""))
    rows.append(("-- ACOG Level-III discordance --", ""))
    rows.append(("  Counties >30 min from Level III",
                 _fmt(df["gap30_level3"] * 100, pct=True)))
    rows.append(("  Counties >60 min from Level III",
                 _fmt(df["gap60_level3"] * 100, pct=True)))
    rows.append(("  Counties >90 min from Level III",
                 _fmt(df["gap90_level3"] * 100, pct=True)))
    rows.append(("  Maternity desert (composite)",
                 _fmt(df["maternity_desert"] * 100, pct=True)))
    rows.append(("", ""))
    rows.append(("-- SVI quartiles (% of counties) --", ""))
    for q in [1, 2, 3, 4]:
        rows.append((f"  Q{q}", f"{(df['svi_quartile']==q).mean()*100:.1f}%"))
    rows.append(("", ""))
    rows.append(("-- Demographics, median (IQR) % --", ""))
    rows.append(("  % Minority (non-white non-Hispanic, SVI)",
                 _fmt(df["pct_minority"])))
    rows.append(("  % Black",  _fmt(df["pct_black"])))
    rows.append(("  % AIAN",   _fmt(df["pct_aian"])))
    rows.append(("  % Asian",  _fmt(df["pct_asian"])))
    rows.append(("  % in poverty (SVI)", _fmt(df["ep_pov"])))
    rows.append(("  % no HS diploma (SVI)", _fmt(df["ep_nohsdp"])))
    rows.append(("  % no vehicle (SVI)", _fmt(df["ep_noveh"])))
    rows.append(("", ""))
    rows.append(("-- Outcomes, median (IQR) per 1,000 PRG discharges --", ""))
    rows.append(("  SMM-proxy composite",
                 _fmt(df["smm_rate_per1000"])))
    rows.append(("  Hypertensive disorders of pregnancy",
                 _fmt(df["hdp_rate_per1000"])))
    rows.append(("  Childbirth complications",
                 _fmt(df["childbirth_cx_rate_per1000"])))
    return pd.DataFrame(rows, columns=["Characteristic", "Value"])


def stratified_table(df: pd.DataFrame, group_col: str, outcome: str) -> pd.DataFrame:
    g = df.groupby(group_col, dropna=True)[outcome]
    out = g.agg(n="count", mean="mean", median="median", sd="std",
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75)).round(2)
    return out.reset_index()


def figure_headline(df: pd.DataFrame) -> None:
    """Four-panel figure for the paper."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("Maternal Care Access and Complication Burden Across U.S. Counties "
                 "(HCUP 2017-2019, 37 states, n={})".format(len(df)),
                 fontsize=12, fontweight="bold")

    # (A) Distribution of travel time to Level III.
    ax = axes[0, 0]
    t3 = df["time3_imputed"].clip(upper=300)
    ax.hist(t3.dropna(), bins=50, color="#5b8def", edgecolor="white")
    for thr, col, lbl in [(30, "#2ca02c", "30 min"),
                          (60, "#ff7f0e", "60 min"),
                          (90, "#d62728", "90 min")]:
        ax.axvline(thr, color=col, linestyle="--", linewidth=1.2,
                   label=f">{lbl}: {(df['time3_imputed']>thr).mean()*100:.0f}%")
    ax.set_xlabel("Travel time to nearest ACOG Level-III OB facility (min, capped at 300)")
    ax.set_ylabel("Counties")
    ax.set_title("(A) County-level access to Level-III maternal care")
    ax.legend(fontsize=8, loc="upper right")

    # (B) SMM rate by ACOG discordance tier.
    ax = axes[0, 1]
    tier_labels = {0: "<=30", 1: "30-60", 2: "60-90", 3: ">90"}
    order = [0, 1, 2, 3]
    box_data = [df.loc[df["loc_discordance_tier"] == t, TARGET_PRIMARY].dropna()
                for t in order]
    bp = ax.boxplot(box_data, labels=[tier_labels[t] for t in order],
                    showfliers=False, patch_artist=True)
    for patch, col in zip(bp["boxes"], ["#2ca02c", "#9acd32", "#ff7f0e", "#d62728"]):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax.set_xlabel("Travel time to Level-III OB (min)")
    ax.set_ylabel("SMM-proxy rate per 1,000 PRG discharges")
    ax.set_title("(B) Complication burden by Level-III discordance")

    # (C) SMM rate by SVI quartile.
    ax = axes[1, 0]
    svi_data = [df.loc[df["svi_quartile"] == q, TARGET_PRIMARY].dropna()
                for q in [1, 2, 3, 4]]
    bp = ax.boxplot(svi_data, labels=["Q1\n(least)", "Q2", "Q3", "Q4\n(most)"],
                    showfliers=False, patch_artist=True)
    for patch, col in zip(bp["boxes"], ["#c6dbef", "#9ecae1", "#4292c6", "#08519c"]):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)
    ax.set_xlabel("CDC Social Vulnerability Index quartile")
    ax.set_ylabel("SMM-proxy rate per 1,000 PRG discharges")
    ax.set_title("(C) Complication burden by social vulnerability")

    # (D) 2x2 heat: access tier x SVI quartile.
    ax = axes[1, 1]
    piv = df.pivot_table(index="loc_discordance_tier",
                         columns="svi_quartile",
                         values=TARGET_PRIMARY,
                         aggfunc="median")
    im = ax.imshow(piv.values, aspect="auto", cmap="Reds")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([f"Q{int(c)}" for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([tier_labels[int(i)] for i in piv.index])
    ax.set_xlabel("SVI quartile")
    ax.set_ylabel("Level-III travel time (min)")
    ax.set_title("(D) Median SMM rate: access x SVI")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        color="white" if val > np.nanmedian(piv.values) else "black",
                        fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = C.FIG_DIR / "fig1_headline.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] figure saved -> {out.relative_to(C.PROJECT_ROOT)}")


def run() -> dict:
    _ensure_dirs()
    df = build_analytic()
    summary: dict = {"n_counties": len(df),
                     "n_states": int(df["state_fips"].nunique())}

    # Table 1
    t1 = table1(df)
    t1.to_csv(C.TAB_DIR / "table1_cohort.csv", index=False)
    print("[ok] Table 1 ->", (C.TAB_DIR / "table1_cohort.csv").name)

    # Stratified tables
    for g in ["loc_discordance_tier", "svi_quartile",
              "effective_access_tier_30min", "maternity_desert"]:
        out = stratified_table(df, g, TARGET_PRIMARY)
        out.to_csv(C.TAB_DIR / f"strat_{g}.csv", index=False)
        print("[ok] strat ->", f"strat_{g}.csv")

    # Headline figure
    figure_headline(df)

    # Save cohort snapshot for reproducibility
    (C.OUTPUT_ROOT / "cohort_snapshot.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(run())
