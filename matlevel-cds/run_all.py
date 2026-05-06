"""End-to-end pipeline: descriptive stats -> models -> causal -> explain -> report.

Run from project root:
    python run_all.py
"""
from __future__ import annotations
import time

from src import (causal_gnn, describe, evaluate, explain, gnn, gnn_deploy,
                 gnn_eval, model, report, shap_gnn)


STEPS = [
    ("descriptive statistics + Table 1 + headline figure", describe.run),
    ("NB-GLM + XGBoost spatial CV (legacy baseline,\n"
     "  retained as manuscript-evidence comparator only)", model.run),
    ("DoubleML causal analysis with GraphSAGE nuisances\n"
     "  (XGBoost retired from causal.py too)",            causal_gnn.run),
    ("Fairness stratification (descriptive)",              explain.run),
    ("summary forest plot + equity bars",                  report.run),
    ("full performance metrics (baselines, calibration,\n"
     "  ECE, per-state, binary AUC/F1/Brier)",             evaluate.run),
    ("GraphSAGE spatial GNN vs XGBoost (R2/MAE)",          gnn.run),
    ("GraphSAGE vs XGBoost APPLES-TO-APPLES\n"
     "  (identical splits, bootstrap CIs, calibration,\n"
     "  ECE, AUROC, stability, binary head)",              gnn_eval.run),
    ("Train + cache GraphSAGE for production scoring\n"
     "  (all 6 CDS outcomes; primary model on Render)",    gnn_deploy.run),
    ("SHAP-style attribution figs from the GraphSAGE\n"
     "  primary model (fig2 summary, fig3 dependence)",    shap_gnn.run),
]


def main():
    t0 = time.time()
    for name, fn in STEPS:
        print("\n" + "=" * 72)
        print(f"RUNNING: {name}")
        print("=" * 72)
        fn()
    print(f"\n[done] full pipeline in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
