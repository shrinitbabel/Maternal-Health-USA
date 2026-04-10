"""End-to-end pipeline: descriptive stats -> models -> causal -> explain -> report.

Run from project root:
    python run_all.py
"""
from __future__ import annotations
import time

from src import describe, model, causal, explain, report


STEPS = [
    ("descriptive statistics + Table 1 + headline figure", describe.run),
    ("NB-GLM + XGBoost spatial CV",                        model.run),
    ("DoubleML causal analysis",                           causal.run),
    ("SHAP explainability + fairness stratification",      explain.run),
    ("summary forest plot + equity bars",                  report.run),
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
