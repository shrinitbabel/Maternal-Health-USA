# Maternal Health USA

Live tool: **[babels.ai/maternal](https://babels.ai/maternal)**.

This project asks one question, with machine learning: **how much does a U.S. county's drive time to advanced obstetric care affect its rate of postpartum complications?** By "postpartum complications" we mean the medical events that happen to a woman *after* she has delivered her baby and gone home — chiefly **postpartum hemorrhage requiring readmission**, **postpartum sepsis** (typically 3–14 days after discharge), **postpartum venous thromboembolism / blood clots** (up to six weeks), **postpartum preeclampsia and eclampsia** (up to six weeks), and **postpartum cardiomyopathy** (weeks to months). These are leading causes of late maternal death in the United States.

---

## 1. The question

How much does a U.S. county's drive time to a Level III obstetric hospital affect its rate of postpartum complications?

## 2. The answer

Doubling a woman's drive to advanced obstetric care is associated with about a **2.8% increase** in the postpartum complication rate nationally (95% certainty range 1.2% to 4.4%, p = 0.001), and roughly an **8.9% increase in rural counties** (95% range 2.9% to 14.9%, p = 0.004).

## 3. Why this needs machine learning

A linear regression cannot recover this effect because the confounding is high-dimensional and *spatial* — a county's outcome depends on its neighbors' demographics and travel patterns, not just its own — so we need a model that can represent spatial structure paired with a debiased estimator that turns its predictions into an unbiased causal coefficient.

---

## 4. The method

**Target.** Each county's rate of postpartum complications per 1,000 pregnancy-related hospital visits.

**Features per county.** Drive time to the nearest Level III obstetric hospital; the four CDC Social Vulnerability Index themes (socioeconomic status; household composition and disability; minority status and language; housing type and transportation); total population; share White / Black / AIAN / Asian; reporting completeness across the 15 PRG outcome categories; and whether the county itself hosts a Level III facility.

**Model.** A 2-layer **GraphSAGE** graph neural network on an 8-nearest-neighbor county graph, built from the haversine distance between Census 2019 Gazetteer county centroids. Each county is represented as a vector that depends on its own features and the features of its eight nearest spatial neighbors. This lets the model use spatial structure that a tabular model (XGBoost, random forests) cannot see.

**Estimator.** **DoubleML** (Chernozhukov et al. 2018) — a debiasing technique that uses cross-fitted ML predictions to recover an unbiased causal coefficient on drive time, with valid 95% confidence intervals. Folds are leave-state-out to prevent within-state leakage.

---

## 5. The data

Source: **PRJ-6250 — Maternal Health Equity Linked Datasets** (Colf et al., *Data in Brief* 2026, DOI 10.1016/j.dib.2026.112713). PRJ-6250 links four public sources at the U.S. county level for 2017–2019: AHRQ HCUP hospital discharge data (the outcomes), CDC Social Vulnerability Index 2018 (the social confounders), U.S. Census ACS / Decennial 2020 (demographics), and ArcGIS-computed drive times from each county centroid to the nearest hospital at each of four ACOG levels of maternal care. Cohort: **2,054 counties in 37 states**, covering 256 million Americans. The data descriptor PDF is in the repo root — read it first.

---

## 6. Why drive time matters even after a woman is at the hospital for delivery

A common, fair question: if she's already at the hospital to deliver, why does her residential drive time matter at all? The resolution is that **postpartum complications almost all present *after* the woman has been discharged home**. The relevant drive time is therefore not "from home to delivery" — it is "from home back to a hospital that can manage a time-sensitive subspecialty emergency": massive transfusion, ICU, maternal-fetal medicine, interventional radiology, neuro-OB. A woman two hours from a Level III who develops postpartum hemorrhage at home doesn't have a two-hour budget — the drive is the loss-of-care window. Antepartum and intrapartum complications, by contrast, are managed where the woman is currently being cared for, which is exactly why their cases get transferred to receiving hospitals and recorded there. Postpartum is the only window in which residential geography has real, immediate causal force, which is also why it is the only window in which the question is answerable from this data.

---

## 7. Repository layout

```
Maternal-Health-USA/
├── PRJ-6250/                                     ← raw dataset (5 MB CSV)
├── matlevel-cds/                                 ← Python pipeline + FastAPI backend
│   ├── src/
│   │   ├── analytic.py        ← analytic frame
│   │   ├── outcomes.py        ← rate construction
│   │   ├── geo.py             ← centroids, k-NN adjacency, drive-time model
│   │   ├── gnn.py             ← GraphSAGE architecture
│   │   ├── gnn_eval.py        ← apples-to-apples GNN vs XGBoost head-to-head
│   │   ├── gnn_deploy.py      ← train + cache GNN predictions per county
│   │   ├── causal_gnn.py      ← DoubleML with GraphSAGE nuisance learners (the main method)
│   │   ├── shap_gnn.py        ← gradient × input attribution for the GNN
│   │   ├── cds.py             ← ACOG-CO-726 rule engine (for the demo panel)
│   │   ├── counterfactual.py  ← hospital-siting engine
│   │   └── ...
│   ├── api/main.py            ← FastAPI: /access /recommend /counterfactual /graph/* /shap
│   ├── outputs/
│   │   ├── tables/dml_results.json           ← causal coefficients + CIs
│   │   ├── tables/gnn_xgb_head_to_head.json  ← apples-to-apples evidence
│   │   ├── models/gnn_predictions.json       ← cached GNN predictions per county × outcome
│   │   └── figures/                          ← fig1–fig9 referenced in the manuscript
│   ├── run_all.py             ← end-to-end pipeline
│   ├── render.yaml            ← Render Blueprint config
│   └── DEPLOY_RENDER.md       ← deployment notes
└── 1-s2.0-S2352340926002660-main.pdf          ← PRJ-6250 data descriptor (read first)
```

The frontend (Next.js 14 + deck.gl + maplibre on Vercel) lives in a separate `babels.ai` repo as `app/maternal/page.tsx` plus `app/api/maternal/*/route.ts` proxy handlers.

---

## 8. Glossary

- **Postpartum complications (puerperal complications)** — medical events affecting the mother after delivery, typically in the first six weeks: postpartum hemorrhage, postpartum sepsis, venous thromboembolism, postpartum preeclampsia / eclampsia, postpartum cardiomyopathy.
- **Level III obstetric hospital** — ACOG / SMFM Committee Opinion 726 designation for a subspecialty maternal-care facility (on-site MFM, adult ICU, blood bank, advanced imaging). Level IV adds the most complex maternal medical / surgical care; the source data does not separate the two.
- **HCUP** — Healthcare Cost and Utilization Project (AHRQ). National hospital discharge data.
- **PRG categories** — HCUP's pregnancy-related Clinical Classifications Software Refined diagnostic groupings.
- **SVI** — Social Vulnerability Index (CDC / ATSDR). Composite of four themes; higher = more vulnerable.
- **AIAN** — American Indian / Alaska Native.
- **Drive time (`time3`)** — minutes from the county centroid to the nearest Level III obstetric facility, computed by the PRJ-6250 team in ArcGIS.
- **GraphSAGE** — Hamilton et al. 2017. A graph neural network that updates each node's representation by aggregating from its neighbors. We use a 2-layer version on the 8-nearest-neighbor county graph.
- **DoubleML** — double / debiased machine learning (Chernozhukov 2018). A plug-in that recovers an unbiased causal coefficient using cross-fitted ML for the nuisance functions.
- **Leave-state-out CV** — cross-validation in which entire states are held out, to prevent within-state leakage.
- **Referral capture** — the data-design phenomenon by which a discharge after a patient transfer is recorded at the receiving hospital. The reason the geographic question is only honestly answerable for postpartum outcomes.

---

## Suggested citation

> *How much does a U.S. county's drive time to advanced obstetric care affect its rate of postpartum complications? An interactive geospatial-AI companion.* Shrinit Babel, Leremy A. Colf, et al., 2026.
> Live tool: [babels.ai/maternal](https://babels.ai/maternal).
> Data: Colf et al., *Data in Brief* 2026, DOI 10.17603/ds2-62zy-zh93.
