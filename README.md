# Maternal Health USA

Live tool: **[babels.ai/maternal](https://babels.ai/maternal)**.

This project asks one question, with machine learning: **how much does a U.S. county's drive time to advanced obstetric care affect its rate of postpartum complications?** By "postpartum complications" we mean the medical events that happen to a woman *after* she has delivered her baby and gone home — chiefly **postpartum hemorrhage requiring readmission**, **postpartum sepsis** (typically 3–14 days after discharge), **postpartum venous thromboembolism / blood clots** (up to six weeks), **postpartum preeclampsia and eclampsia** (up to six weeks), and **postpartum cardiomyopathy** (weeks to months). These are leading causes of late maternal death in the United States.

---

## 1. The question

How much does a U.S. county's drive time to a Level III obstetric hospital affect its rate of postpartum complications?

## 2. The answer

Doubling a woman's drive to advanced obstetric care is associated with about a **2.8% increase** in the postpartum complication rate nationally (95% certainty range 1.2% to 4.4%, p = 0.001), and roughly a **6.4% increase in rural counties** (95% range 2.0% to 10.9%, p = 0.004) — about 2.3× the national-average effect. Detailed results in section 7.


---

## 3. The method

**The overall approach, in plain English.** We train a graph neural network to predict a county's postpartum complication rate, and its drive time to advanced care, from each county's own social and geographic features plus the features of its 8 nearest spatial neighbors. We then feed those two predictions into a debiasing estimator (DoubleML), which subtracts the part of the outcome and the part of the exposure that the model could already explain from the confounders, and reads the causal effect of drive time off the residuals. Four moving pieces, named below.

**Target.** Each county's rate of postpartum complications per 1,000 pregnancy-related hospital visits.

**Predictors per county.** Drive time to the nearest Level III obstetric hospital; the four CDC Social Vulnerability Index themes (socioeconomic status; household composition and disability; minority status and language; housing type and transportation); total population; share White / Black / AIAN / Asian; reporting completeness across the 15 PRG outcome categories; and whether the county itself hosts a Level III facility.

**Model.** A 2-layer **GraphSAGE** graph neural network on an 8-nearest-neighbor county graph, built from the haversine distance between Census 2019 Gazetteer county centroids. Each county is represented as a vector that depends on its own features and the features of its eight nearest spatial neighbors. This lets the model use spatial structure that a tabular model (XGBoost, random forests) cannot see.

**Estimator.** **DoubleML** (Chernozhukov et al. 2018) — a debiasing technique that uses cross-fitted ML predictions to recover an unbiased causal coefficient on drive time, with valid 95% confidence intervals. Folds are leave-state-out to prevent within-state leakage.

---

## 5. The dataset used

Source: **PRJ-6250 — Maternal Health Equity Linked Datasets** (Colf et al., *Data in Brief* 2026, DOI 10.1016/j.dib.2026.112713). PRJ-6250 links four public sources at the U.S. county level for 2017–2019: AHRQ HCUP hospital discharge data (the outcomes), CDC Social Vulnerability Index 2018 (the social confounders), U.S. Census ACS / Decennial 2020 (demographics), and ArcGIS-computed drive times from each county centroid to the nearest hospital at each of four ACOG levels of maternal care. Cohort: **2,054 counties in 37 states**, covering 256 million Americans. The data descriptor PDF is in the repo root — read it first.

---

## 6. Why drive time matters even after a woman is at the hospital for delivery

A common, fair question: if she's already at the hospital to deliver, why does her residential drive time matter at all? The resolution is that **postpartum complications almost all present *after* the woman has been discharged home**. The relevant drive time is therefore not "from home to delivery" — it is "from home back to a hospital that can manage a time-sensitive subspecialty emergency": massive transfusion, ICU, maternal-fetal medicine, interventional radiology, neuro-OB. A woman two hours from a Level III who develops postpartum hemorrhage at home doesn't have a two-hour budget — the drive is the loss-of-care window. Antepartum and intrapartum complications, by contrast, are managed where the woman is currently being cared for, which is exactly why their cases get transferred to receiving hospitals and recorded there. Postpartum is the only window in which residential geography has real, immediate causal force, which is also why it is the only window in which the question is answerable from this data.

---

## 7. Detailed results

### Cohort overview

| Field | Value |
|---|---|
| Counties | 2,054 |
| States | 37 |
| Population covered | ~256 million |
| Years | 2017–2019 |
| Median drive time to Level III | 67.5 min (IQR 42.9–105.6) |
| % counties >30 min from Level III | 87.5% |
| % counties >60 min from Level III | 57.6% |
| % counties >90 min from Level III | 32.3% |
| % maternity deserts (`time3 > 90 min OR time1 > 60 min`) | 34.9% |

### Headline causal estimates — postpartum complications

DoubleML × GraphSAGE, partially-linear specification on `log(time3)`, leave-state-out spatial cross-validation, influence-function-based standard errors. Source: [`outputs/tables/dml_results.json`](matlevel-cds/outputs/tables/dml_results.json).

| Specification | n counties | θ on log(drive time) | 95% CI on θ | p | **Effect per doubling of drive time** |
|---|---:|---:|---|---:|---|
| Main (all eligible counties) | 1,135 | +0.040 | (+0.018, +0.063) | 0.001 | **+2.8% (+1.2%, +4.4%)** |
| Excluding Level III host counties | 1,128 | +0.040 | (+0.018, +0.061) | <0.001 | +2.8% (+1.2%, +4.3%) |
| Rural subsample (population decile ≤ 5) | 266 | +0.089 | (+0.029, +0.149) | 0.004 | **+6.4% (+2.0%, +10.9%)** |

The estimate is robust to dropping counties that *host* a Level III facility (so the result is not driven by metropolitan counties that are themselves the destination), and the rural subsample magnifies the effect ~2.3× — consistent with the clinical mechanism that long drive times become decisive when local hospitals lack the capacity to manage time-sensitive postpartum emergencies.

### The sign-flip pattern across PRG outcomes

The same DoubleML × GraphSAGE specification was applied to all four PRG outcomes that share the residential-geography question. The pattern across outcomes is the strongest internal evidence that the postpartum estimate captures the real residential-access signal.

| Outcome | n | θ on log(drive time) | 95% CI | p | Reading |
|---|---:|---:|---|---:|---|
| Severe maternal morbidity (composite) | 1,995 | −0.016 | (−0.028, −0.004) | 0.007 | Wrong-signed — system is transferring high-risk cases to Level III |
| Hypertensive disorders of pregnancy | 1,991 | −0.043 | (−0.060, −0.025) | <0.001 | Wrong-signed — referral capture |
| Childbirth complications | 2,049 | −0.020 | (−0.034, −0.005) | 0.008 | Wrong-signed — referral capture |
| **Postpartum complications (puerperal)** | **1,135** | **+0.040** | **(+0.018, +0.063)** | **0.001** | **Correctly signed — the residential-access signal** |

Three negative coefficients on outcomes that result in patient transfers. One positive on the outcome that does not. The sign flip is the empirical contribution: it lets us distinguish a real residential-access effect from a data-design artefact, on the same data, under the same model, in the same run.

### GraphSAGE vs XGBoost — methods comparison

Apples-to-apples evaluation under identical leave-state-out CV splits, bootstrap 95% CIs, and 5-seed stability resampling. Source: [`outputs/tables/gnn_xgb_head_to_head.json`](matlevel-cds/outputs/tables/gnn_xgb_head_to_head.json). **Bold = winner.**

| Outcome | R² GNN | R² XGB | AUROC GNN | AUROC XGB | Reg-ECE GNN | Reg-ECE XGB | Stab. SD GNN | Stab. SD XGB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Postpartum (puerperal) | **0.357** | 0.300 | **0.815** | 0.796 | **0.43** | 0.74 | **1.06** | 1.31 |
| Hypertensive (HDP) | 0.194 | **0.223** | **0.727** | 0.707 | **2.67** | 5.22 | **6.87** | 8.50 |
| SMM-proxy | 0.074 | **0.137** | 0.632 | **0.635** | **5.97** | 10.08 | **13.30** | 16.80 |
| Childbirth complications | 0.081 | **0.150** | **0.671** | 0.615 | **5.81** | 11.96 | **10.46** | 12.81 |
| Placental disorders | **0.080** | 0.052 | **0.643** | 0.633 | **0.49** | 1.07 | **0.95** | 1.09 |
| Diabetes in pregnancy | **0.077** | 0.069 | **0.653** | 0.621 | **1.32** | 2.70 | **3.06** | 3.69 |

GraphSAGE wins **calibration (regression ECE) and seed stability on every one of the six outcomes** — typically reducing ECE by 40–50% and stability variance by ~15–20%. AUROC favors GraphSAGE on 5 of 6 outcomes (ties on SMM-proxy). R² is split 3–3, with XGBoost's wins concentrated on the outcomes that are referral-capture-confounded (HDP, SMM, childbirth), where R² is the least meaningful indicator. The magnitude of the causal estimate θ on the puerperal outcome is identical across both nuisance learners to four decimal places, so the headline causal claim does not depend on the choice of ML model — but the GraphSAGE-based reporting has tighter calibration and tighter prediction stability, which matters for any downstream use of the cached county-level predictions (the live tool).

A draft abstract for the eventual manuscript lives in [`ABSTRACT.md`](ABSTRACT.md).

---

## 8. Repository layout

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

## 9. Glossary

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


## extra. Why this needs machine learning

A linear regression cannot recover this effect because the confounding is high-dimensional and *spatial* — a county's outcome depends on its neighbors' demographics and travel patterns, not just its own — so we need a model that can represent spatial structure paired with a debiased estimator that turns its predictions into an unbiased causal coefficient.