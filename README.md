# Maternal Health USA

Live tool: **[babels.ai/maternal](https://babels.ai/maternal)**.

This project asks one question, with machine learning: **how much does a U.S. county's drive time to advanced obstetric care affect its rate of postpartum complications?** By "postpartum complications" we mean the medical events that happen to a woman after she has delivered her baby and gone home — chiefly postpartum hemorrhage requiring readmission, postpartum sepsis (typically 3–14 days after discharge), postpartum blood clots (up to six weeks), postpartum preeclampsia and eclampsia (up to six weeks), and postpartum cardiomyopathy (weeks to months). These are leading causes of late maternal death in the United States.

---

## 1. The question

How much does a U.S. county's drive time to a Level III obstetric hospital affect its rate of postpartum complications?

## 2. The answer

Doubling a woman's drive to advanced obstetric care is associated with about a **2.8% increase** in the postpartum complication rate nationally (95% certainty range 1.2% to 4.4%, p = 0.001), and roughly a **6.4% increase in rural counties** (95% range 2.0% to 10.9%, p = 0.004) — about 2.3× the national-average effect. Detailed results in section 7.


---

## 3. The method

**The overall approach, in plain English.** Two steps. First, an AI model learns to predict each county's postpartum complication rate — and its drive time to advanced care — from a set of demographic and social features for the county itself plus its 8 nearest neighbors. Second, a statistical method called DoubleML "subtracts out" the part of the rate that the AI could already explain from those features, and reads the *causal* effect of drive time off what's left over.

**What the model predicts (target).** Each county's rate of postpartum complications, expressed as cases per 1,000 pregnancy-related hospital visits.

**What goes into the model (features per county).** Drive time to the nearest Level III obstetric hospital; the four CDC Social Vulnerability Index themes (poverty and employment, household composition, minority status and language, housing and transportation); total population; share White / Black / American Indian or Alaska Native / Asian; how completely the county reports across HCUP's 15 pregnancy-related diagnosis categories; and whether the county itself contains a Level III hospital.

**The AI model: GraphSAGE.** A "graph neural network" — a type of AI model designed for data with a network structure. We treat each county as a node and connect it to its 8 nearest geographic neighbors (by straight-line distance between county centers). The model learns each county's prediction by mixing in information from those 8 neighbors, which lets it pick up spatial patterns that a regular row-by-row model (like XGBoost or random forest) cannot see, because those treat counties as independent.

**The causal estimator: DoubleML** (Chernozhukov et al. 2018). A statistical method that turns the AI's predictions into a clean causal estimate with 95% confidence intervals. To stop the model from memorizing one state's quirks, we hold out an entire state at a time during evaluation: the model trains on 36 states and is tested on the 37th, then we rotate.

---

## 5. The dataset used

Source: PRJ-6250 — Maternal Health Equity Linked Datasets (Colf et al., *Data in Brief* 2026, DOI 10.1016/j.dib.2026.112713). PRJ-6250 links four public sources at the U.S. county level for 2017–2019: AHRQ HCUP hospital discharge data (the outcomes), CDC Social Vulnerability Index 2018 (the social confounders), U.S. Census ACS / Decennial 2020 (demographics), and ArcGIS-computed drive times from each county centroid to the nearest hospital at each of four ACOG levels of maternal care. Our cohort: 2,054 counties in 37 states, covering 256 million Americans. The data descriptor PDF is in the repo root — read it first.

---

## 6. Why drive time matters even after a woman is at the hospital for delivery

A fair question: if she's already at the hospital to deliver, why does her drive time from home matter at all? The answer is that postpartum complications almost all show up *after* the woman has been discharged home. The relevant drive time is not "from home to delivery" — it's "from home back to a hospital that can manage a time-sensitive emergency" (massive transfusion, ICU, MFM, interventional radiology, neuro-OB). A woman two hours from a Level III who develops postpartum hemorrhage at home doesn't have a two-hour budget — the drive is the loss-of-care window.

Complications during pregnancy or delivery, by contrast, are managed wherever the woman is currently being cared for. That's exactly why those cases get transferred to bigger hospitals and recorded there. So postpartum is the only window where residential geography has real, immediate causal force, which is also why it's the only window where the question is answerable from this data.

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

The numbers below come from running the DoubleML × GraphSAGE pipeline on the postpartum complication rate, holding out one state at a time. Source: [`outputs/tables/dml_results.json`](matlevel-cds/outputs/tables/dml_results.json) (the raw model coefficients are in the JSON if you want them; the table here translates them into a more interpretable percentage).

The "% per doubling of drive time" column means: if a county's drive time were doubled (say, from 30 minutes to 60), how much would the model expect the postpartum complication rate to change, holding everything else fixed?

| Specification | n counties | % per doubling of drive time | 95% certainty range | p |
|---|---:|---:|---|---:|
| Main (all eligible counties) | 1,135 | +2.8% | (+1.2%, +4.4%) | 0.001 |
| Excluding counties that have a Level III hospital | 1,128 | +2.8% | (+1.2%, +4.3%) | <0.001 |
| Rural counties only (population decile ≤ 5) | 266 | +6.4% | (+2.0%, +10.9%) | 0.004 |

The result holds even if we drop the big-city counties that already have a Level III hospital — so the finding isn't being driven by cities being healthier in general. And the rural-only version is about 2.3× larger, which fits the medical mechanism: long drives matter most where the local hospital can't manage a postpartum emergency on its own.

### The sign-flip pattern across pregnancy outcomes

The same pipeline was run on all four pregnancy-related outcomes that share the residential-geography question. The pattern across outcomes is the strongest evidence inside the analysis itself that what we found for postpartum is the real geographic effect, not a coincidence.

| Outcome | n | % per doubling of drive time | 95% certainty range | p | What this means |
|---|---:|---:|---|---:|---|
| Severe maternal morbidity (composite) | 1,995 | −1.1% | (−1.9%, −0.3%) | 0.007 | Misleading — these high-risk patients get transferred to a Level III hospital, so the data records the case in the receiving county |
| Hypertensive disorders of pregnancy | 1,991 | −2.9% | (−4.1%, −1.7%) | <0.001 | Misleading — same transfer mechanism |
| Childbirth complications | 2,049 | −1.3% | (−2.3%, −0.4%) | 0.008 | Misleading — same transfer mechanism |
| Postpartum complications (puerperal) | 1,135 | +2.8% | (+1.2%, +4.4%) | 0.001 | The real residential-access signal — postpartum events happen at the local hospital, not after a transfer |

Three negative numbers on outcomes that get transferred away. One positive on the outcome that does not. The flip from negative to positive lets us tell a real residential-access effect apart from a quirk in how the data was collected — on the same data, under the same model, in the same run.

### GraphSAGE vs XGBoost — methods comparison

We ran the GraphSAGE GNN and a regular XGBoost (gradient-boosted trees) model on the same data, with the same train/test splits, and compared them. **Bold = winner per metric.** The four metrics:

- **R²** — how much of the variation in county rates the model explains (higher = better).
- **AUROC** — how well the model separates the highest-burden counties from the rest (higher = better).
- **Reg-ECE** (regression Expected Calibration Error) — how well predicted rates match actual rates (lower = better).
- **Stab. SD** (stability standard deviation) — how much the prediction wobbles when we re-train with different random starting points (lower = better).

Source: [`outputs/tables/gnn_xgb_head_to_head.json`](matlevel-cds/outputs/tables/gnn_xgb_head_to_head.json).

| Outcome | R² GNN | R² XGB | AUROC GNN | AUROC XGB | Reg-ECE GNN | Reg-ECE XGB | Stab. SD GNN | Stab. SD XGB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Postpartum (puerperal) | **0.357** | 0.300 | **0.815** | 0.796 | **0.43** | 0.74 | **1.06** | 1.31 |
| Hypertensive (HDP) | 0.194 | **0.223** | **0.727** | 0.707 | **2.67** | 5.22 | **6.87** | 8.50 |
| SMM-proxy | 0.074 | **0.137** | 0.632 | **0.635** | **5.97** | 10.08 | **13.30** | 16.80 |
| Childbirth complications | 0.081 | **0.150** | **0.671** | 0.615 | **5.81** | 11.96 | **10.46** | 12.81 |
| Placental disorders | **0.080** | 0.052 | **0.643** | 0.633 | **0.49** | 1.07 | **0.95** | 1.09 |
| Diabetes in pregnancy | **0.077** | 0.069 | **0.653** | 0.621 | **1.32** | 2.70 | **3.06** | 3.69 |

GraphSAGE wins on calibration and stability for every one of the six outcomes — typically cutting calibration error by 40–50% and prediction wobble by 15–20%. AUROC favors GraphSAGE on 5 of 6 outcomes (tied on the SMM composite). R² is split 3–3, with XGBoost's wins clustered on the outcomes that are confused by referral capture (HDP, SMM, childbirth) — where R² is misleading anyway. Reassuringly, the headline causal estimate for postpartum (+2.8% per doubling) is the same whether we use GraphSAGE or XGBoost as the AI model, so our main finding does not depend on which model we chose. The GraphSAGE pipeline just gives nicer-behaved predictions for the county-level numbers shown on the live site.


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

A simple linear regression won't work here for two reasons. First, there are too many confounders — county demographics, social vulnerability, race composition, hospital landscape — and they don't combine in a clean linear way. Second, the confounding is *spatial*: a county's complication rate depends on its neighbors' situations, not just its own. So we need a model that can represent the neighborhood structure (the GNN) paired with a method that can pull a clean causal estimate out of that model (DoubleML).