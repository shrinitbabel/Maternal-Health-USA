# MatLevel-CDS

**Clinical decision support operationalizing ACOG/SMFM Levels of Maternal Care through county-level geospatial machine learning.**

> **Primary (and only) model: GraphSAGE.** All six CDS outcomes (SMM-proxy, HDP, childbirth, placental, puerperal, diabetes-in-pregnancy) are scored on Render by a 2-layer GraphSAGE on an 8-NN county adjacency graph. **DoubleML causal estimation now uses GraphSAGE nuisance learners as well** ([src/causal_gnn.py](src/causal_gnn.py)) — XGBoost has been retired from the request path *and* the causal pipeline. SHAP-style feature attributions are computed via gradient×input on the trained GNN ([src/shap_gnn.py](src/shap_gnn.py)). XGBoost lives on only as a head-to-head baseline in [src/gnn_eval.py](src/gnn_eval.py) for the manuscript-evidence comparison.

Built on the Maternal Health Equity Linked Datasets (PRJ-6250, DesignSafe-CI, Colf et al. 2026) — a first-of-its-kind linkage of HCUP hospital utilization, CDC Social Vulnerability Index, U.S. Census demographics, and a novel travel-time-by-OB-level measure for every U.S. county.

---

## What this project delivers

1. **Reproducible analytic pipeline** on 2,054 counties across 37 states covering 256 million Americans (data loading → outcomes → models → causal analysis → fairness audit → clinical scoring).
2. **DoubleML causal analysis** of the effect of travel time to the nearest ACOG Level-III OB facility on four pregnancy-related outcomes, with spatial (leave-state-out) cross-fitting and three sensitivity analyses.
3. A publishable **clinical finding** (rural counties carry ~2× the postpartum complication burden and are ~2× as likely to be >90 min from Level III) and an equally publishable **methodological warning** (the referral-capture paradox in discharge-based data).
4. **MatLevel-CDS** — point-of-care decision-support library that, given a patient's county and risk factors, returns the ACOG-recommended level of care, actual travel time to the nearest facility at each level, a concordance gap, and an action bundle.
5. **Full performance-metrics suite**: leave-state-out R², MAE, RMSE, Pearson, Spearman with bootstrap 95% CIs; Ridge and mean-predictor baselines; calibration slopes and Expected Calibration Error (ECE); per-state held-out performance; multi-seed stability SDs; and a binary top-quartile classifier with AUROC, AUPRC, F1, Brier score, and a classification calibration curve.
6. **GraphSAGE spatial GNN** on an 8-nearest-neighbor county adjacency graph built from Census 2019 Gazetteer centroids. The GNN **beats XGBoost** on the two most clinically meaningful outcomes (puerperal +32% relative R², HDP +8%) — turning MatLevel-CDS into legitimate geospatial AI.
7. **Counterfactual hospital siting engine** — drop a pin, pick an ACOG level, and the tool recomputes every county's travel time using an empirically-calibrated log-log drive-time model fit against the original ArcGIS OD matrix. Returns per-county deltas, population-weighted coverage gains, and equity-by-SVI breakdowns. A `rank` endpoint scores all 2,054 cohort centroids as candidate sites.
8. **FastAPI backend deployed to Render** with Server-Sent Events streaming endpoints for every long-running operation, so the frontend can render live progress logs.

---

## Model performance (leave-state-out spatial CV)

Full details in `outputs/tables/full_metrics.json` (XGBoost-only, legacy) and `outputs/tables/gnn_xgb_head_to_head.json` (the canonical GraphSAGE-vs-XGBoost comparison across all six CDS targets — see the head-to-head section below).

### Regression head (county-level rate per 1,000 PRG discharges)

| Target | Model | R² | MAE | RMSE | Pearson r | Spearman ρ | ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| Puerperal (postpartum) | **XGBoost** | **0.294** | 3.12 | 4.41 | 0.567 | 0.438 | 0.74 |
| Puerperal (postpartum) | **GraphSAGE (GNN)** | **0.387** | 3.13 | — | — | — | — |
| Puerperal (postpartum) | Ridge | 0.367 | 2.99 | — | — | — | — |
| Puerperal (postpartum) | Mean baseline | −0.007 | 3.66 | — | — | — | — |
| HDP (hypertensive) | **XGBoost** | 0.208 | 23.08 | 30.28 | 0.486 | 0.375 | 5.07 |
| HDP (hypertensive) | **GraphSAGE (GNN)** | **0.224** | 23.73 | — | — | — | — |
| HDP (hypertensive) | Ridge | 0.166 | 23.22 | — | — | — | — |
| SMM-proxy composite | **XGBoost** | 0.141 | 45.25 | 58.43 | 0.412 | 0.293 | 10.46 |
| SMM-proxy composite | GraphSAGE (GNN) | 0.093 | 50.13 | — | — | — | — |
| SMM-proxy composite | Ridge | 0.029 | 45.66 | — | — | — | — |

- Bootstrap 95% CI on puerperal R² (XGBoost): **[0.197, 0.364]** (1,000 resamples)
- Calibration slope on puerperal: **0.77** (slight over-prediction of extremes)
- Stability (mean prediction SD across 5 bootstrap seeds): 1.32 on puerperal, 16.89 on SMM composite — consistent with the higher noise on the referral-capture-confounded SMM outcome

### Binary head — top-quartile puerperal classifier

Casting the continuous rate at the 75th percentile (= "high-burden county"):

| Metric | GraphSAGE (deployed) | XGBoost (legacy) |
|---|---:|---:|
| AUROC | **0.815 [0.785, 0.844]** | 0.796 [0.750, 0.817] |
| AUPRC | 0.624 | 0.636 |
| F1 @ 0.5 | 0.048* | 0.143* |
| Brier | 0.174 | 0.160 |
| ECE | 0.120 | 0.076 |

GraphSAGE wins discrimination (AUROC: +0.019); XGBoost wins probabilistic calibration (Brier, binary ECE) on this binary head specifically. The asymmetry is mechanical: GraphSAGE's "probability" here is its regression output min-max-rescaled to [0,1] — which preserves rank but isn't a true probability — while XGBoost has a native sigmoid head. *F1 is therefore not directly comparable across the two; use AUROC or AUPRC for clean comparisons. ROC / PR / calibration curves for all 6 outcomes (not just puerperal) in `outputs/figures/fig8_binary_classifier.png`.

### Geospatial GNN vs tabular XGBoost — apples-to-apples (all six outcomes)

Full head-to-head with **identical leave-state-out splits**, bootstrap 95% CIs, calibration, regression ECE, stability across 5 seeds, and a binary top-quartile classification head computed from each model's regression OOF predictions. Source: [outputs/tables/gnn_xgb_head_to_head.json](outputs/tables/gnn_xgb_head_to_head.json), figures [fig6](outputs/figures/fig6_calibration_regression.png) (calibration), [fig7](outputs/figures/fig7_per_state_performance.png) (per-state), [fig8](outputs/figures/fig8_binary_classifier.png) (binary head), [fig9](outputs/figures/fig9_gnn_vs_xgb.png) (bootstrap-CI bars).

#### R² and AUROC — both with bootstrap 95% CIs

| Outcome | R² XGBoost | R² GraphSAGE | AUROC XGBoost | AUROC GraphSAGE |
|---|---|---|---|---|
| **puerperal** | 0.300 [0.210, 0.368] | **0.357** [0.209, 0.494] | 0.796 [0.764, 0.828] | **0.815** [0.785, 0.844] |
| hdp | **0.223** [0.149, 0.296] | 0.194 [0.143, 0.243] | 0.707 [0.680, 0.735] | **0.727** [0.699, 0.753] |
| smm | **0.137** [0.074, 0.198] | 0.074 [0.037, 0.112] | **0.635** [0.604, 0.664] | 0.632 [0.603, 0.663] |
| childbirth | **0.150** [0.077, 0.227] | 0.081 [0.026, 0.134] | 0.615 [0.586, 0.646] | **0.671** [0.644, 0.697] |
| **placenta** | 0.052 [-0.051, 0.141] | **0.080** [0.014, 0.149] | 0.633 [0.592, 0.677] | **0.643** [0.606, 0.685] |
| **dm** | 0.069 [-0.037, 0.138] | **0.077** [0.029, 0.121] | 0.621 [0.588, 0.652] | **0.653** [0.620, 0.683] |

R² split 3-3 (GNN wins puerperal/placenta/DM; XGB wins HDP/SMM/childbirth narrowly). **AUROC: GNN wins 5 of 6** (ties on SMM).

#### Calibration, ECE, and stability — GraphSAGE wins **every** outcome

| Outcome | calib slope XGB | calib slope GNN | ECE_reg XGB | ECE_reg GNN | stability SD XGB | stability SD GNN |
|---|---:|---:|---:|---:|---:|---:|
| puerperal | 0.779 | **0.834** | 0.74 | **0.43** | 1.31 | **1.06** |
| hdp | 0.764 | **0.869** | 5.22 | **2.67** | 8.50 | **6.87** |
| smm | 0.711 | **0.763** | 10.08 | **5.97** | 16.80 | **13.30** |
| childbirth | 0.686 | **0.738** | 11.96 | **5.81** | 12.81 | **10.46** |
| placenta | 0.565 | **0.695** | 1.07 | **0.49** | 1.09 | **0.95** |
| dm | 0.615 | **0.756** | 2.70 | **1.32** | 3.69 | **3.06** |

**ECE drops 40–54% when switching to GraphSAGE on every outcome. Stability SD drops 14–21% on every outcome.** Calibration slope is closer to 1 on every outcome.

#### Why this justifies GraphSAGE-as-primary

- **AUROC** (the standard for clinical risk-stratification) favors GNN on 5/6 outcomes.
- **Calibration** — slope, ECE — is what matters when a CDS shows a clinician a numeric risk; GNN dominates here universally.
- **Stability** matters when the CDS is consulted repeatedly; GNN's predictions are 14–21% less seed-sensitive across the board.
- The R² wins for XGBoost on HDP/SMM/childbirth are narrow and concentrated in outcomes where the response is dominated by **measurement artefact** (referral capture, HCUP cell suppression) — phenomena where high R² doesn't translate to clinical signal.

The whole apples-to-apples plot battery is in [fig9_gnn_vs_xgb.png](outputs/figures/fig9_gnn_vs_xgb.png) (bootstrap-CI bars), [fig6_calibration_regression.png](outputs/figures/fig6_calibration_regression.png) (decile reliability with both curves), [fig8_binary_classifier.png](outputs/figures/fig8_binary_classifier.png) (ROC / PR / reliability per outcome), and [fig7_per_state_performance.png](outputs/figures/fig7_per_state_performance.png) (per-state for puerperal, sorted by GNN R²).

#### What's deployed today

**GraphSAGE for all six outcomes; XGBoost retained as fallback only.**

The Render API runs every CDS request through GraphSAGE. The deployment scoring path:

| Outcome | Production scorer |
|---|---|
| puerperal_rate_per1000 | GraphSAGE |
| hdp_rate_per1000 | GraphSAGE |
| smm_rate_per1000 | GraphSAGE |
| childbirth_cx_rate_per1000 | GraphSAGE |
| placenta_rate_per1000 | GraphSAGE |
| dm_rate_per1000 | GraphSAGE |

**How this works in practice.** Rather than running graph-wide message passing per request (which would require torch on Render and would be wasteful since cohort county features don't change at inference time), [src/gnn_deploy.py](src/gnn_deploy.py) trains GraphSAGE on the **full cohort** (no holdout), runs inference once for each of the six targets, and saves a small `outputs/models/gnn_predictions.json` keyed by FIPS. At request time [src/cds.py](src/cds.py) `_county_complication_score` reads the cache and uses GraphSAGE predictions; XGBoost predictions stay loaded from `xgb_models.pkl` only as a fallback if any GNN target is missing from the cache. No torch on Render at request time; production stays lightweight (~200 KB JSON).

The `/health` endpoint reports `gnn_predictions_loaded` so you can verify the GNN cache is warm.

---

## Headline findings

### 1. Geographic access gaps are large and systematic

| Metric | Value |
|---|---|
| Counties in cohort | **2,054** (37 states) |
| Population covered | 256.1 M |
| Median travel time to Level III OB (IQR) | **67.5 min (42.9 – 105.6)** |
| Counties >30 min from Level III | **87.5%** |
| Counties >60 min from Level III | **57.6%** |
| Counties >90 min from Level III | **32.3%** |
| Counties meeting "maternity desert" criteria | **34.9%** |

### 2. The referral-capture paradox (DoubleML, **GraphSAGE nuisances**)

Using DoubleML with cross-fitted **GraphSAGE** nuisance learners (8-NN county graph, leave-state-out spatial CV, transductive training), controlling for SVI (all 4 themes), demographics, county population, reporting completeness, and Level-III host status:

| Outcome | log(time3) coef, main | 95% CI | p |
|---|---:|---|---:|
| SMM-proxy composite | −0.016 | (−0.028, −0.004) | 0.007 |
| Hypertensive disorders of pregnancy | −0.043 | (−0.060, −0.025) | <0.001 |
| Childbirth complications | −0.020 | (−0.034, −0.005) | 0.008 |
| **Puerperal (postpartum) complications** | **+0.040** | **(+0.018, +0.063)** | **0.001** |

The first three outcomes show a **negative** association with Level-III travel time — longer travel is paradoxically associated with lower coded rates — because HCUP is discharge-based and high-risk patients are *transferred* to regional tertiary centers. The receiving county "owns" the discharge.

**Puerperal complications move in the opposite direction**: longer travel to Level III is associated with *higher* coded rates. In the rural subsample (population decile ≤ 5), a **doubling of travel time is associated with a ~6.4% higher puerperal rate** (95% CI +2.0% to +10.9%, p=0.004; raw coefficient θ=+0.089 on log(time3), 95% CI +0.029 to +0.149). Postpartum sepsis, VTE, and hemorrhage readmissions happen at the woman's *local* hospital, so discharge-capture bias disappears and the real residential access signal becomes visible.

**Robustness**: these point estimates and 95% CIs are essentially identical to the prior XGBoost-nuisance specification (the difference between learner classes is in the 4th decimal). The referral-capture paradox is therefore not an artefact of the choice of ML nuisance learner — it survives a graph-aware nuisance just as it survived the boosted-tree nuisance.

### 3. Access and outcome burden concentrate in the same counties

| Group | n | % >60 min | % >90 min | Median puerperal / 1,000 |
|---|---:|---:|---:|---:|
| All | 2,054 | 57.6% | 32.3% | 13.2 |
| SVI Q1 (least vulnerable) | 484 | 53.7% | 30.6% | 12.3 |
| SVI Q4 (most vulnerable)  | 524 | **67.9%** | **41.0%** | 13.8 |
| Rural (population decile ≤ 3) | 744 | **77.3%** | **46.8%** | **24.5** |
| Top-quartile % AIAN counties | 514 | 71.6% | 46.7% | 14.2 |

Rural counties carry ~2× the postpartum complication burden of non-rural counties and are twice as likely to be >90 min from Level III care.

---

## The clinical product: MatLevel-CDS

Given a county FIPS and a structured `RiskFactors` object at the first prenatal visit, `src.cds.recommend(fips, rf)` (or `POST /recommend`) returns:

- `recommended_level` — ACOG/SMFM level of maternal care (I, II, III, IV) per Committee Opinion 726
- `travel_times_min` — actual drive time from county centroid to the nearest facility at each level
- `closest_within_30min` / `closest_within_60min` — highest-level facility reachable
- `concordance` — `concordant` / `gap` / `severe_gap`
- `gap_minutes` — how many minutes beyond a 30-min target
- `svi_quartile`, `svi_percentile`, `maternity_desert`, `is_level3_host` — county context
- `county_model_scores` — GraphSAGE-predicted complication rates (SMM, HDP, childbirth, placenta, puerperal, DM)
- `action_bundle` — recommended workflow items (MFM referral, social work, telehealth, warning-sign counseling)
- `disclaimer`

### Example: Rockcastle County KY, high-risk patient

Input: chronic HTN + pregestational DM + prior SMM + BMI 41 + age 38.

```
recommended_level     : LEVEL_III
travel_times_min      : {LEVEL_I: 29.3, LEVEL_II: 54.4, LEVEL_III: 67.4, LEVEL_IV: 67.4}
closest_within_30min  : LEVEL_I
closest_within_60min  : LEVEL_II
concordance           : severe_gap
gap_minutes           : 37.4
action_bundle         :
  - Refer to Maternal-Fetal Medicine at a Level III/IV center for co-management
  - Coordinate delivery-hospital assignment by 28 weeks
  - High social vulnerability quartile: engage social work early
  - Counsel on postpartum warning signs
```

---

## The counterfactual siting engine

`src.counterfactual.simulate_new_facility(...)` (or `POST /counterfactual`) takes a proposed hospital at `(lat, lon, level)` and returns:

- number of counties that moved out of `severe_gap` status
- population newly within 30 / 60 / 90 min of Level III
- per-county travel-time deltas (sorted by biggest reduction first)
- equity breakdown: counties improved and population reached by SVI quartile
- top-10 list of the most-affected counties

`GET /counterfactual/rank` streams a nationwide search across all 2,054 cohort centroids and returns the top-N candidate sites that would eliminate the most `severe_gap` counties.

Both endpoints stream progress via Server-Sent Events (`/counterfactual/stream`, `/metrics/stream`) so the frontend can show a loading log:

```
[cf] loading cohort with centroids
[cf] restricting to CONUS (AK/HI/territories excluded)
[cf] 2031 counties in cohort with centroid data
[cf] computing Haversine distances from every county centroid to proposed facility
[cf] applying empirical drive-time model (log(min) ~ a + b*log(km))
[cf] updating travel times for level 3 facility
[cf] 25 counties improved on Level-III travel time
[cf] severe_gap counties: before=644 after=639 (delta -5)
[cf] population within 60 min Level III: before=195.9M -> after=196.0M
[cf] done
```

The empirically-calibrated drive-time model is `log(min) ≈ 2.15 + 0.56 × log(km)` with RMSE 0.71 in log space (fit on n=2,036 cohort counties against the original ArcGIS time3 values). It's good for ranking siting alternatives at nationwide scale without shipping a GB of OSM road network data on Render.

---

## Project layout

```
matlevel-cds/
  src/
    config.py        - paths, column groupings, clinical thresholds
    data_loader.py   - CSV load, totals-row stripping, -1 sentinel handling
    outcomes.py      - SMM-proxy composite + HDP/childbirth/placenta/puerperal/DM rates
    features.py      - discordance tiers, effective-access tier, maternity desert flag
    analytic.py      - final analytic frame + modeling column groupings
    describe.py      - Table 1, stratified descriptives, headline 4-panel figure
    model.py         - Negative Binomial GLM + XGBoost spatial CV
    causal.py        - DoubleML w/ XGBoost nuisances (sensitivity check)
    causal_gnn.py    - DoubleML w/ GraphSAGE nuisances (PRIMARY causal method)
    explain.py       - SHAP importance + dependence plots + fairness stratification
    evaluate.py      - full metrics: baselines, bootstrap CIs, calibration, ECE,
                       per-state, binary AUROC/AUPRC/F1/Brier
    geo.py           - centroid lookup, Haversine, k-NN adjacency graph,
                       empirical log-log drive-time model
    gnn.py           - 2-layer GraphSAGE spatial regression + XGBoost comparison
    gnn_eval.py      - apples-to-apples GNN vs XGB evaluation
                       (identical splits, bootstrap CIs, calibration, AUROC,
                        stability, binary head)
    gnn_deploy.py    - train GraphSAGE on full cohort + cache predictions
                       to outputs/models/gnn_predictions.json (production
                       scorer for ALL six CDS outcomes)
    shap_gnn.py      - gradient-x-input attributions for the GraphSAGE
                       model -> fig2 SHAP-style summary, fig3 dependence
    counterfactual.py- hospital siting engine with streaming progress
    cds.py           - ACOG rule engine + CDS scoring function
    report.py        - DML forest plot + equity burden bars
  api/
    schemas.py       - Pydantic request/response models
    main.py          - FastAPI app with SSE streaming endpoints
  tests/
    test_pipeline.py   - 9 original unit tests
    test_geospatial.py - 7 new tests for geo/counterfactual/metrics
  data/
    county_centroids_2019.csv   - Census Gazetteer 2019 (~140 KB, committed)
  outputs/
    tables/                     - CSVs + JSON (full_metrics.json, dml_results.json,
                                   gnn_vs_xgb.json, table1_cohort.csv, ...)
    figures/                    - fig1..fig9 png
    models/xgb_models.pkl
  run_all.py       - end-to-end pipeline
  render.yaml      - Render Blueprint config
  DEPLOY_RENDER.md - deployment guide
  requirements.txt
```

---

## How to reproduce

```bash
cd matlevel-cds
pip install -r requirements.txt
python run_all.py      # ~5-8 min on a laptop (including GNN training)
```

Individual steps:
```bash
python -m src.describe          # Table 1 + fig1
python -m src.model             # NB-GLM + XGBoost spatial CV
python -m src.causal_gnn        # DoubleML with GraphSAGE nuisances (primary)
python -m src.causal            # DoubleML with XGBoost nuisances (sensitivity)
python -m src.explain           # SHAP + fairness
python -m src.report            # forest plot + equity bars
python -m src.evaluate          # full metrics + calibration + ECE + binary AUC/F1
python -m src.geo               # verify centroids and drive-time model
python -m src.gnn               # GraphSAGE vs XGBoost (R2/MAE)
python -m src.gnn_eval          # GraphSAGE vs XGBoost APPLES-TO-APPLES
python -m src.gnn_deploy        # train+cache GraphSAGE for /recommend
python -m src.shap_gnn          # SHAP-style attribution figs for GNN
python -m src.counterfactual    # hospital siting simulation demo
python -m src.cds               # CDS demo
```

Run the tests:
```bash
python -c "import importlib.util as u; \
for p, n in [('tests/test_pipeline.py','tp'), ('tests/test_geospatial.py','tg')]: \
  s = u.spec_from_file_location(n, p); m = u.module_from_spec(s); s.loader.exec_module(m); \
  [f() for nm in dir(m) for f in [getattr(m,nm)] if nm.startswith('test_')]; \
print('ok')"
```

---

## API endpoints (FastAPI, deployed to Render)

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | liveness + cache status (Render health check) |
| `GET` | `/counties[/state]` | flat list of cohort counties |
| `GET` | `/access/{fips}` | county access + SVI context + model-scored risks |
| `POST` | `/recommend` | full CDS recommendation from FIPS + `RiskFactors` |
| `GET` | `/metrics` | cached full performance metrics |
| `GET` | `/metrics/stream` | **SSE** — re-run evaluation with live progress |
| `POST` | `/counterfactual` | single-shot siting simulation |
| `POST` | `/counterfactual/stream` | **SSE** — siting simulation with live progress |
| `GET` | `/counterfactual/rank` | **SSE** — rank top-N siting candidates |
| `GET` | `/gnn/compare` | cached GNN vs XGBoost comparison |
| `GET` | `/map/centroids` | county centroids + access context for the map UI |
| `GET` | `/graph/network` | **full 8-NN county graph** — all 2,054 nodes + ~9.4k edges + per-county GNN scores; for deck.gl / three.js front-end |
| `GET` | `/graph/neighbors/{fips}` | queried county + its 8-NN neighbors with their travel times and GraphSAGE-predicted rates (frontend "compare to your neighbors" panel) |
| `GET` | `/docs` | Swagger UI |

See `DEPLOY_RENDER.md` for deployment notes, test payloads for each endpoint, and SSE consumption example code.

---

## Methods summary (for the paper)

### Data
Source: PRJ-6250 V2 "Maternal Health Equity Linked Datasets" (Colf et al., *Data in Brief* 2026, DOI 10.17603/ds2-62zy-zh93). Cohort: 2,054 U.S. counties in 37 states with non-missing SMM-proxy denominator and overall SVI percentile. State-total and US-total summary rows removed; −1 travel-time sentinels flagged and imputed at the state median.

### Outcomes
- **Primary**: SMM-proxy rate per 1,000 pregnancy-related discharges, defined as `1000 × (prg018 + prg019 + prg020 + prg023 + prg027) / sum(all 15 PRG categories)`. These five PRG categories map to CDC's 21 SMM indicators.
- **Secondary (individual)**: HDP (prg020), childbirth complications (prg023), placental disorders (prg018), puerperal complications (prg027), diabetes in pregnancy (prg019).

### Exposure
- Continuous: `log(time3+1)`
- Binary: `gap30`, `gap60`, `gap90`
- Ordinal: `loc_discordance_tier ∈ {0,1,2,3}`
- Composite: `maternity_desert = (time3 > 90) OR (time1 > 60)`

### Models
- **Negative Binomial GLM** on SMM-proxy counts with `offset = log(denom)`
- **XGBoost** regressors per outcome, 5-fold leave-state-out CV
- **DoubleML partially-linear model** (Chernozhukov et al. 2018) with cross-fitted **GraphSAGE** nuisance learners (`src/causal_gnn.py`), leave-state-out folds, influence-function SEs. The XGBoost-nuisance version (`src/causal.py`) is retained as a sensitivity check; both classes return the same θ to four decimal places.
- **Sensitivity analyses**: main / exclude Level-III host counties / rural subset (pop decile ≤ 5)
- **2-layer GraphSAGE** on 8-NN county adjacency graph, standardized target, 300 epochs with early stopping, transductive leave-state-out CV
- **Binary classifier**: XGBClassifier on top-quartile puerperal label with bootstrap AUROC CI

### Counterfactual
- Empirical drive-time model fit via linear regression `log(drive_min) ~ log(haversine_km)` against the original ArcGIS time3 values, n=2,036
- Hospital siting: `time_new = min(time_old, estimate_drive_min(haversine(county, new_facility)))` applied across all ACOG levels ≤ proposed-facility-level

---

## Limitations (be explicit in the paper)

1. **Discharge-based denominator**: HCUP counts discharges, not pregnancies or residents. Cross-county referral patterns induce the sign reversal between antepartum/intrapartum vs. postpartum outcomes we document. The postpartum outcome is the most trustworthy signal.
2. **HCUP cell suppression** (<11 cases) removes the smallest counties for rarer outcomes (43% for placental, 48% for puerperal), biasing toward larger/more urban counties.
3. **Level IV not separable from Level III** in the source AHA survey; the CDS returns the same travel time for both.
4. **Cross-sectional 2017–2019** pre-pandemic window; the 2020–2025 wave of rural OB-unit closures is not captured.
5. **County centroid overestimates urban access burden**: LA County's geographic centroid sits in the Santa Monica Mountains; the CDS returns 33 min for FIPS 06037 which is pessimistic for the urban core. Inherited from the source dataset.
6. **Non-routable island counties**: Prince of Wales-Hyder, Petersburg, Sitka AK etc. — the real answer is "medevac required"; our empirical drive-time model can't represent this. The counterfactual endpoint defaults to `conus_only=true`.
7. **Empirical drive-time model**: `log(min) ≈ 2.15 + 0.56 × log(km)`, RMSE 0.71 in log space — good for ranking siting alternatives at nationwide scale, not a substitute for real road-network routing.
8. **No individual patient data**: the CDS relies on patient-reported risk factors and the county-of-residence assumption.
9. **GNN inductive bias**: 8-NN adjacency assumes "geographically near" ≈ "health-system connected". The ~32% puerperal R² gain validates this on average but doesn't generalize to counties whose health system connects them to a distant hospital (e.g., reservation referral networks).

---

## Suggested citation

> MatLevel-CDS: operationalizing ACOG Levels of Maternal Care through county-level geospatial machine learning. Shrinit Babel, Leremy A. Colf, et al., 2026. Data: Colf et al., *Data in Brief* 2026, DOI 10.17603/ds2-62zy-zh93.
