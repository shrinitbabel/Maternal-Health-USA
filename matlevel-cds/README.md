# MatLevel-CDS

**Clinical decision support operationalizing ACOG/SMFM Levels of Maternal Care through county-level geographic access modeling.**

Built on the Maternal Health Equity Linked Datasets (PRJ-6250, DesignSafe-CI, Colf et al. 2026), a first-of-its-kind linkage of HCUP hospital utilization, CDC Social Vulnerability Index, U.S. Census demographics, and a novel travel-time-by-OB-level measure for every U.S. county.

---

## What this project delivers

1. A reproducible analytic pipeline (data loading -> outcomes -> models -> causal analysis -> fairness audit -> clinical scoring) on 2,054 counties across 37 states covering 256 million Americans.
2. A **DoubleML** causal analysis of the effect of travel time to the nearest ACOG Level-III OB facility on four pregnancy-related outcomes, with spatial (leave-state-out) cross-fitting and three sensitivity analyses.
3. A publishable **clinical finding** and an equally publishable **methodological warning**.
4. **MatLevel-CDS** — a point-of-care decision-support library that, given a patient's county and risk factors, returns the ACOG-recommended level of care, actual travel time to the nearest facility at each level, a concordance gap, and an action bundle.

---

## Headline findings

### 1. Geographic access gaps are large and systematic

| Metric | Value |
|---|---|
| Counties in cohort | **2,054** (37 states) |
| Population covered | 256.1 M |
| Median travel time to Level III OB (IQR) | **67.5 min (42.9 - 105.6)** |
| Counties >30 min from Level III | **87.5%** |
| Counties >60 min from Level III | **57.6%** |
| Counties >90 min from Level III | **32.3%** |
| Counties meeting "maternity desert" criteria | **34.9%** |

### 2. The referral-capture paradox

Using DoubleML with cross-fitted XGBoost nuisance learners and leave-state-out spatial cross-validation, controlling for SVI (all 4 themes), demographics, county population, and reporting completeness, we find:

| Outcome | log(time3) coef, main | 95% CI | p |
|---|---:|---|---:|
| SMM-proxy composite | -0.016 | (-0.028, -0.004) | 0.007 |
| Hypertensive disorders of pregnancy | -0.043 | (-0.060, -0.025) | <0.001 |
| Childbirth complications | -0.020 | (-0.034, -0.005) | 0.008 |
| **Puerperal (postpartum) complications** | **+0.040** | **(+0.018, +0.063)** | **0.001** |

The first three outcomes show a **negative** association with Level-III travel time — longer travel is paradoxically associated with lower coded rates — because HCUP is a discharge-based data source and high-risk patients are *transferred* to regional tertiary centers. The receiving county "owns" the discharge, inflating its antepartum/intrapartum complication rates relative to the patient's county of residence.

**The puerperal complication rate moves in the opposite direction**: longer travel to Level III is associated with *higher* coded postpartum complications. In the rural subsample (population decile <= 5), a doubling of travel time to Level III is associated with ~6% higher log puerperal complication rate (95% CI +2% to +11%, p=0.004). This is biologically coherent: post-discharge postpartum morbidity (sepsis, VTE, hemorrhage-related readmissions) is captured at the woman's local hospital, not the distant referral center.

### 3. Access burden and outcome burden concentrate in the same counties

From `outputs/tables/equity_summary.csv`:

| Group | n | % >60 min | % >90 min | Median puerperal / 1,000 |
|---|---:|---:|---:|---:|
| All | 2,054 | 57.6% | 32.3% | 13.2 |
| SVI Q1 (least vulnerable) | 484 | 53.7% | 30.6% | 12.3 |
| SVI Q4 (most vulnerable)  | 524 | **67.9%** | **41.0%** | 13.8 |
| Rural (population decile <=3) | 744 | **77.3%** | **46.8%** | **24.5** |
| Top quartile % AIAN counties | 514 | 71.6% | 46.7% | 14.2 |

Rural counties carry roughly **twice** the postpartum complication burden of non-rural counties and nearly half are >90 min from Level III care.

---

## The clinical product: MatLevel-CDS

Given a county FIPS and a structured `RiskFactors` object at the first prenatal visit, `src.cds.recommend(fips, rf)` returns:

- `recommended_level` - ACOG/SMFM level of maternal care (I, II, III, IV) per Committee Opinion 726
- `travel_times_min` - actual drive time from county centroid to the nearest facility at each level
- `closest_within_30min` / `closest_within_60min` - highest-level facility reachable
- `concordance` - `concordant` / `gap` / `severe_gap`
- `gap_minutes` - how many minutes beyond a 30-min target
- `svi_quartile`, `svi_percentile`, `maternity_desert`, `is_level3_host` - county context
- `county_model_scores` - XGBoost-predicted complication rates
- `action_bundle` - recommended workflow items (MFM referral, social work, telehealth, warning-sign counseling)
- `disclaimer`

The ACOG rule engine in `src/cds.py` implements the published risk-factor list, including Level IV indications (placenta accreta spectrum, NYHA III/IV cardiac disease, fetal anomaly requiring surgery), Level III indications (severe preeclampsia, delivery <32w, severe pulmonary/renal/autoimmune disease, prior SMM), Level II indications (chronic hypertension, pregestational diabetes, multiple gestation, TOLAC, age >=40, BMI >=40), and Level I defaults.

**Example** - Rockcastle County, KY (median-access, SVI Q4), high-risk patient with chronic HTN + pregestational DM + prior SMM + BMI 41:

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

## Project layout

```
matlevel-cds/
  src/
    config.py      - paths, column groupings, clinical thresholds, ACOG-level-of-care crosswalk
    data_loader.py - CSV load, totals-row stripping, -1 sentinel handling, state-median imputation
    outcomes.py    - SMM-proxy composite + HDP / childbirth cx / placenta / puerperal / DM rates
    features.py    - Level-III discordance tiers, effective-access tier within 30 min,
                     maternity desert flag, rural/SVI quartiles
    analytic.py    - final analytic frame + modeling column groupings
    describe.py    - Table 1, stratified descriptive tables, headline 4-panel figure
    model.py       - Negative Binomial GLM (rate-ratio table) + XGBoost spatial CV
    causal.py      - DoubleML partially-linear model, cross-fitted by state, 3 sensitivity analyses
    explain.py     - SHAP global importance + dependence plots + fairness stratification tables
    cds.py         - ACOG rule engine + CDS scoring function
    report.py      - forest plot of DML effects, equity burden bars
  tests/
    test_pipeline.py
  outputs/
    tables/        - all CSV and JSON result files
    figures/       - fig1_headline.png, fig2_shap_*.png, fig3_shap_dep_*.png,
                     fig4_dml_forest.png, fig5_equity_bars.png
    models/xgb_models.pkl
  run_all.py       - end-to-end pipeline
  requirements.txt
```

---

## How to reproduce

```bash
cd matlevel-cds
pip install -r requirements.txt
python run_all.py      # ~1-2 min on a laptop
```

Individual steps:
```bash
python -m src.describe    # Table 1 + fig1
python -m src.model       # NB-GLM + XGBoost spatial CV
python -m src.causal      # DoubleML
python -m src.explain     # SHAP + fairness
python -m src.report      # forest plot + equity bars
python -m src.cds         # CDS demo
```

Smoke tests:
```bash
python -c "import importlib.util as u; s=u.spec_from_file_location('t','tests/test_pipeline.py'); m=u.module_from_spec(s); s.loader.exec_module(m); [f() for n in dir(m) for f in [getattr(m,n)] if n.startswith('test_')]; print('ok')"
```

---

## Methods summary (for the paper)

### Data
- **Source**: PRJ-6250 V2 "Maternal Health Equity Linked Datasets" (Colf et al., *Data in Brief* 2026, DOI 10.17603/ds2-62zy-zh93). Links HCUP HCUPnet 2017-2019 community inpatient statistics (county-level pregnancy-related CCSR aggregates), CDC/ATSDR SVI 2018, 2018 ACS 1-year estimates, Decennial 2020 PL 94-171 redistricting, Census Bureau Gazetteer 2019 county centroids, and a novel ArcGIS Pro origin-destination cost matrix of travel time to the nearest hospital at each of 4 OB care levels (AHA 2019 Annual Survey).
- **Cohort**: 2,054 U.S. counties in 37 states with non-missing SMM-proxy denominator and overall SVI percentile. State-total and US-total summary rows removed; -1 travel-time sentinels flagged and imputed at the state median.

### Outcomes
- Primary: **SMM-proxy rate per 1,000 pregnancy-related discharges**, defined as `1000 * (prg018 + prg019 + prg020 + prg023 + prg027) / sum(all 15 PRG categories)`. These five PRG categories map to CDC's 21 Severe Maternal Morbidity indicators via placental disorders, diabetic complications, severe hypertensive disease, intrapartum complications, and puerperal complications respectively.
- Secondary (individual): HDP (prg020), childbirth complications (prg023), placental disorders (prg018), puerperal complications (prg027), diabetes in pregnancy (prg019) - each expressed as rate per 1,000 PRG discharges.

### Exposure
- Continuous: `log(time3+1)` - natural log of travel time in minutes to the nearest ACOG Level III OB facility
- Binary: `gap30`, `gap60`, `gap90` - indicators for >30/60/90 min
- Ordinal: `loc_discordance_tier` in {0,1,2,3} for {<=30, 30-60, 60-90, >90} min
- Derived: `effective_access_tier_30min` (highest ACOG level reachable within 30 min)
- Composite: `maternity_desert = (time3 > 90) OR (time1 > 60)`

### Confounders
- SVI themes 1-4 (socioeconomic, household composition/disability, minority/language, housing/transportation)
- % Black, % AIAN, % Asian (from Decennial 2020 P1)
- `log(p1_001n)` population
- `reporting_completeness` (fraction of 15 PRG categories non-suppressed) to account for HCUP-suppression-related selection
- `is_level3_host` indicator (time3 <= 5 min) to flag regional referral centers

### Models
- **Negative Binomial GLM** (statsmodels) on SMM-proxy counts with `offset = log(denom)`
- **XGBoost** regressors for each rate outcome, 5-fold leave-state-out cross-validation
- **DoubleML partially-linear model** (Chernozhukov et al. 2018): Y and D residualized against X via cross-fitted XGBoost regressors/classifiers across state folds, then OLS of residual Y on residual D, with influence-function-based SE
- **Sensitivity analyses**: main / exclude Level-III host counties / rural subset (pop decile <= 5)

### Clinical scoring
- ACOG rule engine implements Committee Opinion 726 risk stratification
- Travel times used directly (no regression) to compute concordance gap
- XGBoost model outputs provide the auxiliary county-level risk context

---

## Limitations (be explicit in the paper)

1. **Discharge-based denominator**: HCUP counts discharges, not pregnancies or residents. Cross-county referral patterns induce the sign reversal between antepartum/intrapartum vs. postpartum outcomes that we document. The postpartum outcome is the most trustworthy signal.
2. **HCUP cell suppression** (<11 cases) removes the smallest counties for rarer outcomes (43% for placental, 48% for puerperal), biasing toward larger/more urban counties.
3. **Level IV not separable from Level III** in the source AHA survey; the CDS returns the same travel time for both.
4. **Cross-sectional 2017-2019** pre-pandemic window; the 2020-2025 wave of rural OB-unit closures is not captured.
5. **County centroid** distances approximate population-weighted distances (ArcGIS OD cost matrix), but for large counties with clustered population, true patient-specific distances will differ.
6. **No individual patient data**: the CDS recommendation relies on patient-reported risk factors and the county-of-residence assumption.

---

## Suggested citation

> Shrin [your name here], Colf LA, et al. MatLevel-CDS: operationalizing ACOG Levels of Maternal Care through county-level geographic access modeling. [journal TBD], 2026. Data: Colf et al., *Data in Brief* 2026, DOI 10.17603/ds2-62zy-zh93.
