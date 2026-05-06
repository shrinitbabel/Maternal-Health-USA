# Maternal Health USA 

The live tool is at **[babels.ai/maternal](https://babels.ai/maternal)**.
Click around the map for ten minutes before reading further — that's the fastest way to get the gist of what we built.

---

## 1. The story

The U.S. has the highest maternal mortality rate of any high-income
country, and the deaths are not distributed evenly: rural, low-income,
and certain communities carry disproportionate burden. ACOG (American College of Obstetrics and Gynecology) and SMFM (Society of Maternal-Fetal Medicine) defined four **Levels of Maternal Care** (I–IV) so that high-risk
pregnancies can be matched to facilities equipped to handle them. Level III/IV centers are advanced OBGYN centers that offer maternal-fetal medicine, ICU, and advanced neonatal care; Levels I–II do not.

Our clinical question: **how far do U.S. women live from the level of
care their pregnancy actually requires, and does that distance translate into worse outcomes?**

We answered this at the county level for 2,054 counties (37 states,
256 million people) by linking four public datasets, fitting predictive and causal models, and packaging the result as a point-of-care decision-support tool that — given a patient's county and risk factors — returns the recommended level of care, the drive time to the nearest facility at that level, and an action bundle.

Two findings drive the manuscript:

1. **Clinical.** Rural counties carry roughly 2× the postpartum
   complication burden of non-rural counties and are roughly 2× as
   likely to be more than 90 minutes from a Level III facility. The two
   facts are linked.
2. **Methodological.** Discharge-based data (HCUP) systematically
   *under*-counts severe morbidity in remote counties because high-risk patients are transferred to regional centers and the receiving county "owns" the discharge. We name this the **referral-capture paradox**.
   Postpartum complications escape this bias because postpartum events
   present at the woman's local hospital — which is why the postpartum
   outcome is our most trustworthy signal.

---

## 2. Clinical background

### Levels of Maternal Care (ACOG/SMFM)

| Level | Capabilities | Example patient |
|---|---|---|
| **I — Basic** | Uncomplicated term deliveries, basic emergencies, stabilize and transfer | Healthy multipara, low-risk pregnancy |
| **II — Specialty** | Level I + moderate-risk care, OB anesthesia, some MFM consultation | Well-controlled gestational DM, mild preeclampsia |
| **III — Subspecialty** | Level II + on-site MFM, adult ICU, blood bank, advanced imaging | Severe preeclampsia, placenta accreta, prior SMM |
| **IV — Regional perinatal center** | Level III + complex maternal medical/surgical care, ECMO | Pulmonary HTN, complex congenital heart disease in pregnancy |

Source: ACOG/SMFM Obstetric Care Consensus #2 (2015, reaffirmed 2019);
ACOG Committee Opinion 726. In our data, **Level III and IV are not
separable** because the source hospital survey doesn't distinguish them; we treat "Level III" as "Level III or higher".

### The outcomes we measure

All are **county-level rates per 1,000 pregnancy-related (PRG) discharges**.

- **SMM-proxy** — Severe Maternal Morbidity (CDC's 21-indicator list:
  eclampsia, AMI, sepsis, hysterectomy, transfusion, etc.).
- **HDP** — Hypertensive Disorders of Pregnancy: gestational HTN,
  preeclampsia, eclampsia, HELLP.
- **Childbirth complications** — intrapartum hemorrhage, obstructed
  labor, perineal trauma.
- **Placental disorders** — abruption, previa, accreta.
- **Puerperal (postpartum) complications** — postpartum hemorrhage
  readmission, postpartum sepsis, postpartum VTE, retained products.
  **Our most trustworthy outcome (see below).**
- **Diabetes in pregnancy** — gestational and pregestational.

### Why the postpartum window is special

Antepartum and intrapartum care can be regionalized — a high-risk
patient is referred to a Level III/IV center for delivery, and the
discharge code is captured at the receiving hospital. Postpartum events present locally: a woman discharged after delivery goes home, and if she develops postpartum sepsis, hemorrhage, or VTE in the next six weeks, she presents to her **local** ED. The discharge code is captured at her county of residence. That's why the puerperal outcome is the only one in which discharge-based data faithfully reflects residential access.

### Maternity desert

Our operational definition is access-to-care-level based:
`time to Level III > 90 min OR time to Level I > 60 min`. About 35% of
cohort counties meet it.

---

## 3. The data

The single linked dataset that powers the project is **PRJ-6250 —
Maternal Health Equity Linked Datasets** (Colf et al., *Data in Brief*
2026). The data descriptor PDF is in the project root —
read it first.

PRJ-6250 links four public sources at the U.S. county level for 2017–2019:

1. **HCUP State Inpatient Databases** — every inpatient discharge in
   participating states. We use the Pregnancy-Related (PRG) categorization
   for the outcomes above.
2. **CDC Social Vulnerability Index (SVI), 2018** — four themes
   (socioeconomic, household, minority/language, housing/transportation)
   plus an overall percentile.
3. **U.S. Census ACS demographics** — population, age, race/ethnicity
   (including % AIAN), rurality decile.
4. **Travel time to the nearest hospital at each ACOG level** — computed
   in ArcGIS by the PRJ-6250 team using the AHA Annual Survey to identify
   each hospital's level and the road network for drive time from the
   county centroid.

**Cohort:** 2,054 counties in 37 states.

**Caveat for the limitations section.** HCUP suppresses any county/category
cell with fewer than 11 cases. About 43% of cohort counties are suppressed
for placental disorders and 48% for puerperal — biasing the analytic
frame toward larger, more urban counties, which *attenuates* the rural
disadvantage we report.

---

## 4. The methods

You don't need to derive any of this. You need to (a) say what each
method does in one sentence, (b) explain why we chose it, and (c) handle
basic reviewer questions.

### XGBoost predictive models

Gradient-boosted decision trees: the algorithm fits trees sequentially,
each one trained on the residuals of the prior ensemble. Result is a
flexible, non-linear regressor that handles interactions automatically —
which we want, because the relationship between SVI, demographics, and
maternal outcomes isn't linear.

We use **leave-state-out cross-validation**: hold out entire states at
a time rather than individual counties. This prevents within-state
leakage (HCUP coding, Medicaid policy, and OB workforce vary by state)
and makes our reported performance a conservative estimate of
out-of-state generalization.

### GraphSAGE graph neural network

A neural network that operates on a *graph* of counties, where edges
connect each county to its 8 nearest geographic neighbors. With two
layers, every county aggregates information from two hops out.

We use it because maternal outcomes and access patterns spill across
county lines (women cross county lines for OB care, especially in rural areas) and a tabular model can't see that. The GNN improves prediction on postpartum and HDP outcomes — the two where residential access is the dominant signal — and *loses* on the SMM composite, which is the outcome most contaminated by referral-capture. The selective gain is itself evidence that residential access is the right signal.

### DoubleML causal model

The naive approach — regressing outcome on travel time with confounders
as covariates — is biased because high-risk areas get more hospitals,
hospitals are sited in cities, cities have lower SVI, and SVI predicts
outcomes. Confounding everywhere.

DoubleML (Chernozhukov et al. 2018) handles this by (1) training a
model to predict the outcome from confounders, (2) training a model to
predict the exposure (travel time) from confounders, (3) computing
residuals from both, and (4) regressing residual outcome on residual
exposure. The estimate is **doubly robust** — if either nuisance model
is well-specified, the causal estimate is consistent. We use cross-fitted
XGBoost for both nuisance stages and cross-fit leave-state-out to
prevent spatial leakage.

We run three sensitivity analyses for every outcome — main cohort,
excluding Level III host counties, and rural subsample (population
decile ≤ 5). Our main causal claim (longer travel to Level III →
more postpartum complications in rural counties) survives all three.

### The clinical decision support tool (CDS)

Given a county FIPS and the patient's risk factors at the first prenatal visit, the CDS returns:

- the ACOG-recommended **level of maternal care**
- the **actual drive time** to the nearest facility at each level
- a **concordance gap** (concordant / gap / severe gap)
- county context (SVI quartile, maternity-desert flag)
- model-predicted complication rates for the county
- an **action bundle** (MFM referral, social work, telehealth, postpartum
  warning-sign counseling, delivery-hospital pre-assignment)

Try it: **[babels.ai/maternal](https://babels.ai/maternal)**.

---

## 5. Glossary

- **ACOG / SMFM** — American College of Obstetricians and Gynecologists
  / Society for Maternal-Fetal Medicine.
- **SMM** — Severe Maternal Morbidity. CDC's 21-indicator list.
- **HDP** — Hypertensive Disorders of Pregnancy.
- **MFM** — Maternal-Fetal Medicine. Subspecialty for high-risk
  pregnancy; available at Level III/IV.
- **HCUP** — Healthcare Cost and Utilization Project (AHRQ). National
  hospital discharge data.
- **PRG categories** — HCUP's pregnancy-related diagnostic categories.
- **SVI** — Social Vulnerability Index (CDC/ATSDR). Composite of
  socioeconomic, household, minority/language, and housing/transportation
  themes. Higher = more vulnerable.
- **AIAN** — American Indian / Alaska Native.
- **time3** — drive time in minutes from the county centroid to the
  nearest Level III OB facility.
- **Concordance / gap / severe gap** — relationship between the level a
  patient *should* deliver at vs. the level she can actually *reach*
  within a given drive time.
- **Maternity desert** — county with no/limited OB infrastructure; here,
  `time3 > 90 min OR time1 > 60 min`.
- **DoubleML** — causal-inference method that uses ML for nuisance
  functions and gives an interpretable scalar treatment effect.
- **Leave-state-out CV** — cross-validation in which entire states are
  held out, to prevent within-state leakage.
- **GNN / GraphSAGE** — graph neural network; updates each county's
  representation by aggregating information from its graph neighbors.
- **Counterfactual siting** — "if we placed a hospital here, who
  benefits?" simulation.
- **CDS** — Clinical Decision Support.

