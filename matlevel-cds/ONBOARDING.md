# MatLevel-CDS — Onboarding

Welcome. This document is for a second-year medical student joining the
project to help write the abstract and manuscript. You do not need to read
or run any code. Read this once, top to bottom, and then come back to
sections as you draft.

The live model is here: **[babels.ai/maternal](https://babels.ai/maternal)**.
Open it before you read further — clicking around the map for ten minutes is
the fastest way to get the gestalt of what we built.

---

## 1. The story

The U.S. has the highest maternal mortality rate of any high-income country,
and the deaths are not distributed evenly: rural, low-income, and Black/AIAN
communities carry disproportionate burden. ACOG and SMFM defined four
**Levels of Maternal Care** (I–IV) so that high-risk pregnancies can be
matched to facilities equipped to handle them. Level III/IV centers offer
maternal-fetal medicine, ICU, and advanced neonatal care; Levels I–II do
not.

Our clinical question: **how far do U.S. women live from the level of care
their pregnancy actually requires, and does that distance translate into
worse outcomes?**

We answered this at the county level for 2,054 counties (37 states,
256 million people) by linking four public datasets, fitting predictive
and causal models, and packaging the result as a point-of-care decision-
support tool that, given a patient's county and risk factors, returns the
recommended level of care, the actual drive time to the nearest facility
at that level, and an action bundle.

Two findings drive the manuscript:

1. **A clinical finding.** Rural counties carry roughly 2× the postpartum
   complication burden of non-rural counties and are roughly 2× as likely
   to be more than 90 minutes from a Level III facility. The two facts are
   linked.
2. **A methodological warning.** Discharge-based data (HCUP) systematically
   *under*-counts severe morbidity in remote counties because high-risk
   patients are transferred to regional centers and the receiving county
   "owns" the discharge. We name this the **referral-capture paradox**.
   Postpartum complications escape this bias because postpartum events
   present at the woman's local hospital — which is why the postpartum
   outcome is our most trustworthy signal.

---

## 2. Clinical background

These are the concepts that need to appear in the introduction and
discussion. Reviewers will probe them.

### Levels of Maternal Care (ACOG/SMFM)

| Level | Capabilities | Example patient |
|---|---|---|
| **I — Basic** | Uncomplicated term deliveries, basic emergencies, stabilize and transfer | Healthy multipara, low-risk pregnancy |
| **II — Specialty** | Level I + moderate-risk care, OB anesthesia, some MFM consultation | Well-controlled gestational DM, mild preeclampsia |
| **III — Subspecialty** | Level II + on-site MFM, adult ICU, blood bank, advanced imaging | Severe preeclampsia, placenta accreta, prior SMM |
| **IV — Regional perinatal center** | Level III + complex maternal medical/surgical care, ECMO | Pulmonary HTN, complex congenital heart disease in pregnancy |

Source: ACOG/SMFM Obstetric Care Consensus #2 (Levels of Maternal Care,
2015, reaffirmed 2019); ACOG Committee Opinion 726.

In our data, **Level III and Level IV are not separable** because the
source hospital survey does not distinguish them; we treat "Level III" as
"Level III or higher".

### The outcomes we measure

All outcomes are **county-level rates per 1,000 pregnancy-related (PRG)
discharges**.

- **SMM-proxy composite** — Severe Maternal Morbidity. CDC's 21-indicator
  list (eclampsia, AMI, sepsis, hysterectomy, transfusion, etc.).
- **HDP** — Hypertensive Disorders of Pregnancy: gestational HTN,
  preeclampsia, eclampsia, HELLP.
- **Childbirth complications** — intrapartum hemorrhage, obstructed labor,
  perineal trauma.
- **Placental disorders** — abruption, previa, accreta.
- **Puerperal (postpartum) complications** — postpartum hemorrhage
  requiring readmission, postpartum sepsis, postpartum VTE, retained
  products. **This is our most trustworthy outcome — see below.**
- **Diabetes in pregnancy** — gestational and pregestational.

### Why the postpartum window is special

Antepartum and intrapartum care can be regionalized: a high-risk patient
is referred to a Level III/IV center for delivery, and the discharge
(and the SMM/HDP code) is captured at the receiving hospital. Postpartum
events present locally — a woman discharged after delivery goes home, and
if she develops postpartum sepsis, hemorrhage, or VTE in the following six
weeks, she presents to her **local** ED. The discharge code is captured
at her county of residence.

This is why the puerperal outcome is the only one in which discharge-based
data faithfully reflects residential access, and why it is the outcome on
which travel time has its expected (positive) sign.

### Maternity desert

March of Dimes defines a maternity desert as a county with no obstetric
hospital, no birth center, and fewer than two OB providers per 10,000
women. Our analogous **operational** definition is access-to-care-level
based:

> `maternity_desert = (time to Level III > 90 min) OR (time to Level I > 60 min)`

About 35% of cohort counties meet this criterion.

---

## 3. The data

The single linked dataset that powers the project is **PRJ-6250**,
"Maternal Health Equity Linked Datasets" (Colf et al., *Data in Brief*
2026). The data descriptor PDF is in the project root
(`1-s2.0-S2352340926002660-main.pdf`) — read it first.

PRJ-6250 links four public sources at the U.S. county level for 2017–2019:

1. **HCUP State Inpatient Databases** — every inpatient discharge in
   participating states. We use the Pregnancy-Related (PRG) categorization
   to compute the outcomes above.
2. **CDC Social Vulnerability Index (SVI), 2018** — four themes
   (socioeconomic, household composition/disability, minority/language,
   housing/transportation) plus an overall percentile.
3. **U.S. Census ACS demographics** — population, age structure,
   race/ethnicity (including % AIAN), rurality decile.
4. **Travel time to the nearest hospital at each ACOG level** — computed
   in ArcGIS by the PRJ-6250 authors using the AHA Annual Survey to
   identify each hospital's level and the road network for drive time
   from the county centroid.

**Cohort:** 2,054 counties in 37 states. Counties drop out when their
state did not participate in HCUP for 2017–2019 or when SVI/centroid data
is missing.

**Caveat to mention in the limitations section.** HCUP suppresses any
county/category cell with fewer than 11 cases. For our two rarer outcomes
this is consequential — about 43% of cohort counties are suppressed for
placental disorders and 48% for puerperal. This biases the analytic frame
toward larger, more urban counties, which *attenuates* the rural
disadvantage we report (i.e., the true effect is likely larger than what
we measure).

---

## 4. The methods

You don't need to derive any of this. You need to be able to (a) say what
each method is doing in one sentence, (b) explain why we chose it, and
(c) handle basic reviewer questions.

### XGBoost predictive models

Gradient-boosted decision trees. Each tree is a piecewise-constant
function of the inputs; the algorithm fits trees sequentially, each one
trained on the residuals of the prior ensemble. The end result is a
flexible, non-linear regressor that handles interactions automatically.

We chose it because the relationship between SVI, demographics, and
maternal outcomes is not linear (e.g., minority composition interacts
with rurality), and tree ensembles do not require us to specify those
interactions in advance.

We use **leave-state-out cross-validation**: rather than randomly holding
out individual counties, we hold out entire states at a time. This
prevents within-state leakage — HCUP coding practices, Medicaid policy,
and OB workforce vary at the state level — and makes our reported model
performance a conservative estimate of out-of-state generalization.

### GraphSAGE graph neural network

A neural network that operates on a *graph* of counties, where edges
connect each county to its 8 nearest geographic neighbors. With two
layers, every county aggregates information from two hops out.

We use it because maternal outcomes and access patterns spill across
county lines (women cross county lines for OB care, especially in rural
areas), and a purely tabular model cannot see that. The GNN's smoothing
across neighbors meaningfully improves prediction on the postpartum and
HDP outcomes — the two outcomes where residential access is the dominant
signal — and *loses* on the SMM composite, which is the outcome most
contaminated by referral-capture. The selective gain is itself evidence
that residential access is the right signal.

### DoubleML causal model

The naive approach — regressing outcome on travel time with confounders
as covariates — is biased because high-risk areas get more hospitals,
hospitals are sited in cities, cities have lower SVI, and SVI predicts
outcomes. Confounding everywhere.

DoubleML (Chernozhukov et al. 2018) handles this by (1) training a model
to predict the outcome from confounders, (2) training a model to predict
the exposure (travel time) from confounders, (3) computing residuals
from both, and (4) regressing the residual outcome on the residual
exposure. The estimate is **doubly robust**: if either the outcome model
or the exposure model is well-specified, the causal estimate is
consistent. We use cross-fitted XGBoost as the nuisance learner for both
stages, and we cross-fit leave-state-out to prevent spatial leakage.

We run three sensitivity analyses for every outcome: main cohort,
excluding counties that *host* a Level III facility, and rural subsample
(population decile ≤ 5). Our main causal claim — that longer travel time
to Level III increases postpartum complications in rural counties —
survives all three sensitivities.

### The clinical decision support tool (CDS)

Given a county FIPS code and a patient's risk factors at the first
prenatal visit, the CDS returns:

- the ACOG-recommended **level of maternal care**
- the **actual drive time** to the nearest facility at each level
- a **concordance gap** (concordant / gap / severe gap)
- county context (SVI quartile, maternity-desert flag)
- model-predicted complication rates for the county
- an **action bundle** (MFM referral, social work, telehealth referral,
  postpartum warning-sign counseling, delivery-hospital pre-assignment)

The CDS is **not** a diagnostic tool. It is a workflow accelerant for the
first prenatal visit — it makes the geographic gap visible to the
clinician in seconds, in time to coordinate referrals.

You can try it live at **[babels.ai/maternal](https://babels.ai/maternal)**.
The site also exposes a **counterfactual hospital siting tool**: drop a
pin, pick a level, and the engine recomputes every county's travel time
and reports per-county deltas, population-weighted coverage gains, and
equity-by-SVI breakdowns. That tool is what makes the paper relevant to
state Title-V planning conversations — not just an academic exercise.

---

## 5. Glossary

- **ACOG / SMFM** — American College of Obstetricians and Gynecologists /
  Society for Maternal-Fetal Medicine.
- **SMM** — Severe Maternal Morbidity. CDC's 21-indicator list.
- **HDP** — Hypertensive Disorders of Pregnancy.
- **MFM** — Maternal-Fetal Medicine. Subspecialty trained to manage
  high-risk pregnancies; available at Level III/IV.
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
- **Maternity desert** — county with no/limited OB infrastructure; in
  this paper, operationally `time3 > 90 min OR time1 > 60 min`.
- **DoubleML** — causal-inference method that uses machine learning for
  nuisance functions and gives an interpretable scalar treatment effect.
- **Leave-state-out CV** — cross-validation in which entire states are
  held out, to prevent within-state leakage.
- **GNN / GraphSAGE** — graph neural network; updates each county's
  representation by aggregating information from its graph neighbors.
- **Counterfactual siting** — "if we placed a hospital here, who
  benefits?" simulation.
- **CDS** — Clinical Decision Support.

---

Welcome to the team.
