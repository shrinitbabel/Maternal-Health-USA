"""Project configuration: paths, constants, clinical thresholds."""
from __future__ import annotations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "PRJ-6250" / "Project--maternal-health-equity-linked-datasets--V2" / "data"
RAW_CSV = DATA_ROOT / "Maternal Health HCUP SV CENSUS TIME.csv"

OUTPUT_ROOT = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_ROOT / "figures"
TAB_DIR = OUTPUT_ROOT / "tables"
MODEL_DIR = OUTPUT_ROOT / "models"

# -------- Travel-time columns (novel variable from the dataset) --------
# time0: nearest hospital with NO maternal services
# time1: Level I  (OB Level 1) – uncomplicated maternal/newborn care
# time2: Level II (OB Level 2) – uncomplicated + most complicated
# time3: Level III (OB Level 3) – serious illnesses & abnormalities
TIME_COLS = ["time0", "time1", "time2", "time3"]
ACOG_LEVEL_OF = {"time0": 0, "time1": 1, "time2": 2, "time3": 3}

# Sentinel value used in the dataset for non-routable counties (islands etc.)
TIME_MISSING_SENTINEL = -1.0

# -------- PRG (pregnancy-related CCSR) outcome variables --------
# Source: HCUP HCUPnet, county-level 3-year aggregate 2017-2019.
ALL_PRG = [
    "prg011", "prg012", "prg013", "prg014", "prg016",
    "prg017", "prg018", "prg019", "prg020", "prg022",
    "prg023", "prg024", "prg026", "prg027", "prg029",
]
PRG_LABELS = {
    "prg011": "Early or threatened labor",
    "prg012": "Multiple gestation",
    "prg013": "Maternal care related to fetal conditions",
    "prg014": "Polyhydramnios and other amniotic cavity problems",
    "prg016": "Previous C-section",
    "prg017": "Maternal care for abnormality of pelvic organs",
    "prg018": "Placental disorders",
    "prg019": "Diabetes in pregnancy",
    "prg020": "Hypertensive disorders of pregnancy",
    "prg022": "Prolonged pregnancy",
    "prg023": "Complications specified during childbirth",
    "prg024": "Malposition / disproportion / labor complications",
    "prg026": "OB trauma to perineum/vulva",
    "prg027": "Complications specified during the puerperium",
    "prg029": "Uncomplicated pregnancy/delivery/puerperium",
}

# Severe Maternal Morbidity (SMM) *proxy* composite.
# These PRG categories map most closely onto CDC's 21 SMM indicators
# (severe preeclampsia / eclampsia -> prg020, hemorrhage / placental disorders
# -> prg018, severe diabetes with complications -> prg019, childbirth complications
# -> prg023, puerperal complications including sepsis/AFE/DIC -> prg027).
SMM_PROXY_PRG = ["prg018", "prg019", "prg020", "prg023", "prg027"]

# Total pregnancy-related discharges denominator (sum across complicated +
# uncomplicated categories). Used to build rate outcomes.
DENOM_PRG = ALL_PRG  # rate = sum(SMM_PROXY) / sum(ALL_PRG)

# -------- CDC Social Vulnerability Index themes (confounders) --------
SVI_THEMES = ["rpl_theme1", "rpl_theme2", "rpl_theme3", "rpl_theme4"]
SVI_THEME_LABELS = {
    "rpl_theme1": "Socioeconomic status",
    "rpl_theme2": "Household composition & disability",
    "rpl_theme3": "Minority status & language",
    "rpl_theme4": "Housing type & transportation",
}
SVI_OVERALL = "rpl_themes"

# A compact set of SVI raw percentages to use as interpretable confounders.
SVI_EP = [
    "ep_pov", "ep_unemp", "ep_nohsdp", "ep_age65", "ep_age17",
    "ep_disabl", "ep_sngpnt", "ep_minrty", "ep_limeng", "ep_munit",
    "ep_mobile", "ep_crowd", "ep_noveh", "ep_groupq",
]

# -------- Census / demographic exposures --------
# Total population from Decennial 2020 redistricting PL 94-171, table P1.
POP_TOTAL = "p1_001n"
# Decennial 2020 PL 94-171 Table P1 offsets verified against Table of Variables:
#   p1_003n White alone; p1_004n Black alone; p1_005n AIAN alone;
#   p1_006n Asian alone; p1_007n NHPI alone.
RACE_VARS = {
    "pct_white": "p1_003n",
    "pct_black": "p1_004n",
    "pct_aian":  "p1_005n",
    "pct_asian": "p1_006n",
    "pct_nhpi":  "p1_007n",
}

# -------- Clinical thresholds (from trauma systems / perinatal regionalization literature) --------
# 30 / 60 / 90 minute thresholds are conventional in the access literature.
ACCESS_THRESHOLDS_MIN = [30, 60, 90]

# -------- Modeling --------
RANDOM_STATE = 20260410
N_SPLITS_STATE_CV = 5  # leave-group-out by state for spatial CV
