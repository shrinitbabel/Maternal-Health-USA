# Deploying MatLevel-CDS to Render

You already created the web service — here are the exact settings to fill in, plus what the two committed files (`render.yaml`, `api/main.py`) give you.

---

## 1. Service settings (copy into the Render dashboard)

| Field | Value |
|---|---|
| **Runtime / Environment** | `Python` |
| **Region** | whatever's closest to your Vercel edge (e.g. Oregon) |
| **Branch** | `main` |
| **Root Directory** | `matlevel-cds` |
| **Build Command** | `pip install --no-cache-dir -r requirements.txt` |
| **Start Command** | `uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1` |
| **Python Version** | `3.12` (set as env var `PYTHON_VERSION=3.12` — Render reads this) |
| **Health Check Path** | `/health` |
| **Instance Type** | Starter ($7/mo) recommended. Free tier works but cold-starts will be slow because the XGBoost pickle + CSV take ~15-25s to warm. |
| **Auto-Deploy** | On |

### Environment variables to add in the dashboard

| Key | Value | Notes |
|---|---|---|
| `PYTHON_VERSION` | `3.12` | pins the interpreter |
| `PYTHONUNBUFFERED` | `1` | flushes logs to Render immediately |
| `PYTHONIOENCODING` | `utf-8` | avoids Windows-style cp1252 errors in any print statements |

That's it. The commands above match `render.yaml` exactly, so if you ever switch to "Use render.yaml (Blueprint)", it picks up the same config.

---

## 2. Why **Root Directory = `matlevel-cds`** matters

Your GitHub repo is `Maternal Health USA` with this layout:

```
Maternal Health USA/              <-- repo root
├── PRJ-6250/                     <-- raw dataset (5 MB CSV)
├── matlevel-cds/                 <-- the Python project we just built
│   ├── api/
│   ├── src/
│   ├── outputs/models/xgb_models.pkl   <-- 7 MB, trained model
│   ├── requirements.txt
│   ├── render.yaml
│   └── ...
├── README.md
└── LICENSE
```

Setting Render's **Root Directory** to `matlevel-cds` means:
- `pip install -r requirements.txt` resolves to the right file
- `uvicorn api.main:app` finds the package
- `src/config.py`'s path logic walks **up one level** to find `PRJ-6250/...` (via `PROJECT_ROOT.parent`), which is exactly `Maternal Health USA/PRJ-6250/` — the CSV is found on the very first candidate and the loader boots.

If you ever move the data folder, set `MATLEVEL_DATA_ROOT` as an env var and the loader will use that instead (fallback chain is documented in `src/config.py`).

---

## 3. Endpoints the Vercel frontend will call

All already live, tested with `httpx.TestClient` end-to-end:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | service banner (sanity) |
| `GET` | `/health` | liveness + readiness. Returns `{status, version, counties_loaded, model_loaded}`. **Render's health check hits this.** |
| `GET` | `/counties` | all 2,054 counties: `[{fips, county, state_fips}, ...]` |
| `GET` | `/counties/{state}` | same, filtered to a 2-digit state FIPS (e.g. `21` for KY) |
| `GET` | `/access/{fips}` | county access & SVI context + model-scored county complication rates, **no patient data required** — ideal for the map popover |
| `POST` | `/recommend` | full CDS recommendation. Body: `{fips, risk: {...}}` |
| `GET` | `/docs` | auto-generated Swagger UI (FastAPI) |
| `GET` | `/openapi.json` | OpenAPI schema (useful for the Vercel frontend client) |

### Example `POST /recommend` body
```json
{
  "fips": "21203",
  "risk": {
    "chronic_hypertension": true,
    "pregestational_diabetes": true,
    "prior_severe_maternal_morbidity": true,
    "bmi": 41,
    "age_years": 38
  }
}
```

### Example `POST /recommend` response
```json
{
  "fips": "21203",
  "county": "Rockcastle",
  "state_fips": "21",
  "patient_risk_factors": [
    "Prior severe maternal morbidity",
    "Chronic hypertension",
    "Pregestational diabetes",
    "Class-III obesity (BMI 41.0)"
  ],
  "recommended_level": "LEVEL_III",
  "recommended_level_int": 3,
  "travel_times_min": {"LEVEL_I": 29.3, "LEVEL_II": 54.4, "LEVEL_III": 67.4, "LEVEL_IV": 67.4},
  "closest_within_30min": "LEVEL_I",
  "closest_within_60min": "LEVEL_II",
  "concordance": "severe_gap",
  "gap_minutes": 37.4,
  "svi_percentile": 0.788,
  "svi_quartile": 4,
  "maternity_desert": false,
  "is_level3_host": false,
  "county_model_scores": {
    "smm_rate_per1000": 342.0,
    "hdp_rate_per1000": 115.06,
    "puerperal_rate_per1000": 21.76
  },
  "action_bundle": [
    "Refer to Maternal-Fetal Medicine at a Level III/IV center for co-management",
    "Coordinate delivery-hospital assignment by 28 weeks",
    "High social vulnerability quartile: engage social work early for transportation, housing, and insurance barriers",
    "Counsel on recognition and immediate presentation for postpartum warning signs"
  ],
  "disclaimer": "..."
}
```

---

## 4. CORS

`api/main.py` currently allows `*` origins (convenient for first deploy + Swagger UI). **Before you wire the Vercel frontend, pin the origin** to your domain:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-site.vercel.app", "https://www.yourdomain.com"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

Or read it from an env var so you don't have to redeploy on domain changes.

---

## 5. After you hit "Manual Deploy"

Watch the logs for:
1. `pip install` succeeds (~2-3 min the first time; `xgboost` + `shap` are heavy)
2. `Uvicorn running on http://0.0.0.0:PORT`
3. `[ok] ...` startup messages from the CDS cache warm
4. Render marks the service healthy (green) once `/health` returns 200

Then hit:
```
https://matlevel-cds-api.onrender.com/health
https://matlevel-cds-api.onrender.com/docs
https://matlevel-cds-api.onrender.com/counties/21
```

and you're live.

---

## 6. Known caveats on Render

- **Cold-start latency on free tier**: the pkl model is 7 MB and pandas loads a 5 MB CSV on first request — expect ~15-25 s warmup. The `@app.on_event("startup")` handler already pre-warms both caches before Render runs the health check, so users never see the cold start.
- **Ephemeral disk**: Render containers have ephemeral `/tmp`. We don't write to disk at runtime, so this doesn't bite us.
- **Single worker**: `--workers 1` intentional. Multi-worker would load the 7 MB model once per worker and balloon the memory footprint. If you outgrow 1 worker, move to the Standard plan (512 MB → 2 GB).
- **Model drift**: there is none yet — this service serves a frozen cohort (HCUP 2017-2019). When you refresh the data in a future release, retrain `xgb_models.pkl` locally via `python -m src.model`, commit it, and Render auto-deploys.
