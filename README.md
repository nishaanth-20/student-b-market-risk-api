# Student B – Market Risk Index (MRI) API

FastAPI service for AT3 (36120). It predicts the **Market Risk Index** of a cryptocurrency
**7 days ahead** with a LightGBM regressor, and serves Student B's engineered features
(MA5, MA20, TrendStrength) to the rest of the group.

| | |
|---|---|
| Group | Group 15 |
| Student B | Nishaanth Govindaraj (26044382) |
| Live API | https://student-b-market-risk-api.onrender.com (docs at `/docs`) |
| Data source | Kraken public OHLC API, daily candles, completed UTC days only |
| Tokens | BTC, ETH, XRP, SOL, DOGE, XMR, XLM, TRX |
| Model training | Group experiments repo, notebook `36120_26SP_AT3_15_26044382_experiment.ipynb` |

## Endpoints

| Method | Path | What it returns |
|---|---|---|
| GET | `/` | Project overview, endpoint list, GitHub link |
| GET | `/health` | `status`, whether the model is loaded, which peer URLs are set |
| POST | `/features` | `{"token", "date", "MA5", "MA20", "TrendStrength"}` (never calls another API) |
| POST | `/predict` | MRI forecast for `date + 7` |
| GET | `/model-metadata` | Target, algorithm, horizon, features, feature importance, best params, test metrics |

Request body for `/features`: `{"token": "BTC"}`.
Request body for `/predict`: `{"token": "BTC", "forecast_horizon": 7}`.

Example `/predict` response:

```json
{
  "token": "BTC",
  "prediction": 16.82,
  "risk_level": "Low",
  "risk_warning_level": "Low",
  "forecast_horizon": 7,
  "date": "2026-10-29",
  "prediction_date": "2026-11-05",
  "current_mri": 19.93,
  "features_used": {"Return20": -0.08, "Volatility20": 0.03, "MaxPrice20": 165.0, "GapRisk": 0.001,
                    "DrawdownRisk": 0.05, "TrendStrength": 0.05},
  "missing_features": [],
  "feature_sources": {"student_a": "ok", "student_c": "ok"},
  "warnings": []
}
```

Errors follow the group format: unsupported token, missing token or `forecast_horizon` other
than 7 → **422** `{"error": ..., "supported_tokens": [...]}`; Kraken unreachable or no model
file → **503** `{"error": ...}`.

## How a prediction is made

1. Download daily candles from Kraken, drop today's unfinished candle (`date` = last completed UTC day),
   and compute my own TrendStrength.
2. Call Student A and Student C's `/features` **at the same time** with a 10-second limit.
3. Model inputs (order matters): `Return20` (A), `Volatility20`, `GapRisk` (C),
   `DrawdownRisk = (MaxPrice20 − Close) / MaxPrice20` (MaxPrice20 from C, close from Kraken),
   and `TrendStrength` (this API).
4. If a peer is down, times out, or returns a different `date`, its features are sent to the
   model as missing values. LightGBM routes missing values natively, and the model was trained
   with simulated outages of Student A and Student C, so it still gives a usable forecast.
   TrendStrength is always available.
5. The prediction is clipped to 0–100 and returned with HTTP 200, with:
   - `risk_level`: the band of the forecast (Low < 25 ≤ Moderate < 50 ≤ High < 75 ≤ Very High)
   - `current_mri`: today's MRI from the formula (only when A and C are both up)
   - `risk_warning_level`: the band of **max(today's MRI, forecast)**. The model tends to
     under-forecast rare high-risk weeks; this conservative rule misses fewer of them
     (46.7% vs 65.7% on the 2024–2025 test set, notebook section J.5).

## Model

| | |
|---|---|
| Algorithm | LightGBM regressor, L1 (absolute error) loss, tuned with Optuna (60 trials) |
| Features | Return20, Volatility20, GapRisk, DrawdownRisk, TrendStrength |
| Split | train ≤ 2022-12-31, validation 2023-01-08 to 2023-12-31, test from 2024-01-08 |
| Test MAE | **6.75** (no-change baseline 7.58), R² 0.56 |
| Most important | Volatility20 (48%), DrawdownRisk (25%), TrendStrength (16%) |

The full experiment (target construction, feature selection, outage simulation, Optuna search,
model selection and business impact) is in the group experiments repository. The files in
`models/` are copied from that notebook's output (`student_b_mri_lgbm.joblib` →
`mri_lgbm.joblib`, `student_b_mri_metadata.json` → `model_metadata.json`).

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run locally

```bash
export STUDENT_A_URL=https://<student-a>.onrender.com     # PowerShell: $env:STUDENT_A_URL = "..."
export STUDENT_C_URL=https://<student-c>.onrender.com
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs.

### Tests

```bash
python -m pytest -q
```

The tests replace Kraken and the peer APIs with fake responses, so they run offline.

## Deploy on Render

1. New → Web Service → connect this repo (or use `render.yaml` as a Blueprint).
2. Build command `pip install -r requirements.txt`, start command
   `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, health check path `/health`.
3. Environment variables: `PYTHON_VERSION=3.12.13`, `STUDENT_A_URL`, `STUDENT_C_URL`, `GITHUB_URL`.

## Project structure

```
app/
  main.py        endpoints
  config.py      tokens, Kraken pairs, feature names, peer URLs
  data.py        Kraken download and cleaning
  features.py    feature and MRI formulas
  peers.py       parallel calls to Students A and C
models/          trained model and metadata (from the experiments notebook)
tests/
  test_api.py
```
