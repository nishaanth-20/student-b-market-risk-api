# Student B – Market Risk Index (MRI) API

FastAPI service for AT3 (36120). It predicts the **Market Risk Index** of a cryptocurrency
**7 days ahead** with a LightGBM regressor, and serves Student B's engineered features
(MA5, MA20, TrendStrength) to the rest of the group.

| | |
|---|---|
| Group | Group 15 |
| Student B | Nishaanth Govindaraj 26044382 |
| Live API | https://student-b-market-risk-api.onrender.com (docs at `/docs`) |
| Data source | Kraken public OHLC API, daily candles, completed UTC days only |
| Tokens | BTC, ETH, XRP, SOL, DOGE, XMR, XLM, TRX |

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
  "prediction": 38.21,
  "risk_level": "Moderate",
  "forecast_horizon": 7,
  "date": "2026-10-24",
  "prediction_date": "2026-10-31",
  "current_mri": 35.9,
  "features_used": {"Return20": -0.04, "Volatility20": 0.021, "MaxPrice20": 71250.0, "GapRisk": 0.002, "DrawdownRisk": 0.06},
  "missing_features": [],
  "feature_sources": {"student_a": "ok", "student_c": "ok"},
  "warnings": []
}
```

Errors follow the group format: unsupported token, missing token or `forecast_horizon` other
than 7 → **422** `{"error": ..., "supported_tokens": [...]}`; Kraken unreachable or no model
file → **503** `{"error": ...}`.

## How a prediction is made

1. Download daily candles from Kraken, drop today's unfinished candle (`date` = last completed UTC day).
2. Call Student A and Student C's `/features` **at the same time** with a 10-second limit.
3. Model inputs: `Return20` (A), `Volatility20`, `MaxPrice20`, `GapRisk` (C), and
   `DrawdownRisk = (MaxPrice20 − Close) / MaxPrice20` (Student B's intermediate variable).
4. If a peer is down, times out, or returns a different `date`, its features are sent to the
   model as missing values. LightGBM routes missing values natively, and the model was trained
   with extra copies of rows where one peer's features were blanked, so it has learned how to do this.
5. The prediction is clipped to 0–100 and returned with HTTP 200.

## Target

```
DR  = (MaxPrice20 − Close_t) / MaxPrice20          RR  = |min(Return20, 0)|
VRN = min(Volatility20 / 0.10, 1)                   DRN = min(DR / 0.40, 1)
RRN = min(RR / 0.30, 1)                             GRN = min(GapRisk / 0.10, 1)
MRI = 100 × (0.40·VRN + 0.30·DRN + 0.20·RRN + 0.10·GRN)
```

The model learns features at day *t* → MRI at day *t + 7*.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Train the model

Put the provided dataset CSVs in a folder (filenames must contain the coin name or symbol,
e.g. `Bitcoin_...csv`), then:

```bash
python -m training.train_mri --data-dir path/to/dataset --trials 50
```

This writes `models/mri_lgbm.joblib`, `models/model_metadata.json` and
`models/training_report.json` (search space, best parameters, every Optuna trial, baselines,
per-token and missing-feature results for the report). Commit the `models/` folder so Render can load it.

Split (group agreement): train ≤ 2022, validation 2023, test 2024–2025, with the last 7 days
before each boundary dropped so no target leaks into the next split. Each token is put on a
daily calendar without filling gaps, so no window or target crosses a gap (e.g. XLM 2023).

### Run locally

```bash
export STUDENT_A_URL=https://<student-a>.onrender.com
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
  features.py    feature, target and MRI formulas (shared by training and serving)
  peers.py       parallel calls to Students A and C
models/          trained model and metadata
training/
  train_mri.py   LightGBM + Optuna training script
tests/
  test_api.py
```
