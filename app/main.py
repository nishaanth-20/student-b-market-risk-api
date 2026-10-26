"""Student B - Market Risk Index (MRI) API.

Predicts the Market Risk Index of a cryptocurrency 7 days ahead with LightGBM,
and serves Student B's engineered features (MA5, MA20, TrendStrength).
"""

import json
import math
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import (
    FEATURE_SOURCE,
    FORECAST_HORIZON,
    GITHUB_URL,
    METADATA_PATH,
    MODEL_FEATURES,
    MODEL_PATH,
    PEER_URLS,
    SUPPORTED_TOKENS,
    VERSION,
)
from app.data import DataSourceError, fetch_daily_candles
from app.features import add_own_features, market_risk_index, risk_level, to_daily_calendar
from app.peers import as_float, fetch_peer_features

app = FastAPI(
    title="Market Risk Index API (Student B)",
    description="Predicts the Market Risk Index (MRI) of a cryptocurrency 7 days ahead.",
    version=VERSION,
)

# ------------------------------------------------------------------ model
model = None
metadata: dict = {}
if Path(MODEL_PATH).exists():
    model = joblib.load(MODEL_PATH)
if Path(METADATA_PATH).exists():
    metadata = json.loads(Path(METADATA_PATH).read_text())


# --------------------------------------------------------------- requests
class FeaturesRequest(BaseModel):
    token: str


class PredictRequest(BaseModel):
    token: str
    forecast_horizon: int = FORECAST_HORIZON


# ----------------------------------------------------------------- errors
def error_422(message: str) -> JSONResponse:
    """Agreed error format for bad requests (point 6)."""
    return JSONResponse(
        status_code=422,
        content={"error": message, "supported_tokens": SUPPORTED_TOKENS},
    )


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    # FastAPI's default is {"detail": [...]}; reshape it to the group format.
    first = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(p) for p in first.get("loc", []) if p != "body")
    msg = first.get("msg", "Invalid request")
    return error_422(f"Invalid request: {field} {msg}".strip())


@app.exception_handler(DataSourceError)
async def data_source_handler(request: Request, exc: DataSourceError):
    return JSONResponse(status_code=503, content={"error": str(exc)})


def check_token(token: str) -> str | None:
    token = token.strip().upper()
    return token if token in SUPPORTED_TOKENS else None


def clean(value):
    """NaN/inf -> None so the JSON shows null."""
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) or math.isinf(value) else value


def latest_own_features(token: str) -> dict:
    df = to_daily_calendar(fetch_daily_candles(token))
    df = add_own_features(df)
    last = df.iloc[-1]
    return {
        "token": token,
        "date": last["date"].strftime("%Y-%m-%d"),
        "close": clean(last["close"]),
        **{name: clean(last[name]) for name in ["MA5", "MA20", "TrendStrength"]},
    }


# -------------------------------------------------------------- endpoints
@app.get("/")
def root():
    return {
        "project": "Market Risk Index API",
        "student": "Student B",
        "description": (
            "Predicts the Market Risk Index (MRI, 0-100, higher = riskier) of a "
            "cryptocurrency 7 days into the future, and serves Student B's "
            "engineered features (MA5, MA20, TrendStrength)."
        ),
        "algorithm": "LightGBM Regressor",
        "forecast_horizon": FORECAST_HORIZON,
        "data_source": "Kraken public OHLC API, daily candles (completed UTC days only)",
        "supported_tokens": SUPPORTED_TOKENS,
        "version": VERSION,
        "github": GITHUB_URL,
        "endpoints": ["/", "/health", "/features", "/predict", "/model-metadata"],
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "peers_configured": {name: bool(url) for name, url in PEER_URLS.items()},
        "version": VERSION,
    }


@app.post("/features")
def features(req: FeaturesRequest):
    """Student B's features. Never calls another API (point 2)."""
    token = check_token(req.token)
    if token is None:
        return error_422("Unsupported cryptocurrency")
    feats = latest_own_features(token)
    feats.pop("close")
    return feats  # token, date, MA5, MA20, TrendStrength


@app.post("/predict")
async def predict(req: PredictRequest):
    token = check_token(req.token)
    if token is None:
        return error_422("Unsupported cryptocurrency")
    if req.forecast_horizon != FORECAST_HORIZON:
        return error_422(f"forecast_horizon must be {FORECAST_HORIZON}")
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model file not found. Train it with training/train_mri.py."},
        )

    own = latest_own_features(token)  # raises DataSourceError -> 503
    date = own["date"]
    close = own["close"]

    peers = await fetch_peer_features(token)
    a = peers["student_a"].get("data", {}) if peers["student_a"]["ok"] else {}
    c = peers["student_c"].get("data", {}) if peers["student_c"]["ok"] else {}

    # Only use peer values for the same market date as ours.
    warnings = []
    for name, data in (("student_a", a), ("student_c", c)):
        if data and data.get("date") != date:
            warnings.append(f"{name} returned date {data.get('date')}, expected {date}; ignored")
            data.clear()

    values = {
        "Return20": as_float(a.get("Return20")),
        "Volatility20": as_float(c.get("Volatility20")),
        "MaxPrice20": as_float(c.get("MaxPrice20")),
        "GapRisk": as_float(c.get("GapRisk")),
    }
    mp = values["MaxPrice20"]
    values["DrawdownRisk"] = (mp - close) / mp if mp and close is not None else None

    row = pd.DataFrame(
        [[np.nan if values[f] is None else values[f] for f in MODEL_FEATURES]],
        columns=MODEL_FEATURES,
    )
    score = float(np.clip(model.predict(row)[0], 0, 100))

    # Today's MRI from the formula, shown next to the forecast for context.
    current = None
    if all(values[k] is not None for k in ["Return20", "Volatility20", "DrawdownRisk", "GapRisk"]):
        current = round(float(market_risk_index(
            values["Return20"], values["Volatility20"], values["DrawdownRisk"], values["GapRisk"]
        )), 2)

    missing = [f for f in MODEL_FEATURES if values[f] is None]
    prediction_date = (pd.Timestamp(date) + timedelta(days=FORECAST_HORIZON)).strftime("%Y-%m-%d")

    return {
        "token": token,
        "prediction": round(score, 2),
        "risk_level": risk_level(score),
        "forecast_horizon": FORECAST_HORIZON,
        "date": date,
        "prediction_date": prediction_date,
        "current_mri": current,
        "features_used": {k: clean(v) for k, v in values.items()},
        "missing_features": missing,
        "feature_sources": {
            name: "ok" if res["ok"] else f"unavailable ({res['error']})"
            for name, res in peers.items()
        },
        "warnings": warnings,
    }


@app.get("/model-metadata")
def model_metadata():
    base = {
        "target": "Market Risk Index (MRI)",
        "target_description": (
            "MRI = 100 x (0.40 x VRN + 0.30 x DRN + 0.20 x RRN + 0.10 x GRN), "
            "where VRN = min(Volatility20/0.10, 1), DRN = min(DrawdownRisk/0.40, 1), "
            "RRN = min(|min(Return20, 0)|/0.30, 1), GRN = min(GapRisk/0.10, 1). "
            "Range 0-100, higher means riskier. The model predicts MRI at t+7."
        ),
        "algorithm": "LightGBM Regressor",
        "forecast_horizon": FORECAST_HORIZON,
        "features": MODEL_FEATURES,
        "feature_sources": FEATURE_SOURCE,
        "own_features": ["MA5", "MA20", "TrendStrength"],
        "risk_levels": {"Low": "0-25", "Moderate": "25-50", "High": "50-75", "Very High": "75-100"},
        "model_loaded": model is not None,
    }
    # Training results (feature importance, best params, metrics) from the training run.
    return {**base, **metadata}
