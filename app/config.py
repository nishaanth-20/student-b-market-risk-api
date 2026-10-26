"""Shared settings for the Student B (Market Risk Index) API.

Everything that other students also rely on (token list, feature names,
timeouts) lives here so the group agreement is written down in one place.
"""

import os

VERSION = "1.0.0"
FORECAST_HORIZON = 7

GITHUB_URL = os.getenv(
    "GITHUB_URL", "https://github.com/<your-username>/student-b-market-risk-api"
)

# Kraken daily candles (group agreement, point 1).
KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"
KRAKEN_INTERVAL = 1440  # minutes -> daily candles
KRAKEN_TIMEOUT = 15  # seconds
CACHE_SECONDS = 600  # reuse Kraken data for 10 minutes

# Kraken pair names. Bitcoin is XBT and Dogecoin is XDG on Kraken.
KRAKEN_PAIRS = {
    "BTC": "XBTUSD",
    "ETH": "ETHUSD",
    "XRP": "XRPUSD",
    "SOL": "SOLUSD",
    "DOGE": "XDGUSD",
    "XMR": "XMRUSD",
    "XLM": "XLMUSD",
    "TRX": "TRXUSD",
}
SUPPORTED_TOKENS = list(KRAKEN_PAIRS)

# Feature names, exact spelling (group agreement, point 3).
OWN_FEATURES = ["MA5", "MA20", "TrendStrength"]          # Student B
STUDENT_A_FEATURES = ["DailyReturn", "Return5", "Return20", "Momentum"]
STUDENT_C_FEATURES = ["Volatility20", "MaxPrice20", "MaxDrawdown", "GapRisk"]
STUDENT_D_FEATURES = ["VolumeMA20", "RelativeVolume"]

# Model inputs. The four required features come from Students A and C.
# DrawdownRisk is Student B's own intermediate variable, built from
# MaxPrice20 (Student C) and today's close (Kraken).
MODEL_FEATURES = ["Return20", "Volatility20", "MaxPrice20", "GapRisk", "DrawdownRisk"]

# Where each model input comes from, used to report missing sources.
FEATURE_SOURCE = {
    "Return20": "student_a",
    "Volatility20": "student_c",
    "MaxPrice20": "student_c",
    "GapRisk": "student_c",
    "DrawdownRisk": "student_c",
}

# Peer APIs (set these as environment variables on Render).
PEER_URLS = {
    "student_a": os.getenv("STUDENT_A_URL", "").rstrip("/"),
    "student_c": os.getenv("STUDENT_C_URL", "").rstrip("/"),
}
PEER_TIMEOUT = 10  # seconds, all peers called at the same time (point 7)

MODEL_PATH = os.getenv("MODEL_PATH", "models/mri_lgbm.joblib")
METADATA_PATH = os.getenv("METADATA_PATH", "models/model_metadata.json")
