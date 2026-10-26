"""Live daily candles from Kraken (no API key needed)."""

import time
from datetime import datetime, timezone

import httpx
import pandas as pd

from app.config import (
    CACHE_SECONDS,
    KRAKEN_INTERVAL,
    KRAKEN_OHLC_URL,
    KRAKEN_PAIRS,
    KRAKEN_TIMEOUT,
)

_cache: dict[str, tuple[float, pd.DataFrame]] = {}


class DataSourceError(Exception):
    """Kraken could not be reached or returned something unusable (-> 503)."""


def parse_kraken_ohlc(payload: dict) -> pd.DataFrame:
    """Turn Kraken's OHLC JSON into a frame of completed UTC days."""
    if payload.get("error"):
        raise DataSourceError(f"Kraken error: {', '.join(payload['error'])}")
    result = payload.get("result") or {}
    # The pair key varies (e.g. XXBTZUSD), so take the one that isn't "last".
    rows = next((v for k, v in result.items() if k != "last"), None)
    if not rows:
        raise DataSourceError("Kraken returned no candles")

    # Each row: [time, open, high, low, close, vwap, volume, count]
    df = pd.DataFrame(
        rows, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"]
    )
    df["date"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True).dt.tz_localize(None).dt.normalize()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # Keep completed days only: today's UTC candle is still forming.
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    df = df[df["date"] < today]
    if df.empty:
        raise DataSourceError("Kraken returned no completed days")
    return df[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def fetch_daily_candles(token: str) -> pd.DataFrame:
    """Completed daily candles for a token (about 720 days), cached briefly."""
    cached = _cache.get(token)
    if cached and time.time() - cached[0] < CACHE_SECONDS:
        return cached[1]

    pair = KRAKEN_PAIRS[token]
    try:
        resp = httpx.get(
            KRAKEN_OHLC_URL,
            params={"pair": pair, "interval": KRAKEN_INTERVAL},
            timeout=KRAKEN_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise DataSourceError(f"Could not reach Kraken for {pair}: {exc}") from exc

    df = parse_kraken_ohlc(payload)
    _cache[token] = (time.time(), df)
    return df
