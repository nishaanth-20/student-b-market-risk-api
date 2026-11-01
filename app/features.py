"""Feature and MRI formulas used when serving.

They match the shared `at3` package in the group experiments repo, which was
used to build the training data, so live features are built the same way.

Conventions agreed by the group (point 4):
- every rolling window includes today (t)
- returns are decimals (0.05 = 5%)
- Volatility20 uses the sample standard deviation (ddof=1)
- GapRisk is the mean of |Open_t - Close_t-1| / Close_t-1 over the last 20 days

The input frame must have one row per calendar day (missing days as NaN rows),
so that a window never silently joins two sides of a data gap.
"""

import numpy as np
import pandas as pd


def to_daily_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by date and insert NaN rows for missing calendar days."""
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    return df.set_index("date").reindex(full).rename_axis("date").reset_index()


# ---------------------------------------------------------------- Student B
def add_own_features(df: pd.DataFrame) -> pd.DataFrame:
    """MA5, MA20 and TrendStrength (owned by Student B)."""
    out = df.copy()
    out["MA5"] = out["close"].rolling(5).mean()
    out["MA20"] = out["close"].rolling(20).mean()
    out["TrendStrength"] = out["MA5"] / out["MA20"] - 1
    return out


# -------------------------------------------------- Student B target formula
def market_risk_index(return20, volatility20, drawdown_risk, gap_risk):
    """MRI from the brief. Works on scalars, arrays or Series."""
    vrn = np.minimum(volatility20 / 0.10, 1)
    drn = np.minimum(drawdown_risk / 0.40, 1)
    rr = np.abs(np.minimum(return20, 0))
    rrn = np.minimum(rr / 0.30, 1)
    grn = np.minimum(gap_risk / 0.10, 1)
    return 100 * (0.40 * vrn + 0.30 * drn + 0.20 * rrn + 0.10 * grn)


def risk_level(score: float) -> str:
    """Plain-language band for the Streamlit page."""
    if score < 25:
        return "Low"
    if score < 50:
        return "Moderate"
    if score < 75:
        return "High"
    return "Very High"
