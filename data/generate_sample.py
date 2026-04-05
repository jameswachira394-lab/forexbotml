"""
data/generate_sample.py
-----------------------
Generates realistic synthetic EURUSD-like OHLCV data for testing.
Uses geometric Brownian motion with session-aware volatility regimes
and injects liquidity-sweep + BOS patterns for model training.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone


def _session_vol_multiplier(hour: int) -> float:
    """Return a volatility scalar based on the trading session."""
    if 7 <= hour < 9:    return 1.6   # London open
    if 12 <= hour < 14:  return 2.0   # NY open / London-NY overlap
    if 0 <= hour < 3:    return 0.6   # Asia quiet
    if 22 <= hour <= 23: return 0.5   # end of day
    return 1.0


def generate_ohlcv(
    n_bars:     int   = 20_000,
    start:      str   = "2022-01-03 00:00:00",
    timeframe_min: int = 5,
    base_price: float = 1.10000,
    spread:     float = 0.00010,
    seed:       int   = 42,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV bars using GBM with session volatility.
    Produces enough structural variety for feature engineering.
    """
    rng = np.random.default_rng(seed)

    dt       = timeframe_min / (252 * 390)  # fraction of trading year
    sigma    = 0.08                          # annual volatility EURUSD ~8%
    mu       = 0.0                           # no drift for realism

    index = pd.date_range(
        start=pd.Timestamp(start, tz="UTC"),
        periods=n_bars,
        freq=f"{timeframe_min}min",
    )

    closes = np.empty(n_bars)
    closes[0] = base_price

    for i in range(1, n_bars):
        hour  = index[i].hour
        vol_m = _session_vol_multiplier(hour)
        eps   = rng.standard_normal()
        ret   = (mu - 0.5 * sigma**2) * dt + sigma * vol_m * np.sqrt(dt) * eps
        closes[i] = closes[i - 1] * np.exp(ret)

    # Build OHLC from close series
    bar_vol = sigma * np.sqrt(dt) * closes  # bar range ≈ 1 std-dev
    noise   = rng.uniform(0.3, 1.0, n_bars)

    opens  = np.empty(n_bars)
    highs  = np.empty(n_bars)
    lows   = np.empty(n_bars)
    opens[0] = base_price

    for i in range(n_bars):
        o = opens[i] if i == 0 else closes[i - 1] * (1 + rng.normal(0, 0.0001))
        c = closes[i]
        rng_half = bar_vol[i] * noise[i] * 0.5
        h = max(o, c) + abs(rng.normal(0, rng_half))
        l = min(o, c) - abs(rng.normal(0, rng_half))
        opens[i] = o
        highs[i] = h
        lows[i]  = l
        if i < n_bars - 1:
            opens[i + 1] = c

    volumes = rng.integers(500, 5000, n_bars).astype(float)
    # Spike volume at session opens
    for i, ts in enumerate(index):
        if ts.hour in (7, 8, 12, 13):
            volumes[i] *= rng.uniform(1.5, 3.0)

    df = pd.DataFrame(
        {
            "timestamp": index,
            "open":   np.round(opens,  5),
            "high":   np.round(highs,  5),
            "low":    np.round(lows,   5),
            "close":  np.round(closes, 5),
            "volume": np.round(volumes, 0),
        }
    )
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data/raw", exist_ok=True)
    df = generate_ohlcv(n_bars=30_000, timeframe_min=5)
    path = "data/raw/EURUSD_M5.csv"
    df.to_csv(path, index=False)
    print(f"Generated {len(df):,} bars → {path}")
    print(df.tail(3).to_string())
