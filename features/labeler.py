"""
features/labeler.py
-------------------
Fixed-horizon, probabilistic labeling pipeline.

Instead of strictly defining a setup via SMC deterministic logic (Sweep -> Displacement -> CHoCH),
this labeler considers EVERY bar as a potential trade entry in both directions.

It applies fixed-horizon TP and SL multipliers relative to ATR.
Label Outcome simulation incorporates "worst-case" scenario assumption on same-candle hits.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

class LabelConfig:
    pass  # No longer used in favor of centralized config.py constants

class SetupLabeler:
    """
    Scans enriched DataFrame and labels EVERY bar for both long and short outcomes.
    Returns a DataFrame doubled in size where each row = one directional trade context.
    """

    def __init__(self, config=None):
        pass

    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        import config as cfg
        tp_mult  = getattr(cfg, "TP_ATR_MULT", 2.0)
        sl_mult  = getattr(cfg, "SL_ATR_MULT", 1.0)
        max_bars = getattr(cfg, "MAX_HOLD_BARS", 60)

        logger.info(f"Labeling all rows. TP: {tp_mult}x, SL: {sl_mult}x, Horizon: {max_bars} bars.")

        records = []
        size    = len(df)

        high  = df["high"].values
        low   = df["low"].values
        close = df["close"].values
        atr   = df["atr"].values

        # Fast pre-allocation
        timestamps = df.index
        
        # We only evaluate bars that have room to look ahead
        valid_idxs = range(size - max_bars)
        
        for i in valid_idxs:
            if atr[i] < 1e-9:
                continue
                
            row_base = df.iloc[i].to_dict()
            ts = timestamps[i]

            # ── LONG Outcome ──
            ep_long = float(close[i])
            sl_long = ep_long - sl_mult * atr[i]
            tp_long = ep_long + tp_mult * atr[i]
            
            label_long, duration_long = _simulate_outcome_horizon(
                direction=1, entry_idx=i, tp_price=tp_long, sl_price=sl_long,
                high=high, low=low, size=size, max_bars=max_bars
            )
            
            rec_long = row_base.copy()
            rec_long.update(
                timestamp=ts,
                direction=1,
                entry_price=ep_long,
                sl_price=sl_long,
                tp_price=tp_long,
                time_to_outcome=duration_long,
                label=1 if label_long == 1 else 0, # timeout or SL equals 0
                rr_actual=tp_mult / sl_mult if sl_mult > 0 else 0
            )
            records.append(rec_long)

            # ── SHORT Outcome ──
            ep_short = float(close[i])
            sl_short = ep_short + sl_mult * atr[i]
            tp_short = ep_short - tp_mult * atr[i]
            
            label_short, duration_short = _simulate_outcome_horizon(
                direction=-1, entry_idx=i, tp_price=tp_short, sl_price=sl_short,
                high=high, low=low, size=size, max_bars=max_bars
            )
            
            rec_short = row_base.copy()
            rec_short.update(
                timestamp=ts,
                direction=-1,
                entry_price=ep_short,
                sl_price=sl_short,
                tp_price=tp_short,
                time_to_outcome=duration_short,
                label=1 if label_short == 1 else 0,
                rr_actual=tp_mult / sl_mult if sl_mult > 0 else 0
            )
            records.append(rec_short)

        if not records:
            return pd.DataFrame()

        labeled = pd.DataFrame(records).set_index("timestamp")
        pos_rate = labeled["label"].mean()
        
        logger.info(
            f"Labeler complete: {len(labeled):,} context rows | "
            f"Overall TP hit rate: {pos_rate:.1%}"
        )
        return labeled


def _simulate_outcome_horizon(direction, entry_idx, tp_price, sl_price, high, low, size, max_bars):
    """
    Worst-case assumption: if candle hits both SL and TP, SL is triggered first.
    Return (outcome, duration).
    Outcome: 1 = TP, 0 = SL, -1 = timeout
    """
    end = min(size, entry_idx + max_bars + 1)
    for i in range(entry_idx + 1, end):
        duration = i - entry_idx
        if direction == 1:
            if low[i] <= sl_price:
                return 0, duration
            if high[i] >= tp_price:
                return 1, duration
        else:
            if high[i] >= sl_price:
                return 0, duration
            if low[i] <= tp_price:
                return 1, duration
                
    return -1, max_bars

# ──────────────────────────────────────────────────────────────
# Feature column list (used by model training)
# ──────────────────────────────────────────────────────────────

MODEL_FEATURES = [
    # Time / session
    "hour", "weekday", "session",
    "is_london_open", "is_ny_open", "mins_since_session_open",

    # Candle
    "body_pct", "body_ratio", "upper_wick", "lower_wick",
    "candle_dir", "range_expansion",
    "momentum_persistence",

    # Volatility
    "atr", "atr_percentile",

    # Market structure
    "hh", "hl", "lh", "ll", "bos", "trend", "choch",
    "dist_to_last_sh", "dist_to_last_sl",

    # Liquidity
    "eq_high", "eq_low",
    "dist_prev_50_high", "dist_prev_50_low",
    "dist_prev_day_high", "dist_prev_day_low",
    "bull_sweep", "bear_sweep", "sweep_strength",

    # Session H/L distances
    "asia_high_dist", "asia_low_dist",
    "london_high_dist", "london_low_dist",
    "ny_high_dist", "ny_low_dist",

    # FVG / Order Blocks precision metrics
    "fvg_bull", "fvg_bear", "fvg_size",
    "ob_size", "entry_in_discount", "impulse",

    # HTF
    "htf_trend",

    # Direction requested
    "direction",
]

def get_feature_columns(df: pd.DataFrame) -> list:
    """Return MODEL_FEATURES columns that actually exist in df."""
    return [c for c in MODEL_FEATURES if c in df.columns]
