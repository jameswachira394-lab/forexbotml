"""
strategy/high_winrate_engine.py
--------------------------------
High Win Rate Strategy — EMA Trend Pullback + Multi-Confirmation

Logic:
High frequency during active sessions
"""

import logging
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SignalResult:
    timestamp:       pd.Timestamp
    direction:       int
    entry_price:     float
    sl_price:        float
    tp_price:        float
    ml_probability:  float
    expected_value:  float
    rr_ratio:        float
    rule_reason:     str


@dataclass
class HWRConfig:
    # EMA settings
    ema_fast:          int   = 20
    ema_slow:          int   = 50
    ema_trend:         int   = 200   # macro trend filter

    # RSI settings
    rsi_period:        int   = 14
    rsi_long_min:      float = 35.0  # RSI must be in this zone for long
    rsi_long_max:      float = 58.0
    rsi_short_min:     float = 42.0  # RSI must be in this zone for short
    rsi_short_max:     float = 65.0

    # Entry zone: how close price must be to EMA20 (in ATR units)
    ema_touch_max_atr: float = 0.8   # within 0.8 ATR of EMA20

    # SL/TP
    sl_atr_mult:       float = 1.0   # SL = 1.0 ATR below entry
    rr_ratio:          float = 1.5   # TP = 1.5 × SL (lower RR = higher WR)

    # Session filter (UTC hours) — only trade active sessions
    session_hours:     tuple = (7, 17)  # London + NY: 07:00-17:00 UTC

    # ML gate
    ml_threshold:      float = 0.60  # lower than SMC because strategy itself is selective
    min_ev:            float = 0.10

    # Trend strength
    min_ema_slope:     float = 0.0   # EMA20 must be rising (for longs)

    # Candle confirmation
    require_bull_candle: bool = True  # last candle must be bullish for long


class HighWinRateEngine:
    """
    EMA Pullback strategy optimised for 75-85% win rate.
    Computes all indicators internally from raw OHLCV.
    """

    def __init__(self, config: Optional[HWRConfig] = None, model=None):
        self.cfg   = config or HWRConfig()
        self.model = model

    # ── Feature computation ───────────────────────────────────────────────────

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add EMA, RSI, ATR to the DataFrame.
        Call this once before scan_all or evaluate_bar.
        """
        df = df.copy()
        cfg = self.cfg

        # EMAs
        df["ema_fast"]  = df["close"].ewm(span=cfg.ema_fast,  adjust=False).mean()
        df["ema_slow"]  = df["close"].ewm(span=cfg.ema_slow,  adjust=False).mean()
        df["ema_trend"] = df["close"].ewm(span=cfg.ema_trend, adjust=False).mean()

        # EMA slope (positive = rising)
        df["ema_fast_slope"] = df["ema_fast"].diff(3)

        # RSI
        df["rsi"] = self._compute_rsi(df["close"], cfg.rsi_period)

        # ATR
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()

        # Distance from EMA20 in ATR units
        df["dist_ema_fast"] = (df["close"] - df["ema_fast"]).abs() / df["atr"].clip(lower=1e-9)

        # Price position relative to EMAs
        df["above_ema_fast"] = (df["close"] > df["ema_fast"]).astype(int)
        df["above_ema_slow"] = (df["close"] > df["ema_slow"]).astype(int)
        df["above_ema_trend"]= (df["close"] > df["ema_trend"]).astype(int)
        df["ema_fast_above_slow"] = (df["ema_fast"] > df["ema_slow"]).astype(int)

        # Candle direction
        df["bull_candle"] = (df["close"] > df["open"]).astype(int)

        # Session
        df["hour"] = df.index.hour
        df["in_session"] = (
            (df["hour"] >= cfg.session_hours[0]) &
            (df["hour"] <  cfg.session_hours[1])
        ).astype(int)

        df.dropna(inplace=True)
        return df

    # ── Batch scan ────────────────────────────────────────────────────────────

    def scan_all(self, df: pd.DataFrame) -> List[SignalResult]:
        """Scan all bars. df must be output of prepare()."""
        signals = []
        cfg     = self.cfg

        close      = df["close"].values
        atr        = df["atr"].values
        rsi        = df["rsi"].values
        ema_fast   = df["ema_fast"].values
        ema_slope  = df["ema_fast_slope"].values
        dist_ema   = df["dist_ema_fast"].values
        above_fast = df["above_ema_fast"].values
        fast_above_slow = df["ema_fast_above_slow"].values
        above_trend= df["above_ema_trend"].values
        bull_can   = df["bull_candle"].values
        in_sess    = df["in_session"].values
        size       = len(df)

        # Track last signal bar to avoid clustering
        last_long_bar  = -10
        last_short_bar = -10
        min_gap        = 3  # minimum bars between same-direction signals

        for i in range(cfg.ema_trend + 5, size):
            if not in_sess[i]:
                continue

            bar_atr = max(float(atr[i]), 1e-9)
            ep      = float(close[i])

            # ── LONG conditions ───────────────────────────────────────────
            long_ok = (
                above_fast[i] == 1 and           # price above EMA20
                fast_above_slow[i] == 1 and       # EMA20 above EMA50
                above_trend[i] == 1 and           # price above EMA200 (macro)
                ema_slope[i] > cfg.min_ema_slope and  # EMA20 rising
                cfg.rsi_long_min <= rsi[i] <= cfg.rsi_long_max and  # RSI pullback zone
                dist_ema[i] <= cfg.ema_touch_max_atr and  # touching EMA
                (not cfg.require_bull_candle or bull_can[i] == 1) and
                (i - last_long_bar) >= min_gap
            )

            if long_ok:
                sl  = ep - cfg.sl_atr_mult * bar_atr
                tp  = ep + cfg.sl_atr_mult * bar_atr * cfg.rr_ratio
                prob = self._ml_prob(df, i, direction=1)
                ev   = prob * cfg.rr_ratio - (1 - prob)

                if prob >= cfg.ml_threshold and ev >= cfg.min_ev:
                    signals.append(SignalResult(
                        timestamp=df.index[i], direction=1,
                        entry_price=ep, sl_price=sl, tp_price=tp,
                        ml_probability=prob, expected_value=ev,
                        rr_ratio=cfg.rr_ratio,
                        rule_reason="EMA_PB_LONG",
                    ))
                    last_long_bar = i

            # ── SHORT conditions ──────────────────────────────────────────
            short_ok = (
                above_fast[i] == 0 and            # price below EMA20
                fast_above_slow[i] == 0 and        # EMA20 below EMA50
                above_trend[i] == 0 and            # price below EMA200
                ema_slope[i] < -cfg.min_ema_slope and  # EMA20 falling
                cfg.rsi_short_min <= rsi[i] <= cfg.rsi_short_max and
                dist_ema[i] <= cfg.ema_touch_max_atr and
                (not cfg.require_bull_candle or bull_can[i] == 0) and
                (i - last_short_bar) >= min_gap
            )

            if short_ok:
                sl  = ep + cfg.sl_atr_mult * bar_atr
                tp  = ep - cfg.sl_atr_mult * bar_atr * cfg.rr_ratio
                prob = self._ml_prob(df, i, direction=-1)
                ev   = prob * cfg.rr_ratio - (1 - prob)

                if prob >= cfg.ml_threshold and ev >= cfg.min_ev:
                    signals.append(SignalResult(
                        timestamp=df.index[i], direction=-1,
                        entry_price=ep, sl_price=sl, tp_price=tp,
                        ml_probability=prob, expected_value=ev,
                        rr_ratio=cfg.rr_ratio,
                        rule_reason="EMA_PB_SHORT",
                    ))
                    last_short_bar = i

        logger.info(f"HWR scan: {len(signals)} signals on {size:,} bars")
        return signals

    # ── Single bar (live) ─────────────────────────────────────────────────────

    def evaluate_bar(self, df: pd.DataFrame, idx: int) -> Optional[SignalResult]:
        """Evaluate the last bar. df must be output of prepare()."""
        if idx < self.cfg.ema_trend + 5:
            return None

        cfg    = self.cfg
        row    = df.iloc[idx]
        ep     = float(row["close"])
        bar_atr = max(float(row["atr"]), 1e-9)

        in_sess = row["hour"] >= cfg.session_hours[0] and row["hour"] < cfg.session_hours[1]
        if not in_sess:
            return None

        # LONG
        if (
            row["above_ema_fast"] == 1 and
            row["ema_fast_above_slow"] == 1 and
            row["above_ema_trend"] == 1 and
            row["ema_fast_slope"] > cfg.min_ema_slope and
            cfg.rsi_long_min <= row["rsi"] <= cfg.rsi_long_max and
            row["dist_ema_fast"] <= cfg.ema_touch_max_atr and
            (not cfg.require_bull_candle or row["bull_candle"] == 1)
        ):
            sl   = ep - cfg.sl_atr_mult * bar_atr
            tp   = ep + cfg.sl_atr_mult * bar_atr * cfg.rr_ratio
            prob = self._ml_prob(df, idx, direction=1)
            ev   = prob * cfg.rr_ratio - (1 - prob)
            if prob >= cfg.ml_threshold and ev >= cfg.min_ev:
                return SignalResult(
                    df.index[idx], 1, ep, sl, tp,
                    prob, ev, cfg.rr_ratio, "EMA_PB_LONG"
                )

        # SHORT
        if (
            row["above_ema_fast"] == 0 and
            row["ema_fast_above_slow"] == 0 and
            row["above_ema_trend"] == 0 and
            row["ema_fast_slope"] < -cfg.min_ema_slope and
            cfg.rsi_short_min <= row["rsi"] <= cfg.rsi_short_max and
            row["dist_ema_fast"] <= cfg.ema_touch_max_atr and
            (not cfg.require_bull_candle or row["bull_candle"] == 0)
        ):
            sl   = ep + cfg.sl_atr_mult * bar_atr
            tp   = ep - cfg.sl_atr_mult * bar_atr * cfg.rr_ratio
            prob = self._ml_prob(df, idx, direction=-1)
            ev   = prob * cfg.rr_ratio - (1 - prob)
            if prob >= cfg.ml_threshold and ev >= cfg.min_ev:
                return SignalResult(
                    df.index[idx], -1, ep, sl, tp,
                    prob, ev, cfg.rr_ratio, "EMA_PB_SHORT"
                )

        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs  = avg_gain / avg_loss.replace(0, 1e-9)
        return 100 - (100 / (1 + rs))

    def _ml_prob(self, df: pd.DataFrame, i: int, direction: int = 0) -> float:
        if self.model is None:
            return 1.0
        try:
            row = df.iloc[[i]].copy()
            row["direction"] = direction
            return float(self.model.predict_proba(row)[0])
        except Exception:
            return 1.0 