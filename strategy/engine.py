"""
strategy/engine.py
------------------
Optimized O(n) rule-based strategy engine using rolling-state scan.
Detects: HTF trend alignment → liquidity sweep → BOS → pullback → ML gate.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SignalResult:
    timestamp:       pd.Timestamp
    direction:       int            # +1 LONG, -1 SHORT
    entry_price:     float
    sl_price:        float
    tp_price:        float
    ml_probability:  float
    rule_reason:     str


@dataclass
class StrategyConfig:
    ml_threshold:       float = 0.55
    sl_atr_mult:        float = 1.5
    rr_ratio:           float = 2.0
    require_htf_align:  bool  = False
    max_sweep_bos_gap:  int   = 30    # max bars between sweep and current bar
    pullback_atr_min:   float = 0.1
    pullback_atr_max:   float = 3.5


class StrategyEngine:
    """
    O(n) scanner: maintains rolling state for last seen sweep and BOS,
    fires a signal when pullback conditions are met.
    """

    def __init__(self, config: Optional[StrategyConfig] = None, model=None):
        self.cfg   = config or StrategyConfig()
        self.model = model

    # ── Batch scan (backtest) ─────────────────────────────────────────────────

    def scan_all(self, df: pd.DataFrame) -> list:
        """O(n) scan. Returns list[SignalResult]."""
        signals = []
        size    = len(df)
        cfg     = self.cfg

        close   = df["close"].values
        atr     = df["atr"].values
        bos_arr = df["bos"].values
        bull_sw = df["bull_sweep"].values
        bear_sw = df["bear_sweep"].values
        htf_tr  = df["htf_trend"].values if "htf_trend" in df.columns else np.zeros(size)

        last_bull_sweep_i = -1
        last_bear_sweep_i = -1
        last_bull_bos_i   = -1
        last_bear_bos_i   = -1

        for i in range(1, size):
            prev = i - 1

            # Update sweep / BOS memory from previous bar
            if bull_sw[prev]:  last_bull_sweep_i = prev
            if bear_sw[prev]:  last_bear_sweep_i = prev
            if bos_arr[prev] == 1:  last_bull_bos_i = prev
            if bos_arr[prev] == -1: last_bear_bos_i = prev

            bar_atr = max(float(atr[i]), 1e-9)

            # ── LONG setup ────────────────────────────────────────
            if (
                last_bull_sweep_i >= 0
                and last_bull_bos_i > last_bull_sweep_i
                and (i - last_bull_sweep_i) <= cfg.max_sweep_bos_gap
                and (not cfg.require_htf_align or htf_tr[i] >= 0)
            ):
                bos_close = close[last_bull_bos_i]
                pb_dist   = (bos_close - close[i]) / bar_atr

                if cfg.pullback_atr_min <= pb_dist <= cfg.pullback_atr_max:
                    ep   = float(close[i])
                    sl   = ep - cfg.sl_atr_mult * bar_atr
                    tp   = ep + abs(ep - sl) * cfg.rr_ratio
                    prob = self._ml_prob(df, i, direction=1)

                    if prob >= cfg.ml_threshold:
                        signals.append(SignalResult(
                            timestamp=df.index[i], direction=1,
                            entry_price=ep, sl_price=sl, tp_price=tp,
                            ml_probability=prob, rule_reason="BullSweep+BOS+PB",
                        ))
                    # Reset regardless of ML filter to avoid duplicate signals
                    last_bull_sweep_i = -1
                    last_bull_bos_i   = -1

            # ── SHORT setup ───────────────────────────────────────
            if (
                last_bear_sweep_i >= 0
                and last_bear_bos_i > last_bear_sweep_i
                and (i - last_bear_sweep_i) <= cfg.max_sweep_bos_gap
                and (not cfg.require_htf_align or htf_tr[i] <= 0)
            ):
                bos_close = close[last_bear_bos_i]
                pb_dist   = (close[i] - bos_close) / bar_atr

                if cfg.pullback_atr_min <= pb_dist <= cfg.pullback_atr_max:
                    ep   = float(close[i])
                    sl   = ep + cfg.sl_atr_mult * bar_atr
                    tp   = ep - abs(sl - ep) * cfg.rr_ratio
                    prob = self._ml_prob(df, i, direction=-1)

                    if prob >= cfg.ml_threshold:
                        signals.append(SignalResult(
                            timestamp=df.index[i], direction=-1,
                            entry_price=ep, sl_price=sl, tp_price=tp,
                            ml_probability=prob, rule_reason="BearSweep+BOS+PB",
                        ))
                    last_bear_sweep_i = -1
                    last_bear_bos_i   = -1

        logger.info(f"Strategy scan: {len(signals)} signals on {size:,} bars")
        return signals

    # ── Single bar (live) ─────────────────────────────────────────────────────

    def evaluate_bar(self, df: pd.DataFrame, idx: int) -> Optional[SignalResult]:
        """Evaluate only the last bar. Maintains no state — caller manages window."""
        cfg  = self.cfg
        size = len(df)
        if idx < 2:
            return None

        close   = df["close"].values
        atr_v   = df["atr"].values
        bos_arr = df["bos"].values
        bull_sw = df["bull_sweep"].values
        bear_sw = df["bear_sweep"].values
        htf_tr  = df["htf_trend"].values if "htf_trend" in df.columns else np.zeros(size)

        lookback = min(idx, cfg.max_sweep_bos_gap)

        def _last_idx_where(arr, val, end):
            for j in range(end - 1, end - lookback - 1, -1):
                if j >= 0 and arr[j] == val:
                    return j
            return -1

        bull_sw_i = _last_idx_where(bull_sw, 1, idx)
        bear_sw_i = _last_idx_where(bear_sw, 1, idx)

        bar_atr = max(float(atr_v[idx]), 1e-9)

        # LONG
        if bull_sw_i >= 0 and (not cfg.require_htf_align or htf_tr[idx] >= 0):
            bull_bos_i = _last_idx_where(bos_arr, 1, idx)
            if bull_bos_i > bull_sw_i:
                pb = (close[bull_bos_i] - close[idx]) / bar_atr
                if cfg.pullback_atr_min <= pb <= cfg.pullback_atr_max:
                    ep   = float(close[idx])
                    sl   = ep - cfg.sl_atr_mult * bar_atr
                    tp   = ep + abs(ep - sl) * cfg.rr_ratio
                    prob = self._ml_prob(df, idx, direction=1)
                    if prob >= cfg.ml_threshold:
                        return SignalResult(df.index[idx], 1, ep, sl, tp, prob, "BullSweep+BOS+PB")

        # SHORT
        if bear_sw_i >= 0 and (not cfg.require_htf_align or htf_tr[idx] <= 0):
            bear_bos_i = _last_idx_where(bos_arr, -1, idx)
            if bear_bos_i > bear_sw_i:
                pb = (close[idx] - close[bear_bos_i]) / bar_atr
                if cfg.pullback_atr_min <= pb <= cfg.pullback_atr_max:
                    ep   = float(close[idx])
                    sl   = ep + cfg.sl_atr_mult * bar_atr
                    tp   = ep - abs(sl - ep) * cfg.rr_ratio
                    prob = self._ml_prob(df, idx, direction=-1)
                    if prob >= cfg.ml_threshold:
                        return SignalResult(df.index[idx], -1, ep, sl, tp, prob, "BearSweep+BOS+PB")

        return None

    # ── ML helper ─────────────────────────────────────────────────────────────

    def _ml_prob(self, df: pd.DataFrame, i: int, direction: int) -> float:
        """Get model probability for the bar at index i, with a given direction."""
        if self.model is None:
            return 1.0
        try:
            # Create a 1-row DataFrame copy and inject metadata features known at entry-time
            row = df.iloc[[i]].copy()
            row["direction"] = direction
            
            return float(self.model.predict_proba(row)[0])
        except Exception as exc:
            logger.warning(f"ML predict failed at bar {i}: {exc}")
            return 0.0
