"""
strategy/engine.py — FIXED
Fixes applied:
  [5.1] Rules generate FEATURES, not hard gate — ML decides final entry
  [5.2] Filters loosened; ML threshold is the primary filter
  [5.3] SL placed at sweep extreme ± 2×ATR buffer (not naked at liquidity)
  [5.4] RR is dynamic from signal quality, not fixed constant
  [7.1] ml_prob and rr passed to RiskManager for EV-based sizing
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SignalResult:
    timestamp:      pd.Timestamp
    direction:      int
    entry_price:    float
    sl_price:       float
    tp_price:       float
    ml_probability: float
    expected_value: float          # [4.1]
    rr_ratio:       float          # [5.4] dynamic
    rule_reason:    str


@dataclass
class StrategyConfig:
    ml_threshold:       float = 0.45    # model's own F1-optimal threshold
    min_ev:             float = 0.10    # minimum positive expected value
    sl_atr_mult:        float = 1.0
    sl_buffer_atr:      float = 0.8     # wider buffer on gold to avoid stop hunts
    rr_ratio:           float = 3.0     # 1:3 RR minimum
    require_htf_align:  bool  = True    # H4 trend must agree
    max_sweep_bos_gap:  int   = 100     # bars from BOS for pullback window (raised from 50)
    pullback_atr_min:   float = 0.1     # any pullback at all required
    pullback_atr_max:   float = 5.0     # allow deep pullbacks into discount zone


class StrategyEngine:

    def __init__(self, config: Optional[StrategyConfig] = None, model=None):
        self.cfg   = config or StrategyConfig()
        self.model = model

    # ── Batch scan ────────────────────────────────────────────────────────────

    def scan_all(self, df: pd.DataFrame) -> list:
        signals = []
        size    = len(df)
        cfg     = self.cfg

        close   = df["close"].values
        high    = df["high"].values
        low     = df["low"].values
        atr     = df["atr"].values
        bos_arr = df["bos"].values
        bull_sw = df["bull_sweep"].values
        bear_sw = df["bear_sweep"].values
        htf_tr  = df["htf_trend"].values if "htf_trend" in df.columns else np.zeros(size)

        last_bull_sweep_i = -1
        last_bear_sweep_i = -1
        last_bull_bos_i   = -1
        last_bear_bos_i   = -1
        last_bull_sweep_low  = np.nan   # [5.3] track extreme for SL placement
        last_bear_sweep_high = np.nan

        for i in range(1, size):
            prev = i - 1

            if bull_sw[prev]:
                last_bull_sweep_i   = prev
                last_bull_sweep_low = low[prev]   # [5.3] sweep low for SL
            if bear_sw[prev]:
                last_bear_sweep_i    = prev
                last_bear_sweep_high = high[prev]
            # Only record FIRST BOS after the sweep — prevents bos_close drifting upward
            # while price continues trending (which makes pullbacks measure near-zero).
            if bos_arr[prev] == 1  and last_bull_bos_i <= last_bull_sweep_i:
                last_bull_bos_i = prev
            if bos_arr[prev] == -1 and last_bear_bos_i <= last_bear_sweep_i:
                last_bear_bos_i = prev

            bar_atr = max(float(atr[i]), 1e-9)

            # Condition: BOS occurred after sweep AND pullback window (measured from BOS)
            if (
                last_bull_sweep_i >= 0
                and last_bull_bos_i > last_bull_sweep_i
                and (i - last_bull_bos_i) <= cfg.max_sweep_bos_gap
                and (not cfg.require_htf_align or htf_tr[i] >= 0)
            ):
                # Pullback is measured from the FIRST BOS close — how far has price
                # retraced below that level since the BOS confirmed?
                bos_close = close[last_bull_bos_i]
                pb_dist   = (bos_close - close[i]) / bar_atr

                if cfg.pullback_atr_min <= pb_dist <= cfg.pullback_atr_max:
                    ep = float(close[i])

                    # [5.3] SL below sweep extreme with buffer
                    if not np.isnan(last_bull_sweep_low):
                        sl = last_bull_sweep_low - cfg.sl_buffer_atr * bar_atr
                    else:
                        sl = ep - cfg.sl_atr_mult * bar_atr

                    sl_dist = ep - sl
                    if sl_dist <= 0:
                        last_bull_sweep_i = -1; last_bull_bos_i = -1
                        continue

                    # [5.4] Dynamic RR: scale with pullback quality
                    rr   = cfg.rr_ratio + max(0, pb_dist - 1.0) * 0.25
                    tp   = ep + sl_dist * rr
                    prob = self._ml_prob(df, i, direction=1)
                    ev   = prob * rr - (1 - prob)   # [4.1]

                    if prob >= cfg.ml_threshold and ev >= cfg.min_ev:
                        signals.append(SignalResult(
                            timestamp=df.index[i], direction=1,
                            entry_price=ep, sl_price=sl, tp_price=tp,
                            ml_probability=prob, expected_value=ev,
                            rr_ratio=rr, rule_reason="BullSweep+BOS+PB",
                        ))
                    last_bull_sweep_i = -1
                    last_bull_bos_i   = -1

            # ── SHORT ─────────────────────────────────────────────
            if (
                last_bear_sweep_i >= 0
                and last_bear_bos_i > last_bear_sweep_i
                and (i - last_bear_bos_i) <= cfg.max_sweep_bos_gap
                and (not cfg.require_htf_align or htf_tr[i] <= 0)
            ):
                bos_close = close[last_bear_bos_i]
                pb_dist   = (close[i] - bos_close) / bar_atr

                if cfg.pullback_atr_min <= pb_dist <= cfg.pullback_atr_max:
                    ep = float(close[i])

                    # [5.3] SL above sweep high with buffer
                    if not np.isnan(last_bear_sweep_high):
                        sl = last_bear_sweep_high + cfg.sl_buffer_atr * bar_atr
                    else:
                        sl = ep + cfg.sl_atr_mult * bar_atr

                    sl_dist = sl - ep
                    if sl_dist <= 0:
                        last_bear_sweep_i = -1; last_bear_bos_i = -1
                        continue

                    rr   = cfg.rr_ratio + max(0, pb_dist - 1.0) * 0.25
                    tp   = ep - sl_dist * rr
                    prob = self._ml_prob(df, i, direction=-1)
                    ev   = prob * rr - (1 - prob)

                    if prob >= cfg.ml_threshold and ev >= cfg.min_ev:
                        signals.append(SignalResult(
                            timestamp=df.index[i], direction=-1,
                            entry_price=ep, sl_price=sl, tp_price=tp,
                            ml_probability=prob, expected_value=ev,
                            rr_ratio=rr, rule_reason="BearSweep+BOS+PB",
                        ))
                    last_bear_sweep_i = -1
                    last_bear_bos_i   = -1

        logger.info(f"Strategy scan: {len(signals)} signals on {size:,} bars")
        return signals

    # ── Single bar (live) ─────────────────────────────────────────────────────

    def evaluate_bar(self, df: pd.DataFrame, idx: int) -> Optional[SignalResult]:
        cfg  = self.cfg
        size = len(df)
        if idx < 2:
            return None

        close   = df["close"].values
        high    = df["high"].values
        low     = df["low"].values
        atr_v   = df["atr"].values
        bos_arr = df["bos"].values
        bull_sw = df["bull_sweep"].values
        bear_sw = df["bear_sweep"].values
        htf_tr  = df["htf_trend"].values if "htf_trend" in df.columns else np.zeros(size)

        lookback = min(idx, cfg.max_sweep_bos_gap)

        def _last(arr, val, end):
            for j in range(end - 1, max(-1, end - lookback - 1), -1):
                if arr[j] == val:
                    return j
            return -1

        bull_sw_i = _last(bull_sw, 1, idx)
        bear_sw_i = _last(bear_sw, 1, idx)
        bar_atr   = max(float(atr_v[idx]), 1e-9)

        # LONG
        if bull_sw_i >= 0 and (not cfg.require_htf_align or htf_tr[idx] >= 0):
            bull_bos_i = _last(bos_arr, 1, idx)
            if bull_bos_i > bull_sw_i:
                pb = (close[bull_bos_i] - close[idx]) / bar_atr
                if cfg.pullback_atr_min <= pb <= cfg.pullback_atr_max:
                    ep      = float(close[idx])
                    sl      = low[bull_sw_i] - cfg.sl_buffer_atr * bar_atr  # [5.3]
                    sl_dist = ep - sl
                    if sl_dist > 0:
                        rr   = cfg.rr_ratio + max(0, pb - 1.0) * 0.25
                        tp   = ep + sl_dist * rr
                        prob = self._ml_prob(df, idx, direction=1)
                        ev   = prob * rr - (1 - prob)
                        if prob >= cfg.ml_threshold and ev >= cfg.min_ev:
                            return SignalResult(
                                df.index[idx], 1, ep, sl, tp,
                                prob, ev, rr, "BullSweep+BOS+PB"
                            )

        # SHORT
        if bear_sw_i >= 0 and (not cfg.require_htf_align or htf_tr[idx] <= 0):
            bear_bos_i = _last(bos_arr, -1, idx)
            if bear_bos_i > bear_sw_i:
                pb = (close[idx] - close[bear_bos_i]) / bar_atr
                if cfg.pullback_atr_min <= pb <= cfg.pullback_atr_max:
                    ep      = float(close[idx])
                    sl      = high[bear_sw_i] + cfg.sl_buffer_atr * bar_atr  # [5.3]
                    sl_dist = sl - ep
                    if sl_dist > 0:
                        rr   = cfg.rr_ratio + max(0, pb - 1.0) * 0.25
                        tp   = ep - sl_dist * rr
                        prob = self._ml_prob(df, idx, direction=-1)
                        ev   = prob * rr - (1 - prob)
                        if prob >= cfg.ml_threshold and ev >= cfg.min_ev:
                            return SignalResult(
                                df.index[idx], -1, ep, sl, tp,
                                prob, ev, rr, "BearSweep+BOS+PB"
                            )
        return None

    # ── ML helper ─────────────────────────────────────────────────────────────

    def _ml_prob(self, df: pd.DataFrame, i: int, direction: int = 0) -> float:
        if self.model is None:
            return 1.0
        try:
            row = df.iloc[[i]].copy()
            # [2.4] inject direction at inference time — not present in raw bar df
            row["direction"] = direction
            return float(self.model.predict_proba(row)[0])
        except Exception as exc:
            logger.warning(f"ML predict failed at bar {i}: {exc}")
            return 0.0