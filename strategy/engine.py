"""
strategy/engine.py
------------------
Probabilistic strategy engine.
Replaces deterministic SMC boolean logic with pure Expected Value (EV) evaluation
driven by the ML model's probabilities.
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
    direction:       int            # +1 LONG, -1 SHORT
    entry_price:     float
    sl_price:        float
    tp_price:        float
    ml_probability:  float
    expected_value:  float
    rule_reason:     str


class StrategyEngine:
    """
    Evaluates every bar. Uses batch ML prediction to get P(win) for longs and shorts.
    Calculates Expected Value (EV). Triggers tradeoff if EV > 0.
    """

    def __init__(self, config=None, model=None):
        import config as cfg
        self.tp_mult = getattr(cfg, "TP_ATR_MULT", 2.0)
        self.sl_mult = getattr(cfg, "SL_ATR_MULT", 1.0)
        # Add slight buffer above 0 EV to overcome spread/fees
        self.min_ev  = getattr(cfg, "MIN_EV", 0.05) 
        self.model   = model

    # ── Batch scan (backtest/walkforward) ─────────────────────────────────────────────────

    def scan_all(self, df: pd.DataFrame) -> List[SignalResult]:
        """O(n) scan using batched ML probabilities for speed."""
        signals = []
        size    = len(df)
        
        if size == 0:
            return signals

        close = df["close"].values
        atr   = df["atr"].values
        htf_tr = df["htf_trend"].values if "htf_trend" in df.columns else np.zeros(size)

        # Batch predict Longs
        df_long = df.copy()
        df_long["direction"] = 1
        probs_long = self._batch_ml_prob(df_long)

        # Batch predict Shorts
        df_short = df.copy()
        df_short["direction"] = -1
        probs_short = self._batch_ml_prob(df_short)

        for i in range(1, size):
            if atr[i] < 1e-9:
                continue

            bar_atr = float(atr[i])
            ep      = float(close[i])

            # ── LONG Evaluation ──
            prob_L = probs_long[i]
            ev_L   = (prob_L * self.tp_mult) - ((1.0 - prob_L) * self.sl_mult)

            # Trend filter (optional: rely entirely on ML or keep light structural guidance)
            if ev_L > self.min_ev and htf_tr[i] >= 0:
                sl = ep - self.sl_mult * bar_atr
                tp = ep + self.tp_mult * bar_atr
                signals.append(SignalResult(
                    timestamp=df.index[i], direction=1,
                    entry_price=ep, sl_price=sl, tp_price=tp,
                    ml_probability=prob_L, expected_value=ev_L,
                    rule_reason="+EV Long"
                ))

            # ── SHORT Evaluation ──
            prob_S = probs_short[i]
            ev_S   = (prob_S * self.tp_mult) - ((1.0 - prob_S) * self.sl_mult)

            if ev_S > self.min_ev and htf_tr[i] <= 0:
                sl = ep + self.sl_mult * bar_atr
                tp = ep - self.tp_mult * bar_atr
                signals.append(SignalResult(
                    timestamp=df.index[i], direction=-1,
                    entry_price=ep, sl_price=sl, tp_price=tp,
                    ml_probability=prob_S, expected_value=ev_S,
                    rule_reason="+EV Short"
                ))

        logger.info(f"Strategy scan: {len(signals)} +EV signals generated over {size:,} bars")
        return signals

    # ── Single bar (live) ─────────────────────────────────────────────────────

    def evaluate_bar(self, df: pd.DataFrame, idx: int) -> Optional[SignalResult]:
        """Evaluate only the last bar (used in live execution)."""
        if idx < 0 or idx >= len(df):
            return None

        # Require a valid ATR
        bar_atr = df["atr"].iloc[idx]
        if bar_atr < 1e-9:
            return None
            
        ep = float(df["close"].iloc[idx])
        htf_tr = df["htf_trend"].iloc[idx] if "htf_trend" in df.columns else 0

        # Evaluate Long
        prob_L = self._single_ml_prob(df, idx, direction=1)
        ev_L   = (prob_L * self.tp_mult) - ((1.0 - prob_L) * self.sl_mult)

        # Evaluate Short
        prob_S = self._single_ml_prob(df, idx, direction=-1)
        ev_S   = (prob_S * self.tp_mult) - ((1.0 - prob_S) * self.sl_mult)

        # Return the best option if it meets criteria
        if ev_L > self.min_ev and ev_L > ev_S and htf_tr >= 0:
            sl = ep - self.sl_mult * bar_atr
            tp = ep + self.tp_mult * bar_atr
            return SignalResult(
                timestamp=df.index[idx], direction=1,
                entry_price=ep, sl_price=sl, tp_price=tp,
                ml_probability=prob_L, expected_value=ev_L,
                rule_reason="+EV Long"
            )

        if ev_S > self.min_ev and ev_S > ev_L and htf_tr <= 0:
            sl = ep + self.sl_mult * bar_atr
            tp = ep - self.tp_mult * bar_atr
            return SignalResult(
                timestamp=df.index[idx], direction=-1,
                entry_price=ep, sl_price=sl, tp_price=tp,
                ml_probability=prob_S, expected_value=ev_S,
                rule_reason="+EV Short"
            )

        return None

    # ── ML helpers ────────────────────────────────────────────────────────────

    def _batch_ml_prob(self, df: pd.DataFrame) -> np.ndarray:
        """Batch predict array of probabilities."""
        if self.model is None or not hasattr(self.model, "predict_proba"):
            return np.zeros(len(df))
        try:
            return self.model.predict_proba(df)
        except Exception as exc:
            logger.warning(f"ML batch predict failed: {exc}")
            return np.zeros(len(df))

    def _single_ml_prob(self, df: pd.DataFrame, i: int, direction: int) -> float:
        """Get model probability for the bar at index i, with a given direction."""
        if self.model is None or not hasattr(self.model, "predict_proba"):
            return 0.0
        try:
            row = df.iloc[[i]].copy()
            row["direction"] = direction
            return float(self.model.predict_proba(row)[0])
        except Exception as exc:
            logger.warning(f"ML single predict failed at bar {i}: {exc}")
            return 0.0
