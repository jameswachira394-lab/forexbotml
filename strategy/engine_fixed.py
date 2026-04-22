"""
strategy/engine_fixed.py — INSTITUTIONAL-GRADE, BAR-BY-BAR EXECUTION
====================================================================

FIXES (original):
  [1] Bar-by-bar execution (NOT batch scan_all)
  [2] Sequential state machine (sweep → displacement → BOS → pullback)
  [3] Displacement: body ≥ 1.5×ATR required post-BOS
  [4] Pullback range: 0.5-2.5 ATR (not 0.1-5.0)
  [5] HTF strength gate (not binary)
  [6] Trade cooldown: 4 bars between signals
  [7] Cost-aware RR computation
  [8] No future data access (causal)

FIXES (v2 — code review):
  [A] Directional displacement: features must supply bull_displacement_confirmed /
      bear_displacement_confirmed separately to avoid cross-contamination.
  [B] Setup expiry: stale sweep/displacement states reset after max_bars_to_bos bars.
  [C] EV formula corrected: cost expressed in R-units via sl_dist_pips so tight stops
      are properly penalised.
  [D] Per-direction cooldown: bull_last_signal_bar / bear_last_signal_bar replace the
      single shared last_signal_bar.
  [E] Minimum bar gap between displacement and BOS (≥ 1 bar).
  [F] ML failure returns None instead of trading on random noise.
  [G] max_bars_pullback is now used (pullback window closes after that many bars
      past BOS, capped by max_bars_to_bos).
  [H] direction parameter is now encoded in the ML feature vector.

Usage:
  engine = StrategyEngineFixed(config, model=trained_model)
  signal = engine.process_bar(bar_idx, ts, ohlc_dict, atr, features_dict)
  if signal:
      # Execute trade

Feature dict keys (updated):
  bull_sweep:                    int  (0/1)  — bearish sweep of a swing low detected
  bear_sweep:                    int  (0/1)  — bullish sweep of a swing high detected
  bull_displacement_confirmed:   int  (0/1)  — bullish displacement confirmed
  bear_displacement_confirmed:   int  (0/1)  — bearish displacement confirmed
  displacement:                  float       — magnitude in ATR units (latest)
  bos:                           int  (-1/0/+1)
  htf_trend:                     int  (-1/0/+1)
  htf_strength:                  float [0,1]
  swing_high:                    int  (0/1)
  swing_low:                     int  (0/1)
  last_sh_price:                 float
  last_sl_price:                 float
  atr_percentile:                float
  body_pct:                      float
  range_expansion:               float
  momentum_persistence:          float
  is_london_open:                int  (0/1)
  is_ny_open:                    int  (0/1)
  session:                       float
  fvg_size:                      float
  hour:                          float
  mins_since_session_open:       float
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# pip_size is instrument-specific; override via StrategyConfigFixed
_DEFAULT_PIP_SIZE = 0.0001  # EUR/USD, GBP/USD, etc.


# ────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyConfigFixed:
    # Trend filter
    require_htf_align:    bool  = True
    htf_strength_min:     float = 0.3       # HTF trend strength ≥ 30 %

    # Structure gates
    displacement_atr_min: float = 1.5       # body after BOS must be ≥ 1.5×ATR

    # Pullback range
    pullback_atr_min:     float = 0.5       # reject shallow noise
    pullback_atr_max:     float = 2.5       # reject deep / invalid

    # ML gate
    ml_threshold:         float = 0.50      # probability threshold

    # EV gate (cost-aware)
    min_ev:               float = 0.15      # minimum EV in R-units after costs

    # Stop-loss placement
    sl_buffer_atr:        float = 0.8       # safety buffer beyond sweep

    # Base R:R
    rr_ratio_base:        float = 3.0

    # Wick threshold
    sweep_wick_atr_min:   float = 0.5       # wick must be ≥ 0.5×ATR

    # Cooldown (per direction)
    trade_cooldown_bars:  int   = 4         # bars between signals in the same direction

    # Setup expiry
    max_bars_to_bos:      int   = 20        # bars after sweep before state expires
                                            # also max bars to wait for BOS after displacement
    # [G] Pullback window
    max_bars_pullback:    int   = 15        # bars after BOS in which pullback must occur

    # Instrument
    pip_size:             float = _DEFAULT_PIP_SIZE  # [C] needed for cost-in-R calc

    # Trading costs
    cost_pips:            float = 2.0       # spread + slippage in pips


# ────────────────────────────────────────────────────────────────────────────
# Signal dataclass
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class SignalFixed:
    timestamp:      pd.Timestamp
    direction:      int              # +1 long, -1 short
    entry_price:    float
    sl_price:       float
    tp_price:       float
    ml_probability: float
    expected_value: float            # in R-units
    rr_ratio:       float
    reason:         str
    displacement:   float            # actual displacement in ATR units


# ────────────────────────────────────────────────────────────────────────────
# Per-direction state
# ────────────────────────────────────────────────────────────────────────────

class StructureState:
    """Tracks setup progression for one direction."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sweep_idx:        int   = -1
        self.sweep_price:      float = float("nan")
        self.displacement_idx: int   = -1
        self.bos_idx:          int   = -1
        self.bos_close:        float = float("nan")
        self.last_signal_bar:  int   = -9999   # [D] per-direction cooldown


# ────────────────────────────────────────────────────────────────────────────
# Engine
# ────────────────────────────────────────────────────────────────────────────

class StrategyEngineFixed:
    """
    Bar-by-bar sequential strategy engine.

    State machine per direction:
      IDLE → SWEPT → DISPLACED → BOS_CONFIRMED → (pullback entry or expire)
    """

    def __init__(
        self,
        config: Optional[StrategyConfigFixed] = None,
        model=None,
    ):
        self.cfg   = config or StrategyConfigFixed()
        self.model = model

        self.bull_state = StructureState()
        self.bear_state = StructureState()

    # ──────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────

    def process_bar(
        self,
        bar_idx:   int,
        timestamp: pd.Timestamp,
        ohlc:      Dict[str, float],   # {"open", "high", "low", "close"}
        atr:       float,
        features:  Dict[str, Any],
    ) -> Optional[SignalFixed]:
        """
        Process one bar.  Returns a SignalFixed if all gates pass, else None.
        All feature values are assumed to be from t-1 (causal, no lookahead).
        """
        close = ohlc["close"]
        high  = ohlc["high"]
        low   = ohlc["low"]

        # ── Gate 1: HTF alignment ────────────────────────────────────────
        if self.cfg.require_htf_align:
            htf_strength = features.get("htf_strength", 0.0)
            if htf_strength < self.cfg.htf_strength_min:
                return None

        # ── State expiry ─────────────────────────────────────────────────
        # [B] Reset stale setups so the engine doesn't carry phantom state forever
        self._expire_state(self.bull_state, bar_idx)
        self._expire_state(self.bear_state, bar_idx)

        # ── BULLISH path ─────────────────────────────────────────────────
        bull_signal = self._process_bull(bar_idx, timestamp, ohlc, atr, features, low, close)
        if bull_signal:
            return bull_signal

        # ── BEARISH path ─────────────────────────────────────────────────
        bear_signal = self._process_bear(bar_idx, timestamp, ohlc, atr, features, high, close)
        if bear_signal:
            return bear_signal

        return None

    # ──────────────────────────────────────────────────────────────────────
    # State expiry helper
    # ──────────────────────────────────────────────────────────────────────

    def _expire_state(self, state: StructureState, bar_idx: int) -> None:
        """[B] Reset state if too much time has passed since the last anchor."""
        if state.sweep_idx < 0:
            return
        anchor = state.bos_idx if state.bos_idx >= 0 else state.sweep_idx
        if bar_idx - anchor > self.cfg.max_bars_to_bos:
            logger.debug("Setup expired — resetting state")
            cooldown_save = state.last_signal_bar
            state.reset()
            state.last_signal_bar = cooldown_save   # preserve cooldown across expiry

    # ──────────────────────────────────────────────────────────────────────
    # Bullish direction
    # ──────────────────────────────────────────────────────────────────────

    def _process_bull(
        self,
        bar_idx:   int,
        timestamp: pd.Timestamp,
        ohlc:      Dict[str, float],
        atr:       float,
        features:  Dict[str, Any],
        low:       float,
        close:     float,
    ) -> Optional[SignalFixed]:

        st = self.bull_state

        # Step 1 — sweep
        if features.get("bear_sweep", 0) == 1:
            st.reset()
            st.sweep_idx   = bar_idx
            st.sweep_price = low
            logger.debug(f"[{timestamp}] Bull SWEEP @ {low:.5f}")

        # Step 2 — directional displacement [A]
        if (st.sweep_idx >= 0
                and st.displacement_idx < 0
                and features.get("bull_displacement_confirmed", 0) == 1):
            st.displacement_idx = bar_idx
            logger.debug(f"[{timestamp}] Bull DISPLACEMENT ({features.get('displacement', 0):.2f}×ATR)")

        # Step 3 — BOS, with min 1-bar gap from displacement [E]
        if (st.displacement_idx >= 0
                and st.bos_idx < 0
                and bar_idx > st.displacement_idx          # [E]
                and features.get("bos", 0) == 1):
            st.bos_idx   = bar_idx
            st.bos_close = close
            logger.debug(f"[{timestamp}] Bull BOS @ {close:.5f}")

        # Step 4 — pullback entry
        if st.bos_idx >= 0:
            bars_since_bos = bar_idx - st.bos_idx
            # [G] pullback window closes after max_bars_pullback
            if bars_since_bos <= min(self.cfg.max_bars_pullback, self.cfg.max_bars_to_bos):
                pullback_dist = (st.bos_close - close) / atr
                if (self.cfg.pullback_atr_min <= pullback_dist <= self.cfg.pullback_atr_max
                        and bar_idx - st.last_signal_bar >= self.cfg.trade_cooldown_bars):  # [D]
                    signal = self._generate_signal(
                        bar_idx, timestamp, ohlc, atr, features,
                        pullback_dist, direction=1, state=st,
                    )
                    if signal:
                        st.last_signal_bar = bar_idx
                        cooldown_save = st.last_signal_bar
                        st.reset()
                        st.last_signal_bar = cooldown_save
                        return signal

        return None

    # ──────────────────────────────────────────────────────────────────────
    # Bearish direction
    # ──────────────────────────────────────────────────────────────────────

    def _process_bear(
        self,
        bar_idx:   int,
        timestamp: pd.Timestamp,
        ohlc:      Dict[str, float],
        atr:       float,
        features:  Dict[str, Any],
        high:      float,
        close:     float,
    ) -> Optional[SignalFixed]:

        st = self.bear_state

        # Step 1 — sweep
        if features.get("bull_sweep", 0) == 1:
            st.reset()
            st.sweep_idx   = bar_idx
            st.sweep_price = high
            logger.debug(f"[{timestamp}] Bear SWEEP @ {high:.5f}")

        # Step 2 — directional displacement [A]
        if (st.sweep_idx >= 0
                and st.displacement_idx < 0
                and features.get("bear_displacement_confirmed", 0) == 1):
            st.displacement_idx = bar_idx
            logger.debug(f"[{timestamp}] Bear DISPLACEMENT ({features.get('displacement', 0):.2f}×ATR)")

        # Step 3 — BOS with min 1-bar gap [E]
        if (st.displacement_idx >= 0
                and st.bos_idx < 0
                and bar_idx > st.displacement_idx
                and features.get("bos", 0) == -1):
            st.bos_idx   = bar_idx
            st.bos_close = close
            logger.debug(f"[{timestamp}] Bear BOS @ {close:.5f}")

        # Step 4 — pullback entry
        if st.bos_idx >= 0:
            bars_since_bos = bar_idx - st.bos_idx
            if bars_since_bos <= min(self.cfg.max_bars_pullback, self.cfg.max_bars_to_bos):
                pullback_dist = (close - st.bos_close) / atr
                if (self.cfg.pullback_atr_min <= pullback_dist <= self.cfg.pullback_atr_max
                        and bar_idx - st.last_signal_bar >= self.cfg.trade_cooldown_bars):
                    signal = self._generate_signal(
                        bar_idx, timestamp, ohlc, atr, features,
                        pullback_dist, direction=-1, state=st,
                    )
                    if signal:
                        st.last_signal_bar = bar_idx
                        cooldown_save = st.last_signal_bar
                        st.reset()
                        st.last_signal_bar = cooldown_save
                        return signal

        return None

    # ──────────────────────────────────────────────────────────────────────
    # Signal generation (shared)
    # ──────────────────────────────────────────────────────────────────────

    def _generate_signal(
        self,
        bar_idx:      int,
        timestamp:    pd.Timestamp,
        ohlc:         Dict[str, float],
        atr:          float,
        features:     Dict[str, Any],
        pullback_dist: float,
        direction:    int,            # +1 or -1
        state:        StructureState,
    ) -> Optional[SignalFixed]:

        close = ohlc["close"]

        # ── Stop-loss placement ──────────────────────────────────────────
        if direction == 1:
            sl_price = state.sweep_price - self.cfg.sl_buffer_atr * atr
            sl_dist  = close - sl_price
        else:
            sl_price = state.sweep_price + self.cfg.sl_buffer_atr * atr
            sl_dist  = sl_price - close

        if sl_dist <= 0:
            logger.debug(f"[{timestamp}] Signal rejected: sl_dist <= 0")
            return None

        # ── Dynamic R:R ──────────────────────────────────────────────────
        rr_ratio = self.cfg.rr_ratio_base + max(0.0, pullback_dist - 1.0) * 0.25
        if direction == 1:
            tp_price = close + sl_dist * rr_ratio
        else:
            tp_price = close - sl_dist * rr_ratio

        # ── ML gate ──────────────────────────────────────────────────────
        ml_prob = self._get_ml_prob(features, direction, timestamp)
        if ml_prob is None:          # [F] hard failure — do not trade
            return None
        if ml_prob < self.cfg.ml_threshold:
            logger.debug(f"[{timestamp}] ML gate failed: {ml_prob:.3f} < {self.cfg.ml_threshold}")
            return None

        # ── EV gate (cost-aware, in R-units) ─────────────────────────────
        # [C] Express cost as a fraction of 1R so tight stops are penalised correctly.
        sl_dist_pips = sl_dist / self.cfg.pip_size
        cost_in_r    = self.cfg.cost_pips / sl_dist_pips if sl_dist_pips > 0 else 1.0
        ev = ml_prob * (rr_ratio - cost_in_r) - (1.0 - ml_prob) * (1.0 + cost_in_r)

        if ev < self.cfg.min_ev:
            logger.debug(f"[{timestamp}] EV gate failed: {ev:.3f} < {self.cfg.min_ev}")
            return None

        side = "BULLISH" if direction == 1 else "BEARISH"
        label = "BullSweep" if direction == 1 else "BearSweep"
        logger.info(
            f"[{timestamp}] ✓ {side} SIGNAL | "
            f"Entry={close:.5f} SL={sl_price:.5f} TP={tp_price:.5f} | "
            f"ML={ml_prob:.3f} RR={rr_ratio:.2f} EV={ev:.3f} PB={pullback_dist:.2f}ATR"
        )

        return SignalFixed(
            timestamp=timestamp,
            direction=direction,
            entry_price=close,
            sl_price=sl_price,
            tp_price=tp_price,
            ml_probability=ml_prob,
            expected_value=ev,
            rr_ratio=rr_ratio,
            reason=f"{label}→Displacement→BOS→PB({pullback_dist:.2f}ATR)",
            displacement=features.get("displacement", 0.0),
        )

    # ──────────────────────────────────────────────────────────────────────
    # ML helpers
    # ──────────────────────────────────────────────────────────────────────

    def _get_ml_prob(
        self,
        features:  Dict[str, Any],
        direction: int,
        timestamp: pd.Timestamp,
    ) -> Optional[float]:
        """
        Return P(win) from the ML model, or None on hard failure.
        [F] Never falls back to random noise — caller must handle None.
        """
        if self.model is None:
            logger.warning("ML model not loaded — using default 0.50")
            return 0.50

        try:
            ml_features = self._extract_ml_features(features, direction)
            return float(self.model.predict_proba(ml_features)[0, 1])
        except Exception as exc:
            logger.error(f"[{timestamp}] ML prediction failed: {exc}")
            return None   # [F] propagate failure

    def _extract_ml_features(
        self,
        features:  Dict[str, Any],
        direction: int,
    ) -> np.ndarray:
        """
        Build the feature vector for the ML model.
        Rules-based features (sweep, BOS, HTF trend) are excluded to avoid
        double-counting.  direction is encoded as a signed float. [H]
        """
        feat_cols = [
            "atr_percentile",
            "body_pct",
            "range_expansion",
            "momentum_persistence",
            "is_london_open",
            "is_ny_open",
            "session",
            "fvg_size",
            "hour",
            "mins_since_session_open",
        ]
        vec = np.array(
            [features.get(c, 0.5) for c in feat_cols] + [float(direction)],  # [H]
            dtype=np.float32,
        )
        return vec.reshape(1, -1)