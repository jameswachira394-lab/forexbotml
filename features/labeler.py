"""
features/labeler.py  (FIXED)
-----------------------------
Fixes applied:
  [1.1]  No lookahead: TP/SL computed from ATR at entry bar only (no future swing refs)
  [3.1]  Candle ambiguity: SL checked before TP on same bar (worst-case)
  [3.2]  Spread applied to entry and TP/SL levels
  [2.5]  time_to_outcome added as feature
  [2.1]  Unresolved trades skipped (not forced-labelled)
  [2.4]  direction stored as explicit feature for mixed-entry separation
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LabelConfig:
    rr_ratio:           float = 2.0
    sl_atr_mult:        float = 1.5
    max_bars_to_bos:    int   = 20
    max_bars_to_entry:  int   = 25
    pullback_pct:       float = 0.20
    min_atr_move:       float = 0.2
    spread_pips:        float = 1.5    # [3.2] applied to entry and levels
    pip_size:           float = 0.0001 # override for JPY pairs


class SetupLabeler:

    def __init__(self, config: Optional[LabelConfig] = None):
        self.cfg = config or LabelConfig()

    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        records  = []
        high     = df["high"].values
        low      = df["low"].values
        close    = df["close"].values
        atr      = df["atr"].values
        bos      = df["bos"].values
        bull_sw  = df["bull_sweep"].values
        bear_sw  = df["bear_sweep"].values
        size     = len(df)

        for i in range(size):
            if bull_sw[i]:
                r = self._resolve(df, i, 1, high, low, close, atr, bos, size)
                if r is not None:
                    records.append(r)
            if bear_sw[i]:
                r = self._resolve(df, i, -1, high, low, close, atr, bos, size)
                if r is not None:
                    records.append(r)

        if not records:
            logger.warning("No valid setups found.")
            return pd.DataFrame()

        labeled   = pd.DataFrame(records)
        labeled.set_index("timestamp", inplace=True)
        pos_rate  = labeled["label"].mean()
        long_cnt  = (labeled["direction"] == 1).sum()
        short_cnt = (labeled["direction"] == -1).sum()
        logger.info(
            f"SMC Labeler: {len(labeled):,} setups | "
            f"Win rate: {pos_rate:.1%} | "
            f"Long: {long_cnt} | Short: {short_cnt}"
        )
        return labeled

    # ── Core resolution ───────────────────────────────────────────────────────

    def _resolve(self, df, sweep_idx, direction,
                 high, low, close, atr, bos, size) -> Optional[dict]:
        cfg = self.cfg

        # Step 1: BOS after sweep
        bos_idx = None
        for j in range(sweep_idx + 1,
                       min(sweep_idx + cfg.max_bars_to_bos + 1, size)):
            if bos[j] == direction:
                bos_idx = j
                break
        if bos_idx is None:
            return None

        bos_move = abs(close[bos_idx] - close[sweep_idx])
        if bos_move < cfg.min_atr_move * atr[bos_idx]:
            return None

        # Step 2: Pullback entry
        bos_level   = close[bos_idx]
        retrace_min = bos_move * cfg.pullback_pct
        entry_idx   = None
        entry_price = None

        for k in range(bos_idx + 1,
                       min(bos_idx + cfg.max_bars_to_entry + 1, size)):
            pullback = (bos_level - low[k]) if direction == 1 else (high[k] - bos_level)
            if pullback >= retrace_min:
                entry_idx   = k
                entry_price = close[k]
                break
        if entry_idx is None:
            return None

        # Step 3: [1.1] Levels from ATR at entry only — no future swing refs
        spread    = cfg.spread_pips * cfg.pip_size
        sl_dist   = cfg.sl_atr_mult * atr[entry_idx]
        tp_dist   = sl_dist * cfg.rr_ratio

        # [3.2] Apply spread: long buyer pays ask (entry higher), SL/TP adjusted
        if direction == 1:
            actual_entry = entry_price + spread          # fill at ask
            sl_price     = actual_entry - sl_dist
            tp_price     = actual_entry + tp_dist - spread   # TP sell at bid
        else:
            actual_entry = entry_price - spread          # fill at bid (short sell)
            sl_price     = actual_entry + sl_dist
            tp_price     = actual_entry - tp_dist + spread   # TP buy at ask

        # Step 4: [3.1] Worst-case candle ambiguity — SL checked before TP
        label, bars_to_outcome = self._simulate_outcome(
            direction, entry_idx, tp_price, sl_price, high, low, size
        )
        if label is None:
            return None

        row = df.iloc[entry_idx].to_dict()
        row.update(
            timestamp        = df.index[entry_idx],
            entry_price      = actual_entry,
            sl_price         = sl_price,
            tp_price         = tp_price,
            direction        = direction,          # [2.4] explicit feature
            sweep_idx        = sweep_idx,
            bos_idx          = bos_idx,
            label            = label,
            time_to_outcome  = bars_to_outcome,    # [2.5] trade duration feature
            sl_dist_atr      = sl_dist / max(atr[entry_idx], 1e-9),
            rr_achieved      = cfg.rr_ratio if label == 1 else -(sl_dist / max(tp_dist, 1e-9)),
        )
        return row

    @staticmethod
    def _simulate_outcome(direction, entry_idx, tp_price, sl_price,
                          high, low, size):
        """
        [3.1] Worst-case rule: on any bar check SL first, then TP.
        This prevents inflating the win rate by assuming TP always fills first
        when both are touched in the same candle.
        """
        for i in range(entry_idx + 1, size):
            if direction == 1:
                if low[i]  <= sl_price:   return 0, i - entry_idx  # SL first
                if high[i] >= tp_price:   return 1, i - entry_idx
            else:
                if high[i] >= sl_price:   return 0, i - entry_idx  # SL first
                if low[i]  <= tp_price:   return 1, i - entry_idx
        return None, None


# ── Feature columns ──────────────────────────────────────────────────────────

MODEL_FEATURES = [
    # Time
    "hour", "weekday", "session",
    # Candle
    "body_pct", "body_ratio", "upper_wick", "lower_wick", "candle_dir",
    # ATR
    "atr",
    # Market structure
    "hh", "hl", "lh", "ll", "bos", "trend",
    "dist_to_last_sh", "dist_to_last_sl",
    # Liquidity
    "eq_high", "eq_low",
    "dist_prev_50_high", "dist_prev_50_low",
    "bull_sweep", "bear_sweep",
    # Session HL
    "asia_high_dist", "asia_low_dist",
    "london_high_dist", "london_low_dist",
    "ny_high_dist", "ny_low_dist",
    # Impulse / momentum
    "impulse",
    # HTF
    "htf_trend",
    # direction injected at inference time by strategy engine — not in raw bars
    # "time_to_outcome" is training-only; excluded from live inference
]


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in MODEL_FEATURES if c in df.columns]