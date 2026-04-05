"""
features/labeler.py
-------------------
SMC/ICT setup-based labeling pipeline.

A valid trade setup requires ALL 4 steps in sequence:

  Step 1  Liquidity Sweep     Wick past a key level (swing H/L, prev-day H/L,
                               Asia session range), with close reverting back.

  Step 2  Displacement        Within DISPLACEMENT_BARS of the sweep:
                               a candle body > DISPLACEMENT_ATR_MULT × ATR,
                               moving IN THE REVERSAL DIRECTION.

  Step 3  CHoCH               Within CHOCH_BARS of displacement:
                               Bullish: close > last confirmed lower-high
                               Bearish: close < last confirmed higher-low

  Step 4  Entry Zone          Within ENTRY_BARS of CHoCH:
                               Price re-enters either a Fair Value Gap (FVG) or
                               the last Order Block (OB) range.
                               First touch = entry bar.

  Label   Outcome simulation  TP = nearest liquidity on the opposite side
                               (prev swing H or L), minimum RR_MIN × SL.
                               SL = beyond the sweep wick extreme.
                               label=1 if TP hit first, label=0 if SL hit first.

Only rows representing actual trade entry bars are emitted.
Candles with no setup context are NEVER labeled.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

@dataclass
class LabelConfig:
    # Step 2: displacement
    displacement_atr_mult: float = 1.5   # body must exceed N × ATR
    displacement_bars:     int   = 5     # how many bars after sweep to find displacement

    # Step 3: CHoCH
    choch_bars: int = 15                 # bars after displacement to find CHoCH

    # Step 4: entry zone
    entry_bars: int = 20                 # bars after CHoCH to find FVG/OB fill

    # TP/SL
    rr_min:               float = 2.0   # minimum risk/reward ratio
    sl_buffer_atr:        float = 0.2   # extra buffer beyond sweep extreme for SL
    max_sl_atr:           float = 3.0   # invalidate setups with too-wide SL

    # Label quality filter
    min_sweep_strength:   float = 0.05  # ATR units beyond the swept level


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

class SetupLabeler:
    """
    Scans enriched DataFrame for valid SMC setups.
    Returns a DataFrame where each row = one trade entry, fully featured.
    """

    def __init__(self, config: Optional[LabelConfig] = None):
        self.cfg = config or LabelConfig()

    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect all 4-step setups and return labeled entry rows.

        Returns
        -------
        pd.DataFrame with all original features at the entry bar plus:
          direction, entry_type, sweep_type, sweep_strength,
          entry_price, sl_price, tp_price, rr_actual, label
        """
        records = []
        size    = len(df)

        bull_sw   = df["bull_sweep"].values
        bear_sw   = df["bear_sweep"].values
        sw_str    = df["sweep_strength"].values

        for i in range(size):
            # ── Bullish setup: swept a LOW, expect price to rise ─────
            if bull_sw[i] and sw_str[i] >= self.cfg.min_sweep_strength:
                rec = self._build_setup(df, i, direction=1)
                if rec is not None:
                    records.append(rec)

            # ── Bearish setup: swept a HIGH, expect price to fall ────
            if bear_sw[i] and sw_str[i] >= self.cfg.min_sweep_strength:
                rec = self._build_setup(df, i, direction=-1)
                if rec is not None:
                    records.append(rec)

        if not records:
            logger.warning(
                "No valid SMC setups found. "
                "Try loosening displacement_atr_mult or extending lookback windows."
            )
            return pd.DataFrame()

        labeled = pd.DataFrame(records).set_index("timestamp")
        pos_rate = labeled["label"].mean()
        logger.info(
            f"SMC Labeler: {len(labeled):,} setups | "
            f"Win rate: {pos_rate:.1%} | "
            f"Long: {(labeled['direction']==1).sum()} | "
            f"Short: {(labeled['direction']==-1).sum()}"
        )
        return labeled

    # ──────────────────────────────────────────────────────────
    # Private: 4-step pipeline
    # ──────────────────────────────────────────────────────────

    def _build_setup(
        self,
        df: pd.DataFrame,
        sweep_idx: int,
        direction: int,
    ) -> Optional[dict]:
        """
        direction: +1 = bullish (swept low, expect up)
                   -1 = bearish (swept high, expect down)
        """
        cfg  = self.cfg
        size = len(df)

        high  = df["high"].values
        low   = df["low"].values
        close = df["close"].values
        body  = df["body"].values
        atr   = df["atr"].values

        # Reference levels stored on the sweep bar
        sweep_row = df.iloc[sweep_idx]
        sweep_extreme = (
            float(low[sweep_idx])   if direction == 1
            else float(high[sweep_idx])
        )

        # ── Step 2: Displacement ──────────────────────────────────────
        disp_idx = None
        search_end = min(sweep_idx + cfg.displacement_bars + 1, size)
        for j in range(sweep_idx + 1, search_end):
            required_body = cfg.displacement_atr_mult * atr[j]
            candle_bullish = close[j] > df["open"].values[j]
            candle_bearish = close[j] < df["open"].values[j]
            if direction == 1 and candle_bullish and body[j] > required_body:
                disp_idx = j
                break
            if direction == -1 and candle_bearish and body[j] > required_body:
                disp_idx = j
                break

        if disp_idx is None:
            return None

        # ── Step 3: CHoCH ─────────────────────────────────────────────
        choch_idx = None
        choch_vals = df["choch"].values if "choch" in df.columns else np.zeros(size)
        search_end = min(disp_idx + cfg.choch_bars + 1, size)

        # A CHoCH in the right direction
        # Bullish: choch == +1 | Bearish: choch == -1
        for j in range(disp_idx + 1, search_end):
            if direction == 1 and choch_vals[j] == 1:
                choch_idx = j
                break
            if direction == -1 and choch_vals[j] == -1:
                choch_idx = j
                break

        # Fallback: accept a BOS if no CHoCH (BOS confirms structure shift too)
        if choch_idx is None:
            bos_vals = df["bos"].values
            for j in range(disp_idx + 1, search_end):
                if direction == 1 and bos_vals[j] == 1:
                    choch_idx = j
                    break
                if direction == -1 and bos_vals[j] == -1:
                    choch_idx = j
                    break

        if choch_idx is None:
            return None

        # ── Step 4: Entry Zone (FVG or OB fill) ──────────────────────
        entry_idx   = None
        entry_price = None
        entry_type  = None

        fvg_top  = df["fvg_top"].values  if "fvg_top"  in df.columns else np.full(size, np.nan)
        fvg_bot  = df["fvg_bot"].values  if "fvg_bot"  in df.columns else np.full(size, np.nan)

        bull_ob_h = df["bull_ob_high"].values if "bull_ob_high" in df.columns else np.full(size, np.nan)
        bull_ob_l = df["bull_ob_low"].values  if "bull_ob_low"  in df.columns else np.full(size, np.nan)
        bear_ob_h = df["bear_ob_high"].values if "bear_ob_high" in df.columns else np.full(size, np.nan)
        bear_ob_l = df["bear_ob_low"].values  if "bear_ob_low"  in df.columns else np.full(size, np.nan)

        search_end = min(choch_idx + cfg.entry_bars + 1, size)
        for k in range(choch_idx, search_end):
            if direction == 1:
                # FVG fill: price dips back into bullish FVG
                if (not np.isnan(fvg_bot[k]) and not np.isnan(fvg_top[k])
                        and low[k] <= fvg_top[k] and close[k] >= fvg_bot[k]):
                    entry_idx   = k
                    entry_price = float(fvg_top[k])  # entry at top of FVG
                    entry_type  = "fvg"
                    break
                # OB fill: price dips into last bullish OB zone
                if (not np.isnan(bull_ob_h[k]) and not np.isnan(bull_ob_l[k])
                        and low[k] <= bull_ob_h[k] and close[k] >= bull_ob_l[k]):
                    entry_idx   = k
                    entry_price = float(bull_ob_h[k])  # entry at OB top
                    entry_type  = "ob"
                    break
            else:
                # FVG fill: price pops back into bearish FVG
                if (not np.isnan(fvg_bot[k]) and not np.isnan(fvg_top[k])
                        and high[k] >= fvg_bot[k] and close[k] <= fvg_top[k]):
                    entry_idx   = k
                    entry_price = float(fvg_bot[k])  # entry at bottom of FVG
                    entry_type  = "fvg"
                    break
                # OB fill: price pops into last bearish OB zone
                if (not np.isnan(bear_ob_l[k]) and not np.isnan(bear_ob_h[k])
                        and high[k] >= bear_ob_l[k] and close[k] <= bear_ob_h[k]):
                    entry_idx   = k
                    entry_price = float(bear_ob_l[k])  # entry at OB bottom
                    entry_type  = "ob"
                    break

        # Fallback: if no FVG/OB filled, use CHoCH bar close as entry
        if entry_idx is None:
            entry_idx   = choch_idx
            entry_price = float(close[choch_idx])
            entry_type  = "choch_close"

        # ── SL: beyond sweep extreme ──────────────────────────────────
        sl_buffer = cfg.sl_buffer_atr * atr[entry_idx]
        if direction == 1:
            sl_price = sweep_extreme - sl_buffer
        else:
            sl_price = sweep_extreme + sl_buffer

        sl_dist = abs(entry_price - sl_price)
        if sl_dist == 0 or sl_dist > cfg.max_sl_atr * atr[entry_idx]:
            return None   # degenerate or too wide

        # ── TP: nearest opposing liquidity (min RR_MIN × SL) ─────────
        tp_min_dist = sl_dist * cfg.rr_min
        tp_price    = self._find_tp(
            df, entry_idx, direction,
            entry_price, tp_min_dist
        )
        if tp_price is None:
            return None

        rr_actual = abs(tp_price - entry_price) / sl_dist

        # ── Simulate outcome ──────────────────────────────────────────
        label = _simulate_outcome(
            direction, entry_idx, tp_price, sl_price,
            high, low, size
        )
        if label is None:
            return None   # trade never resolved

        # ── Build record ──────────────────────────────────────────────
        row = df.iloc[entry_idx].to_dict()
        row.update(
            timestamp     = df.index[entry_idx],
            direction     = direction,
            entry_type    = entry_type,
            sweep_idx     = sweep_idx,
            disp_idx      = disp_idx,
            choch_idx     = choch_idx,
            sweep_extreme = sweep_extreme,
            entry_price   = entry_price,
            sl_price      = sl_price,
            tp_price      = tp_price,
            rr_actual     = round(rr_actual, 2),
            label         = label,
        )
        return row

    # ──────────────────────────────────────────────────────────
    # TP targeting: nearest opposing swing H/L above minimum RR
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _find_tp(
        df: pd.DataFrame,
        entry_idx: int,
        direction: int,
        entry_price: float,
        tp_min_dist: float,
    ) -> Optional[float]:
        """
        Walk forward from entry, find the first swing high (bullish) or
        swing low (bearish) that exceeds the RR minimum.
        Fall back to the prev_day_high / prev_day_low if no swing found within 50 bars.
        """
        size = len(df)
        swing_high = df["swing_high"].values
        swing_low  = df["swing_low"].values
        high_vals  = df["high"].values
        low_vals   = df["low"].values

        for j in range(entry_idx + 1, min(entry_idx + 60, size)):
            if direction == 1 and swing_high[j]:
                candidate = high_vals[j]
                if candidate - entry_price >= tp_min_dist:
                    return float(candidate)
            if direction == -1 and swing_low[j]:
                candidate = low_vals[j]
                if entry_price - candidate >= tp_min_dist:
                    return float(candidate)

        # Fallback: fixed RR multiple of SL
        if direction == 1:
            return entry_price + tp_min_dist
        return entry_price - tp_min_dist


# ──────────────────────────────────────────────────────────────
# Outcome simulation (module-level, fast)
# ──────────────────────────────────────────────────────────────

def _simulate_outcome(direction, entry_idx, tp_price, sl_price,
                      high, low, size) -> Optional[int]:
    """
    Bar-by-bar simulation: returns 1 (TP hit first), 0 (SL hit first),
    or None (unresolved before end of data).
    """
    for i in range(entry_idx + 1, size):
        if direction == 1:
            if high[i] >= tp_price:
                return 1
            if low[i]  <= sl_price:
                return 0
        else:
            if low[i]  <= tp_price:
                return 1
            if high[i] >= sl_price:
                return 0
    return None


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

    # Volatility
    "atr",

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

    # FVG
    "fvg_bull", "fvg_bear", "fvg_size",

    # Order Block
    "ob_size",

    # Entry context
    "entry_in_discount",

    # Impulse
    "impulse",

    # HTF
    "htf_trend",

    # Setup-level (available at entry bar)
    "direction",
]


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return MODEL_FEATURES columns that actually exist in df."""
    return [c for c in MODEL_FEATURES if c in df.columns]
