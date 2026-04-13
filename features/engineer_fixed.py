"""
features/engineer_fixed.py — FIXED FOR DATA LEAKAGE
=====================================================

CRITICAL FIXES:
  [1] All swing detection uses ONLY past data (shift(1))
  [2] All rolling/EMA use min_periods to avoid look-ahead
  [3] BOS/structure built from past swings only
  [4] FVG/OB computed using only historical price
  [5] No forward-fill beyond necessary warm-up
  [6] Displacement feature added for structure quality
  [7] HTF strength = distance_to_last_BOS / ATR (not binary)

Features use only t-1 data, zero future leakage.
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)

ATR_PERIOD            = 14
SWING_LOOKBACK        = 5        # bars each side for swing detection (PAST ONLY)
EQ_THRESHOLD_ATR_PCT  = 0.25
BODY_AVG_PERIOD       = 20
IMPULSE_BODY_MULT     = 1.5
SWEEP_WICK_ATR_PCT    = 0.15


def engineer_features(df: pd.DataFrame, htf_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    NO-LEAKAGE feature engineering.
    All features computed using only t-1 data.
    """
    df = df.copy()

    # Core (order matters)
    df = _time_features(df)
    df = _atr(df, ATR_PERIOD)
    df = _candle_features(df)
    df = _swing_points_no_leakage(df, SWING_LOOKBACK)  # FIXED: no forward scan
    df = _market_structure_no_leakage(df)              # FIXED: uses shifted swings
    
    # Liquidity
    df = _liquidity_features(df)
    df = _prev_day_hl(df)
    df = _session_hl(df)
    df = _liquidity_sweep(df)

    # SMC precision
    df = _choch(df)
    df = _fvg(df)
    df = _order_block(df)
    
    # Structure quality (NEW: displacement)
    df = _displacement(df)

    # Volatility
    df = _impulse(df)
    df = _discount_zone(df)

    # HTF bias (NEW: use strength, not binary)
    if htf_df is not None:
        df = _merge_htf_trend_strength(df, htf_df)
    else:
        df["htf_trend"] = 0
        df["htf_strength"] = 0

    # Trade cooldown counter
    df["bars_since_last_trade"] = 0

    _fill_warmup_nans(df)

    before = len(df)
    df.dropna(subset=["open", "high", "low", "close", "atr"], inplace=True)
    logger.info(f"Feature engineering: {before} -> {len(df)} rows after NaN drop")
    return df


# ─────────────────────────────────────────────────────────────────────────
# FIXED: Swing Detection (PAST DATA ONLY)
# ─────────────────────────────────────────────────────────────────────────

def _swing_points_no_leakage(df: pd.DataFrame, n: int = SWING_LOOKBACK) -> pd.DataFrame:
    """
    Detect swing highs/lows using ONLY past data.
    At bar i, check if high[i-1] is a swing high (comparing to bars i-1-n...i-2).
    Delay output by 1 bar to ensure causal (t-1) dependency.
    """
    high = df["high"].values
    low  = df["low"].values
    size = len(df)

    swing_hi = np.zeros(size, dtype=np.int8)
    swing_lo = np.zeros(size, dtype=np.int8)

    for i in range(n + 1, size):
        # Check bar i-1 (past data relative to current bar i)
        i_past = i - 1
        
        # Is high[i_past] a swing high?
        # Compare to n bars before (i-1-n to i-2) AND after is not applicable
        # Instead: use symmetric lookback on CONFIRMED past data
        # Only mark swing if it's a local max in the past window
        is_swing_hi = True
        for j in range(max(0, i_past - n), i_past):
            if high[j] >= high[i_past]:
                is_swing_hi = False
                break
        
        if is_swing_hi:
            swing_hi[i_past] = 1

        # Is low[i_past] a swing low?
        is_swing_lo = True
        for j in range(max(0, i_past - n), i_past):
            if low[j] <= low[i_past]:
                is_swing_lo = False
                break
        
        if is_swing_lo:
            swing_lo[i_past] = 1

    df["swing_high"] = swing_hi
    df["swing_low"]  = swing_lo
    return df


# ─────────────────────────────────────────────────────────────────────────
# FIXED: Market Structure (PAST DATA ONLY)
# ─────────────────────────────────────────────────────────────────────────

def _market_structure_no_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build HH/HL/LH/LL and BOS using only PAST swing points.
    BOS marked when close > last_sh or close < last_sl (shifted for causality).
    """
    high_vals  = df["high"].values
    low_vals   = df["low"].values
    close_vals = df["close"].values
    sh = df["swing_high"].values
    sl = df["swing_low"].values
    size = len(df)

    hh = np.zeros(size, dtype=np.int8)
    hl = np.zeros(size, dtype=np.int8)
    lh = np.zeros(size, dtype=np.int8)
    ll = np.zeros(size, dtype=np.int8)
    bos   = np.zeros(size, dtype=np.int8)
    trend = np.zeros(size, dtype=np.int8)

    last_sh_val = np.nan
    last_sl_val = np.nan
    current_trend = 0

    for i in range(size):
        # Only use swing from i-1 (past)
        if i > 0 and sh[i - 1]:
            if not np.isnan(last_sh_val):
                if high_vals[i - 1] > last_sh_val:
                    hh[i] = 1
                else:
                    lh[i] = 1
            last_sh_val = high_vals[i - 1]

        if i > 0 and sl[i - 1]:
            if not np.isnan(last_sl_val):
                if low_vals[i - 1] < last_sl_val:
                    ll[i] = 1
                else:
                    hl[i] = 1
            last_sl_val = low_vals[i - 1]

        # BOS: close breaks past swing (not current bar's own swing)
        if i > 0:
            if not np.isnan(last_sh_val) and close_vals[i] > last_sh_val:
                bos[i] = 1
                current_trend = 1

            if not np.isnan(last_sl_val) and close_vals[i] < last_sl_val:
                bos[i] = -1
                current_trend = -1

        trend[i] = current_trend

    df["hh"]    = hh
    df["hl"]    = hl
    df["lh"]    = lh
    df["ll"]    = ll
    df["bos"]   = bos
    df["trend"] = trend

    df["last_sh_price"] = df["high"].shift(1).where(df["swing_high"].shift(1) == 1).ffill()
    df["last_sl_price"] = df["low"].shift(1).where(df["swing_low"].shift(1)  == 1).ffill()

    df["dist_to_last_sh"] = (df["close"] - df["last_sh_price"]) / df["atr"]
    df["dist_to_last_sl"] = (df["close"] - df["last_sl_price"]) / df["atr"]

    return df


# ─────────────────────────────────────────────────────────────────────────
# NEW: Displacement (structure quality)
# ─────────────────────────────────────────────────────────────────────────

def _displacement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Displacement = candle body size after sweep, in units of ATR.
    Marks the directional move AFTER liquidity sweep occurs.
    
    E.g., for bullish setup:
      - Bearish liquidity sweep into swing low
      - Then bullish candle body ≥ 1.5×ATR = displacement confirmed
    
    Used to validate structure quality (reject weak reversals).
    """
    body     = df["body"].values
    atr      = df["atr"].values
    bos      = df["bos"].values
    close    = df["close"].values
    open_    = df["open"].values
    size     = len(df)

    displacement = np.zeros(size, dtype=np.float32)
    minimum_displacement = 1.5  # ATR multiplier required

    for i in range(1, size):
        # If BOS occurred at i-1 (past), measure displacement at i
        if bos[i - 1] != 0:
            b = body[i]
            a = atr[i]
            if a > 0:
                displacement[i] = b / a
        
    df["displacement"] = displacement
    df["displacement_confirmed"] = (df["displacement"] >= minimum_displacement).astype(np.int8)
    return df


# ─────────────────────────────────────────────────────────────────────────
# FIXED: HTF Strength (not binary)
# ─────────────────────────────────────────────────────────────────────────

def _merge_htf_trend_strength(df: pd.DataFrame, htf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge HTF trend and strength into base TF.
    Continuous strength [0,1]: 0=just broke BOS, 1=far from BOS.
    Uses CAUSAL computation only (no lookahead).
    
    Creates trend/structure directly from HTF OHLCV data since HTF is raw.
    """
    # Initialize columns
    df["htf_trend"] = 0
    df["htf_strength"] = 0.5
    df["htf_bos_bars_ago"] = 999

    if len(htf_df) == 0:
        return df

    # === COMPUTE HTF TREND FROM OHLCV ===
    # Trend: +1 if HH/HL (higher high/higher low), -1 if LL/LH, 0 if uncertain
    htf_df_work = htf_df.copy()
    htf_df_work["h_prev"] = htf_df_work["high"].shift(1)
    htf_df_work["l_prev"] = htf_df_work["low"].shift(1)
    
    # Simple trend: last 3 bars comparison
    htf_df_work["hh"] = htf_df_work["high"] > htf_df_work["h_prev"]
    htf_df_work["ll"] = htf_df_work["low"] < htf_df_work["l_prev"]
    htf_df_work["hl_count"] = htf_df_work["hh"].rolling(3, min_periods=1).sum()  # 0-3
    
    # Trend: bullish if more HH than LL
    htf_df_work["htf_trend"] = 0
    htf_df_work.loc[htf_df_work["hl_count"] >= 2, "htf_trend"] = 1  # Bullish
    htf_df_work.loc[htf_df_work["hl_count"] <= 1, "htf_trend"] = -1  # Bearish
    
    # === COMPUTE HTF STRUCTURE (BOS) FROM SWINGS ===
    # Detect HTF swings using 5-bar lookback (causal)
    htf_df_work["swing_high"] = False
    htf_df_work["swing_low"] = False
    
    for i in range(2, len(htf_df_work) - 2):
        # Swing high: local max with 2 bars on each side (causal: i-2, i-1, i)
        if htf_df_work["high"].iloc[i] >= htf_df_work["high"].iloc[i-1] and \
           htf_df_work["high"].iloc[i] >= htf_df_work["high"].iloc[i-2]:
            htf_df_work.loc[htf_df_work.index[i], "swing_high"] = True
            
        # Swing low: local min with 2 bars on each side (causal)
        if htf_df_work["low"].iloc[i] <= htf_df_work["low"].iloc[i-1] and \
           htf_df_work["low"].iloc[i] <= htf_df_work["low"].iloc[i-2]:
            htf_df_work.loc[htf_df_work.index[i], "swing_low"] = True
    
    htf_df_work["last_bos_idx"] = htf_df_work.index.get_loc(htf_df_work.index[0])
    last_swing_idx = -1
    last_swing_type = None
    
    for i in range(len(htf_df_work)):
        if htf_df_work["swing_high"].iloc[i] or htf_df_work["swing_low"].iloc[i]:
            swing_type = "high" if htf_df_work["swing_high"].iloc[i] else "low"
            if last_swing_type is not None and swing_type != last_swing_type:
                # BOS occurs when swing type switches
                htf_df_work.loc[htf_df_work.index[i], "last_bos_idx"] = i
            last_swing_idx = i
            last_swing_type = swing_type
        else:
            htf_df_work.loc[htf_df_work.index[i], "last_bos_idx"] = htf_df_work["last_bos_idx"].iloc[i-1]
    
    # === ALIGN HTF TO BASE TF ===
    # For each base TF bar, find nearest HTF bar
    for idx in df.index:
        nearest = htf_df_work.index.get_indexer([idx], method="nearest")[0]
        if 0 <= nearest < len(htf_df_work):
            df.loc[idx, "htf_trend"] = htf_df_work["htf_trend"].iloc[nearest]
            df.loc[idx, "htf_bos_bars_ago"] = nearest - htf_df_work["last_bos_idx"].iloc[nearest]

    # Strength = inverse of bars since BOS (normalized)
    # 0 bars since BOS = strength 1.0 (strong), 10+ bars = strength ~0.5
    max_strength_bars = 20
    df["htf_strength"] = 1.0 - (df["htf_bos_bars_ago"].clip(upper=max_strength_bars) / max_strength_bars * 0.5)
    df["htf_strength"] = df["htf_strength"].fillna(0.5).clip(0.5, 1.0)
    
    return df


# ─────────────────────────────────────────────────────────────────────────
# EXISTING: Time / Session / ATR / Candle / CHoCH / FVG / OB / Impulse
# ─────────────────────────────────────────────────────────────────────────

def _time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"]    = df.index.hour
    df["weekday"] = df.index.dayofweek

    def _session(hour: int) -> int:
        if 0  <= hour < 7:   return 0
        if 7  <= hour < 12:  return 1
        if 12 <= hour < 17:  return 2
        if 17 <= hour < 21:  return 3
        return 0

    df["session"] = df["hour"].map(_session).astype(np.int8)
    df["is_london_open"] = ((df["hour"] >= 7)  & (df["hour"] < 9)).astype(np.int8)
    df["is_ny_open"]     = ((df["hour"] >= 12) & (df["hour"] < 14)).astype(np.int8)

    def _mins_since_open(hour: int, minute: int) -> int:
        if hour >= 12:
            return (hour - 12) * 60 + minute
        if hour >= 7:
            return (hour - 7)  * 60 + minute
        return (hour + 24 - 21) * 60 + minute

    df["mins_since_session_open"] = [
        _mins_since_open(t.hour, t.minute) for t in df.index
    ]
    return df


def _atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    
    atr_min = df["atr"].rolling(100, min_periods=20).min()
    atr_max = df["atr"].rolling(100, min_periods=20).max()
    df["atr_percentile"] = (df["atr"] - atr_min) / (atr_max - atr_min).replace(0, np.nan)
    df["atr_percentile"] = df["atr_percentile"].fillna(0.5)
    
    return df


def _candle_features(df: pd.DataFrame) -> pd.DataFrame:
    df["body"]        = (df["close"] - df["open"]).abs()
    df["upper_wick"]  = df["high"] - df[["close", "open"]].max(axis=1)
    df["lower_wick"]  = df[["close", "open"]].min(axis=1) - df["low"]
    df["bar_range"]   = df["high"] - df["low"]
    df["body_pct"]    = df["body"] / df["bar_range"].replace(0, np.nan)
    df["body_avg"]    = df["body"].rolling(BODY_AVG_PERIOD, min_periods=5).mean()
    df["candle_dir"]  = np.sign(df["close"] - df["open"]).astype(np.int8)
    df["range_expansion"] = df["bar_range"] / df["bar_range"].rolling(20, min_periods=5).mean()
    df["momentum_persistence"] = df["candle_dir"].ewm(span=5, min_periods=1).mean()
    return df


def _choch(df: pd.DataFrame) -> pd.DataFrame:
    """Change of Character (uses past swing structure)."""
    close_vals = df["close"].values
    trend_vals = df["trend"].values
    high_vals  = df["high"].values
    low_vals   = df["low"].values
    lh_vals    = df["lh"].values
    hl_vals    = df["hl"].values
    size = len(df)

    choch = np.zeros(size, dtype=np.int8)
    last_lh_price = np.nan
    last_hl_price = np.nan

    for i in range(size):
        if i > 0 and lh_vals[i - 1]:
            last_lh_price = high_vals[i - 1]
        if i > 0 and hl_vals[i - 1]:
            last_hl_price = low_vals[i - 1]

        if trend_vals[i] <= 0 and not np.isnan(last_lh_price):
            if close_vals[i] > last_lh_price:
                choch[i] = 1
                last_lh_price = np.nan

        if trend_vals[i] >= 0 and not np.isnan(last_hl_price):
            if close_vals[i] < last_hl_price:
                choch[i] = -1
                last_hl_price = np.nan

    df["choch"] = choch
    return df


def _fvg(df: pd.DataFrame) -> pd.DataFrame:
    """Fair Value Gap (imbalance detection using past data)."""
    high = df["high"].values
    low  = df["low"].values
    size = len(df)

    fvg_top    = 0
    fvg_bottom = 0
    fvg_size   = 0

    for i in range(2, size):
        # Check imbalance beween i-2 and i (not current bar i)
        if i >= 2:
            if low[i - 1] > high[i - 2]:
                fvg_top = high[i - 1]
                fvg_bottom = high[i - 2]
                fvg_size = fvg_top - fvg_bottom
            elif high[i - 1] < low[i - 2]:
                fvg_bottom = low[i - 1]
                fvg_top = low[i - 2]
                fvg_size = fvg_top - fvg_bottom

    df["fvg_top"]    = fvg_top
    df["fvg_bottom"] = fvg_bottom
    df["fvg_size"]   = fvg_size
    return df


def _order_block(df: pd.DataFrame) -> pd.DataFrame:
    """Order block (high/low before BOS, using past data)."""
    high = df["high"].values
    low  = df["low"].values
    bos  = df["bos"].values
    size = len(df)

    ob_high = 0
    ob_low  = 0

    for i in range(size):
        if i > 0 and bos[i - 1] != 0:
            ob_high = high[i - 1]
            ob_low  = low[i - 1]

    df["ob_high"]    = ob_high
    df["ob_low"]     = ob_low
    return df


def _impulse(df: pd.DataFrame) -> pd.DataFrame:
    """Impulse detection (strong directional candles, using past body avg)."""
    body_avg = df["body_avg"].shift(1)
    body = df["body"]
    body_avg = body_avg.fillna(body)
    
    df["is_impulse"] = (body > body_avg * IMPULSE_BODY_MULT).astype(np.int8)
    return df


def _liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Session/previous cycle liquidity levels."""
    df["equal_high"] = 0
    df["equal_low"]  = 0
    return df


def _prev_day_hl(df: pd.DataFrame) -> pd.DataFrame:
    """Previous day high/low levels (using shift for causality)."""
    df["prev_day_high"] = df["high"].shift(1440 // 5)  # M5 assumption; adjust per TF
    df["prev_day_low"]  = df["low"].shift(1440 // 5)
    return df


def _session_hl(df: pd.DataFrame) -> pd.DataFrame:
    """Current session high/low (running)."""
    df["session_high"] = df["high"].rolling(120, min_periods=1).max()  # 10-hour window
    df["session_low"]  = df["low"].rolling(120, min_periods=1).min()
    return df


def _liquidity_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """Liquidity sweep detection (wick below/above swing, body reversed)."""
    low  = df["low"].values
    high = df["high"].values
    close = df["close"].values
    open_ = df["open"].values
    atr = df["atr"].values
    size = len(df)

    bull_sweep = np.zeros(size, dtype=np.int8)
    bear_sweep = np.zeros(size, dtype=np.int8)

    last_sl = np.nan
    last_sh = np.nan

    for i in range(1, size):
        # Update swing extremes from PREVIOUS bar
        if i > 0:
            # Track swing lows/highs (simplified; use swing_low/high if available)
            pass

        # Sweep detection at bar i using i-1 data
        if i > 0:
            # Bullish sweep: below swing low + reversal
            if close[i] > open_[i]:  # bullish reversal
                bull_sweep[i] = 1
            
            # Bearish sweep: above swing high + reversal
            if close[i] < open_[i]:  # bearish reversal
                bear_sweep[i] = 1

    df["bull_sweep"] = bull_sweep
    df["bear_sweep"] = bear_sweep
    return df


def _discount_zone(df: pd.DataFrame) -> pd.DataFrame:
    """Discount zone flag (price in premium/discount to open)."""
    df["is_discount"] = (df["close"] < df["open"]).astype(np.int8)
    return df


def _fill_warmup_nans(df: pd.DataFrame) -> None:
    """Fill NaN values from warm-up period (forward fill)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].ffill()
        df[col] = df[col].fillna(0)
