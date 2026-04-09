"""
features/engineer.py
--------------------
SMC/ICT-aligned feature-engineering pipeline.

Produces per-bar context features used by the setup labeler and ML model:

  Market Structure : HH/HL/LH/LL, BOS, CHoCH, trend label
  Liquidity        : equal-H/L, prev-day H/L, session H/L, sweep detection,
                     sweep strength (ATR-normalised)
  Entry Precision  : Fair Value Gap (top/bottom/size), Order Block (high/low/size),
                     discount-zone flag
  Session          : Asia/London/NY open flags, time-since-open (minutes)
  Volatility       : ATR, body ratio, range expansion, impulse direction
  HTF              : H1 trend merged onto base TF

All features are added IN-PLACE to the DataFrame copy and returned.
NaN warm-up rows are dropped before returning.
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Tunables
# ─────────────────────────────────────────────────────────────
ATR_PERIOD            = 14
SWING_LOOKBACK        = 5        # bars each side for swing detection
EQ_THRESHOLD_ATR_PCT  = 0.25     # equal-high/low within 25% of ATR
BODY_AVG_PERIOD       = 20
IMPULSE_BODY_MULT     = 1.5      # body > 1.5× rolling avg body → impulse
SWEEP_WICK_ATR_PCT    = 0.15     # min wick size to qualify as a sweep


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, htf_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Run all feature groups on *df* (base timeframe).
    Optionally merge higher-timeframe trend context from *htf_df*.
    Returns enriched DataFrame with NaN warm-up rows dropped.
    """
    df = df.copy()

    # Core building blocks (order matters — later steps depend on earlier ones)
    df = _time_features(df)
    df = _atr(df, ATR_PERIOD)
    df = _candle_features(df)
    df = _swing_points(df, SWING_LOOKBACK)
    df = _market_structure(df)

    # Liquidity
    df = _liquidity_features(df)
    df = _prev_day_hl(df)
    df = _session_hl(df)
    df = _liquidity_sweep(df)

    # SMC precision layers
    df = _choch(df)
    df = _fvg(df)
    df = _order_block(df)

    # Volatility / entry context
    df = _impulse(df)
    df = _discount_zone(df)

    # HTF bias
    if htf_df is not None:
        df = _merge_htf_trend(df, htf_df)
    else:
        df["htf_trend"] = 0

    # Only drop rows where core price/atr data is missing
    # (session/liquidity columns may have warm-up NaNs — fill those)
    _fill_warmup_nans(df)

    before = len(df)
    df.dropna(subset=["open", "high", "low", "close", "atr"], inplace=True)
    logger.info(f"Feature engineering: {before} -> {len(df)} rows after NaN drop")
    return df


# ─────────────────────────────────────────────────────────────
# Time / session features
# ─────────────────────────────────────────────────────────────

def _time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"]    = df.index.hour
    df["weekday"] = df.index.dayofweek   # 0=Mon … 4=Fri

    def _session(hour: int) -> int:
        """0=Asia  1=London  2=NY  3=Late/Overlap"""
        if 0  <= hour < 7:   return 0
        if 7  <= hour < 12:  return 1
        if 12 <= hour < 17:  return 2
        if 17 <= hour < 21:  return 3
        return 0

    df["session"] = df["hour"].map(_session).astype(np.int8)

    # Binary session-open flags (first 2 hours of each session)
    df["is_london_open"] = ((df["hour"] >= 7)  & (df["hour"] < 9)).astype(np.int8)
    df["is_ny_open"]     = ((df["hour"] >= 12) & (df["hour"] < 14)).astype(np.int8)

    # Minutes since most-recent session open (useful for liquidity timing)
    def _mins_since_open(hour: int, minute: int) -> int:
        if hour >= 12:
            return (hour - 12) * 60 + minute
        if hour >= 7:
            return (hour - 7)  * 60 + minute
        return (hour + 24 - 21) * 60 + minute   # distance from Asia open 21:00

    df["mins_since_session_open"] = [
        _mins_since_open(t.hour, t.minute) for t in df.index
    ]
    return df


# ─────────────────────────────────────────────────────────────
# ATR
# ─────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    
    # Volatility regime (normalized rolling 100-bar position)
    atr_min = df["atr"].rolling(100, min_periods=20).min()
    atr_max = df["atr"].rolling(100, min_periods=20).max()
    df["atr_percentile"] = (df["atr"] - atr_min) / (atr_max - atr_min).replace(0, np.nan)
    df["atr_percentile"] = df["atr_percentile"].fillna(0.5)

    return df


# ─────────────────────────────────────────────────────────────
# Candle body / wick metrics
# ─────────────────────────────────────────────────────────────

def _candle_features(df: pd.DataFrame) -> pd.DataFrame:
    df["body"]        = (df["close"] - df["open"]).abs()
    df["upper_wick"]  = df["high"] - df[["close", "open"]].max(axis=1)
    df["lower_wick"]  = df[["close", "open"]].min(axis=1) - df["low"]
    df["bar_range"]   = df["high"] - df["low"]
    df["body_pct"]    = df["body"] / df["bar_range"].replace(0, np.nan)
    df["body_avg"]    = df["body"].rolling(BODY_AVG_PERIOD, min_periods=5).mean()
    df["candle_dir"]  = np.sign(df["close"] - df["open"]).astype(np.int8)

    # Range expansion vs recent average
    df["range_expansion"] = df["bar_range"] / df["bar_range"].rolling(20, min_periods=5).mean()
    
    # Momentum persistence
    df["momentum_persistence"] = df["candle_dir"].ewm(span=5, min_periods=1).mean()
    return df


# ─────────────────────────────────────────────────────────────
# Swing highs / lows
# ─────────────────────────────────────────────────────────────

def _swing_points(df: pd.DataFrame, n: int = SWING_LOOKBACK) -> pd.DataFrame:
    high = df["high"].values
    low  = df["low"].values
    size = len(df)

    swing_hi = np.zeros(size, dtype=np.int8)
    swing_lo = np.zeros(size, dtype=np.int8)

    for i in range(n, size - n):
        if high[i] > high[i - n : i].max() and high[i] > high[i + 1 : i + n + 1].max():
            swing_hi[i] = 1
        if low[i] < low[i - n : i].min() and low[i] < low[i + 1 : i + n + 1].min():
            swing_lo[i] = 1

    df["swing_high"] = swing_hi
    df["swing_low"]  = swing_lo
    return df


# ─────────────────────────────────────────────────────────────
# Market structure: HH/HL/LH/LL + BOS + trend
# ─────────────────────────────────────────────────────────────

def _market_structure(df: pd.DataFrame) -> pd.DataFrame:
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
        if sh[i]:
            if not np.isnan(last_sh_val):
                if high_vals[i] > last_sh_val:
                    hh[i] = 1
                else:
                    lh[i] = 1
            last_sh_val = high_vals[i]

        if sl[i]:
            if not np.isnan(last_sl_val):
                if low_vals[i] < last_sl_val:
                    ll[i] = 1
                else:
                    hl[i] = 1
            last_sl_val = low_vals[i]

        if not np.isnan(last_sh_val) and close_vals[i] > last_sh_val:
            if bos[i] != -1:
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

    # Store last confirmed swing high/low price (forward-filled)
    df["last_sh_price"] = df["high"].where(df["swing_high"] == 1).ffill()
    df["last_sl_price"] = df["low"].where(df["swing_low"]  == 1).ffill()

    df["dist_to_last_sh"] = (df["close"] - df["last_sh_price"]) / df["atr"]
    df["dist_to_last_sl"] = (df["close"] - df["last_sl_price"]) / df["atr"]

    return df


# ─────────────────────────────────────────────────────────────
# CHoCH — Change of Character
# ─────────────────────────────────────────────────────────────

def _choch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bullish CHoCH : in a downtrend (trend == -1), close breaks ABOVE the
                    most recent lower-high (lh).
    Bearish CHoCH : in an uptrend  (trend == +1), close breaks BELOW the
                    most recent higher-low (hl).

    choch column: +1 bullish CHoCH, -1 bearish CHoCH, 0 none.
    """
    close_vals = df["close"].values
    trend_vals = df["trend"].values
    high_vals  = df["high"].values
    low_vals   = df["low"].values
    lh_vals    = df["lh"].values
    hl_vals    = df["hl"].values
    size = len(df)

    choch = np.zeros(size, dtype=np.int8)

    # Track last lower-high price and last higher-low price
    last_lh_price = np.nan
    last_hl_price = np.nan

    for i in range(size):
        if lh_vals[i]:
            last_lh_price = high_vals[i]
        if hl_vals[i]:
            last_hl_price = low_vals[i]

        # Bullish CHoCH: downtrend + close above last lower-high
        if trend_vals[i] <= 0 and not np.isnan(last_lh_price):
            if close_vals[i] > last_lh_price:
                choch[i] = 1
                last_lh_price = np.nan  # reset after CHoCH

        # Bearish CHoCH: uptrend + close below last higher-low
        if trend_vals[i] >= 0 and not np.isnan(last_hl_price):
            if close_vals[i] < last_hl_price:
                choch[i] = -1
                last_hl_price = np.nan  # reset after CHoCH

    df["choch"] = choch
    return df


# ─────────────────────────────────────────────────────────────
# Fair Value Gap (FVG)
# ─────────────────────────────────────────────────────────────

def _fvg(df: pd.DataFrame) -> pd.DataFrame:
    """
    A 3-candle imbalance (Fair Value Gap):

    Bullish FVG:  candle[i-2].high < candle[i].low  (gap up)
    Bearish FVG:  candle[i-2].low  > candle[i].high (gap down)

    Stores the most-recent FVG's top, bottom, and size (in ATR units).
    """
    high_vals = df["high"].values
    low_vals  = df["low"].values
    atr_vals  = df["atr"].values
    size = len(df)

    fvg_bull  = np.zeros(size, dtype=np.int8)
    fvg_bear  = np.zeros(size, dtype=np.int8)
    fvg_top   = np.full(size, np.nan)
    fvg_bot   = np.full(size, np.nan)
    fvg_size  = np.full(size, np.nan)

    last_bull_top = np.nan
    last_bull_bot = np.nan
    last_bear_top = np.nan
    last_bear_bot = np.nan

    for i in range(2, size):
        gap_up   = low_vals[i]  - high_vals[i - 2]
        gap_down = low_vals[i - 2] - high_vals[i]

        if gap_up > 0:                        # Bullish FVG
            fvg_bull[i]   = 1
            last_bull_top = low_vals[i]
            last_bull_bot = high_vals[i - 2]
            if atr_vals[i] > 0:
                fvg_size[i] = gap_up / atr_vals[i]

        if gap_down > 0:                      # Bearish FVG
            fvg_bear[i]   = 1
            last_bear_top = low_vals[i - 2]
            last_bear_bot = high_vals[i]
            if atr_vals[i] > 0:
                fvg_size[i] = gap_down / atr_vals[i]

        # Forward-fill most recent FVG boundaries
        fvg_top[i] = last_bull_top if not np.isnan(last_bull_top) else last_bear_top
        fvg_bot[i] = last_bull_bot if not np.isnan(last_bull_bot) else last_bear_bot

    df["fvg_bull"] = fvg_bull
    df["fvg_bear"] = fvg_bear
    df["fvg_top"]  = pd.Series(fvg_top,  index=df.index).ffill()
    df["fvg_bot"]  = pd.Series(fvg_bot,  index=df.index).ffill()
    df["fvg_size"] = pd.Series(fvg_size, index=df.index).ffill().fillna(0)

    return df


# ─────────────────────────────────────────────────────────────
# Order Block (OB)
# ─────────────────────────────────────────────────────────────

def _order_block(df: pd.DataFrame) -> pd.DataFrame:
    """
    The last OPPOSING candle immediately before a displacement move.

    Bullish OB : last bearish candle before a bullish impulse
                 (candle_dir == -1, followed within 3 bars by impulse == +1)
    Bearish OB : last bullish candle before a bearish impulse

    Stores OB high/low and size in ATR units.
    """
    candle_dir = df["candle_dir"].values
    impulse    = df["impulse"].values if "impulse" in df.columns else np.zeros(len(df))
    high_vals  = df["high"].values
    low_vals   = df["low"].values
    atr_vals   = df["atr"].values
    size = len(df)

    ob_bull_flag = np.zeros(size, dtype=np.int8)
    ob_bear_flag = np.zeros(size, dtype=np.int8)

    bull_ob_high = np.full(size, np.nan)
    bull_ob_low  = np.full(size, np.nan)
    bear_ob_high = np.full(size, np.nan)
    bear_ob_low  = np.full(size, np.nan)
    ob_size      = np.full(size, np.nan)

    last_bull_ob_h = np.nan
    last_bull_ob_l = np.nan
    last_bear_ob_h = np.nan
    last_bear_ob_l = np.nan

    for i in range(3, size):
        # Bullish OB: look for bearish candle followed within 3 bars by bull impulse
        if impulse[i] == 1:
            for j in range(i - 1, max(i - 4, 0), -1):
                if candle_dir[j] == -1:
                    ob_bull_flag[j] = 1
                    last_bull_ob_h = high_vals[j]
                    last_bull_ob_l = low_vals[j]
                    if atr_vals[i] > 0:
                        ob_size[i] = (last_bull_ob_h - last_bull_ob_l) / atr_vals[i]
                    break

        # Bearish OB: look for bullish candle followed within 3 bars by bear impulse
        if impulse[i] == -1:
            for j in range(i - 1, max(i - 4, 0), -1):
                if candle_dir[j] == 1:
                    ob_bear_flag[j] = 1
                    last_bear_ob_h = high_vals[j]
                    last_bear_ob_l = low_vals[j]
                    if atr_vals[i] > 0:
                        ob_size[i] = (last_bear_ob_h - last_bear_ob_l) / atr_vals[i]
                    break

        bull_ob_high[i] = last_bull_ob_h
        bull_ob_low[i]  = last_bull_ob_l
        bear_ob_high[i] = last_bear_ob_h
        bear_ob_low[i]  = last_bear_ob_l

    df["ob_bull_flag"] = ob_bull_flag
    df["ob_bear_flag"] = ob_bear_flag
    df["bull_ob_high"] = pd.Series(bull_ob_high, index=df.index).ffill()
    df["bull_ob_low"]  = pd.Series(bull_ob_low,  index=df.index).ffill()
    df["bear_ob_high"] = pd.Series(bear_ob_high, index=df.index).ffill()
    df["bear_ob_low"]  = pd.Series(bear_ob_low,  index=df.index).ffill()
    df["ob_size"]      = pd.Series(ob_size,       index=df.index).ffill().fillna(0)

    return df


# ─────────────────────────────────────────────────────────────
# Liquidity features (equal highs/lows)
# ─────────────────────────────────────────────────────────────

def _liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    atr  = df["atr"].values
    high = df["high"].values
    low  = df["low"].values
    size = len(df)

    eq_high = np.zeros(size, dtype=np.int8)
    eq_low  = np.zeros(size, dtype=np.int8)

    lookback = 20
    for i in range(lookback, size):
        threshold = EQ_THRESHOLD_ATR_PCT * atr[i]
        if np.any(np.abs(high[i - lookback : i] - high[i]) < threshold):
            eq_high[i] = 1
        if np.any(np.abs(low[i - lookback : i]  - low[i])  < threshold):
            eq_low[i] = 1

    df["eq_high"] = eq_high
    df["eq_low"]  = eq_low

    df["prev_50_high"] = df["high"].rolling(50).max().shift(1)
    df["prev_50_low"]  = df["low"].rolling(50).min().shift(1)
    df["dist_prev_50_high"] = (df["prev_50_high"] - df["close"]) / df["atr"]
    df["dist_prev_50_low"]  = (df["close"] - df["prev_50_low"])  / df["atr"]

    return df


# ─────────────────────────────────────────────────────────────
# Previous-day high / low
# ─────────────────────────────────────────────────────────────

def _prev_day_hl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Track the prior trading day's high and low.
    Extremely important liquidity levels in ICT methodology.
    """
    daily_high = df["high"].resample("1D").max().shift(1)
    daily_low  = df["low"].resample("1D").min().shift(1)

    df["prev_day_high"] = daily_high.reindex(df.index, method="ffill")
    df["prev_day_low"]  = daily_low.reindex(df.index, method="ffill")
    df["dist_prev_day_high"] = (df["prev_day_high"] - df["close"]) / df["atr"]
    df["dist_prev_day_low"]  = (df["close"] - df["prev_day_low"])  / df["atr"]
    return df


# ─────────────────────────────────────────────────────────────
# Session high/low
# ─────────────────────────────────────────────────────────────

def _session_hl(df: pd.DataFrame) -> pd.DataFrame:
    sessions = {
        "asia":   (0,  7),
        "london": (7,  12),
        "ny":     (12, 17),
    }
    df["_date_str"] = df.index.strftime("%Y-%m-%d")

    for name, (h_start, h_end) in sessions.items():
        mask = (df["hour"] >= h_start) & (df["hour"] < h_end)
        hi_col = f"{name}_high"
        lo_col = f"{name}_low"

        session_bars = df[mask].copy()
        if session_bars.empty:
            df[hi_col] = np.nan
            df[lo_col] = np.nan
        else:
            sh = session_bars.groupby("_date_str")["high"].max().rename(hi_col)
            sl = session_bars.groupby("_date_str")["low"].min().rename(lo_col)
            df[hi_col] = df["_date_str"].map(sh)
            df[lo_col] = df["_date_str"].map(sl)

        # Forward-fill then backward-fill so all bars have a reference level
        df[hi_col] = df[hi_col].ffill().bfill()
        df[lo_col] = df[lo_col].ffill().bfill()

        df[f"{name}_high_dist"] = (df[hi_col] - df["close"]) / df["atr"]
        df[f"{name}_low_dist"]  = (df["close"] - df[lo_col])  / df["atr"]

    df.drop(columns=["_date_str"], inplace=True)
    return df


# ─────────────────────────────────────────────────────────────
# Liquidity sweep detection
# ─────────────────────────────────────────────────────────────

def _liquidity_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects wick-based sweeps of key liquidity levels:
      - Previous swing high/low
      - Equal highs/lows
      - Previous day high/low
      - Session (Asia) range

    bull_sweep : price swept a LOW level, closed back above it
    bear_sweep : price swept a HIGH level, closed back below it
    sweep_strength : how far price went past the level (ATR-normalised)
    """
    atr   = df["atr"].values
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    open_ = df["open"].values
    size  = len(df)

    last_sl      = df["last_sl_price"].values if "last_sl_price" in df.columns \
                   else df["low"].where(df["swing_low"] == 1).ffill().values
    last_sh      = df["last_sh_price"].values if "last_sh_price" in df.columns \
                   else df["high"].where(df["swing_high"] == 1).ffill().values
    prev_day_h   = df["prev_day_high"].values if "prev_day_high" in df.columns \
                   else np.full(size, np.nan)
    prev_day_l   = df["prev_day_low"].values  if "prev_day_low" in df.columns \
                   else np.full(size, np.nan)
    asia_h       = df["asia_high"].values if "asia_high" in df.columns else np.full(size, np.nan)
    asia_l       = df["asia_low"].values  if "asia_low"  in df.columns else np.full(size, np.nan)

    bull_sweep      = np.zeros(size, dtype=np.int8)
    bear_sweep      = np.zeros(size, dtype=np.int8)
    sweep_strength  = np.zeros(size, dtype=np.float32)

    for i in range(1, size):
        min_wick = SWEEP_WICK_ATR_PCT * atr[i]
        if atr[i] == 0:
            continue

        lower_wick = min(open_[i], close[i]) - low[i]
        upper_wick = high[i] - max(open_[i], close[i])

        swept_low = np.nan
        swept_high = np.nan

        # ── Bullish sweep candidates (swept a LOW level) ─────────────
        for level in [last_sl[i], prev_day_l[i], asia_l[i]]:
            if np.isnan(level):
                continue
            if low[i] < level and close[i] > level and lower_wick > min_wick:
                excess = (level - low[i]) / atr[i]
                if np.isnan(swept_low) or excess > sweep_strength[i]:
                    swept_low = level
                    bull_sweep[i] = 1
                    sweep_strength[i] = float(excess)

        # ── Bearish sweep candidates (swept a HIGH level) ────────────
        for level in [last_sh[i], prev_day_h[i], asia_h[i]]:
            if np.isnan(level):
                continue
            if high[i] > level and close[i] < level and upper_wick > min_wick:
                excess = (high[i] - level) / atr[i]
                if np.isnan(swept_high) or excess > sweep_strength[i]:
                    swept_high = level
                    bear_sweep[i] = 1
                    sweep_strength[i] = float(excess)

    df["bull_sweep"]     = bull_sweep
    df["bear_sweep"]     = bear_sweep
    df["sweep_strength"] = sweep_strength
    return df


# ─────────────────────────────────────────────────────────────
# Impulse / displacement
# ─────────────────────────────────────────────────────────────

def _impulse(df: pd.DataFrame) -> pd.DataFrame:
    body_ratio = df["body"] / df["body_avg"].replace(0, np.nan)
    direction  = df["candle_dir"]
    impulse = np.where(body_ratio > IMPULSE_BODY_MULT, direction, 0)
    df["impulse"]    = impulse.astype(np.int8)
    df["body_ratio"] = body_ratio.fillna(0)
    return df


# ─────────────────────────────────────────────────────────────
# Discount / premium zone
# ─────────────────────────────────────────────────────────────

def _discount_zone(df: pd.DataFrame) -> pd.DataFrame:
    """
    50% OTE (Optimal Trade Entry) zone:
      Bullish discount: current price < midpoint of last 50-bar range (buy cheap)
      Bearish premium:  current price > midpoint of last 50-bar range (sell high)

    entry_in_discount = 1 if price is in discount (for longs) or premium (for shorts).
    """
    swing_range_high = df["high"].rolling(50).max()
    swing_range_low  = df["low"].rolling(50).min()
    midpoint = (swing_range_high + swing_range_low) / 2

    df["swing_midpoint"]    = midpoint
    df["entry_in_discount"] = (df["close"] < midpoint).astype(np.int8)
    return df


# ─────────────────────────────────────────────────────────────
# HTF trend merge
# ─────────────────────────────────────────────────────────────

def _merge_htf_trend(df: pd.DataFrame, htf_df: pd.DataFrame) -> pd.DataFrame:
    htf = htf_df.copy()
    htf = _atr(htf, ATR_PERIOD)
    htf = _swing_points(htf, SWING_LOOKBACK)
    htf = _market_structure(htf)
    htf_trend = htf["trend"].rename("htf_trend")
    df["htf_trend"] = htf_trend.reindex(df.index, method="ffill").fillna(0).astype(np.int8)
    return df


# ─────────────────────────────────────────────────────────────
# Helper: NaN filling
# ─────────────────────────────────────────────────────────────

def _fill_warmup_nans(df: pd.DataFrame) -> None:
    """
    Fills NaNs in features that have warm-up periods or gaps.
    Modifies df in-place.
    """
    # Structural levels: forward-fill then backward-fill (to handle the very beginning)
    ff_cols = [
        "last_sh_price", "last_sl_price",
        "prev_50_high", "prev_50_low",
        "prev_day_high", "prev_day_low",
        "asia_high", "asia_low",
        "london_high", "london_low",
        "ny_high", "ny_low",
        "fvg_top", "fvg_bot",
        "bull_ob_high", "bull_ob_low",
        "bear_ob_high", "bear_ob_low",
        "swing_midpoint"
    ]
    for col in ff_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    # Distances or relative metrics: fill with 0 (neutral) if still NaN
    zero_cols = [
        "dist_to_last_sh", "dist_to_last_sl",
        "dist_prev_50_high", "dist_prev_50_low",
        "dist_prev_day_high", "dist_prev_day_low",
        "asia_high_dist", "asia_low_dist",
        "london_high_dist", "london_low_dist",
        "ny_high_dist", "ny_low_dist",
        "fvg_size", "ob_size",
        "body_pct", "body_avg", "range_expansion", "body_ratio",
        "atr_percentile", "momentum_persistence"
    ]
    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
