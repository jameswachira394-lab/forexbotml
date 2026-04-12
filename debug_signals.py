"""
debug_signals.py  v2 — post-BOS-fix diagnostic
Shows gate-by-gate funnel WITH the first-BOS-only fix, then scores ML probabilities.
Run: python -X utf8 debug_signals.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import logging
logging.basicConfig(level=logging.WARNING)

from data.histdata_parser import parse_histdata
from data.loader import OHLCVLoader
from features.engineer import engineer_features
from models.ml_model import ForexMLModel
import config as cfg

# ── 1. Load & feature-engineer ──────────────────────────────────────────────
data_path = "data/raw/XAUUSD_M5_mt5.csv"
symbol    = "XAUUSD"

base_df = parse_histdata(data_path, symbol=symbol, target_tf=cfg.BASE_TF)
loader  = OHLCVLoader.__new__(OHLCVLoader)
loader.timeframe = "M1"
htf_df  = loader.resample(base_df, cfg.HTF_FOR_TREND)
feat_df = engineer_features(base_df, htf_df=htf_df)

print(f"\n{'='*60}")
print(f"  XAUUSD  |  {len(feat_df):,} bars  |  {cfg.BASE_TF}")
print(f"  PULLBACK_ATR: [{cfg.PULLBACK_ATR_MIN}, {cfg.PULLBACK_ATR_MAX}]")
print(f"  ML_THRESHOLD: {cfg.ML_THRESHOLD}")
print(f"{'='*60}")

# ── 2. Load model ─────────────────────────────────────────────────────────
model = ForexMLModel(model_name=f"{symbol}_{cfg.MODEL_NAME}")
try:
    model.load()
    print(f"  Model loaded OK  |  saved threshold: {model.threshold:.3f}")
except FileNotFoundError:
    model = None
    print("  WARNING: No model found — will skip ML scoring")

# ── 3. Gate funnel with FIRST-BOS-only fix ──────────────────────────────
bull_sw  = feat_df["bull_sweep"].values
bear_sw  = feat_df["bear_sweep"].values
bos_arr  = feat_df["bos"].values
htf_tr   = feat_df["htf_trend"].values
atr      = feat_df["atr"].values
close    = feat_df["close"].values
high_v   = feat_df["high"].values
low_v    = feat_df["low"].values
size     = len(feat_df)

MAX_GAP      = 50
PULLBACK_MIN = cfg.PULLBACK_ATR_MIN
PULLBACK_MAX = cfg.PULLBACK_ATR_MAX
SL_BUFFER    = cfg.SL_BUFFER_ATR

last_bull_sweep_i    = -1
last_bear_sweep_i    = -1
last_bull_bos_i      = -1
last_bear_bos_i      = -1
last_bull_sweep_low  = np.nan
last_bear_sweep_high = np.nan

c = dict(
    LONG_has_sweep=0, LONG_bos_after_sweep=0, LONG_within_gap=0,
    LONG_htf_ok=0, LONG_pb_too_small=0, LONG_pb_too_large=0,
    LONG_pb_ok=0, LONG_sl_ok=0, LONG_ml_scored=0,
    SHORT_has_sweep=0, SHORT_bos_after_sweep=0, SHORT_within_gap=0,
    SHORT_htf_ok=0, SHORT_pb_too_small=0, SHORT_pb_too_large=0,
    SHORT_pb_ok=0, SHORT_sl_ok=0, SHORT_ml_scored=0,
)

ml_probs_long  = []
ml_probs_short = []
pb_long_ok_vals = []
pb_short_ok_vals = []

for i in range(1, size):
    prev = i - 1

    if bull_sw[prev]:
        last_bull_sweep_i   = prev
        last_bull_sweep_low = low_v[prev]
        # Reset BOS tracker for new sweep
        last_bull_bos_i = -1

    if bear_sw[prev]:
        last_bear_sweep_i    = prev
        last_bear_sweep_high = high_v[prev]
        last_bear_bos_i = -1

    # FIRST-BOS-ONLY: only record BOS if we haven't seen one since the sweep
    if bos_arr[prev] == 1  and last_bull_bos_i <  0:
        last_bull_bos_i = prev
    if bos_arr[prev] == -1 and last_bear_bos_i < 0:
        last_bear_bos_i = prev

    bar_atr = max(float(atr[i]), 1e-9)

    # ── LONG ────────────────────────────────────────────────
    if last_bull_sweep_i >= 0:
        c["LONG_has_sweep"] += 1
        if last_bull_bos_i > last_bull_sweep_i:
            c["LONG_bos_after_sweep"] += 1
            if (i - last_bull_sweep_i) <= MAX_GAP:
                c["LONG_within_gap"] += 1
                if not cfg.REQUIRE_HTF_ALIGN or htf_tr[i] >= 0:
                    c["LONG_htf_ok"] += 1
                    bos_close = close[last_bull_bos_i]
                    pb_dist   = (bos_close - close[i]) / bar_atr
                    if pb_dist < PULLBACK_MIN:
                        c["LONG_pb_too_small"] += 1
                    elif pb_dist > PULLBACK_MAX:
                        c["LONG_pb_too_large"] += 1
                    else:
                        c["LONG_pb_ok"] += 1
                        pb_long_ok_vals.append(pb_dist)
                        ep = float(close[i])
                        sl = last_bull_sweep_low - SL_BUFFER * bar_atr \
                             if not np.isnan(last_bull_sweep_low) else ep - bar_atr
                        if (ep - sl) > 0:
                            c["LONG_sl_ok"] += 1
                            if model is not None:
                                row = feat_df.iloc[[i]].copy()
                                row["direction"] = 1
                                try:
                                    prob = float(model.predict_proba(row)[0])
                                    ml_probs_long.append(prob)
                                    c["LONG_ml_scored"] += 1
                                except Exception:
                                    pass
                        last_bull_sweep_i = -1
                        last_bull_bos_i   = -1

    # ── SHORT ───────────────────────────────────────────────
    if last_bear_sweep_i >= 0:
        c["SHORT_has_sweep"] += 1
        if last_bear_bos_i > last_bear_sweep_i:
            c["SHORT_bos_after_sweep"] += 1
            if (i - last_bear_sweep_i) <= MAX_GAP:
                c["SHORT_within_gap"] += 1
                if not cfg.REQUIRE_HTF_ALIGN or htf_tr[i] <= 0:
                    c["SHORT_htf_ok"] += 1
                    bos_close = close[last_bear_bos_i]
                    pb_dist   = (close[i] - bos_close) / bar_atr
                    if pb_dist < PULLBACK_MIN:
                        c["SHORT_pb_too_small"] += 1
                    elif pb_dist > PULLBACK_MAX:
                        c["SHORT_pb_too_large"] += 1
                    else:
                        c["SHORT_pb_ok"] += 1
                        pb_short_ok_vals.append(pb_dist)
                        ep = float(close[i])
                        sl = last_bear_sweep_high + SL_BUFFER * bar_atr \
                             if not np.isnan(last_bear_sweep_high) else ep + bar_atr
                        if (sl - ep) > 0:
                            c["SHORT_sl_ok"] += 1
                            if model is not None:
                                row = feat_df.iloc[[i]].copy()
                                row["direction"] = -1
                                try:
                                    prob = float(model.predict_proba(row)[0])
                                    ml_probs_short.append(prob)
                                    c["SHORT_ml_scored"] += 1
                                except Exception:
                                    pass
                        last_bear_sweep_i = -1
                        last_bear_bos_i   = -1

print(f"\n[GATE FUNNEL — LONG setups]")
keys_l = [k for k in c if k.startswith("LONG")]
for k in keys_l:
    print(f"  {k:<30}: {c[k]:>8,}")

print(f"\n[GATE FUNNEL — SHORT setups]")
keys_s = [k for k in c if k.startswith("SHORT")]
for k in keys_s:
    print(f"  {k:<30}: {c[k]:>8,}")

# ── 4. ML probability distribution ─────────────────────────────────────
for label, probs, thresh in [
    ("LONG",  ml_probs_long,  cfg.ML_THRESHOLD),
    ("SHORT", ml_probs_short, cfg.ML_THRESHOLD),
]:
    if not probs:
        print(f"\n[ML probs {label}]: No samples scored")
        continue
    arr = np.array(probs)
    print(f"\n[ML PROBABILITY DISTRIBUTION — {label}  (n={len(arr):,})]")
    print(f"  min={arr.min():.4f}  p10={np.percentile(arr,10):.4f}  "
          f"p25={np.percentile(arr,25):.4f}  median={np.median(arr):.4f}  "
          f"p75={np.percentile(arr,75):.4f}  p90={np.percentile(arr,90):.4f}  "
          f"p95={np.percentile(arr,95):.4f}  max={arr.max():.4f}")
    for t in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.80]:
        count = (arr >= t).sum()
        print(f"  >= {t:.2f} : {count:>5,}  ({count/len(arr):.1%})")

print(f"\n{'='*60}")
print("  Diagnosis complete (v2 — first-BOS-only fix applied).")
print(f"{'='*60}\n")
