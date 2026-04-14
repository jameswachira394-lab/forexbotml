"""
audit_labeler.py - Analyze why setup win rates are so low (20-25%)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from data.loader import OHLCVLoader
from data.histdata_parser import parse_histdata
from features.engineer_fixed import engineer_features
from features.labeler import SetupLabeler, LabelConfig
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# === AUDIT EURUSD (small dataset) ===
csv_file = Path("data/raw/EURUSD_M5_mt5.csv")
logger.info(f"\n{'='*80}")
logger.info(f"AUDITING: {csv_file.name}")
logger.info(f"{'='*80}\n")

# Parse data
df_m1 = parse_histdata(str(csv_file), target_tf="M1")
logger.info(f"Parsed: {len(df_m1):,} M1 bars")

# Resample to M15
loader = OHLCVLoader.__new__(OHLCVLoader)
loader.timeframe = "M1"
df_m15 = loader.resample(df_m1, "M15")
df_h4 = loader.resample(df_m1, "H4")
logger.info(f"Resampled: {len(df_m15):,} M15 bars, {len(df_h4):,} H4 bars\n")

# Engineer features
df_feat = engineer_features(df_m15, htf_df=df_h4)
logger.info(f"Features: {len(df_feat):,} rows\n")

# Label setups
labeler = SetupLabeler(LabelConfig())
df_labeled = labeler.label(df_feat)

if len(df_labeled) == 0:
    logger.error("No labeled setups!")
    exit(1)

# === ANALYSIS ===
logger.info("\n" + "="*80)
logger.info("LABELED SETUP STATISTICS")
logger.info("="*80)

logger.info(f"\nWin Rate: {df_labeled['label'].mean():.1%}")
logger.info(f"Losses: {(df_labeled['label']==0).sum():,}")
logger.info(f"Wins: {(df_labeled['label']==1).sum():,}")
logger.info(f"Long: {(df_labeled['direction']==1).sum():,}")
logger.info(f"Short: {(df_labeled['direction']==-1).sum():,}")

# === EXAMINE A FEW SETUPS ===
logger.info("\n" + "="*80)
logger.info("SAMPLE SETUPS (First 10)")
logger.info("="*80 + "\n")

for idx, (ts, row) in enumerate(df_labeled.head(10).iterrows()):
    if idx >= 10:
        break
    
    direction_str = "LONG" if row['direction'] == 1 else "SHORT"
    outcome = "WIN" if row['label'] == 1 else "LOSS"
    
    logger.info(f"Setup {idx+1}: {direction_str} {outcome}")
    logger.info(f"  Entry:  {row['entry_price']:.5f}")
    logger.info(f"  SL:     {row['sl_price']:.5f} (dist: {abs(row['sl_price'] - row['entry_price']):.5f})")
    logger.info(f"  TP:     {row['tp_price']:.5f} (dist: {abs(row['tp_price'] - row['entry_price']):.5f})")
    logger.info(f"  R:R:    {row['rr_achieved']:.2f}")
    logger.info(f"  Bars to outcome: {row['time_to_outcome']:.0f}")
    logger.info("")

# === CHECK FOR PATHOLOGIES ===
logger.info("="*80)
logger.info("PATHOLOGY CHECKS")
logger.info("="*80 + "\n")

# 1. SL/TP distances
sl_dist = np.abs(df_labeled['sl_price'] - df_labeled['entry_price'])
tp_dist = np.abs(df_labeled['tp_price'] - df_labeled['entry_price'])
ratio = tp_dist / (sl_dist + 1e-9)

logger.info(f"SL Distance: mean={sl_dist.mean():.5f}, std={sl_dist.std():.5f}")
logger.info(f"TP Distance: mean={tp_dist.mean():.5f}, std={tp_dist.std():.5f}")
logger.info(f"Ratio (TP/SL): mean={ratio.mean():.2f}, std={ratio.std():.2f}")
logger.info(f"Expected ratio: 2.0 (if 1.5 ATR SL, 3.0 ATR TP)\n")

# 2. Entry price location (should be reasonable pullback)
logger.info(f"Time to outcome:")
logger.info(f"  Mean: {df_labeled['time_to_outcome'].mean():.1f} bars")
logger.info(f"  Median: {df_labeled['time_to_outcome'].median():.1f} bars")
logger.info(f"  Min: {df_labeled['time_to_outcome'].min():.0f} bars")
logger.info(f"  Max: {df_labeled['time_to_outcome'].max():.0f} bars")

hits_1bar = (df_labeled['time_to_outcome'] == 1).sum()
logger.info(f"  Setups that hit SL/TP on BAR 1: {hits_1bar} ({hits_1bar/len(df_labeled):.1%})")
logger.info("")

# 3. Direction balance
long_wins = ((df_labeled['direction']==1) & (df_labeled['label']==1)).sum()
long_cnt = (df_labeled['direction']==1).sum()
short_wins = ((df_labeled['direction']==-1) & (df_labeled['label']==1)).sum()
short_cnt = (df_labeled['direction']==-1).sum()

logger.info(f"Win rate by direction:")
logger.info(f"  LONG: {long_wins}/{long_cnt} = {long_wins/max(long_cnt,1):.1%}")
logger.info(f"  SHORT: {short_wins}/{short_cnt} = {short_wins/max(short_cnt,1):.1%}")
logger.info("")

# 4. Entry price outliers - are entries placed at very extreme levels?
logger.info(f"Entry price percentile stats:")
for q in [0, 25, 50, 75, 100]:
    val = df_labeled['entry_price'].quantile(q/100)
    logger.info(f"  {q}th percentile: {val:.5f}")

logger.info("\n" + "="*80)
logger.info("CONCLUSION")
logger.info("="*80)
logger.info("""
If win rate is ~25%:
  1. Check if SL is being hit immediately (bar 1)
  2. Check if entry prices are at resistance (not pullback support)
  3. Check if pullback detection is too shallow (20% of BOS)
  4. Consider that pure setup quality is just 25% + ML filter on top
  
If SL is hit on bar 1 frequently:
  → Entry placement is wrong (at wrong level)
  
If ratio TP/SL is not 2.0:
  → Labeler calculation is inconsistent
  
If long vs short win rates differ significantly:
  → Market has directional bias
""")
logger.info("="*80)
