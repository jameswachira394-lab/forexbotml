"""
Debug script to examine engineered features
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from data.histdata_parser import parse_histdata
from data.loader import OHLCVLoader, load_multi_timeframe
from features.engineer import engineer_features

# Load data
symbol = "XAUUSD"
data_path = "data/raw/XAUUSD_M5_mt5.csv"

print("Loading data...")
base_df = parse_histdata(data_path, symbol=symbol, target_tf=cfg.BASE_TF)
loader  = OHLCVLoader.__new__(OHLCVLoader)
loader.timeframe = "M1"
htf_df  = loader.resample(base_df, cfg.HTF_FOR_TREND)

print(f"Base TF bars: {len(base_df)}")
print(f"HTF bars: {len(htf_df)}")

# Engineer features
feat_df = engineer_features(base_df, htf_df=htf_df)
print(f"\nEngineered features: {len(feat_df)} rows")
print(f"Columns ({len(feat_df.columns)}):")
for i, col in enumerate(feat_df.columns):
    print(f"  {i+1:2d}. {col}")

# Check for sweep/BOS features
sweep_cols = [c for c in feat_df.columns if 'sweep' in c.lower() or 'bos' in c.lower() or 'displacement' in c.lower()]
print(f"\nSetup-related columns: {sweep_cols}")

# Sample rows
print("\nSample row 1000:")
row = feat_df.iloc[1000]
for col in sweep_cols:
    print(f"  {col}: {row.get(col, 'N/A')}")

print("\nSample row 5000:")
row = feat_df.iloc[5000]
for col in sweep_cols:
    print(f"  {col}: {row.get(col, 'N/A')}")

# Check for non-zero sweep values
sweep_detected = feat_df[[c for c in feat_df.columns if 'sweep' in c.lower()]].sum()
bos_detected = feat_df[[c for c in feat_df.columns if 'bos' in c.lower()]].sum()

print(f"\nSweep detections: {sweep_detected.to_dict()}")
print(f"BOS detections: {bos_detected.to_dict()}")
