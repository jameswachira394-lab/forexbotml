#!/usr/bin/env python
"""
train_fixed.py — Train ML models using the institutional-grade FIXED system

Usage:
  python train_fixed.py --symbol EURUSD --data data/raw/EURUSD.csv
  python train_fixed.py --symbol XAUUSD  (uses config.py path)
  python train_fixed.py --all  (all symbols from config)

Features:
  ✓ Uses engineer_fixed.py (NO LEAKAGE)
  ✓ Uses ml_integration_fixed.py (CALIBRATED)
  ✓ Walk-forward compatible
  ✓ TimeSeriesSplit CV (no future bleed)
  ✓ Full diagnostics output
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from data.histdata_parser import parse_histdata
from data.loader import OHLCVLoader
from features.engineer_fixed import engineer_features
from features.labeler import SetupLabeler, LabelConfig, get_feature_columns
from models.ml_integration_fixed import ForexMLModelFixed, get_ml_feature_columns


def train_symbol_fixed(
    symbol: str,
    csv_paths: list,
    base_tf: str = "M15",
    htf: str = "H4",
    model_name: str = "forex_xgb",
) -> dict:
    """
    Train model for one symbol using FIXED system (NO LEAKAGE).
    
    Parameters
    ----------
    symbol : str
        Symbol name (e.g., 'EURUSD')
    csv_paths : list
        List of CSV file paths for this symbol
    base_tf : str
        Base timeframe (e.g., 'M15')
    htf : str
        Higher timeframe for trend (e.g., 'H4')
    model_name : str
        Base model name
    
    Returns
    -------
    dict with training metrics
    """
    
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING {symbol} - FIXED SYSTEM")
    logger.info(f"{'='*80}\n")
    
    try:
        # ── 1. Load and parse CSV ─────────────────────────────────────────────
        logger.info(f"[1/6] Loading CSV files...")
        frames = []
        for csv_path in csv_paths:
            if not Path(csv_path).exists():
                logger.warning(f"  ⚠ File not found: {csv_path}")
                continue
            
            try:
                logger.info(f"  ✓ Parsing {Path(csv_path).name}...")
                df = parse_histdata(csv_path, symbol=symbol, target_tf="M1")
                logger.info(f"    → {len(df):,} M1 bars loaded")
                frames.append(df)
            except Exception as e:
                logger.error(f"  ✗ Parse failed: {e}")
        
        if not frames:
            logger.error(f"No valid CSV data loaded for {symbol}")
            return None
        
        # Merge and deduplicate
        df_m1 = pd.concat(frames).sort_index()
        df_m1 = df_m1[~df_m1.index.duplicated(keep='last')]
        logger.info(f"  ✓ Total M1 bars: {len(df_m1):,}")
        logger.info(f"    Date range: {df_m1.index[0].date()} → {df_m1.index[-1].date()}")
        
        # ── 2. Resample to base TF and HTF ────────────────────────────────────
        logger.info(f"\n[2/6] Resampling to {base_tf} and {htf}...")
        loader = OHLCVLoader.__new__(OHLCVLoader)
        loader.timeframe = "M1"
        
        df_base = loader.resample(df_m1, base_tf)
        df_htf = loader.resample(df_m1, htf)
        
        logger.info(f"  ✓ {base_tf}: {len(df_base):,} bars")
        logger.info(f"  ✓ {htf}: {len(df_htf):,} bars")
        
        if len(df_base) < 500:
            logger.error(f"Too few bars ({len(df_base)} < 500)")
            return None
        
        # ── 3. Feature engineering (FIXED - NO LEAKAGE) ────────────────────────
        logger.info(f"\n[3/6] Engineering features (FIXED system, NO leakage)...")
        try:
            df_feat = engineer_features(df_base, htf_df=df_htf)
            logger.info(f"  ✓ Features engineered: {len(df_feat):,} rows")
            logger.info(f"    Columns: {df_feat.shape[1]}")
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return None
        
        # ── 4. Labeling setups ────────────────────────────────────────────────
        logger.info(f"\n[4/6] Labeling setups...")
        try:
            label_cfg = LabelConfig(
                rr_ratio=cfg.TP_ATR_MULT / cfg.SL_ATR_MULT,
                sl_atr_mult=cfg.SL_ATR_MULT,
                max_bars_to_bos=20,
                max_bars_to_entry=25,
            )
            labeler = SetupLabeler(label_cfg)
            df_labeled = labeler.label(df_feat)
            
            num_labeled = len(df_labeled)
            win_rate = df_labeled["label"].mean()
            
            logger.info(f"  ✓ Labeled: {num_labeled:,} setups")
            logger.info(f"    Win rate: {win_rate:.1%}")
            
            if num_labeled < 30:
                logger.error(f"Too few labeled setups ({num_labeled} < 30)")
                return None
        except Exception as e:
            logger.error(f"Labeling failed: {e}")
            return None
        
        # ── 5. Prepare features for ML ────────────────────────────────────────
        logger.info(f"\n[5/6] Preparing ML features...")
        try:
            # Get ML features only (remove redundancy)
            feat_cols = get_ml_feature_columns(df_labeled)
            logger.info(f"  ✓ ML features: {len(feat_cols)} columns")
            
            X = df_labeled[feat_cols].fillna(0)
            y = df_labeled["label"]
            
            logger.info(f"    Features shape: {X.shape}")
            logger.info(f"    Labels: {len(y)}")
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return None
        
        # ── 6. Train model (CALIBRATED) ───────────────────────────────────────
        logger.info(f"\n[6/6] Training model with calibration...")
        try:
            model = ForexMLModelFixed(f"{symbol}_{model_name}")
            
            metrics = model.train(
                X=X,
                y=y,
                test_size=0.15,
                val_size=0.15,
                n_cv_splits=5,
                rr_ratio=cfg.TP_ATR_MULT / cfg.SL_ATR_MULT,
            )
            
            # Save model
            model.save()
            
            # Display results
            logger.info(f"\n{'─'*80}")
            logger.info(f"TRAINING COMPLETE: {symbol}")
            logger.info(f"{'─'*80}")
            logger.info(f"  CV AUC:           {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
            logger.info(f"  Test AUC:         {metrics['test_auc']:.4f}")
            logger.info(f"  Test Brier:       {metrics['test_brier']:.4f}")
            logger.info(f"  Threshold:        {metrics['threshold']:.3f}")
            logger.info(f"  Is Calibrated:    {metrics['is_calibrated']}")
            logger.info(f"  Calibration:      {metrics['calibration_method']}")
            logger.info(f"  Model path:       {model.model_path}")
            logger.info(f"{'─'*80}\n")
            
            return {
                "symbol": symbol,
                "status": "SUCCESS",
                "model": model,
                "metrics": metrics,
                "num_setups": num_labeled,
                "win_rate": win_rate,
            }
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return None
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Train ML models (FIXED system)")
    parser.add_argument("--symbol", type=str, help="Symbol to train (e.g., EURUSD)")
    parser.add_argument("--data", type=str, help="CSV data path")
    parser.add_argument("--all", action="store_true", help="Train all symbols from config")
    parser.add_argument("--force", action="store_true", help="Force retrain regardless of model age")
    
    args = parser.parse_args()
    
    # Determine which symbols to train
    symbols_to_train = {}
    
    if args.all:
        # Train all configured symbols
        symbols_to_train = dict(cfg.SYMBOL_CSV_MAP)
    elif args.symbol and args.data:
        # Single symbol with explicit data
        symbols_to_train = {args.symbol.upper(): [args.data]}
    elif args.symbol:
        # Single symbol from config
        symbol = args.symbol.upper()
        if symbol in cfg.SYMBOL_CSV_MAP:
            symbols_to_train = {symbol: cfg.SYMBOL_CSV_MAP[symbol]}
        else:
            logger.error(f"Symbol {symbol} not in config.py SYMBOL_CSV_MAP")
            return
    else:
        # Default: train configured symbols
        symbols_to_train = dict(cfg.SYMBOL_CSV_MAP)
    
    # Filter out symbols without data
    symbols_to_train = {s: paths for s, paths in symbols_to_train.items() if paths}
    
    if not symbols_to_train:
        logger.error("No symbols to train (no CSV data configured)")
        logger.info("Usage:")
        logger.info("  python train_fixed.py --symbol EURUSD --data data/raw/EURUSD.csv")
        logger.info("  python train_fixed.py --all")
        return
    
    logger.info(f"Training {len(symbols_to_train)} symbols: {', '.join(symbols_to_train.keys())}\n")
    
    # Train each symbol
    results = {}
    for symbol, csv_paths in symbols_to_train.items():
        result = train_symbol_fixed(
            symbol=symbol,
            csv_paths=csv_paths,
            base_tf=cfg.BASE_TF,
            htf=cfg.HTF_FOR_TREND,
            model_name=cfg.MODEL_NAME,
        )
        results[symbol] = result
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING SUMMARY")
    logger.info(f"{'='*80}")
    
    for symbol, result in results.items():
        if result and result["status"] == "SUCCESS":
            metrics = result["metrics"]
            logger.info(
                f"✓ {symbol:10s} | "
                f"AUC={metrics['test_auc']:.4f} | "
                f"Threshold={metrics['threshold']:.3f} | "
                f"Calibrated={metrics['is_calibrated']}"
            )
        else:
            logger.info(f"✗ {symbol:10s} | FAILED")
    
    logger.info(f"{'='*80}\n")
    
    # Final validation message
    logger.info("Next steps:")
    logger.info("  1. Run validation: python validate_system_fixed.py")
    logger.info("  2. Run backtest:   python main.py --mode backtest --symbol XAUUSD")
    logger.info("  3. Run walk-forward: See INTEGRATION_GUIDE_FIXED.md section 5")


if __name__ == "__main__":
    main()
