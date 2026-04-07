"""
models/trainer.py
-----------------
Multi-symbol training pipeline.

For each symbol:
  1. Parse HistData / Dukascopy CSV  (or re-fetch from MT5)
  2. Resample to BASE_TF
  3. Feature-engineer
  4. Label setups
  5. Train XGBoost
  6. Save model as  saved_models/{SYMBOL}_{MODEL_NAME}.joblib

Also supports:
  - Merging multiple CSV files for the same symbol (e.g. yearly exports)
  - Training a single shared model across all symbols (transfer-style)
  - Retraining only symbols whose model files are older than N days
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def train_symbol(
    symbol:      str,
    csv_paths:   List[str],
    base_tf:     str          = "M5",
    htf:         str          = "H1",
    model_name:  str          = "forex_xgb",
    label_config = None,
    source:      str          = "csv",    # "csv" or "mt5"
    force_retrain: bool       = True,
    max_age_days: int         = 7,
) -> Optional[object]:
    """
    Full train pipeline for a single symbol.

    Parameters
    ----------
    symbol      : e.g. "EURUSD"
    csv_paths   : list of HistData/Dukascopy CSV files for this symbol
    base_tf     : primary timeframe, e.g. "M5"
    htf         : higher timeframe for trend filter, e.g. "H1"
    model_name  : base name for saved model file
    label_config: LabelConfig instance (default if None)
    force_retrain: skip age check and always retrain
    max_age_days : skip retrain if saved model is newer than this

    Returns
    -------
    Trained ForexMLModel instance, or None if skipped / failed.
    """
    from data.histdata_parser   import parse_histdata
    from features.engineer      import engineer_features
    from features.labeler       import SetupLabeler, LabelConfig, get_feature_columns
    from models.ml_model        import ForexMLModel
    from execution.mt5_streamer import fetch_live_bars, MT5_AVAILABLE

    saved_path = Path(f"saved_models/{symbol}_{model_name}.joblib")

    # Skip if model is fresh enough
    if not force_retrain and saved_path.exists():
        age = datetime.now() - datetime.fromtimestamp(saved_path.stat().st_mtime)
        if age < timedelta(days=max_age_days):
            logger.info(
                f"[{symbol}] Skipping retrain – model is {age.days}d old "
                f"(max_age={max_age_days}d). Use force_retrain=True to override."
            )
            model = ForexMLModel(model_name=f"{symbol}_{model_name}")
            model.load()
            return model

    logger.info(f"[{symbol}] ---- Training --------------------------------")

    # ── 1. Data Fetching ──────────────────────────────────────────────────────
    if source == "mt5":
        import config as cfg
        from data.loader import MT5SyncLoader
        
        # Determine sync path; try first CSV in list or default to data/raw/{symbol}_mt5.csv
        sync_path = csv_paths[0] if csv_paths else f"data/raw/{symbol}_{base_tf}_mt5.csv"
        
        logger.info(f"[{symbol}] Syncing data from MT5 to {sync_path}...")
        
        # MT5 Sync Logic (fetches base_tf and htf)
        MT5SyncLoader.sync_symbol(
            symbol    = symbol,
            timeframe = base_tf,
            n_bars    = 100_000,   # Deep fetch for training
            save_path = sync_path,
            login     = cfg.MT5_LOGIN,
            password  = cfg.MT5_PASSWORD,
            server    = cfg.MT5_SERVER,
        )
        
        # Re-fetch for HTF if it's different and not already covered by resampling
        # (Though OHLCVLoader handles resampling better, SyncMT5Loader can be used twice)
        # For simplicity, we just use the synced CSV as our source now
        csv_paths = [sync_path]
        # Fall-through to CSV loader logic below to handle parsing/resampling uniformly
        source = "csv" 

    if source == "csv":
        # ── 1. Load and merge all CSVs ────────────────────────────────────────────
        frames = []
        for p in csv_paths:
            try:
                df = parse_histdata(p, symbol=symbol, target_tf="M1")
                logger.info(f"  [+] {Path(p).name}: {len(df):,} M1 bars")
                frames.append(df)
            except Exception as exc:
                logger.error(f"  [!] Failed to parse {p}: {exc}")

        if not frames:
            logger.error(f"[{symbol}] No valid CSV data loaded. Skipping.")
            return None

        m1_df = pd.concat(frames).sort_index()
        m1_df = m1_df[~m1_df.index.duplicated(keep="last")]
        logger.info(f"[{symbol}] Total M1 bars: {len(m1_df):,} "
                    f"| {m1_df.index[0].date()} -> {m1_df.index[-1].date()}")

        # ── 2. Resample to base TF and HTF ───────────────────────────────────────
        from data.loader import OHLCVLoader, TIMEFRAME_MINUTES
        loader = OHLCVLoader.__new__(OHLCVLoader)
        loader.timeframe = "M1"

        base_df = loader.resample(m1_df, base_tf)
        htf_df  = loader.resample(m1_df, htf)
    
    logger.info(f"[{symbol}] {base_tf}: {len(base_df):,} bars | {htf}: {len(htf_df):,} bars")

    # ── 3. Feature engineering ────────────────────────────────────────────────
    logger.info(f"[{symbol}] Engineering features…")
    try:
        feat_df = engineer_features(base_df, htf_df=htf_df)
    except Exception as exc:
        logger.error(f"[{symbol}] Feature engineering failed: {exc}")
        return None

    if len(feat_df) < 500:
        logger.error(f"[{symbol}] Too few feature rows ({len(feat_df)}). Need ≥500.")
        return None

    # ── 4. Labeling ───────────────────────────────────────────────────────────
    lc = label_config or LabelConfig()
    labeler = SetupLabeler(lc)
    logger.info(f"[{symbol}] Labeling setups…")
    labeled = labeler.label(feat_df)

    if labeled.empty or len(labeled) < 30:
        logger.error(
            f"[{symbol}] Too few labeled setups ({len(labeled)}). "
            "Consider longer data range or looser label parameters."
        )
        return None

    pos_rate = labeled["label"].mean()
    logger.info(f"[{symbol}] Setups: {len(labeled):,} | Win rate: {pos_rate:.1%}")

    feat_cols = get_feature_columns(labeled)
    X = labeled[feat_cols].fillna(0)
    y = labeled["label"]

    # ── 5. Train model ────────────────────────────────────────────────────────
    model = ForexMLModel(model_name=f"{symbol}_{model_name}")
    try:
        metrics = model.train(X, y, test_size=0.20, n_cv_splits=5)
    except Exception as exc:
        logger.error(f"[{symbol}] Training failed: {exc}")
        return None

    logger.info(
        f"[{symbol}] CV AUC={metrics['cv_auc_mean']:.4f}±{metrics['cv_auc_std']:.4f} "
        f"| Test AUC={metrics['test_auc']:.4f} "
        f"| Threshold={metrics['threshold']:.2f}"
    )

    # Feature importance summary
    fi = metrics.get("feature_importance")
    if fi is not None and not fi.empty:
        top5 = ", ".join(f"{k}({v:.3f})" for k, v in fi.head(5).items())
        logger.info(f"[{symbol}] Top features: {top5}")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    model.save()
    logger.info(f"[{symbol}] Model saved -> {saved_path}")

    return model


def train_all_symbols(
    symbol_csv_map: Dict[str, List[str]],
    base_tf:        str  = "M5",
    htf:            str  = "H1",
    model_name:     str  = "forex_xgb",
    source:         str  = "csv",
    force_retrain:  bool = False,
    max_age_days:   int  = 7,
    label_config    = None,
) -> Dict[str, object]:
    """
    Train models for all symbols in *symbol_csv_map*.

    Parameters
    ----------
    symbol_csv_map : {"EURUSD": ["path/EURUSD_2022.csv", "path/EURUSD_2023.csv"], ...}

    Returns
    -------
    dict mapping symbol → trained ForexMLModel (or None if failed)
    """
    results = {}
    total   = len(symbol_csv_map)

    for i, (symbol, csv_list) in enumerate(symbol_csv_map.items(), 1):
        logger.info(f"\n{'='*55}")
        logger.info(f"  Symbol {i}/{total}: {symbol}")
        logger.info(f"{'='*55}")

        model = train_symbol(
            symbol        = symbol,
            csv_paths     = csv_list,
            base_tf       = base_tf,
            htf           = htf,
            model_name    = model_name,
            label_config  = label_config,
            source        = source,
            force_retrain = force_retrain,
            max_age_days  = max_age_days,
        )
        results[symbol] = model

    # Summary
    ok     = [s for s, m in results.items() if m is not None]
    failed = [s for s, m in results.items() if m is None]
    logger.info(f"\nTraining complete: {len(ok)} OK | {len(failed)} failed")
    if failed:
        logger.warning(f"Failed symbols: {failed}")

    return results


def load_all_models(
    symbols:    List[str],
    model_name: str = "forex_xgb",
) -> Dict[str, object]:
    """
    Load pre-trained models for all symbols.
    Returns a dict symbol → ForexMLModel (skips missing models with a warning).
    """
    from models.ml_model import ForexMLModel

    models = {}
    for sym in symbols:
        m = ForexMLModel(model_name=f"{sym}_{model_name}")
        try:
            m.load()
            models[sym] = m
            logger.info(f"Loaded model: {sym}")
        except FileNotFoundError:
            logger.warning(
                f"No saved model for {sym}. "
                "Run training first: python main.py --mode train"
            )
    return models
