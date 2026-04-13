"""
main.py  –  Unified CLI entry point for the Forex Trading System.

Modes
-----
  generate      Generate synthetic sample data (testing only)
  train         Train ML models from real HistData/Dukascopy CSVs
  backtest      Run historical simulation on a single symbol
  walkforward   Walk-forward out-of-sample validation
  live          Start multi-symbol live trading via MT5

Quick start
-----------
  # 1. Put your CSVs in data/raw/ and update SYMBOL_CSV_MAP in config.py
  # 2. python main.py --mode train
  # 3. python main.py --mode backtest --symbol EURUSD --data data/raw/EURUSD_2023.csv
  # 4. Set MT5 credentials in config.py
  # 5. python main.py --mode live
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from execution.logger import setup_logging
import config as cfg

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Mode: generate
# ─────────────────────────────────────────────────────────────────────────────

def mode_generate(args) -> None:
    from data.generate_sample import generate_ohlcv
    import os

    os.makedirs("data/raw", exist_ok=True)
    n_bars = args.bars or 157_000
    path   = args.data or cfg.DATA_PATH

    logger.info(f"Generating {n_bars:,} synthetic bars → {path}")
    df = generate_ohlcv(n_bars=n_bars, timeframe_min=cfg.BASE_TF_MINUTES, seed=42)
    df.to_csv(path, index=False)
    logger.info(f"Done. {len(df):,} rows | "
                f"{df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")


# ─────────────────────────────────────────────────────────────────────────────
# Mode: train  (real data, multi-symbol)
# ─────────────────────────────────────────────────────────────────────────────

def mode_train(args) -> None:
    """Train ML models using FIXED system (NO LEAKAGE, CALIBRATED)."""
    from data.histdata_parser import parse_histdata
    from data.loader import OHLCVLoader
    from features.engineer_fixed import engineer_features
    from features.labeler import SetupLabeler, LabelConfig, get_feature_columns
    from models.ml_integration_fixed import ForexMLModelFixed, get_ml_feature_columns

    logger.info("=== TRAINING MODE (FIXED SYSTEM - NO LEAKAGE) ===\n")

    # Build the CSV map: CLI overrides config
    sym_csv_map = dict(cfg.SYMBOL_CSV_MAP)

    if args.symbol and args.data:
        # Single-symbol override from CLI
        sym_csv_map = {args.symbol.upper(): [args.data]}
        logger.info(f"Single-symbol: {args.symbol} ← {args.data}\n")

    # Filter to symbols that have at least one CSV file
    runnable = {s: paths for s, paths in sym_csv_map.items() if paths}
    skipped  = [s for s, paths in sym_csv_map.items() if not paths]

    if skipped:
        logger.warning(f"Skipped (no CSV): {skipped}")

    if not runnable:
        logger.error(
            "No symbols with CSV data found.\n"
            "  → Add HistData/Dukascopy CSVs to SYMBOL_CSV_MAP in config.py\n"
            "  → Or: python main.py --mode train --symbol EURUSD --data path/to/file.csv"
        )
        return

    label_cfg = LabelConfig(
        rr_ratio=cfg.TP_ATR_MULT / cfg.SL_ATR_MULT if hasattr(cfg, 'TP_ATR_MULT') else 2.0,
        sl_atr_mult=cfg.SL_ATR_MULT,
        max_bars_to_bos=20,
        max_bars_to_entry=25,
    )

    results = {}
    for symbol, csv_paths in runnable.items():
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"Training {symbol} (FIXED SYSTEM)")
            logger.info(f"{'='*80}\n")
            
            # Load CSVs
            frames = []
            for csv_path in csv_paths:
                if not Path(csv_path).exists():
                    logger.warning(f"  ⚠ Not found: {csv_path}")
                    continue
                try:
                    logger.info(f"  Loading {Path(csv_path).name}...")
                    df = parse_histdata(csv_path, symbol=symbol, target_tf="M1")
                    logger.info(f"    → {len(df):,} M1 bars")
                    frames.append(df)
                except Exception as e:
                    logger.error(f"    Parse failed: {e}")
            
            if not frames:
                logger.error(f"No valid CSVs for {symbol}")
                results[symbol] = {"status": "FAILED", "reason": "No data"}
                continue
            
            # Merge and resample
            df_m1 = pd.concat(frames).sort_index()
            df_m1 = df_m1[~df_m1.index.duplicated(keep='last')]
            
            loader = OHLCVLoader.__new__(OHLCVLoader)
            loader.timeframe = "M1"
            df_base = loader.resample(df_m1, cfg.BASE_TF)
            df_htf = loader.resample(df_m1, cfg.HTF_FOR_TREND)
            
            logger.info(f"  Resampled: {len(df_base):,} {cfg.BASE_TF} bars")
            
            if len(df_base) < 500:
                logger.error(f"Too few bars ({len(df_base)} < 500)")
                results[symbol] = {"status": "FAILED", "reason": "Too few bars"}
                continue
            
            # Features (FIXED - no leakage)
            logger.info(f"  Engineering features (NO LEAKAGE)...")
            df_feat = engineer_features(df_base, htf_df=df_htf)
            logger.info(f"    → {len(df_feat):,} rows engineered")
            
            # Label
            logger.info(f"  Labeling setups...")
            labeler = SetupLabeler(label_cfg)
            df_labeled = labeler.label(df_feat)
            num_labeled = len(df_labeled)
            win_rate = df_labeled["label"].mean() if len(df_labeled) > 0 else 0
            logger.info(f"    → {num_labeled:,} labeled (win rate: {win_rate:.1%})")
            
            if num_labeled < 30:
                logger.error(f"Too few setups ({num_labeled} < 30)")
                results[symbol] = {"status": "FAILED", "reason": "Too few labeled setups"}
                continue
            
            # ML features
            feat_cols = get_ml_feature_columns(df_labeled)
            X = df_labeled[feat_cols].fillna(0)
            y = df_labeled["label"]
            
            logger.info(f"  Training ML (CALIBRATED)...")
            model = ForexMLModelFixed(f"{symbol}_{cfg.MODEL_NAME}")
            
            metrics = model.train(
                X=X, y=y,
                test_size=0.15, val_size=0.15, n_cv_splits=5,
                rr_ratio=cfg.TP_ATR_MULT / cfg.SL_ATR_MULT if hasattr(cfg, 'TP_ATR_MULT') else 2.0,
            )
            
            model.save()
            
            logger.info(f"\n  Results:")
            logger.info(f"    • AUC (train):      {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
            logger.info(f"    • AUC (test):       {metrics['test_auc']:.4f}")
            logger.info(f"    • Brier score:      {metrics['test_brier']:.4f}")
            logger.info(f"    • Threshold:        {metrics['threshold']:.3f}")
            logger.info(f"    • Is calibrated:    {metrics['is_calibrated']}")
            logger.info(f"    • Model path:       {model.model_path}")
            
            results[symbol] = {"status": "SUCCESS", "metrics": metrics}
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            results[symbol] = {"status": "FAILED", "reason": str(e)}
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING SUMMARY")
    logger.info(f"{'='*80}")
    for symbol, result in results.items():
        if result["status"] == "SUCCESS":
            m = result["metrics"]
            logger.info(f"✓ {symbol:10s} | AUC={m['test_auc']:.4f} | Cal={m['is_calibrated']}")
        else:
            logger.info(f"✗ {symbol:10s} | {result.get('reason', 'FAILED')}")
    logger.info(f"{'='*80}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Mode: backtest
# ─────────────────────────────────────────────────────────────────────────────

def mode_backtest(args) -> None:
    """Backtest using FIXED system (NO LEAKAGE, SEQUENTIAL)."""
    from data.histdata_parser import parse_histdata
    from data.loader import load_multi_timeframe, OHLCVLoader
    from features.engineer_fixed import engineer_features
    from models.ml_integration_fixed import ForexMLModelFixed
    from strategy.engine_fixed import StrategyEngineFixed, StrategyConfigFixed
    from risk.manager_fixed import RiskManager, RiskConfig, get_symbol_spec
    from backtest.engine import BacktestEngine, BacktestConfig

    symbol = (args.symbol or cfg.SYMBOL).upper()
    data_path = args.data or cfg.DATA_PATH

    logger.info(f"\n{'='*80}")
    logger.info(f"BACKTEST (FIXED SYSTEM - BAR-BY-BAR, NO LEAKAGE)")
    logger.info(f"Symbol: {symbol} | Data: {data_path}")
    logger.info(f"{'='*80}\n")

    path = Path(data_path)
    if not path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    # Load data
    try:
        logger.info(f"Loading data...")
        base_df = parse_histdata(data_path, symbol=symbol, target_tf=cfg.BASE_TF)
        loader = OHLCVLoader.__new__(OHLCVLoader)
        loader.timeframe = "M1"
        htf_df = loader.resample(base_df, cfg.HTF_FOR_TREND)
        logger.info(f"  ✓ {len(base_df):,} {cfg.BASE_TF} bars loaded")
    except Exception as e:
        logger.warning(f"HistData parse failed: {e}, trying fallback loader...")
        try:
            tfs = load_multi_timeframe(data_path, cfg.BASE_TF, cfg.HIGHER_TFS)
            base_df = tfs[cfg.BASE_TF]
            htf_df = tfs.get(cfg.HTF_FOR_TREND)
        except Exception as e2:
            logger.error(f"Failed to load data: {e2}")
            return

    # Features (FIXED - no leakage)
    logger.info(f"Engineering features...")
    feat_df = engineer_features(base_df, htf_df=htf_df)
    logger.info(f"  ✓ {len(feat_df):,} feature rows")

    # Load ML model
    logger.info(f"Loading ML model...")
    model = ForexMLModelFixed(f"{symbol}_{cfg.MODEL_NAME}")
    try:
        model.load()
        if not model.is_calibrated:
            logger.warning(f"  ⚠ Model not calibrated (may have lower accuracy)")
        else:
            logger.info(f"  ✓ Model loaded and calibrated")
    except Exception as e:
        logger.warning(f"  ⚠ Model not found: {e}")
        logger.info(f"    Will run with strategy rules only (no ML)")
        model = None

    # Strategy config (FIXED)
    s_cfg = StrategyConfigFixed(
        require_htf_align=cfg.REQUIRE_HTF_ALIGN if hasattr(cfg, 'REQUIRE_HTF_ALIGN') else True,
        htf_strength_min=0.3,
        displacement_atr_min=1.5,
        pullback_atr_min=cfg.PULLBACK_ATR_MIN if hasattr(cfg, 'PULLBACK_ATR_MIN') else 0.5,
        pullback_atr_max=cfg.PULLBACK_ATR_MAX if hasattr(cfg, 'PULLBACK_ATR_MAX') else 2.5,
        ml_threshold=cfg.ML_THRESHOLD if hasattr(cfg, 'ML_THRESHOLD') else 0.50,
        min_ev=cfg.MIN_EV if hasattr(cfg, 'MIN_EV') else 0.15,
        sl_buffer_atr=cfg.SL_BUFFER_ATR if hasattr(cfg, 'SL_BUFFER_ATR') else 0.8,
        rr_ratio_base=cfg.RR_MIN if hasattr(cfg, 'RR_MIN') else 3.0,
        trade_cooldown_bars=10,
    )

    engine = StrategyEngineFixed(s_cfg, model=model)

    # Process bar-by-bar (SEQUENTIAL - no lookahead)
    logger.info(f"Processing bars (bar-by-bar execution)...")
    signals = []
    for i in range(100, len(feat_df)):  # Warm-up: first 100 bars
        ts = feat_df.index[i]
        row = feat_df.iloc[i]
        
        ohlc = {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
        
        features = row.to_dict()
        atr = float(row.get("atr", 0.05))
        
        signal = engine.process_bar(
            bar_idx=i,
            timestamp=ts,
            ohlc=ohlc,
            atr=atr,
            features=features,
        )
        
        if signal:
            signals.append(signal)

    logger.info(f"  ✓ {len(signals)} signals generated")

    if not signals:
        logger.warning("No signals generated")
        return

    # Risk manager
    logger.info(f"Risk-approval gating...")
    approved_signals = []
    risk_cfg = RiskConfig(
        account_balance=cfg.INITIAL_BALANCE if hasattr(cfg, 'INITIAL_BALANCE') else 100.0,
        risk_per_trade_pct=cfg.RISK_PER_TRADE_PCT if hasattr(cfg, 'RISK_PER_TRADE_PCT') else 2.0,
        min_profit_target=10.0,
        max_trades_per_day=cfg.MAX_TRADES_PER_DAY if hasattr(cfg, 'MAX_TRADES_PER_DAY') else 2,
        daily_loss_limit_pct=cfg.DAILY_LOSS_LIMIT_PCT if hasattr(cfg, 'DAILY_LOSS_LIMIT_PCT') else 2.0,
        min_ev_after_cost=0.15,
    )
    risk_mgr = RiskManager(risk_cfg)

    for sig in signals:
        approved, reason = risk_mgr.approve_trade(
            entry_price=sig.entry_price,
            sl_price=sig.sl_price,
            tp_price=sig.tp_price,
            direction=sig.direction,
            symbol=symbol,
            ml_prob=sig.ml_probability,
            rr_ratio=sig.rr_ratio,
            spread_pips=cfg.SPREAD_PIPS if hasattr(cfg, 'SPREAD_PIPS') else 1.5,
            slippage_pips=cfg.SLIPPAGE_PIPS if hasattr(cfg, 'SLIPPAGE_PIPS') else 0.5,
            atr=base_df.iloc[sig.timestamp.searchsorted(base_df.index)].get("atr", 0.05) if hasattr(base_df.index, 'searchsorted') else 0.05,
            atr_percentile=0.5,
        )
        if approved:
            approved_signals.append(sig)

    logger.info(f"  ✓ {len(approved_signals)} passed risk gate")

    # Backtest
    logger.info(f"Running backtest...")
    b_cfg = BacktestConfig(
        initial_balance=cfg.INITIAL_BALANCE if hasattr(cfg, 'INITIAL_BALANCE') else 10000.0,
        risk_per_trade=cfg.RISK_PER_TRADE_PCT if hasattr(cfg, 'RISK_PER_TRADE_PCT') else 1.0,
        spread_pips=cfg.SPREAD_PIPS if hasattr(cfg, 'SPREAD_PIPS') else 1.5,
        slippage_pips=cfg.SLIPPAGE_PIPS if hasattr(cfg, 'SLIPPAGE_PIPS') else 0.5,
        symbol=symbol,
    )
    bt = BacktestEngine(b_cfg)
    results = bt.run(base_df, approved_signals)

    # Export results
    Path("logs").mkdir(exist_ok=True)
    trade_log = risk_mgr.get_trade_log()
    if not trade_log.empty:
        trade_log.to_csv(f"logs/{symbol}_backtest_trades.csv", index=False)
        logger.info(f"  ✓ Trades exported: logs/{symbol}_backtest_trades.csv")

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"BACKTEST RESULTS: {symbol}")
    logger.info(f"{'='*80}")
    logger.info(f"  Initial balance:     ${b_cfg.initial_balance:,.2f}")
    logger.info(f"  Final equity:        ${bt.equity:,.2f}")
    logger.info(f"  P&L:                 ${bt.equity - b_cfg.initial_balance:+,.2f}")
    logger.info(f"  Total trades:        {len(bt.trades)}")
    if len(bt.trades) > 0:
        wins = sum(1 for t in bt.trades if t.pnl_usd > 0)
        logger.info(f"  Wins:                {wins}/{len(bt.trades)} ({wins/len(bt.trades):.1%})")
        avg_profit = sum(t.pnl_usd for t in bt.trades if t.pnl_usd > 0) / max(wins, 1) if wins > 0 else 0
        avg_loss = abs(sum(t.pnl_usd for t in bt.trades if t.pnl_usd < 0)) / max(len(bt.trades) - wins, 1)
        logger.info(f"  Avg profit/loss:     ${avg_profit:,.2f} / ${avg_loss:,.2f}")
    logger.info(f"{'='*80}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Mode: walkforward
# ─────────────────────────────────────────────────────────────────────────────

def mode_walkforward(args) -> None:
    import copy, numpy as np
    from data.histdata_parser   import parse_histdata
    from data.loader            import load_multi_timeframe, OHLCVLoader
    from features.engineer      import engineer_features
    from features.labeler       import LabelConfig
    from models.ml_model        import ForexMLModel
    from strategy.engine        import StrategyConfig
    from backtest.engine        import BacktestConfig, WalkForwardValidator

    symbol    = (args.symbol or cfg.SYMBOL).upper()
    data_path = args.data or cfg.DATA_PATH

    logger.info(f"=== WALK-FORWARD | {symbol} ===")

    path = Path(data_path)
    if not path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    try:
        base_df = parse_histdata(data_path, symbol=symbol, target_tf=cfg.BASE_TF)
        loader  = OHLCVLoader.__new__(OHLCVLoader)
        loader.timeframe = "M1"
        htf_df  = loader.resample(base_df, cfg.HTF_FOR_TREND)
    except Exception:
        tfs     = load_multi_timeframe(data_path, cfg.BASE_TF, cfg.HIGHER_TFS)
        base_df = tfs[cfg.BASE_TF]
        htf_df  = tfs.get(cfg.HTF_FOR_TREND)

    feat_df = engineer_features(base_df, htf_df=htf_df)

    label_cfg = LabelConfig(
        rr_ratio          = getattr(cfg, 'RR_MIN', getattr(cfg, 'RR_RATIO', 2.0)),
        sl_atr_mult       = cfg.SL_ATR_MULT,
        max_bars_to_bos   = getattr(cfg, 'MAX_BARS_TO_BOS', 20),
        max_bars_to_entry = getattr(cfg, 'MAX_BARS_TO_ENTRY', 25),
    )
    s_cfg = StrategyConfig(
        ml_threshold      = cfg.ML_THRESHOLD,
        sl_atr_mult       = cfg.SL_ATR_MULT,
        sl_buffer_atr     = cfg.SL_BUFFER_ATR,
        rr_ratio          = getattr(cfg, 'RR_MIN', getattr(cfg, 'RR_RATIO', 2.0)),
        require_htf_align = cfg.REQUIRE_HTF_ALIGN,
        min_ev            = cfg.MIN_EV,
        pullback_atr_min  = cfg.PULLBACK_ATR_MIN,
        pullback_atr_max  = cfg.PULLBACK_ATR_MAX,
    )
    b_cfg = BacktestConfig(
        initial_balance      = cfg.INITIAL_BALANCE,
        risk_per_trade       = cfg.RISK_PER_TRADE_PCT,
        spread_pips          = cfg.SPREAD_PIPS,
        slippage_pips        = cfg.SLIPPAGE_PIPS,
    )

    wf = WalkForwardValidator(n_splits=args.folds or 4, oos_fraction=getattr(cfg, 'WF_OOS_FRACTION', 0.20))
    results = wf.run(
        feature_df      = feat_df,
        raw_df          = base_df,
        label_config    = label_cfg,
        strategy_config = s_cfg,
        backtest_config = b_cfg,
        model_class     = ForexMLModel,
        model_kwargs    = {"model_name": f"{symbol}_{cfg.MODEL_NAME}_wf"},
    )

    print(f"\n{'═'*50}")
    print(f"  WALK-FORWARD: {symbol}")
    print(f"{'═'*50}")
    for r in results:
        print(
            f"  Fold {r['fold']:<2} | "
            f"Trades={r.get('total_trades',0):<4} "
            f"WR={r.get('win_rate',0):.1%}  "
            f"PF={r.get('profit_factor',0):.2f}  "
            f"Sharpe={r.get('sharpe',0):.2f}  "
            f"DD={r.get('max_drawdown',0):.1%}"
        )
    if results:
        wrs = [r.get("win_rate", 0) for r in results if r.get("total_trades", 0) > 0]
        pfs = [r.get("profit_factor", 0) for r in results if r.get("total_trades", 0) > 0]
        if wrs:
            print(f"{'─'*50}")
            print(f"  Avg WR: {np.mean(wrs):.1%}  |  Avg PF: {np.mean(pfs):.2f}")
    print(f"{'═'*50}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Mode: live  (multi-symbol)
# ─────────────────────────────────────────────────────────────────────────────

def mode_live(args) -> None:
    from models.trainer             import load_all_models
    from execution.multi_symbol_trader import MultiSymbolTrader

    logger.info("=== LIVE TRADING MODE (Multi-Symbol) ===")

    symbols = args.symbols.split(",") if args.symbols else cfg.SYMBOLS
    symbols = [s.strip().upper() for s in symbols]
    logger.info(f"Symbols: {symbols}")

    if not cfg.MT5_LOGIN:
        logger.warning(
            "MT5_LOGIN is 0 – running in SIMULATION mode.\n"
            "Set MT5_LOGIN / MT5_PASSWORD / MT5_SERVER in config.py or env vars."
        )

    # Load per-symbol models; fall back to generic model if symbol-specific missing
    models = load_all_models(symbols, model_name=cfg.MODEL_NAME)
    if not models:
        logger.warning("No trained models found. Running without ML filter.")

    # MultiSymbolTrader uses the first available model for all symbols
    # (extend to per-symbol routing if desired)
    first_model = next(iter(models.values()), None) if models else None

    trader = MultiSymbolTrader.from_config(model=first_model)
    # Override symbols from CLI if provided
    trader.symbols = symbols
    trader._states = {s: trader._states.get(s, __import__(
        'execution.multi_symbol_trader', fromlist=['SymbolState']
    ).SymbolState(s)) for s in symbols}

    trader.run()


# ─────────────────────────────────────────────────────────────────────────────────
# Mode: sync  (download historical bars from MT5 to CSV)
# ─────────────────────────────────────────────────────────────────────────────────

def mode_sync(args) -> None:
    """
    Download historical M5 bars from MT5 and save to CSV.

    Usage:
        python main.py --mode sync --symbol XAUUSD --bars 400000
        python main.py --mode sync --symbol GBPUSD --bars 400000 --data data/raw/GBPUSD_M5.csv

    Downloads in chunks of 100,000 bars to stay within MT5 buffer limits.
    Output CSV columns: timestamp,open,high,low,close,volume
    """
    import os
    import pandas as pd

    symbol     = (args.symbol or cfg.SYMBOL).upper()
    n_bars     = args.bars or 400_000
    batch_size = 100_000          # MT5 copy_rates_from_pos limit per call
    out_path   = args.data or f"data/raw/{symbol}_M5_mt5.csv"

    logger.info(f"=== SYNC | {symbol} | {n_bars:,} bars | -> {out_path} ===")

    # ── Try real MT5 first, fall back gracefully ──────────────────────────────
    try:
        import MetaTrader5 as mt5
    except ImportError:
        logger.error(
            "MetaTrader5 library not installed.\n"
            "  pip install MetaTrader5\n"
            "  MT5 terminal must also be running on this machine."
        )
        return

    # ── Connect ───────────────────────────────────────────────────────────────
    if not mt5.initialize(
        login    = cfg.MT5_LOGIN,
        password = cfg.MT5_PASSWORD,
        server   = cfg.MT5_SERVER,
    ):
        err = mt5.last_error()
        logger.error(f"MT5 connection failed: {err}")
        mt5.shutdown()
        return

    acc = mt5.account_info()
    logger.info(f"Connected | Account: {acc.login} | Server: {acc.server}")

    # ── Ensure symbol is active in MT5 ────────────────────────────────────────
    if not mt5.symbol_select(symbol, True):
        logger.error(f"Symbol {symbol} not found in MT5 market-watch.")
        mt5.shutdown()
        return

    TF = mt5.TIMEFRAME_M5

    # ── Download in batches (newest -> oldest) ────────────────────────────────
    # copy_rates_from_pos(symbol, tf, start_pos, count):
    #   start_pos=0 is the most recent CLOSED bar.
    #   Each successive batch starts further back in time.
    all_frames = []
    downloaded = 0

    while downloaded < n_bars:
        want  = min(batch_size, n_bars - downloaded)
        start = downloaded          # bars back from the most-recent closed bar

        rates = mt5.copy_rates_from_pos(symbol, TF, start, want)
        if rates is None or len(rates) == 0:
            logger.warning(f"MT5 returned no data at offset {start}. Stopping.")
            break

        batch = pd.DataFrame(rates)
        batch.rename(columns={"time": "timestamp", "tick_volume": "volume"}, inplace=True)
        batch["timestamp"] = pd.to_datetime(batch["timestamp"], unit="s")
        batch = batch[["timestamp", "open", "high", "low", "close", "volume"]]

        all_frames.append(batch)
        downloaded += len(batch)
        logger.info(
            f"  Batch {start:>7,} – {start+len(batch)-1:>7,} | "
            f"{batch['timestamp'].iloc[0].date()} -> {batch['timestamp'].iloc[-1].date()} | "
            f"Total so far: {downloaded:,}"
        )

        if len(batch) < want:
            logger.info("Reached beginning of MT5 history.")
            break

    mt5.shutdown()

    if not all_frames:
        logger.error("No data downloaded.")
        return

    # ── Merge, sort chronologically, deduplicate, save ────────────────────────
    df = pd.concat(all_frames, ignore_index=True)
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates(subset="timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)

    span = f"{df['timestamp'].iloc[0].date()} -> {df['timestamp'].iloc[-1].date()}"
    logger.info(
        f"Saved {len(df):,} M5 bars | {span} | {out_path}"
    )
    print(f"\nDownloaded: {len(df):,} bars | {span}")
    print(f"Saved to  : {out_path}")
    print(f"\nNext step  : python main.py --mode train --symbol {symbol} --data {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Forex Trading System — Real Data Edition",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--mode", required=True,
        choices=["generate", "train", "backtest", "walkforward", "live", "sync"],
        help=(
            "generate    → synthetic data (testing)\n"
            "train       → train from real HistData/Dukascopy CSVs\n"
            "backtest    → single-symbol historical simulation\n"
            "walkforward → expanding walk-forward validation\n"
            "live        → multi-symbol MT5 live trading\n"
            "sync        → download historical bars from MT5 to CSV"
        ),
    )
    p.add_argument("--symbol",  default=None,
                   help="Symbol override, e.g. EURUSD")
    p.add_argument("--symbols", default=None,
                   help="[live] Comma-separated list, e.g. EURUSD,GBPUSD,USDJPY")
    p.add_argument("--data",    default=None,
                   help="Path to CSV file (overrides config DATA_PATH / SYMBOL_CSV_MAP)")
    p.add_argument("--bars",    type=int, default=None,
                   help="[generate] Number of bars to produce")
    p.add_argument("--folds",   type=int, default=4,
                   help="[walkforward] Number of OOS folds")
    p.add_argument("--force",   action="store_true",
                   help="[train] Force retrain even if model is recent")
    p.add_argument("--log-level", dest="log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(level=args.log_level)

    dispatch = {
        "generate":    mode_generate,
        "train":       mode_train,
        "backtest":    mode_backtest,
        "walkforward": mode_walkforward,
        "live":        mode_live,
        "sync":        mode_sync,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()