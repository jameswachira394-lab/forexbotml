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

    logger.info(f"Generating {n_bars:,} synthetic bars -> {path}")
    df = generate_ohlcv(n_bars=n_bars, timeframe_min=cfg.BASE_TF_MINUTES, seed=42)
    df.to_csv(path, index=False)
    logger.info(f"Done. {len(df):,} rows | "
                f"{df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")


# ─────────────────────────────────────────────────────────────────────────────
# Mode: train  (real data, multi-symbol)
# ─────────────────────────────────────────────────────────────────────────────

def mode_train(args) -> None:
    from features.labeler   import LabelConfig
    from models.trainer     import train_all_symbols, train_symbol

    logger.info("=== TRAINING MODE ===")

    label_cfg = LabelConfig(
        displacement_atr_mult = cfg.DISPLACEMENT_ATR_MULT,
        displacement_bars     = cfg.DISPLACEMENT_BARS,
        choch_bars            = cfg.CHOCH_BARS,
        entry_bars            = cfg.ENTRY_BARS,
        rr_min                = cfg.RR_MIN,
        sl_buffer_atr         = cfg.SL_BUFFER_ATR,
        max_sl_atr            = cfg.MAX_SL_ATR,
        min_sweep_strength    = cfg.MIN_SWEEP_STRENGTH,
    )

    # Build the CSV map: CLI overrides config
    sym_csv_map = dict(cfg.SYMBOL_CSV_MAP)

    if args.symbol and args.data:
        # Single-symbol override from CLI
        sym_csv_map = {args.symbol.upper(): [args.data]}
        logger.info(f"Single-symbol mode: {args.symbol} ← {args.data}")

    # Filter to symbols that have at least one CSV file
    runnable = {s: paths for s, paths in sym_csv_map.items() if paths}
    skipped  = [s for s, paths in sym_csv_map.items() if not paths]

    if skipped:
        logger.warning(
            f"No CSV files configured for: {skipped}. "
            "Add paths to SYMBOL_CSV_MAP in config.py."
        )

    if not runnable:
        logger.error(
            "No symbols with CSV data found.\n"
            "  → Add your HistData/Dukascopy CSV paths to SYMBOL_CSV_MAP in config.py\n"
            "  → Or run: python main.py --mode train --symbol EURUSD --data path/to/file.csv"
        )
        return

    train_all_symbols(
        symbol_csv_map = runnable,
        base_tf        = cfg.BASE_TF,
        htf            = cfg.HTF_FOR_TREND,
        model_name     = cfg.MODEL_NAME,
        source         = args.source,
        force_retrain  = args.force or cfg.FORCE_RETRAIN,
        max_age_days   = cfg.MAX_MODEL_AGE_DAYS,
        label_config   = label_cfg,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mode: backtest
# ─────────────────────────────────────────────────────────────────────────────

def mode_backtest(args) -> None:
    from data.histdata_parser   import parse_histdata
    from data.loader            import load_multi_timeframe, OHLCVLoader
    from features.engineer      import engineer_features
    from models.ml_model        import ForexMLModel
    from strategy.engine        import StrategyEngine, StrategyConfig
    from backtest.engine        import BacktestEngine, BacktestConfig

    symbol    = (args.symbol or cfg.SYMBOL).upper()
    data_path = args.data or cfg.DATA_PATH

    logger.info(f"=== BACKTEST | {symbol} | {data_path} ===")

    # ── Load data ─────────────────────────────────────────────────
    path = Path(data_path)
    if not path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    # Auto-detect format: HistData/Dukascopy vs native loader
    try:
        base_df = parse_histdata(data_path, symbol=symbol, target_tf=cfg.BASE_TF)
        loader  = OHLCVLoader.__new__(OHLCVLoader)
        loader.timeframe = "M1"
        htf_df  = loader.resample(base_df, cfg.HTF_FOR_TREND)
    except Exception:
        # Fallback to native loader (already-resampled CSV)
        tfs     = load_multi_timeframe(data_path, cfg.BASE_TF, cfg.HIGHER_TFS)
        base_df = tfs[cfg.BASE_TF]
        htf_df  = tfs.get(cfg.HTF_FOR_TREND)

    logger.info(f"Bars: {len(base_df):,} [{cfg.BASE_TF}]")

    # ── Features ──────────────────────────────────────────────────
    feat_df = engineer_features(base_df, htf_df=htf_df)

    # ── Load model ────────────────────────────────────────────────
    model = ForexMLModel(model_name=f"{symbol}_{cfg.MODEL_NAME}")
    try:
        model.load()
        logger.info(f"Loaded model: {symbol}_{cfg.MODEL_NAME}")
    except FileNotFoundError:
        # Try generic model
        try:
            model = ForexMLModel(model_name=cfg.MODEL_NAME)
            model.load()
            logger.warning("Using generic model (no symbol-specific model found).")
        except FileNotFoundError:
            logger.warning("No model found – running without ML filter.")
            model = None

    # ── Strategy ──────────────────────────────────────────────────
    s_cfg = StrategyConfig(
        ml_threshold      = cfg.ML_THRESHOLD,
        sl_atr_mult       = getattr(cfg, 'SL_BUFFER_ATR', 0.2),
        rr_ratio          = cfg.RR_MIN,
        require_htf_align = cfg.REQUIRE_HTF_ALIGN,
    )
    engine  = StrategyEngine(s_cfg, model=model)
    signals = engine.scan_all(feat_df)

    if not signals:
        logger.warning("No signals generated.")
        return

    # ── Backtest ──────────────────────────────────────────────────
    b_cfg = BacktestConfig(
        initial_balance      = cfg.INITIAL_BALANCE,
        risk_per_trade       = cfg.RISK_PER_TRADE_PCT,
        spread_pips          = cfg.SPREAD_PIPS,
        slippage_pips        = cfg.SLIPPAGE_PIPS,
        max_trades_per_day   = cfg.MAX_TRADES_PER_DAY,
        daily_loss_limit_pct = cfg.DAILY_LOSS_LIMIT_PCT,
        timeframe_mins       = cfg.BASE_TF_MINUTES,
    )
    bt  = BacktestEngine(b_cfg)
    res = bt.run(base_df, signals)

    res.print_summary()
    Path("logs").mkdir(exist_ok=True)
    res.save_trades(f"logs/{symbol}_backtest_trades.csv")
    res.save_equity_curve(f"logs/{symbol}_equity_curve.csv")


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
        displacement_atr_mult = cfg.DISPLACEMENT_ATR_MULT,
        displacement_bars     = cfg.DISPLACEMENT_BARS,
        choch_bars            = cfg.CHOCH_BARS,
        entry_bars            = cfg.ENTRY_BARS,
        rr_min                = cfg.RR_MIN,
        sl_buffer_atr         = cfg.SL_BUFFER_ATR,
        max_sl_atr            = cfg.MAX_SL_ATR,
        min_sweep_strength    = cfg.MIN_SWEEP_STRENGTH,
    )
    s_cfg = StrategyConfig(
        ml_threshold      = cfg.ML_THRESHOLD,
        sl_atr_mult       = getattr(cfg, 'SL_BUFFER_ATR', 0.2),
        rr_ratio          = cfg.RR_MIN,
        require_htf_align = cfg.REQUIRE_HTF_ALIGN,
    )
    
    validator = WalkForwardValidator(n_splits=args.folds)
    results = validator.run(
        feature_df      = feat_df,
        raw_df          = base_df,
        label_config    = label_cfg,
        strategy_config = s_cfg,
        backtest_config = BacktestConfig(
            initial_balance      = cfg.INITIAL_BALANCE,
            risk_per_trade       = cfg.RISK_PER_TRADE_PCT,
            max_trades_per_day   = cfg.MAX_TRADES_PER_DAY,
            ml_threshold         = cfg.ML_THRESHOLD,
        ),
        model_class     = ForexMLModel,
    )

    print(f"\n{'='*50}")
    print(f"  WALK-FORWARD: {symbol}")
    print(f"\n{'='*50}")
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
            print(f"{'-'*50}")
            print(f"  Avg WR: {np.mean(wrs):.1%}  |  Avg PF: {np.mean(pfs):.2f}")
    print(f"{'='*50}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Mode: verify
# ─────────────────────────────────────────────────────────────────────────────

def mode_verify(args) -> None:
    """
    Task 7: Run backtest and live sim on the same data and compare.
    """
    import pandas as pd
    from data.loader import load_multi_timeframe
    from features.engineer import engineer_features
    from strategy.engine import StrategyEngine, StrategyConfig
    from backtest.engine import BacktestEngine, BacktestConfig
    from models.ml_model import ForexMLModel
    from execution.multi_symbol_trader import MultiSymbolTrader
    from execution.mt5_broker import MT5Broker
    from risk.manager import RiskConfig

    symbol = (args.symbol or cfg.SYMBOL).upper()
    data_path = args.data or cfg.DATA_PATH
    logger.info(f"=== VERIFY MODE | {symbol} ===")

    tfs = load_multi_timeframe(data_path, cfg.BASE_TF, cfg.HIGHER_TFS)
    base_df = tfs[cfg.BASE_TF]
    htf_df = tfs.get(cfg.HTF_FOR_TREND)
    feat_df = engineer_features(base_df, htf_df=htf_df)

    model = ForexMLModel(model_name=f"{symbol}_{cfg.MODEL_NAME}")
    try:
        model.load()
    except Exception:
        logger.warning("Using generic model for verify.")
        model = ForexMLModel(model_name=cfg.MODEL_NAME)
        model.load()

    s_cfg = StrategyConfig(
        ml_threshold=cfg.ML_THRESHOLD,
        sl_atr_mult=cfg.SL_BUFFER_ATR,
        rr_ratio=cfg.RR_MIN,
        require_htf_align=cfg.REQUIRE_HTF_ALIGN,
    )

    # 1. Run Backtest
    b_cfg = BacktestConfig(
        initial_balance=cfg.INITIAL_BALANCE,
        risk_per_trade=cfg.RISK_PER_TRADE_PCT,
        max_trades_per_day=cfg.MAX_TRADES_PER_DAY,
        pip_value=cfg.PIP_VALUE,
        ml_threshold=cfg.ML_THRESHOLD,
    )
    bt_engine = BacktestEngine(b_cfg)
    strat = StrategyEngine(s_cfg, model=model)
    signals = strat.scan_all(feat_df)
    bt_res = bt_engine.run(base_df, signals, symbol=symbol)
    bt_trades = bt_res.trades

    # 2. Run Live simulation
    # Clear previous logs/live_trades.csv to ensure clean comparison
    live_log_path = Path("logs/live_trades.csv")
    if live_log_path.exists():
        live_log_path.unlink()

    broker = MT5Broker() # Simulation mode
    r_cfg = RiskConfig(
        account_balance=cfg.INITIAL_BALANCE,
        max_trades_per_day=cfg.MAX_TRADES_PER_DAY,
        risk_per_trade_pct=cfg.RISK_PER_TRADE_PCT,
        pip_value=cfg.PIP_VALUE,
    )
    trader = MultiSymbolTrader(
        symbols=[symbol], model=model, broker=broker,
        strategy_config=s_cfg, risk_config=r_cfg, poll_secs=0
    )
    
    logger.info("Running parallel live simulation...")
    warm_up = getattr(cfg, "LIVE_WARM_BARS", 300)
    for i in range(warm_up, len(base_df)):
        window = base_df.iloc[max(0, i-warm_up):i+1]
        trader._on_new_bar(symbol, window)

    # Read results
    if live_log_path.exists():
        live_df = pd.read_csv(live_log_path)
        live_df = live_df[live_df['symbol'] == symbol]
        # Filter for closed trades to match backtest list usually
        live_trades_count = len(live_df[live_df['exit_reason'] != 'OPEN'])
    else:
        live_trades_count = 0

    # 3. Compare
    logger.info("\n" + "="*40)
    logger.info("  VERIFICATION RESULTS")
    logger.info("="*40)
    logger.info(f"  Backtest Trades: {len(bt_trades)}")
    logger.info(f"  Live Sim Trades: {live_trades_count}")
    
    if len(bt_trades) == live_trades_count:
        logger.info("  [SUCCESS] Trade counts match perfectly.")
    else:
        logger.warning(f"  [DIVERGENCE] Difference: {abs(len(bt_trades) - live_trades_count)} trades.")
        logger.warning("  Check logs/live_trades.csv and system logs for comparison.")
    logger.info("="*40 + "\n")


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
    
    # Re-init states for the new symbol list
    from execution.multi_symbol_trader import SymbolState
    trader._states = {s: SymbolState(s) for s in symbols}

    trader.run()


# ─────────────────────────────────────────────────────────────────────────────
# Mode: report
# ─────────────────────────────────────────────────────────────────────────────

def mode_report(args) -> None:
    from reporting.dashboard_generator import DashboardGenerator
    symbol = (args.symbol or cfg.SYMBOL).upper()
    
    logger.info(f"=== REPORT MODE | {symbol} ===")
    
    try:
        gen = DashboardGenerator(symbol=symbol)
        output_path = gen.generate()
        print(f"\n{'='*55}")
        print(f"  DASHBOARD GENERATED SUCCESSFULLY")
        print(f"  Symbol: {symbol}")
        print(f"  Path:   {output_path}")
        print(f"  (Open this file in your browser to view results)")
        print(f"{'='*55}\n")
    except Exception as e:
        logger.error(f"Failed to generate dashboard: {e}")
        if "template" in str(e).lower():
            logger.error("Ensure reporting/template.html exists.")


# ─────────────────────────────────────────────────────────────────────────────
# Mode: sync
# ─────────────────────────────────────────────────────────────────────────────

def mode_sync(args) -> None:
    from data.loader import MT5SyncLoader
    import config as cfg

    symbol    = (args.symbol or cfg.SYMBOL).upper()
    timeframe = args.timeframe or cfg.BASE_TF
    n_bars    = args.bars or 100_000
    save_path = args.data or f"data/raw/{symbol}_{timeframe}_mt5.csv"

    logger.info(f"=== SYNC MODE | {symbol} [{timeframe}] ===")
    
    MT5SyncLoader.sync_symbol(
        symbol    = symbol,
        timeframe = timeframe,
        n_bars    = n_bars,
        save_path = save_path,
        login     = cfg.MT5_LOGIN,
        password  = cfg.MT5_PASSWORD,
        server    = cfg.MT5_SERVER,
    )
    logger.info("Sync complete.")


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
        choices=["generate", "train", "backtest", "walkforward", "live", "sync", "report", "verify"],
        help=(
            "generate    → synthetic data (testing)\n"
            "train       → train from real HistData/Dukascopy CSVs\n"
            "backtest    → single-symbol historical simulation\n"
            "walkforward → expanding walk-forward validation\n"
            "live        → multi-symbol MT5 live trading\n"
            "sync        → download data from MT5 terminal\n"
            "report      → generate HTML dashboard\n"
            "verify      → Task 7: parity check backtest vs live"
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
    p.add_argument("--source", default="csv", choices=["csv", "mt5"],
                   help="[train] Data source: 'csv' (local files) or 'mt5' (live terminal)")
    p.add_argument("--timeframe", default=None,
                   help="[sync] Timeframe override, e.g. M1, M5, H1")
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
        "report":      mode_report,
        "verify":      mode_verify,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
