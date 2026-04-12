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

    logger.info(f"Generating {n_bars:,} synthetic bars → {path}")
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
        rr_ratio          = getattr(cfg, 'RR_MIN', getattr(cfg, 'RR_RATIO', 2.0)),
        sl_atr_mult       = cfg.SL_ATR_MULT,
        max_bars_to_bos   = getattr(cfg, 'MAX_BARS_TO_BOS', 20),
        max_bars_to_entry = getattr(cfg, 'MAX_BARS_TO_ENTRY', 25),
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
        sl_atr_mult       = cfg.SL_ATR_MULT,
        sl_buffer_atr     = cfg.SL_BUFFER_ATR,
        rr_ratio          = getattr(cfg, 'RR_MIN', getattr(cfg, 'RR_RATIO', 2.0)),
        require_htf_align = cfg.REQUIRE_HTF_ALIGN,
        min_ev            = cfg.MIN_EV,
        pullback_atr_min  = cfg.PULLBACK_ATR_MIN,
        pullback_atr_max  = cfg.PULLBACK_ATR_MAX,
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
        symbol               = symbol,
        max_trades_per_day   = cfg.MAX_TRADES_PER_DAY,
        daily_loss_limit_pct = cfg.DAILY_LOSS_LIMIT_PCT,
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