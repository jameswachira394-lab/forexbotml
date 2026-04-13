"""
INTEGRATION_GUIDE_FIXED.md

Institutional-Grade Forex Trading System — Refactored

This document explains how to use the refactored components to ensure:
  ✓ Zero data leakage
  ✓ Sequential bar-by-bar execution
  ✓ Proper structure logic (displacement required)
  ✓ Cost-aware EV filtering
  ✓ Calibrated ML probabilities
  ✓ Walk-forward out-of-sample validation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. FEATURE ENGINEERING (NO LEAKAGE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Use: features/engineer_fixed.py

Key fixes:
  - All features use only t-1 data (shift(1))
  - Swing detection: no forward scan
  - BOS/CHoCH: built from past swings
  - New: displacement detection (body ≥ 1.5×ATR post-BOS)
  - HTF strength: not binary (distance_to_BOS / ATR)

Usage:

    from features.engineer_fixed import engineer_features
    
    df = pd.read_csv("EURUSD_M5.csv")
    df.index = pd.to_datetime(df['timestamp'])
    
    htf_df = resample_to_htf(df, "H1")  # Optional HTF for trend
    
    df_feat = engineer_features(df, htf_df=htf_df)
    
    # Confirm NO leakage: features at bar i use only data up to bar i-1
    assert df_feat['bos'].iloc[10] uses df.iloc[0:10]  # ✓

Key features produced:
  - bos: +1/-1 when close breaks past swing
  - displacement: body after BOS in ATR units
  - displacement_confirmed: 1 if ≥ 1.5×ATR
  - htf_trend: -1/0/+1 from HTF data
  - htf_strength: 0-1 scale (not binary)
  - sweep detection: bull_sweep / bear_sweep from t-1
  - atr_percentile: volatility regime [0, 1]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. ML MODEL WITH CALIBRATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Use: models/ml_integration_fixed.py

Key fixes:
  - Platt scaling calibration on validation set (NO test leakage)
  - Redundant feature removal (sweep, BOS, trend removed)
  - Probability validation: buckets check (0.7-0.8 pred → 70-80% wins)
  - Fail-safe: raises if model missing

Features kept (microstructure, not rules):
  - atr_percentile
  - body_pct
  - range_expansion
  - momentum_persistence
  - is_london_open, is_ny_open
  - session timing
  - fvg_size
  - hour / mins_since_open

Features removed (already enforced by strategy):
  - sweep
  - bos
  - htf_trend
  - choch
  - liquidity levels

Training procedure:

    from models.ml_integration_fixed import ForexMLModelFixed, get_ml_feature_columns
    from features.labeler import SetupLabeler, LabelConfig
    
    # 1. Feature engineer and label
    df_feat = engineer_features(df)
    labeler = SetupLabeler(LabelConfig())
    df_labeled = labeler.label(df_feat)
    
    # 2. Get ML features only (removes redundancy)
    feat_cols = get_ml_feature_columns(df_labeled)
    X = df_labeled[feat_cols].fillna(0)
    y = df_labeled["label"]
    
    # 3. Train with calibration
    model = ForexMLModelFixed("EURUSD_xgb")
    metrics = model.train(
        X, y,
        test_size=0.15,   # test set (never touched until end)
        val_size=0.15,    # validation for threshold + calibration
        n_cv_splits=5     # TimeSeriesSplit (no future leakage)
    )
    model.save()
    
    # 4. Verify calibration
    logger.info(f"Is calibrated: {model.is_calibrated}")
    logger.info(f"Threshold: {model.threshold:.3f}")
    
    # 5. Load and use
    model = ForexMLModelFixed("EURUSD_xgb")
    model.load_or_die()  # Raises if missing (fail-safe)
    
    # Predict with calibrated probabilities
    X_new = get_features()
    prob = model.predict_proba_calibrated(X_new)
    # prob[i] ≈ true win rate for that prediction

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. STRATEGY ENGINE (BAR-BY-BAR, NO BATCH SCANNING)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Use: strategy/engine_fixed.py

Key fixes:
  - Sequential bar-by-bar processing (NOT scan_all)
  - State machine (sweep → displacement → BOS → pullback)
  - Displacement gate: body ≥ 1.5×ATR required
  - Pullback range: 0.5-2.5 ATR (not 0.1-5.0)
  - Trade cooldown: 10 bars between trades
  - HTF strength gate (not binary trend)

Setup sequence for BULLISH entry:

    1. SWEEP: bearish sweep detected (wick below swing low, close above wick)
    2. DISPLACEMENT: next bullish candle has body ≥ 1.5×ATR
    3. BOS: close breaks above last swing high
    4. PULLBACK: price retraces 0.5-2.5×ATR below BOS close
    5. ENTRY: at pullback location with ML + EV gates

Usage:

    from strategy.engine_fixed import StrategyEngineFixed, StrategyConfigFixed
    from models.ml_integration_fixed import ForexMLModelFixed
    
    # Load trained ML model
    model = ForexMLModelFixed("EURUSD_xgb")
    model.load_or_die()
    
    # Create engine
    config = StrategyConfigFixed(
        ml_threshold=0.50,
        min_ev=0.15,
        pullback_atr_min=0.5,
        pullback_atr_max=2.5,
        trade_cooldown_bars=10,
    )
    engine = StrategyEngineFixed(config, model=model)
    
    # Process each bar in order
    for i, (ts, row) in enumerate(df.iterrows()):
        if i < 100:
            continue  # Warm-up
        
        ohlc = {
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
        }
        
        features = row.to_dict()  # Must include: bos, displacement, sweep, etc.
        
        signal = engine.process_bar(
            bar_idx=i,
            timestamp=ts,
            ohlc=ohlc,
            atr=row["atr"],
            features=features
        )
        
        if signal:
            # signal = SignalFixed
            #   .direction: +1/-1
            #   .entry_price, .sl_price, .tp_price
            #   .ml_probability, .expected_value, .rr_ratio
            #   .reason: for debugging
            
            # Execute trade (see Risk Manager)
            pass

Key signals produced:
  - direction: +1 for long, -1 for short
  - entry_price: ATR-derived entry level (pullback close)
  - sl_price: below sweep extreme ± safety buffer
  - tp_price: RR-derived target
  - ml_probability: calibrated ML score
  - expected_value: cost-aware EV computation
  - rr_ratio: dynamic per pullback depth

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. RISK MANAGER (COST-AWARE EV, DIAGNOSTICS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Use: risk/manager_fixed.py

Key fixes:
  - Cost-aware EV: EV = P(win)×(RR-cost) - P(loss)×(1+cost)
  - Trade diagnostics: MAE/MFE tracked per trade
  - Per-symbol pip specifications (XAUUSD, GBPUSD, USDJPY all different)
  - Volatility-adjusted sizing (high ATR percentile → reduce size)
  - Minimum profit target gate

Usage:

    from risk.manager_fixed import RiskManager, RiskConfig, get_symbol_spec
    
    # 1. Initialize risk manager
    config = RiskConfig(
        account_balance=100.0,
        risk_per_trade_pct=2.0,
        min_profit_target=10.0,      # min USD per winning trade
        max_trades_per_day=2,
        daily_loss_limit_pct=2.0,
        min_ev_after_cost=0.15,
    )
    risk_mgr = RiskManager(config)
    
    # 2. Approve trade (cost-aware)
    approved, reason = risk_mgr.approve_trade(
        entry_price=1.0950,
        sl_price=1.0930,
        tp_price=1.0980,
        direction=1,
        symbol="EURUSD",
        ml_prob=0.60,
        rr_ratio=2.0,
        spread_pips=1.5,
        slippage_pips=0.5,
        atr=0.0025,
        atr_percentile=0.5,
    )
    
    if not approved:
        logger.info(f"Trade rejected: {reason}")
        return
    
    # 3. Calculate position size (EV-scaled, vol-adjusted)
    pos_size = risk_mgr.calculate_position_size(
        entry_price=1.0950,
        sl_price=1.0930,
        tp_price=1.0980,
        direction=1,
        symbol="EURUSD",
        ml_prob=0.60,
        rr_ratio=2.0,
        atr=0.0025,
        atr_percentile=0.5,
    )
    
    # 4. Log trade opening
    risk_mgr.open_trade(
        entry_ts=pd.Timestamp.now(),
        direction=1,
        entry_price=1.0950,
        sl_price=1.0930,
        tp_price=1.0980,
        symbol="EURUSD",
        position_size_units=pos_size,
        ml_prob=0.60,
        rr_ratio=2.0,
        cost_pips=2.0,
    )
    
    # 5. Log trade closing (when hit TP/SL)
    trade = risk_mgr.close_trade(
        symbol="EURUSD",
        exit_ts=pd.Timestamp.now(),
        exit_price=1.0980,
        exit_reason="TP",
    )
    
    # 6. Export diagnostics
    trade_log = risk_mgr.get_trade_log()  # DataFrame with MAE/MFE/RR_realized
    trade_log.to_csv("trades.csv")

EV calculation:

    ev = P(win) × (RR - spread_pips - slippage_pips) 
       - P(loss) × (1 + spread_pips + slippage_pips)

    Example: P(win)=0.60, RR=3.0, cost=2.0 pips
    ev = 0.60 × (3.0 - 2.0) - 0.40 × (1.0 + 2.0)
       = 0.60 × 1.0 - 0.40 × 3.0
       = 0.60 - 1.20
       = -0.60  ← REJECTED (negative EV)
    
    Requires: EV ≥ 0.15 to execute

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. WALK-FORWARD VALIDATION (NO LEAKAGE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Use: backtest/walk_forward.py

Ensures pure out-of-sample validation:
  - Each fold: trains on past data, tests on future data
  - No future data ever touches training
  - Realistic performance estimates

Usage:

    from backtest.walk_forward import run_walk_forward_backtest
    
    # Load data
    df = load_bars("EURUSD", years=5)
    df_feat = engineer_features(df)
    
    # Run walk-forward validation
    results_df, trade_logs = run_walk_forward_backtest(
        df_feat,
        symbol="EURUSD",
        initial_train_months=24,  # first 2 years train
        test_period_months=3,     # next 3 months test
    )
    
    # results_df columns:
    # - fold_idx
    # - num_trades
    # - total_pnl
    # - win_rate
    # - sharpe
    # - max_dd
    
    # Aggregate across all folds
    total_trades = results_df["num_trades"].sum()
    avg_win_rate = results_df["win_rate"].mean()
    total_pnl = results_df["total_pnl"].sum()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. COMPLETE WORKFLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Step 1: Load & feature engineer with NO LEAKAGE
df = pd.read_csv("EURUSD_5Y.csv", parse_dates=['timestamp'], index_col='timestamp')
df_feat = engineer_features(df)

# Step 2: Walk-forward validation
results_df, trades_df = run_walk_forward_backtest(df_feat, symbol="EURUSD")

# Step 3: Verify results
print("Total trades:", len(trades_df))
print("Win rate:", (trades_df['pnl_usd'] > 0).mean())
print("Avg R:R:", trades_df['realized_rr'].mean())

# Step 4: Check trade quality
print("\\nTrade diagnostics:")
print(trades_df[['entry_ts', 'exit_ts', 'pnl_usd', 'mae_pips', 'mfe_pips']])

# Step 5: Deploy with confidence
model = ForexMLModelFixed("EURUSD_xgb")
model.load_or_die()

engine = StrategyEngineFixed(config, model=model)
risk_mgr = RiskManager(config)

# Live bar-by-bar processing (same as backtest)
for ts, row in live_stream():
    signal = engine.process_bar(...)
    if signal:
        approved, reason = risk_mgr.approve_trade(...)
        if approved:
            # Execute

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. VALIDATION CHECKLIST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before deploying with real capital:

□ Data Leakage:
  - Run validation_check_leakage.py
  - Confirm all features use only t-1 data
  
□ ML Calibration:
  - Check model.is_calibrated == True
  - Validate calibration buckets (±10% tolerance)
  - Check is_calibrated return code
  
□ Walk-Forward Performance:
  - Run 5+ folds
  - Win rate ≥ 50% in most folds
  - PnL positive across folds (or at least most)
  - Sharpe ratio ≥ 1.0
  
□ Strategy Sequential:
  - Backtest ≈ live (within slippage/spread)
  - No scan_all() calls
  - All signals from bar-by-bar loop
  
□ Risk Controls:
  - Cost-aware EV enabled
  - Maximum drawdown ≤ 10%
  - Daily loss limits set
  - Cooldown prevents signal clustering
  
□ Trade Quality:
  - MAE/MFE tracked
  - Displaced setups only (body ≥ 1.5×ATR)
  - Pullback depth in range (0.5-2.5 ATR)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

print(__doc__)
