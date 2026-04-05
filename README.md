# Forex Trading System — Liquidity + ML
## Production-Grade Quantitative Forex Bot

A modular, end-to-end algorithmic trading system combining **Smart Money Concepts (SMC)**
(liquidity sweeps + break of structure) with an **XGBoost probability filter** and
**MetaTrader 5** live execution.

---

## Architecture

```
forex_bot/
├── config.py                  # Central configuration (all tunables)
├── main.py                    # Unified CLI entry point
│
├── data/
│   ├── loader.py              # OHLCV CSV loading, validation, resampling
│   └── generate_sample.py     # Synthetic GBM data generator (testing)
│
├── features/
│   ├── engineer.py            # Full feature pipeline (structure, liquidity, momentum, time)
│   └── labeler.py             # Supervised label generation (TP/SL simulation)
│
├── models/
│   └── ml_model.py            # XGBoost classifier with CV, calibration, save/load
│
├── strategy/
│   └── engine.py              # O(n) rule-based signal engine with ML gate
│
├── risk/
│   └── manager.py             # Position sizing, daily limits, drawdown guard
│
├── backtest/
│   └── engine.py              # Event-driven simulator + walk-forward validator
│
├── execution/
│   ├── mt5_broker.py          # MetaTrader5 API wrapper (sim fallback if MT5 absent)
│   ├── live_trader.py         # Live trading loop (bar polling + signal execution)
│   └── logger.py              # Structured CSV trade log + CLI dashboard
│
├── saved_models/              # Persisted model files (auto-created)
└── logs/                      # Trade CSVs, equity curve, system log (auto-created)
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install numpy pandas scikit-learn xgboost joblib
# For live trading (Windows only):
pip install MetaTrader5
```

### 2. Generate Sample Data

```bash
python main.py --mode generate --bars 157000
```
Produces `data/raw/EURUSD_M5.csv` with ~18 months of synthetic M5 bars.

**For real trading, replace this with actual OHLCV data** from your broker or a provider
like Dukascopy, HistData, or the MT5 terminal itself. The CSV must have columns:
```
timestamp, open, high, low, close, volume
```

### 3. Train the ML Model

```bash
python main.py --mode train
```

Output includes:
- Time-series cross-validation AUC per fold
- Hold-out test AUC and Brier score
- Feature importance table
- Saved model → `saved_models/forex_xgb.joblib`

### 4. Run a Backtest

```bash
python main.py --mode backtest
```

Output:
```
=======================================================
  BACKTEST RESULTS
=======================================================
  total_trades              270
  win_rate                57.04%
  profit_factor             1.86
  net_pnl_usd           21785.93
  max_drawdown             -5.65%
  sharpe                    1.811
  cagr                    117.06%
=======================================================
```

Full trade log → `logs/backtest_trades.csv`  
Equity curve  → `logs/equity_curve.csv`

### 5. Walk-Forward Validation

```bash
python main.py --mode walkforward --folds 4
```

Each fold trains on all prior data and tests on the next unseen window.
This is the primary anti-overfitting check.

### 6. Live Trading

```bash
python main.py --mode live
```

**Requirements:**
1. MetaTrader5 installed and logged in
2. Credentials set in `config.py`:
   ```python
   MT5_LOGIN    = 123456        # your account number
   MT5_PASSWORD = "yourpass"
   MT5_SERVER   = "ICMarkets-Demo"
   ```
3. Model trained (`saved_models/forex_xgb.joblib` exists)

The live trader polls MT5 every `LIVE_POLL_SECONDS` seconds, checks for new bars,
runs the full feature → strategy → risk pipeline, and places orders when conditions align.

---

## Configuration Reference (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `DATA_PATH` | `data/raw/EURUSD_M5.csv` | Input CSV path |
| `BASE_TF` | `M5` | Primary timeframe |
| `HIGHER_TFS` | `["M15","H1"]` | Timeframes for HTF context |
| `HTF_FOR_TREND` | `H1` | Timeframe for trend filter |
| `RR_RATIO` | `2.0` | Take-profit / stop-loss ratio |
| `SL_ATR_MULT` | `1.5` | Stop = entry ± `SL_ATR_MULT × ATR` |
| `ML_THRESHOLD` | `0.55` | Minimum ML probability to trade |
| `REQUIRE_HTF_ALIGN` | `False` | Enforce HTF trend direction |
| `RISK_PER_TRADE_PCT` | `1.0` | % of equity risked per trade |
| `MAX_TRADES_PER_DAY` | `5` | Daily trade cap |
| `DAILY_LOSS_LIMIT_PCT` | `3.0` | Halt after X% daily loss |
| `SPREAD_PIPS` | `1.5` | Simulated spread cost |
| `SLIPPAGE_PIPS` | `0.5` | Simulated slippage |

---

## Feature Engineering Summary

### Market Structure
- Swing highs/lows (N-bar fractal detection)
- HH / HL / LH / LL classification
- Break of Structure (BOS) — both bullish and bearish
- Trend state (+1 bull, -1 bear, 0 range)
- Distance to last swing high/low (ATR-normalised)

### Liquidity
- Equal highs/lows (within ATR threshold)
- 50-bar rolling previous high/low distance
- Session H/L for Asia, London, NY sessions
- **Liquidity sweeps**: wick-based detection (close reverts after breaching swing)

### Momentum / Displacement
- ATR (14-period Wilder EMA of True Range)
- Candle body size vs 20-bar rolling average
- Impulse detection (body > 2× average → direction flag)
- Upper/lower wick ratios

### Time
- Hour of day (0–23)
- Day of week (0–4)
- Session code (0=Asia, 1=London, 2=NY, 3=Overlap)

---

## Strategy Logic

### Long Entry
1. HTF trend ≥ 0 (or `REQUIRE_HTF_ALIGN = False`)
2. Bullish liquidity sweep detected within last N bars
3. Bullish BOS occurred after the sweep
4. Current bar is a pullback: `PULLBACK_ATR_MIN ≤ (BOS_close − current_close)/ATR ≤ PULLBACK_ATR_MAX`
5. ML probability ≥ `ML_THRESHOLD`

### Short Entry
Mirror conditions (bearish sweep → bearish BOS → pullback upward).

### Trade Levels
- **Entry**: current close (market order)
- **Stop Loss**: entry − `SL_ATR_MULT × ATR` (long) or + (short)
- **Take Profit**: entry + `SL_dist × RR_RATIO` (long) or − (short)

---

## ML Model Details

- **Algorithm**: XGBoost (falls back to RandomForest if XGBoost unavailable)
- **Output**: P(TP hit before SL) — a continuous probability, NOT a binary signal
- **Split**: Chronological 80/20 train/test — no data leakage
- **CV**: `TimeSeriesSplit` with 5 folds (no shuffling)
- **Calibration**: Platt scaling via `CalibratedClassifierCV`
- **Anti-overfit**: `max_depth=4`, `min_child_weight`, `early_stopping_rounds=30`
- **Imbalance**: `scale_pos_weight` set to negative/positive ratio
- **Persistence**: `joblib` → `saved_models/forex_xgb.joblib`

---

## Backtest Engine Details

- **Bar-by-bar simulation**: no look-ahead, each bar sees only past data
- **Spread + slippage**: applied on entry (configurable in pips)
- **Position sizing**: % risk method — lot size calculated from SL distance
- **Daily limits**: max trades/day + daily loss cutoff enforced per calendar day
- **One trade at a time**: conservative default (extend `max_open_positions` for portfolio)
- **Forced EOD close**: open trades closed at last bar's price

### Performance Metrics
| Metric | Formula |
|---|---|
| Win Rate | Winners / Total trades |
| Profit Factor | Gross profit / Gross loss |
| Max Drawdown | Max peak-to-trough equity decline |
| Sharpe Ratio | Annualised return / Annualised std dev |
| Sortino Ratio | Annualised return / Annualised downside std dev |
| CAGR | Compound Annual Growth Rate |

---

## Walk-Forward Validation

Uses an **expanding window** approach:
- Fold 1: Train on first 40% → test on next 15%
- Fold 2: Train on first 55% → test on next 15%
- Fold 3: Train on first 70% → test on next 15%
- Fold 4: Train on first 85% → test on last 15%

Each fold trains its own model and uses the **model's own calibrated threshold** for the OOS scan — no data leakage across folds.

---

## Notes on Synthetic Data vs Real Data

The included sample generator produces **geometric Brownian motion** data. By design:
- Price has no persistent autocorrelation (random walk)
- Patterns are not predictive — ML AUC will be near 0.5 on true OOS

**On real EURUSD data**, you should expect:
- More frequent and cleaner liquidity sweeps
- Stronger session-driven patterns
- ML CV AUC in the range 0.58–0.68 (good signal for this type of system)
- Walk-forward Profit Factor consistently > 1.3 (target before live deployment)

### Sourcing Real Data
- **MT5 terminal**: `copy_rates_range()` for any broker's historical feed
- **Dukascopy**: free tick/M1 data going back to 2003
- **HistData.com**: free M1 OHLCV for major pairs

---

## Live Deployment Checklist

- [ ] Replace synthetic data with real broker data (≥ 2 years M5)
- [ ] Retrain model on real data, verify CV AUC > 0.57
- [ ] Run walk-forward: confirm Avg PF > 1.3 across all folds
- [ ] Paper trade for 4+ weeks, compare to backtest
- [ ] Set `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER` in `config.py`
- [ ] Start on demo account: `python main.py --mode live`
- [ ] Monitor `logs/live_trades.csv` and `logs/system.log`
- [ ] Move to live account only after consistent demo performance

---

## Extending the System

- **Add pairs**: run separate instances per symbol with symbol-specific configs
- **Portfolio**: increase `MAX_OPEN_POSITIONS` and update `live_trader.py` to track per-symbol state
- **Telegram alerts**: add a notifier call in `execution/logger.py`
- **Database logging**: replace CSV writer with SQLite/PostgreSQL in `TradeLogger`
- **Dashboard**: pipe `logs/equity_curve.csv` into Grafana or a Dash app
- **Additional features**: funding rates, COT data, macro calendar events
