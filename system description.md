# Forex Trading System — System Description

## What Is This System?

An **algorithmic forex trading bot** that combines traditional Smart Money Concepts (SMC) with a machine learning probability filter to automatically detect, evaluate, and execute high-quality trade setups across multiple currency pairs and gold.

The system does not rely on indicators like RSI, MACD, or moving averages. Instead it reads raw price structure — the same way institutional traders do — then uses a trained XGBoost model to decide whether a detected setup is worth taking.

---

## What It Trades

| Symbol | Type | Notes |
|---|---|---|
| EURUSD | Forex Major | Primary pair, most data |
| GBPUSD | Forex Major | London session sensitive |
| USDJPY | Forex Major | Tokyo session driven |
| XAUUSD | Commodity | Gold — requires more data |

**Timeframe:** M15 (15-minute bars) as primary entry  
**Higher timeframe:** H1 for trend context  
**Sessions:** Asia, London, New York — all tracked separately

---

## How the System Works — End to End

```
Raw OHLCV Data (MT5 / CSV)
        ↓
  Feature Engineering
  (market structure, liquidity, session levels, momentum)
        ↓
  Setup Detection (rule-based)
  Liquidity Sweep → Break of Structure → Pullback
        ↓
  ML Probability Filter (XGBoost)
  Only take trades where P(win) × RR − P(loss) > 0
        ↓
  Risk Management Gate
  Position size, drawdown check, daily limits
        ↓
  Order Execution (MetaTrader5)
  Market order with SL at sweep extreme, TP at RR target
        ↓
  Trade Monitoring + Logging
  CSV logs, equity curve, real-time dashboard
```

---

## The Trading Logic (Strategy)

### What the system looks for

**Step 1 — Liquidity Sweep**  
Price temporarily breaks below a swing low (or above a swing high), triggering stop losses of retail traders. This is called a sweep. The candle closes back inside the range, leaving a wick — this is the institutional footprint.

**Step 2 — Break of Structure (BOS)**  
After the sweep, price moves impulsively in the opposite direction, breaking the most recent swing high (for longs) or swing low (for shorts). This confirms that the smart money direction has shifted.

**Step 3 — Pullback Entry**  
Price retraces back toward the BOS level. The system waits for this pullback — entering at a discount rather than chasing the move.

**Step 4 — ML Probability Filter**  
The XGBoost model scores the setup from 0 to 1. Only setups with positive expected value pass:
```
EV = P(win) × RR − P(loss) × 1 > 0.05
```

**Step 5 — Stop Loss Placement**  
SL is placed at the sweep extreme (the wick low or high) plus an ATR buffer — not a fixed pip distance. This respects structure.

**Step 6 — Take Profit**  
TP is set at RR 2:1 by default. Deeper pullbacks dynamically earn a slightly higher RR.

---

## Feature Engineering — What the Model Sees

The ML model receives 45+ features extracted from raw OHLCV data. No external data sources are needed.

### Market Structure
- Higher Highs / Higher Lows / Lower Highs / Lower Lows
- Break of Structure (bullish / bearish)
- Trend direction (bullish / bearish / ranging)
- Distance to last swing high and low (ATR-normalised)

### Liquidity Detection
- Equal highs and equal lows (liquidity pools within ATR threshold)
- Previous 50-bar high/low distance from current price
- Bullish and bearish liquidity sweeps (wick-based detection)

### Session Levels
- Asia session high and low
- London session high and low
- New York session high and low
- Distance from current price to each session level

### Momentum & Displacement
- ATR (14-period, Wilder EMA of True Range)
- Candle body size vs 20-bar rolling average
- Impulse detection — body > 2× average signals displacement
- Upper and lower wick ratios

### Time Features
- Hour of day (0–23)
- Day of week (Monday–Friday)
- Session code (Asia / London / NY / Overlap)

### Setup Features
- Trade direction (long / short)
- SL distance in ATR units
- Time-to-outcome (how many bars to TP or SL)

---

## The Machine Learning Model

### Algorithm
**XGBoost** (Extreme Gradient Boosting) — a tree-based ensemble model. Chosen because:
- Handles tabular financial data well
- Interpretable via feature importance
- Resistant to overfitting with proper regularisation
- No deep learning complexity — robust and fast

### What It Predicts
**P(TP hit before SL)** — the probability that a detected setup reaches its take profit before hitting the stop loss.

This is combined with the R:R ratio to compute Expected Value:
```
EV = P(win) × RR − (1 − P(win)) × 1
```
Trades are only taken when EV > 0.05 (positive edge).

### How It Was Trained

**Data:** Real MT5 OHLCV data  
- EURUSD: 100,000 M1 bars (Nov 2024 – Apr 2026)  
- GBPUSD: 100,000 M1 bars (Nov 2024 – Apr 2026)  
- USDJPY: 50,000 M1 bars (Jul 2025 – Mar 2026)  
- XAUUSD: 50,000 M1 bars (Jul 2025 – Apr 2026)

**Labeling:** Each detected setup is simulated forward bar by bar. If price hits TP before SL → label 1. If SL hits first → label 0. Worst-case candle ambiguity rule: SL is always checked before TP on the same bar.

**Data split (chronological — no shuffling):**
```
[─── 70% Train ───][─ 15% Validation ─][─ 15% Test ─]
```
- **Train:** Model learns from this
- **Validation:** Threshold is tuned here (not on test — no leakage)
- **Test:** Final evaluation only — never touched during training

**Cross-validation:** 5-fold TimeSeriesSplit on the training set only. Each fold trains on earlier data and validates on later data — respects time order.

**Calibration:** Platt scaling via `CalibratedClassifierCV` with `TimeSeriesSplit` — ensures probabilities are realistic, not just rankings.

**Anti-overfitting measures:**
- `max_depth = 4` (shallow trees)
- `early_stopping_rounds = 30`
- `subsample = 0.8`, `colsample_bytree = 0.7`
- `reg_alpha = 0.1`, `reg_lambda = 1.0`
- `min_child_weight` scaled to dataset size

### Training Results (Real Data)

| Symbol | CV AUC | Test AUC | Accuracy | Setups | Verdict |
|---|---|---|---|---|---|
| EURUSD | 0.654 ±0.16 | 0.712 | 74% | 114 | ✅ Deploy-ready |
| GBPUSD | 0.587 ±0.09 | 0.521 | 61% | 177 | ⚠️ Paper trade first |
| USDJPY | 0.638 ±0.20 | 0.850 | 85% | 61 | ⚠️ Need more data |
| XAUUSD | 0.826 ±0.17 | 0.500 | 50% | 69 | ⛔ Overfit — retrain |

---

## Risk Management

Every trade passes through a multi-layer risk gate before execution.

### Position Sizing — Probability-Scaled
Position size scales with the expected value of the trade — higher-confidence setups get slightly larger size, marginal setups get smaller size:
```
risk_amount = equity × base_risk% × EV_scale
lots = risk_amount / (SL_pips × pip_value)
```
`EV_scale` ranges from 0.25× (low confidence) to 1.5× (high confidence).

### Drawdown Control
| Drawdown Level | Action |
|---|---|
| > 10% from peak | Position size halved |
| > 20% from peak | Trading halts entirely |

### Daily Limits
- Maximum 50 trades per day across all symbols
- Maximum 3 simultaneous open positions
- Trading stops if daily loss exceeds 3% of equity

### SL Placement
Stop loss is anchored to the **sweep extreme** (the actual wick) plus an ATR buffer — not a fixed pip distance. This prevents SL from sitting inside the liquidity zone that already got swept.

---

## Backtesting

The system includes a full event-driven backtesting engine:

- **Bar-by-bar simulation** — no lookahead, each bar only sees past data
- **Spread + slippage** applied to every entry (1.5 pip spread + 0.5 pip slippage)
- **Worst-case candle fill** — if both SL and TP are touched in one bar, SL fills first
- **EV-scaled lot sizing** — matches live trading behaviour exactly
- **Walk-forward validation** — expanding window, train on past, test on unseen future

### Walk-Forward Structure
```
Fold 1: [Train on 40%] → [Test 15%]
Fold 2: [Train on 55%] → [Test 15%]
Fold 3: [Train on 70%] → [Test 15%]
Fold 4: [Train on 85%] → [Test 15%]
```
Each fold trains its own model and uses that model's calibrated threshold — no data leakage between folds.

### Metrics Reported
- Win rate
- Profit factor (gross profit / gross loss)
- Max drawdown
- Sharpe ratio (annualised)
- Sortino ratio
- CAGR
- Equity curve (saved to CSV)
- Full trade log with ML probability per trade

---

## Live Trading

### Connection
The system connects to **MetaTrader 5** via the official Python API (`MetaTrader5` package). MT5 must be running on the same Windows machine.

### What happens on each bar close
1. MT5 streamer detects a new completed M15 bar
2. Feature engineering runs on the last 300 bars
3. Strategy engine scans for sweep → BOS → pullback
4. ML model scores the setup
5. Risk manager checks all gates (EV, drawdown, daily limits)
6. If approved: market order placed with SL and TP
7. Position monitored on every subsequent bar
8. When MT5 closes the position (TP or SL hit): P&L logged

### Multi-Symbol
All symbols run simultaneously in a single background thread. A threading lock prevents race conditions when multiple symbols fire at the same time.

### Logging
Every event is written to:
- `logs/system.log` — full system log with timestamps
- `logs/live_trades.csv` — every trade: entry, exit, P&L, ML probability, reason

---

## Project Structure

```
forex_bot/
├── config.py                  ← All settings in one place
├── main.py                    ← CLI: train / backtest / live / report / sync
│
├── data/
│   ├── loader.py              ← OHLCV loading, resampling
│   ├── histdata_parser.py     ← Auto-detects HistData / Dukascopy / MT5 CSV formats
│   └── generate_sample.py     ← Synthetic data for testing
│
├── features/
│   ├── engineer.py            ← 45+ features: structure, liquidity, session, momentum
│   └── labeler.py             ← TP/SL simulation → supervised labels
│
├── models/
│   ├── ml_model.py            ← XGBoost: train, calibrate, save, load, predict EV
│   └── trainer.py             ← Multi-symbol training pipeline
│
├── strategy/
│   └── engine.py              ← O(n) scanner: sweep → BOS → pullback → EV gate
│
├── risk/
│   └── manager.py             ← EV sizing, drawdown control, daily limits
│
├── backtest/
│   └── engine.py              ← Bar-by-bar simulation + walk-forward validator
│
├── execution/
│   ├── mt5_broker.py          ← MT5 order placement, position management
│   ├── mt5_streamer.py        ← Multi-symbol live bar streamer
│   ├── multi_symbol_trader.py ← Live trading loop (thread-safe)
│   └── logger.py             ← Trade logging, CLI dashboard
│
├── reporting/
│   └── dashboard_generator.py ← HTML results dashboard
│
├── saved_models/              ← Trained model files (.joblib)
└── logs/                      ← Trade logs, equity curves
```

---

## How to Run

### 1. Train models from your CSV data
```bash
python main.py --mode train
```

### 2. Backtest a symbol
```bash
python main.py --mode backtest --symbol XAUUSD
```

### 3. Walk-forward validation
```bash
python main.py --mode walkforward --symbol XAUUSD --folds 4
```

### 4. Start live trading
```bash
python main.py --mode live --symbols XAUUSD
```

### 5. Sync fresh data from MT5
```bash
python main.py --mode sync --symbol EURUSD --bars 100000
```

### 6. Generate results dashboard
```bash
python main.py --mode report --symbol EURUSD
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| ML Model | XGBoost 2.0 |
| Data processing | Pandas, NumPy |
| Model persistence | Joblib |
| Broker connection | MetaTrader5 Python API |
| Backtesting | Custom event-driven engine |
| Logging | Python logging + rotating file handler |
| Platform | Windows (required for MT5) |

---

## Key Design Decisions

**No indicators** — the system uses raw OHLCV and derived structure features only. RSI, MACD, Bollinger Bands etc. are lagging and not used.

**ML as filter, not signal** — the rule-based engine finds setups; the ML model decides whether to take them. This separation keeps the logic interpretable.

**Expected value over win rate** — the system optimises for positive EV, not just high win rate. A 45% win rate with RR 2:1 is more valuable than a 60% win rate with RR 1:1.

**Worst-case simulation** — all backtest assumptions are conservative: SL fills before TP on ambiguous bars, spread and slippage are always applied.

**Per-symbol models** — each currency pair gets its own trained model because EURUSD and USDJPY have different volatility profiles, session sensitivities, and structural behaviours.

**Chronological splits only** — no shuffling anywhere in the pipeline. The future never leaks into the past.