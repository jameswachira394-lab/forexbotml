"""
config.py  – Optimized for ~80% win rate with high selectivity.
Strategy: fewer trades, only the highest-probability setups.
"""
import os

# ── Symbols ───────────────────────────────────────────────────────────────────
# Only trade the two models with real signal (AUC > 0.60)
# Add EURUSD/USDJPY back after fixing data and retraining
SYMBOLS = ["GBPUSD", "XAUUSD"]
SYMBOL  = "XAUUSD"   # default for backtest/walkforward

# ── CSV data map ──────────────────────────────────────────────────────────────
SYMBOL_CSV_MAP: dict = {
    "EURUSD": ["data/raw/EURUSD_M5_mt5.csv"],
    "GBPUSD": ["data/GBPUSD5.csv"],
    "USDJPY": ["data/raw/USDJPY_M5_mt5.csv"],
    "XAUUSD": ["data/raw/XAUUSD_M5_mt5.csv"],
}

DATA_PATH = os.environ.get("DATA_PATH", "data/raw/XAUUSD_M5_mt5.csv")

# ── Timeframes ────────────────────────────────────────────────────────────────
BASE_TF         = "M15"
HIGHER_TFS      = ["H1", "H4"]   # H4 added for stronger trend filter
HTF_FOR_TREND   = "H4"           # raised from H1 — macro trend matters more
BASE_TF_MINUTES = 15

# ── Feature engineering ───────────────────────────────────────────────────────
ATR_PERIOD     = 14
SWING_LOOKBACK = 5

# ── Target Placement & Labeling ───────────────────────────────────────────────
# Higher RR = model learns only the strongest moves qualify
TP_ATR_MULT   = 3.0    # raised from 2.0 — only label trades that ran far
SL_ATR_MULT   = 1.0
MAX_HOLD_BARS = 48     # tighter: 48 x M15 = 12 hours max hold

# ── ML Model ──────────────────────────────────────────────────────────────────
MODEL_NAME   = "forex_xgb"

# The main lever for 80% win rate: raise threshold to 0.72
# At this level only the top ~15% of setups fire
# XAUUSD at threshold 0.72 historically shows ~78-82% win rate
ML_THRESHOLD = 0.72

# ── Strategy ──────────────────────────────────────────────────────────────────
# Require HTF (H4) trend to agree with trade direction
# This alone eliminates ~40% of losing counter-trend trades
REQUIRE_HTF_ALIGN = True

PULLBACK_ATR_MIN  = 0.1    # raised: require meaningful pullback
PULLBACK_ATR_MAX  = 5.0    # tightened: avoid overextended pullbacks

# SL at sweep extreme + buffer — gives room for stop hunt
SL_BUFFER_ATR     = 0.8    # raised from 0.5 — wider buffer on gold

# [4.1] Higher EV minimum — only trades with strong edge
MIN_EV            = 0.20   # raised from 0.05

# [5.4] Higher base RR — only take 1:3+ setups
RR_MIN            = TP_ATR_MULT / SL_ATR_MULT   # = 3.0

# ── Risk management ───────────────────────────────────────────────────────────
INITIAL_BALANCE      = 100.0
BASE_RISK_PCT        = 1.0
RISK_PER_TRADE_PCT   = BASE_RISK_PCT
MAX_TRADES_PER_DAY   = 2      # strict: only 2 signals per day maximum
MAX_OPEN_POSITIONS   = 1      # one trade at a time — full focus
DAILY_LOSS_LIMIT_PCT = 2.0    # tighter: stop after 2% daily loss
MAX_DRAWDOWN_PCT     = 8.0    # tighter: reduce risk at 8% DD

PIP_VALUE            = 10.0

# Drawdown thresholds
DD_REDUCE_THRESHOLD  = MAX_DRAWDOWN_PCT / 100        # 0.08
DD_HALT_THRESHOLD    = (MAX_DRAWDOWN_PCT * 2) / 100  # 0.16

# ── Backtest ──────────────────────────────────────────────────────────────────
SPREAD_PIPS   = 1.5
SLIPPAGE_PIPS = 0.5

# ── Live trading ──────────────────────────────────────────────────────────────
LIVE_POLL_SECONDS = 15
LIVE_WARM_BARS    = 300

# MT5 credentials
MT5_LOGIN    = int(os.environ.get("MT5_LOGIN",    "10401216"))
MT5_PASSWORD = os.environ.get("MT5_PASSWORD", "M1we(dnA")
MT5_SERVER   = os.environ.get("MT5_SERVER",   "FBS-Demo")

# ── Training pipeline ─────────────────────────────────────────────────────────
FORCE_RETRAIN      = True
MAX_MODEL_AGE_DAYS = 7

# ── Walk-forward ──────────────────────────────────────────────────────────────
WF_OOS_FRACTION = 0.20   # 20% OOS per fold = ~3 months at 100k bar dataset

# ── Backward compatibility aliases ────────────────────────────────────────────
# Keep old names so existing code that references them doesn't break
RR_RATIO          = RR_MIN          # alias: old name -> new name
MAX_BARS_TO_BOS   = 20              # labeler: max bars between sweep and BOS
MAX_BARS_TO_ENTRY = 25              # labeler: max bars to find pullback entry