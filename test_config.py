"""
test_config.py - Relaxed parameters for backtesting

This config has much higher signal generation to test the backtest engine.
Use with: python main.py --mode backtest --symbol XAUUSD (after modifying main.py to import test_config)

Or update config.py with these values for testing.
"""

# ── Symbols ───────────────────────────────────────────────────────────────────
SYMBOLS = ["GBPUSD", "XAUUSD"]
SYMBOL  = "XAUUSD"

# ── CSV data map ──────────────────────────────────────────────────────────────
SYMBOL_CSV_MAP = {
    "EURUSD": ["data/raw/EURUSD_M5_mt5.csv"],
    "GBPUSD": ["data/GBPUSD5.csv"],
    "USDJPY": ["data/raw/USDJPY_M5_mt5.csv"],
    "XAUUSD": ["data/raw/XAUUSD_M5_mt5.csv"],
}

DATA_PATH = "data/raw/XAUUSD_M5_mt5.csv"

# ── Timeframes ────────────────────────────────────────────────────────────────
BASE_TF         = "M15"
HIGHER_TFS      = ["H1", "H4"]
HTF_FOR_TREND   = "H4"
BASE_TF_MINUTES = 15

# ── Feature engineering ───────────────────────────────────────────────────────
ATR_PERIOD     = 14
SWING_LOOKBACK = 5

# ── Target Placement & Labeling ───────────────────────────────────────────────
TP_ATR_MULT   = 3.0
SL_ATR_MULT   = 1.0
MAX_HOLD_BARS = 48

# ── ML Model ──────────────────────────────────────────────────────────────────
MODEL_NAME   = "forex_xgb"

# #### TEST CONFIG: RELAXED THRESHOLD ####
# Original strict: 0.72
# Testing: 0.50 allows ~20-30% of setups to trade
ML_THRESHOLD = 0.50

# ── Strategy ──────────────────────────────────────────────────────────────────
# Keep HTF filter enabled but can disable for testing
REQUIRE_HTF_ALIGN = True

PULLBACK_ATR_MIN  = 0.3
PULLBACK_ATR_MAX  = 2.5

# SL placement
SL_BUFFER_ATR     = 0.8

# #### TEST CONFIG: RELAXED EV ####
# Original: 0.20
# Testing: 0.05 allows weaker edge trades through
MIN_EV            = 0.05

# Base RR
RR_MIN            = TP_ATR_MULT / SL_ATR_MULT   # = 3.0

# ── Risk management ───────────────────────────────────────────────────────────
INITIAL_BALANCE      = 100.0
BASE_RISK_PCT        = 3.0
RISK_PER_TRADE_PCT   = BASE_RISK_PCT
MAX_TRADES_PER_DAY   = 10
MAX_OPEN_POSITIONS   = 5
DAILY_LOSS_LIMIT_PCT = 2.0
MAX_DRAWDOWN_PCT     = 8.0

PIP_VALUE            = 10.0

DD_REDUCE_THRESHOLD  = MAX_DRAWDOWN_PCT / 100
DD_HALT_THRESHOLD    = (MAX_DRAWDOWN_PCT * 2) / 100

# ── Backtest ──────────────────────────────────────────────────────────────────
SPREAD_PIPS   = 1.5
SLIPPAGE_PIPS = 0.5

# ── Live trading ──────────────────────────────────────────────────────────────
LIVE_POLL_SECONDS = 15
LIVE_WARM_BARS    = 300

# MT5 credentials
import os
MT5_LOGIN    = int(os.environ.get("MT5_LOGIN",    "10401216"))
MT5_PASSWORD = os.environ.get("MT5_PASSWORD", "M1we(dnA")
MT5_SERVER   = os.environ.get("MT5_SERVER",   "FBS-Demo")

# ── Training pipeline ─────────────────────────────────────────────────────────
FORCE_RETRAIN      = True
MAX_MODEL_AGE_DAYS = 7

# ── Walk-forward ──────────────────────────────────────────────────────────────
WF_OOS_FRACTION = 0.20

# ── Backward compatibility aliases ────────────────────────────────────────────
RR_RATIO          = RR_MIN
MAX_BARS_TO_BOS   = 20
MAX_BARS_TO_ENTRY = 25

MIN_PROFIT_TARGET = 10.0
ACTIVE_STRATEGY   = "hwr"

# High Win Rate params
HWR_RR             = 1.5
HWR_EMA_FAST       = 20
HWR_EMA_SLOW       = 50
HWR_EMA_TREND      = 200
HWR_RSI_LONG_MIN   = 35.0
HWR_RSI_LONG_MAX   = 58.0
HWR_RSI_SHORT_MIN  = 42.0
HWR_RSI_SHORT_MAX  = 65.0
HWR_EMA_TOUCH_ATR  = 0.8
