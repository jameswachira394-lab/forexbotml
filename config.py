"""
config.py  – Central configuration for the Forex Trading System.
Edit this file.  Do NOT hardcode values inside source modules.
"""
import os

# ── Symbols ───────────────────────────────────────────────────────────────────
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY","XAUUSD"]
SYMBOL  = "EURUSD"   # default single-symbol for backtest / generate modes

# ── CSV data map (HistData / Dukascopy) ───────────────────────────────────────
# List every CSV file you have per symbol.  Multiple files are merged.
# Leave [] if you have no CSV yet — those symbols are skipped in training.
SYMBOL_CSV_MAP: dict = {
    "EURUSD": ["data/raw/EURUSD_M5_mt5.csv"],
    "GBPUSD": ["data/GBPUSD5.csv"],
    "USDJPY": ["data/raw/USDJPY_M5_mt5.csv"],
    "XAUUSD": ["data/raw/XAUUSD_M5_mt5.csv"]
}

# Fallback path used by --mode generate / backtest (single symbol)
DATA_PATH = os.environ.get("DATA_PATH", "data/raw/EURUSD_M5_mt5.csv")

# ── Timeframes ────────────────────────────────────────────────────────────────
BASE_TF         = "M15"
HIGHER_TFS      = ["M45", "H1"]
HTF_FOR_TREND   = "H1"
BASE_TF_MINUTES = 15

# ── Feature engineering ───────────────────────────────────────────────────────
ATR_PERIOD     = 14
SWING_LOOKBACK = 5

# ── Target Placement & Labeling ───────────────────────────────────────────────
TP_ATR_MULT = 2.0
SL_ATR_MULT = 1.0
MAX_HOLD_BARS = 60

# ── ML Model ──────────────────────────────────────────────────────────────────
MODEL_NAME   = "forex_xgb"
ML_THRESHOLD = 0.55

# ── Strategy ──────────────────────────────────────────────────────────────────
REQUIRE_HTF_ALIGN = False
PULLBACK_ATR_MIN  = 0.1
PULLBACK_ATR_MAX  = 3.5

# ── Risk management ───────────────────────────────────────────────────────────
INITIAL_BALANCE      = 100.0
BASE_RISK_PCT        = 1.0    # Base risk to be multiplied by win probability (fractional Kelly)
MAX_TRADES_PER_DAY   = 50
MAX_OPEN_POSITIONS   = 3
DAILY_LOSS_LIMIT_PCT = 3.0
MAX_DRAWDOWN_PCT     = 10.0   # Halve risk if DD exceeds this
PIP_VALUE            = 10.0   # USD/pip/standard-lot for XXXUSD pairs

# ── Backtest ──────────────────────────────────────────────────────────────────
SPREAD_PIPS   = 1.
SLIPPAGE_PIPS = 0.5

# ── Live trading ──────────────────────────────────────────────────────────────
LIVE_POLL_SECONDS = 15
LIVE_WARM_BARS    = 300

# MT5 credentials  (prefer env vars for security)
MT5_LOGIN    = int(os.environ.get("MT5_LOGIN",    "10401216"))
MT5_PASSWORD = os.environ.get("MT5_PASSWORD", "M1we(dnA")
MT5_SERVER   = os.environ.get("MT5_SERVER",   "FBS-Demo")

# ── Training pipeline ─────────────────────────────────────────────────────────
FORCE_RETRAIN      = True
MAX_MODEL_AGE_DAYS = 7
