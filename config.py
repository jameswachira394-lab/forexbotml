"""
config.py  – Central configuration for the Forex Trading System.
Edit this file.  Do NOT hardcode values inside source modules.
"""
import os

# ── Symbols ───────────────────────────────────────────────────────────────────
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
SYMBOL  = "EURUSD"   # default single-symbol for backtest / generate modes

# ── CSV data map (HistData / Dukascopy) ───────────────────────────────────────
# List every CSV file you have per symbol.  Multiple files are merged.
# Leave [] if you have no CSV yet — those symbols are skipped in training.
SYMBOL_CSV_MAP: dict = {
    "EURUSD": [],   # e.g. ["data/raw/EURUSD_2022.csv", "data/raw/EURUSD_2023.csv"]
    "GBPUSD": ["data/GBPUSD5.csv"],
    "USDJPY": [],
    "USDCHF": [],
    "AUDUSD": [],
    "USDCAD": [],
    "NZDUSD": [],
}

# Fallback path used by --mode generate / backtest (single symbol)
DATA_PATH = os.environ.get("DATA_PATH", "data/raw/EURUSD_M5.csv")

# ── Timeframes ────────────────────────────────────────────────────────────────
BASE_TF         = "M5"
HIGHER_TFS      = ["M15", "H1"]
HTF_FOR_TREND   = "H1"
BASE_TF_MINUTES = 5

# ── Feature engineering ───────────────────────────────────────────────────────
ATR_PERIOD     = 14
SWING_LOOKBACK = 5

# ── Labeling (SMC/ICT) ────────────────────────────────────────────────────────
# Step 2: displacement candle must have body >= this multiple of ATR
DISPLACEMENT_ATR_MULT = 1.5
# Bars to search for displacement after sweep
DISPLACEMENT_BARS     = 5
# Bars to search for CHoCH/BOS after displacement
CHOCH_BARS            = 15
# Bars to search for FVG/OB entry fill after CHoCH
ENTRY_BARS            = 20
# Minimum risk/reward ratio — setups below this are skipped
RR_MIN                = 2.0
# ATR buffer added beyond sweep extreme for SL placement
SL_BUFFER_ATR         = 0.2
# Maximum SL width in ATR units (wider = discard setup)
MAX_SL_ATR            = 3.0
# Minimum sweep strength (ATR units price moved past the level)
MIN_SWEEP_STRENGTH    = 0.05

# ── ML Model ──────────────────────────────────────────────────────────────────
MODEL_NAME   = "forex_xgb"
ML_THRESHOLD = 0.55

# ── Strategy ──────────────────────────────────────────────────────────────────
REQUIRE_HTF_ALIGN = False
PULLBACK_ATR_MIN  = 0.1
PULLBACK_ATR_MAX  = 3.5

# ── Risk management ───────────────────────────────────────────────────────────
INITIAL_BALANCE      = 10_000.0
RISK_PER_TRADE_PCT   = 1.0
MAX_TRADES_PER_DAY   = 5
MAX_OPEN_POSITIONS   = 3
DAILY_LOSS_LIMIT_PCT = 3.0
MIN_RR               = 1.5
PIP_VALUE            = 10.0   # USD/pip/standard-lot for XXXUSD pairs

# ── Backtest ──────────────────────────────────────────────────────────────────
SPREAD_PIPS   = 1.
SLIPPAGE_PIPS = 0.5

# ── Live trading ──────────────────────────────────────────────────────────────
LIVE_POLL_SECONDS = 15
LIVE_WARM_BARS    = 300

# MT5 credentials  (prefer env vars for security)
MT5_LOGIN    = int(os.environ.get("10401216",    0))
MT5_PASSWORD = os.environ.get("M1we(dnA", "")
MT5_SERVER   = os.environ.get("FBS-Demo",   "")

# ── Training pipeline ─────────────────────────────────────────────────────────
FORCE_RETRAIN      = true
MAX_MODEL_AGE_DAYS = 7
