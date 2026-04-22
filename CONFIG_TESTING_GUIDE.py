"""
QUICK-START GUIDE: How to Test the System with More Signals

All edits below should be made to config.py (around lines 40-60)
"""

# ============================================================================
# SCENARIO 1: Just test if system works (very relaxed)
# ============================================================================
# Replace config.py lines ~40-60 with:

ML_THRESHOLD = 0.50        # (was 0.72) - Relax to see 20-30% of setups
MIN_EV = 0.05              # (was 0.20) - Allow weaker edges
REQUIRE_HTF_ALIGN = False  # (was True)  - Allow counter-trend trades
PULLBACK_ATR_MIN = 0.1     # (was 0.3)  - Shallower pullbacks OK
PULLBACK_ATR_MAX = 3.5     # (was 2.5)  - Allow deeper pullbacks

# Then run:
# python main.py --mode backtest --symbol XAUUSD --data data/raw/XAUUSD_M5_mt5.csv

# Expected: 20-50 signals on 157k bars


# ============================================================================
# SCENARIO 2: Moderate settings (balanced testing)
# ============================================================================

ML_THRESHOLD = 0.60        # Mid-range confidence - find good setups
MIN_EV = 0.10              # Allow some edge trades
REQUIRE_HTF_ALIGN = True   # Keep only trend-aligned
PULLBACK_ATR_MIN = 0.25    # Some minimum pullback
PULLBACK_ATR_MAX = 3.0     # Reasonable limit

# Expected: 10-30 signals


# ============================================================================
# SCENARIO 3: Conservative (production-ready)
# ============================================================================

ML_THRESHOLD = 0.68         # High confidence
MIN_EV = 0.15               # Good edge requirement
REQUIRE_HTF_ALIGN = True    # Only trend trades
PULLBACK_ATR_MIN = 0.3      # Meaningful pullback
PULLBACK_ATR_MAX = 2.5      # Tight pullback zone

# Expected: 1-5 signals (current config - highest quality only)


# ============================================================================
# HOW TO CHANGE config.py
# ============================================================================

# 1. Open: config.py
# 2. Find the section labeled:
#    "# ── ML Model ──────────────────────────────────────────────────────────"
# 3. Change these lines:
#    ML_THRESHOLD = 0.50  (instead of 0.72)
#    MIN_EV = 0.05        (instead of 0.20)
#    REQUIRE_HTF_ALIGN = False  (instead of True)
#    PULLBACK_ATR_MIN = 0.1     (instead of 0.3)
#    PULLBACK_ATR_MAX = 3.5     (instead of 2.5)
#
# 4. Save the file
# 5. Run backtest:
#    python main.py --mode backtest --symbol XAUUSD --data data/raw/XAUUSD_M5_mt5.csv
#
# 6. Check results:
#    - Trades: logs/XAUUSD_backtest_trades.csv
#    - Equity:  logs/XAUUSD_equity_curve.csv


# ============================================================================
# WHAT EACH PARAMETER DOES
# ============================================================================

# ML_THRESHOLD (0.0-1.0)
#   ├─ 0.50 = "Relaxed"      - Allow all middle signals
#   ├─ 0.60 = "Balanced"     - Good quality setups
#   ├─ 0.72 = "Strict"       - Only best signals (current)
#   └─ 1.00 = "Impossible"    - No signals generated

# MIN_EV (0.0+)
#   ├─ 0.00 = "Any trade"    - All setups with positive EV
#   ├─ 0.05 = "Relaxed"      - Lightweight edge requirement
#   ├─ 0.20 = "Conservative" - Good edge (current)
#   └─ 0.50 = "Strict"       - Very rare signals

# REQUIRE_HTF_ALIGN (True/False)
#   ├─ True  = Trend-aligned only (safer)
#   └─ False = Any directional setup (more signals)

# PULLBACK_ATR_MIN (0-2)
#   ├─ 0.1 = "Shallow pullbacks OK" (more entries)
#   ├─ 0.3 = "Meaningful pullback"  (current)
#   └─ 0.5 = "Strong pullback only"

# PULLBACK_ATR_MAX (1-5)
#   ├─ 2.5 = "Tight range" (current)
#   ├─ 3.5 = "Moderate range"
#   └─ 5.0 = "Allow deep pullbacks"


# ============================================================================
# BACKTEST COMMANDS
# ============================================================================

# Single symbol (returns logs/SYMBOL_backtest_trades.csv)
# python main.py --mode backtest --symbol XAUUSD --data data/raw/XAUUSD_M5_mt5.csv

# Single symbol - GBPUSD
# python main.py --mode backtest --symbol GBPUSD --data data/raw/GBPUSD_M5_mt5.csv

# Walk-forward validation (4 out-of-sample folds)
# python main.py --mode walkforward --symbol GBPUSD

# Always check results:
# - Open: logs/XAUUSD_backtest_trades.csv to see individual trades
# - Open: logs/XAUUSD_equity_curve.csv to see P&L over time


# ============================================================================
# TIPS FOR TESTING
# ============================================================================

# 1. Start with SCENARIO 1 to verify system generates signals
# 2. Check logs/SYMBOL_backtest_trades.csv for trade details
# 3. Gradually tighten parameters toward SCENARIO 3
# 4. Monitor Win Rate, Profit Factor, Max Drawdown
# 5. Print latest equity_curve and trades CSVs to see results
# 6. Only trust out-of-sample results (use walkforward mode)

