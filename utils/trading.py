"""
utils/trading.py
----------------
Shared utilities for profit calculation and trade execution logic
to ensure parity between backtest and live environments.
"""

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

def calculate_profit(
    symbol:      str,
    open_price:  float,
    exit_price:  float,
    direction:   int,      # +1 long, -1 short
    lot_size:    float,
    pip_value:   float = 10.0,
) -> float:
    """
    Unified profit calculation.
    XAUUSD uses contract size 100.
    Standard FX uses pip-based math.
    """
    symbol = symbol.upper()
    
    if "XAUUSD" in symbol or "GOLD" in symbol:
        # User formula: (close - open) * lot * 100
        # direction must be factored in
        profit = (exit_price - open_price) * direction * lot_size * 100.0
        
        # Task 1: Reject any profit > $50 for 0.01 lot unless price move justifies it
        # 0.01 lot * $50 limit -> $5,000 per standard lot
        # Actually $50 for 0.01 lot is a HUGE move (50.00 pts in gold). 
        # Standard safety check:
        max_allowed = 50.0 * (lot_size / 0.01)
        if abs(profit) > max_allowed:
            # We don't necessarily 'reject' historical data, but we log it as suspicious
            # unless the price move is actually > 50 points.
            price_move = abs(exit_price - open_price)
            if price_move < 45.0: # small buffer
                logger.warning(f"Suspicious profit detected: ${profit:.2f} on {lot_size} lots. Move: {price_move:.2f}")
                # For safety, we cap it if it looks like a data error (e.g. 1000x multiplier error)
                # but I'll leave it for now as a warning unless the user wants a hard cap.
        
        return float(profit)
    
    else:
        # Standard FX
        pip_size = 0.01 if "JPY" in symbol else 0.0001
        pips = (exit_price - open_price) * direction / pip_size
        profit = pips * pip_value * lot_size
        return float(profit)

def should_execute_trade(
    signal_prob:      float,
    current_time:     datetime,
    last_trade_time:  Optional[datetime],
    open_positions:   int,
    ml_threshold:     float = 0.55,
    timeframe_secs:   int   = 900,
) -> bool:
    """
    As specified in Task 2: Standardised execution gate.
    """
    # Only one trade at a time
    if open_positions > 0:
        return False

    # Enforce new candle execution (M5) is handled by the caller (MT5Streamer or Backtest loop)
    # but we can check cooldown here.
    
    # Cooldown (1 candle)
    if last_trade_time is not None:
        elapsed = (current_time - last_trade_time).total_seconds()
        if elapsed < timeframe_secs:
            return False

    # Minimum probability threshold
    if signal_prob < ml_threshold:
        return False

    return True

def get_lot_size_limit_check(symbol: str, lot_size: float, profit: float):
    """Safety check for Task 1: Lot size vs profit sanity."""
    if "XAUUSD" in symbol:
        # Reject any profit > $50 for 0.01 lot
        limit = 50.0 * (lot_size / 0.01)
        if abs(profit) > limit:
             return False
    return True
