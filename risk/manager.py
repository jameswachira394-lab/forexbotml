"""
risk/manager.py — FIXED
Fixes:
  [7.1] EV-proportional sizing
  [7.2] Daily cap
  [7.3] Drawdown control
  [NEW] Per-symbol pip value table — XAUUSD/GBPUSD/USDJPY all different
  [NEW] MIN_PROFIT_TARGET: lot size floored so TP >= target profit per trade
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Per-symbol market specifications ─────────────────────────────────────────
# pip_value  = USD profit per 1 pip move per 1 standard lot
# pip_size   = price movement that equals 1 pip
#
# XAUUSD: 1 lot = 100 oz, price in USD/oz, pip = $0.01
#   → pip_value = 100 oz × $0.01 = $1.00 per pip per lot
#   → but brokers often quote gold in "points" where 1 point = $0.01
#   → a 100-point ($1.00) move on 1 lot = $100 profit
#
# GBPUSD: standard FX, pip = 0.0001, 1 lot = 100,000 units
#   → pip_value = $10 per pip per lot
#
# USDJPY: pip = 0.01 (JPY quoted), 1 lot = 100,000 units
#   → pip_value ≈ $9.xx (fluctuates with JPY rate), use $9.00 conservatively

SYMBOL_SPECS = {
    "XAUUSD": {"pip_size": 0.01,   "pip_value": 1.0},    # Gold: $1/pip/lot
    "XAGUSD": {"pip_size": 0.001,  "pip_value": 5.0},    # Silver
    "EURUSD": {"pip_size": 0.0001, "pip_value": 10.0},
    "GBPUSD": {"pip_size": 0.0001, "pip_value": 10.0},
    "AUDUSD": {"pip_size": 0.0001, "pip_value": 10.0},
    "NZDUSD": {"pip_size": 0.0001, "pip_value": 10.0},
    "USDCAD": {"pip_size": 0.0001, "pip_value": 7.5},    # approx
    "USDCHF": {"pip_size": 0.0001, "pip_value": 11.0},   # approx
    "USDJPY": {"pip_size": 0.01,   "pip_value": 9.0},    # approx at 145 JPY/USD
    "EURJPY": {"pip_size": 0.01,   "pip_value": 9.0},
    "GBPJPY": {"pip_size": 0.01,   "pip_value": 9.0},
}

# Fallback for unknown symbols
DEFAULT_SPEC = {"pip_size": 0.0001, "pip_value": 10.0}


def get_symbol_spec(symbol: str) -> dict:
    return SYMBOL_SPECS.get(symbol.upper(), DEFAULT_SPEC)


@dataclass
class RiskConfig:
    account_balance:      float = 100.0
    risk_per_trade_pct:   float = 3.0        # % of balance risked per trade
    min_profit_target:    float = 10.0       # minimum USD profit per winning trade
    max_trades_per_day:   int   = 2
    daily_loss_limit_pct: float = 2.0
    max_open_positions:   int   = 1
    min_rr:               float = 3.0
    # EV scaling bounds
    min_risk_scale:       float = 0.5
    max_risk_scale:       float = 2.0        # allow up to 2× base risk on high-EV trades
    # Drawdown control
    dd_reduce_threshold:  float = 0.08
    dd_halt_threshold:    float = 0.16
    dd_risk_scale:        float = 0.50


@dataclass
class DayStats:
    date:         date  = field(default_factory=date.today)
    trades_taken: int   = 0
    gross_pnl:    float = 0.0
    winning:      int   = 0
    losing:       int   = 0


class RiskManager:

    def __init__(self, config: Optional[RiskConfig] = None):
        self.cfg         = config or RiskConfig()
        self.equity      = self.cfg.account_balance
        self.peak_equity = self.cfg.account_balance
        self._day        = DayStats()
        self._open_pos   = 0

    # ── Approval gate ─────────────────────────────────────────────────────────

    def approve_trade(
        self,
        entry_price: float,
        sl_price:    float,
        tp_price:    float,
        direction:   int,
        symbol:      str   = "EURUSD",
        ml_prob:     float = 0.55,
        rr_ratio:    float = 3.0,
    ) -> bool:
        self._refresh_day()

        # Drawdown halt
        dd = self._current_drawdown()
        if dd >= self.cfg.dd_halt_threshold:
            logger.info(f"[RISK BLOCK] DD halt: {dd:.1%} >= {self.cfg.dd_halt_threshold:.1%}")
            return False

        # Daily trade cap
        if self._day.trades_taken >= self.cfg.max_trades_per_day:
            logger.info(f"[RISK BLOCK] Max trades/day ({self._day.trades_taken}/{self.cfg.max_trades_per_day})")
            return False

        # Daily loss limit
        daily_limit = self.equity * self.cfg.daily_loss_limit_pct / 100
        if self._day.gross_pnl <= -daily_limit:
            logger.info(f"[RISK BLOCK] Daily loss limit hit (P&L={self._day.gross_pnl:.2f})")
            return False

        # Max open positions
        if self._open_pos >= self.cfg.max_open_positions:
            logger.info(f"[RISK BLOCK] Max open positions ({self._open_pos}/{self.cfg.max_open_positions})")
            return False

        # RR gate
        sl_dist = abs(entry_price - sl_price)
        tp_dist = abs(tp_price - entry_price)
        if sl_dist <= 0 or (tp_dist / sl_dist) < self.cfg.min_rr:
            logger.info(f"[RISK BLOCK] R:R {tp_dist/max(sl_dist,1e-9):.2f} < {self.cfg.min_rr}")
            return False

        # Positive EV gate
        ev = ml_prob * rr_ratio - (1 - ml_prob)
        if ev <= 0:
            logger.info(f"[RISK BLOCK] Negative EV ({ev:.3f})")
            return False

        return True

    # ── Position sizing ───────────────────────────────────────────────────────

    def calculate_lot_size(
        self,
        entry_price: float,
        sl_price:    float,
        tp_price:    float   = 0.0,   # needed for profit target floor
        symbol:      str     = "EURUSD",
        ml_prob:     float   = 0.55,
        rr_ratio:    float   = 3.0,
        lot_step:    float   = 0.01,
        min_lot:     float   = 0.01,
        max_lot:     float   = 10.0,
    ) -> float:
        """
        Lot size calculation with three layers:
          1. % risk sizing   (base_risk / sl_pips / pip_value)
          2. EV scaling      (higher probability = larger size, up to 2×)
          3. Profit floor    (size up if TP profit < min_profit_target)
        """
        spec      = get_symbol_spec(symbol)
        pip_size  = spec["pip_size"]
        pip_value = spec["pip_value"]

        # Base risk in dollars
        base_risk = self.equity * (self.cfg.risk_per_trade_pct / 100.0)

        # EV scaling
        ev        = ml_prob * rr_ratio - (1 - ml_prob)
        max_ev    = rr_ratio  # theoretical max at p=1.0
        ev_scale  = np.clip(ev / max(max_ev, 1e-9), 0.0, 1.0)
        risk_scale = self.cfg.min_risk_scale + (
            (self.cfg.max_risk_scale - self.cfg.min_risk_scale) * ev_scale
        )

        # Drawdown reduction
        if self._current_drawdown() >= self.cfg.dd_reduce_threshold:
            risk_scale *= self.cfg.dd_risk_scale

        risk_amount = base_risk * risk_scale

        # Pips to SL
        sl_pips = abs(entry_price - sl_price) / pip_size
        if sl_pips <= 0:
            logger.warning("SL distance zero — returning min lot")
            return min_lot

        # Lot from % risk
        raw_lots = risk_amount / (sl_pips * pip_value)

        # ── Profit target floor ───────────────────────────────────────────────
        # If TP profit at computed lot size < min_profit_target, size up
        if tp_price > 0 and self.cfg.min_profit_target > 0:
            tp_pips          = abs(tp_price - entry_price) / pip_size
            profit_at_raw    = raw_lots * tp_pips * pip_value
            if profit_at_raw < self.cfg.min_profit_target:
                # Minimum lots needed to hit profit target
                min_lots_for_target = self.cfg.min_profit_target / (tp_pips * pip_value)
                raw_lots = max(raw_lots, min_lots_for_target)
                logger.debug(
                    f"[{symbol}] Profit floor applied: "
                    f"sized up to {raw_lots:.3f} lots for ${self.cfg.min_profit_target} target"
                )

        # Round and clamp
        lots = round(raw_lots / lot_step) * lot_step
        lots = float(max(min_lot, min(lots, max_lot)))

        # Log the expected P&L
        tp_pips   = abs(tp_price - entry_price) / pip_size if tp_price > 0 else sl_pips * rr_ratio
        sl_pips_f = abs(entry_price - sl_price) / pip_size
        logger.info(
            f"[{symbol}] Lots={lots} | "
            f"Risk=${lots*sl_pips_f*pip_value:.2f} | "
            f"TP profit=${lots*tp_pips*pip_value:.2f} | "
            f"EV_scale={risk_scale:.2f}"
        )
        return lots

    # ── Trade tracking ────────────────────────────────────────────────────────

    def record_trade_open(self) -> None:
        self._open_pos        += 1
        self._day.trades_taken += 1

    def record_trade_close(self, pnl: float, won: bool) -> None:
        self._open_pos       = max(0, self._open_pos - 1)
        self._day.gross_pnl += pnl
        self.equity          += pnl
        self.peak_equity      = max(self.peak_equity, self.equity)
        if won: self._day.winning += 1
        else:   self._day.losing  += 1
        logger.info(
            f"Trade closed P&L={pnl:+.2f} | "
            f"Equity={self.equity:.2f} | "
            f"DD={self._current_drawdown():.1%} | "
            f"Day W/L={self._day.winning}/{self._day.losing}"
        )

    def day_summary(self) -> dict:
        return {
            "date":      str(self._day.date),
            "trades":    self._day.trades_taken,
            "gross_pnl": round(self._day.gross_pnl, 2),
            "winners":   self._day.winning,
            "losers":    self._day.losing,
            "equity":    round(self.equity, 2),
            "drawdown":  round(self._current_drawdown(), 4),
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _current_drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    def _refresh_day(self) -> None:
        today = date.today()
        if self._day.date != today:
            if self._day.trades_taken > 0:
                logger.info(f"Day reset | {self.day_summary()}")
            self._day = DayStats(date=today)