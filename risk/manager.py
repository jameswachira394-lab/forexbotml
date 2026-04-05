"""
risk/manager.py
---------------
Risk management layer.

Responsibilities:
  - Position sizing based on % risk per trade
  - Daily loss cutoff and max-trades-per-day enforcement
  - Drawdown tracking
  - Trade validation gate (all rules must pass before execution)

All monetary values are in account currency units.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    account_balance:      float = 10_000.0   # Starting/current account balance
    risk_per_trade_pct:   float = 1.0        # % of balance risked per trade
    max_trades_per_day:   int   = 5
    daily_loss_limit_pct: float = 3.0        # Stop trading if daily loss > X%
    max_open_positions:   int   = 3
    min_rr:               float = 1.5        # Minimum R:R to take a trade
    spread_pips:          float = 1.5        # Typical spread in pips
    pip_value:            float = 10.0       # USD value of 1 pip per standard lot


@dataclass
class DayStats:
    date:          date  = field(default_factory=date.today)
    trades_taken:  int   = 0
    gross_pnl:     float = 0.0
    winning:       int   = 0
    losing:        int   = 0


class RiskManager:
    """
    Gate-keeper that must approve every trade before execution.
    Also tracks intraday statistics and account equity.
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.cfg       = config or RiskConfig()
        self.equity    = self.cfg.account_balance
        self._day      = DayStats()
        self._open_positions: int = 0

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def approve_trade(
        self,
        entry_price:  float,
        sl_price:     float,
        tp_price:     float,
        direction:    int,     # +1 long, -1 short
        symbol:       str = "EURUSD",
    ) -> bool:
        """
        Return True only if ALL risk rules pass.
        Logs the specific rule that blocks a trade.
        """
        self._refresh_day()

        # 1. Daily trade count
        if self._day.trades_taken >= self.cfg.max_trades_per_day:
            logger.info(
                f"[RISK BLOCK] Max trades/day reached "
                f"({self._day.trades_taken}/{self.cfg.max_trades_per_day})"
            )
            return False

        # 2. Daily loss limit
        daily_loss_limit = self.equity * self.cfg.daily_loss_limit_pct / 100
        if self._day.gross_pnl <= -daily_loss_limit:
            logger.info(
                f"[RISK BLOCK] Daily loss limit hit "
                f"(P&L: {self._day.gross_pnl:.2f} / Limit: {-daily_loss_limit:.2f})"
            )
            return False

        # 3. Max open positions
        if self._open_positions >= self.cfg.max_open_positions:
            logger.info(
                f"[RISK BLOCK] Max open positions "
                f"({self._open_positions}/{self.cfg.max_open_positions})"
            )
            return False

        # 4. RR check
        sl_dist = abs(entry_price - sl_price)
        tp_dist = abs(tp_price  - entry_price)
        if sl_dist <= 0 or (tp_dist / sl_dist) < self.cfg.min_rr:
            logger.info(
                f"[RISK BLOCK] R:R too low "
                f"({tp_dist / sl_dist:.2f} < {self.cfg.min_rr})"
            )
            return False

        return True

    def calculate_lot_size(
        self,
        entry_price: float,
        sl_price:    float,
        symbol:      str = "EURUSD",
        lot_step:    float = 0.01,
        min_lot:     float = 0.01,
        max_lot:     float = 10.0,
    ) -> float:
        """
        Position size in lots so that a 1-SL loss equals exactly
        risk_per_trade_pct % of current equity.

        Formula:
            lots = (equity * risk_pct/100) / (sl_pips * pip_value_per_lot)
        """
        sl_pips = abs(entry_price - sl_price) / self._pip_size(symbol)
        if sl_pips <= 0:
            logger.warning("SL distance is zero – returning minimum lot size.")
            return min_lot

        risk_amount = self.equity * (self.cfg.risk_per_trade_pct / 100.0)
        raw_lots    = risk_amount / (sl_pips * self.cfg.pip_value)

        # Round to lot_step and clamp
        lots = round(raw_lots / lot_step) * lot_step
        lots = max(min_lot, min(lots, max_lot))
        return lots

    def record_trade_open(self) -> None:
        self._open_positions += 1
        self._day.trades_taken += 1

    def record_trade_close(self, pnl: float, won: bool) -> None:
        self._open_positions  = max(0, self._open_positions - 1)
        self._day.gross_pnl  += pnl
        self.equity           += pnl
        if won:
            self._day.winning += 1
        else:
            self._day.losing  += 1
        logger.info(
            f"Trade closed P&L={pnl:+.2f} | "
            f"Equity={self.equity:.2f} | "
            f"Day W/L={self._day.winning}/{self._day.losing}"
        )

    def day_summary(self) -> dict:
        return {
            "date":           str(self._day.date),
            "trades":         self._day.trades_taken,
            "gross_pnl":      round(self._day.gross_pnl, 2),
            "winners":        self._day.winning,
            "losers":         self._day.losing,
            "equity":         round(self.equity, 2),
        }

    # ──────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────

    def _refresh_day(self) -> None:
        today = date.today()
        if self._day.date != today:
            if self._day.trades_taken > 0:
                logger.info(f"Day reset | Previous: {self.day_summary()}")
            self._day = DayStats(date=today)

    @staticmethod
    def _pip_size(symbol: str) -> float:
        """Return the pip size (0.0001 for most FX, 0.01 for JPY pairs)."""
        if "JPY" in symbol.upper():
            return 0.01
        return 0.0001
