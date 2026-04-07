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
    account_balance:      float = 100.0   # Starting/current account balance
    risk_per_trade_pct:   float = 1.0        # % of balance risked per trade
    max_trades_per_day:   int   = 23
    daily_loss_limit_pct: float = 3.0        # Stop trading if daily loss > X%
    max_open_positions:   int   = 3
    min_rr:               float = 1.5        # Minimum R:R to take a trade
    spread_pips:          float = 1.5        # Typical spread in pips
    pip_value:            float = 10.0       # USD value of 1 pip per standard lot
    min_win_rate:         float = 0.50       # Task 5: block if win rate < 50%
    min_model_accuracy:   float = 0.55       # Task 6: block if accuracy < 0.55


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

    def __init__(self, config: Optional[RiskConfig] = None, model_accuracy: float = 1.0):
        self.cfg       = config or RiskConfig()
        self.equity    = self.cfg.account_balance
        self._day      = DayStats()
        self._open_positions: int = 0
        self.model_accuracy = model_accuracy   # Task 6

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

        # 5. Overtrading check: Win rate (Task 5)
        total_finished = self._day.winning + self._day.losing
        if total_finished >= 5: # check after 5 trades
            win_rate = self._day.winning / total_finished
            if win_rate < self.cfg.min_win_rate:
                logger.warning(f"[RISK BLOCK] Win rate too low ({win_rate:.1%} < {self.cfg.min_win_rate:.1%})")
                return False

        # 6. Model accuracy gate (Task 6)
        if self.model_accuracy < self.cfg.min_model_accuracy:
            logger.warning(f"[RISK BLOCK] Model accuracy too low ({self.model_accuracy:.3f} < {self.cfg.min_model_accuracy:.3f})")
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
        risk_amount = self.equity * (self.cfg.risk_per_trade_pct / 100.0)
        
        if "XAUUSD" in symbol.upper():
            # Profit = (p2 - p1) * lot * 100
            # risk_amount = abs(entry - sl) * lot * 100
            # lot = risk_amount / (abs(entry - sl) * 100)
            sl_dist = abs(entry_price - sl_price)
            raw_lots = risk_amount / (max(sl_dist, 0.01) * 100.0)
        else:
            sl_pips = abs(entry_price - sl_price) / self._pip_size(symbol)
            raw_lots = risk_amount / (max(sl_pips, 0.1) * self.cfg.pip_value)

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
