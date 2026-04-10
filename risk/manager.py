"""
risk/manager.py — FIXED
Fixes applied:
  [7.1] Probability-scaled position sizing: risk = base_risk * EV_scaled
  [7.2] MAX_TRADES_PER_DAY default tightened; trade count gate enforced
  [7.3] Drawdown control: reduce risk when equity drawdown > threshold
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    account_balance:      float = 10_000.0
    risk_per_trade_pct:   float = 1.0        # base risk — scaled by EV [7.1]
    max_trades_per_day:   int   = 4          # [7.2] tightened default
    daily_loss_limit_pct: float = 3.0
    max_open_positions:   int   = 2
    min_rr:               float = 1.5
    spread_pips:          float = 1.5
    pip_value:            float = 10.0
    # [7.3] Drawdown control
    dd_reduce_threshold:  float = 0.05      # reduce risk when DD > 5%
    dd_halt_threshold:    float = 0.10      # stop trading when DD > 10%
    dd_risk_scale:        float = 0.50      # scale risk to 50% under drawdown
    # [7.1] EV-based sizing limits
    min_risk_scale:       float = 0.25      # floor: never risk less than 25% of base
    max_risk_scale:       float = 1.50      # cap:   never risk more than 150% of base


@dataclass
class DayStats:
    date:         date  = field(default_factory=date.today)
    trades_taken: int   = 0
    gross_pnl:    float = 0.0
    winning:      int   = 0
    losing:       int   = 0


class RiskManager:

    def __init__(self, config: Optional[RiskConfig] = None):
        self.cfg           = config or RiskConfig()
        self.equity        = self.cfg.account_balance
        self.peak_equity   = self.cfg.account_balance   # [7.3] for drawdown tracking
        self._day          = DayStats()
        self._open_pos:int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def approve_trade(
        self,
        entry_price: float,
        sl_price:    float,
        tp_price:    float,
        direction:   int,
        symbol:      str   = "EURUSD",
        ml_prob:     float = 0.55,    # [7.1] used for EV-gating
        rr_ratio:    float = 2.0,
    ) -> bool:
        self._refresh_day()

        # [7.3] Hard halt on deep drawdown
        current_dd = self._current_drawdown()
        if current_dd >= self.cfg.dd_halt_threshold:
            logger.info(
                f"[RISK BLOCK] Drawdown halt: DD={current_dd:.1%} "
                f"≥ {self.cfg.dd_halt_threshold:.1%}"
            )
            return False

        # [7.2] Daily trade cap
        if self._day.trades_taken >= self.cfg.max_trades_per_day:
            logger.info(
                f"[RISK BLOCK] Max trades/day "
                f"({self._day.trades_taken}/{self.cfg.max_trades_per_day})"
            )
            return False

        # Daily loss limit
        daily_limit = self.equity * self.cfg.daily_loss_limit_pct / 100
        if self._day.gross_pnl <= -daily_limit:
            logger.info(
                f"[RISK BLOCK] Daily loss limit "
                f"(P&L={self._day.gross_pnl:.2f} / limit={-daily_limit:.2f})"
            )
            return False

        # Open position cap
        if self._open_pos >= self.cfg.max_open_positions:
            logger.info(
                f"[RISK BLOCK] Max open positions "
                f"({self._open_pos}/{self.cfg.max_open_positions})"
            )
            return False

        # RR gate
        sl_dist = abs(entry_price - sl_price)
        tp_dist = abs(tp_price - entry_price)
        if sl_dist <= 0 or (tp_dist / sl_dist) < self.cfg.min_rr:
            logger.info(
                f"[RISK BLOCK] R:R {tp_dist/max(sl_dist,1e-9):.2f} "
                f"< {self.cfg.min_rr}"
            )
            return False

        # [4.1] Positive EV gate
        ev = ml_prob * rr_ratio - (1 - ml_prob)
        if ev <= 0:
            logger.info(f"[RISK BLOCK] Negative EV ({ev:.3f}) at p={ml_prob:.2f}")
            return False

        return True

    def calculate_lot_size(
        self,
        entry_price: float,
        sl_price:    float,
        symbol:      str   = "EURUSD",
        ml_prob:     float = 0.55,    # [7.1] scales position size
        rr_ratio:    float = 2.0,
        lot_step:    float = 0.01,
        min_lot:     float = 0.01,
        max_lot:     float = 10.0,
    ) -> float:
        """
        [7.1] Risk scales with expected value (EV-proportional sizing).
        [7.3] Risk halved when in drawdown > dd_reduce_threshold.
        """
        # Base risk amount
        base_risk = self.equity * (self.cfg.risk_per_trade_pct / 100.0)

        # [7.1] Scale by EV — higher-probability trades get more size
        ev          = ml_prob * rr_ratio - (1 - ml_prob)
        max_ev      = 1.0 * rr_ratio - 0.0   # theoretical max EV
        ev_scale    = np.clip(ev / max(max_ev, 1e-9), 0.0, 1.0)
        risk_scale  = self.cfg.min_risk_scale + (
            (self.cfg.max_risk_scale - self.cfg.min_risk_scale) * ev_scale
        )

        # [7.3] Halve risk during moderate drawdown
        if self._current_drawdown() >= self.cfg.dd_reduce_threshold:
            risk_scale *= self.cfg.dd_risk_scale

        risk_amount = base_risk * risk_scale

        # Lot calculation
        sl_pips = abs(entry_price - sl_price) / self._pip_size(symbol)
        if sl_pips <= 0:
            logger.warning("SL distance zero — returning min lot")
            return min_lot

        raw_lots = risk_amount / (sl_pips * self.cfg.pip_value)
        lots     = round(raw_lots / lot_step) * lot_step
        return float(max(min_lot, min(lots, max_lot)))

    def record_trade_open(self) -> None:
        self._open_pos        += 1
        self._day.trades_taken += 1

    def record_trade_close(self, pnl: float, won: bool) -> None:
        self._open_pos    = max(0, self._open_pos - 1)
        self._day.gross_pnl += pnl
        self.equity         += pnl
        self.peak_equity     = max(self.peak_equity, self.equity)
        if won: self._day.winning += 1
        else:   self._day.losing  += 1
        dd = self._current_drawdown()
        logger.info(
            f"Trade closed P&L={pnl:+.2f} | Equity={self.equity:.2f} | "
            f"DD={dd:.1%} | Day W/L={self._day.winning}/{self._day.losing}"
        )

    def day_summary(self) -> dict:
        return {
            "date":       str(self._day.date),
            "trades":     self._day.trades_taken,
            "gross_pnl":  round(self._day.gross_pnl, 2),
            "winners":    self._day.winning,
            "losers":     self._day.losing,
            "equity":     round(self.equity, 2),
            "drawdown":   round(self._current_drawdown(), 4),
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

    @staticmethod
    def _pip_size(symbol: str) -> float:
        return 0.01 if "JPY" in symbol.upper() else 0.0001


# numpy needed for np.clip in calculate_lot_size
import numpy as np