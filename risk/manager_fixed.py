"""
risk/manager_fixed.py — INSTITUTIONAL-GRADE
=============================================

FIXES:
  [1] Cost-aware EV: EV = P(win) × (RR - cost) - P(loss) × (1 + cost)
  [2] Trade diagnostics: MAE/MFE per trade
  [3] Failure-safe: rejects if cost too high
  [4] Regime-aware sizing (volatility adjustment)
  [5] Walk-forward compatible (no future data)
  [6] Per-symbol pip tables (XAUUSD, GBPUSD, USDJPY all different)
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Per-symbol market specifications
SYMBOL_SPECS = {
    "XAUUSD": {"pip_size": 0.01,   "pip_value": 1.0, "description": "Gold: $1/pip/lot"},
    "XAGUSD": {"pip_size": 0.001,  "pip_value": 5.0, "description": "Silver: $5/pip/lot"},
    "EURUSD": {"pip_size": 0.0001, "pip_value": 10.0, "description": "Major: $10/pip/lot"},
    "GBPUSD": {"pip_size": 0.0001, "pip_value": 10.0, "description": "Major: $10/pip/lot"},
    "AUDUSD": {"pip_size": 0.0001, "pip_value": 10.0, "description": "Minor: $10/pip/lot"},
    "NZDUSD": {"pip_size": 0.0001, "pip_value": 10.0, "description": "Minor: $10/pip/lot"},
    "USDCAD": {"pip_size": 0.0001, "pip_value": 7.5, "description": "Cross: $7.5/pip/lot"},
    "USDCHF": {"pip_size": 0.0001, "pip_value": 11.0, "description": "Cross: $11/pip/lot"},
    "USDJPY": {"pip_size": 0.01,   "pip_value": 9.0, "description": "JPY: $9/pip/lot"},
    "EURJPY": {"pip_size": 0.01,   "pip_value": 9.0, "description": "JPY: $9/pip/lot"},
    "GBPJPY": {"pip_size": 0.01,   "pip_value": 9.0, "description": "JPY: $9/pip/lot"},
}

DEFAULT_SPEC = {"pip_size": 0.0001, "pip_value": 10.0, "description": "Default FX pair"}


def get_symbol_spec(symbol: str) -> dict:
    return SYMBOL_SPECS.get(symbol.upper(), DEFAULT_SPEC)


@dataclass
class TradeEntry:
    """Diagnostics record per trade."""
    entry_ts:        pd.Timestamp = None
    exit_ts:         Optional[pd.Timestamp] = None
    direction:       int = 0           # +1 long, -1 short
    entry_price:     float = 0.0
    sl_price:        float = 0.0
    tp_price:        float = 0.0
    exit_price:      float = 0.0
    symbol:          str = ""
    
    # Trade metrics
    position_size_units: float = 0.0
    pnl_price_units:     float = 0.0  # price move × position_size
    pnl_usd:             float = 0.0
    
    # Diagnostics (NEW)
    mae_pips:        float = 0.0       # max adverse excursion
    mfe_pips:        float = 0.0       # max favorable excursion
    mae_usd:         float = 0.0
    mfe_usd:         float = 0.0
    
    # Context
    ml_prob:         float = 0.0
    rr_ratio:        float = 0.0
    ev_computed:     float = 0.0       # EV at entry
    cost_in_pips:    float = 0.0
    
    # Exit reason
    exit_reason:     str = ""
    
    @property
    def days_held(self) -> float:
        if self.exit_ts and self.entry_ts:
            return (self.exit_ts - self.entry_ts).total_seconds() / (86400)
        return 0
    
    @property
    def realized_rr(self) -> float:
        """Realized R:R = actual_profit / actual_loss."""
        if self.pnl_usd == 0:
            return 0
        # Compute loss from SL distance, profit from actual
        spec = get_symbol_spec(self.symbol)
        pip_sz = spec["pip_size"]
        pip_val = spec["pip_value"]
        
        sl_dist_pips = abs(self.entry_price - self.sl_price) / pip_sz
        risk_usd = sl_dist_pips * pip_val * self.position_size_units
        
        if risk_usd > 0:
            return self.pnl_usd / risk_usd
        return 0


@dataclass
class RiskConfig:
    account_balance:      float = 100.0
    risk_per_trade_pct:   float = 2.0        # % of balance risked
    min_profit_target:    float = 10.0       # USD minimum per winning trade
    max_trades_per_day:   int   = 2
    daily_loss_limit_pct: float = 2.0
    max_open_positions:   int   = 1
    min_rr:               float = 3.0
    
    # Cost gates
    max_spread_pips:      float = 2.0        # reject if spread > this
    max_slippage_pips:    float = 1.0        # reject if slippage risks > this
    min_ev_after_cost:    float = 0.10       # minimum EV accounting for costs
    
    # EV scaling
    min_risk_scale:       float = 0.5
    max_risk_scale:       float = 2.0
    
    # Drawdown control
    dd_reduce_threshold:  float = 0.08
    dd_halt_threshold:    float = 0.16
    dd_risk_scale:        float = 0.50
    
    # Volatility regime adjustment
    atr_multiplier_lo:    float = 0.75      # reduce risk in high volatility
    atr_multiplier_hi:    float = 1.25      # increase risk in stable markets


@dataclass
class DayStats:
    date:           date = field(default_factory=date.today)
    trades_taken:   int = 0
    trades_won:     int = 0
    gross_pnl:      float = 0.0
    commissions:    float = 0.0
    net_pnl:        float = 0.0


class RiskManager:
    """
    Institutional-grade risk management with cost-aware EV filtering.
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.cfg           = config or RiskConfig()
        self.equity        = self.cfg.account_balance
        self.peak_equity   = self.cfg.account_balance
        self._day          = DayStats()
        self._open_trades: Dict[str, TradeEntry] = {}  # symbol -> TradeEntry
        self._trade_log:   List[TradeEntry] = []
        self._bars_since_trade: Dict[str, int] = {}

    # ───────────────────────────────────────────────────────────────────────────
    # Main approval gate with cost-aware EV
    # ───────────────────────────────────────────────────────────────────────────

    def approve_trade(
        self,
        entry_price:     float,
        sl_price:        float,
        tp_price:        float,
        direction:       int,
        symbol:          str = "EURUSD",
        ml_prob:         float = 0.55,
        rr_ratio:        float = 3.0,
        spread_pips:     float = 1.5,
        slippage_pips:   float = 0.5,
        atr:             float = 1.0,
        atr_percentile:  float = 0.5,
    ) -> tuple[bool, str]:
        """
        Comprehensive approval gate.
        
        Returns
        -------
        (approved, reason_string)
        """
        self._refresh_day()

        spec = get_symbol_spec(symbol)
        pip_sz = spec["pip_size"]

        # ────── 1. Drawdown halt ──────────────────────────────────────────────
        dd = self._current_drawdown()
        if dd >= self.cfg.dd_halt_threshold:
            return False, f"DD halt: {dd:.1%} >= {self.cfg.dd_halt_threshold:.1%}"

        # ────── 2. Daily trade cap ────────────────────────────────────────────
        if self._day.trades_taken >= self.cfg.max_trades_per_day:
            return False, f"Max trades/day: {self._day.trades_taken}/{self.cfg.max_trades_per_day}"

        # ────── 3. Daily loss limit ───────────────────────────────────────────
        daily_limit = self.equity * self.cfg.daily_loss_limit_pct / 100
        if self._day.gross_pnl <= -daily_limit:
            return False, f"Daily loss limit: {self._day.gross_pnl:.2f} <= -{daily_limit:.2f}"

        # ────── 4. Max open positions ──────────────────────────────────────────
        if len(self._open_trades) >= self.cfg.max_open_positions:
            return False, f"Max open positions: {len(self._open_trades)}/{self.cfg.max_open_positions}"

        # ────── 5. R:R gate ────────────────────────────────────────────────────
        sl_dist = abs(entry_price - sl_price) / pip_sz  # pips
        tp_dist = (tp_price - entry_price) * direction / pip_sz  # pips
        if sl_dist <= 0:
            return False, "SL distance ≤ 0"
        
        actual_rr = tp_dist / sl_dist
        if actual_rr < self.cfg.min_rr:
            return False, f"R:R {actual_rr:.2f} < {self.cfg.min_rr}"

        # ────── 6. Cost gates (NEW) ────────────────────────────────────────────
        cost_pips = spread_pips + slippage_pips
        if spread_pips > self.cfg.max_spread_pips:
            return False, f"Spread {spread_pips:.1f} > {self.cfg.max_spread_pips}"
        
        if cost_pips > 3.0:
            return False, f"Total cost {cost_pips:.1f} too high"

        # ────── 7. Cost-aware EV gate (CRITICAL NEW FIX) ──────────────────────
        ev = self._compute_ev_with_cost(ml_prob, actual_rr, cost_pips)
        if ev < self.cfg.min_ev_after_cost:
            return False, f"EV {ev:.3f} < min {self.cfg.min_ev_after_cost:.3f} (cost-adjusted)"

        # ────── 8. Minimum profit target ───────────────────────────────────────
        # Risk amount: sl_dist * position_size * pip_value
        # Check: if we win, do we make at least min_profit_target?
        # This requires knowing position size, which is computed later
        # For now, we accept and check during sizing

        return True, "APPROVED"

    # ───────────────────────────────────────────────────────────────────────────
    # Cost-aware EV calculation
    # ───────────────────────────────────────────────────────────────────────────

    def _compute_ev_with_cost(
        self,
        ml_prob: float,
        rr_ratio: float,
        cost_pips: float,
    ) -> float:
        """
        Cost-aware EV:
        
        EV = P(win) × (RR - cost_pips) - P(loss) × (1 + cost_pips)
        
        Interpretation:
          - Entry costs cost_pips in slippage (against us)
          - Effective RR is RR - cost_pips (we lose cost to slippage at start)
          - On exit, another cost (spread at exit), so losing side pays extra
        """
        p_win = ml_prob
        p_loss = 1 - ml_prob
        
        # Effective RR after cost bite at entry
        rr_net = max(0, rr_ratio - cost_pips)
        
        # On losing side, we pay exit spread too
        loss_with_exit_cost = 1 + cost_pips
        
        ev = p_win * rr_net - p_loss * loss_with_exit_cost
        return ev

    # ───────────────────────────────────────────────────────────────────────────
    # Position sizing (EV-scaled, volatility-adjusted)
    # ───────────────────────────────────────────────────────────────────────────

    def calculate_position_size(
        self,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        direction: int,
        symbol: str = "EURUSD",
        ml_prob: float = 0.55,
        rr_ratio: float = 3.0,
        atr: float = 1.0,
        atr_percentile: float = 0.5,
    ) -> float:
        """
        Calculate position size (units) using:
          1. Risk per trade (%)
          2. EV-proportional scaling
          3. Volatility adjustment
          4. Minimum profit target
          
        Returns
        -------
        Position size in units (instrument-dependent).
        """
        spec = get_symbol_spec(symbol)
        pip_sz = spec["pip_size"]
        pip_val = spec["pip_value"]

        # Base risk amount
        base_risk_usd = self.equity * self.cfg.risk_per_trade_pct / 100

        # Drawdown scale
        dd = self._current_drawdown()
        if dd >= self.cfg.dd_reduce_threshold:
            base_risk_usd *= self.cfg.dd_risk_scale

        # Volatility adjustment (high ATR percentile = higher vol = reduce size)
        # atr_percentile ∈ [0, 1], 0=low vol, 1=high vol
        vol_scale = self.cfg.atr_multiplier_lo + \
                    (self.cfg.atr_multiplier_hi - self.cfg.atr_multiplier_lo) * (1 - atr_percentile)
        base_risk_usd *= vol_scale

        # EV scaling (higher EV = larger position)
        spread_pips = 1.5
        slippage_pips = 0.5
        cost_pips = spread_pips + slippage_pips
        ev = self._compute_ev_with_cost(ml_prob, rr_ratio, cost_pips)
        ev_scale = max(0.25, min(2.0, 1.0 + ev * 2))  # 0.25-2.0 range
        base_risk_usd *= ev_scale

        # Convert to position units
        sl_dist_pips = abs(entry_price - sl_price) / pip_sz
        if sl_dist_pips <= 0:
            return 0.0

        position_units = base_risk_usd / (sl_dist_pips * pip_val)

        # Apply minimum profit target (if we win, profit ≥ min_target)
        # profit_if_win = position_units × sl_dist_pips × pip_val × rr_ratio
        # Require: profit_if_win ≥ min_profit_target
        tp_dist_pips = (tp_price - entry_price) * direction / pip_sz
        profit_if_win = position_units * tp_dist_pips * pip_val
        
        if profit_if_win < self.cfg.min_profit_target:
            # Scale up to meet minimum
            scaling_factor = self.cfg.min_profit_target / max(profit_if_win, 1e-9)
            position_units *= scaling_factor
            logger.debug(
                f"[{symbol}] Scaled position to meet min profit: "
                f"{profit_if_win:.2f} → {profit_if_win * scaling_factor:.2f} USD"
            )

        return position_units

    # ───────────────────────────────────────────────────────────────────────────
    # Trade logging (diagnostics)
    # ───────────────────────────────────────────────────────────────────────────

    def open_trade(
        self,
        entry_ts: pd.Timestamp,
        direction: int,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        symbol: str,
        position_size_units: float,
        ml_prob: float,
        rr_ratio: float,
        cost_pips: float,
    ) -> None:
        """Create trade record."""
        spec = get_symbol_spec(symbol)
        pip_sz = spec["pip_size"]
        
        sl_dist_pips = abs(entry_price - sl_price) / pip_sz
        ev = self._compute_ev_with_cost(ml_prob, rr_ratio, cost_pips)

        trade = TradeEntry(
            entry_ts=entry_ts,
            direction=direction,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            symbol=symbol,
            position_size_units=position_size_units,
            ml_prob=ml_prob,
            rr_ratio=rr_ratio,
            ev_computed=ev,
            cost_in_pips=cost_pips,
        )
        self._open_trades[symbol] = trade
        logger.info(
            f"OPEN {symbol} {'+' if direction==1 else '-'} "
            f"@ {entry_price:.5f} | SL={sl_price:.5f} | TP={tp_price:.5f} | "
            f"Units={position_size_units:.4f} | EV={ev:.3f}"
        )

    def close_trade(
        self,
        symbol: str,
        exit_ts: pd.Timestamp,
        exit_price: float,
        exit_reason: str,
    ) -> Optional[TradeEntry]:
        """Close trade and record diagnostics."""
        if symbol not in self._open_trades:
            return None

        trade = self._open_trades.pop(symbol)
        spec = get_symbol_spec(symbol)
        pip_sz = spec["pip_size"]
        pip_val = spec["pip_value"]

        trade.exit_ts = exit_ts
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason

        # P&L calculation
        price_diff = (exit_price - trade.entry_price) * trade.direction
        trade.pnl_price_units = price_diff
        trade.pnl_usd = price_diff * pip_val * trade.position_size_units

        # MAE/MFE (simplified; requires keeping track during bars)
        # For now, compute approximate MAE at exit
        if exit_reason == "SL":
            trade.mae_pips = abs(exit_price - trade.entry_price) / pip_sz
            trade.mae_usd = trade.mae_pips * pip_val * trade.position_size_units
        elif exit_reason == "TP":
            trade.mfe_pips = (trade.tp_price - trade.entry_price) * trade.direction / pip_sz
            trade.mfe_usd = trade.mfe_pips * pip_val * trade.position_size_units

        self._trade_log.append(trade)

        # Update daily stats
        self._day.trades_taken += 1
        self._day.gross_pnl += trade.pnl_usd
        if trade.pnl_usd > 0:
            self._day.trades_won += 1

        # Update equity
        self.equity += trade.pnl_usd
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        logger.info(
            f"CLOSE {symbol} @ {exit_price:.5f} | {exit_reason} | "
            f"P&L={trade.pnl_usd:+.2f} USD | Equity={self.equity:.2f}"
        )

        return trade

    # ───────────────────────────────────────────────────────────────────────────
    # Utilities
    # ───────────────────────────────────────────────────────────────────────────

    def _refresh_day(self) -> None:
        today = date.today()
        if self._day.date != today:
            self._day = DayStats(date=today)

    def _current_drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0
        return max(0, (self.peak_equity - self.equity) / self.peak_equity)

    def get_trade_log(self) -> pd.DataFrame:
        """Export all trades as DataFrame."""
        if not self._trade_log:
            return pd.DataFrame()

        records = []
        for t in self._trade_log:
            records.append({
                "entry_ts": t.entry_ts,
                "exit_ts": t.exit_ts,
                "symbol": t.symbol,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "sl_price": t.sl_price,
                "tp_price": t.tp_price,
                "pnl_usd": t.pnl_usd,
                "realized_rr": t.realized_rr,
                "ml_prob": t.ml_prob,
                "mae_pips": t.mae_pips,
                "mfe_pips": t.mfe_pips,
                "days_held": t.days_held,
                "exit_reason": t.exit_reason,
            })
        return pd.DataFrame(records)
