"""
backtest/engine.py
------------------
Event-driven backtesting engine.

Features:
  - Bar-by-bar simulation (no look-ahead)
  - Spread + slippage modelling
  - One trade at a time (simplified; extend for portfolio)
  - Full trade log with entry/exit timestamps and P&L
  - Equity curve generation
  - Performance metrics: CAGR, Sharpe, Sortino, max drawdown, win rate
  - Walk-forward validation framework
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    initial_balance:  float = 10_000.0
    risk_per_trade:   float = 1.0           # % of equity per trade
    spread_pips:      float = 1.5           # spread in pips
    slippage_pips:    float = 0.5           # slippage in pips
    symbol:           str   = "EURUSD"      # used to derive pip_size for friction
    lot_step:         float = 0.01
    min_lot:          float = 0.01
    max_lot:          float = 500.0         # raised for gold (position_units can be large)
    max_trades_per_day: int = 5
    daily_loss_limit_pct: float = 3.0       # halt if day loss > X%


# ──────────────────────────────────────────────────────────────────────────────
# Trade record
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    entry_ts:       pd.Timestamp
    exit_ts:        Optional[pd.Timestamp]
    direction:      int           # +1 long, -1 short
    entry_price:    float
    exit_price:     float = 0.0
    sl_price:       float = 0.0
    tp_price:       float = 0.0
    position_units: float = 0.0   # price-units of exposure (oz for gold, units for forex)
    pnl_price:      float = 0.0   # price-unit P&L (same units as position_units × price move)
    pnl_usd:        float = 0.0
    exit_reason:    str   = ""
    ml_prob:        float = 0.0
    equity_after:   float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────────────────────────

class BacktestEngine:

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.cfg    = config or BacktestConfig()
        self.equity = self.cfg.initial_balance
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[dict]  = []
        self._open_trade: Optional[TradeRecord] = None
        self._day_trades: int  = 0
        self._day_pnl:    float = 0.0
        self._current_day: Optional[pd.Timestamp] = None

    # ──────────────────────────────────────────────────────────────
    # Main simulation loop
    # ──────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame, signals: list) -> "BacktestResults":
        """
        Simulate all signals against the OHLCV data in *df*.
        *signals* is a list of SignalResult objects from the strategy engine.
        """
        # Index signals by timestamp for fast lookup
        sig_map: Dict[pd.Timestamp, Any] = {s.timestamp: s for s in signals}

        logger.info(f"Starting backtest | {len(df):,} bars | {len(signals)} signals")

        for i, (ts, row) in enumerate(df.iterrows()):
            self._refresh_day(ts)
            self._record_equity(ts, row["close"])

            # ── Check if open trade hits TP or SL ───────────────
            if self._open_trade is not None:
                self._check_trade_exit(ts, row)
                if self._open_trade is not None:
                    continue   # still open, skip new entry

            # ── Check for new signal ─────────────────────────────
            if ts in sig_map and self._can_trade():
                sig = sig_map[ts]
                self._open_new_trade(ts, sig, row)

        # Force-close any remaining open trade at last bar
        if self._open_trade is not None:
            last_ts  = df.index[-1]
            last_row = df.iloc[-1]
            self._force_close(last_ts, last_row["close"])

        return BacktestResults(self.trades, self.equity_curve, self.cfg.initial_balance)

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _can_trade(self) -> bool:
        if self._day_trades >= self.cfg.max_trades_per_day:
            return False
        daily_loss_limit = self.equity * self.cfg.daily_loss_limit_pct / 100
        if self._day_pnl <= -daily_loss_limit:
            return False
        return True

    def _open_new_trade(self, ts, sig, row) -> None:
        # Friction: convert pips to price units using the symbol's pip size
        pip_sz    = self._pip_size(self.cfg.symbol)
        friction  = (self.cfg.spread_pips + self.cfg.slippage_pips) * pip_sz
        entry     = sig.entry_price + friction if sig.direction == 1 \
                    else sig.entry_price - friction

        sl_dist = abs(entry - sig.sl_price)
        if sl_dist < 1e-9:
            logger.warning(f"SL distance near-zero at {ts}, skipping.")
            return

        # EV-scaled risk
        rr       = getattr(sig, "rr_ratio", 2.0)
        ml_prob  = sig.ml_probability
        ev       = ml_prob * rr - (1 - ml_prob)
        ev_scale = max(0.25, min(1.5, ev / max(rr - 1, 0.5)))

        # position_units: how many price-units of exposure we hold
        # P&L = position_units × price_change × direction
        # By construction: if price moves sl_dist against us -> loss = risk_usd  ✓
        # if price moves sl_dist × rr in our favour -> profit = risk_usd × rr  ✓
        risk_usd       = self.equity * self.cfg.risk_per_trade / 100 * ev_scale
        position_units = risk_usd / sl_dist

        t = TradeRecord(
            entry_ts       = ts,
            exit_ts        = None,
            direction      = sig.direction,
            entry_price    = entry,
            sl_price       = sig.sl_price,
            tp_price       = sig.tp_price,
            position_units = position_units,
            ml_prob        = ml_prob,
        )
        self._open_trade  = t
        self._day_trades += 1
        logger.debug(
            f"  OPEN {'+' if sig.direction==1 else '-'} @ {entry:.5f} | "
            f"SL={sig.sl_price:.5f} | TP={sig.tp_price:.5f} | "
            f"pos_units={position_units:.4f} | EV={ev:.3f}"
        )

    def _check_trade_exit(self, ts: pd.Timestamp, row) -> None:
        t  = self._open_trade
        h  = row["high"]
        lo = row["low"]

        exit_price  = None
        exit_reason = None

        # [3.1] Worst-case: check SL before TP on same candle
        if t.direction == 1:
            if lo <= t.sl_price:
                exit_price, exit_reason = t.sl_price, "SL"   # SL checked first
            elif h >= t.tp_price:
                exit_price, exit_reason = t.tp_price, "TP"
        else:
            if h >= t.sl_price:
                exit_price, exit_reason = t.sl_price, "SL"
            elif lo <= t.tp_price:
                exit_price, exit_reason = t.tp_price, "TP"

        if exit_price is not None:
            self._close_trade(ts, exit_price, exit_reason)

    def _close_trade(self, ts: pd.Timestamp, exit_price: float, reason: str) -> None:
        t = self._open_trade
        # Direct price arithmetic — instrument agnostic
        price_diff = (exit_price - t.entry_price) * t.direction
        pnl_usd    = t.position_units * price_diff

        t.exit_ts      = ts
        t.exit_price   = exit_price
        t.pnl_price    = round(price_diff, 5)
        t.pnl_usd      = round(pnl_usd, 2)
        t.exit_reason  = reason

        self.equity     += pnl_usd
        self._day_pnl   += pnl_usd
        t.equity_after   = round(self.equity, 2)

        self.trades.append(t)
        self._open_trade = None
        logger.debug(f"  CLOSE {reason} @ {exit_price:.5f} | P&L=${pnl_usd:+.2f}")

    def _force_close(self, ts: pd.Timestamp, price: float) -> None:
        self._close_trade(ts, price, reason="EOD")

    def _record_equity(self, ts: pd.Timestamp, price: float) -> None:
        mark = 0.0
        if self._open_trade is not None:
            t    = self._open_trade
            mark = t.position_units * (price - t.entry_price) * t.direction
        self.equity_curve.append({"timestamp": ts, "equity": self.equity + mark})

    def _refresh_day(self, ts: pd.Timestamp) -> None:
        day = ts.date()
        if self._current_day != day:
            self._current_day = day
            self._day_trades  = 0
            self._day_pnl     = 0.0

    def _calc_lots(self, sl_pips: float, ev_scale: float = 1.0) -> float:
        """Kept for walk-forward compatibility; not used in main backtest."""
        cfg  = self.cfg
        risk = self.equity * cfg.risk_per_trade / 100 * ev_scale
        raw  = risk / max(sl_pips * self._pip_size(cfg.symbol), 1e-9)
        lots = round(raw / cfg.lot_step) * cfg.lot_step
        return float(np.clip(lots, cfg.min_lot, cfg.max_lot))

    @staticmethod
    def _pip_size(symbol: str) -> float:
        """Smallest price increment for the symbol (used only for friction/spread)."""
        sym = symbol.upper()
        if "JPY" in sym:
            return 0.01      # JPY pairs: 2 decimal places
        if "XAU" in sym or "GOLD" in sym:
            return 0.01      # Gold: priced to 2dp
        if "XAG" in sym:
            return 0.001     # Silver
        return 0.0001        # Standard forex (4dp)
# Results analysis
# ──────────────────────────────────────────────────────────────────────────────

class BacktestResults:

    def __init__(
        self,
        trades: List[TradeRecord],
        equity_curve: List[dict],
        initial_balance: float,
    ):
        self.trades          = trades
        self.equity_curve    = pd.DataFrame(equity_curve).set_index("timestamp")
        self.initial_balance = initial_balance

    def metrics(self) -> Dict[str, Any]:
        if not self.trades:
            return {"error": "No trades executed"}

        df = self._trades_df()
        equity = self.equity_curve["equity"]

        total_trades  = len(df)
        winners       = df[df["pnl_usd"] > 0]
        losers        = df[df["pnl_usd"] <= 0]
        win_rate      = len(winners) / total_trades
        avg_win       = winners["pnl_usd"].mean() if len(winners) else 0
        avg_loss      = losers["pnl_usd"].mean()  if len(losers)  else 0
        profit_factor = (
            abs(winners["pnl_usd"].sum()) / abs(losers["pnl_usd"].sum())
            if abs(losers["pnl_usd"].sum()) > 0 else np.inf
        )
        net_pnl       = df["pnl_usd"].sum()
        final_equity  = self.initial_balance + net_pnl

        # Drawdown
        peak    = equity.cummax()
        dd      = (equity - peak) / peak
        max_dd  = float(dd.min())

        # Returns
        ret = equity.pct_change().dropna()
        sharpe  = float(ret.mean() / ret.std() * np.sqrt(252 * 6.5 * 12)) if ret.std() > 0 else 0.0
        down_r  = ret[ret < 0]
        sortino = float(ret.mean() / down_r.std() * np.sqrt(252 * 6.5 * 12)) if down_r.std() > 0 else 0.0

        # CAGR
        n_days = (equity.index[-1] - equity.index[0]).days
        years  = max(n_days / 365.25, 1e-6)
        cagr   = float((final_equity / self.initial_balance) ** (1 / years) - 1)

        return {
            "total_trades":   total_trades,
            "win_rate":       round(win_rate, 4),
            "avg_win_usd":    round(avg_win,  2),
            "avg_loss_usd":   round(avg_loss, 2),
            "profit_factor":  round(profit_factor, 2),
            "net_pnl_usd":    round(net_pnl, 2),
            "initial_equity": self.initial_balance,
            "final_equity":   round(final_equity, 2),
            "max_drawdown":   round(max_dd, 4),
            "sharpe":         round(sharpe,  3),
            "sortino":        round(sortino, 3),
            "cagr":           round(cagr,    4),
        }

    def print_summary(self) -> None:
        m = self.metrics()
        print("\n" + "=" * 55)
        print("  BACKTEST RESULTS")
        print("=" * 55)
        for k, v in m.items():
            if isinstance(v, float):
                if "rate" in k or "cagr" in k or "drawdown" in k:
                    print(f"  {k:<22} {v:>10.2%}")
                else:
                    print(f"  {k:<22} {v:>10.4f}")
            else:
                print(f"  {k:<22} {v:>10}")
        print("=" * 55)

    def save_trades(self, path: str = "logs/backtest_trades.csv") -> None:
        df = self._trades_df()
        df.to_csv(path)
        logger.info(f"Trade log saved -> {path}")

    def save_equity_curve(self, path: str = "logs/equity_curve.csv") -> None:
        self.equity_curve.to_csv(path)
        logger.info(f"Equity curve saved -> {path}")

    def _trades_df(self) -> pd.DataFrame:
        rows = []
        for t in self.trades:
            rows.append({
                "entry_ts":      t.entry_ts,
                "exit_ts":       t.exit_ts,
                "direction":     t.direction,
                "entry_price":   t.entry_price,
                "exit_price":    t.exit_price,
                "sl_price":      t.sl_price,
                "tp_price":      t.tp_price,
                "position_units": t.position_units,
                "pnl_price":     t.pnl_price,
                "pnl_usd":       t.pnl_usd,
                "exit_reason":   t.exit_reason,
                "ml_prob":       t.ml_prob,
                "equity_after":  t.equity_after,
            })
        return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Walk-forward validation
# ──────────────────────────────────────────────────────────────────────────────

class WalkForwardValidator:
    """
    Expanding-window walk-forward validation.
    Each fold trains on all data up to the fold boundary, tests on the next window.
    Uses the model's calibrated threshold for the OOS strategy scan.
    """

    def __init__(
        self,
        n_splits:      int   = 4,
        oos_fraction:  float = 0.15,   # fraction of total data per OOS window
    ):
        self.n_splits     = n_splits
        self.oos_fraction = oos_fraction

    def run(
        self,
        feature_df:     pd.DataFrame,
        raw_df:         pd.DataFrame,
        label_config,
        strategy_config,
        backtest_config,
        model_class,
        model_kwargs: dict = None,
    ) -> List[Dict]:
        from features.labeler import SetupLabeler, get_feature_columns
        from strategy.engine import StrategyEngine, StrategyConfig
        import copy

        model_kwargs = model_kwargs or {}
        n        = len(feature_df)
        oos_size = max(int(n * self.oos_fraction), 500)
        results  = []

        # Minimum train start: leave room for at least 2 OOS windows before end
        min_train_end = n - self.n_splits * oos_size

        for fold in range(self.n_splits):
            oos_start = min_train_end + fold * oos_size
            oos_end   = oos_start + oos_size
            if oos_end > n:
                break

            train_df = feature_df.iloc[:oos_start]
            oos_df   = feature_df.iloc[oos_start:oos_end]
            oos_raw  = raw_df.iloc[oos_start:oos_end]

            logger.info(
                f"Fold {fold+1}/{self.n_splits} | "
                f"Train: {len(train_df):,} | OOS: {len(oos_df):,}"
            )

            if len(train_df) < 200:
                logger.warning(f"Fold {fold+1}: insufficient training data, skipping.")
                continue

            # Label
            labeler = SetupLabeler(label_config)
            labeled = labeler.label(train_df)
            if labeled.empty or len(labeled) < 10:
                logger.warning(f"Fold {fold+1}: too few labeled setups ({len(labeled)}), skipping.")
                continue

            feat_cols = get_feature_columns(labeled)
            X = labeled[feat_cols].fillna(0)
            y = labeled["label"]

            # Train model for this fold
            fold_model = model_class(**model_kwargs)
            fold_model.train(X, y, test_size=0.20, n_cv_splits=3)

            # Use the model's own calibrated threshold for OOS scan
            fold_cfg = copy.copy(strategy_config)
            fold_cfg.ml_threshold = fold_model.threshold

            engine  = StrategyEngine(fold_cfg, model=fold_model)
            signals = engine.scan_all(oos_df)

            if not signals:
                logger.info(f"  Fold {fold+1}: no OOS signals generated.")
                m = {"fold": fold+1, "total_trades": 0, "win_rate": 0,
                     "profit_factor": 0, "max_drawdown": 0, "sharpe": 0,
                     "net_pnl_usd": 0, "error": "no_signals"}
                results.append(m)
                continue

            bt  = BacktestEngine(backtest_config)
            res = bt.run(oos_raw, signals)
            m   = res.metrics()
            m["fold"] = fold + 1
            results.append(m)
            logger.info(
                f"  Fold {fold+1} | Trades={m.get('total_trades',0)} | "
                f"WR={m.get('win_rate',0):.1%} | "
                f"PF={m.get('profit_factor',0):.2f} | "
                f"DD={m.get('max_drawdown',0):.1%}"
            )

        if results:
            logger.info("\n── Walk-Forward Summary ──")
            for r in results:
                logger.info(
                    f"  Fold {r['fold']}: WR={r.get('win_rate',0):.1%} "
                    f"PF={r.get('profit_factor',0):.2f} "
                    f"DD={r.get('max_drawdown',0):.1%}"
                )
        return results