"""
execution/live_trader.py
------------------------
Live trading loop.

Polls MT5 for new bars, runs feature engineering on the latest window,
evaluates strategy, and executes orders when signals fire.

Design:
  - Stateless per bar (all state reconstructed from rolling window)
  - Graceful shutdown on KeyboardInterrupt
  - All decisions logged
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from execution.mt5_broker import MT5Broker
from execution.logger import TradeLogger
from features.engineer import engineer_features
from strategy.engine import StrategyEngine, StrategyConfig
from risk.manager import RiskManager, RiskConfig

logger = logging.getLogger(__name__)

# Number of historical bars to keep in the rolling window for feature calculation
WARM_UP_BARS = 200


class LiveTrader:
    """
    Bar-event live trading loop.

    Parameters
    ----------
    broker          : MT5Broker instance (real or sim)
    model           : trained ForexMLModel (or None to skip ML filter)
    strategy_config : StrategyConfig
    risk_config     : RiskConfig
    symbol          : FX symbol to trade (e.g. "EURUSD")
    timeframe       : MT5 timeframe string e.g. "M5"
    poll_seconds    : seconds between bar checks (set ~= bar duration)
    """

    def __init__(
        self,
        broker:           MT5Broker,
        model             = None,
        strategy_config:  Optional[StrategyConfig] = None,
        risk_config:      Optional[RiskConfig]     = None,
        symbol:           str   = "EURUSD",
        timeframe:        str   = "M5",
        poll_seconds:     float = 30.0,
    ):
        self.broker    = broker
        self.model     = model
        self.engine    = StrategyEngine(strategy_config, model=model)
        self.risk      = RiskManager(risk_config)
        self.trade_log = TradeLogger()
        self.symbol    = symbol
        self.tf        = timeframe
        self.poll_sec  = poll_seconds

        self._last_bar_time: Optional[pd.Timestamp] = None
        self._open_ticket:   Optional[int]           = None
        self._open_signal    = None    # SignalResult of open trade

    # ──────────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        logger.info(f"LiveTrader starting | {self.symbol} | {self.tf}")

        if not self.broker.connect():
            logger.error("Could not connect to broker. Aborting.")
            return

        try:
            while True:
                self._tick()
                time.sleep(self.poll_sec)
        except KeyboardInterrupt:
            logger.info("LiveTrader stopped by user (KeyboardInterrupt).")
        finally:
            self.broker.disconnect()

    # ──────────────────────────────────────────────────────────────
    # Single poll cycle
    # ──────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        # Fetch recent bars from MT5
        raw_df = self._fetch_bars(WARM_UP_BARS + 50)
        if raw_df is None or len(raw_df) < WARM_UP_BARS:
            logger.warning("Insufficient data received from broker.")
            return

        latest_ts = raw_df.index[-1]
        if latest_ts == self._last_bar_time:
            return   # no new bar yet

        self._last_bar_time = latest_ts
        logger.info(f"New bar: {latest_ts}")

        # Feature engineering on rolling window
        try:
            feat_df = engineer_features(raw_df)
        except Exception as exc:
            logger.error(f"Feature engineering failed: {exc}")
            return

        if feat_df.empty:
            return

        # ── Check if open position has hit TP/SL ──────────────────
        if self._open_ticket is not None:
            self._monitor_open_position()
            if self._open_ticket is not None:
                return   # still in trade, skip new entry

        # ── Evaluate latest bar for a new signal ──────────────────
        idx = len(feat_df) - 1
        sig = self.engine.evaluate_bar(feat_df, idx)

        if sig is None:
            logger.debug("No signal this bar.")
            return

        logger.info(
            f"Signal detected: {'LONG' if sig.direction==1 else 'SHORT'} "
            f"p={sig.ml_probability:.2f} reason={sig.rule_reason}"
        )

        # ── Risk gate ─────────────────────────────────────────────
        if not self.risk.approve_trade(
            entry_price = sig.entry_price,
            sl_price    = sig.sl_price,
            tp_price    = sig.tp_price,
            direction   = sig.direction,
            symbol      = self.symbol,
        ):
            return

        lot_size = self.risk.calculate_lot_size(
            entry_price = sig.entry_price,
            sl_price    = sig.sl_price,
            tp_price    = sig.tp_price,
            symbol      = self.symbol,
            ml_prob     = sig.ml_probability,
            rr_ratio    = getattr(sig, 'rr_ratio', 3.0),
        )

        # ── Execute ───────────────────────────────────────────────
        result = self.broker.place_market_order(
            symbol    = self.symbol,
            direction = sig.direction,
            volume    = lot_size,
            sl        = sig.sl_price,
            tp        = sig.tp_price,
            comment   = f"ForexBot p={sig.ml_probability:.2f}",
        )

        if result.success:
            self._open_ticket = result.ticket
            self._open_signal = sig
            self.risk.record_trade_open()
            self.trade_log.log_open(
                symbol      = self.symbol,
                direction   = sig.direction,
                entry_price = result.price,
                sl_price    = sig.sl_price,
                tp_price    = sig.tp_price,
                lot_size    = lot_size,
                ml_prob     = sig.ml_probability,
                rule_reason = sig.rule_reason,
            )
        else:
            logger.error(f"Order placement failed: {result.error_msg}")

    # ──────────────────────────────────────────────────────────────
    # Open position monitor
    # ──────────────────────────────────────────────────────────────

    def _monitor_open_position(self) -> None:
        """Check if the open position has been closed by MT5 (TP/SL hit)."""
        positions = self.broker.get_open_positions(self.symbol)
        tickets   = {p.ticket for p in positions}

        if self._open_ticket not in tickets:
            # Position no longer open → closed externally (TP/SL/manual)
            price_info = self.broker.get_current_price(self.symbol)
            exit_price = price_info["bid"] if self._open_signal.direction == 1 else price_info["ask"]

            # Estimate P&L from entry (approximate; actual from broker statement)
            pip_size = 0.0001
            pip_val  = self.risk.cfg.pip_value
            lot      = self.risk.calculate_lot_size(
                self._open_signal.entry_price, self._open_signal.sl_price, self.symbol
            )
            pnl_pips = (exit_price - self._open_signal.entry_price) * self._open_signal.direction / pip_size
            pnl_usd  = pnl_pips * pip_val * lot
            won      = pnl_usd > 0

            self.risk.record_trade_close(pnl_usd, won)
            self.trade_log.log_close(
                symbol       = self.symbol,
                direction    = self._open_signal.direction,
                entry_price  = self._open_signal.entry_price,
                exit_price   = exit_price,
                exit_reason  = "TP" if won else "SL",
                pnl_usd      = pnl_usd,
                equity_after = self.risk.equity,
                ml_prob      = self._open_signal.ml_probability,
            )

            self._open_ticket = None
            self._open_signal = None

    # ──────────────────────────────────────────────────────────────
    # Bar fetching
    # ──────────────────────────────────────────────────────────────

    def _fetch_bars(self, n: int) -> Optional[pd.DataFrame]:
        """Fetch *n* latest OHLCV bars from MT5."""
        try:
            import MetaTrader5 as mt5
            tf_map = {
                "M1": mt5.TIMEFRAME_M1,  "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,  "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1,
            }
            tf_code = tf_map.get(self.tf.upper(), mt5.TIMEFRAME_M5)
            rates   = mt5.copy_rates_from_pos(self.symbol, tf_code, 0, n)
            if rates is None:
                return None
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df.rename(columns={"time": "timestamp", "tick_volume": "volume"}, inplace=True)
            df.set_index("timestamp", inplace=True)
            return df[["open", "high", "low", "close", "volume"]]
        except ImportError:
            # MT5 not available – return None (caller handles gracefully)
            return None