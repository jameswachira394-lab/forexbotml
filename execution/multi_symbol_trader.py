"""
execution/multi_symbol_trader.py
---------------------------------
Multi-symbol live trading engine.

Architecture
------------
1. MT5Streamer fires on_new_bar(symbol, df) in a background thread
   each time a bar closes for any of the configured symbols.

2. MultiSymbolTrader.on_new_bar():
   a. Runs feature engineering on the rolling window
   b. Evaluates the strategy engine for a signal
   c. Passes through risk management (global + per-symbol limits)
   d. Executes the trade via MT5Broker
   e. Monitors open positions for TP/SL hits

3. All state is protected by a threading.Lock so no race conditions
   occur when multiple symbols fire callbacks simultaneously.

Usage
-----
    trader = MultiSymbolTrader.from_config()   # reads config.py
    trader.run()                               # blocks; Ctrl-C to stop
"""

import logging
import threading
import time
from typing import Dict, Optional

import pandas as pd

from execution.mt5_streamer import MT5Streamer
from execution.mt5_broker   import MT5Broker, OrderResult
from execution.logger       import TradeLogger
from features.engineer      import engineer_features
from models.ml_model        import ForexMLModel
from strategy.engine        import StrategyEngine, StrategyConfig, SignalResult
from risk.manager           import RiskManager, RiskConfig
import config as cfg

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Per-symbol state container
# ─────────────────────────────────────────────────────────────────────────────

class SymbolState:
    """Holds open-trade state for one symbol."""
    def __init__(self, symbol: str):
        self.symbol          = symbol
        self.open_ticket:    Optional[int]         = None
        self.open_signal:    Optional[SignalResult] = None
        self.bars_since_entry: int                 = 0


# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------

class MultiSymbolTrader:
    """
    Runs the full signal->risk->execution pipeline for multiple FX pairs.
    One MT5Streamer feeds bar events; this class handles them thread-safely.
    """

    def __init__(
        self,
        symbols:         list,
        model:           Optional[ForexMLModel],
        broker:          MT5Broker,
        strategy_config: StrategyConfig,
        risk_config:     RiskConfig,
        timeframe:       str   = "M5",
        warm_bars:       int   = 300,
        poll_secs:       float = 15.0,
    ):
        self.symbols  = [s.upper() for s in symbols]
        self.model    = model
        self.broker   = broker
        self.tf       = timeframe
        self.warm_bars = warm_bars

        # One strategy engine (shared, stateless per-bar)
        self.engine    = StrategyEngine(strategy_config, model=model)
        self.risk      = RiskManager(risk_config)
        self.trade_log = TradeLogger(path="logs/live_trades.csv")

        # Per-symbol state
        self._states: Dict[str, SymbolState] = {
            s: SymbolState(s) for s in self.symbols
        }

        # Global lock for thread safety across symbol callbacks
        self._lock = threading.Lock()

        # Streamer (created in run())
        self._streamer: Optional[MT5Streamer] = None
        self._poll_secs = poll_secs

    # ─── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, model: Optional[ForexMLModel] = None) -> "MultiSymbolTrader":
        """Build instance from config.py settings."""
        broker = MT5Broker(
            login    = cfg.MT5_LOGIN,
            password = cfg.MT5_PASSWORD,
            server   = cfg.MT5_SERVER,
        )
        s_cfg = StrategyConfig(
            ml_threshold      = cfg.ML_THRESHOLD,
            sl_atr_mult       = cfg.SL_BUFFER_ATR,
            rr_ratio          = cfg.RR_MIN,
            require_htf_align = cfg.REQUIRE_HTF_ALIGN,
        )
        r_cfg = RiskConfig(
            account_balance      = cfg.INITIAL_BALANCE,
            risk_per_trade_pct   = cfg.RISK_PER_TRADE_PCT,
            max_trades_per_day   = cfg.MAX_TRADES_PER_DAY,
            daily_loss_limit_pct = cfg.DAILY_LOSS_LIMIT_PCT,
            max_open_positions   = cfg.MAX_OPEN_POSITIONS,
        )
        return cls(
            symbols         = cfg.SYMBOLS,
            model           = model,
            broker          = broker,
            strategy_config = s_cfg,
            risk_config     = r_cfg,
            timeframe       = cfg.BASE_TF,
            warm_bars       = cfg.LIVE_WARM_BARS,
            poll_secs       = cfg.LIVE_POLL_SECONDS,
        )

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Connect broker, start streamer, block until Ctrl-C."""
        logger.info(f"MultiSymbolTrader starting | symbols={self.symbols} | TF={self.tf}")
        logger.warning("LIVE MODE – real orders will be placed if MT5 is connected.")

        if not self.broker.connect():
            logger.error("Broker connection failed. Aborting.")
            return

        self._streamer = MT5Streamer(
            symbols    = self.symbols,
            timeframe  = self.tf,
            warm_bars  = self.warm_bars,
            poll_secs  = self._poll_secs,
            on_new_bar = self._on_new_bar,
            login      = cfg.MT5_LOGIN,
            password   = cfg.MT5_PASSWORD,
            server     = cfg.MT5_SERVER,
        )

        try:
            self._streamer.start()   # blocks
        except KeyboardInterrupt:
            logger.info("Stopped by user.")
        finally:
            self._streamer.stop()
            self.broker.disconnect()

    # ─── Bar callback (called from streamer thread) ───────────────────────────

    def _on_new_bar(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Called by MT5Streamer each time a new bar closes for *symbol*.
        df: rolling window of completed bars (warm_bars deep).
        """
        with self._lock:
            state = self._states[symbol]

            # Step 1: monitor any open position for this symbol
            if state.open_ticket is not None:
                self._monitor_position(state, df)
                if state.open_ticket is not None:
                    return   # still in trade; skip new entry check

            # Step 2: feature engineering
            try:
                feat_df = engineer_features(df)
            except Exception as exc:
                logger.error(f"[{symbol}] Feature engineering failed: {exc}")
                return

            if len(feat_df) < 50:
                return  # not enough warm-up data yet

            # Step 3: strategy evaluation on the last bar
            idx = len(feat_df) - 1
            sig = self.engine.evaluate_bar(feat_df, idx)
            if sig is None:
                return

            logger.info(
                f"[{symbol}] Signal: {'LONG' if sig.direction==1 else 'SHORT'} "
                f"| p={sig.ml_probability:.2f} | {sig.rule_reason}"
            )

            # Step 4: risk gate
            if not self.risk.approve_trade(
                entry_price = sig.entry_price,
                sl_price    = sig.sl_price,
                tp_price    = sig.tp_price,
                direction   = sig.direction,
                symbol      = symbol,
            ):
                return

            lot_size = self.risk.calculate_lot_size(
                sig.entry_price, sig.sl_price, symbol
            )

            # Step 5: execute
            result = self.broker.place_market_order(
                symbol    = symbol,
                direction = sig.direction,
                volume    = lot_size,
                sl        = sig.sl_price,
                tp        = sig.tp_price,
                comment   = f"MST p={sig.ml_probability:.2f}",
            )

            if result.success:
                state.open_ticket = result.ticket
                state.open_signal = sig
                state.bars_since_entry = 0
                self.risk.record_trade_open()

                self.trade_log.log_open(
                    symbol      = symbol,
                    direction   = sig.direction,
                    entry_price = result.price,
                    sl_price    = sig.sl_price,
                    tp_price    = sig.tp_price,
                    lot_size    = lot_size,
                    ml_prob     = sig.ml_probability,
                    rule_reason = sig.rule_reason,
                )
            else:
                logger.error(f"[{symbol}] Order failed: {result.error_msg}")

    # ─── Position monitor ─────────────────────────────────────────────────────

    def _monitor_position(self, state: SymbolState, df: pd.DataFrame) -> None:
        """Check if the open position is still active in MT5."""
        positions    = self.broker.get_open_positions(state.symbol)
        open_tickets = {p.ticket for p in positions}

        state.bars_since_entry += 1

        if state.open_ticket not in open_tickets:
            # Position closed by MT5 (TP or SL hit)
            sig = state.open_signal
            price_info = self.broker.get_current_price(state.symbol)
            if price_info:
                exit_px = price_info["bid"] if sig.direction == 1 else price_info["ask"]
            else:
                exit_px = df.iloc[-1]["close"]

            pip_size = 0.01 if "JPY" in state.symbol else 0.0001
            pnl_pips = (exit_px - sig.entry_price) * sig.direction / pip_size
            pnl_usd  = pnl_pips * cfg.PIP_VALUE * self.risk.calculate_lot_size(
                sig.entry_price, sig.sl_price, state.symbol
            )
            won = pnl_usd > 0

            self.risk.record_trade_close(pnl_usd, won)
            self.trade_log.log_close(
                symbol       = state.symbol,
                direction    = sig.direction,
                entry_price  = sig.entry_price,
                exit_price   = exit_px,
                exit_reason  = "TP" if won else "SL",
                pnl_usd      = pnl_usd,
                equity_after = self.risk.equity,
                ml_prob      = sig.ml_probability,
            )
            state.open_ticket = None
            state.open_signal = None
