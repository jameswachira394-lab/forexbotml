"""
execution/mt5_broker.py
-----------------------
MetaTrader 5 live execution layer.

Wraps the MetaTrader5 Python API to provide:
  - Connection management with auto-reconnect
  - Market order placement with SL/TP
  - Position monitoring and modification
  - Symbol info (pip size, lot constraints)
  - Error handling and retry logic

NOTE: MetaTrader5 must be installed:  pip install MetaTrader5
      This module gracefully degrades to a simulation stub if MT5 is unavailable
      (useful for backtest / CI environments).
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# ── Try to import the real MT5 library ────────────────────────────────────────
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning(
        "MetaTrader5 library not found. "
        "ExecutionBroker will run in SIMULATION mode."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OrderResult:
    success:    bool
    ticket:     int     = 0
    order_type: str     = ""
    volume:     float   = 0.0
    price:      float   = 0.0
    sl:         float   = 0.0
    tp:         float   = 0.0
    comment:    str     = ""
    error_code: int     = 0
    error_msg:  str     = ""


@dataclass
class PositionInfo:
    ticket:     int
    symbol:     str
    direction:  int     # +1 long, -1 short
    volume:     float
    open_price: float
    sl:         float
    tp:         float
    profit:     float


# ──────────────────────────────────────────────────────────────────────────────
# Broker class
# ──────────────────────────────────────────────────────────────────────────────

class MT5Broker:
    """
    Thin wrapper around MetaTrader5 Python API.
    Falls back to simulation stub when MT5 is not installed.
    """

    MAX_RETRIES       = 3
    RETRY_DELAY_SEC   = 2.0
    MAGIC_NUMBER      = 20250101   # unique ID for this bot's orders

    def __init__(self, login: int = 0, password: str = "", server: str = ""):
        self._login    = login
        self._password = password
        self._server   = server
        self._connected = False
        self._sim_ticket_counter = 1000
        self._sim_positions: Dict[int, PositionInfo] = {}

    # ──────────────────────────────────────────────────────────────
    # Connection
    # ──────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        if not MT5_AVAILABLE:
            logger.info("[SIM] MT5 broker connected in simulation mode.")
            self._connected = True
            return True

        for attempt in range(1, self.MAX_RETRIES + 1):
            if mt5.initialize(
                login=self._login,
                password=self._password,
                server=self._server,
            ):
                account = mt5.account_info()
                logger.info(
                    f"MT5 connected | Account: {account.login} | "
                    f"Balance: {account.balance:.2f} | Server: {account.server}"
                )
                self._connected = True
                return True
            else:
                err = mt5.last_error()
                logger.warning(f"MT5 init attempt {attempt}/{self.MAX_RETRIES} failed: {err}")
                time.sleep(self.RETRY_DELAY_SEC)

        logger.error("Failed to connect to MT5 after all retries.")
        return False

    def disconnect(self) -> None:
        if MT5_AVAILABLE and self._connected:
            mt5.shutdown()
        self._connected = False
        logger.info("MT5 disconnected.")

    def ensure_connected(self) -> bool:
        """Check connection and reconnect if needed."""
        if not self._connected:
            return self.connect()
        if MT5_AVAILABLE:
            if mt5.terminal_info() is None:
                logger.warning("MT5 connection lost. Reconnecting…")
                return self.connect()
        return True

    # ──────────────────────────────────────────────────────────────
    # Order placement
    # ──────────────────────────────────────────────────────────────

    def place_market_order(
        self,
        symbol:    str,
        direction: int,       # +1 BUY, -1 SELL
        volume:    float,
        sl:        float,
        tp:        float,
        comment:   str = "ForexBot",
    ) -> OrderResult:
        if not self.ensure_connected():
            return OrderResult(success=False, error_msg="Not connected")

        if not MT5_AVAILABLE:
            return self._sim_place_order(symbol, direction, volume, sl, tp, comment)

        # Validate symbol
        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            return OrderResult(success=False, error_msg=f"Symbol {symbol} not found")
        if not sym_info.visible:
            mt5.symbol_select(symbol, True)

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(success=False, error_msg="Could not fetch tick data")

        order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
        price      = tick.ask if direction == 1 else tick.bid

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       float(volume),
            "type":         order_type,
            "price":        price,
            "sl":           float(sl),
            "tp":           float(tp),
            "deviation":    10,          # max price deviation in points
            "magic":        self.MAGIC_NUMBER,
            "comment":      comment,
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err_msg = f"Order failed: retcode={getattr(result,'retcode',None)}"
            logger.error(err_msg)
            return OrderResult(
                success=False,
                error_code=getattr(result, "retcode", -1),
                error_msg=err_msg,
            )

        logger.info(
            f"Order placed | Ticket={result.order} | "
            f"{'BUY' if direction==1 else 'SELL'} {volume} {symbol} @ {price:.5f} | "
            f"SL={sl:.5f} | TP={tp:.5f}"
        )
        return OrderResult(
            success    = True,
            ticket     = result.order,
            order_type = "BUY" if direction == 1 else "SELL",
            volume     = volume,
            price      = price,
            sl         = sl,
            tp         = tp,
        )

    # ──────────────────────────────────────────────────────────────
    # Position management
    # ──────────────────────────────────────────────────────────────

    def get_open_positions(self, symbol: Optional[str] = None) -> List[PositionInfo]:
        if not MT5_AVAILABLE:
            return list(self._sim_positions.values())

        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if positions is None:
            return []

        result = []
        for p in positions:
            if p.magic != self.MAGIC_NUMBER:
                continue
            result.append(PositionInfo(
                ticket     = p.ticket,
                symbol     = p.symbol,
                direction  = 1 if p.type == 0 else -1,
                volume     = p.volume,
                open_price = p.price_open,
                sl         = p.sl,
                tp         = p.tp,
                profit     = p.profit,
            ))
        return result

    def modify_sl_tp(
        self,
        ticket: int,
        new_sl: float,
        new_tp: float,
    ) -> bool:
        if not MT5_AVAILABLE:
            if ticket in self._sim_positions:
                self._sim_positions[ticket].sl = new_sl
                self._sim_positions[ticket].tp = new_tp
            return True

        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl":       new_sl,
            "tp":       new_tp,
        }
        result = mt5.order_send(request)
        ok = (result is not None and result.retcode == mt5.TRADE_RETCODE_DONE)
        if not ok:
            logger.error(f"Modify SL/TP failed for ticket {ticket}")
        return ok

    def close_position(self, ticket: int, symbol: str, volume: float, direction: int) -> bool:
        if not MT5_AVAILABLE:
            self._sim_positions.pop(ticket, None)
            logger.info(f"[SIM] Closed position ticket={ticket}")
            return True

        close_type = mt5.ORDER_TYPE_SELL if direction == 1 else mt5.ORDER_TYPE_BUY
        tick       = mt5.symbol_info_tick(symbol)
        price      = tick.bid if direction == 1 else tick.ask

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       volume,
            "type":         close_type,
            "position":     ticket,
            "price":        price,
            "deviation":    10,
            "magic":        self.MAGIC_NUMBER,
            "comment":      "ForexBot close",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        ok = (result is not None and result.retcode == mt5.TRADE_RETCODE_DONE)
        if ok:
            logger.info(f"Position {ticket} closed.")
        else:
            logger.error(f"Failed to close position {ticket}: retcode={getattr(result,'retcode',None)}")
        return ok

    # ──────────────────────────────────────────────────────────────
    # Account info
    # ──────────────────────────────────────────────────────────────

    def get_account_balance(self) -> float:
        if not MT5_AVAILABLE:
            return 10_000.0
        info = mt5.account_info()
        return info.balance if info else 0.0

    def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        if not MT5_AVAILABLE:
            return {"bid": 1.10000, "ask": 1.10015}
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return {"bid": tick.bid, "ask": tick.ask, "time": tick.time}

    # ──────────────────────────────────────────────────────────────
    # Simulation stubs (used when MT5 not available)
    # ──────────────────────────────────────────────────────────────

    def _sim_place_order(self, symbol, direction, volume, sl, tp, comment) -> OrderResult:
        ticket = self._sim_ticket_counter
        self._sim_ticket_counter += 1
        price  = 1.10000   # dummy price
        pos = PositionInfo(
            ticket=ticket, symbol=symbol, direction=direction,
            volume=volume, open_price=price, sl=sl, tp=tp, profit=0.0
        )
        self._sim_positions[ticket] = pos
        logger.info(
            f"[SIM] Order | {'BUY' if direction==1 else 'SELL'} {volume} {symbol} | "
            f"SL={sl:.5f} | TP={tp:.5f} | ticket={ticket}"
        )
        return OrderResult(
            success=True, ticket=ticket,
            order_type="BUY" if direction==1 else "SELL",
            volume=volume, price=price, sl=sl, tp=tp,
        )
