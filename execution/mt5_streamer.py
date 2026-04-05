"""
execution/mt5_streamer.py
-------------------------
Multi-symbol, multi-timeframe live bar streamer via the MetaTrader5 Python API.

Design
------
- Polls MT5 for OHLCV bars at a configurable interval.
- Fires a callback ONLY when a NEW completed bar is available (not on every tick).
- Handles disconnections with exponential back-off reconnect.
- Streams up to N symbols simultaneously in a single thread.
- Maintains a rolling history window per symbol so the feature engine
  always has enough warm-up bars without re-fetching everything.

Usage (standalone)
------------------
    streamer = MT5Streamer(
        symbols    = ["EURUSD", "GBPUSD", "USDJPY"],
        timeframe  = "M5",
        warm_bars  = 300,
        on_new_bar = my_callback,   # fn(symbol, df) -> None
    )
    streamer.start()   # blocks; Ctrl-C to stop

Callback signature
------------------
    def on_new_bar(symbol: str, df: pd.DataFrame) -> None:
        # df: rolling window of the last `warm_bars` bars for this symbol
        # The most recent bar (df.iloc[-1]) is the newly completed bar.
        ...
"""

import logging
import time
import threading
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Try real MT5 import ───────────────────────────────────────────────────────
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 not installed – streamer will run in SIMULATION mode.")

TIMEFRAME_MAP: Dict[str, int] = {}   # filled after MT5 import below

if MT5_AVAILABLE:
    TIMEFRAME_MAP = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
    }

TIMEFRAME_SECONDS = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400,
}


# ─────────────────────────────────────────────────────────────────────────────
# Streamer
# ─────────────────────────────────────────────────────────────────────────────

class MT5Streamer:
    """
    Polls MT5 for completed OHLCV bars across multiple symbols.
    Fires `on_new_bar(symbol, df)` each time a new bar closes.
    """

    RECONNECT_DELAYS = [5, 10, 30, 60, 120]   # seconds, exponential back-off

    def __init__(
        self,
        symbols:    List[str],
        timeframe:  str = "M5",
        warm_bars:  int = 300,
        poll_secs:  float = 10.0,
        on_new_bar: Optional[Callable] = None,
        login:      int  = 0,
        password:   str  = "",
        server:     str  = "",
    ):
        self.symbols    = [s.upper() for s in symbols]
        self.timeframe  = timeframe.upper()
        self.warm_bars  = warm_bars
        self.poll_secs  = poll_secs
        self.on_new_bar = on_new_bar or (lambda sym, df: None)
        self._login     = login
        self._password  = password
        self._server    = server

        self._stop_event = threading.Event()
        self._connected  = False

        # Per-symbol rolling history: symbol → DataFrame
        self._history: Dict[str, pd.DataFrame] = {}
        # Last seen bar timestamp per symbol
        self._last_bar_ts: Dict[str, Optional[pd.Timestamp]] = {s: None for s in symbols}

    # ─── Public control ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Block the calling thread. Use start_async() for background operation."""
        logger.info(
            f"MT5Streamer starting | symbols={self.symbols} | "
            f"TF={self.timeframe} | warm={self.warm_bars} bars"
        )
        self._connect_with_retry()
        self._initialise_history()
        self._run_loop()

    def start_async(self) -> threading.Thread:
        """Start streamer in a background daemon thread."""
        t = threading.Thread(target=self.start, daemon=True, name="MT5Streamer")
        t.start()
        return t

    def stop(self) -> None:
        self._stop_event.set()
        if MT5_AVAILABLE and self._connected:
            mt5.shutdown()
        logger.info("MT5Streamer stopped.")

    # ─── Connection ───────────────────────────────────────────────────────────

    def _connect_with_retry(self) -> None:
        for attempt, delay in enumerate(self.RECONNECT_DELAYS + [None]):
            if self._connect():
                return
            if delay is None:
                raise RuntimeError("MT5: exhausted all reconnect attempts.")
            logger.warning(f"MT5 connect failed (attempt {attempt+1}). Retrying in {delay}s…")
            time.sleep(delay)

    def _connect(self) -> bool:
        if not MT5_AVAILABLE:
            logger.info("[SIM] MT5Streamer connected in simulation mode.")
            self._connected = True
            return True

        ok = mt5.initialize(
            login    = self._login    or None,
            password = self._password or None,
            server   = self._server   or None,
        )
        if not ok:
            logger.error(f"mt5.initialize() failed: {mt5.last_error()}")
            return False

        info = mt5.account_info()
        logger.info(
            f"MT5 connected | Account {info.login} | "
            f"Balance {info.balance:.2f} {info.currency} | Server {info.server}"
        )
        # Enable all symbols
        for sym in self.symbols:
            if not mt5.symbol_select(sym, True):
                logger.warning(f"Symbol {sym} could not be selected in MT5 Market Watch.")

        self._connected = True
        return True

    def _ensure_connected(self) -> bool:
        if not MT5_AVAILABLE:
            return True
        if mt5.terminal_info() is None:
            logger.warning("MT5 connection lost – reconnecting…")
            self._connected = False
            self._connect_with_retry()
        return self._connected

    # ─── History bootstrap ────────────────────────────────────────────────────

    def _initialise_history(self) -> None:
        """Fetch warm-up bars for every symbol before entering the poll loop."""
        for sym in self.symbols:
            df = self._fetch_bars(sym, self.warm_bars + 2)
            if df is not None and not df.empty:
                self._history[sym]     = df
                self._last_bar_ts[sym] = df.index[-1]
                logger.info(f"  {sym}: loaded {len(df)} warm-up bars "
                            f"(last: {df.index[-1]})")
            else:
                logger.warning(f"  {sym}: could not fetch warm-up bars.")

    # ─── Main poll loop ───────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        logger.info("Entering poll loop…")
        reconnect_idx = 0

        while not self._stop_event.is_set():
            if not self._ensure_connected():
                delay = self.RECONNECT_DELAYS[min(reconnect_idx, len(self.RECONNECT_DELAYS)-1)]
                time.sleep(delay)
                reconnect_idx += 1
                continue
            reconnect_idx = 0

            for sym in self.symbols:
                try:
                    self._poll_symbol(sym)
                except Exception as exc:
                    logger.error(f"Error polling {sym}: {exc}", exc_info=True)

            time.sleep(self.poll_secs)

    def _poll_symbol(self, symbol: str) -> None:
        """Fetch latest bars, detect new completed bar, fire callback."""
        df = self._fetch_bars(symbol, 5)   # only need the last few bars
        if df is None or df.empty:
            return

        latest_ts = df.index[-1]

        # A new bar has closed if its timestamp is newer than what we've seen
        if self._last_bar_ts[symbol] is None or latest_ts > self._last_bar_ts[symbol]:
            logger.info(f"New bar: {symbol} @ {latest_ts}")
            self._last_bar_ts[symbol] = latest_ts

            # Append new bar(s) to rolling history
            hist = self._history.get(symbol, pd.DataFrame())
            if not hist.empty:
                # Only append rows newer than what we have
                new_rows = df[df.index > hist.index[-1]]
                if not new_rows.empty:
                    hist = pd.concat([hist, new_rows])
            else:
                hist = df

            # Keep only the last `warm_bars` rows to bound memory
            hist = hist.iloc[-self.warm_bars:]
            self._history[symbol] = hist

            # Fire callback with the rolling window
            try:
                self.on_new_bar(symbol, hist.copy())
            except Exception as exc:
                logger.error(f"on_new_bar callback error for {symbol}: {exc}", exc_info=True)

    # ─── MT5 bar fetching ─────────────────────────────────────────────────────

    def _fetch_bars(self, symbol: str, n: int) -> Optional[pd.DataFrame]:
        if not MT5_AVAILABLE:
            return self._sim_bars(symbol, n)

        tf_code = TIMEFRAME_MAP.get(self.timeframe)
        if tf_code is None:
            raise ValueError(f"Unknown timeframe: {self.timeframe}")

        rates = mt5.copy_rates_from_pos(symbol, tf_code, 0, n)
        if rates is None or len(rates) == 0:
            logger.debug(f"No rates returned for {symbol}: {mt5.last_error()}")
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)

        # Drop the currently-forming (incomplete) bar — it's the last one
        # if its timestamp is within the current bar period
        tf_secs  = TIMEFRAME_SECONDS[self.timeframe]
        now_utc  = pd.Timestamp.utcnow().tz_localize("UTC") if pd.Timestamp.utcnow().tzinfo is None \
                   else pd.Timestamp.utcnow()
        last_bar_age = (now_utc - df.index[-1]).total_seconds()
        if last_bar_age < tf_secs:
            df = df.iloc[:-1]   # drop incomplete bar

        if df.empty:
            return None

        return df[["open", "high", "low", "close", "volume"]].copy()

    # ─── Simulation stub ─────────────────────────────────────────────────────

    def _sim_bars(self, symbol: str, n: int) -> pd.DataFrame:
        """Generate fake bars for testing without MT5."""
        from data.generate_sample import generate_ohlcv
        import hashlib

        seed  = int(hashlib.md5(symbol.encode()).hexdigest(), 16) % (2**32)
        tf_m  = TIMEFRAME_SECONDS.get(self.timeframe, 300) // 60
        df    = generate_ohlcv(n_bars=max(n, 50), timeframe_min=tf_m, seed=seed)
        df.set_index("timestamp", inplace=True)
        # Shift so last bar ends "now"
        now = pd.Timestamp.utcnow().replace(second=0, microsecond=0).tz_localize("UTC")
        shift = now - df.index[-1]
        df.index = df.index + shift
        return df[["open", "high", "low", "close", "volume"]].tail(n)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: single-symbol snapshot (no callback, just returns DataFrame)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_live_bars(
    symbol:    str,
    timeframe: str = "M5",
    n_bars:    int = 300,
    login:     int = 0,
    password:  str = "",
    server:    str = "",
) -> pd.DataFrame:
    """
    One-shot fetch of the last *n_bars* completed bars for *symbol*.
    Useful for bootstrapping the live trader or quick checks.
    """
    streamer = MT5Streamer(
        symbols   = [symbol],
        timeframe = timeframe,
        warm_bars = n_bars,
        login     = login,
        password  = password,
        server    = server,
    )
    streamer._connect_with_retry()
    df = streamer._fetch_bars(symbol, n_bars + 2)
    if MT5_AVAILABLE:
        mt5.shutdown()
    return df if df is not None else pd.DataFrame()
