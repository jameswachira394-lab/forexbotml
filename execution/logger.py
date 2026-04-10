"""
execution/logger.py
-------------------
Structured trade logger and CLI dashboard.

Writes:
  - Trade log (CSV)
  - System log (rotating file handler)
  - Real-time CLI summary after each trade
"""

import csv
import logging
import logging.handlers
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# System logger configuration
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO", log_file: str = "logs/system.log") -> None:
    """
    Configure root logger with:
      - Console handler (INFO)
      - Rotating file handler (DEBUG, 5 MB × 5 backups)
    """
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console — force UTF-8 on Windows to prevent cp1252 UnicodeEncodeError
    import sys
    if sys.platform == "win32":
        import io
        stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    else:
        stream = sys.stdout
    ch = logging.StreamHandler(stream)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File (rotating) — always UTF-8 regardless of OS locale
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)


# ──────────────────────────────────────────────────────────────────────────────
# Trade logger (CSV)
# ──────────────────────────────────────────────────────────────────────────────

TRADE_LOG_FIELDS = [
    "ts_logged", "symbol", "direction", "entry_price",
    "sl_price",  "tp_price", "lot_size",
    "exit_price", "exit_reason", "pnl_usd", "equity_after",
    "ml_probability", "rule_reason",
]


class TradeLogger:
    """Append-only CSV trade log with an auto-created header."""

    def __init__(self, path: str = "logs/live_trades.csv"):
        self.path = Path(path)
        self._init_file()

    def _init_file(self) -> None:
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=TRADE_LOG_FIELDS)
                writer.writeheader()

    def log_open(
        self,
        symbol:       str,
        direction:    int,
        entry_price:  float,
        sl_price:     float,
        tp_price:     float,
        lot_size:     float,
        ml_prob:      float,
        rule_reason:  str,
    ) -> None:
        row = {
            "ts_logged":    _now(),
            "symbol":       symbol,
            "direction":    "LONG" if direction == 1 else "SHORT",
            "entry_price":  entry_price,
            "sl_price":     sl_price,
            "tp_price":     tp_price,
            "lot_size":     lot_size,
            "exit_price":   "",
            "exit_reason":  "OPEN",
            "pnl_usd":      "",
            "equity_after": "",
            "ml_probability": round(ml_prob, 4),
            "rule_reason":  rule_reason,
        }
        self._append(row)
        self._print_open(row)

    def log_close(
        self,
        symbol:       str,
        direction:    int,
        entry_price:  float,
        exit_price:   float,
        exit_reason:  str,
        pnl_usd:      float,
        equity_after: float,
        ml_prob:      float   = 0.0,
        rule_reason:  str     = "",
    ) -> None:
        row = {
            "ts_logged":    _now(),
            "symbol":       symbol,
            "direction":    "LONG" if direction == 1 else "SHORT",
            "entry_price":  entry_price,
            "sl_price":     "",
            "tp_price":     "",
            "lot_size":     "",
            "exit_price":   exit_price,
            "exit_reason":  exit_reason,
            "pnl_usd":      round(pnl_usd, 2),
            "equity_after": round(equity_after, 2),
            "ml_probability": round(ml_prob, 4),
            "rule_reason":  rule_reason,
        }
        self._append(row)
        self._print_close(row)

    # ── Console dashboard ─────────────────────────────────────────

    @staticmethod
    def _print_open(row: dict) -> None:
        d = row["direction"]
        e = row["entry_price"]
        print(
            f"\n  ┌─ NEW TRADE ──────────────────────────────\n"
            f"  │  {d:<6}  {row['symbol']}  entry={e}  "
            f"SL={row['sl_price']}  TP={row['tp_price']}\n"
            f"  │  lots={row['lot_size']}  "
            f"ML_prob={row['ml_probability']:.2%}  "
            f"reason={row['rule_reason']}\n"
            f"  └──────────────────────────────────────────"
        )

    @staticmethod
    def _print_close(row: dict) -> None:
        pnl  = row["pnl_usd"]
        icon = "WIN" if pnl > 0 else "LOSS"
        print(
            f"\n  ┌─ CLOSE {row['exit_reason']} {'─'*33}\n"
            f"  │  {icon}  {row['direction']:<6}  {row['symbol']}  "
            f"exit={row['exit_price']}  P&L={pnl:+.2f}\n"
            f"  │  equity={row['equity_after']}\n"
            f"  └──────────────────────────────────────────"
        )

    def _append(self, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRADE_LOG_FIELDS)
            writer.writerow(row)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")