"""
data/histdata_parser.py
-----------------------
Parses CSV exports from HistData.com and Dukascopy into the system's
standard OHLCV DataFrame format (UTC-indexed, 5 columns).

Supported formats
-----------------
1. HistData ASCII M1 (most common download):
   20220103 000000;1.13218;1.13221;1.13200;1.13205;120

2. HistData ASCII D1:
   20220103;1.13218;1.13221;1.13200;1.13205;1200

3. Dukascopy JForex CSV (with or without BOM):
   Time (UTC),Open,High,Low,Close,Volume
   03.01.2022 00:00:00,1.13218,1.13221,1.13200,1.13205,120

4. Dukascopy *.csv tick export (OHLC summarised):
   Gmt time,Open,High,Low,Close,Volume

5. MT5 direct export (tab/comma):
   <DATE>	<TIME>	<OPEN>	<HIGH>	<LOW>	<CLOSE>	<TICKVOL>

The parser auto-detects format by inspecting the first non-empty line.
After parsing, data is resampled to the requested target timeframe.
"""

import io
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# --- pip size table ----------------------------------------------------------
JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "CADJPY", "AUDJPY", "NZDJPY",
             "CHFJPY", "ZARJPY", "MXNJPY", "SGDJPY"}

TIMEFRAME_MINUTES = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440,
}


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def parse_histdata(
    filepath: str,
    symbol: str = "EURUSD",
    target_tf: str = "M5",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Parse a HistData or Dukascopy CSV and return a clean OHLCV DataFrame
    resampled to *target_tf*, filtered to [start, end] date range.

    Parameters
    ----------
    filepath  : path to the CSV file
    symbol    : e.g. "EURUSD"  (used for pip-size validation only)
    target_tf : target timeframe string, e.g. "M5", "H1"
    start     : optional ISO date string "2022-01-01"
    end       : optional ISO date string "2023-12-31"

    Returns
    -------
    pd.DataFrame with columns [open, high, low, close, volume]
    indexed by UTC DatetimeIndex.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    logger.info(f"Parsing {path.name} -> {symbol} {target_tf}")

    raw_text = _read_file(path)
    fmt      = _detect_format(raw_text)
    logger.info(f"Detected format: {fmt}")

    df = _parse_dispatch(raw_text, fmt)
    df = _validate_ohlcv(df, symbol)
    df = _filter_dates(df, start, end)

    if target_tf != "M1":
        df = _resample(df, target_tf)

    logger.info(
        f"Parsed {len(df):,} {target_tf} bars | "
        f"{df.index[0].date()} -> {df.index[-1].date()}"
    )
    return df


# -----------------------------------------------------------------------------
# Format detection
# -----------------------------------------------------------------------------

def _read_file(path: Path) -> str:
    """Read file, strip BOM, handle encodings."""
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode file: {path}")


def _detect_format(text: str) -> str:
    """
    Return one of:
      'histdata_m1'    – semicolon-delimited, YYYYMMDD HHMMSS
      'histdata_d1'    – semicolon-delimited, YYYYMMDD (no time)
      'dukascopy'      – comma-delimited, DD.MM.YYYY HH:MM:SS
      'mt5_tab'        – tab-delimited MT5 export with <DATE> <TIME> headers
      'mt5_csv'        – comma-delimited MT5 export
      'generic_csv'    – fall-through: try comma, auto-detect columns
    """
    # look at first data line (skip header-like rows)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # skip BOM artifact or empty
    first = next((l for l in lines if l and not l.startswith("\ufeff")), "")

    if "\t" in first and "<DATE>" in text[:500]:
        return "mt5_tab"

    if first.startswith("<DATE>") or ("<DATE>" in text[:200]):
        return "mt5_csv"

    # HistData: 20220103 000000;1.13… or 20220103;1.13…
    if re.match(r"^\d{8} \d{6};", first):
        return "histdata_m1"
    if re.match(r"^\d{8};", first):
        return "histdata_d1"

    # Dukascopy: DD.MM.YYYY HH:MM:SS,  or  Gmt time header
    if re.match(r"^\d{2}\.\d{2}\.\d{4}", first):
        return "dukascopy"
    if "Gmt time" in text[:300] or "Time (UTC)" in text[:300]:
        return "dukascopy"

    return "generic_csv"


# -----------------------------------------------------------------------------
# Format-specific parsers
# -----------------------------------------------------------------------------

def _parse_dispatch(text: str, fmt: str) -> pd.DataFrame:
    parsers = {
        "histdata_m1": _parse_histdata_m1,
        "histdata_d1": _parse_histdata_d1,
        "dukascopy":   _parse_dukascopy,
        "mt5_tab":     _parse_mt5_tab,
        "mt5_csv":     _parse_mt5_csv,
        "generic_csv": _parse_generic_csv,
    }
    return parsers[fmt](text)


def _parse_histdata_m1(text: str) -> pd.DataFrame:
    """
    Format: 20220103 000000;1.13218;1.13221;1.13200;1.13205;120
    """
    df = pd.read_csv(
        io.StringIO(text),
        sep=";",
        header=None,
        names=["datetime", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["datetime"], format="%Y%m%d %H%M%S", utc=True)
    return _finalise(df)


def _parse_histdata_d1(text: str) -> pd.DataFrame:
    """
    Format: 20220103;1.13218;1.13221;1.13200;1.13205;1200
    """
    df = pd.read_csv(
        io.StringIO(text),
        sep=";",
        header=None,
        names=["datetime", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["datetime"], format="%Y%m%d", utc=True)
    return _finalise(df)


def _parse_dukascopy(text: str) -> pd.DataFrame:
    """
      Time (UTC),Open,High,Low,Close,Volume   <- header line
      Gmt time,Open,High,Low,Close,Volume      <- header line
    """
    # Normalise: strip Windows line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Detect if there's a header row
    first_line = text.split("\n")[0].strip()
    has_header = any(
        kw in first_line.lower()
        for kw in ("time", "open", "high", "low", "close", "volume", "date")
    )

    df = pd.read_csv(
        io.StringIO(text),
        sep=",",
        header=0 if has_header else None,
    )

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]
    aliases = {
        "time (utc)": "timestamp", "gmt time": "timestamp",
        "time": "timestamp", "date": "timestamp",
        "open": "open", "high": "high", "low": "low",
        "close": "close", "volume": "volume",
    }
    df.rename(columns={k: v for k, v in aliases.items() if k in df.columns}, inplace=True)

    if "timestamp" not in df.columns:
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]

    # Parse Dukascopy datetime: "03.01.2022 00:00:00.000" or "03.01.2022 00:00:00"
    def _parse_dt(s):
        s = str(s).strip()
        # Strip trailing milliseconds: keep only first 19 chars if timestamp is longer
        # e.g. "03.01.2022 00:00:00.000" -> "03.01.2022 00:00:00"
        if len(s) > 19 and s[19:20] in (".", ","):
            s = s[:19]
        for fmt in ("%d.%m.%Y %H:%M:%S", "%Y.%m.%d %H:%M:%S",
                    "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
            try:
                return pd.to_datetime(s, format=fmt, utc=True)
            except Exception:
                pass
        return pd.NaT

    df["timestamp"] = df["timestamp"].apply(_parse_dt)
    df.dropna(subset=["timestamp"], inplace=True)
    return _finalise(df)


def _parse_mt5_tab(text: str) -> pd.DataFrame:
    """
    MT5 tab-delimited: <DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>...
    Date format: 2022.01.03  Time format: 00:00
    Handles both <TICKVOL> and <VOL> columns (keeps only the first).
    """
    df = pd.read_csv(io.StringIO(text), sep="\t", dtype=str)
    df.columns = [c.strip().strip("<>").lower() for c in df.columns]

    # Drop ambiguous extra columns (vol, spread) before renaming
    for drop_col in ["vol", "spread"]:
        if drop_col in df.columns and "tickvol" in df.columns:
            df.drop(columns=[drop_col], inplace=True, errors="ignore")

    df.rename(columns={"tickvol": "volume", "tick_volume": "volume"}, inplace=True)
    if "volume" not in df.columns:
        df["volume"] = "0"

    # Build timestamp
    date_col = next((c for c in df.columns if "date" in c), None)
    time_col = next((c for c in df.columns if "time" in c and c != date_col), None)
    if date_col and time_col:
        dt_str = df[date_col].str.strip() + " " + df[time_col].str.strip()
        df["timestamp"] = pd.to_datetime(dt_str, utc=True, errors="coerce")
    elif date_col:
        df["timestamp"] = pd.to_datetime(df[date_col].str.strip(), utc=True, errors="coerce")

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")

    df.dropna(subset=["timestamp", "open", "high", "low", "close"], inplace=True)
    return _finalise(df)


def _parse_mt5_csv(text: str) -> pd.DataFrame:
    """
    MT5 comma-delimited export (same columns as tab but with commas).
    """
    df = pd.read_csv(io.StringIO(text), sep=",")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    date_col = next((c for c in df.columns if "date" in c), None)
    time_col = next((c for c in df.columns if "time" in c and c != date_col), None)
    if date_col and time_col:
        df["timestamp"] = pd.to_datetime(
            df[date_col].astype(str) + " " + df[time_col].astype(str), utc=True
        )
    elif date_col:
        df["timestamp"] = pd.to_datetime(df[date_col], utc=True)
    df.rename(columns={"tickvol": "volume", "tick_volume": "volume"}, inplace=True)
    if "volume" not in df.columns:
        df["volume"] = 0
    return _finalise(df)


def _parse_generic_csv(text: str) -> pd.DataFrame:
    """
    Best-effort parser: auto-detect separator and column mapping.
    Handles both headered and headerless files.
    """
    # Detect separator
    if text.count("\t") > text.count(",") and text.count("\t") > text.count(";"):
        sep = "\t"
    else:
        sep = ";" if text.count(";") > text.count(",") else ","

    # Peek at first line to decide if there is a header
    first_line = text.lstrip("\ufeff").splitlines()[0].strip()
    fields = first_line.split(sep)

    # If the first field looks like a date/datetime value → headerless file
    def _looks_like_date(s: str) -> bool:
        s = s.strip().strip('"').strip("'")
        return bool(re.match(r"^\d{4}[.\-/]\d{2}[.\-/]\d{2}", s) or
                    re.match(r"^\d{2}[.\-/]\d{2}[.\-/]\d{4}", s) or
                    re.match(r"^\d{8}\s+\d{6}$", s))

    if _looks_like_date(fields[0]):
        # Headerless – read without header and assign standard names
        col_names = ["timestamp", "open", "high", "low", "close", "volume"]
        n = len(fields)
        if n < 6:
            col_names = col_names[:n]
        elif n > 6:
            col_names = col_names + [f"extra_{i}" for i in range(n - 6)]
        df = pd.read_csv(io.StringIO(text), sep=sep, header=None, names=col_names)
    else:
        df = pd.read_csv(io.StringIO(text), sep=sep, header=0)
        df.columns = [str(c).strip().lower() for c in df.columns]

        aliases = {
            "date": "timestamp", "datetime": "timestamp", "time": "timestamp",
            "o": "open", "h": "high", "l": "low", "c": "close",
            "v": "volume", "vol": "volume", "tick_volume": "volume",
        }
        df.rename(columns={k: v for k, v in aliases.items() if k in df.columns}, inplace=True)

        if "timestamp" not in df.columns and len(df.columns) >= 6:
            df.columns = ["timestamp", "open", "high", "low", "close", "volume"] + \
                         list(df.columns[6:])

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)
    return _finalise(df)


# -----------------------------------------------------------------------------
# Shared utilities
# -----------------------------------------------------------------------------

def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    """Set index, cast numerics, sort, de-duplicate."""
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    # Cast ALL price/volume columns to float64 first — critical before any arithmetic
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    vol = df["volume"] if "volume" in df.columns else pd.Series(0, index=df.index)
    df["volume"] = pd.to_numeric(vol, errors="coerce").fillna(0)

    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    return df


def _validate_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Drop rows that violate OHLC sanity or have implausible prices."""
    before = len(df)

    # Ensure numeric types (safety — _finalise should have done this already)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = (
        (df["high"] >= df["open"])  &
        (df["high"] >= df["close"]) &
        (df["high"] >= df["low"])   &
        (df["low"]  <= df["open"])  &
        (df["low"]  <= df["close"]) &
        (df["open"] > 0) &
        (df["close"] > 0)
    )

    # Spike filter: bar range > 200× the median bar range → data error.
    # Median is resistant to the outlier polluting the reference.
    hl_range    = (df["high"] - df["low"]).abs()
    median_range = hl_range.median()
    spike_limit = max(median_range * 200, 1e-6)
    valid      &= hl_range <= spike_limit

    df = df[valid].copy()
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped:,} invalid/spike OHLCV rows from {symbol}")
    return df


def _filter_dates(df: pd.DataFrame,
                  start: Optional[str],
                  end: Optional[str]) -> pd.DataFrame:
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]
    return df


def _resample(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    freq = f"{TIMEFRAME_MINUTES[target_tf]}min"
    out  = df.resample(freq).agg(
        open   = ("open",   "first"),
        high   = ("high",   "max"),
        low    = ("low",    "min"),
        close  = ("close",  "last"),
        volume = ("volume", "sum"),
    ).dropna(subset=["open"])
    return out
