"""
data/loader.py
--------------
Handles loading, validation, and preprocessing of OHLCV forex data.
Supports multiple timeframes and handles missing/corrupt data robustly.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

TIMEFRAME_MINUTES = {
    "M1":  1,
    "M5":  5,
    "M15": 15,
    "M30": 30,
    "H1":  60,
    "H4":  240,
    "D1":  1440,
}


class OHLCVLoader:
    """
    Loads and cleans OHLCV data from CSV files.
    Performs basic sanity checks and resampling between timeframes.
    """

    def __init__(self, filepath: str, timeframe: str = "M15"):
        self.filepath  = Path(filepath)
        self.timeframe = timeframe.upper()
        self._validate_timeframe()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """Load, validate, clean and return a DataFrame indexed by timestamp."""
        logger.info(f"Loading data from {self.filepath} [{self.timeframe}]")
        df = self._read_csv()
        df = self._normalise_columns(df)
        df = self._parse_timestamps(df)
        df = self._validate_ohlcv(df)
        df = self._remove_duplicates(df)
        df = self._fill_gaps(df)
        df = self._cast_numeric(df)
        logger.info(f"Loaded {len(df):,} rows | {df.index[0]} → {df.index[-1]}")
        return df

    def resample(self, df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """
        Resample a DataFrame from its native timeframe to *target_tf*.
        Useful for producing HTF context from tick/M1 data.
        """
        target_tf = target_tf.upper()
        self._validate_timeframe(target_tf)
        freq = f"{TIMEFRAME_MINUTES[target_tf]}min"
        resampled = df.resample(freq).agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low",  "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        ).dropna()
        logger.info(f"Resampled to {target_tf}: {len(resampled):,} bars")
        return resampled

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_timeframe(self, tf: Optional[str] = None) -> None:
        tf = tf or self.timeframe
        if tf not in TIMEFRAME_MINUTES:
            raise ValueError(
                f"Unknown timeframe '{tf}'. Valid: {list(TIMEFRAME_MINUTES.keys())}"
            )

    def _read_csv(self) -> pd.DataFrame:
        if not self.filepath.exists():
            raise FileNotFoundError(f"Data file not found: {self.filepath}")
        try:
            df = pd.read_csv(
                self.filepath,
                header=0,
                low_memory=False,
                na_values=["", "N/A", "n/a", "nan", "NULL", "null"],
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to read CSV: {exc}") from exc
        if df.empty:
            raise ValueError("CSV file is empty.")
        return df

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lower-case column names and map common aliases."""
        df.columns = [c.strip().lower() for c in df.columns]
        aliases: Dict[str, str] = {
            "time": "timestamp", "date": "timestamp", "datetime": "timestamp",
            "o": "open", "h": "high", "l": "low", "c": "close",
            "v": "volume", "vol": "volume", "tick_volume": "volume",
        }
        df.rename(columns=aliases, inplace=True)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return df[REQUIRED_COLUMNS]

    def _parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df

    def _validate_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows that violate OHLCV sanity constraints."""
        before = len(df)
        # High must be >= all other price columns
        valid = (
            (df["high"] >= df["open"]) &
            (df["high"] >= df["close"]) &
            (df["high"] >= df["low"]) &
            (df["low"]  <= df["open"]) &
            (df["low"]  <= df["close"]) &
            (df["open"] > 0) &
            (df["close"] > 0) &
            (df["volume"] >= 0)
        )
        df = df[valid].copy()
        dropped = before - len(df)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows with invalid OHLCV values.")
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df[~df.index.duplicated(keep="last")]
        dropped = before - len(df)
        if dropped > 0:
            logger.warning(f"Removed {dropped} duplicate timestamps.")
        return df

    def _fill_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward-fill small gaps (up to 3 missing bars) so features don't
        have NaN spikes.  Weekends / sessions gaps are untouched (>3 bars).
        """
        freq    = f"{TIMEFRAME_MINUTES[self.timeframe]}min"
        full_ix = pd.date_range(df.index[0], df.index[-1], freq=freq, tz="UTC")
        df      = df.reindex(full_ix)
        # Only forward-fill gaps ≤ 3 bars
        mask = df["close"].isna()
        run  = mask.groupby((~mask).cumsum()).cumcount()
        df.loc[run <= 3] = df.loc[run <= 3].ffill()
        remaining = df["close"].isna().sum()
        if remaining > 0:
            logger.info(f"Dropping {remaining} rows with unfillable gaps (weekend/session).")
            df.dropna(subset=["close"], inplace=True)
        return df

    def _cast_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(inplace=True)
        return df


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def load_multi_timeframe(
    filepath: str,
    base_tf: str = "M5",
    higher_tfs: Optional[list] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load a single CSV and return a dict of DataFrames keyed by timeframe string.
    The base timeframe is loaded as-is; higher timeframes are resampled from it.
    """
    if higher_tfs is None:
        higher_tfs = ["M15", "H1"]

    loader  = OHLCVLoader(filepath, base_tf)
    base_df = loader.load()

    result = {base_tf: base_df}
    for tf in higher_tfs:
        if TIMEFRAME_MINUTES[tf] > TIMEFRAME_MINUTES[base_tf]:
            result[tf] = loader.resample(base_df, tf)
        else:
            logger.warning(f"Skipping {tf}: not higher than base {base_tf}")
    return result
