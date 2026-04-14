"""
features/hwr_labeler.py
-----------------------
Labeler for the High Win Rate (EMA Pullback) strategy.

Uses fixed-horizon ATR-based TP/SL — no lookahead bias.
SL = 1.0 ATR, TP = 1.5 ATR (RR 1:1.5)
Worst-case: SL checked before TP on same bar.
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class HWRLabelConfig:
    sl_atr_mult:   float = 1.0
    tp_atr_mult:   float = 1.5
    spread_pips:   float = 1.5
    pip_size:      float = 0.0001


class HWRLabeler:

    def __init__(self, config: Optional[HWRLabelConfig] = None):
        self.cfg = config or HWRLabelConfig()

    def label(self, df: pd.DataFrame, signals: list) -> pd.DataFrame:
        """
        Given the prepared DataFrame and list of SignalResult,
        simulate outcomes and return labeled DataFrame.
        """
        if not signals:
            return pd.DataFrame()

        high  = df["high"].values
        low   = df["low"].values
        atr   = df["atr"].values
        size  = len(df)
        idx_map = {ts: i for i, ts in enumerate(df.index)}

        records = []
        for sig in signals:
            i = idx_map.get(sig.timestamp)
            if i is None:
                continue

            spread   = self.cfg.spread_pips * self.cfg.pip_size
            bar_atr  = float(atr[i])
            ep       = sig.entry_price
            sl       = sig.sl_price
            tp       = sig.tp_price

            label, bars_held = self._simulate(sig.direction, i, tp, sl, high, low, size)
            if label is None:
                continue

            records.append({
                "timestamp":      sig.timestamp,
                "direction":      sig.direction,
                "entry_price":    ep,
                "sl_price":       sl,
                "tp_price":       tp,
                "label":          label,
                "bars_held":      bars_held,
                "ml_probability": sig.ml_probability,
                "rule_reason":    sig.rule_reason,
                # Features from the bar
                **{c: df.iloc[i][c] for c in df.columns
                   if c not in ("open","high","low","close","volume")},
            })

        if not records:
            return pd.DataFrame()

        out = pd.DataFrame(records).set_index("timestamp")
        wr  = out["label"].mean()
        logger.info(
            f"HWR Labeler: {len(out):,} setups | "
            f"Win rate: {wr:.1%} | "
            f"Long: {(out['direction']==1).sum()} | "
            f"Short: {(out['direction']==-1).sum()}"
        )
        return out

    @staticmethod
    def _simulate(direction, entry_idx, tp, sl, high, low, size):
        for i in range(entry_idx + 1, size):
            if direction == 1:
                if low[i]  <= sl: return 0, i - entry_idx
                if high[i] >= tp: return 1, i - entry_idx
            else:
                if high[i] >= sl: return 0, i - entry_idx
                if low[i]  <= tp: return 1, i - entry_idx
        return None, None