"""
strategy/engine.py – Adapter for engine_fixed.py (institutional-grade engine)

Provides compatibility layer for main.py and execution modules.
Wraps StrategyEngineFixed to add scan_all() method for batch processing.
"""

import logging
from typing import Optional, List, Dict, Any
import pandas as pd

from strategy.engine_fixed import (
    StrategyEngineFixed,
    StrategyConfigFixed,
    SignalFixed,
)

logger = logging.getLogger(__name__)


class StrategyConfig(StrategyConfigFixed):
    """Adapter config that accepts old parameter names for compatibility."""
    
    def __init__(
        self,
        ml_threshold: float = None,
        sl_atr_mult: float = None,
        rr_ratio: float = None,
        require_htf_align: bool = None,
        **kwargs
    ):
        # Map old parameter names to new ones
        if ml_threshold is not None:
            kwargs['ml_threshold'] = ml_threshold
        
        if require_htf_align is not None:
            kwargs['require_htf_align'] = require_htf_align
        
        # Note: sl_atr_mult and rr_ratio from old API don't directly map to the fixed engine
        # The fixed engine uses sl_buffer_atr and rr_ratio_base instead.
        # For now, just pass through to parent config
        
        super().__init__(**kwargs)


class StrategyEngine(StrategyEngineFixed):
    """Wrapper adding scan_all() method for batch processing."""
    
    def __init__(self, config: Optional[StrategyConfig] = None, model=None):
        super().__init__(config, model)
    
    def scan_all(self, df: pd.DataFrame) -> List[SignalFixed]:
        """
        Scan entire dataframe bar-by-bar and return list of signals.
        
        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered dataframe with columns:
            - ohlc: open, high, low, close
            - atr
            - Features (sweep, bos, displacement, etc.)
        
        Returns
        -------
        List[SignalFixed]
            All signals generated from the dataframe
        """
        signals = []
        
        for bar_idx, (ts, row) in enumerate(df.iterrows()):
            # Skip warmup period (first 100 bars)
            if bar_idx < 100:
                continue
            
            ohlc = {
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "close": float(row.get("close", 0)),
            }
            
            atr = float(row.get("atr", 1.0))
            if atr == 0:
                atr = 1.0
            
            # Pass entire row as features dictionary
            features = row.to_dict()
            
            # Process this bar
            signal = self.process_bar(bar_idx, ts, ohlc, atr, features)
            
            if signal:
                signals.append(signal)
                logger.debug(f"Signal #{len(signals)}: {signal.direction:+d} @ {ts}")
        
        logger.info(f"Scan complete: {len(signals)} signals from {len(df)} bars")
        return signals


# Aliases for compatibility
SignalResult = SignalFixed

__all__ = ["StrategyEngine", "StrategyConfig", "SignalResult"]
