"""
backtest/walk_forward.py — WALK-FORWARD FRAMEWORK
================================================

NO FUTURE DATA LEAKAGE. Out-of-sample validation using expanding windows.

Pattern:
  - [2020-01-01 to 2022-01-01]   → Train  
  - [2022-01-01 to 2022-03-01]   → Test (3 months)
  - [2022-01-01 to 2022-04-01]   → Train
  - [2022-04-01 to 2022-06-01]   → Test (3 months)
  - ... repeat ...

Each test period:
  - Uses model trained only on past data
  - No training on test period data
  - Walk-forward compatible

"""

import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardSegment:
    """One fold of walk-forward validation."""
    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_data: pd.DataFrame = None
    test_data: pd.DataFrame = None
    model_path: Path = None


class WalkForwardValidator:
    """
    Walk-forward out-of-sample validator.
    Ensures NO future data leakage in backtesting.
    """

    def __init__(
        self,
        initial_train_months: int = 24,
        test_period_months: int = 3,
        roll_forward_months: int = 1,
    ):
        """
        Parameters
        ----------
        initial_train_months : e.g. 24 = first 24 months of training
        test_period_months : e.g. 3 = test on next 3 months
        roll_forward_months : e.g. 1 = slide by 1 month for next fold
        """
        self.initial_train_months = initial_train_months
        self.test_period_months = test_period_months
        self.roll_forward_months = roll_forward_months
        self.segments: List[WalkForwardSegment] = []

    def generate_segments(self, df: pd.DataFrame) -> List[WalkForwardSegment]:
        """
        Generate walk-forward segments from time-series data.
        
        Parameters
        ----------
        df : DataFrame indexed by timestamp
        
        Returns
        -------
        List of WalkForwardSegment
        """
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        start_date = df.index[0]
        end_date = df.index[-1]
        
        logger.info(
            f"WF segments: {start_date.date()} → {end_date.date()} | "
            f"train={self.initial_train_months}m, test={self.test_period_months}m, "
            f"roll={self.roll_forward_months}m"
        )
        
        self.segments = []
        fold_idx = 0
        
        train_start = start_date
        train_end = train_start + pd.DateOffset(months=self.initial_train_months)
        
        while train_end < end_date:
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.test_period_months)
            
            # Ensure test_end doesn't exceed data
            if test_end > end_date:
                test_end = end_date
            
            # Ensure we have enough test data
            test_data = df[(df.index >= test_start) & (df.index < test_end)]
            if len(test_data) < 100:
                break
            
            # Get training data
            train_data = df[(df.index >= train_start) & (df.index < train_end)]
            if len(train_data) < 500:
                break
            
            segment = WalkForwardSegment(
                fold_idx=fold_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_data=train_data,
                test_data=test_data,
            )
            self.segments.append(segment)
            
            logger.info(
                f"  Fold {fold_idx}: "
                f"Train [{train_start.date()} → {train_end.date()}] ({len(train_data)} bars) | "
                f"Test [{test_start.date()} → {test_end.date()}] ({len(test_data)} bars)"
            )
            
            # Slide window forward
            train_start = train_start + pd.DateOffset(months=self.roll_forward_months)
            train_end = train_start + pd.DateOffset(months=self.initial_train_months)
            fold_idx += 1
        
        logger.info(f"Generated {len(self.segments)} walk-forward segments")
        return self.segments


class WalkForwardBacktestRunner:
    """
    Run backtest across all walk-forward segments.
    Trains model on training period, tests on test period.
    """

    def __init__(self, validator: WalkForwardValidator):
        self.validator = validator
        self.results: List[Dict] = []

    def run_all(
        self,
        df: pd.DataFrame,
        strategy_fn,  # callable(train_df, test_df, segment) -> results_dict
        symbol: str = "EURUSD",
    ) -> pd.DataFrame:
        """
        Run walk-forward validation.
        
        For each segment:
          1. Train model on train_data
          2. Backtest on test_data
          3. Record metrics
        
        Parameters
        ----------
        df : full time-series
        strategy_fn : function(train_df, test_df, segment) -> results dict
        symbol : symbol being tested
        
        Returns
        -------
        DataFrame with results per segment
        """
        segments = self.validator.generate_segments(df)
        
        for segment in segments:
            logger.info(f"\n=== Walk-Forward Fold {segment.fold_idx} ===")
            
            try:
                # Run strategy on this segment
                result = strategy_fn(
                    segment.train_data,
                    segment.test_data,
                    segment
                )
                result["fold_idx"] = segment.fold_idx
                result["symbol"] = symbol
                self.results.append(result)
                
                logger.info(f"Fold {segment.fold_idx}: ✓")
            except Exception as e:
                logger.error(f"Fold {segment.fold_idx} failed: {e}")
        
        # Aggregate results
        if not self.results:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(self.results)
        self._log_aggregate_metrics(results_df)
        
        return results_df

    def _log_aggregate_metrics(self, results_df: pd.DataFrame) -> None:
        """Log summary statistics across folds."""
        if results_df.empty:
            return
        
        logger.info("\n=== Walk-Forward Summary ===")
        
        # Aggregate stats
        total_trades = results_df["num_trades"].sum()
        total_pnl = results_df["total_pnl"].sum()
        avg_win_rate = results_df["win_rate"].mean()
        avg_sharpe = results_df["sharpe"].mean()
        avg_dd = results_df["max_dd"].mean()
        
        logger.info(
            f"Total trades: {total_trades} | "
            f"Total P&L: {total_pnl:.2f} USD | "
            f"Avg win rate: {avg_win_rate:.1%} | "
            f"Avg Sharpe: {avg_sharpe:.2f} | "
            f"Avg DD: {avg_dd:.1%}"
        )
        
        # Per-fold detail
        for _, row in results_df.iterrows():
            fold = row["fold_idx"]
            logger.info(
                f"  Fold {fold}: {row['num_trades']:.0f} trades | "
                f"Win {row['win_rate']:.1%} | "
                f"P&L {row['total_pnl']:+.2f} | "
                f"Sharpe {row['sharpe']:.2f}"
            )


# ──────────────────────────────────────────────────────────────────────────
# Integration: How to use walk-forward with fixed backtest engine
# ──────────────────────────────────────────────────────────────────────────

def run_walk_forward_backtest(
    df: pd.DataFrame,
    symbol: str = "EURUSD",
    initial_train_months: int = 24,
    test_period_months: int = 3,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Full walk-forward validation pipeline.
    
    Example usage:
    ──────────────
    # Load 5 years of M5 bars
    df = load_bars("EURUSD", years=5, timeframe="M5")
    df = engineer_features(df)
    
    # Run walk-forward
    results_df, trade_logs = run_walk_forward_backtest(df, symbol="EURUSD")
    
    # results_df: one row per fold
    # trade_logs: full list of trades across all folds
    
    Returns
    -------
    results_df : DataFrame with { fold, trades, pnl, sharpe, ... }
    trade_logs : List[TradeEntry] from all folds
    """
    from features.engineer_fixed import engineer_features
    from models.ml_integration_fixed import ForexMLModelFixed, get_ml_feature_columns
    from strategy.engine_fixed import StrategyEngineFixed, StrategyConfigFixed
    from risk.manager_fixed import RiskManager, RiskConfig
    from backtest.engine import BacktestEngine, BacktestConfig
    
    # WF setup
    validator = WalkForwardValidator(
        initial_train_months=initial_train_months,
        test_period_months=test_period_months,
    )
    runner = WalkForwardBacktestRunner(validator)
    
    def strategy_fn(train_df: pd.DataFrame, test_df: pd.DataFrame, segment) -> Dict:
        """Strategy function for one fold."""
        from features.labeler import SetupLabeler, LabelConfig
        
        # Train ML model on training period
        logger.info(f"[Fold {segment.fold_idx}] Training model on {len(train_df)} bars...")
        
        # Feature labeling
        labeler = SetupLabeler(LabelConfig())
        labeled_train = labeler.label(train_df)
        
        if len(labeled_train) < 30:
            logger.warning(f"[Fold {segment.fold_idx}] Too few labeled trades")
            return {
                "num_trades": 0, "total_pnl": 0, "win_rate": 0,
                "sharpe": 0, "max_dd": 0,
            }
        
        # Train model
        ml_model = ForexMLModelFixed(f"{symbol}_xgb_fold{segment.fold_idx}")
        feat_cols = get_ml_feature_columns(labeled_train)
        X_train = labeled_train[feat_cols].fillna(0)
        y_train = labeled_train["label"]
        
        metrics = ml_model.train(X_train, y_train)
        ml_model.save()
        
        # Backtest on test period
        logger.info(f"[Fold {segment.fold_idx}] Backtesting on {len(test_df)} bars...")
        
        engine = StrategyEngineFixed(StrategyConfigFixed(), model=ml_model)
        risk_mgr = RiskManager(RiskConfig())
        backtest = BacktestEngine(BacktestConfig(symbol=symbol))
        
        signals = []
        for i, (ts, row) in enumerate(test_df.iterrows()):
            if i < 100:
                continue  # Warm-up
            
            # Extract features
            features = row.to_dict()
            
            # Process bar
            signal = engine.process_bar(
                bar_idx=i,
                timestamp=ts,
                ohlc={
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                },
                atr=row.get("atr", 0.05),
                features=features,
            )
            
            if signal:
                # Risk approval
                approved, reason = risk_mgr.approve_trade(
                    entry_price=signal.entry_price,
                    sl_price=signal.sl_price,
                    tp_price=signal.tp_price,
                    direction=signal.direction,
                    symbol=symbol,
                    ml_prob=signal.ml_probability,
                    rr_ratio=signal.rr_ratio,
                )
                
                if approved:
                    signals.append(signal)
        
        # Run backtest
        results = backtest.run(test_df, signals)
        
        # Metrics
        num_trades = len(results.trades)
        total_pnl = sum(t.pnl_usd for t in results.trades)
        win_rate = sum(1 for t in results.trades if t.pnl_usd > 0) / max(num_trades, 1)
        sharpe = results.sharpe() if hasattr(results, "sharpe") else 0
        max_dd = results.max_dd if hasattr(results, "max_dd") else 0
        
        return {
            "num_trades": num_trades,
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "trades": results.trades,
        }
    
    # Run all folds
    results_df = runner.run_all(df, strategy_fn, symbol=symbol)
    
    # Aggregate trades
    all_trades = []
    for _, row in results_df.iterrows():
        all_trades.extend(row.get("trades", []))
    
    return results_df, all_trades
