"""
models/ml_model.py — FIXED
Fixes applied:
  [1.2] Calibration uses TimeSeriesSplit not cv=None (time-ordered)
  [1.3] Threshold tuned on held-out validation split, NOT the test set
  [4.1] Added expected_value() method alongside raw probability
  [4.2] Dynamic class weighting recalculated per training call
  [4.4] CV uses proper expanding-window TimeSeriesSplit
"""

import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, brier_score_loss,
    classification_report, precision_recall_curve,
)
from sklearn.calibration import CalibratedClassifierCV

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

SAVED_MODELS_DIR = Path("saved_models")
SAVED_MODELS_DIR.mkdir(exist_ok=True)


def _build_estimator(n_samples: int, class_weight: float) -> Any:
    if XGB_AVAILABLE:
        return xgb.XGBClassifier(
            n_estimators          = 400,
            max_depth             = 4,
            learning_rate         = 0.05,
            subsample             = 0.8,
            colsample_bytree      = 0.7,
            min_child_weight      = max(1, int(n_samples * 0.005)),
            scale_pos_weight      = class_weight,   # [4.2] dynamic per call
            reg_alpha             = 0.1,
            reg_lambda            = 1.0,
            eval_metric           = "auc",
            early_stopping_rounds = 30,
            random_state          = 42,
            verbosity             = 0,
        )
    else:
        logger.warning("XGBoost not found – using RandomForestClassifier fallback.")
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators     = 300,
            max_depth        = 6,
            min_samples_leaf = max(5, int(n_samples * 0.005)),
            class_weight     = "balanced",
            random_state     = 42,
            n_jobs           = -1,
        )


class ForexMLModel:

    def __init__(self, model_name: str = "forex_xgb"):
        self.model_name    = model_name
        self.model_path    = SAVED_MODELS_DIR / f"{model_name}.joblib"
        self.meta_path     = SAVED_MODELS_DIR / f"{model_name}_meta.json"
        self.pipeline      = None
        self.feature_names: list = []
        self.threshold:    float = 0.55
        self.rr_ratio:     float = 2.0   # [4.1] stored for EV calculation
        self.meta:         Dict[str, Any] = {}

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X:           pd.DataFrame,
        y:           pd.Series,
        test_size:   float = 0.20,
        val_size:    float = 0.15,   # [1.3] separate validation split for threshold
        n_cv_splits: int   = 5,
        rr_ratio:    float = 2.0,
    ) -> Dict[str, Any]:
        """
        Chronological 3-way split:
          [-------- train --------][-- val --][-- test --]
          threshold tuned on val; final eval on test only.
        """
        self.feature_names = list(X.columns)
        self.rr_ratio      = rr_ratio
        n = len(X)

        # [1.3] Three-way chronological split
        train_end = int(n * (1 - test_size - val_size))
        val_end   = int(n * (1 - test_size))

        X_train = X.iloc[:train_end]
        X_val   = X.iloc[train_end:val_end]
        X_test  = X.iloc[val_end:]
        y_train = y.iloc[:train_end]
        y_val   = y.iloc[train_end:val_end]
        y_test  = y.iloc[val_end:]

        logger.info(
            f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}"
        )
        logger.info(f"Label balance (train) – 1: {y_train.mean():.1%}")

        # [4.2] Dynamic class weight computed from THIS training set
        n_pos = max(y_train.sum(), 1)
        n_neg = max(len(y_train) - n_pos, 1)
        cw    = n_neg / n_pos

        estimator = _build_estimator(len(X_train), cw)

        # ── [4.4] Time-series CV on train only ───────────────────
        n_splits = min(n_cv_splits, max(2, len(X_train) // 3))
        tscv     = TimeSeriesSplit(n_splits=n_splits)
        cv_aucs  = []

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
            Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            ytr, yval = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            if len(yval.unique()) < 2:
                logger.info(f"  CV Fold {fold}: skipped (single class in validation)")
                continue

            if XGB_AVAILABLE:
                estimator.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
            else:
                estimator.fit(Xtr, ytr)

            try:
                prob = estimator.predict_proba(Xval)[:, 1]
                auc  = roc_auc_score(yval, prob)
                cv_aucs.append(auc)
                logger.info(f"  CV Fold {fold}/{n_splits} AUC: {auc:.4f}")
            except Exception:
                pass

        mean_cv_auc = float(np.mean(cv_aucs)) if cv_aucs else float("nan")
        std_cv_auc  = float(np.std(cv_aucs))  if cv_aucs else float("nan")
        logger.info(f"CV AUC: {mean_cv_auc:.4f} ± {std_cv_auc:.4f}")

        # ── Final fit on train set ────────────────────────────────
        if XGB_AVAILABLE:
            estimator.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            estimator.fit(X_train, y_train)

        # [1.2] Calibration with TimeSeriesSplit — preserves time order
        # Build a clone WITHOUT early_stopping_rounds so CalibratedClassifierCV
        # can refit internally without needing an eval_set each fold.
        try:
            if XGB_AVAILABLE:
                cal_estimator = xgb.XGBClassifier(
                    n_estimators     = estimator.best_iteration + 1 if hasattr(estimator, "best_iteration") and estimator.best_iteration else 200,
                    max_depth        = 4,
                    learning_rate    = 0.05,
                    subsample        = 0.8,
                    colsample_bytree = 0.7,
                    min_child_weight = max(1, int(len(X_train) * 0.005)),
                    scale_pos_weight = cw,
                    reg_alpha        = 0.1,
                    reg_lambda       = 1.0,
                    eval_metric      = "auc",
                    random_state     = 42,
                    verbosity        = 0,
                    # NO early_stopping_rounds — calibration refits without eval_set
                )
            else:
                cal_estimator = estimator

            cal_cv = TimeSeriesSplit(n_splits=min(3, max(2, len(X_train) // 4)))
            calibrated = CalibratedClassifierCV(
                cal_estimator, method="sigmoid", cv=cal_cv
            )
            calibrated.fit(X_train, y_train)
            self.pipeline = calibrated
        except Exception as exc:
            logger.warning(f"Calibration failed ({exc}), using raw estimator.")
            self.pipeline = estimator

        # [1.3] Threshold tuned on VALIDATION set — not test
        proba_val   = self.pipeline.predict_proba(X_val)[:, 1]
        if len(y_val.unique()) > 1:
            self.threshold = self._find_optimal_threshold(y_val, proba_val)
        else:
            self.threshold = 0.55
        # Floor: never use a threshold below 0.45 regardless of what val optimises to.
        # A very low threshold on a weak model passes too many bad trades.
        self.threshold = max(self.threshold, 0.45)

        # ── Test-set evaluation (never touched during training) ───
        proba_test = self.pipeline.predict_proba(X_test)[:, 1]
        try:
            test_auc   = roc_auc_score(y_test, proba_test)
            test_brier = brier_score_loss(y_test, proba_test)
        except Exception:
            test_auc, test_brier = float("nan"), float("nan")

        y_pred = (proba_test >= self.threshold).astype(int)
        try:
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            logger.info(f"\n{classification_report(y_test, y_pred, zero_division=0)}")
        except Exception:
            report = {}

        logger.info(
            f"Test AUC: {test_auc:.4f} | "
            f"Brier: {test_brier:.4f} | "
            f"Threshold (from val): {self.threshold:.2f}"
        )

        fi = self._feature_importance()
        if not fi.empty:
            logger.info(f"\nTop-10 Feature Importances:\n{fi.head(10).to_string()}")

        self.meta = {
            "model_name":    self.model_name,
            "feature_names": self.feature_names,
            "threshold":     float(self.threshold),
            "rr_ratio":      float(rr_ratio),
            "cv_auc_mean":   float(mean_cv_auc),
            "cv_auc_std":    float(std_cv_auc),
            "test_auc":      float(test_auc),
            "test_brier":    float(test_brier),
            "n_train":       int(len(X_train)),
            "n_val":         int(len(X_val)),
            "n_test":        int(len(X_test)),
        }

        return {**self.meta, "cv_aucs": cv_aucs,
                "report": report, "feature_importance": fi}

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.pipeline.predict_proba(self._align(X))[:, 1]

    def expected_value(self, X: pd.DataFrame) -> np.ndarray:
        """
        [4.1] Expected value per unit risked:
          EV = P(win)*RR - P(loss)*1
        Positive EV -> trade has edge. Use this for position sizing.
        """
        p   = self.predict_proba(X)
        rr  = self.rr_ratio
        return p * rr - (1 - p)

    def is_valid_setup(self, X: pd.DataFrame,
                       threshold: Optional[float] = None) -> np.ndarray:
        t = threshold if threshold is not None else self.threshold
        return self.predict_proba(X) >= t

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        self._check_fitted()
        joblib.dump(self.pipeline, self.model_path)
        with open(self.meta_path, "w") as f:
            safe = {k: v for k, v in self.meta.items()
                    if isinstance(v, (str, int, float, list, dict, bool))}
            json.dump(safe, f, indent=2)
        logger.info(f"Model saved -> {self.model_path}")

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"No saved model at {self.model_path}")
        self.pipeline      = joblib.load(self.model_path)
        if self.meta_path.exists():
            with open(self.meta_path) as f:
                self.meta = json.load(f)
            self.feature_names = self.meta.get("feature_names", [])
            self.threshold     = self.meta.get("threshold", 0.55)
            self.rr_ratio      = self.meta.get("rr_ratio", 2.0)
        logger.info(f"Model loaded <- {self.model_path}")

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _feature_importance(self) -> pd.Series:
        try:
            pipe = self.pipeline
            if hasattr(pipe, "calibrated_classifiers_"):
                base = pipe.calibrated_classifiers_[0].estimator
            elif hasattr(pipe, "estimator"):
                base = pipe.estimator
            else:
                base = pipe
            if hasattr(base, "feature_importances_"):
                return pd.Series(
                    base.feature_importances_,
                    index=self.feature_names,
                ).sort_values(ascending=False)
        except Exception:
            pass
        return pd.Series(dtype=float)

    @staticmethod
    def _find_optimal_threshold(y_true, y_prob) -> float:
        """Maximise F1 — called on VALIDATION set only."""
        prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
        denom = prec + rec
        # Suppress divide-by-zero RuntimeWarning when both prec and rec are 0
        with np.errstate(invalid="ignore"):
            f1 = np.where(denom > 0, 2 * prec * rec / np.where(denom > 0, denom, 1), 0)
        best = np.argmax(f1)
        return float(thresholds[best]) if best < len(thresholds) else 0.55

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names:
            return X
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            logger.warning(f"Zero-filling missing features: {missing}")
        return X.reindex(columns=self.feature_names, fill_value=0)

    def _check_fitted(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

    # Keep alias for backward compat
    def _feature_importance_series(self):
        return self._feature_importance()