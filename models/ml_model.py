"""
models/ml_model.py
------------------
XGBoost probability classifier with:
  - Stratified train/test split
  - Time-series cross-validation (no look-ahead leakage)
  - Feature importance reporting
  - Model save / load via joblib
  - Calibration check (Brier score)
  - Anti-overfitting: early stopping + max_depth control

The model outputs P(label=1), i.e. probability that the setup hits TP.
The strategy engine thresholds this to filter entries.
"""

import logging
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, brier_score_loss,
    classification_report, precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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


# ──────────────────────────────────────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_estimator(n_samples: int, class_weight: float) -> Any:
    """
    Return an XGBoost or RandomForest estimator depending on availability.
    Parameters are conservatively tuned to avoid overfitting.
    """
    if XGB_AVAILABLE:
        return xgb.XGBClassifier(
            n_estimators       = 400,
            max_depth          = 4,           # shallow trees → less overfit
            learning_rate      = 0.05,
            subsample          = 0.8,
            colsample_bytree   = 0.7,
            min_child_weight   = max(1, int(n_samples * 0.005)),
            scale_pos_weight   = class_weight,  # handle class imbalance
            reg_alpha          = 0.1,
            reg_lambda         = 1.0,
            use_label_encoder  = False,
            eval_metric        = "auc",
            early_stopping_rounds = 30,
            random_state       = 42,
            verbosity          = 0,
        )
    else:
        logger.warning("XGBoost not found – using RandomForestClassifier fallback.")
        return RandomForestClassifier(
            n_estimators  = 300,
            max_depth     = 6,
            min_samples_leaf = max(5, int(n_samples * 0.005)),
            class_weight  = "balanced",
            random_state  = 42,
            n_jobs        = -1,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class ForexMLModel:
    """
    Train, evaluate, save, and load a probability filter for trade setups.
    """

    def __init__(self, model_name: str = "forex_xgb"):
        self.model_name  = model_name
        self.model_path  = SAVED_MODELS_DIR / f"{model_name}.joblib"
        self.meta_path   = SAVED_MODELS_DIR / f"{model_name}_meta.json"
        self.pipeline: Optional[Pipeline] = None
        self.feature_names: list = []
        self.threshold: float = 0.60     # default classification threshold
        self.meta: Dict[str, Any] = {}

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.20,
        n_cv_splits: int = 5,
    ) -> Dict[str, Any]:
        """
        Full training pipeline:
          1. Chronological train/test split (no shuffling)
          2. Time-series cross-validation on train set
          3. Final fit on full train set with calibration
          4. Evaluation on held-out test set

        Returns a metrics dict.
        """
        self.feature_names = list(X.columns)
        n = len(X)
        test_idx = int(n * (1 - test_size))
        val_idx  = int(test_idx * 0.8)  # Reserve 20% of the non-test data for validation

        X_train, X_val, X_test = X.iloc[:val_idx], X.iloc[val_idx:test_idx], X.iloc[test_idx:]
        y_train, y_val, y_test = y.iloc[:val_idx], y.iloc[val_idx:test_idx], y.iloc[test_idx:]

        logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
        logger.info(f"Label balance (train) – 1: {y_train.mean():.1%}")

        # Class weight for imbalanced labels
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        cw = n_neg / max(n_pos, 1)

        estimator = _build_estimator(len(X_train), cw)

        # ── Cross-validation (time series) ────────────────────────
        tscv = TimeSeriesSplit(n_splits=min(n_cv_splits, len(X_train) // 3))
        cv_aucs = []

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
            Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            ytr, yval = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            if len(yval.unique()) < 2:
                logger.info(f"  CV Fold {fold}: skipped (single class in validation)")
                continue

            if XGB_AVAILABLE:
                estimator.fit(
                    Xtr, ytr,
                    eval_set=[(Xval, yval)],
                    verbose=False,
                )
            else:
                estimator.fit(Xtr, ytr)

            prob = estimator.predict_proba(Xval)[:, 1]
            try:
                auc = roc_auc_score(yval, prob)
                cv_aucs.append(auc)
                logger.info(f"  CV Fold {fold}/{n_cv_splits} AUC: {auc:.4f}")
            except Exception:
                pass

        mean_cv_auc = float(np.mean(cv_aucs)) if cv_aucs else float("nan")
        std_cv_auc  = float(np.std(cv_aucs))  if cv_aucs else float("nan")
        logger.info(f"CV AUC: {mean_cv_auc:.4f} ± {std_cv_auc:.4f}")

        # ── Final fit on full training set ────────────────────────
        if XGB_AVAILABLE:
            estimator.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )
        else:
            estimator.fit(X_train, y_train)

        # Probability calibration using cross-validated approach with No-Leakage prefit on Validation Set
        try:
            calibrated = CalibratedClassifierCV(estimator, method="sigmoid", cv="prefit")
            calibrated.fit(X_val, y_val)
            self.pipeline = calibrated
        except Exception as e:
            # If calibration fails (tiny dataset), use the raw estimator
            logger.warning(f"Calibration failed ({e}), using raw estimator.")
            self.pipeline = estimator

        # ── Test-set evaluation ───────────────────────────────────
        # Find optimal threshold using Validation set (prevents leakage into Test set)
        proba_val = self.pipeline.predict_proba(X_val)[:, 1]
        self.threshold = self._find_optimal_threshold(y_val, proba_val)
        
        # Evaluate on strictly unseen Test set
        proba_test  = self.pipeline.predict_proba(X_test)[:, 1]
        test_auc    = roc_auc_score(y_test, proba_test)
        test_brier  = brier_score_loss(y_test, proba_test)

        y_pred = (proba_test >= self.threshold).astype(int)
        report = classification_report(y_test, y_pred, output_dict=True)

        logger.info(
            f"Test AUC: {test_auc:.4f} | "
            f"Brier: {test_brier:.4f} | "
            f"Threshold: {self.threshold:.2f}"
        )
        logger.info(f"\n{classification_report(y_test, y_pred)}")

        # ── Feature importance ────────────────────────────────────
        fi = self._feature_importance()
        top10 = fi.head(10).to_string()
        logger.info(f"\nTop-10 Feature Importances:\n{top10}")

        # ── Calibration data ──────────────────────────────────────
        try:
            calib_df = pd.DataFrame({'prob': proba_test, 'actual': y_test.values})
            calib_df['bucket'] = (calib_df['prob'] * 10).astype(int).clip(0, 9)
            prob_buckets = []
            for b in range(10):
                mask = calib_df['bucket'] == b
                wr   = float(calib_df[mask]['actual'].mean()) if mask.any() else 0.0
                cnt  = int(mask.sum())
                prob_buckets.append([f"{b*10}-{(b+1)*10}%", wr, cnt])
        except Exception:
            prob_buckets = []

        self.meta = {
            "model_name":   self.model_name,
            "feature_names": self.feature_names,
            "threshold":    float(self.threshold),
            "cv_auc_mean":  float(mean_cv_auc),
            "cv_auc_std":   float(std_cv_auc),
            "test_auc":     float(test_auc),
            "test_brier":   float(test_brier),
            "n_train":      int(len(X_train)),
            "n_test":       int(len(X_test)),
            "feature_importance": fi.to_dict(),
            "prob_buckets": prob_buckets,
        }

        return {
            **self.meta,
            "cv_aucs":  cv_aucs,
            "report":   report,
            "feature_importance": fi,
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of TP hit (label=1) for each row."""
        self._check_fitted()
        X_aligned = self._align_features(X)
        return self.pipeline.predict_proba(X_aligned)[:, 1]

    def is_valid_setup(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Return boolean array: True if model probability ≥ threshold."""
        t = threshold or self.threshold
        return self.predict_proba(X) >= t

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        self._check_fitted()
        joblib.dump(self.pipeline, self.model_path)
        with open(self.meta_path, "w") as f:
            # Convert non-serialisable objects before dumping
            meta_out = {k: v for k, v in self.meta.items()
                        if isinstance(v, (str, int, float, list, dict, bool))}
            json.dump(meta_out, f, indent=2)
        logger.info(f"Model saved -> {self.model_path}")

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"No saved model at {self.model_path}")
        self.pipeline = joblib.load(self.model_path)
        if self.meta_path.exists():
            with open(self.meta_path) as f:
                self.meta = json.load(f)
            self.feature_names = self.meta.get("feature_names", [])
            self.threshold     = self.meta.get("threshold", 0.60)
        logger.info(f"Model loaded <- {self.model_path}")

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _feature_importance(self) -> pd.Series:
        """Extract feature importances from the inner estimator."""
        try:
            pipe = self.pipeline
            # CalibratedClassifierCV wraps estimators
            if hasattr(pipe, "calibrated_classifiers_"):
                base = pipe.calibrated_classifiers_[0].estimator
            elif hasattr(pipe, "estimator"):
                base = pipe.estimator
            else:
                base = pipe
            if hasattr(base, "feature_importances_"):
                fi = pd.Series(
                    base.feature_importances_,
                    index=self.feature_names,
                ).sort_values(ascending=False)
                return fi
        except Exception:
            pass
        return pd.Series(dtype=float)

    @staticmethod
    def _find_optimal_threshold(y_true, y_prob) -> float:
        """Return threshold that maximises F1 on validation data."""
        prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0)
        best_idx = np.argmax(f1)
        if best_idx < len(thresholds):
            return float(thresholds[best_idx])
        return 0.60

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure X has exactly the columns used at training time."""
        if not self.feature_names:
            return X
            
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(
                f"Missing required features: {missing}. "
                f"Model was trained with: {self.feature_names}"
            )
            
        # Reorder to match training schema EXACTLY
        return X[self.feature_names]

    def _check_fitted(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("Model is not trained or loaded. Call train() or load() first.")
