"""DefenseModel: supervised gradient boosting + unsupervised anomaly detector
with score fusion, plus a tuned decision threshold.

The fusion matters for the challenge's "novel attacks" framing: the supervised
head catches known patterns; the Isolation Forest (fit on legit only) still
flags out-of-distribution transactions from attack families it never saw during
supervised training. The tuned threshold keeps the legit false-positive rate at
or below the configured target.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

import config
from src.defend.features import FEATURE_COLUMNS


@dataclass
class DefenseModel:
    supervised: HistGradientBoostingClassifier | None = None
    anomaly: IsolationForest | None = None
    scaler: StandardScaler | None = None
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    anomaly_min: float = 0.0
    anomaly_max: float = 1.0
    threshold: float = 0.5
    fusion_weight: float = 0.15  # weight on the anomaly signal
    # Isotonic calibration of the fused score onto an observed fraud rate. Fitted
    # on a held-out slice. Isotonic is monotone, so calibration cannot change the
    # model's ranking: ROC-AUC, PR-AUC, and any quantile-selected threshold behave
    # exactly as before. What changes is that a displayed score of 0.30 now means
    # "about 30% of transactions scoring here were fraudulent" instead of being an
    # arbitrary blend of two heads.
    calibrator: IsotonicRegression | None = None

    # -- training ---------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.ndarray,
            sample_weight: np.ndarray | None = None) -> "DefenseModel":
        """Fit both heads. ``sample_weight`` lets the adaptive loop emphasise
        replayed adversarial examples without physically duplicating rows, which
        keeps the training frame small and the emphasis explicit."""
        X = X[self.feature_columns]
        self.supervised = HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.08, max_iter=350,
            l2_regularization=1.0, class_weight="balanced",
            random_state=config.GLOBAL_SEED,
        ).fit(X, y, sample_weight=sample_weight)

        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        # Fit anomaly detector on legit only => better novelty detection.
        legit = Xs[y == 0]
        contamination = float(min(0.05, max(0.005, y.mean())))
        self.anomaly = IsolationForest(
            n_estimators=200, contamination=contamination,
            random_state=config.GLOBAL_SEED,
        ).fit(legit if len(legit) else Xs)

        raw = -self.anomaly.decision_function(Xs)  # higher => more anomalous
        self.anomaly_min = float(raw.min())
        self.anomaly_max = float(raw.max())
        return self

    # -- scoring ----------------------------------------------------------
    def _anomaly_norm(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self.scaler.transform(X[self.feature_columns])
        raw = -self.anomaly.decision_function(Xs)
        span = (self.anomaly_max - self.anomaly_min) or 1.0
        return np.clip((raw - self.anomaly_min) / span, 0.0, 1.0)

    def fused_scores(self, X: pd.DataFrame) -> np.ndarray:
        """The DETECTION score: the blend of the supervised and anomaly heads.

        Deliberately the uncalibrated fusion. It is continuous and fine-grained,
        which is what the operating threshold needs: a calibrated score is a step
        function, and thresholding a step function cannot hit a false-positive
        budget precisely — the first tied block it admits overshoots. Calibration
        is monotone, so ranking metrics (ROC-AUC, PR-AUC) are identical either way.
        """
        p_sup = self.supervised.predict_proba(X[self.feature_columns])[:, 1]
        a = self._anomaly_norm(X)
        w = self.fusion_weight
        return np.clip((1 - w) * p_sup + w * a, 0.0, 1.0)

    # Backwards-compatible alias for the raw fusion.
    raw_fused_scores = fused_scores

    def risk_probability(self, X: pd.DataFrame) -> np.ndarray:
        """The DISPLAYED score: an estimated probability that this authorization
        is fraudulent.

        This is what the decision policy tiers on and what a reviewer sees, so
        the number has to mean something: 0.30 should mean roughly three in ten
        transactions scoring here were fraudulent. Without calibration the fused
        blend is on no meaningful scale at all.
        """
        raw = self.fused_scores(X)
        if self.calibrator is None:
            return raw
        return np.clip(self.calibrator.predict(raw), 0.0, 1.0)

    @property
    def threshold_probability(self) -> float:
        """The operating threshold expressed as a fraud probability."""
        if self.calibrator is None:
            return float(self.threshold)
        return float(np.clip(self.calibrator.predict([self.threshold])[0], 0.0, 1.0))

    def fit_calibration(self, X: pd.DataFrame, y: np.ndarray) -> "DefenseModel":
        """Fit isotonic calibration on a held-out slice (never on training rows)."""
        if len(np.unique(y)) < 2:
            return self
        raw = self.fused_scores(X)
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.calibrator.fit(raw, y)
        return self

    def component_scores(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        return {
            "supervised": self.supervised.predict_proba(X[self.feature_columns])[:, 1],
            "anomaly": self._anomaly_norm(X),
            "fused": self.fused_scores(X),
            "probability": self.risk_probability(X),
        }

    def predict(self, X: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        t = self.threshold if threshold is None else threshold
        return (self.fused_scores(X) >= t).astype(int)

    # -- threshold tuning -------------------------------------------------
    def tune_threshold(self, X: pd.DataFrame, y: np.ndarray,
                       target_fpr: float = config.TARGET_MAX_FPR) -> float:
        """Pick the lowest threshold whose legit false-positive rate <= target
        (maximizes recall subject to the FP budget).

        The threshold is chosen by evaluating the REALIZED false-positive rate at
        each distinct score, not by taking a quantile. Calibrated scores are a
        step function, so many genuine transactions share an identical score; a
        quantile lands inside one of those tied blocks and the ``>=`` comparison
        then admits the whole block, quietly spending several times the agreed
        budget. Searching realized rates cannot do that.
        """
        scores = self.fused_scores(X)
        legit_scores = np.sort(scores[y == 0])
        n = len(legit_scores)
        if n == 0:
            self.threshold = 0.5
            return self.threshold
        candidates = np.unique(legit_scores)
        # FPR when the threshold is c: the share of genuine scores at or above c.
        fprs = (n - np.searchsorted(legit_scores, candidates, side="left")) / n
        ok = np.flatnonzero(fprs <= target_fpr)
        if len(ok):
            self.threshold = float(candidates[ok[0]])
        else:
            # Even the very top score exceeds the budget (a degenerate model);
            # sit just above everything rather than flag the whole book.
            self.threshold = float(candidates[-1]) + 1e-9
        self.threshold = float(min(max(self.threshold, 1e-6), 1.0))
        return self.threshold

    # -- persistence ------------------------------------------------------
    def save(self, path=config.MODEL_PATH):
        joblib.dump(self, path)

    @staticmethod
    def load(path=config.MODEL_PATH) -> "DefenseModel":
        return joblib.load(path)
