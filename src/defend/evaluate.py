"""Evaluation metrics for the defense model.

Reports the headline classification metrics plus the two things this challenge
cares about specifically: per-attack-type recall (does the defense catch every
family?) and the false-positive rate on legitimate traffic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

import config
from src.defend.model import DefenseModel


def evaluate(model: DefenseModel, X: pd.DataFrame, y: np.ndarray,
             attack_type: np.ndarray, threshold: float | None = None) -> dict:
    scores = model.fused_scores(X)
    t = model.threshold if threshold is None else threshold
    pred = (scores >= t).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    legit_mask = y == 0
    fpr = float(fp / max(1, legit_mask.sum()))

    metrics = {
        "threshold": float(t),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else None,
        "pr_auc": float(average_precision_score(y, scores)) if len(np.unique(y)) > 1 else None,
        "false_positive_rate": fpr,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_eval": int(len(y)),
        "n_fraud_eval": int(y.sum()),
    }
    metrics["per_attack_recall"] = _per_attack_recall(y, pred, attack_type)
    return metrics


def wilson_interval(p: float | None, n: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson score interval for a proportion.

    Family-level recall is a proportion measured on a handful of transactions.
    Quoting "1.00" from nine examples invites exactly one question, and the
    interval answers it before it is asked.
    """
    if p is None or n <= 0:
        return None
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / denom
    return [round(float(max(0.0, centre - half)), 4), round(float(min(1.0, centre + half)), 4)]


def _per_attack_recall(y, pred, attack_type, min_n: int | None = None) -> dict:
    out = {}
    at = np.asarray(attack_type)
    floor = min_n if min_n is not None else config.FAMILY_EVAL["min_n_to_report"]
    for a in sorted(set(at)):
        if a == "legit":
            continue
        mask = (at == a) & (y == 1)
        n = int(mask.sum())
        if n == 0:
            continue
        r = float(pred[mask].mean())
        out[a] = {"recall": r, "n": n, "ci95": wilson_interval(r, n),
                  "sufficient_n": bool(n >= floor)}
    return out


def scored_frame(model: DefenseModel, df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Attach component + fused scores and the flag decision to a copy of df."""
    comp = model.component_scores(X)
    out = df.copy()
    out["score_supervised"] = comp["supervised"]
    out["score_anomaly"] = comp["anomaly"]
    out["risk_score"] = comp["fused"]
    out["flagged"] = (comp["fused"] >= model.threshold).astype(int)
    return out
