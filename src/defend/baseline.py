"""Transparent rule-based fraud baseline + a 3-way comparison harness.

The rules baseline lets judges see whether the ML actually adds value. It is a
small set of intuitive, *untuned* card-fraud rules operating on the same
authorization-time features as the model. The harness compares:

    1. Rules baseline
    2. Static ML model (the trained DefenseModel)
    3. Adaptive ML (loaded from the closed-loop artifacts, when available)

across the overall dataset and per attack family.

CLI:  python -m src.defend.baseline
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

import config
from src.defend.features import build_xy
from src.defend.model import DefenseModel


# --------------------------------------------------------------------------- #
# Rule-based detector
# --------------------------------------------------------------------------- #
# A competent, transparent rule set — not a strawman. It includes the
# network-level rules a real fraud team would write once it had the same
# counters the model gets, so the comparison measures what machine learning adds
# on top of good rules, not what it adds on top of a deliberately weak baseline.
# The rules are intentionally UNTUNED: thresholds are round, intuitive numbers
# chosen from domain reasoning, never fitted to this dataset.
RULES = [
    ("high_velocity_1h", lambda X: X["velocity_1h"] >= 4),
    ("very_high_amount_new_payee", lambda X: (X["log_amount"] >= np.log1p(60000)) &
        (X["is_new_payee"] == 1)),
    ("new_device_high_risk_cnp", lambda X: (X["device_changed"] == 1) &
        (X["is_cnp"] == 1) & (X["mcc_risk"] >= 0.7)),
    # Impossible travel means far away SOON AFTER being somewhere else — distance
    # alone just flags people who travel.
    ("impossible_travel", lambda X: (X["log_distance"] >= np.log1p(1500)) &
        (X["is_foreign"] == 1) & (X["time_since_last_hours"] <= 6)),
    ("new_account_high_risk_spend", lambda X: (X["is_new_account"] == 1) &
        (X["log_amount"] >= np.log1p(15000)) & (X["mcc_risk"] >= 0.6)),
    ("high_risk_country", lambda X: X["is_high_risk_country"] == 1),
    ("structuring", lambda X: (X["near_threshold"] == 1) & (X["velocity_24h"] >= 4)),
    ("amount_zscore_spike", lambda X: X["amount_zscore"] >= 6),
    ("repeat_disputer", lambda X: X["card_prior_dispute_rate"] >= 0.34),
    # --- network-level rules ------------------------------------------------
    # The relational counters reach the model log-compressed, so the rule
    # thresholds are written as log1p of the count a fraud analyst would state.
    ("device_shared_across_cards", lambda X: X["device_card_count_prior"] >= np.log1p(3)),
    ("card_device_hopping", lambda X: X["card_device_count_prior"] >= np.log1p(6)),
    ("merchant_new_card_flood", lambda X: (X["merchant_new_card_ratio_prior"] >= 0.9) &
        (X["merchant_txn_count_prior"] >= np.log1p(50))),
    ("new_payee_large_transfer", lambda X: (X["is_new_payee"] == 1) &
        (X["mcc_risk"] >= 0.85) & (X["log_amount"] >= np.log1p(40000))),
]


def rule_hits(X: pd.DataFrame) -> pd.DataFrame:
    """Boolean matrix of which rule fired for each row (for reason codes)."""
    return pd.DataFrame({name: fn(X).astype(int) for name, fn in RULES}, index=X.index)


def rules_predict(X: pd.DataFrame, min_rules: int = 1) -> np.ndarray:
    """Flag a transaction when at least `min_rules` rules fire."""
    return (rule_hits(X).sum(axis=1) >= min_rules).astype(int).to_numpy()


def rules_score(X: pd.DataFrame) -> np.ndarray:
    """A crude 0..1 'risk' from the fraction of rules that fire (for ROC/PR)."""
    return (rule_hits(X).sum(axis=1) / len(RULES)).to_numpy()


# --------------------------------------------------------------------------- #
# Metrics + comparison
# --------------------------------------------------------------------------- #
def _metrics(y, pred, score=None, attack=None) -> dict:
    legit = y == 0
    fp = int(((pred == 1) & legit).sum())
    out = {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "false_positive_rate": float(fp / max(1, legit.sum())),
        "n_flagged": int(pred.sum()),
        "fraud_caught_per_1000_legit_hit": None,
    }
    out["false_positives_per_1000_legit"] = round(out["false_positive_rate"] * 1000, 2)
    out["fraud_caught_per_1000_fraud"] = round(out["recall"] * 1000, 1)
    if fp > 0:
        out["fraud_caught_per_1000_legit_hit"] = round(
            float(((pred == 1) & (y == 1)).sum()) / fp * 1000, 1)
    if score is not None and len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, score))
        out["pr_auc"] = float(average_precision_score(y, score))
    if attack is not None:
        per = {}
        for a in sorted(set(attack)):
            if a == "legit":
                continue
            m = (attack == a) & (y == 1)
            if m.sum():
                per[a] = {"recall": round(float(pred[m].mean()), 4), "n": int(m.sum())}
        out["per_attack_recall"] = per
    return out


def _threshold_for_fpr(score: np.ndarray, y: np.ndarray, target_fpr: float) -> float:
    """Lowest threshold whose realized false-positive rate stays within budget."""
    legit = np.sort(score[y == 0])
    n = len(legit)
    if n == 0:
        return float(np.max(score)) + 1e-9
    cand = np.unique(legit)
    fprs = (n - np.searchsorted(legit, cand, side="left")) / n
    ok = np.flatnonzero(fprs <= target_fpr)
    return float(cand[ok[0]]) if len(ok) else float(cand[-1]) + 1e-9


def compare(df: pd.DataFrame | None = None, adaptive_model: DefenseModel | None = None,
            models: dict | None = None, on_test_split: bool = True,
            match_fpr: bool = True) -> dict:
    """Rules vs static ML vs adaptive ML, at their native operating points AND at
    a matched false-positive rate.

    Comparing detectors at whatever operating point each happens to sit on is
    close to meaningless: any of them can buy recall by challenging more genuine
    customers. The matched-rate view re-thresholds every model to spend the SAME
    false-positive budget on the same test split, which is the comparison a fraud
    team would actually run before choosing one.

    Evaluated on the held-out TIME-SPLIT TEST set by default, so the static-ML
    numbers match the honest reported metrics (no scoring on training rows).
    """
    if df is None:
        df = pd.read_parquet(config.DATASET_PARQUET)
    X_val = y_val = None
    if on_test_split:
        # Same global-feature-build-then-split contract the trainer uses, so the
        # static-ML numbers here are the same numbers reported in metrics.json.
        from src.defend.train import split_xy
        parts = split_xy(df)
        X, y, attack = parts["test"]
        X_val, y_val, _ = parts["val"]
    else:
        X, y, attack = build_xy(df.sort_values("timestamp").reset_index(drop=True))

    scored: dict[str, np.ndarray] = {"rules_baseline": rules_score(X)}
    preds: dict[str, np.ndarray] = {"rules_baseline": rules_predict(X)}

    if models is None:
        models = {}
        if config.MODEL_PATH.exists():
            models["static_ml"] = DefenseModel.load()
        if adaptive_model is not None:
            models["adaptive_ml"] = adaptive_model
    for name, m in models.items():
        s = m.fused_scores(X)
        scored[name] = s
        preds[name] = (s >= m.threshold).astype(int)

    result = {name: _metrics(y, preds[name], scored[name], attack) for name in preds}

    if match_fpr:
        budget = config.TARGET_MAX_FPR
        matched = {}
        # Matched thresholds are selected on the VALIDATION slice and then applied
        # unchanged to the test slice. Choosing them on the test set's own labels
        # would let every detector peek at the answer, and the comparison would
        # flatter whichever one happens to have the most exploitable score
        # distribution there.
        for name, s in scored.items():
            if X_val is not None:
                s_val = (rules_score(X_val) if name == "rules_baseline"
                         else models[name].fused_scores(X_val))
                t = _threshold_for_fpr(s_val, y_val, budget)
            else:
                t = _threshold_for_fpr(s, y, budget)
            p = (s >= t).astype(int)
            m = _metrics(y, p, s, attack)
            m["threshold"] = round(float(t), 6)
            matched[name] = m
        result["_fpr_matched"] = {
            "target_false_positive_rate": budget,
            "threshold_selected_on": "validation slice" if X_val is not None else "same split",
            "note": ("Every detector re-thresholded to spend the same false-positive budget, "
                     "with the threshold chosen on the validation slice and applied unchanged "
                     "to the held-out test slice. This is the like-for-like comparison; the "
                     "blocks above are each detector at its own native operating point. "
                     "Realized test false-positive rates will differ slightly from the budget "
                     "because the threshold was not fitted to the test set."),
            "models": matched,
        }
    result["_meta"] = {
        "schema_version": config.SCHEMA_VERSION,
        "n_eval": int(len(y)), "n_fraud_eval": int(y.sum()),
        "evaluated_on": "held-out time-split test set" if on_test_split else "full dataset",
    }
    return result


def main():
    # Ship the model governance actually approved. The last CANDIDATE the loop
    # produced is reported alongside it, labelled, so a rejected challenger is
    # never presented as the working defense.
    models = {}
    if config.MODEL_PATH.exists():
        models["static_ml"] = DefenseModel.load()
    if config.LOOP_CHAMPION_MODEL_PATH.exists():
        models["adaptive_ml"] = DefenseModel.load(config.LOOP_CHAMPION_MODEL_PATH)
    if config.LOOP_ADAPTED_MODEL_PATH.exists():
        models["adaptive_candidate_unpromoted"] = DefenseModel.load(
            config.LOOP_ADAPTED_MODEL_PATH)
    res = compare(models=models)
    config.BASELINE_COMPARISON_PATH.write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    for name, m in res.items():
        if name.startswith("_"):
            continue
        line = (f"{name:32s} recall={m['recall']:.3f} precision={m['precision']:.3f} "
                f"F1={m['f1']:.3f} FPR={m['false_positive_rate']:.3f}")
        if "pr_auc" in m:
            line += f" PR-AUC={m['pr_auc']:.3f}"
        print(line)
    if "_fpr_matched" in res:
        print(f"\nAt a matched {res['_fpr_matched']['target_false_positive_rate']:.1%} "
              "false-positive budget:")
        for name, m in res["_fpr_matched"]["models"].items():
            print(f"  {name:32s} recall={m['recall']:.3f} "
                  f"precision={m['precision']:.3f} FPR={m['false_positive_rate']:.4f}")
    print(f"\nSaved {config.MODELS_DIR / 'baseline_comparison.json'}")


if __name__ == "__main__":
    main()
