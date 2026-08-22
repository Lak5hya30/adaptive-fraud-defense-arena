"""Risk-based decision policy — a prototype issuer/network decisioning layer.

Maps a fused risk score to an action instead of a binary block:

    score >= DECLINE threshold  -> DECLINE
    score >= STEP_UP threshold  -> STEP_UP  (3DS challenge / manual review)
    otherwise                   -> APPROVE

Each decision carries short, human-readable reason codes derived from which
rule-style signals fired, so an analyst sees *why*. Thresholds are configurable
in config.DECISION_THRESHOLDS.

NOTE: this is a prototype decisioning layer for simulation, NOT a description of
any production payment-network system.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.defend.baseline import rule_hits

APPROVE, STEP_UP, DECLINE = "APPROVE", "STEP_UP", "DECLINE"

# Human-readable labels for the rule signals used as reason codes.
REASON_LABELS = {
    "high_velocity_1h": "abnormal velocity",
    "very_high_amount_new_payee": "unusually high amount to a new payee",
    "new_device_high_risk_cnp": "new device on risky online spend",
    "impossible_travel": "geographic anomaly",
    "new_account_high_risk_spend": "large risky spend on a new account",
    "high_risk_country": "high-risk geography",
    "structuring": "sub-threshold structuring",
    "amount_zscore_spike": "amount far above cardholder norm",
    "repeat_disputer": "repeat-dispute history",
    "device_shared_across_cards": "device shared across multiple cards",
    "card_device_hopping": "card seen on many devices",
    "merchant_new_card_flood": "merchant seeing almost only new cards",
    "new_payee_large_transfer": "large transfer to a new payee",
}


def resolve_thresholds(thresholds: dict | None = None,
                       step_up: float | None = None) -> dict:
    """Fill in the step-up tier from the model's tuned operating threshold.

    Keeping the step-up boundary tied to the point already chosen against the
    false-positive budget avoids inventing a second, unjustified cutoff.
    """
    t = dict(thresholds or config.DECISION_THRESHOLDS)
    if t.get("step_up") is None:
        t["step_up"] = float(step_up) if step_up is not None else 0.5
    return t


def decide(probabilities: np.ndarray, thresholds: dict | None = None,
           step_up: float | None = None) -> np.ndarray:
    """Vectorized fraud-probability -> action mapping.

    Takes the CALIBRATED probability, not the raw detection score, so the tiers
    are statements about how likely fraud is rather than positions on an
    arbitrary scale.
    """
    t = resolve_thresholds(thresholds, step_up)
    actions = np.full(len(probabilities), APPROVE, dtype=object)
    actions[probabilities >= t["step_up"]] = STEP_UP
    actions[probabilities >= t["decline"]] = DECLINE
    return actions


def reason_codes(X: pd.DataFrame, top: int = 3) -> list[list[str]]:
    """Per-row reason codes from the rules that fired (most specific first)."""
    hits = rule_hits(X)
    out = []
    for _, row in hits.iterrows():
        fired = [REASON_LABELS.get(name, name) for name, v in row.items() if v]
        out.append(fired[:top])
    return out


def decision_frame(df: pd.DataFrame, X: pd.DataFrame, probabilities: np.ndarray,
                   thresholds: dict | None = None,
                   step_up: float | None = None) -> pd.DataFrame:
    """Attach fraud probability, action, and reason codes to a copy of df."""
    actions = decide(probabilities, thresholds, step_up)
    reasons = reason_codes(X)
    out = df.copy()
    out["risk_score"] = probabilities
    out["action"] = actions
    out["reason_codes"] = ["; ".join(r) if r else "-" for r in reasons]
    return out


def policy_summary(probabilities: np.ndarray, y: np.ndarray,
                   thresholds: dict | None = None,
                   step_up: float | None = None) -> dict:
    """How the tiered policy distributes traffic and catches fraud."""
    actions = decide(probabilities, thresholds, step_up)
    legit = y == 0
    fraud = y == 1
    summ = {"thresholds": resolve_thresholds(thresholds, step_up), "tiers": {}}
    for tier in (APPROVE, STEP_UP, DECLINE):
        m = actions == tier
        summ["tiers"][tier] = {
            "share_all": round(float(m.mean()), 4),
            "share_of_legit": round(float((m & legit).sum() / max(1, legit.sum())), 4),
            "share_of_fraud": round(float((m & fraud).sum() / max(1, fraud.sum())), 4),
        }
    # Fraud auto-declined; fraud sent to friction (step-up+decline); legit auto-approved.
    friction = np.isin(actions, [STEP_UP, DECLINE])
    summ["fraud_declined"] = round(float((actions[fraud] == DECLINE).mean()), 4)
    summ["fraud_to_friction"] = round(float(friction[fraud].mean()), 4)
    summ["legit_auto_approved"] = round(float((actions[legit] == APPROVE).mean()), 4)
    summ["legit_hard_declined"] = round(float((actions[legit] == DECLINE).mean()), 4)
    return summ


if __name__ == "__main__":
    import json

    from src.defend.features import build_xy
    from src.defend.model import DefenseModel

    df = pd.read_parquet(config.DATASET_PARQUET)
    X, y, _ = build_xy(df)
    model = DefenseModel.load()
    probs = model.risk_probability(X)
    summ = policy_summary(probs, y, step_up=model.threshold_probability)
    print(json.dumps(summ, indent=2))
