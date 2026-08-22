"""Phase 1 tests: authorization-time leakage, legitimate-data realism, and
temporal causality of historical features.

    python -m pytest tests/test_data_quality.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from src.defend.features import (POST_OUTCOME_COLUMNS, assert_auth_time_safe,
                                 build_features, build_xy)
from src.generate.simulate import simulate


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #
def test_post_outcome_fields_never_reach_model():
    df, _ = simulate(n_transactions=6000, fraud_rate=0.02, seed=3)
    X = build_features(df)
    for col in POST_OUTCOME_COLUMNS:
        assert col not in X.columns, f"{col} leaked into the feature matrix"


def test_assert_guard_catches_injected_leakage():
    df, _ = simulate(n_transactions=3000, fraud_rate=0.02, seed=3)
    X = build_features(df)
    X_bad = X.copy()
    X_bad["refund_flag"] = 1  # simulate an accidental leak
    with pytest.raises(ValueError):
        assert_auth_time_safe(X_bad)


def test_refund_flag_still_present_in_raw_dataset():
    # It must survive in the raw data for post-hoc analysis, just not as a feature.
    df, _ = simulate(n_transactions=2000, fraud_rate=0.02, seed=3)
    assert "refund_flag" in df.columns


# --------------------------------------------------------------------------- #
# Legitimate-data realism  (the anti-shortcut checks)
# --------------------------------------------------------------------------- #
def test_realism_bundle():
    df, _ = simulate(n_transactions=20000, fraud_rate=0.02, seed=7)
    legit = df[df.is_fraud == 0]
    fraud = df[df.is_fraud == 1]

    # multiple devices per legitimate card
    dev_per_card = legit.groupby("card_id").device_id.nunique()
    assert (dev_per_card > 1).sum() > 0, "no legit card uses >1 device"

    # genuine new payees among legit (so is_new_payee != fraud)
    assert legit.is_new_payee.sum() > 0, "no legit transaction is a new payee"
    assert 0.05 < legit.is_new_payee.mean() < 0.98, "legit new-payee rate degenerate"

    # legit high-value purchases exist
    assert (legit.amount > legit.amount.quantile(0.999)).sum() >= 0
    assert legit.amount.max() > 5 * legit.amount.median(), "no genuine large purchases"

    # legit foreign / unusual-hour activity exists
    assert (legit.country != "IN").sum() > 0, "no legit foreign activity"
    assert (pd.to_datetime(legit.timestamp).dt.hour < 6).sum() > 0, "no legit night activity"

    # thin-file genuine customers exist
    assert (legit.account_age_days < 90).sum() > 0, "no thin-file legit customers"


def test_device_and_payee_are_not_perfect_shortcuts():
    """The whole point of Phase 1: neither device change nor new-payee may
    perfectly separate fraud from legit."""
    df, _ = simulate(n_transactions=20000, fraud_rate=0.02, seed=7)
    X, y, _ = build_xy(df)
    for feat in ["is_new_payee", "device_changed"]:
        legit_pos = X.loc[y == 0, feat].mean()
        assert legit_pos > 0.02, f"{feat} is ~0 for legit -> perfect shortcut"


def test_fraud_and_legit_feature_distributions_overlap():
    df, _ = simulate(n_transactions=20000, fraud_rate=0.02, seed=7)
    X, y, _ = build_xy(df)
    # log_amount ranges must overlap substantially (not cleanly separable).
    lo_l, hi_l = np.quantile(X.loc[y == 0, "log_amount"], [0.05, 0.95])
    lo_f, hi_f = np.quantile(X.loc[y == 1, "log_amount"], [0.05, 0.95])
    overlap = min(hi_l, hi_f) - max(lo_l, lo_f)
    assert overlap > 0, "legit and fraud amount distributions do not overlap"


# --------------------------------------------------------------------------- #
# Temporal causality
# --------------------------------------------------------------------------- #
def test_historical_features_use_only_the_past():
    """A row's historical features must not change when FUTURE rows are added.
    Compute features on a card's first k rows, then on all its rows, and confirm
    the first k rows are identical."""
    df, _ = simulate(n_transactions=15000, fraud_rate=0.02, seed=5)
    df = df.sort_values("timestamp").reset_index(drop=True)
    # pick a legit card with plenty of history
    counts = df[df.is_fraud == 0].card_id.value_counts()
    card = counts.index[0]
    card_rows = df[df.card_id == card].sort_values("timestamp")
    assert len(card_rows) >= 6

    hist_feats = ["amount_zscore", "velocity_1h", "velocity_24h", "device_changed",
                  "card_history_depth", "card_prior_dispute_rate", "time_since_last_hours"]
    full = build_features(card_rows)[hist_feats].reset_index(drop=True)
    prefix = build_features(card_rows.iloc[:4])[hist_feats].reset_index(drop=True)

    pd.testing.assert_frame_equal(full.iloc[:4], prefix, check_dtype=False,
                                  rtol=1e-9, atol=1e-9)


def test_first_transaction_history_is_neutral():
    df, _ = simulate(n_transactions=8000, fraud_rate=0.02, seed=5)
    X = build_features(df.sort_values("timestamp"))
    df_sorted = df.sort_values("timestamp")
    first_idx = df_sorted.groupby("card_id").head(1).index
    # first-ever txn per card has no prior history -> neutral defaults
    assert (X.loc[first_idx, "amount_zscore"] == 0.0).all()
    assert (X.loc[first_idx, "card_prior_dispute_rate"] == 0.0).all()
    assert (X.loc[first_idx, "card_history_depth"] == 0.0).all()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nAll data-quality tests passed.")
