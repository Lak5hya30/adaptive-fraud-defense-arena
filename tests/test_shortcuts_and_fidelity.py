"""Guards against the failure mode that quietly destroys a fraud-simulation
project: the detector learning the generator instead of the fraud.

Two real shortcuts existed in this simulator and were removed. Both have a test
here so they cannot come back:

  * ``is_new_payee`` once fired on 99.9% of fraud versus 46% of genuine traffic,
    because every fraudulent row happened to be a first-ever card/merchant pair.
  * The velocity features were dead — identically 1.0 on the two families whose
    entire definition is burst behaviour — because each probing transaction used
    a freshly minted card id.

The tests are two-sided on purpose. Asserting only that a feature is non-zero on
genuine traffic is what let the first shortcut survive.

    python -m pytest tests/test_shortcuts_and_fidelity.py -q
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
from sklearn.metrics import roc_auc_score

import config
from src.defend.features import (FEATURE_COLUMNS, NETWORK_FEATURES, ORACLE_COLUMNS,
                                 POST_OUTCOME_COLUMNS, build_features, build_xy)
from src.generate.fidelity import SEPARABILITY_FAIL, quality_checks, separability
from src.generate.simulate import simulate

N = 26_000
SEED = 21


@pytest.fixture(scope="module")
def frame():
    df, _ = simulate(n_transactions=N, fraud_rate=0.02, seed=SEED)
    df = df.sort_values("timestamp").reset_index(drop=True)
    X, y, atk = build_xy(df)
    return df, X, y, atk


# No single feature may separate the classes
def test_no_single_feature_separates_fraud_from_genuine(frame):
    _, X, y, _ = frame
    worst = separability(X, y)[0]
    assert worst["auc"] < SEPARABILITY_FAIL, (
        f"{worst['feature']} ranks fraud above genuine traffic at AUC {worst['auc']:.3f} — "
        "that is a generator artifact, not a fraud signal")


def test_binary_features_are_not_one_sided_shortcuts(frame):
    """Both directions. A flag that is ~always true on fraud is as much of a
    shortcut as one that is ~never true on genuine traffic."""
    _, X, y, _ = frame
    binary = [c for c in FEATURE_COLUMNS if set(np.unique(X[c].dropna())) <= {0.0, 1.0}]
    assert binary, "expected some binary features"
    for c in binary:
        legit_rate = float(X.loc[y == 0, c].mean())
        fraud_rate = float(X.loc[y == 1, c].mean())
        assert not (fraud_rate > 0.95 and legit_rate < 0.60), (
            f"{c}: fires on {fraud_rate:.1%} of fraud vs {legit_rate:.1%} of genuine "
            "traffic — a near-perfect shortcut")


def test_new_payee_is_a_signal_not_a_label(frame):
    _, X, y, _ = frame
    legit_rate = float(X.loc[y == 0, "is_new_payee"].mean())
    fraud_rate = float(X.loc[y == 1, "is_new_payee"].mean())
    assert 0.05 < legit_rate < 0.95
    assert fraud_rate < 0.95, (
        f"is_new_payee fires on {fraud_rate:.1%} of fraud — attacks must reuse merchants")


def test_velocity_features_are_alive_for_burst_families(frame):
    """card_testing and velocity_smurfing are defined by burst behaviour. If their
    velocity features sit at 1.0 the simulator is not producing the pattern its
    own family names claim."""
    _, X, y, atk = frame
    for fam in ("card_testing", "velocity_smurfing"):
        m = (atk == fam) & (y == 1)
        if m.sum() < 10:
            continue
        legit_mean = float(X.loc[y == 0, "velocity_24h"].mean())
        fam_mean = float(X.loc[m, "velocity_24h"].mean())
        assert fam_mean > legit_mean, (
            f"{fam} shows velocity_24h {fam_mean:.2f} vs {legit_mean:.2f} on genuine "
            "traffic — the burst signature is not being generated")


def test_absence_of_history_is_not_a_synonym_for_fraud(frame):
    _, X, y, _ = frame
    d0 = X["card_history_depth"] == 0
    legit_share = float(d0[y == 0].mean())
    fraud_share = float(d0[y == 1].mean())
    assert legit_share > 0.02, "no genuine first-ever transactions in the portfolio"
    assert fraud_share / max(legit_share, 1e-9) < 6.0, (
        f"first-ever transactions are {fraud_share:.1%} of fraud vs {legit_share:.1%} of "
        "genuine traffic — 'unknown card' is standing in for 'fraud'")


def test_fraud_sometimes_looks_entirely_ordinary(frame):
    df, X, y, _ = frame
    known_device = float((X.loc[y == 1, "device_changed"] == 0).mean())
    assert known_device > 0.10, "no fraud runs from a device the card already used"
    legit_amounts = df.loc[df.is_fraud == 0, "amount"]
    lo, hi = np.quantile(legit_amounts, [0.25, 0.75])
    in_range = float(((df.loc[df.is_fraud == 1, "amount"] >= lo) &
                      (df.loc[df.is_fraud == 1, "amount"] <= hi)).mean())
    assert in_range > 0.10, "no fraud sits inside the ordinary amount range"


def test_fraud_actors_generate_cover_traffic(frame):
    df, _, _, _ = frame
    cover = int((df.actor_role == "fraud_actor_cover").sum())
    assert cover > 0, "mule and bust-out accounts must build history before being used"
    assert (df.loc[df.actor_role == "fraud_actor_cover", "is_fraud"] == 0).all(), (
        "cover traffic is legitimate at authorization time and must be labelled so")


def test_all_fidelity_checks_pass(frame):
    df, X, y, _ = frame
    checks = quality_checks(df, X, y, separability(X, y))
    failing = [c for c in checks if c["status"] == "FAIL"]
    assert not failing, f"failing fidelity checks: {[c['check'] for c in failing]}"


# Leakage
def test_post_outcome_and_oracle_columns_never_reach_the_model(frame):
    _, X, _, _ = frame
    for col in POST_OUTCOME_COLUMNS + ORACLE_COLUMNS:
        assert col not in X.columns, f"{col} leaked into the feature matrix"


def test_the_leakage_guard_actually_fires():
    from src.defend.features import assert_auth_time_safe
    df, _ = simulate(n_transactions=3000, fraud_rate=0.02, seed=3)
    X = build_features(df)
    for bad in ("refund_flag", "attack_type"):
        X_bad = X.copy()
        X_bad[bad] = 1
        with pytest.raises(ValueError):
            assert_auth_time_safe(X_bad)


# Temporal causality, including the network counters
def test_network_counters_only_look_backwards():
    """A row's features must not change when LATER rows are appended. The
    relational counters are the easy place to get this wrong, because they are
    computed across cards rather than within one."""
    df, _ = simulate(n_transactions=9000, fraud_rate=0.02, seed=5)
    df = df.sort_values("timestamp").reset_index(drop=True)
    prefix_len = 4000
    head = df.index[:prefix_len]
    # build_features returns rows in card-sorted order, so reindex to the original
    # chronological order before comparing — otherwise this compares different rows.
    full = (build_features(df).reindex(df.index)
            .loc[head, NETWORK_FEATURES].reset_index(drop=True))
    prefix = (build_features(df.iloc[:prefix_len]).reindex(head)[NETWORK_FEATURES]
              .reset_index(drop=True))
    pd.testing.assert_frame_equal(full, prefix, check_dtype=False, rtol=1e-9, atol=1e-9)


def test_first_transaction_history_is_neutral():
    df, _ = simulate(n_transactions=8000, fraud_rate=0.02, seed=5)
    df_sorted = df.sort_values("timestamp")
    X = build_features(df_sorted)
    first_idx = df_sorted.groupby("card_id").head(1).index
    assert (X.loc[first_idx, "amount_zscore"] == 0.0).all()
    assert (X.loc[first_idx, "card_prior_dispute_rate"] == 0.0).all()
    assert (X.loc[first_idx, "card_history_depth"] == 0.0).all()
    assert (X.loc[first_idx, "amount_vs_card_max_prior"] == 0.0).all()


# Reproducibility and configurability
def test_simulation_is_reproducible():
    a, sa = simulate(n_transactions=4000, fraud_rate=0.02, seed=99)
    b, sb = simulate(n_transactions=4000, fraud_rate=0.02, seed=99)
    pd.testing.assert_frame_equal(a, b)
    assert sa["n_fraud"] == sb["n_fraud"]


def test_a_different_seed_gives_a_different_portfolio():
    a, _ = simulate(n_transactions=4000, fraud_rate=0.02, seed=1)
    b, _ = simulate(n_transactions=4000, fraud_rate=0.02, seed=2)
    assert not a["card_id"].equals(b["card_id"])


@pytest.mark.parametrize("rate", [0.005, 0.03])
def test_fraud_rate_is_configurable_and_honoured(rate):
    df, summary = simulate(n_transactions=8000, fraud_rate=rate, seed=7)
    assert abs(summary["fraud_rate"] - rate) < max(0.004, rate * 0.35)


def test_portfolio_contains_every_customer_archetype():
    _, summary = simulate(n_transactions=6000, fraud_rate=0.02, seed=7)
    present = set(summary["customer_archetypes"])
    assert present == set(config.CUSTOMER_ARCHETYPES), (
        f"missing archetypes: {sorted(set(config.CUSTOMER_ARCHETYPES) - present)}")


def test_legit_volume_does_not_drift_across_the_window():
    """Card issuance during the window pushes genuine volume later; attrition
    balances it. Without that balance the fraud RATE drifts across the window and
    a chronological split inherits a different base rate from its training data."""
    df, _ = simulate(n_transactions=30000, fraud_rate=0.02, seed=11)
    df = df.sort_values("timestamp").reset_index(drop=True)
    bounds = np.linspace(0, len(df), 5).astype(int)
    rates = [float(df.is_fraud.iloc[lo:hi].mean())
             for lo, hi in zip(bounds[:-1], bounds[1:])]
    assert max(rates) / max(min(rates), 1e-9) < 2.5, (
        f"fraud rate drifts across the window: {[round(r, 4) for r in rates]}")
