"""Phase 2/3 tests: rules baseline, tiered decision policy, and the cumulative
adaptive loop (cumulative replay, no catastrophic forgetting, metric-derived
status). Also reproducibility of the simulator.

    python -m pytest tests/test_defense_and_loop.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import config
from src.defend.baseline import rules_predict, rules_score
from src.defend.decision_policy import (APPROVE, DECLINE, STEP_UP, decide,
                                        policy_summary, resolve_thresholds)
from src.defend.features import build_xy
from src.generate.simulate import simulate
from src.loop.redteam_loop import _bounded_append, _focus_mix, _status, run_loop


# Rules baseline
def test_rules_produce_valid_predictions():
    df, _ = simulate(n_transactions=4000, fraud_rate=0.02, seed=3)
    X, y, _ = build_xy(df)
    pred = rules_predict(X)
    assert set(np.unique(pred)).issubset({0, 1})
    assert len(pred) == len(X)
    assert (rules_score(X) <= 1.0).all() and (rules_score(X) >= 0.0).all()


# Tiered decision policy
def test_thresholds_map_to_actions():
    """The step-up tier defaults to the model's own tuned operating threshold, so
    the policy cannot drift away from the point chosen against the FP budget."""
    step_up = 0.2
    t = resolve_thresholds(step_up=step_up)
    assert t["step_up"] == step_up
    probs = np.array([0.0, t["step_up"] - 1e-6, t["step_up"],
                      t["decline"] - 1e-6, t["decline"], 1.0])
    actions = decide(probs, step_up=step_up)
    assert list(actions) == [APPROVE, APPROVE, STEP_UP, STEP_UP, DECLINE, DECLINE]


def test_decline_tier_is_stricter_than_step_up():
    t = resolve_thresholds(step_up=0.2)
    assert t["decline"] > t["step_up"], "declining must require more confidence than a challenge"


def test_policy_summary_partitions_traffic():
    probs = np.linspace(0, 1, 1000)
    y = (probs > 0.5).astype(int)
    summ = policy_summary(probs, y, step_up=0.3)
    shares = [summ["tiers"][t]["share_all"] for t in (APPROVE, STEP_UP, DECLINE)]
    assert abs(sum(shares) - 1.0) < 1e-6


# Loop internals (fast, pure)
def test_status_is_derived_from_metrics():
    cfg = config.LOOP_CONFIG
    # clear recovery above the ceiling
    assert _status(0.4, 0.4 + cfg["recovery_delta"] + 0.2, cfg) == "adapted"
    # stays below the residual ceiling
    assert _status(0.1, cfg["residual_recall_ceiling"] - 0.05, cfg) == "residual_frontier"
    # improves but not enough / above ceiling without delta -> partial
    assert _status(0.9, 0.91, cfg) == "partial"


def test_focus_mix_boosts_and_normalizes():
    mix = _focus_mix(["otp_relay", "account_takeover"])
    assert abs(sum(mix.values()) - 1.0) < 1e-9
    base = _focus_mix([])  # no boost baseline
    assert mix["otp_relay"] > base["otp_relay"]


def test_replay_buffer_is_cumulative_and_bounded():
    buf_X, buf_y, buf_atk, buf_round = [], [], [], []
    X = pd.DataFrame({"f": np.arange(100.0)})
    y = np.ones(100, dtype=int)
    atk = np.array(["otp_relay"] * 100)
    _bounded_append(buf_X, buf_y, buf_atk, buf_round, X, y, atk, 1, cap=1000)
    n1 = sum(len(a) for a in buf_y)
    _bounded_append(buf_X, buf_y, buf_atk, buf_round, X, y, atk, 2, cap=1000)
    n2 = sum(len(a) for a in buf_y)
    assert n2 > n1, "buffer must accumulate across rounds"
    # cap enforcement
    _bounded_append(buf_X, buf_y, buf_atk, buf_round, X, y, atk, 3, cap=120)
    assert sum(len(a) for a in buf_y) <= 120


# Adaptive loop integration (small, but exercises the real path)
def test_adaptive_loop_end_to_end():
    res = run_loop(rounds=2, n_base=9000, n_round=6000, n_eval=6000,
                   focus=["account_takeover", "otp_relay", "adversarial_mimicry"],
                   save=False)
    hist = res["history"]
    assert len(hist) == 2

    # guards must be disjoint from the families under attack, or the
    # forgetting check is measuring the thing that was deliberately changed
    assert not set(res["guard_families"]) & set(res["focus"])

    # every round records a promotion decision with its reasoning
    for h in hist:
        cc = h["champion_challenger"]
        assert cc["decision"] in {"PROMOTE", "REJECT"}
        assert cc["gates"], "a promotion decision must show the gates it evaluated"

    # cumulative replay grows
    assert hist[1]["replay_buffer_size"] > hist[0]["replay_buffer_size"]

    # every family carries a metric-derived status from the allowed set
    allowed = {"adapted", "partial", "residual_frontier", "n/a"}
    for h in hist:
        for fam, d in h["families"].items():
            assert d["status"] in allowed
            # adapted_recall is an actual fraction, not a hardcoded claim
            assert d["adapted_recall"] is None or 0.0 <= d["adapted_recall"] <= 1.0

    # prior generations are still tracked in later rounds (memory retained)
    assert hist[1]["prior_generation_recall"], "round 2 must evaluate round-1 variants"

    # Catastrophic-forgetting guard: a family that was never attacked must still
    # be caught after retraining. The bar is deliberately loose here because this
    # runs at a fraction of the committed scale; the real number is in
    # models/loop_history.json.
    guards = hist[-1]["guard_families_on_base"]
    assert guards, "the forgetting check must actually evaluate something"
    measured = [v["recall"] for v in guards.values() if v["recall"] is not None]
    assert measured, "guard families produced no measurements — the check is not running"
    assert max(measured) >= 0.5, "adaptation collapsed every previously-learned family"


# Reproducibility
def test_simulator_is_reproducible():
    a, _ = simulate(n_transactions=4000, fraud_rate=0.02, seed=123)
    b, _ = simulate(n_transactions=4000, fraud_rate=0.02, seed=123)
    assert a.shape == b.shape
    assert (a["amount"].to_numpy() == b["amount"].to_numpy()).all()
    assert (a["is_fraud"].to_numpy() == b["is_fraud"].to_numpy()).all()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nAll defense/loop tests passed.")
