"""Tests for the attack-specification contract, the payment-domain constraint
layer, weakness-driven mutation, and the governance gates.

These cover the parts of the system that make claims about *behaviour* rather
than about numbers: that an impossible attack cannot be executed, that "evasion"
always moves toward legitimate behaviour, and that the promotion gates can
actually refuse a model.

    python -m pytest tests/test_attack_spec_and_loop.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

import config
from src.defend.governance import PROMOTE, REJECT, evaluate_candidate
from src.generate.attack_spec import (BASE_SPECS, FAMILY_CONSTRAINTS, SpecRejected,
                                      as_spec, spec_diff, spec_distance, validate_spec)
from src.loop.weakness import (SIGNAL_TO_DIAL, WeaknessReport, _STEALTH_RANK,
                               heuristic_mutation)


# The constraint layer
def test_every_family_has_a_generation_zero_spec():
    from src.generate.attack_injectors import INJECTORS
    missing = sorted(set(INJECTORS) - set(BASE_SPECS))
    assert not missing, f"injectors without a baseline specification: {missing}"


def test_out_of_range_intensity_is_clamped_not_executed():
    spec, report = validate_spec({"intensity": 9.9}, "account_takeover")
    lo, hi = config.ATTACK_SPEC_BOUNDS["intensity"]
    assert lo <= spec.intensity <= hi
    assert any(c["field"] == "intensity" for c in report.corrections)


def test_unknown_vocabulary_falls_back_to_the_family_baseline():
    spec, report = validate_spec({"device_behavior": "teleportation"}, "account_takeover")
    assert spec.device_behavior in config.ATTACK_SPEC_BOUNDS["device_behavior"]
    assert any(c["field"] == "device_behavior" for c in report.corrections)


def test_impossible_family_combination_is_corrected():
    """An authorized push payment scam cannot run from an attacker's device: the
    genuine customer is the one authenticating."""
    spec, report = validate_spec({"device_behavior": "new_device"}, "scam_transfer")
    assert spec.device_behavior in FAMILY_CONSTRAINTS["scam_transfer"]["device_behavior"]
    assert any(c["field"] == "device_behavior" for c in report.corrections)


def test_strict_mode_refuses_rather_than_correcting():
    with pytest.raises(SpecRejected):
        validate_spec({"device_behavior": "new_device"}, "scam_transfer", strict=True)


def test_stealth_is_monotone_within_a_lineage():
    previous = as_spec({"intensity": 0.4}, "account_takeover")
    spec, report = validate_spec({"intensity": 0.95}, "account_takeover", previous=previous)
    assert spec.intensity <= previous.intensity
    assert any("stealth" in c["reason"] for c in report.corrections)


def test_amounts_stay_inside_simulation_bounds():
    for fam, spec in BASE_SPECS.items():
        assert spec.amount_scale > 0, fam
        lo, hi = spec.txns_per_card
        assert 1 <= lo <= hi <= config.ATTACK_SPEC_BOUNDS["max_txns_per_card_per_day"], fam


def test_spec_diff_and_distance_describe_a_real_change():
    a = BASE_SPECS["account_takeover"]
    b, _ = validate_spec({"device_behavior": "trusted_device"}, "account_takeover",
                         previous=a)
    d = spec_diff(a, b)
    assert "device_behavior" in d["changes"]
    assert d["changes"]["device_behavior"]["to"] == "trusted_device"
    assert 0.0 < spec_distance(a, b) <= 1.0
    assert spec_distance(a, a) == 0.0


# Weakness-driven mutation
def _report(family: str, signals: list[str], recall: float = 0.9) -> WeaknessReport:
    return WeaknessReport(
        family=family, recall=recall, n_fraud=100, n_escaped=10,
        relied_signals=[{"feature": s, "auc_drop": 0.1 - i * 0.01}
                        for i, s in enumerate(signals)])


def test_mutation_targets_the_signal_the_model_relies_on():
    prev = BASE_SPECS["account_takeover"]           # device_behavior = new_device
    proposal = heuristic_mutation(_report("account_takeover", ["device_changed"]), prev)
    assert proposal["device_behavior"] == "trusted_device"
    assert "device_changed" in proposal["targets_signal"]


def test_mutation_never_makes_the_attack_louder():
    """Every dial move must take the attack CLOSER to ordinary behaviour.

    Proposing 'stop reusing the victim's device' because the model watches device
    sharing would hand the defense an easier signal, not a harder one.
    """
    for fam, prev in BASE_SPECS.items():
        for signal in SIGNAL_TO_DIAL:
            proposal = heuristic_mutation(_report(fam, [signal]), prev)
            for field_name, ranks in _STEALTH_RANK.items():
                if field_name not in proposal:
                    continue
                before = ranks.get(getattr(prev, field_name), 99)
                after = ranks.get(proposal[field_name], 99)
                assert after <= before, (
                    f"{fam}: proposal moved {field_name} from "
                    f"{getattr(prev, field_name)} to {proposal[field_name]}, which is louder")


def test_mutation_output_survives_the_constraint_layer():
    for fam, prev in BASE_SPECS.items():
        proposal = heuristic_mutation(_report(fam, ["device_changed", "log_amount"]), prev)
        spec, _ = validate_spec(proposal, fam, previous=prev)
        assert spec.attack_family == fam
        assert spec.generation == prev.generation + 1


def test_mutation_reduces_intensity_each_generation():
    prev = BASE_SPECS["otp_relay"]
    proposal = heuristic_mutation(_report("otp_relay", ["device_changed"], recall=0.95), prev)
    assert proposal["intensity"] < prev.intensity


# Champion / challenger gates
def test_gate_promotes_a_genuinely_better_candidate():
    d = evaluate_candidate(
        champion_metrics={"attack_recall": 0.30, "false_positive_rate": 0.015,
                          "pr_auc": 0.50},
        candidate_metrics={"attack_recall": 0.80, "false_positive_rate": 0.016,
                           "pr_auc": 0.55},
        prior_family_recall={"card_testing": {"champion": 0.9, "candidate": 0.88}})
    assert d["decision"] == PROMOTE, d["failed_gates"]


def test_gate_refuses_a_candidate_that_floods_genuine_customers():
    d = evaluate_candidate(
        champion_metrics={"attack_recall": 0.30, "false_positive_rate": 0.015,
                          "pr_auc": 0.50},
        candidate_metrics={"attack_recall": 0.95, "false_positive_rate": 0.22,
                           "pr_auc": 0.55})
    assert d["decision"] == REJECT
    assert "absolute_false_positive_ceiling" in d["failed_gates"]


def test_gate_refuses_a_candidate_that_forgets():
    d = evaluate_candidate(
        champion_metrics={"attack_recall": 0.30, "false_positive_rate": 0.015,
                          "pr_auc": 0.50},
        candidate_metrics={"attack_recall": 0.90, "false_positive_rate": 0.016,
                           "pr_auc": 0.52},
        prior_family_recall={"card_testing": {"champion": 0.95, "candidate": 0.40}})
    assert d["decision"] == REJECT
    assert "no_catastrophic_forgetting" in d["failed_gates"]


def test_gate_refuses_a_candidate_with_no_real_recall_gain():
    d = evaluate_candidate(
        champion_metrics={"attack_recall": 0.80, "false_positive_rate": 0.015,
                          "pr_auc": 0.50},
        candidate_metrics={"attack_recall": 0.81, "false_positive_rate": 0.015,
                           "pr_auc": 0.50})
    assert d["decision"] == REJECT
    assert "attack_recall_gain" in d["failed_gates"]


def test_guard_families_are_never_also_targets():
    from src.loop.redteam_loop import GUARD_CANDIDATES, resolve_guards
    focus = ["card_testing", "bust_out"]
    guards = resolve_guards(focus)
    assert not set(guards) & set(focus)
    assert all(g in GUARD_CANDIDATES for g in guards)
