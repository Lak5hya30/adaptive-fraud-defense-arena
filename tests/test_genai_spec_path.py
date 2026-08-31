"""The live GenAI structured attack-generation path, exercised without a network call.

No API key is available in this environment, so ``models/genai_spec_demo.json``
cannot be produced — and the demo script deliberately refuses to write one from
offline heuristic output, because a "demonstration of the GenAI path" produced by
the thing it is meant to demonstrate would be worthless.

What CAN be verified offline is everything either side of the network boundary:
the prompt is constructed from the measured weakness, a model response is parsed,
the payment-domain constraint layer validates it, and the result is a spec the
simulator can execute. These tests substitute a stub client for the network call
and check exactly that. The stub is a test double and lives only here — it never
writes an artifact.

    python -m pytest tests/test_genai_spec_path.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

import config
from src.generate import demo_specs
from src.generate.attack_spec import BASE_SPECS
from src.generate.llm_agent import propose_attack_spec
from src.llm.client import LLMUnavailable


class _StubClient:
    """Stands in for LLMClient. Returns a canned structured proposal."""

    def __init__(self, payload, record=None):
        self.payload = payload
        self.record = record if record is not None else []

    def structured(self, prompt, system="", **kw):
        self.record.append({"prompt": prompt, "system": system})
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


# The path, end to end, with the network stubbed
def test_model_proposal_becomes_an_executable_spec():
    seed = demo_specs.SEEDS[0]                      # account_takeover
    report = demo_specs._report(seed)
    previous = BASE_SPECS[seed["family"]]

    client = _StubClient({
        "strategy": "run from a device the victim already trusts",
        "amount_profile": "moderate", "velocity_profile": "low_and_slow",
        "device_behavior": "trusted_device", "geo_behavior": "plausible",
        "merchant_behavior": "new_high_risk_merchant", "timing_profile": "customer_normal",
        "intensity": 0.6, "targets_signal": "device_changed",
        "rationale": "the model's ranking of this family depends most on the device change",
        "confidence": 0.82,
    })
    spec, validation = propose_attack_spec(seed["family"], report, previous=previous,
                                           client=client)

    assert spec.source == "llm", "a model-authored proposal must be recorded as such"
    assert spec.device_behavior == "trusted_device"
    assert spec.targets_signal == "device_changed"
    assert spec.generation == previous.generation + 1
    assert validation["accepted"] is True
    # The spec has to be executable: derived knobs the simulator reads must resolve.
    assert spec.amount_scale > 0
    assert spec.txns_per_card[0] >= 1


def test_the_prompt_carries_the_measured_weakness():
    """The model is not asked to invent a target — it is handed the measurement."""
    seed = demo_specs.SEEDS[2]                      # wallet_provisioning
    report = demo_specs._report(seed)
    client = _StubClient({"device_behavior": "secondary_device", "intensity": 0.7})
    propose_attack_spec(seed["family"], report, previous=BASE_SPECS[seed["family"]],
                        client=client)

    sent = client.record[0]["prompt"]
    assert seed["family"] in sent
    for signal in seed["relied_signals"]:
        assert signal in sent, f"the prompt omits the measured signal {signal}"
    assert "0.42" in sent or "42" in sent, "the prompt omits the measured recall"
    # The system prompt must constrain the model to the dial vocabulary.
    system = client.record[0]["system"]
    assert "behavioural dials" in system
    assert "sandboxed simulator" in system


def test_an_impossible_model_proposal_is_corrected_not_executed():
    """The constraint layer sits between the model and the simulator for a reason."""
    seed = {"family": "scam_transfer", "recall": 0.8, "n_fraud": 100, "n_escaped": 20,
            "relied_signals": ["is_new_payee"], "note": ""}
    report = demo_specs._report(seed)
    client = _StubClient({
        "device_behavior": "new_device",       # impossible: the victim authenticates
        "geo_behavior": "high_risk",           # impossible for this family
        "intensity": 4.0,                      # out of range
        "targets_signal": "is_new_payee",
    })
    spec, validation = propose_attack_spec("scam_transfer", report,
                                           previous=BASE_SPECS["scam_transfer"],
                                           client=client)
    assert spec.device_behavior == "trusted_device"
    assert spec.geo_behavior in ("home", "plausible")
    lo, hi = config.ATTACK_SPEC_BOUNDS["intensity"]
    assert lo <= spec.intensity <= hi
    fields = {c["field"] for c in validation["corrections"]}
    assert {"device_behavior", "geo_behavior", "intensity"} <= fields


def test_an_unusable_model_response_falls_back_without_crashing():
    """If the model is unreachable or returns nonsense, the loop still runs — from
    the weakness-driven heuristic, and the spec is labelled accordingly."""
    seed = demo_specs.SEEDS[0]
    report = demo_specs._report(seed)
    for payload in (LLMUnavailable("no key"), {}, "not json at all"):
        spec, validation = propose_attack_spec(
            seed["family"], report, previous=BASE_SPECS[seed["family"]],
            client=_StubClient(payload))
        assert spec.source == "heuristic", (
            "a fallback spec must never be recorded as model-authored")
        assert validation["accepted"] is True


# The demonstration script itself
def test_every_seed_names_a_real_family_and_real_features():
    from src.defend.features import FEATURE_COLUMNS
    assert 3 <= len(demo_specs.SEEDS) <= 5, "the brief asks for 3-5 demonstration seeds"
    for seed in demo_specs.SEEDS:
        assert seed["family"] in BASE_SPECS, f"unknown family {seed['family']}"
        for f in seed["relied_signals"]:
            assert f in FEATURE_COLUMNS, f"{f} is not a real model feature"
        assert 0.0 <= seed["recall"] <= 1.0


def test_the_demo_script_refuses_to_fabricate_without_a_key(monkeypatch):
    """The failure mode that matters: producing a 'GenAI demonstration' from the
    offline heuristic. It must refuse, and it must write nothing."""
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    out = config.MODELS_DIR / "genai_spec_demo.json"
    before = out.exists()
    with pytest.raises(LLMUnavailable):
        demo_specs.run(save=True)
    assert out.exists() == before, "the script wrote an artifact with no key available"


def test_readiness_reports_the_truth():
    r = demo_specs.readiness()
    assert r["llm_available"] == config.llm_available()
    assert r["api_key_present"] == bool(config.ANTHROPIC_API_KEY)
