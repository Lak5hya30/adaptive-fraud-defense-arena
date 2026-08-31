"""Structured attack specifications and the payment-domain constraint layer.

This module is the contract between the *creative* half of the red team and the
*deterministic* half of the simulator:

    Threat intelligence / observed model weakness
                  |
           GenAI red-team agent          <- creative, may propose anything
                  |
        AttackSpec (structured JSON)
                  |
        validate_spec()  <- payment-domain constraints; clamp or reject
                  |
        Constrained simulator engine     <- deterministic, reproducible
                  |
          Synthetic payment transactions

The language model never writes transaction rows. It writes a *specification*:
which behavioural dial to move, in which direction, and why. Everything that
touches the dataset is deterministic given the seed, so results stay reproducible
whether the specification came from Claude or from the offline heuristic.

A specification that violates payment reality — negative amounts, impossible
device/account combinations, transaction rates beyond simulation bounds, a family
requirement that cannot hold — is normalized (clamped to the nearest legal value)
or rejected outright. Nothing invalid is ever executed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace

import config

B = config.ATTACK_SPEC_BOUNDS

# The specification

# Categorical dials, mapped to the concrete numbers the injectors consume.
AMOUNT_SCALE = {"micro": 0.05, "low": 0.45, "moderate": 1.0, "high": 1.9, "extreme": 3.4}
VELOCITY_TXNS = {"single": (1, 1), "low_and_slow": (1, 2), "moderate": (2, 4), "burst": (4, 9)}
VELOCITY_WINDOW_H = {"single": 0.0, "low_and_slow": 72.0, "moderate": 24.0, "burst": 1.5}
# Probability the attack runs from a device the card has used before.
DEVICE_TRUST = {"trusted_device": 1.0, "secondary_device": 0.6, "shared_device": 0.0,
                "new_device": 0.0}
GEO_KM = {"home": (0.0, 12.0), "plausible": (5.0, 60.0), "domestic_far": (400.0, 1800.0),
          "foreign": (1800.0, 7000.0), "high_risk": (2500.0, 8000.0)}


@dataclass(frozen=True)
class AttackSpec:
    """A constrained, executable description of one attack generation."""

    attack_family: str
    strategy: str = "baseline behaviour of the family"
    intensity: float = 1.0                     # 1.0 = blatant, 0.15 = near-legitimate
    amount_profile: str = "moderate"
    velocity_profile: str = "single"
    device_behavior: str = "new_device"
    geo_behavior: str = "plausible"
    merchant_behavior: str = "new_high_risk_merchant"
    timing_profile: str = "any"
    targets_signal: str = ""                   # detector signal this generation aims to defeat
    rationale: str = ""
    confidence: float = 0.5
    source: str = "default"                    # llm | heuristic | default | fixed
    generation: int = 0

    # -- derived numeric knobs used by the injectors ----------------------- #
    @property
    def amount_scale(self) -> float:
        return AMOUNT_SCALE.get(self.amount_profile, 1.0)

    @property
    def txns_per_card(self) -> tuple[int, int]:
        return VELOCITY_TXNS.get(self.velocity_profile, (1, 1))

    @property
    def velocity_window_hours(self) -> float:
        return VELOCITY_WINDOW_H.get(self.velocity_profile, 24.0)

    @property
    def device_trust(self) -> float:
        return DEVICE_TRUST.get(self.device_behavior, 0.0)

    @property
    def geo_km_range(self) -> tuple[float, float]:
        return GEO_KM.get(self.geo_behavior, (5.0, 60.0))

    @property
    def reuse_known_merchant(self) -> bool:
        return self.merchant_behavior == "known_merchant"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["derived"] = {
            "amount_scale": self.amount_scale,
            "txns_per_card": list(self.txns_per_card),
            "velocity_window_hours": self.velocity_window_hours,
            "device_trust": self.device_trust,
            "geo_km_range": list(self.geo_km_range),
        }
        return d


# Generation-0 specifications: how each family behaves before any evolution.
# These are the lineage roots that evolved variants are compared against.
BASE_SPECS: dict[str, AttackSpec] = {
    "card_testing": AttackSpec(
        "card_testing", "validate stolen card numbers with cheap probes across small merchants",
        1.0, "micro", "burst", "shared_device", "home", "new_low_risk_merchant", "any",
        targets_signal="none (pre-evolution baseline)", source="fixed"),
    "bust_out": AttackSpec(
        # The account belongs to the fraudster, so the drain runs from the same
        # device that built the history. There is no device change to catch.
        "bust_out", "groom a synthetic account with ordinary spend, then drain it to the limit",
        1.0, "extreme", "moderate", "trusted_device", "plausible", "cash_like", "customer_normal",
        targets_signal="none (pre-evolution baseline)", source="fixed"),
    "account_takeover": AttackSpec(
        # Deliberately NOT a foreign-geography attack. Real account takeover is
        # overwhelmingly domestic card-not-present spend; making generation 0
        # transact from another continent would hand the detector a geography
        # tell and let the family post a recall that has nothing to do with
        # detecting takeover. What generation 0 does show is the attacker's own
        # device — and removing that is exactly what the closed loop evolves.
        "account_takeover", "spend hard from an attacker device before the victim notices",
        1.0, "high", "moderate", "new_device", "plausible", "new_high_risk_merchant", "any",
        targets_signal="none (pre-evolution baseline)", source="fixed"),
    "adversarial_mimicry": AttackSpec(
        "adversarial_mimicry", "shape stolen-card spend onto the victim's own behavioural centroid",
        1.0, "moderate", "single", "secondary_device", "home", "known_merchant", "customer_normal",
        targets_signal="every individual signal held inside the customer's normal range", source="fixed"),
    "velocity_smurfing": AttackSpec(
        "velocity_smurfing", "split value across a mule ring in sub-threshold amounts",
        1.0, "moderate", "burst", "shared_device", "plausible", "front_merchant", "any",
        targets_signal="none (pre-evolution baseline)", source="fixed"),
    "merchant_laundering": AttackSpec(
        "merchant_laundering", "run stolen cards through a controlled front merchant",
        1.0, "moderate", "moderate", "shared_device", "plausible", "front_merchant", "any",
        targets_signal="none (pre-evolution baseline)", source="fixed"),
    "friendly_fraud": AttackSpec(
        "friendly_fraud", "make a genuine purchase and dispute it afterwards",
        1.0, "moderate", "single", "trusted_device", "home", "known_merchant", "customer_normal",
        targets_signal="authorization-time observability (there is almost none)", source="fixed"),
    "otp_relay": AttackSpec(
        "otp_relay", "walk the victim through a step-up challenge and relay the code live",
        # The victim has to be awake and on the phone for a relay to work, so the
        # transaction lands inside the customer's ordinary waking hours.
        1.0, "high", "low_and_slow", "new_device", "home", "new_high_risk_merchant",
        "customer_normal", targets_signal="none (pre-evolution baseline)", source="fixed"),
    "geo_anomaly": AttackSpec(
        "geo_anomaly", "present a cloned card far from the cardholder's home geography",
        1.0, "moderate", "moderate", "new_device", "foreign", "new_low_risk_merchant", "any",
        targets_signal="none (pre-evolution baseline)", source="fixed"),
    "scam_transfer": AttackSpec(
        "scam_transfer", "persuade the genuine cardholder to authorize a push to a new payee",
        1.0, "high", "low_and_slow", "trusted_device", "home", "cash_like", "customer_normal",
        targets_signal="authentication itself (the real customer approves)", source="fixed"),
    "wallet_provisioning": AttackSpec(
        "wallet_provisioning", "provision a compromised card into an attacker-controlled wallet, "
                              "then spend on the inherited token trust",
        1.0, "high", "moderate", "new_device", "plausible", "new_high_risk_merchant",
        "customer_normal", targets_signal="none (pre-evolution baseline)", source="fixed"),
}

# Payment-domain requirements per family. A specification that contradicts these
# describes something that cannot happen on a real rail, so it is normalized.
FAMILY_CONSTRAINTS: dict[str, dict] = {
    # The genuine customer authenticates on their own device: an "attacker device"
    # authorized push payment is a contradiction in terms.
    "scam_transfer": {"device_behavior": {"trusted_device"},
                      "geo_behavior": {"home", "plausible"},
                      "merchant_behavior": {"cash_like", "new_high_risk_merchant"}},
    "friendly_fraud": {"device_behavior": {"trusted_device"},
                       "geo_behavior": {"home", "plausible"},
                       "merchant_behavior": {"known_merchant", "new_low_risk_merchant"}},
    # Mimicry is defined by staying on the victim's centroid; a foreign
    # high-value burst would simply be a different attack.
    "adversarial_mimicry": {"geo_behavior": {"home", "plausible"},
                            "amount_profile": {"low", "moderate"},
                            "velocity_profile": {"single", "low_and_slow"},
                            "merchant_behavior": {"known_merchant", "new_low_risk_merchant"}},
    # Probing is only economically sensible at trivial value and real volume.
    "card_testing": {"amount_profile": {"micro", "low"},
                     "velocity_profile": {"moderate", "burst"},
                     "device_behavior": {"shared_device", "new_device"}},
    "velocity_smurfing": {"amount_profile": {"low", "moderate"},
                          "velocity_profile": {"moderate", "burst"},
                          "device_behavior": {"shared_device", "new_device"}},
    "merchant_laundering": {"merchant_behavior": {"front_merchant"},
                            "device_behavior": {"shared_device", "new_device"}},
    "bust_out": {"merchant_behavior": {"cash_like", "new_high_risk_merchant"},
                 "amount_profile": {"high", "extreme"}},
    "geo_anomaly": {"geo_behavior": {"domestic_far", "foreign", "high_risk"}},
    "otp_relay": {"merchant_behavior": {"new_high_risk_merchant", "cash_like",
                                        "new_low_risk_merchant"}},
    "wallet_provisioning": {"device_behavior": {"new_device", "secondary_device"}},
}


class SpecRejected(ValueError):
    """Raised when a proposed specification cannot be made legal."""


@dataclass
class ValidationReport:
    """What the constraint layer did to a proposed specification."""

    family: str
    accepted: bool = True
    corrections: list[dict] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)

    def note(self, fieldname: str, frm, to, why: str) -> None:
        self.corrections.append({"field": fieldname, "from": frm, "to": to, "reason": why})

    @property
    def n_corrections(self) -> int:
        return len(self.corrections)

    def to_dict(self) -> dict:
        return {"family": self.family, "accepted": self.accepted,
                "corrections": self.corrections, "rejections": self.rejections}


def _coerce_choice(value, allowed: list[str], fallback: str, fieldname: str,
                   report: ValidationReport) -> str:
    v = str(value).strip().lower().replace(" ", "_") if value is not None else ""
    if v in allowed:
        return v
    report.note(fieldname, value, fallback, "value outside the permitted vocabulary")
    return fallback


def validate_spec(proposed: dict, family: str, previous: AttackSpec | None = None,
                  strict: bool = False) -> tuple[AttackSpec, ValidationReport]:
    """Turn an arbitrary proposal into an executable, payment-legal AttackSpec.

    Anything out of vocabulary is replaced with the family's baseline value;
    anything out of range is clamped; anything contradicting a family
    requirement is corrected to the nearest permitted value. With ``strict=True``
    a proposal that needs a family-level correction is rejected instead, which is
    how the closed loop can report "the red team asked for something impossible".
    """
    if family not in BASE_SPECS:
        raise SpecRejected(f"unknown attack family {family!r}")
    base = previous or BASE_SPECS[family]
    report = ValidationReport(family=family)
    p = dict(proposed or {})

    # --- 1. numeric ranges ------------------------------------------------- #
    lo, hi = B["intensity"]
    try:
        intensity = float(p.get("intensity", base.intensity))
    except (TypeError, ValueError):
        report.note("intensity", p.get("intensity"), base.intensity, "not a number")
        intensity = base.intensity
    if not lo <= intensity <= hi:
        report.note("intensity", intensity, min(max(intensity, lo), hi), "outside permitted range")
        intensity = min(max(intensity, lo), hi)
    # A persistent adversary does not become louder after being caught: stealth
    # is monotone within a lineage.
    if previous is not None and intensity > previous.intensity:
        report.note("intensity", intensity, previous.intensity,
                    "stealth may not regress within an attack lineage")
        intensity = previous.intensity

    try:
        confidence = float(p.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = min(max(confidence, 0.0), 1.0)

    # --- 2. categorical vocabularies --------------------------------------- #
    spec = AttackSpec(
        attack_family=family,
        strategy=str(p.get("strategy", base.strategy))[:400] or base.strategy,
        intensity=intensity,
        amount_profile=_coerce_choice(p.get("amount_profile", base.amount_profile),
                                      B["amount_profile"], base.amount_profile,
                                      "amount_profile", report),
        velocity_profile=_coerce_choice(p.get("velocity_profile", base.velocity_profile),
                                        B["velocity_profile"], base.velocity_profile,
                                        "velocity_profile", report),
        device_behavior=_coerce_choice(p.get("device_behavior", base.device_behavior),
                                       B["device_behavior"], base.device_behavior,
                                       "device_behavior", report),
        geo_behavior=_coerce_choice(p.get("geo_behavior", base.geo_behavior),
                                    B["geo_behavior"], base.geo_behavior,
                                    "geo_behavior", report),
        merchant_behavior=_coerce_choice(p.get("merchant_behavior", base.merchant_behavior),
                                         B["merchant_behavior"], base.merchant_behavior,
                                         "merchant_behavior", report),
        timing_profile=_coerce_choice(p.get("timing_profile", base.timing_profile),
                                      B["timing_profile"], base.timing_profile,
                                      "timing_profile", report),
        targets_signal=str(p.get("targets_signal", ""))[:200],
        rationale=str(p.get("rationale", ""))[:600],
        confidence=confidence,
        source=str(p.get("source", "llm")),
        generation=int(p.get("generation", base.generation + 1)),
    )

    # --- 3. family-level payment-domain requirements ------------------------ #
    reqs = FAMILY_CONSTRAINTS.get(family, {})
    for fieldname, allowed in reqs.items():
        current = getattr(spec, fieldname)
        if current in allowed:
            continue
        fallback = getattr(base, fieldname)
        if fallback not in allowed:
            fallback = sorted(allowed)[0]
        if strict:
            report.rejections.append(
                f"{fieldname}={current!r} is impossible for family {family!r}")
            report.accepted = False
            continue
        report.note(fieldname, current, fallback,
                    f"{family} cannot be executed with {fieldname}={current!r} on a real rail")
        spec = replace(spec, **{fieldname: fallback})

    if not report.accepted:
        raise SpecRejected("; ".join(report.rejections))
    return spec, report


def spec_diff(a: AttackSpec, b: AttackSpec) -> dict:
    """Structured difference between two generations of the same family.

    Used by the attack-lineage view: it is what makes "how did the attack evolve"
    a concrete, inspectable answer rather than a claim.
    """
    fields = ["intensity", "amount_profile", "velocity_profile", "device_behavior",
              "geo_behavior", "merchant_behavior", "timing_profile"]
    changes = {}
    for f in fields:
        va, vb = getattr(a, f), getattr(b, f)
        if isinstance(va, float):
            if abs(va - vb) > 1e-9:
                changes[f] = {"from": round(va, 3), "to": round(vb, 3)}
        elif va != vb:
            changes[f] = {"from": va, "to": vb}
    return {
        "family": b.attack_family,
        "from_generation": a.generation,
        "to_generation": b.generation,
        "changes": changes,
        "n_changed_dials": len(changes),
        "targets_signal": b.targets_signal,
        "strategy": b.strategy,
    }


def spec_distance(a: AttackSpec, b: AttackSpec) -> float:
    """A blunt 0..1 behavioural distance between two generations.

    Categorical dials contribute 1 each when they differ; intensity contributes
    its absolute change. Deliberately simple — it is a readable summary of how
    far the attack has moved, not a claim about a metric space.
    """
    fields = ["amount_profile", "velocity_profile", "device_behavior",
              "geo_behavior", "merchant_behavior", "timing_profile"]
    changed = sum(1 for f in fields if getattr(a, f) != getattr(b, f))
    return round(min(1.0, (changed + abs(a.intensity - b.intensity)) / (len(fields) + 1)), 4)


def as_spec(value, family: str) -> AttackSpec:
    """Accept an AttackSpec, a dict, a bare intensity float, or None."""
    if isinstance(value, AttackSpec):
        return value
    if value is None:
        return BASE_SPECS[family]
    if isinstance(value, (int, float)):
        base = BASE_SPECS[family]
        return replace(base, intensity=float(min(max(value, B["intensity"][0]),
                                                 B["intensity"][1])))
    spec, _ = validate_spec(dict(value), family)
    return spec


def dumps(spec: AttackSpec) -> str:
    return json.dumps(spec.to_dict(), indent=2)
