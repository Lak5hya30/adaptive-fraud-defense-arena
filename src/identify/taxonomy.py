"""Loader + validator for the GenAI payment-fraud Threat Atlas.

The catalog (attacks.json) is the single source of truth for Pillar 1. Its
``maps_to_injector`` field wires an attack to the transaction-level simulator in
Pillar 2, and ``observable_signals`` grounds the features used in Pillar 3.

The catalog is deliberately wider than the simulator. Every entry carries a
``simulator_status`` so research breadth is never presented as simulation
breadth:

    IMPLEMENTED    a dedicated injector reproduces this attack's authorization
                   footprint, and it is in the default simulated mix
    PARAMETERIZED  reachable today as a configuration of an existing injector
                   through the attack-specification dials, without new code
    RESEARCH_ONLY  catalogued and characterized, but not simulated
    FUTURE         named as planned simulator work

Every field describes what a DEFENDER can observe. The catalog contains no
operational guidance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

import config

IMPLEMENTED, PARAMETERIZED, RESEARCH_ONLY, FUTURE = (
    "IMPLEMENTED", "PARAMETERIZED", "RESEARCH_ONLY", "FUTURE")
VALID_STATUS = {IMPLEMENTED, PARAMETERIZED, RESEARCH_ONLY, FUTURE}
OBSERVABILITY_ORDER = {"high": 3, "partial": 2, "low": 1, "none": 0}
DIFFICULTY_ORDER = {"low": 1, "medium": 2, "high": 3, "extreme": 4}


@dataclass(frozen=True)
class Attack:
    id: str
    name: str
    category: str
    rails: list[str]
    channel: str
    genai_mechanism: str
    kill_chain: list[str]
    observable_signals: list[str]
    real_world_grounding: str
    severity: str
    novelty_score: float
    simulatable: bool
    maps_to_injector: str | None = None
    # --- Threat Atlas fields --------------------------------------------- #
    subcategory: str = ""
    attacker_objective: str = ""
    genai_role: str = ""
    transaction_signature: str = ""
    behavioral_signature: str = ""
    auth_time_observability: str = "partial"
    post_transaction_signals: list[str] = field(default_factory=list)
    defense_difficulty: str = "medium"
    expected_impact: str = "medium"
    ethical_notes: str = ""
    simulator_status: str = RESEARCH_ONLY

    @property
    def severity_rank(self) -> int:
        return {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(self.severity, 0)

    @property
    def observability_rank(self) -> int:
        return OBSERVABILITY_ORDER.get(self.auth_time_observability, 0)

    @property
    def difficulty_rank(self) -> int:
        return DIFFICULTY_ORDER.get(self.defense_difficulty, 0)

    @property
    def is_simulated(self) -> bool:
        return self.simulator_status in (IMPLEMENTED, PARAMETERIZED)


@dataclass
class Taxonomy:
    focus: str
    injectors: list[str]
    attacks: list[Attack] = field(default_factory=list)
    status_definitions: dict = field(default_factory=dict)
    honesty_note: str = ""
    provenance: dict = field(default_factory=dict)

    # -- lookups ----------------------------------------------------------
    def by_id(self, attack_id: str) -> Attack | None:
        return next((a for a in self.attacks if a.id == attack_id), None)

    @property
    def categories(self) -> list[str]:
        return sorted({a.category for a in self.attacks})

    @property
    def all_rails(self) -> list[str]:
        return sorted({r for a in self.attacks for r in a.rails})

    @property
    def channels(self) -> list[str]:
        return sorted({a.channel for a in self.attacks})

    @property
    def genai_roles(self) -> list[str]:
        return sorted({a.genai_role for a in self.attacks if a.genai_role})

    def simulatable(self) -> list[Attack]:
        return [a for a in self.attacks if a.is_simulated]

    def by_status(self, status: str) -> list[Attack]:
        return [a for a in self.attacks if a.simulator_status == status]

    def for_injector(self, injector: str) -> list[Attack]:
        return [a for a in self.attacks if a.maps_to_injector == injector]

    def coverage_by_category(self) -> dict[str, dict]:
        """Per-category counts, for the atlas coverage map."""
        out: dict[str, dict] = {}
        for c in self.categories:
            rows = [a for a in self.attacks if a.category == c]
            out[c] = {
                "total": len(rows),
                IMPLEMENTED: sum(1 for a in rows if a.simulator_status == IMPLEMENTED),
                PARAMETERIZED: sum(1 for a in rows if a.simulator_status == PARAMETERIZED),
                RESEARCH_ONLY: sum(1 for a in rows if a.simulator_status == RESEARCH_ONLY),
                FUTURE: sum(1 for a in rows if a.simulator_status == FUTURE),
            }
        return out

    def summary_counts(self) -> dict:
        return {
            "total_attacks": len(self.attacks),
            "categories": len(self.categories),
            "rails": len(self.all_rails),
            "channels": len(self.channels),
            "simulatable": len(self.simulatable()),
            "implemented": len(self.by_status(IMPLEMENTED)),
            "parameterized": len(self.by_status(PARAMETERIZED)),
            "research_only": len(self.by_status(RESEARCH_ONLY)),
            "future": len(self.by_status(FUTURE)),
            "injectors": len(self.injectors),
            "critical": sum(1 for a in self.attacks if a.severity == "critical"),
            "auth_time_hard": sum(1 for a in self.attacks
                                  if a.auth_time_observability in ("low", "none")),
        }


@lru_cache(maxsize=1)
def load_taxonomy() -> Taxonomy:
    raw = json.loads(config.ATTACKS_JSON.read_text(encoding="utf-8"))
    fields = set(Attack.__dataclass_fields__)
    attacks = [Attack(**{k: v for k, v in a.items() if k in fields}) for a in raw["attacks"]]
    tax = Taxonomy(focus=raw["focus"], injectors=raw["injectors"], attacks=attacks,
                   status_definitions=raw.get("status_definitions", {}),
                   honesty_note=raw.get("honesty_note", ""),
                   provenance=raw.get("provenance", {}))
    _validate(tax)
    return tax


def _validate(tax: Taxonomy) -> None:
    ids = [a.id for a in tax.attacks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate attack ids in attacks.json")
    valid_injectors = set(tax.injectors)
    for a in tax.attacks:
        if a.maps_to_injector and a.maps_to_injector not in valid_injectors:
            raise ValueError(
                f"Attack {a.id} maps to unknown injector {a.maps_to_injector!r}")
        if not 0.0 <= a.novelty_score <= 1.0:
            raise ValueError(f"Attack {a.id} novelty_score out of range")
        if a.simulator_status not in VALID_STATUS:
            raise ValueError(f"Attack {a.id} has unknown simulator_status "
                             f"{a.simulator_status!r}")
        # A claimed simulator mapping must actually name an injector, and an
        # unsimulated attack must not pretend to have one.
        if a.simulator_status in (IMPLEMENTED, PARAMETERIZED) and not a.maps_to_injector:
            raise ValueError(f"Attack {a.id} is {a.simulator_status} but maps to no injector")
        if a.simulator_status in (RESEARCH_ONLY, FUTURE) and a.maps_to_injector:
            raise ValueError(f"Attack {a.id} is {a.simulator_status} but claims an injector")
        if a.auth_time_observability not in OBSERVABILITY_ORDER:
            raise ValueError(f"Attack {a.id} has unknown auth_time_observability "
                             f"{a.auth_time_observability!r}")


if __name__ == "__main__":  # quick self-check
    t = load_taxonomy()
    print(json.dumps(t.summary_counts(), indent=2))
    print(f"\nCategories: {t.categories}")
    print("\nCoverage by category:")
    for c, d in t.coverage_by_category().items():
        print(f"  {c:34s} {d['total']:2d} total  "
              f"{d['IMPLEMENTED']} implemented  {d['PARAMETERIZED']} parameterized  "
              f"{d['RESEARCH_ONLY']} research-only  {d['FUTURE']} future")
