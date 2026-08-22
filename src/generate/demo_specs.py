"""Demonstrate the LIVE GenAI structured attack-generation path.

The committed artifacts in this repository were all produced by the deterministic
offline path, because the demo has to run with no API key and no network. That is
the right default, but it means nothing in the committed evidence shows the
language-model half of the design actually working.

This script closes that gap, and only that gap. It takes a handful of safe
synthetic attack seeds, sends each through the real red-team agent, puts the
returned specification through the payment-domain constraint layer, and writes the
result to ``models/genai_spec_demo.json`` as a clearly-labelled DEMONSTRATION
artifact.

It does not touch the dataset, the models, the loop, or any committed metric. It
trains nothing and regenerates nothing.

    python -m src.generate.demo_specs            # needs ANTHROPIC_API_KEY
    python -m src.generate.demo_specs --check    # report readiness and exit

If no key is available the script REFUSES to write anything. A demonstration of
the GenAI path that was actually produced by the offline heuristic would be worse
than having no demonstration at all.
"""
from __future__ import annotations

import argparse
import json

import config
from src.generate.attack_spec import BASE_SPECS, spec_diff, spec_distance
from src.generate.llm_agent import REDTEAM_SPEC_SYSTEM, _spec_prompt
from src.generate.attack_spec import validate_spec
from src.llm.client import LLMClient, LLMUnavailable
from src.loop.weakness import WeaknessReport

# Safe synthetic seeds. Each is a *measured weakness report* of the kind the loop
# produces — a family, the recall observed against it, and which signals the model
# was measured to lean on. Nothing here is real fraud intelligence, and nothing
# here is operational: the model is only ever asked to choose values on a fixed
# set of behavioural dials that a sandboxed simulator will execute.
SEEDS = [
    {
        "family": "account_takeover",
        "recall": 0.91, "n_fraud": 140, "n_escaped": 13,
        "relied_signals": ["device_changed", "log_amount", "mcc_risk"],
        "note": "classic takeover caught mainly by the device-change flag",
    },
    {
        "family": "card_testing",
        "recall": 0.97, "n_fraud": 210, "n_escaped": 6,
        "relied_signals": ["device_card_count_prior", "velocity_1h", "log_amount"],
        "note": "probing caught by the shared-device counter and short-window velocity",
    },
    {
        "family": "wallet_provisioning",
        "recall": 0.42, "n_fraud": 90, "n_escaped": 52,
        "relied_signals": ["mcc_risk", "is_new_payee", "log_amount"],
        "note": "token provisioning abuse, already partly evading",
    },
    {
        "family": "merchant_laundering",
        "recall": 0.87, "n_fraud": 120, "n_escaped": 16,
        "relied_signals": ["merchant_new_card_ratio_prior", "merchant_card_fanin_prior"],
        "note": "front merchant caught by network-level fan-in signals",
    },
    {
        "family": "otp_relay",
        "recall": 0.78, "n_fraud": 160, "n_escaped": 35,
        "relied_signals": ["device_changed", "mcc_risk", "amount_zscore"],
        "note": "relayed step-up still showing an attacker device",
    },
]


def _report(seed: dict) -> WeaknessReport:
    return WeaknessReport(
        family=seed["family"], recall=seed["recall"],
        n_fraud=seed["n_fraud"], n_escaped=seed["n_escaped"],
        relied_signals=[{"feature": f, "auc_drop": round(0.11 - i * 0.02, 4)}
                        for i, f in enumerate(seed["relied_signals"])],
        escape_profile=[], note=seed["note"])


def readiness() -> dict:
    return {
        "api_key_present": bool(config.ANTHROPIC_API_KEY),
        "llm_enabled": config.LLM_ENABLED,
        "llm_available": config.llm_available(),
        "model": config.LLM_MODEL,
        "cache_dir": str(config.CACHE_DIR),
        "cached_responses": len(list(config.CACHE_DIR.glob("*.json"))),
        "output_path": str(config.MODELS_DIR / "genai_spec_demo.json"),
    }


def run(seeds: list[dict] | None = None, save: bool = True) -> dict:
    """Run the live path over the seeds. Raises LLMUnavailable if there is no key."""
    seeds = seeds or SEEDS
    if not config.llm_available():
        raise LLMUnavailable(
            "No ANTHROPIC_API_KEY available. This script will not write a "
            "demonstration artifact produced by the offline heuristic — that would "
            "misrepresent the very thing it exists to demonstrate.")

    client = LLMClient()
    examples = []
    for seed in seeds:
        fam = seed["family"]
        previous = BASE_SPECS[fam]
        rep = _report(seed)
        prompt = _spec_prompt(fam, previous, rep)

        raw = client.structured(prompt, REDTEAM_SPEC_SYSTEM, schema_hint="object")
        proposal = dict(raw) if isinstance(raw, dict) else {}
        proposal["source"] = "llm"
        spec, validation = validate_spec(proposal, fam, previous=previous)

        examples.append({
            "seed": seed,
            "prompt_sent_to_model": prompt,
            "raw_model_proposal": raw,
            "validated_spec": spec.to_dict(),
            "constraint_layer": validation.to_dict(),
            "corrections_applied": validation.n_corrections,
            "diff_from_generation_0": spec_diff(previous, spec),
            "behavioural_distance_from_generation_0": spec_distance(previous, spec),
            "spec_source": spec.source,
        })

    out = {
        "artifact_type": "GENAI_PATH_DEMONSTRATION",
        "purpose": ("Evidence that the live GenAI structured attack-generation path works "
                    "end to end: a measured weakness goes in, the language model returns a "
                    "structured specification, and the payment-domain constraint layer "
                    "validates it before anything could be simulated."),
        "not_used_for": ("Nothing here feeds the dataset, the models, the closed loop, or any "
                         "committed metric. Demo mode continues to use the deterministic "
                         "committed specifications, for reliability."),
        "model": config.LLM_MODEL,
        "schema_version": config.SCHEMA_VERSION,
        "n_examples": len(examples),
        "n_with_corrections": sum(1 for e in examples if e["corrections_applied"] > 0),
        "safety_note": ("Seeds are synthetic measured-weakness reports. The model is only ever "
                        "asked to choose values on a fixed vocabulary of behavioural dials that "
                        "a sandboxed simulator executes; it is never asked for, and cannot "
                        "return, operational guidance."),
        "examples": examples,
    }
    if save:
        (config.MODELS_DIR / "genai_spec_demo.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Demonstrate the live GenAI structured attack-generation path.")
    ap.add_argument("--check", action="store_true",
                    help="report readiness and exit without calling the model")
    ap.add_argument("--n", type=int, default=len(SEEDS),
                    help="how many seeds to run (default: all)")
    args = ap.parse_args()

    r = readiness()
    print(json.dumps(r, indent=2))
    if args.check:
        return
    if not r["llm_available"]:
        print("\nNOT RUN: no ANTHROPIC_API_KEY available.\n"
              "Set it in .env or the environment and re-run:\n"
              "    python -m src.generate.demo_specs\n"
              "Nothing was written. This script never substitutes offline heuristic output "
              "for a live model response.")
        raise SystemExit(2)

    out = run(SEEDS[: args.n])
    print(f"\nWrote {out['n_examples']} live specifications to "
          f"{config.MODELS_DIR / 'genai_spec_demo.json'}")
    print(f"{out['n_with_corrections']} of them needed a correction from the constraint layer.")
    for e in out["examples"]:
        fam = e["validated_spec"]["attack_family"]
        changed = ", ".join(f"{k}: {v['from']} -> {v['to']}"
                            for k, v in e["diff_from_generation_0"]["changes"].items()) or "—"
        print(f"  {fam:22s} targets={e['validated_spec']['targets_signal'][:34]!r} "
              f"changed={changed}")


if __name__ == "__main__":
    main()
