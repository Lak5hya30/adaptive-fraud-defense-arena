"""Simulation orchestrator: assemble a labeled card-transaction dataset + a
GenAI attack-artifact corpus, at configurable scale / fraud-rate / attack-mix.

Attack behaviour is driven by validated :class:`AttackSpec` objects, so the same
entry point serves the static dataset (generation-0 specs) and every evolved
generation the closed loop produces. A bare ``intensity`` float is still accepted
and is converted into the family's baseline spec at that intensity.

CLI:  python -m src.generate.simulate
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import config
from src.generate import base_generator as BG
from src.generate import profiles as P
from src.generate.attack_injectors import DEFAULT_MIX, INJECTORS
from src.generate.attack_spec import BASE_SPECS, as_spec
from src.generate.llm_agent import build_text_corpus


def simulate(
    n_transactions: int = config.DEFAULT_N_TRANSACTIONS,
    fraud_rate: float = config.DEFAULT_FRAUD_RATE,
    n_cardholders: int = config.DEFAULT_N_CARDHOLDERS,
    n_merchants: int = config.DEFAULT_N_MERCHANTS,
    mix: dict[str, float] | None = None,
    intensities: dict[str, float] | None = None,
    specs: dict | None = None,
    seed: int = config.GLOBAL_SEED,
) -> tuple[pd.DataFrame, dict]:
    """Return (dataframe, summary). Deterministic given seed.

    ``specs`` maps a family to an AttackSpec / spec dict; ``intensities`` is the
    older scalar form and is honoured for backwards compatibility.
    """
    rng = P.make_rng(seed)
    mix = mix or DEFAULT_MIX
    intensities = intensities or {}
    specs = specs or {}

    holders = P.generate_cardholders(n_cardholders, rng)
    merchants = P.generate_merchants(n_merchants, rng)
    P.assign_relationships(holders, merchants, rng)

    n_fraud_target = int(round(n_transactions * fraud_rate))

    # --- fraud + the cover traffic its actors generate ---------------------- #
    mix_total = sum(mix.values())
    attack_rows: list[dict] = []
    used_specs: dict[str, dict] = {}
    for atk, weight in mix.items():
        count = int(round(n_fraud_target * weight / mix_total))
        if count <= 0 or atk not in INJECTORS:
            continue
        spec = as_spec(specs.get(atk, intensities.get(atk)), atk)
        used_specs[atk] = spec.to_dict()
        attack_rows.extend(INJECTORS[atk](holders, merchants, rng, count, spec))

    attack_df = pd.DataFrame(attack_rows, columns=BG.COLUMNS)
    n_fraud_actual = int(attack_df["is_fraud"].sum()) if len(attack_df) else 0
    n_cover = len(attack_df) - n_fraud_actual

    # Genuine traffic fills whatever the attack stream did not occupy, so the
    # requested total size and fraud rate both hold.
    n_legit = max(0, n_transactions - len(attack_df))
    legit_df = BG.generate_legit(holders, merchants, n_legit, rng)

    df = pd.concat([legit_df, attack_df], ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["txn_id"] = [f"T{i:08d}" for i in range(len(df))]

    # is_new_payee (auth-time historical fact): first time a card transacts with a
    # given merchant, in chronological order. Computed globally so legit and fraud
    # share one principled definition — genuine new-merchant visits are common
    # enough that "new payee" is a risk signal, not a fraud shortcut.
    df["is_new_payee"] = ~df.duplicated(subset=["card_id", "merchant_id"], keep="first")

    summary = _summarize(df, holders, merchants, used_specs, n_cover)
    return df, summary


def _summarize(df: pd.DataFrame, holders, merchants, used_specs: dict,
               n_cover: int) -> dict:
    counts = df["attack_type"].value_counts().to_dict()
    arch = {}
    for h in holders:
        arch[h.archetype] = arch.get(h.archetype, 0) + 1
    return {
        "schema_version": config.SCHEMA_VERSION,
        "n_transactions": int(len(df)),
        "n_fraud": int(df["is_fraud"].sum()),
        "fraud_rate": round(float(df["is_fraud"].mean()), 4),
        "n_cover_transactions": int(n_cover),
        "n_cardholders": len(holders),
        "n_merchants": len(merchants),
        "customer_archetypes": arch,
        "attack_type_counts": {k: int(v) for k, v in counts.items()},
        "attack_specs": used_specs,
        "date_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
    }


def run_and_save(with_artifacts: bool = True, **kwargs) -> dict:
    df, summary = simulate(**kwargs)
    df.to_parquet(config.DATASET_PARQUET, index=False)
    df.to_csv(config.DATASET_CSV, index=False)

    if with_artifacts:
        corpus = build_text_corpus()
        with config.ARTIFACTS_JSONL.open("w", encoding="utf-8") as f:
            for row in corpus:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["n_text_artifacts"] = len(corpus)
        summary["n_fraud_artifacts"] = sum(1 for r in corpus if r["label"] == 1)

    (config.DATA_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Simulate labeled card-fraud dataset.")
    ap.add_argument("--n", type=int, default=config.DEFAULT_N_TRANSACTIONS)
    ap.add_argument("--fraud-rate", type=float, default=config.DEFAULT_FRAUD_RATE)
    ap.add_argument("--no-artifacts", action="store_true")
    ap.add_argument("--seed", type=int, default=config.GLOBAL_SEED)
    args = ap.parse_args()

    summary = run_and_save(
        with_artifacts=not args.no_artifacts,
        n_transactions=args.n, fraud_rate=args.fraud_rate, seed=args.seed)
    print(json.dumps({k: v for k, v in summary.items() if k != "attack_specs"}, indent=2))


if __name__ == "__main__":
    main()
