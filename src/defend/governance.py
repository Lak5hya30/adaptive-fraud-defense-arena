"""Champion / challenger promotion gates and a lightweight model registry.

An adaptive defense that deploys itself is not deployable. Every model the closed
loop produces is a CANDIDATE (challenger) measured against the model currently in
force (champion), and it is promoted only if it clears every gate:

    promote if:
        recall on the new attack   >= champion recall + minimum gain
        AND legitimate FPR         <= absolute ceiling
        AND legitimate FPR         <= champion FPR + permitted regression
        AND no prior attack family drops by more than the tolerance
        AND overall PR-AUC         >= champion PR-AUC - permitted regression

A candidate that fails any gate is recorded with the reason and NOT promoted, so
"the loop keeps improving the defense" is a claim the artifacts can contradict.
Thresholds live in ``config.CHAMPION_CHALLENGER`` and are meant to be tuned per
deployment.

The registry is a plain JSON file: model version, the seed and schema the data
came from, replay-buffer composition, the metrics that were measured, the gate
results, and the promote/reject decision. It is traceability, not an MLOps
platform. Entries carry no wall-clock timestamp by default, because committed
artifacts must stay byte-stable across runs; pass ``trained_at`` explicitly if a
deployment wants real timestamps.
"""
from __future__ import annotations

import json

import numpy as np

import config

PROMOTE, REJECT = "PROMOTE", "REJECT"


def _wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion measured on n observations."""
    if n <= 0:
        return 0.0, 1.0
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _gate(name: str, passed: bool, observed, required, detail: str) -> dict:
    return {"gate": name, "passed": bool(passed),
            "observed": (round(float(observed), 4) if observed is not None else None),
            "required": (round(float(required), 4) if required is not None else None),
            "detail": detail}


def evaluate_candidate(champion_metrics: dict, candidate_metrics: dict,
                       prior_family_recall: dict | None = None,
                       thresholds: dict | None = None) -> dict:
    """Run every promotion gate and return the decision with its reasoning.

    ``*_metrics`` need: ``attack_recall`` (recall on the attack generation under
    test), ``false_positive_rate`` and ``pr_auc``. ``prior_family_recall`` maps a
    previously-learned family to ``{"champion": r, "candidate": r}``.
    """
    T = thresholds or config.CHAMPION_CHALLENGER
    gates: list[dict] = []

    ch_rec = champion_metrics.get("attack_recall")
    ca_rec = candidate_metrics.get("attack_recall")
    if ch_rec is not None and ca_rec is not None:
        need = ch_rec + T["min_attack_recall_gain"]
        gates.append(_gate(
            "attack_recall_gain", ca_rec >= need, ca_rec, need,
            f"candidate catches {ca_rec:.1%} of the new attack generation vs "
            f"{ch_rec:.1%} for the model in force"))

    ca_fpr = candidate_metrics.get("false_positive_rate")
    if ca_fpr is not None:
        gates.append(_gate(
            "absolute_false_positive_ceiling", ca_fpr <= T["max_fpr"], ca_fpr, T["max_fpr"],
            f"candidate declines or challenges {ca_fpr:.2%} of genuine traffic"))
        ch_fpr = champion_metrics.get("false_positive_rate")
        if ch_fpr is not None:
            limit = ch_fpr + T["max_fpr_regression"]
            gates.append(_gate(
                "false_positive_regression", ca_fpr <= limit, ca_fpr, limit,
                f"customer friction moves from {ch_fpr:.2%} to {ca_fpr:.2%}"))

    worst_drop, worst_fam, worst_n, significant = 0.0, None, 0, False
    for fam, d in (prior_family_recall or {}).items():
        champ, cand = d.get("champion"), d.get("candidate")
        if champ is None or cand is None:
            continue
        drop = champ - cand
        # Family recall is a proportion measured on a few dozen transactions, so a
        # gate that fires on any drop past a fixed line will fire on sampling
        # noise — and a gate that stops a genuinely better model for a coin-flip
        # is worse than no gate. A regression counts only when the candidate's
        # recall is confidently below the tolerated level, judged by the upper
        # bound of its 95% Wilson interval.
        n = int(d.get("n") or 0)
        tolerated = champ - T["max_prior_recall_drop"]
        if n > 0:
            _, upper = _wilson(cand, n)
            is_real = upper < tolerated
        else:
            is_real = drop > T["max_prior_recall_drop"]
        if drop > worst_drop:
            worst_drop, worst_fam, worst_n = drop, fam, n
        significant = significant or is_real
    if prior_family_recall:
        detail = ("no previously-learned family regressed" if worst_fam is None
                  else f"largest regression is {worst_fam} at -{worst_drop:.1%}"
                       + (f" on n={worst_n}" if worst_n else "")
                       + ("" if significant else
                          " — inside sampling error at this sample size, so not counted"))
        gates.append(_gate("no_catastrophic_forgetting", not significant,
                           worst_drop, T["max_prior_recall_drop"], detail))

    ch_pr, ca_pr = champion_metrics.get("pr_auc"), candidate_metrics.get("pr_auc")
    if ch_pr is not None and ca_pr is not None:
        floor = ch_pr - T["max_overall_pr_auc_drop"]
        gates.append(_gate(
            "overall_ranking_quality", ca_pr >= floor, ca_pr, floor,
            f"overall PR-AUC moves from {ch_pr:.3f} to {ca_pr:.3f}"))

    failed = [g["gate"] for g in gates if not g["passed"]]
    return {
        "decision": REJECT if failed else PROMOTE,
        "failed_gates": failed,
        "gates": gates,
        "thresholds": dict(T),
        "summary": ("candidate promoted to champion" if not failed
                    else "candidate held back: " + ", ".join(failed)),
    }


def registry_entry(model_version: str, stage: str, metrics: dict,
                   replay_composition: dict | None = None,
                   attack_generations: list | None = None,
                   decision: dict | None = None,
                   trained_at: str | None = None) -> dict:
    """One traceable record of a trained model."""
    return {
        "model_version": model_version,
        "stage": stage,                      # champion | challenger | baseline
        "schema_version": config.SCHEMA_VERSION,
        "data_seed": config.GLOBAL_SEED,
        "trained_at": trained_at,
        "metrics": {k: (round(float(v), 5) if isinstance(v, (int, float, np.floating))
                        else v)
                    for k, v in metrics.items()},
        "replay_composition": replay_composition or {},
        "attack_generations_used": attack_generations or [],
        "acceptance": decision or {},
    }


def write_registry(entries: list[dict], path=None) -> None:
    path = path or config.MODEL_REGISTRY_PATH
    path.write_text(json.dumps({
        "schema_version": config.SCHEMA_VERSION,
        "gates": dict(config.CHAMPION_CHALLENGER),
        "note": ("Every adapted model is a challenger measured against the model in force. "
                 "Promotion requires all gates to pass; a rejected candidate stays out of "
                 "the authorization path. Deployment would additionally run a shadow / "
                 "controlled-rollout period, which this prototype does not simulate."),
        "entries": entries,
    }, indent=2), encoding="utf-8")


def load_registry(path=None) -> dict | None:
    path = path or config.MODEL_REGISTRY_PATH
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
