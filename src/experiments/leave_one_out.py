"""Leave-one-attack-family-out (LOAO) evaluation.

Empirical evidence for WHY an adaptive defense is needed: a static supervised
model can be weak against a fraud family it never trained on. For each family we
  1. train a model with that family REMOVED from training,
  2. measure recall on that (unseen) family in a held-out test set,
  3. train a model WITH the family included,
  4. measure the recall gain.

Large gains = the family is learnable but genuinely unseen-hard, which is exactly
the gap the closed-loop replay fills. Small gains on an already-easy family, or a
low ceiling even when included, are reported honestly.

CLI:  python -m src.experiments.leave_one_out
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config
from src.defend.evaluate import evaluate
from src.defend.features import build_xy
from src.defend.model import DefenseModel
from src.defend.train import split_xy
from src.generate.simulate import simulate

# Families with enough samples for a meaningful LOAO read.
DEFAULT_FAMILIES = [
    "card_testing", "account_takeover", "otp_relay", "scam_transfer",
    "adversarial_mimicry", "merchant_laundering", "velocity_smurfing", "bust_out",
    "wallet_provisioning", "geo_anomaly",
]


def _wilson(p, n):
    from src.defend.evaluate import wilson_interval
    return wilson_interval(p, n)


def select_hero_family(loao: dict | None) -> tuple[str | None, dict | None, bool]:
    """Pick the family to lead with: the largest measured gain among families that
    clear the sample-size floor.

    Returns ``(family, record, underpowered)``. ``underpowered`` is True when no
    family cleared the floor and the choice fell back to the full set — the caller
    must then refuse to headline the number, because a point estimate on a dozen
    transactions carries an interval no headline can support.

    This lives in one place on purpose. The rule was previously duplicated across
    the README generator, the landing page and the hero-demo page, each with a
    silent ``or`` fallback that would have re-admitted an n=16 family the moment
    the floor stopped being met — and all three would have flipped together,
    invisibly.
    """
    if not loao or not loao.get("families"):
        return None, None, False
    floor = config.FAMILY_EVAL["min_n_to_report"]
    scored = {f: r for f, r in loao["families"].items() if r.get("gain") is not None}
    eligible = {f: r for f, r in scored.items()
                if r.get("sufficient_n", r.get("n_test", 0) >= floor)}
    if eligible:
        f = max(eligible, key=lambda k: eligible[k]["gain"])
        return f, eligible[f], False
    if not scored:
        return None, None, False
    f = max(scored, key=lambda k: scored[k]["gain"])
    return f, scored[f], True


def _recall_on(model: DefenseModel, X, y, attack, family) -> tuple[float | None, int]:
    m = (attack == family) & (y == 1)
    n = int(m.sum())
    if n == 0:
        return None, 0
    flagged = model.fused_scores(X.loc[m]) >= model.threshold
    return float(flagged.mean()), n


def run_loao(families: list[str] | None = None, n: int = 90_000,
             fraud_rate: float = 0.035, seed: int = config.GLOBAL_SEED,
             save: bool = True) -> dict:
    """The frame is deliberately larger and richer in fraud than the headline
    dataset: a per-family recall claim needs enough held-out examples of that
    family to mean anything, and the smallest families would otherwise land at a
    couple of dozen rows."""
    families = families or DEFAULT_FAMILIES
    df = simulate(n_transactions=n, fraud_rate=fraud_rate, seed=seed)[0]
    parts = split_xy(df)
    X_tr, y_tr, atk_tr = parts["train"]
    X_val, y_val, _ = parts["val"]
    X_te, y_te, atk_te = parts["test"]

    results = {}
    for fam in families:
        # (1) train EXCLUDING this family's fraud (legit + other fraud kept)
        keep = ~((atk_tr == fam) & (y_tr == 1))
        m_excl = DefenseModel().fit(X_tr[keep], y_tr[keep])
        m_excl.fit_calibration(X_val, y_val)
        m_excl.tune_threshold(X_val, y_val, target_fpr=config.TARGET_MAX_FPR)
        unseen_recall, n_fam = _recall_on(m_excl, X_te, y_te, atk_te, fam)

        # (2) train INCLUDING the family (full training set)
        m_incl = DefenseModel().fit(X_tr, y_tr)
        m_incl.fit_calibration(X_val, y_val)
        m_incl.tune_threshold(X_val, y_val, target_fpr=config.TARGET_MAX_FPR)
        seen_recall, _ = _recall_on(m_incl, X_te, y_te, atk_te, fam)

        results[fam] = {
            "n_test": n_fam,
            "sufficient_n": bool(n_fam >= config.FAMILY_EVAL["min_n_to_report"]),
            "recall_unseen": None if unseen_recall is None else round(unseen_recall, 4),
            "recall_unseen_ci95": _wilson(unseen_recall, n_fam),
            "recall_after_learning": None if seen_recall is None else round(seen_recall, 4),
            "recall_after_learning_ci95": _wilson(seen_recall, n_fam),
            "gain": (None if (unseen_recall is None or seen_recall is None)
                     else round(seen_recall - unseen_recall, 4)),
        }

    out = {"n_transactions": int(len(df)), "seed": seed,
           "schema_version": config.SCHEMA_VERSION,
           "families": results,
           "interpretation": ("recall_unseen = static model that never saw the family; "
                              "recall_after_learning = same pipeline once the family is in "
                              "training. Large gain => unseen-hard but learnable => motivates "
                              "the adaptive replay loop."),
           "what_this_does_not_show": (
               "This measures UNSEEN -> LEARNED on synthetic data, not zero-shot detection and "
               "not production performance. recall_after_learning is an upper bound obtained by "
               "putting the family into training directly; the closed loop has to reach it by "
               "generating the family itself. Families with a small n_test carry wide "
               "uncertainty and are reported with their sample size for that reason.")}
    if save:
        (config.MODELS_DIR / "leave_one_out.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    out = run_loao()
    print(f"{'family':22s} {'unseen':>8s} {'learned':>8s} {'gain':>7s}  n")
    for fam, r in out["families"].items():
        u = "n/a" if r["recall_unseen"] is None else f"{r['recall_unseen']:.2f}"
        s = "n/a" if r["recall_after_learning"] is None else f"{r['recall_after_learning']:.2f}"
        g = "n/a" if r["gain"] is None else f"{r['gain']:+.2f}"
        print(f"{fam:22s} {u:>8s} {s:>8s} {g:>7s}  {r['n_test']}")
    print(f"\nSaved {config.MODELS_DIR / 'leave_one_out.json'}")


if __name__ == "__main__":
    main()
