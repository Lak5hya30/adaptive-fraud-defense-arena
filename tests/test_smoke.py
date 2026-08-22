"""End-to-end smoke test: taxonomy -> generate -> features -> train -> score -> loop.

Runs at small scale so it's fast. Verifies the whole closed loop wires together
and produces sane, non-degenerate outputs. Works fully offline.

    python -m pytest tests/ -q
    python tests/test_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.defend.evaluate import evaluate
from src.defend.features import FEATURE_COLUMNS, build_xy
from src.defend.model import DefenseModel
from src.defend.train import split_xy
from src.generate.attack_injectors import INJECTORS
from src.generate.llm_agent import propose_evasion
from src.generate.simulate import simulate
from src.identify.taxonomy import load_taxonomy


def test_taxonomy_loads_and_validates():
    tax = load_taxonomy()
    assert len(tax.attacks) >= 40
    # breadth claims in the UI and the README come from these counts
    assert len(tax.categories) >= 5
    assert tax.summary_counts()["research_only"] > 0, (
        "a catalog with nothing marked research-only is claiming to simulate everything")
    # every declared injector is actually implemented
    for inj in tax.injectors:
        assert inj in INJECTORS
    # every simulatable attack's injector exists
    for a in tax.attacks:
        if a.maps_to_injector:
            assert a.maps_to_injector in INJECTORS


def test_simulation_is_labeled_and_covers_all_injectors():
    df, summary = simulate(n_transactions=8000, fraud_rate=0.02, seed=7)
    assert len(df) > 7000
    assert set(df["is_fraud"].unique()) == {0, 1}
    assert 0.01 < df["is_fraud"].mean() < 0.05
    present = set(df.loc[df.is_fraud == 1, "attack_type"].unique())
    assert present == set(INJECTORS.keys()), present  # all families represented


def test_features_are_finite_and_complete():
    df, _ = simulate(n_transactions=6000, fraud_rate=0.02, seed=11)
    X, y, atk = build_xy(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(df) == len(y) == len(atk)
    assert np.isfinite(X.to_numpy()).all()


def test_train_and_score_is_discriminative():
    df, _ = simulate(n_transactions=20000, fraud_rate=0.025, seed=13)
    parts = split_xy(df)
    X_tr, y_tr, _ = parts["train"]
    X_val, y_val, _ = parts["val"]
    X_te, y_te, atk_te = parts["test"]

    model = DefenseModel().fit(X_tr, y_tr)
    model.fit_calibration(X_val, y_val)
    model.tune_threshold(X_val, y_val)
    m = evaluate(model, X_te, y_te, atk_te)

    # Bounds reflect data with genuine legit/fraud overlap and no planted
    # shortcuts, so they are deliberately modest. They are loose enough to be
    # robust to this small sample; the committed model's real numbers are in
    # models/metrics.json.
    assert m["roc_auc"] > 0.75         # still clearly discriminative overall
    assert m["recall"] > 0.3           # catches a meaningful share of fraud
    assert m["pr_auc"] > 0.15          # meaningful at this base rate
    assert m["false_positive_rate"] < 0.06


def test_threshold_tuning_honours_the_false_positive_budget():
    """Regression test for a real bug: calibrated scores are a step function, so a
    quantile threshold lands inside a tied block and `>=` admits the whole block —
    spending several times the agreed budget without saying so."""
    df, _ = simulate(n_transactions=20000, fraud_rate=0.025, seed=17)
    parts = split_xy(df)
    X_tr, y_tr, _ = parts["train"]
    X_val, y_val, _ = parts["val"]

    model = DefenseModel().fit(X_tr, y_tr)
    model.fit_calibration(X_val, y_val)
    for budget in (0.005, 0.02, 0.05):
        model.tune_threshold(X_val, y_val, target_fpr=budget)
        scores = model.fused_scores(X_val)
        realized = float((scores[y_val == 0] >= model.threshold).mean())
        assert realized <= budget + 1e-9, (
            f"budget {budget:.3f} but realized {realized:.4f} on the tuning set")


def test_calibration_does_not_change_the_ranking():
    """Isotonic calibration is monotone, so it must leave ROC-AUC untouched. If it
    does not, the score being reported is not the score being ranked."""
    from sklearn.metrics import roc_auc_score
    df, _ = simulate(n_transactions=15000, fraud_rate=0.025, seed=19)
    parts = split_xy(df)
    X_tr, y_tr, _ = parts["train"]
    X_val, y_val, _ = parts["val"]
    X_te, y_te, _ = parts["test"]

    model = DefenseModel().fit(X_tr, y_tr)
    before = roc_auc_score(y_te, model.fused_scores(X_te))
    model.fit_calibration(X_val, y_val)
    after = roc_auc_score(y_te, model.risk_probability(X_te))
    assert abs(before - after) < 0.02, (
        f"calibration moved ROC-AUC from {before:.4f} to {after:.4f}")


def test_evasion_only_escalates_stealth():
    ev = propose_evasion("adversarial_mimicry", recall=0.9, current_intensity=1.0)
    assert 0.2 <= ev["intensity"] <= 1.0
    assert ev["intensity"] <= 1.0
    # a second escalation is never less stealthy than the first
    ev2 = propose_evasion("adversarial_mimicry", recall=0.9, current_intensity=ev["intensity"])
    assert ev2["intensity"] <= ev["intensity"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nAll smoke tests passed.")
