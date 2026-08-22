"""Deterministic end-to-end artifact regeneration.

Rebuilds every derived artifact from the seeded simulator so a fresh clone can
reproduce the committed metrics/data.

    python -m src.pipeline            # everything (incl. the closed loop)
    python -m src.pipeline --fast     # skip the (slow) closed-loop retraining
"""
from __future__ import annotations

import argparse
import json

import config

N_STEPS = 8


def regenerate(include_loop: bool = True) -> dict:
    out = {"schema_version": config.SCHEMA_VERSION, "seed": config.GLOBAL_SEED}

    print(f"[1/{N_STEPS}] Simulating labeled dataset + GenAI artifact corpus…")
    from src.generate.simulate import run_and_save
    summary = run_and_save()
    out["dataset"] = {k: v for k, v in summary.items() if k != "attack_specs"}

    print(f"[2/{N_STEPS}] Synthetic fidelity diagnostics…")
    from src.generate.fidelity import run_fidelity
    fid = run_fidelity()
    out["fidelity"] = {"max_single_feature_auc": fid["max_single_feature_auc"],
                       "n_fail": fid["n_fail"], "n_warn": fid["n_warn"]}
    if fid["n_fail"]:
        failing = [c["check"] for c in fid["checks"] if c["status"] == "FAIL"]
        print(f"       WARNING: {fid['n_fail']} fidelity check(s) failing: {failing}")

    print(f"[3/{N_STEPS}] Training defense model (authorization-time features)…")
    from src.defend.train import train_and_eval
    _, metrics = train_and_eval()
    out["metrics"] = {k: metrics[k] for k in ("recall", "precision", "f1", "roc_auc",
                                              "pr_auc", "false_positive_rate")}

    print(f"[4/{N_STEPS}] Training text arm (synthetic sanity check, not a detection result)…")
    from src.defend.text_model import train_text_model
    _, tm = train_text_model()
    # Keyed so the score cannot be read as a peer of the detection metrics above,
    # and carrying its own caveat so no consumer can quote it bare.
    out["text_sanity_check_not_detection_evidence"] = {
        "roc_auc": tm["roc_auc"], "roc_auc_cv5": tm["roc_auc_cv5"],
        "n_corpus": tm["n_corpus"], "trivially_separable": tm["trivially_separable"],
        "honest_reading": tm["honest_reading"],
    }

    # Leave-one-out runs BEFORE the loop on purpose. It is the experiment that
    # establishes which families are learnable at authorization time at all, and
    # the loop reads that evidence to decide which families are worth attacking.
    print(f"[5/{N_STEPS}] Leave-one-attack-family-out evaluation…")
    from src.experiments.leave_one_out import run_loao
    out["loao"] = run_loao()["families"]

    if include_loop:
        print(f"[6/{N_STEPS}] Running weakness-driven adaptive loop…")
        from src.loop.redteam_loop import run_loop, unlearnable_families
        excluded, _ = unlearnable_families()
        print(f"       excluded from targeting as structural frontiers: {sorted(excluded)}")
        loop = run_loop(rounds=3, n_base=45000, n_round=20000, n_eval=24000)
        out["loop_rounds"] = loop["rounds"]
        out["loop_focus"] = loop["focus"]
        out["loop_families_attacked"] = loop.get("families_attacked")
        out["loop_retired"] = [r["family"] for r in loop.get("retired_frontiers", [])]
        out["loop_promotions"] = loop["promotion"]["promoted_rounds"]
    else:
        print(f"[6/{N_STEPS}] Skipping closed loop (--fast).")

    print(f"[7/{N_STEPS}] Rules vs static vs adaptive baseline comparison…")
    from src.defend.baseline import compare
    from src.defend.model import DefenseModel
    # The adaptive column is the PROMOTED champion. The final candidate is
    # reported separately and labelled, so a model the governance gates turned
    # down is never presented as the working defense.
    models = {"static_ml": DefenseModel.load()}
    if config.LOOP_CHAMPION_MODEL_PATH.exists():
        models["adaptive_ml"] = DefenseModel.load(config.LOOP_CHAMPION_MODEL_PATH)
    if config.LOOP_ADAPTED_MODEL_PATH.exists():
        models["adaptive_candidate_unpromoted"] = DefenseModel.load(
            config.LOOP_ADAPTED_MODEL_PATH)
    comp = compare(models=models)
    config.BASELINE_COMPARISON_PATH.write_text(json.dumps(comp, indent=2), encoding="utf-8")
    out["baseline"] = {k: {"recall": v["recall"], "fpr": v["false_positive_rate"]}
                       for k, v in comp.items() if not k.startswith("_")}
    out["baseline_fpr_matched"] = {
        k: {"recall": v["recall"], "precision": v["precision"],
            "fpr": v["false_positive_rate"]}
        for k, v in comp.get("_fpr_matched", {}).get("models", {}).items()}

    print(f"[8/{N_STEPS}] Thresholds, calibration, operational metrics, blind spots, "
          "head-to-head…")
    from src.defend.diagnostics import run_all
    diag = run_all()
    out["calibration"] = {"brier_raw": diag["calibration"]["brier_raw"],
                          "brier_calibrated": diag["calibration"]["brier_calibrated"]}
    out["operational"] = {k: v["per_1000"] for k, v in diag["operational"].items()}
    out["blind_spots"] = [d["family"] for d in diag["blind_spots"]["hardest"]]

    (config.MODELS_DIR / "pipeline_summary.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("\nDone. Summary written to models/pipeline_summary.json")
    return out


def main():
    ap = argparse.ArgumentParser(description="Regenerate all artifacts deterministically.")
    ap.add_argument("--fast", action="store_true", help="skip the slow closed-loop step")
    args = ap.parse_args()
    regenerate(include_loop=not args.fast)


if __name__ == "__main__":
    main()
