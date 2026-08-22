"""Operational diagnostics: threshold trade-offs, calibration, cost-sensitive
volumes, and a standing register of the defense's blind spots.

Accuracy alone does not tell a payments team whether a model can be run. What
they need to know is how much genuine traffic it will challenge, how many reviews
that implies, whether the probability on the screen means anything, and where the
model is still weak. Each of those is computed here from measured results and
committed as an artifact, so the web prototype and the walkthrough quote the same
numbers.

The cost figures use ILLUSTRATIVE SYNTHETIC ASSUMPTIONS declared in
``config.OPERATIONAL_SCENARIO``. They are not anyone's real economics and are
labelled as such everywhere they appear.

CLI:  python -m src.defend.diagnostics
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config
from src.defend.decision_policy import decide, policy_summary
from src.defend.features import build_xy
from src.defend.model import DefenseModel


# --------------------------------------------------------------------------- #
# Threshold exploration
# --------------------------------------------------------------------------- #
def threshold_sweep(model: DefenseModel, X: pd.DataFrame, y: np.ndarray,
                    n_points: int = 60) -> dict:
    """Recall / precision / false-positive rate across the whole threshold range.

    There is no magic threshold. This is the curve a fraud team actually argues
    over: every point on it is a different answer to "how much genuine friction
    are we willing to buy detection with".
    """
    scores = model.fused_scores(X)
    legit = y == 0
    n_legit = int(legit.sum())
    n_fraud = int((y == 1).sum())
    # Sample thresholds on the score quantiles so the curve is dense where the
    # decisions actually happen.
    qs = np.linspace(0.0, 1.0, n_points)
    cand = np.unique(np.quantile(scores, qs))
    points = []
    for t in cand:
        pred = scores >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & legit).sum())
        points.append({
            "threshold": round(float(t), 5),
            "recall": round(tp / max(1, n_fraud), 4),
            "precision": round(tp / max(1, tp + fp), 4),
            "false_positive_rate": round(fp / max(1, n_legit), 5),
            "false_positives_per_1000_legit": round(fp / max(1, n_legit) * 1000, 2),
            "flagged_share": round(float(pred.mean()), 5),
        })
    from src.defend.decision_policy import resolve_thresholds
    return {
        "points": points,
        "operating_point": round(float(model.threshold), 5),
        "operating_point_probability": round(float(model.threshold_probability), 5),
        "decision_thresholds": resolve_thresholds(step_up=model.threshold_probability),
        "fpr_budget": config.TARGET_MAX_FPR,
        "n_legit": n_legit, "n_fraud": n_fraud,
        "note": ("The operating point is chosen as the lowest threshold whose false-positive "
                 "rate on genuine traffic stays inside the configured budget, tuned on a "
                 "held-out validation slice rather than on the test set."),
    }


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def calibration_report(model: DefenseModel, X: pd.DataFrame, y: np.ndarray,
                       bins: int = 10) -> dict:
    """Reliability curve and Brier score, before and after calibration."""
    raw = model.fused_scores(X)
    cal = model.risk_probability(X)

    def _curve(scores):
        edges = np.quantile(scores, np.linspace(0, 1, bins + 1))
        edges = np.unique(edges)
        out = []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            m = (scores >= lo) & (scores <= hi if i == len(edges) - 2 else scores < hi)
            if m.sum() < 5:
                continue
            out.append({"predicted": round(float(scores[m].mean()), 5),
                        "observed": round(float(y[m].mean()), 5),
                        "n": int(m.sum())})
        return out

    brier_raw = float(np.mean((raw - y) ** 2))
    brier_cal = float(np.mean((cal - y) ** 2))
    return {
        "calibrated": model.calibrator is not None,
        "brier_raw": round(brier_raw, 6),
        "brier_calibrated": round(brier_cal, 6),
        "brier_improvement": round(brier_raw - brier_cal, 6),
        "reliability_raw": _curve(raw),
        "reliability_calibrated": _curve(cal),
        "note": ("Isotonic calibration is fitted on the held-out validation slice. It is "
                 "monotone, so it cannot change the model's ranking: ROC-AUC and PR-AUC are "
                 "identical either way, and only the meaning of the displayed number changes. "
                 "Detection still thresholds the raw fused score, because a calibrated score "
                 "is a step function and cannot hit a false-positive budget precisely. At a "
                 "~1% base rate a low Brier score is easy to obtain by predicting near zero "
                 "everywhere, so the reliability curve matters more than the scalar."),
    }


# --------------------------------------------------------------------------- #
# Operational volumes and cost scenario
# --------------------------------------------------------------------------- #
def operational_metrics(model: DefenseModel, X: pd.DataFrame, y: np.ndarray,
                        label: str = "static_ml") -> dict:
    """Translate rates into the volumes an operations team would actually staff."""
    S = config.OPERATIONAL_SCENARIO
    scores = model.fused_scores(X)
    probs = model.risk_probability(X)
    actions = decide(probs, step_up=model.threshold_probability)
    legit, fraud = y == 0, y == 1
    n_legit, n_fraud = int(legit.sum()), int(fraud.sum())

    flagged = scores >= model.threshold
    fp = int((flagged & legit).sum())
    tp = int((flagged & fraud).sum())
    fpr = fp / max(1, n_legit)
    recall = tp / max(1, n_fraud)

    step_up_share = float((actions == "STEP_UP").mean())
    decline_share = float((actions == "DECLINE").mean())
    approve_share = float((actions == "APPROVE").mean())
    legit_stepped = float(((actions == "STEP_UP") & legit).mean() / max(1e-9, legit.mean()))
    legit_declined = float(((actions == "DECLINE") & legit).mean() / max(1e-9, legit.mean()))

    monthly = S["monthly_authorizations"]
    monthly_reviews = monthly * step_up_share
    review_hours = monthly_reviews / max(1, S["analyst_reviews_per_hour"])

    return {
        "label": label,
        "per_1000": {
            "false_positives_per_1000_legit": round(fpr * 1000, 2),
            "fraud_caught_per_1000_fraud": round(recall * 1000, 1),
            "flagged_per_1000_all": round(float(flagged.mean()) * 1000, 2),
        },
        "action_distribution": {
            "approve": round(approve_share, 4),
            "step_up": round(step_up_share, 4),
            "decline": round(decline_share, 4),
            "genuine_customers_stepped_up": round(legit_stepped, 4),
            "genuine_customers_declined": round(legit_declined, 4),
        },
        "scenario": {
            "label": S["label"],
            "monthly_authorizations": monthly,
            "monthly_step_up_challenges": int(round(monthly_reviews)),
            "monthly_review_hours_if_all_manually_reviewed": int(round(review_hours)),
            "estimated_genuine_customers_abandoning_after_step_up":
                int(round(monthly * legit_stepped * S["step_up_abandonment_rate"])),
        },
        "policy": policy_summary(probs, y, step_up=model.threshold_probability),
        "disclaimer": ("Volumes are derived by applying measured rates to the declared "
                       "synthetic scenario. They illustrate what a false-positive rate costs "
                       "operationally; they are not a forecast for any real portfolio."),
    }


# --------------------------------------------------------------------------- #
# Blind spots
# --------------------------------------------------------------------------- #
def blind_spots(model: DefenseModel, X: pd.DataFrame, y: np.ndarray,
                attack: np.ndarray, min_n: int = 10, top_k: int = 4) -> dict:
    """Where the defense is still weakest, and what the escaped traffic looks like.

    Deliberately published rather than buried: a defense that cannot name its own
    blind spots is not being evaluated honestly, and the next red-team target is
    chosen from this list.
    """
    from src.loop.weakness import analyze_family, weakest_families

    ranked = weakest_families(model, X, y, attack, min_n=min_n)
    detail = []
    for row in ranked[:top_k]:
        rep = analyze_family(model, X, y, attack, row["family"])
        detail.append({
            "family": row["family"], "recall": row["recall"], "n": row["n"],
            "n_escaped": rep.n_escaped,
            "relied_signals": rep.relied_signals[:4],
            "escape_profile": rep.escape_profile[:4],
        })
    nxt = detail[0] if detail else None
    suggestion = None
    if nxt:
        from src.generate.attack_spec import BASE_SPECS
        from src.loop.weakness import heuristic_mutation
        rep = analyze_family(model, X, y, attack, nxt["family"])
        base = BASE_SPECS.get(nxt["family"])
        if base is not None:
            mut = heuristic_mutation(rep, base)
            suggestion = {"family": nxt["family"], "proposed_change": {
                k: v for k, v in mut.items()
                if k in ("amount_profile", "velocity_profile", "device_behavior",
                         "geo_behavior", "merchant_behavior", "timing_profile", "intensity")},
                "targets_signal": mut.get("targets_signal"),
                "strategy": mut.get("strategy")}
    return {
        "ranked_families": ranked,
        "hardest": detail,
        "next_red_team_target": suggestion,
        "note": ("Recall is measured at the model's operating threshold on held-out synthetic "
                 "traffic. Families whose fraud is authorized by the genuine customer are "
                 "expected to sit near the bottom: that is a property of authorization-time "
                 "observability, not a bug to be tuned away."),
    }


# --------------------------------------------------------------------------- #
# Per-family recall at a sample size worth quoting
# --------------------------------------------------------------------------- #
def family_recall_frame(models: dict[str, DefenseModel] | None = None,
                        save: bool = True) -> dict:
    """Measure per-family recall on a large, fraud-enriched, UNSEEN frame.

    The headline test slice holds a realistic 1.2% base rate, which is right for
    reporting overall precision and false positives — and useless for saying
    anything about a family that appears twenty times in it. This frame is
    generated from a seed the models have never seen, so the cardholders,
    merchants and devices are a different synthetic portfolio entirely: it
    measures whether the defense generalizes, not whether it memorized.
    """
    from src.defend.features import build_xy
    from src.generate.simulate import simulate

    F = config.FAMILY_EVAL
    if models is None:
        models = {"static_ml": DefenseModel.load()}
        if config.LOOP_ADAPTED_MODEL_PATH.exists():
            models["adaptive_ml"] = DefenseModel.load(config.LOOP_ADAPTED_MODEL_PATH)

    df, _ = simulate(n_transactions=F["n_transactions"], fraud_rate=F["fraud_rate"],
                     seed=config.GLOBAL_SEED + F["seed_offset"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    X, y, atk = build_xy(df)

    out = {
        "frame": {"n_transactions": int(len(df)), "n_fraud": int(y.sum()),
                  "fraud_rate": F["fraud_rate"],
                  "seed": config.GLOBAL_SEED + F["seed_offset"]},
        "note": ("Generated from a seed no model was trained or tuned on, so the cardholders, "
                 "merchants and devices are an unseen synthetic portfolio. Fraud is enriched to "
                 f"{F['fraud_rate']:.1%} purely so each family has enough held-out examples to "
                 "quote; overall precision and false-positive figures are reported on the "
                 "realistic-base-rate test slice instead, never on this frame."),
        "min_n_to_report": F["min_n_to_report"],
        "models": {},
    }
    from src.defend.baseline import rules_predict
    r_pred = rules_predict(X)
    per_model = {"rules_baseline": r_pred}
    for name, m in models.items():
        per_model[name] = (m.fused_scores(X) >= m.threshold).astype(int)

    for name, pred in per_model.items():
        fam = {}
        for a in sorted({v for v in atk if v != "legit"}):
            mask = (atk == a) & (y == 1)
            n = int(mask.sum())
            if n == 0:
                continue
            fam[a] = {"recall": round(float(pred[mask].mean()), 4), "n": n,
                      "sufficient_n": bool(n >= F["min_n_to_report"])}
        legit = y == 0
        out["models"][name] = {
            "per_family_recall": fam,
            "overall_recall_on_this_frame": round(float(pred[y == 1].mean()), 4),
            "false_positive_rate_on_this_frame": round(float(pred[legit].mean()), 5),
        }
    if save:
        config.FAMILY_RECALL_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# Static vs adaptive on the attacks that actually moved
# --------------------------------------------------------------------------- #
def head_to_head(save: bool = True) -> dict | None:
    """Score the pre-loop defense and the loop-trained defense on the FINAL
    evolved attack generation.

    This is the measurement the whole project stands on. Every other comparison
    scores detectors on the original attack distribution, which is the static
    model's home ground — it was trained on exactly that, so of course it wins
    there. The question adaptation exists to answer is what happens once the
    attack has moved, and that is only visible on the evolved generation.

    Both models are loaded from the artifacts the loop saved, and the evolved
    generation is regenerated from the specifications recorded in the loop
    history, on a seed neither model has seen.
    """
    import json as _json

    if not (config.LOOP_HISTORY_PATH.exists() and config.LOOP_BASE_MODEL_PATH.exists()
            and config.LOOP_ADAPTED_MODEL_PATH.exists()):
        return None

    from src.defend.features import build_xy
    from src.generate.attack_spec import AttackSpec
    from src.generate.simulate import simulate
    from src.loop.redteam_loop import _focus_mix

    loop = _json.loads(config.LOOP_HISTORY_PATH.read_text(encoding="utf-8"))
    last = loop["history"][-1]
    # The families under attack in the FINAL round — which may differ from where
    # the loop started, because exhausted frontiers are retired and replaced.
    focus = last.get("focus") or loop["focus"]

    fields = set(AttackSpec.__dataclass_fields__)
    specs = {}
    for fam, d in last["specs"].items():
        keep = {k: v for k, v in d.items() if k in fields}
        keep["attack_family"] = fam
        specs[fam] = AttackSpec(**keep)

    df, _ = simulate(n_transactions=config.FAMILY_EVAL["n_transactions"],
                     fraud_rate=0.03, specs=specs, mix=_focus_mix(focus),
                     seed=config.GLOBAL_SEED + 606_060)
    df = df.sort_values("timestamp").reset_index(drop=True)
    X, y, atk = build_xy(df)

    models = {
        "static_defense": DefenseModel.load(config.LOOP_BASE_MODEL_PATH),
        "adaptive_defense": DefenseModel.load(config.LOOP_ADAPTED_MODEL_PATH),
    }
    if config.LOOP_CHAMPION_MODEL_PATH.exists():
        models["promoted_champion"] = DefenseModel.load(config.LOOP_CHAMPION_MODEL_PATH)

    # Two defenses at their own operating points are not comparable: whichever
    # one happens to sit stricter will catch less, and the difference says
    # nothing about which is the better detector. So each model is additionally
    # re-thresholded to spend the SAME false-positive budget, with the threshold
    # chosen on a separate generation-0 frame neither model has seen — never on
    # the evolved frame it is about to be scored on.
    from src.defend.baseline import _threshold_for_fpr
    from src.defend.evaluate import wilson_interval

    cal_df, _ = simulate(n_transactions=30_000, fraud_rate=0.03, specs={},
                         mix=_focus_mix(focus), seed=config.GLOBAL_SEED + 707_070)
    cal_df = cal_df.sort_values("timestamp").reset_index(drop=True)
    X_cal, y_cal, _ = build_xy(cal_df)

    out = {
        "frame": {"n_transactions": int(len(df)), "n_fraud": int(y.sum()),
                  "seed": config.GLOBAL_SEED + 606_060,
                  "generation": last["round"]},
        "focus": focus,
        "specs": {f: s.to_dict() for f, s in specs.items()},
        "matched_false_positive_rate": config.TARGET_MAX_FPR,
        "note": ("The static defense is the model trained before the loop ran — it has never "
                 "seen these evolved variants. The adaptive defense is the same architecture "
                 "trained through the loop on cumulative adversarial replay. Both are scored "
                 "on the same unseen frame of the FINAL evolved generation, and both are "
                 "additionally re-thresholded to spend the same false-positive budget, with "
                 "that threshold chosen on a separate generation-0 frame. The matched numbers "
                 "are the comparable ones."),
        "models": {},
    }

    def _score_block(m, threshold):
        flagged = m.fused_scores(X) >= threshold
        fam = {}
        for f in focus:
            mask = (atk == f) & (y == 1)
            n = int(mask.sum())
            if n == 0:
                continue
            r = float(flagged[mask].mean())
            fam[f] = {"recall": round(r, 4), "n": n, "ci95": wilson_interval(r, n)}
        return {
            "threshold": round(float(threshold), 6),
            "evolved_family_recall": fam,
            "mean_evolved_recall": (round(float(np.mean([v["recall"] for v in fam.values()])), 4)
                                    if fam else None),
            "overall_recall": round(float(flagged[y == 1].mean()), 4),
            "false_positive_rate": round(float(flagged[y == 0].mean()), 5),
        }

    for name, m in models.items():
        native = _score_block(m, m.threshold)
        t_matched = _threshold_for_fpr(m.fused_scores(X_cal), y_cal, config.TARGET_MAX_FPR)
        matched = _score_block(m, t_matched)
        out["models"][name] = {**matched, "native_operating_point": native}
    if save:
        config.HEAD_TO_HEAD_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
def run_all(save: bool = True) -> dict:
    from src.defend.train import load_dataset, split_xy

    df = load_dataset()
    parts = split_xy(df)
    X_te, y_te, atk_te = parts["test"]
    model = DefenseModel.load()

    sweep = threshold_sweep(model, X_te, y_te)
    calib = calibration_report(model, X_te, y_te)
    ops = operational_metrics(model, X_te, y_te, label="static_ml")
    spots = blind_spots(model, X_te, y_te, atk_te)

    if config.LOOP_ADAPTED_MODEL_PATH.exists():
        adapted = DefenseModel.load(config.LOOP_ADAPTED_MODEL_PATH)
        ops = {"static_ml": ops,
               "adaptive_ml": operational_metrics(adapted, X_te, y_te, label="adaptive_ml")}
    else:
        ops = {"static_ml": ops}

    fam = family_recall_frame(save=save)
    h2h = head_to_head(save=save)

    if save:
        # Precomputed scores, actions and reason codes for the held-out test
        # split. The web prototype reads this instead of rebuilding every feature
        # on each rerun — a demo that recomputes on stage is a demo that stalls.
        from src.defend.decision_policy import decision_frame
        test_df = parts["frames"]["test"]
        probs = model.risk_probability(X_te)
        demo = decision_frame(test_df, X_te, probs, step_up=model.threshold_probability)
        demo["detection_score"] = model.fused_scores(X_te)
        demo["flagged"] = (demo["detection_score"] >= model.threshold).astype(int)
        keep = ["timestamp", "card_id", "merchant_id", "mcc", "amount", "channel",
                "geo_city", "distance_from_home_km", "device_id", "is_new_payee",
                "otp_verified", "is_fraud", "attack_type", "actor_role",
                "risk_score", "detection_score", "flagged", "action", "reason_codes"]
        demo[[c for c in keep if c in demo.columns]].to_parquet(
            config.MODELS_DIR / "defend_demo.parquet", index=False)
        config.THRESHOLD_SWEEP_PATH.write_text(json.dumps(sweep, indent=2), encoding="utf-8")
        config.CALIBRATION_PATH.write_text(json.dumps(calib, indent=2), encoding="utf-8")
        config.OPERATIONAL_PATH.write_text(json.dumps(ops, indent=2), encoding="utf-8")
        config.BLIND_SPOTS_PATH.write_text(json.dumps(spots, indent=2), encoding="utf-8")
    return {"threshold_sweep": sweep, "calibration": calib, "operational": ops,
            "blind_spots": spots, "family_recall": fam, "head_to_head": h2h}


def main():
    out = run_all()
    c = out["calibration"]
    print(f"Calibration: Brier {c['brier_raw']:.5f} -> {c['brier_calibrated']:.5f} "
          f"(improvement {c['brier_improvement']:+.5f})")
    ops = out["operational"]["static_ml"]
    print(f"\nOperational (static ML): "
          f"{ops['per_1000']['false_positives_per_1000_legit']} false positives per 1,000 "
          f"genuine payments; {ops['action_distribution']['step_up']:.2%} stepped up, "
          f"{ops['action_distribution']['decline']:.2%} declined.")
    print("\nHardest families:")
    for d in out["blind_spots"]["hardest"]:
        print(f"  {d['family']:22s} recall={d['recall']:.2f} (n={d['n']}) "
              f"leans on {d['relied_signals'][0]['feature'] if d['relied_signals'] else 'n/a'}")
    print("\nPer-family recall on the unseen fraud-enriched frame:")
    fam = out["family_recall"]["models"].get("static_ml", {}).get("per_family_recall", {})
    for a, d in sorted(fam.items(), key=lambda kv: kv[1]["recall"]):
        mark = "" if d["sufficient_n"] else "  (n too small to quote)"
        print(f"  {a:22s} recall={d['recall']:.3f}  n={d['n']}{mark}")
    h2h = out.get("head_to_head")
    if h2h:
        print(f"\nOn the FINAL evolved generation (unseen frame, "
              f"{h2h['frame']['n_fraud']} fraudulent):")
        for name, blk in h2h["models"].items():
            print(f"  {name:20s} mean evolved recall={blk['mean_evolved_recall']}  "
                  f"FPR={blk['false_positive_rate']:.4f}")
    nxt = out["blind_spots"]["next_red_team_target"]
    if nxt:
        print(f"\nNext red-team target: {nxt['family']} — {nxt['strategy']}")
    print(f"\nSaved {config.THRESHOLD_SWEEP_PATH.name}, {config.CALIBRATION_PATH.name}, "
          f"{config.OPERATIONAL_PATH.name}, {config.BLIND_SPOTS_PATH.name}")


if __name__ == "__main__":
    main()
