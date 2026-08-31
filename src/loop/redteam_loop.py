"""Closed-loop red-team / blue-team engine with CUMULATIVE adversarial replay.

The core of the AI Defense Lab. One round is a full red-team experiment, not a
retraining step:

  1. EVALUATE   the defense currently in force on held-out traffic.
  2. LOCATE     its weakest attack families, and for each of them measure which
                signals its ranking actually depends on and what the transactions
                that escaped it look like.
  3. PROPOSE    the next attack generation as a structured specification — the
                GenAI red-team agent chooses which behavioural dial to move to
                remove the signal the defense leans on; offline, a weakness-driven
                heuristic makes the same call deterministically.
  4. CONSTRAIN  the specification against payment-domain rules, clamping or
                rejecting anything impossible before anything is generated.
  5. SIMULATE   the evolved generation, deterministically, from that specification.
  6. STRESS     the stale defense against it -> the gap the red team just opened.
  7. REPLAY     the escaped generation into a bounded, stratified replay buffer
                holding every prior generation.
  8. RETRAIN    a CANDIDATE defense on the base data plus the whole buffer.
  9. RE-EVALUATE the candidate on the new generation, on every prior generation
                (catastrophic forgetting), on families never attacked, and on
                genuine traffic (false positives).
 10. GOVERN     the candidate through champion/challenger gates. A candidate that
                fails is recorded and NOT promoted.
 11. CARRY      the residual weakness into the next round.

Everything is computed in FEATURE SPACE: each round is simulated as a full frame
so per-card history and network counters are correct, features are built once
over the ordered frame, and only the resulting feature rows enter the buffer.

Status per family is DERIVED FROM METRICS, never hardcoded:
  * adapted            - recall clears the residual ceiling AND gains >= recovery_delta
  * partial            - some gain but below target
  * residual_frontier  - recall stays below the residual ceiling after adaptation
A materially worse legit FPR is surfaced as a tradeoff rather than hidden.

CLI:  python -m src.loop.redteam_loop --rounds 3
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import config
from src.defend.decision_policy import decide
from src.defend.evaluate import evaluate
from src.defend.governance import (PROMOTE, evaluate_candidate, registry_entry,
                                   write_registry)
from src.defend.model import DefenseModel
from src.defend.train import split_xy
from src.generate.attack_injectors import DEFAULT_MIX
from src.generate.attack_spec import BASE_SPECS, spec_distance
from src.generate.llm_agent import propose_attack_spec
from src.generate.simulate import simulate
from src.loop.weakness import analyze_family, lineage_entry, weakest_families

# A family whose recall stays below this even when it is handed to the model
# directly in training is not learnable at authorization time by this feature set.
# Adversarial replay cannot beat that ceiling either, because replay is a slower
# route to the same thing: more examples of the family.
LEARNABILITY_FLOOR = 0.50

# Used when the leave-one-out artifact has not been generated yet.
STRUCTURAL_FRONTIERS = {"friendly_fraud", "adversarial_mimicry"}


def unlearnable_families(floor: float = LEARNABILITY_FLOOR) -> tuple[set[str], dict]:
    """Families excluded from red-team targeting, and the evidence for excluding them.

    The criterion is measured, not asserted: the leave-one-attack-family-out
    experiment trains a model WITH each family present, which is the most
    favourable case there is. A family that still cannot be caught under those
    conditions is limited by authorization-time observability, not by the decision
    boundary — the genuine customer authenticated and authorized, or the behaviour
    sits on the legitimate centroid. Aiming the loop at it produces a flat recovery
    curve and spends replay capacity that a family with a real residual signal
    could have used.

    Those families are still simulated, still scored, and still reported. They are
    just not pretended to be solvable by more of the same.
    """
    evidence: dict[str, dict] = {}
    excluded = set(STRUCTURAL_FRONTIERS)
    if config.LEAVE_ONE_OUT_PATH.exists():
        loao = json.loads(config.LEAVE_ONE_OUT_PATH.read_text(encoding="utf-8"))
        excluded = {"friendly_fraud"}
        for fam, r in loao.get("families", {}).items():
            learned = r.get("recall_after_learning")
            if learned is None:
                continue
            evidence[fam] = {"recall_after_learning": learned, "n_test": r.get("n_test")}
            if learned < floor:
                excluded.add(fam)
    return excluded, evidence


AUTH_INVISIBLE, _LEARNABILITY_EVIDENCE = unlearnable_families()

# Fraud share inside loop experiment frames. Higher than the portfolio base rate
# so that per-family recall — including the guard families — rests on enough
# rows to be worth quoting. Never used for precision or false-positive claims.
LOOP_FRAME_FRAUD_RATE = 0.03

# Fallback focus set if automatic selection cannot run (kept stable for the demo).
DEFAULT_FOCUS = ["account_takeover", "otp_relay", "adversarial_mimicry"]
# Preferred guards: families with a clear, stable signature, watched to prove
# adaptation does not forget them. Resolved at run time against whatever the red
# team actually targeted — a family cannot be both attacked and a control.
GUARD_CANDIDATES = ["card_testing", "velocity_smurfing", "merchant_laundering",
                    "bust_out", "geo_anomaly", "scam_transfer"]


def resolve_guards(focus: list[str], k: int = 3) -> list[str]:
    """Guard families are, by definition, the ones NOT under attack."""
    return [f for f in GUARD_CANDIDATES if f not in focus][:k]


# helpers
def _focus_mix(focus: list[str], floor: float = 0.07) -> dict[str, float]:
    """Up-weight focus families so loop frames contain enough of them for
    statistically stable per-family recall.

    A floor keeps every other family present. Boosting the targets without one
    squeezes the untouched families down to a handful of rows, and the
    catastrophic-forgetting check — whose entire job is to watch those families —
    silently reports nothing.
    """
    boost = config.LOOP_CONFIG["focus_mix_boost"]
    mix = dict(DEFAULT_MIX)
    for f in focus:
        mix[f] = mix.get(f, 0.05) * boost
    total = sum(mix.values())
    mix = {k: max(v / total, floor) for k, v in mix.items()}
    total = sum(mix.values())
    return {k: v / total for k, v in mix.items()}


def _prepare_frame(n, seed, specs, mix=None, fraud_rate=None):
    """Simulate a frame and return time-split feature/label/attack arrays.

    Loop frames carry a higher fraud rate than the headline dataset on purpose.
    They are experiment frames: every family — including the guard families whose
    only job is to prove nothing was forgotten — needs enough rows for its recall
    to mean anything. Overall precision and false-positive figures are reported
    from the realistic-base-rate dataset, never from here.
    """
    df, _ = simulate(n_transactions=n, specs=specs, seed=seed, mix=mix,
                     fraud_rate=fraud_rate or LOOP_FRAME_FRAUD_RATE)
    return split_xy(df)


def _eval_frame(n, seed, specs, mix=None, fraud_rate=0.03):
    """A large, fully held-out evaluation frame (never used for training) for
    stable stale-vs-adapted recall on the current attack generation."""
    df, _ = simulate(n_transactions=n, specs=specs, seed=seed, mix=mix,
                     fraud_rate=fraud_rate)
    from src.defend.features import build_xy
    df = df.sort_values("timestamp").reset_index(drop=True)
    return build_xy(df)


def _family_recall(pred_mask_fraud, atk, family):
    """Recall on rows of one family: pred is a boolean 'flagged' over fraud rows."""
    m = atk == family
    n = int(m.sum())
    if n == 0:
        return None, 0
    return float(pred_mask_fraud[m].mean()), n


def _bounded_append(buf_X, buf_y, buf_atk, buf_round, X, y, atk, rnd, cap):
    """Append rows to the replay buffer, then stratified-downsample to `cap`
    while keeping representation of every round already present."""
    buf_X.append(X); buf_y.append(y); buf_atk.append(atk)
    buf_round.append(np.full(len(y), rnd))
    total = sum(len(a) for a in buf_y)
    if total <= cap:
        return
    # Stratified cap: keep an equal per-generation share so earlier attack
    # generations stay represented, while the TOTAL stays within `cap`.
    allX = pd.concat(buf_X); ally = np.concatenate(buf_y)
    allatk = np.concatenate(buf_atk); allrnd = np.concatenate(buf_round)
    rounds_with_data = sorted(set(allrnd.tolist()))
    per = max(1, cap // len(rounds_with_data))
    keep_X, keep_y, keep_atk, keep_round = [], [], [], []
    rng = np.random.default_rng(config.GLOBAL_SEED + rnd)
    for rr in rounds_with_data:
        idx = np.where(allrnd == rr)[0]
        if len(idx) > per:
            idx = rng.choice(idx, per, replace=False)
        keep_X.append(allX.iloc[idx]); keep_y.append(ally[idx])
        keep_atk.append(allatk[idx]); keep_round.append(allrnd[idx])
    buf_X[:] = [pd.concat(keep_X)]; buf_y[:] = [np.concatenate(keep_y)]
    buf_atk[:] = [np.concatenate(keep_atk)]; buf_round[:] = [np.concatenate(keep_round)]


def _buffer_composition(buf_atk, buf_round, buf_y, focus) -> dict:
    if not buf_y:
        return {}
    atk = np.concatenate(buf_atk); rnd = np.concatenate(buf_round)
    y = np.concatenate(buf_y)
    by_family = {a: int(((atk == a) & (y == 1)).sum()) for a in sorted(set(atk)) if a != "legit"}
    by_round = {int(r): int((rnd == r).sum()) for r in sorted(set(rnd.tolist()))}
    evolved = sum(v for k, v in by_family.items() if k in focus)
    return {"total_rows": int(len(y)), "fraud_rows": int(y.sum()),
            "evolved_attack_rows": int(evolved),
            "rehearsal_rows": int(y.sum()) - int(evolved),
            "legit_context_rows": int((y == 0).sum()),
            "by_family": by_family, "by_generation": by_round}


def _status(stale_recall, adapted_recall, cfg) -> str:
    if adapted_recall is None:
        return "n/a"
    gain = adapted_recall - (stale_recall if stale_recall is not None else 0.0)
    if adapted_recall >= cfg["residual_recall_ceiling"] and gain >= cfg["recovery_delta"]:
        return "adapted"
    if adapted_recall < cfg["residual_recall_ceiling"]:
        return "residual_frontier"
    return "partial"


def select_focus(model, X, y, atk, k: int = 3, min_n: int = 12) -> tuple[list[str], list[dict]]:
    """Pick the red team's targets from the defense's MEASURED weakest families.

    This is what makes the loop an experiment rather than a script: nothing about
    which family gets attacked is decided in advance.
    """
    ranked = weakest_families(model, X, y, atk, min_n=min_n)
    eligible = [r for r in ranked if r["family"] not in AUTH_INVISIBLE
                and r["family"] in BASE_SPECS]
    focus = [r["family"] for r in eligible[:k]] or DEFAULT_FOCUS
    return focus, ranked


# main loop
def run_loop(rounds: int = 3, n_base: int = 45_000, n_round: int = 20_000,
             n_eval: int = 24_000, focus: list[str] | None = None,
             seed: int = config.GLOBAL_SEED, save: bool = True,
             auto_focus: bool = True) -> dict:
    cfg = config.LOOP_CONFIG

    # ---- Round 0: the defense currently in force, trained on the known-attack
    # distribution (every family at its generation-0 specification).
    probe = _prepare_frame(n_base, seed, specs={}, mix=_focus_mix(DEFAULT_FOCUS))
    X0_tr, y0_tr, atk0_tr = probe["train"]
    Xp_val, yp_val, _ = probe["val"]
    X0_te, y0_te, atk0_te = probe["test"]

    probe_model = DefenseModel().fit(X0_tr, y0_tr)
    probe_model.fit_calibration(Xp_val, yp_val)
    probe_model.tune_threshold(Xp_val, yp_val, target_fpr=config.TARGET_MAX_FPR)

    # Choose targets from what this defense is measurably worst at.
    if focus is None and auto_focus:
        focus, initial_ranking = select_focus(probe_model, X0_te, y0_te, atk0_te)
    else:
        focus = focus or DEFAULT_FOCUS
        initial_ranking = weakest_families(probe_model, X0_te, y0_te, atk0_te)

    guards = resolve_guards(focus)
    fmix = _focus_mix(focus)
    base = _prepare_frame(n_base, seed, specs={}, mix=fmix)
    X0_tr, y0_tr, _ = base["train"]
    X0_te, y0_te, atk0_te = base["test"]

    # Calibration and threshold tuning happen on a dedicated held-out frame built
    # the SAME way the evaluation frames are built. Tuning on the base frame's
    # validation slice instead would set the threshold against a population whose
    # cards all have mid-window history, and then measure false positives on full
    # frames that include first-ever transactions — so the realized rate would
    # drift above its budget round after round for reasons that have nothing to do
    # with the attack evolving.
    X0_val, y0_val, _ = _eval_frame(max(9000, n_eval // 2), seed + 9000, specs={},
                                    mix=fmix)

    base_model = DefenseModel().fit(X0_tr, y0_tr)
    base_model.fit_calibration(X0_val, y0_val)
    base_model.tune_threshold(X0_val, y0_val, target_fpr=config.TARGET_MAX_FPR)
    base_metrics = evaluate(base_model, X0_te, y0_te, atk0_te)
    base_fpr = base_metrics["false_positive_rate"]

    # Replay buffer holds feature rows of adversarial examples across generations.
    buf_X: list[pd.DataFrame] = []
    buf_y: list[np.ndarray] = []
    buf_atk: list[np.ndarray] = []
    buf_round: list[np.ndarray] = []

    specs = {f: BASE_SPECS[f] for f in focus}
    lineage: dict[str, list[dict]] = {f: [] for f in focus}
    frontier_strikes: dict[str, int] = {f: 0 for f in focus}
    retired: list[dict] = []
    attacked_ever = list(focus)
    prev_model = base_model
    champion = base_model
    variant_tests: dict[int, tuple] = {}   # round -> (X_te, y_te, atk_te) of that gen
    history: list[dict] = []
    registry: list[dict] = [registry_entry(
        "v0-base", "champion",
        {"recall": base_metrics["recall"], "precision": base_metrics["precision"],
         "pr_auc": base_metrics["pr_auc"], "false_positive_rate": base_fpr},
        replay_composition={}, attack_generations=[
            {"family": f, "generation": 0} for f in focus],
        decision={"decision": PROMOTE, "summary": "initial model in force"})]

    for r in range(1, rounds + 1):
        # 1) MEASURE the current defense's dependence on each focus family, on the
        #    held-out frame of the CURRENT attack generation.
        weakness_frames = {}
        round_specs = {}
        round_validation = {}
        reports = {}
        for fam in focus:
            Xw, yw, atkw = _eval_frame(max(9000, n_eval // 2), seed + 5000 + r,
                                       specs=specs, mix=_focus_mix([fam]))
            rep = analyze_family(prev_model, Xw, yw, atkw, fam, seed=seed + r)
            reports[fam] = rep
            weakness_frames[fam] = rep.to_dict()

            # 2) PROPOSE + 3) CONSTRAIN the next generation.
            spec, validation = propose_attack_spec(fam, rep, previous=specs[fam])
            round_specs[fam] = spec
            round_validation[fam] = validation

        specs = {**specs, **round_specs}

        # 4) SIMULATE the evolved generation: a TRAIN frame (source of adversarial
        #    examples) and a large, fully held-out EVAL frame.
        train_parts = _prepare_frame(n_round, seed + r, specs=specs, mix=fmix)
        Xr_tr, yr_tr, atkr_tr = train_parts["train"]
        Xev, yev, atkev = _eval_frame(n_eval, seed + 7000 + r, specs=specs, mix=fmix)
        variant_tests[r] = (Xev, yev, atkev)
        ev_fraud_atk = atkev[yev == 1]

        # 5) STALE model (the defense in force) on the brand-new variant.
        stale_fraud_flag = (prev_model.fused_scores(Xev) >= prev_model.threshold)[yev == 1]
        stale_metrics = evaluate(prev_model, Xev, yev, atkev)

        # 6) REPLAY: this generation's focus adversarial examples, a REHEARSAL sample
        # of every family that was NOT attacked, and legitimate context.
        #
        # The rehearsal set is the fix for a failure this loop actually produced:
        # a buffer containing only the newest, hardest attacks pulls the decision
        # boundary toward them and quietly degrades families the red team never
        # touched — which the forgetting gate then (correctly) refuses to promote.
        # Rehearsing prior tasks alongside the new one is the standard remedy in
        # continual learning, and here it is also the honest one: a defense is not
        # allowed to trade away what it already knew.
        foc_mask = np.isin(atkr_tr, focus) & (yr_tr == 1)
        rehearsal_mask = (~np.isin(atkr_tr, focus)) & (yr_tr == 1)
        n_foc = int(foc_mask.sum())
        legit_idx = np.where(yr_tr == 0)[0]
        rng = np.random.default_rng(seed + r)
        legit_take = rng.choice(legit_idx, min(len(legit_idx), max(n_foc, 200)), replace=False)
        add_idx = np.concatenate([np.where(foc_mask)[0], np.where(rehearsal_mask)[0],
                                  legit_take])
        _bounded_append(buf_X, buf_y, buf_atk, buf_round,
                        Xr_tr.iloc[add_idx], yr_tr[add_idx], atkr_tr[add_idx],
                        r, cfg["replay_buffer_max"])

        # 7) RETRAIN the candidate on base training data + the whole buffer.
        # Replayed adversarial examples are emphasised with a sample weight
        # rather than duplicated rows: same effect on the decision boundary,
        # without inflating the training frame or hiding how strong the emphasis
        # is behind a concatenation.
        k = cfg["replay_oversample"]
        X_train = pd.concat([X0_tr] + buf_X, ignore_index=True)
        y_train = np.concatenate([y0_tr] + buf_y)
        w_train = np.concatenate([np.ones(len(y0_tr))] +
                                 [np.full(len(a), float(k)) for a in buf_y])
        candidate = DefenseModel().fit(X_train, y_train, sample_weight=w_train)
        candidate.fit_calibration(X0_val, y0_val)
        candidate.tune_threshold(X0_val, y0_val, target_fpr=config.TARGET_MAX_FPR)

        # 8) RE-EVALUATE the candidate.
        ad_fraud_flag = (candidate.fused_scores(Xev) >= candidate.threshold)[yev == 1]
        adapted_metrics = evaluate(candidate, Xev, yev, atkev)
        adapted_fpr = adapted_metrics["false_positive_rate"]

        families = {}
        for fam in focus:
            sr, sn = _family_recall(stale_fraud_flag, ev_fraud_atk, fam)
            ar, an = _family_recall(ad_fraud_flag, ev_fraud_atk, fam)
            prev_spec = (lineage[fam][-1]["spec"] if lineage[fam] else BASE_SPECS[fam].to_dict())
            families[fam] = {
                "stale_recall": sr, "adapted_recall": ar, "n": an,
                "delta": (None if (sr is None or ar is None) else round(ar - sr, 4)),
                "status": _status(sr, ar, cfg),
                "intensity": round(specs[fam].intensity, 3),
                "spec": specs[fam].to_dict(),
                "spec_source": specs[fam].source,
                "targets_signal": specs[fam].targets_signal,
                "strategy": specs[fam].strategy,
                "weakness": weakness_frames[fam],
                "constraint_layer": round_validation[fam],
                "novelty_distance_from_generation_0": spec_distance(
                    BASE_SPECS[fam], specs[fam]),
            }
            prior = (BASE_SPECS[fam] if not lineage[fam]
                     else _spec_from_dict(lineage[fam][-1]["spec"], fam))
            lineage[fam].append(lineage_entry(
                fam, prior, specs[fam], reports[fam], round_validation[fam],
                stale_recall=sr, adapted_recall=ar))

        # 9) catastrophic-forgetting check on PRIOR generations.
        prior_generation_recall = {}
        prior_for_gate: dict[str, dict] = {}
        for pr, (Xp, yp, atkp) in variant_tests.items():
            if pr >= r:
                continue
            pflag = (candidate.fused_scores(Xp) >= candidate.threshold)[yp == 1]
            cflag = (champion.fused_scores(Xp) >= champion.threshold)[yp == 1]
            patk = atkp[yp == 1]
            prior_generation_recall[pr] = {}
            for fam in focus:
                cr, cn = _family_recall(pflag, patk, fam)
                hr, _ = _family_recall(cflag, patk, fam)
                prior_generation_recall[pr][fam] = None if cr is None else round(cr, 4)
                if cr is not None and hr is not None:
                    prior_for_gate[f"gen{pr}:{fam}"] = {"champion": hr, "candidate": cr,
                                                        "n": cn}

        # Guard families (never attacked) must stay caught. Measured on a large
        # held-out frame of generation-0 attacks rather than on the base test
        # slice: that slice is small, and a forgetting check that quietly reports
        # "no data" is worse than no check at all.
        guard_recall = {}
        Xg, yg, atkg = _eval_frame(n_eval, seed + 8000, specs={}, mix=_focus_mix(guards))
        cand_flag_fraud = (candidate.fused_scores(Xg) >= candidate.threshold)[yg == 1]
        champ_flag_fraud = (champion.fused_scores(Xg) >= champion.threshold)[yg == 1]
        base_atk_fraud = atkg[yg == 1]
        for fam in guards:
            rc, n = _family_recall(cand_flag_fraud, base_atk_fraud, fam)
            hc, _ = _family_recall(champ_flag_fraud, base_atk_fraud, fam)
            guard_recall[fam] = {"recall": None if rc is None else round(rc, 4), "n": n}
            if rc is not None and hc is not None:
                prior_for_gate[f"guard:{fam}"] = {"champion": hc, "candidate": rc, "n": n}

        # 10) GOVERN: champion / challenger gates decide whether this is deployable.
        focus_stale = [families[f]["stale_recall"] for f in focus
                       if families[f]["stale_recall"] is not None]
        focus_adapted = [families[f]["adapted_recall"] for f in focus
                         if families[f]["adapted_recall"] is not None]
        decision = evaluate_candidate(
            champion_metrics={"attack_recall": float(np.mean(focus_stale)) if focus_stale else None,
                              "false_positive_rate": stale_metrics["false_positive_rate"],
                              "pr_auc": stale_metrics["pr_auc"]},
            candidate_metrics={"attack_recall": float(np.mean(focus_adapted)) if focus_adapted else None,
                               "false_positive_rate": adapted_fpr,
                               "pr_auc": adapted_metrics["pr_auc"]},
            prior_family_recall=prior_for_gate)

        composition = _buffer_composition(buf_atk, buf_round, buf_y, focus)
        registry.append(registry_entry(
            f"v{r}-challenger", "challenger" if decision["decision"] != PROMOTE else "champion",
            {"recall": adapted_metrics["recall"], "precision": adapted_metrics["precision"],
             "pr_auc": adapted_metrics["pr_auc"], "false_positive_rate": adapted_fpr,
             "attack_recall": float(np.mean(focus_adapted)) if focus_adapted else None},
            replay_composition=composition,
            attack_generations=[{"family": f, "generation": specs[f].generation,
                                 "source": specs[f].source} for f in focus],
            decision=decision))

        history.append({
            "round": r,
            "focus": list(focus),
            "intensities": {f: round(specs[f].intensity, 3) for f in focus},
            "specs": {f: specs[f].to_dict() for f in focus},
            "constraint_layer": round_validation,
            "families": families,
            "stale_overall": {
                "recall": stale_metrics["recall"], "precision": stale_metrics["precision"],
                "pr_auc": stale_metrics["pr_auc"],
                "false_positive_rate": stale_metrics["false_positive_rate"],
            },
            "adapted_overall": {
                "recall": adapted_metrics["recall"], "precision": adapted_metrics["precision"],
                "f1": adapted_metrics["f1"], "roc_auc": adapted_metrics["roc_auc"],
                "pr_auc": adapted_metrics["pr_auc"], "false_positive_rate": adapted_fpr,
            },
            "legit_impact": {
                "base_fpr": round(base_fpr, 4), "adapted_fpr": round(adapted_fpr, 4),
                "fpr_regression": round(adapted_fpr - base_fpr, 4),
                "tradeoff": bool(adapted_fpr - base_fpr > cfg["max_fpr_regression"]),
            },
            "prior_generation_recall": prior_generation_recall,
            "guard_families_on_base": guard_recall,
            "replay_buffer_size": int(sum(len(a) for a in buf_y)),
            "replay_composition": composition,
            "champion_challenger": decision,
        })

        prev_model = candidate
        if decision["decision"] == PROMOTE:
            champion = candidate

        # 11) REALLOCATE. A family that stays a residual frontier under repeated
        # attack is not going to yield to more of the same: it sits on the
        # legitimate behavioural centroid, or the genuine customer authorized it.
        # Continuing to hammer it burns replay capacity on examples the model
        # cannot separate — and, as this loop demonstrated, drags down families it
        # could. A real red team moves on and reports the frontier; so does this
        # one. The retirement is recorded, never silently dropped.
        if r < rounds:
            for fam in list(focus):
                fd = families[fam]
                stuck = (fd["status"] == "residual_frontier"
                         and (fd["delta"] or 0.0) < cfg["recovery_delta"])
                frontier_strikes[fam] = frontier_strikes[fam] + 1 if stuck else 0
            exhausted = [f for f in focus if frontier_strikes[f] >= 2]
            if exhausted:
                drop = exhausted[0]
                ranked_now = weakest_families(candidate, Xev, yev, atkev, min_n=25)
                replacement = next(
                    (row["family"] for row in ranked_now
                     if row["family"] not in attacked_ever
                     and row["family"] not in AUTH_INVISIBLE
                     and row["family"] in BASE_SPECS), None)
                retired.append({
                    "round": r, "family": drop,
                    "final_recall": families[drop]["adapted_recall"],
                    "replaced_by": replacement,
                    "reason": ("stayed a residual frontier across two consecutive rounds of "
                               "attack; further replay on it costs other families more than "
                               "it gains"),
                })
                focus = [f for f in focus if f != drop]
                if replacement:
                    focus.append(replacement)
                    attacked_ever.append(replacement)
                    specs[replacement] = BASE_SPECS[replacement]
                    lineage.setdefault(replacement, [])
                    frontier_strikes[replacement] = 0
                guards = resolve_guards(focus)
                fmix = _focus_mix(focus)

    result = {
        "schema_version": config.SCHEMA_VERSION,
        "rounds": rounds, "focus": focus, "families_attacked": attacked_ever,
        "retired_frontiers": retired, "guard_families": guards,
        "focus_selection": {
            "mode": "measured-weakest" if auto_focus else "fixed",
            "initial_ranking": initial_ranking,
            "excluded_as_structural_frontier": sorted(AUTH_INVISIBLE),
            "exclusion_evidence": _LEARNABILITY_EVIDENCE,
            "learnability_floor": LEARNABILITY_FLOOR,
            "why": ("Targets are the families the defense in force handles worst — measured, not "
                    "decided in advance. Families are excluded from targeting only on evidence: "
                    "the leave-one-attack-family-out experiment trains a model with each family "
                    "present, the most favourable case there is, and a family that still cannot "
                    f"clear {LEARNABILITY_FLOOR:.0%} recall under those conditions is limited by "
                    "authorization-time observability rather than by the decision boundary. "
                    "Adversarial replay is a slower route to the same thing — more examples of "
                    "the family — so it cannot beat that ceiling either. Those families are "
                    "still simulated, scored and reported as structural frontiers."),
        },
        "config": {k: cfg[k] for k in cfg},
        "base_metrics": {
            "recall": base_metrics["recall"], "precision": base_metrics["precision"],
            "f1": base_metrics["f1"], "roc_auc": base_metrics["roc_auc"],
            "pr_auc": base_metrics["pr_auc"], "false_positive_rate": base_fpr,
            "per_attack_recall": base_metrics["per_attack_recall"],
        },
        "history": history,
        "promotion": {
            "promoted_rounds": [h["round"] for h in history
                                if h["champion_challenger"]["decision"] == PROMOTE],
            "final_candidate_promoted": bool(
                history and history[-1]["champion_challenger"]["decision"] == PROMOTE),
            "note": ("loop_adapted_model.joblib is the final candidate the loop produced; "
                     "loop_champion_model.joblib is the last candidate that cleared every "
                     "promotion gate. A candidate that improves attack recall but breaches "
                     "the false-positive ceiling is reported, not deployed."),
        },
    }
    if save:
        config.LOOP_HISTORY_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
        config.ATTACK_LINEAGE_PATH.write_text(json.dumps({
            "schema_version": config.SCHEMA_VERSION,
            "note": ("Each node records the specification that produced a generation, the "
                     "measured weakness that motivated it, what the payment-domain constraint "
                     "layer changed, and what it cost the defense."),
            "lineage": lineage}, indent=2), encoding="utf-8")
        write_registry(registry)
        base_model.save(config.LOOP_BASE_MODEL_PATH)      # stale model for the demo
        # Two artifacts, deliberately: the last CANDIDATE the loop produced (what
        # adaptation achieved) and the last PROMOTED model (what governance would
        # actually let into the authorization path). When no candidate clears the
        # gates these are different models, and the difference is the point.
        prev_model.save(config.LOOP_ADAPTED_MODEL_PATH)
        champion.save(config.LOOP_CHAMPION_MODEL_PATH)
        _save_hero_example(base_model, prev_model, specs, seed)
    return result


def _spec_from_dict(d: dict, family: str):
    from src.generate.attack_spec import AttackSpec
    keep = {k: v for k, v in d.items() if k in AttackSpec.__dataclass_fields__}
    keep["attack_family"] = family
    return AttackSpec(**keep)


def _save_hero_example(stale_model, adapted_model, specs, seed,
                       hero_fam: str = "account_takeover") -> None:
    """Persist ONE concrete evolved-attack transaction the STALE model misses but
    the ADAPTED model catches, for the hero demo. Also stores hero-family recall
    for both models on a stable held-out set."""
    from src.defend.features import build_xy
    if hero_fam not in specs:
        hero_fam = next(iter(specs), "account_takeover")
    df, _ = simulate(n_transactions=9000, fraud_rate=0.03, seed=seed + 99,
                     specs=specs, mix=_focus_mix([hero_fam]))
    df = df.sort_values("timestamp").reset_index(drop=True)
    X, y, atk = build_xy(df)
    bs = stale_model.fused_scores(X)
    a = adapted_model.fused_scores(X)
    bflag = bs >= stale_model.threshold
    aflag = a >= adapted_model.threshold

    fam_fraud = (atk == hero_fam) & (y == 1)
    stale_recall = float(bflag[fam_fraud].mean()) if fam_fraud.any() else None
    adapted_recall = float(aflag[fam_fraud].mean()) if fam_fraud.any() else None

    # Pick a transaction where the tiered policy visibly disagrees: the stale model
    # APPROVES it but the adapted model routes it to friction -- a concrete
    # "escaped -> caught" contrast rather than a marginal score difference.
    cand = np.where(fam_fraud & (~bflag) & aflag)[0]
    if len(cand) == 0:
        cand = np.where(fam_fraud)[0]
    pos = int(cand[np.argmax(a[cand] - bs[cand])]) if len(cand) else 0
    raw = df.loc[X.index[pos]]
    txn = {}
    for k, v in raw.to_dict().items():
        txn[k] = (str(v) if isinstance(v, (pd.Timestamp,)) else
                  (bool(v) if isinstance(v, (bool, np.bool_)) else
                   (float(v) if isinstance(v, (np.floating, float)) else
                    (int(v) if isinstance(v, (np.integer, int)) else str(v)))))
    spec = specs.get(hero_fam)
    example = {
        "family": hero_fam,
        "narrative": (f"A {hero_fam.replace('_', ' ')} that EVOLVED specifically to remove the "
                      "signal the defense was leaning on. The stale model waves it through; the "
                      "adapted model — retrained on such variants via cumulative replay — flags it."),
        "evolved_spec": spec.to_dict() if spec is not None else None,
        "targets_signal": spec.targets_signal if spec is not None else None,
        "transaction": txn,
        "stale": {"score": float(bs[pos]), "flagged": bool(bflag[pos]),
                  "threshold": float(stale_model.threshold),
                  "probability": float(stale_model.risk_probability(X.iloc[[pos]])[0]),
                  "action": str(decide(stale_model.risk_probability(X.iloc[[pos]]),
                                       step_up=stale_model.threshold_probability)[0])},
        "adapted": {"score": float(a[pos]), "flagged": bool(aflag[pos]),
                    "threshold": float(adapted_model.threshold),
                    "probability": float(adapted_model.risk_probability(X.iloc[[pos]])[0]),
                    "action": str(decide(adapted_model.risk_probability(X.iloc[[pos]]),
                                         step_up=adapted_model.threshold_probability)[0])},
        "hero_family_recall": {"stale": stale_recall, "adapted": adapted_recall,
                               "n": int(fam_fraud.sum())},
    }
    config.HERO_EXAMPLE_PATH.write_text(json.dumps(example, indent=2), encoding="utf-8")


def _fmt(v):
    return " n/a " if v is None else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser(description="Cumulative-replay adaptive loop.")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--n-base", type=int, default=45_000)
    ap.add_argument("--n-round", type=int, default=20_000)
    ap.add_argument("--n-eval", type=int, default=24_000)
    ap.add_argument("--focus", nargs="*", default=None,
                    help="override the automatically selected weakest families")
    args = ap.parse_args()

    res = run_loop(rounds=args.rounds, n_base=args.n_base, n_round=args.n_round,
                   n_eval=args.n_eval, focus=args.focus)
    print(f"Focus (selected as weakest): {res['focus']}")
    print(f"Guard (not attacked): {res['guard_families']}")
    bm = res["base_metrics"]
    print(f"\nBase model: recall={bm['recall']:.3f} PR-AUC={bm['pr_auc']:.3f} "
          f"FPR={bm['false_positive_rate']:.3f}")
    for h in res["history"]:
        ao = h["adapted_overall"]
        print(f"\n== Round {h['round']} | adapted recall={ao['recall']:.3f} "
              f"PR-AUC={ao['pr_auc']:.3f} FPR={ao['false_positive_rate']:.3f} "
              f"| replay={h['replay_buffer_size']}")
        for fam, d in h["families"].items():
            print(f"   {fam:20s} stale->{_fmt(d['stale_recall'])} "
                  f"adapted->{_fmt(d['adapted_recall'])}  [{d['status']}] "
                  f"targets={d['targets_signal'][:40]!r} src={d['spec_source']}")
        cc = h["champion_challenger"]
        print(f"   champion/challenger: {cc['decision']} — {cc['summary']}")
        gr = ", ".join(f"{k}={_fmt(v['recall'])}" for k, v in h["guard_families_on_base"].items())
        print(f"   guard families (base): {gr}")
    print(f"\nHistory saved to {config.LOOP_HISTORY_PATH}")


if __name__ == "__main__":
    main()
