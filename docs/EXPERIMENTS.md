# Experimental Methodology

*This document is for a judge, a reviewing engineer, or a new contributor who wants to know exactly what was measured, how, and what the measurement is not allowed to claim. After reading it you will be able to re-run every experiment in this repository from the command line, find the artifact each one writes, read that artifact correctly, and recognise the specific ways each result could be misread. Deliberately, this document contains no result values: numbers live in [README.md](../README.md) and in the generated files under `models/`, so this text can never drift out of agreement with them.*

---

## How to read this document

The laboratory runs nine distinct experiments. They are not variations on one benchmark; each answers a different question, and several exist specifically to constrain how the others may be interpreted. The synthetic fidelity diagnostics (Experiment 2) gate the credibility of everything else. The leave-one-attack-family-out experiment (Experiment 5) is the evidence base that decides which families the closed loop is permitted to target. The head-to-head on the final evolved generation (Experiment 7) is the measurement the project actually stands on.

Every experiment writes a JSON artifact under `models/`. The paths are declared once, in `config.py`, and every producer and consumer imports them from there — there are no string literals for artifact paths scattered across the codebase. The whole set is regenerated deterministically by `python -m src.pipeline`, which runs the experiments in the order they depend on each other.

A note that applies everywhere: all data is synthetic, produced by the simulator in `src/generate/`. Nothing in this repository has been validated against real payment data, and no comparison against a real portfolio has been performed. Read every result as a statement about this simulator and this feature set.

### Shared conventions

| Convention | Where it is defined | What it means |
|---|---|---|
| Global seed | `config.GLOBAL_SEED` | Every frame, every model fit, every sampling decision derives from it. Offsets are added to reach an unseen portfolio. |
| Chronological split | `split_points()` in `src/defend/train.py` | Fractions `val_frac=0.15`, `test_frac=0.25`, so the earliest 60 percent trains, the next 15 percent validates, the last 25 percent tests. |
| Feature contract | `FEATURE_COLUMNS` in `src/defend/features.py` | 36 authorization-time features. Post-outcome and simulator-oracle columns are hard-blocked by `assert_auth_time_safe()`. |
| Attack families | `INJECTORS` in `src/generate/attack_injectors.py` | 11 injectors: card testing, bust-out, account takeover, adversarial mimicry, velocity smurfing, merchant laundering, friendly fraud, one-time-password relay, geographic anomaly, scam transfer, wallet provisioning. |
| False-positive budget | `config.TARGET_MAX_FPR` | The single declared budget every threshold-tuning routine targets. |
| Sample-size floor | `config.FAMILY_EVAL["min_n_to_report"]` | Below this count of held-out examples, a per-family proportion is marked insufficient and must not be headlined. |
| Detection score | `DefenseModel.fused_scores()` | The uncalibrated blend of the supervised and anomaly heads. All thresholding happens on this score. |
| Displayed probability | `DefenseModel.risk_probability()` | The isotonically calibrated score. Used by the decision policy and by anything a human reads. |

---

## 1. Held-out detection metrics

### What question it answers

Does the defence rank fraudulent authorizations above genuine ones on traffic it has never seen, at a false-positive budget a payments team would actually agree to? This is the baseline claim on which every later comparison rests.

### How it works

`src/defend/train.py` loads the committed dataset, sorts it by timestamp, and calls `split_xy()`. That function does one thing that is easy to get wrong and matters a great deal: it builds the whole feature matrix once over the complete time-ordered frame, and only then slices it into train, validation, and test. Building features per slice would restart every card's history, every velocity window, and every network counter at the slice boundary. No production system does that, and doing it here would quietly weaken precisely the historical features the defence depends on. Because every historical and relational feature is computed from strictly earlier rows only — expanding windows with a shift, trailing rolling windows, running counters — building globally and splitting afterwards introduces no leakage. A test row still only ever sees its own past.

The split is chronological rather than random for the obvious reason and one less obvious one. The obvious reason is deployment realism: a fraud model is always applied to future traffic, and a random split lets the model train on transactions that occurred after the ones it is scored on. The less obvious reason is that a random split would break the card-level and merchant-level history features by scattering a card's transactions across all three slices, so the model would be evaluated on rows whose own history was in training.

`DefenseModel.fit()` trains both heads: a `HistGradientBoostingClassifier` on the labelled training slice, and an `IsolationForest` fitted on genuine rows only, so it functions as a novelty detector rather than a second supervised model. Scores are blended by `fusion_weight`. Calibration and threshold tuning then happen on the validation slice, never on training rows and never on test rows. `tune_threshold()` picks the lowest score whose *realized* false-positive rate on genuine validation traffic stays inside `config.TARGET_MAX_FPR`; it searches realized rates at each distinct score rather than taking a quantile, because a quantile can land inside a block of tied scores and the `>=` comparison then admits the whole block, silently spending several times the agreed budget.

`evaluate()` in `src/defend/evaluate.py` then scores the test slice and records precision, recall, F1, ROC-AUC, PR-AUC, the false-positive rate, the confusion matrix, and per-attack-family recall with Wilson intervals.

### How to run it

```bash
python -m src.generate.simulate     # only if data/transactions.parquet is missing
python -m src.defend.train
```

Result artifact: `models/metrics.json`.

### How to read the result

Lead with PR-AUC, not ROC-AUC. At the simulated base rate — `config.DEFAULT_FRAUD_RATE`, roughly one percent — the negative class dominates so heavily that ROC-AUC is computed against an enormous pool of easy genuine transactions, and a detector can look excellent on it while its flagged queue is mostly noise. PR-AUC is computed from precision and recall alone, both of which depend on the positive class, so it degrades honestly when the flagged queue fills with genuine traffic. ROC-AUC is still reported, because it is what most readers expect and because a large gap between the two is itself informative, but it does not lead.

Read precision in the context of the base rate rather than against an intuition borrowed from balanced problems. Modest precision at a one-percent base rate is arithmetic, not a defect, and the design answer in this system is tiered decisioning — challenge rather than decline — which Experiment 8 quantifies.

A bad result looks like any of the following: a realized test false-positive rate materially above the budget, which means the validation slice was unrepresentative of the test period; a PR-AUC close to the base rate, which means the ranking carries almost no information; or a large recall with a collapsed precision, which means the threshold search failed and the model is flagging indiscriminately.

### What it does NOT show

It does not show performance on real payments. It does not show performance on attacks that have changed since training — that is the entire point of Experiments 5, 6, and 7. Per-family recall inside this artifact is measured on a realistic-base-rate slice, so the rarer families land at sample sizes too small to quote; that is why they carry a `sufficient_n` flag and why Experiment 3 exists. And it says nothing about whether the underlying synthetic data is separable for the wrong reasons, which is the next experiment.

---

## 2. Synthetic fidelity diagnostics

### What question it answers

Is this simulator a legitimate training ground, or does some field cleanly separate fraudulent from genuine behaviour? This experiment gates the credibility of every other one in the document. If a single feature separates the classes, then the detector has learned the generator rather than the fraud, and every downstream metric — held-out recall, the rules comparison, the closed loop, the head-to-head — becomes a measurement of an artefact. Running it first, and publishing its output, is what makes the rest of the numbers arguable rather than decorative.

### How it works

`src/generate/fidelity.py` builds the feature matrix over the committed dataset and then does three things.

First, `separability()` computes, for every one of the 36 feature columns, the univariate ranking power of that single feature against the fraud label, reported as the larger of the area under the curve and its complement so that direction does not matter. Alongside it, `_overlap_coefficient()` computes a histogram intersection between the genuine and fraudulent marginals of that feature on a shared grid: one means indistinguishable marginals, zero means completely disjoint. Ranking power and overlap answer different questions — a feature can rank well while still overlapping substantially — and both are recorded.

Second, `distributions()` and `portfolio()` emit side-by-side marginals a reviewer can eyeball (log amount, hour of day, day of week, distance from home) and structural statistics of the synthetic portfolio itself (transactions per card and per merchant, repeat-merchant share, distinct merchants and devices per card, weekend share, card-not-present share, foreign share, and the count of rows by actor role).

Third, `quality_checks()` runs a fixed battery of automated guards and emits each as an explicit `PASS`, `WARN`, or `FAIL` record with the observed detail attached. The two separability guards compare the strongest single feature against `SEPARABILITY_FAIL` and `SEPARABILITY_WARN`, both declared at the top of the module. The remaining guards test the properties that make overlap real rather than nominal: that genuine transactions sometimes change device and sometimes visit a new merchant, that fraud sometimes runs from a device the card had used before and sometimes revisits a merchant, that a meaningful share of fraudulent amounts sits inside the genuine inter-quartile amount range, that having no card history is not a synonym for fraud, that fraud is not uniformly nocturnal, that every family clears the module's own `MIN_FAMILY_N` floor, and that fraud actors generated cover traffic at all.

That last check deserves a sentence of explanation, because it is the one most easily mistaken for a labelling bug. Mule accounts, bust-out accounts, and front merchants build ordinary-looking history before the fraudulent authorization. Those rows are generated by a fraud actor and are labelled **legitimate**, because at authorization time that is exactly what they are. A simulator in which every transaction touched by a fraud actor is labelled fraud teaches the model to detect actors rather than authorizations, which is not the problem statement.

### How to run it

```bash
python -m src.generate.fidelity
```

Result artifact: `models/fidelity_report.json`, which carries `separability`, `max_single_feature_auc`, `distributions`, `portfolio`, `checks`, `n_fail`, and `n_warn`. The pipeline runs this as step two of eight and prints a warning listing the failing check names if `n_fail` is non-zero.

### How to read the result

Start at `n_fail`. Any failing check invalidates the downstream numbers until it is fixed, and the pipeline says so on the console rather than continuing quietly. Then read the top of the `separability` list: a feature at the top of that list with very high ranking power and very low overlap is a shortcut, and the right response is to change the simulator, not to change the threshold. A warning on `no_feature_near_perfect_separation` is a signal to inspect that feature's marginals in the `distributions` block before trusting any result that leans on it.

A bad result is a `FAIL` on `no_single_feature_separates_classes`; a `FAIL` on `no_history_is_not_a_fraud_synonym`, which would mean the model can win by flagging every thin-file card; or a `FAIL` on `fraud_actors_have_cover_traffic`, which would mean the actor-level realism silently stopped being generated.

### What it does NOT show

This is stated in the artifact itself, in the `scope` field, and it is worth repeating. These are diagnostics of the simulator against itself. They measure overlap between simulated genuine and simulated fraudulent behaviour. They say nothing whatsoever about similarity to any real payment portfolio, and no such comparison has been performed. Passing every check means the data is not trivially separable; it does not mean the data is realistic.

---

## 3. Per-family recall on an unseen fraud-enriched frame

### What question it answers

For each attack family individually, how much of it does the defence catch — measured on enough examples for the number to mean anything, and on a synthetic portfolio the model has never met?

### How it works

`family_recall_frame()` in `src/defend/diagnostics.py` generates a dedicated frame with the parameters in `config.FAMILY_EVAL`: `n_transactions` of 90,000, `fraud_rate` of 0.030, and a seed of `config.GLOBAL_SEED + FAMILY_EVAL["seed_offset"]`, where the offset is 424,242. The frame is sorted by timestamp, features are built over it, and three detectors are scored on it at their own operating points: the rules baseline via `rules_predict()`, the static model, and — when the closed-loop artifact exists — the adapted model. For each family, recall is recorded together with `n` and a `sufficient_n` flag against `min_n_to_report`, which is 30.

Two design choices carry the experiment.

The separate seed is what makes this a generalization test rather than a memorization test. Because the seed differs, the simulator draws a different set of cardholders, merchants, devices, and relationships. The model is not being asked to re-recognise transactions that resemble its training rows; it is being asked whether what it learned transfers to a different synthetic portfolio built by the same rules.

The enriched fraud rate exists purely to make the denominators large enough. At the realistic base rate, the smaller families appear a couple of dozen times in a held-out slice, and a recall quoted from that many examples carries an interval so wide that the point estimate is not worth printing. Raising the fraud share raises the per-family counts without changing what a family looks like, because enrichment changes the mix, not the injectors.

### How to run it

It runs as part of the diagnostics module, which is the only entry point:

```bash
python -m src.defend.diagnostics
```

Result artifact: `models/family_recall.json`.

### How to read the result

Read the families in ascending order of recall — that is the order the module's own console output prints them in — and read the `n` beside each. Anything with `sufficient_n` set to false is present for completeness and must not be quoted as a finding. Compare families against each other rather than against an absolute target: the interesting signal is *which* families the defence is weak on, because that is what motivates the red-team targeting in Experiment 6.

Expect the families where the genuine customer authenticated and authorized the payment to sit at the bottom. That is a property of authorization-time observability, not a tuning failure, and the blind-spots artifact says so explicitly in its `note` field.

A bad result is a family whose recall collapses relative to the same family in `models/metrics.json`, which would suggest the model memorized portfolio-specific structure rather than behaviour, or a frame in which several families fall below the reporting floor despite the enrichment, which would mean the mix weights need attention.

### What it does NOT show

**This frame must never be used for precision or false-positive claims, and the artifact says so in its own `note`.** Its fraud rate is roughly two and a half times the realistic base rate, and precision is a function of the base rate. Precision computed here would be flattering and meaningless. The artifact does record `false_positive_rate_on_this_frame` for internal comparison between detectors on identical traffic, but the reportable false-positive figure is the one from the realistic-base-rate test slice in Experiment 1. The frame also holds generation-zero attacks only: it says nothing about evolved variants.

---

## 4. Rules versus static versus adaptive, at native and matched budgets

### What question it answers

Does machine learning actually add value over rules a competent fraud team would write, and does the adaptive model differ from the static one — measured in a way that cannot be won by simply challenging more customers?

### How it works

`src/defend/baseline.py` defines a transparent rule set in `RULES`: high velocity in an hour, a very large amount to a new payee, a new device on high-risk card-not-present traffic, impossible travel, high-risk spend from a new account, a high-risk country, structuring near a threshold, an amount z-score spike, a repeat disputer, and four network-level rules covering a device shared across cards, a card hopping devices, a merchant flooded with new cards, and a large transfer to a new payee at a cash-like merchant category.

The rule set is deliberately not a strawman. It includes the network-level rules a real team would write once it had the same relational counters the model receives, so the comparison measures what machine learning adds on top of good rules rather than on top of a weak one. The rules are also deliberately untuned: the thresholds are round, intuitive numbers chosen from domain reasoning, never fitted to this dataset. Fitting them would turn the baseline into a second model and destroy the point of having one.

`compare()` evaluates every detector on the held-out chronological test slice by default, using the same `split_xy()` contract the trainer uses, so the static-model column here is numerically the same model state reported in `models/metrics.json`. It produces two views. The first is each detector at its **native** operating point. The second, the `_fpr_matched` block, re-thresholds every detector — including the rules baseline, via its fraction-of-rules-fired score — to spend the same false-positive budget.

Matching matters because comparing detectors at whatever operating point each happens to occupy is close to meaningless. Any detector can buy recall by challenging more genuine customers. A comparison that does not hold customer friction constant is comparing two different products, not two different detectors.

The way the matched thresholds are chosen is the part most likely to be done wrong elsewhere, so it is worth stating plainly. `_threshold_for_fpr()` selects the lowest threshold whose realized false-positive rate stays inside the budget, and it is run **on the validation slice**, then applied unchanged to the test slice. Choosing it on the test set's own labels would let every detector peek at the answer, and the comparison would flatter whichever detector happened to have the most exploitable score distribution on that particular slice. A consequence the artifact notes explicitly: realized test false-positive rates will differ slightly from the budget, precisely because the threshold was not fitted to the test set. That small mismatch is evidence of honesty, not of a bug.

The adaptive column is the **promoted champion** (`loop_champion_model.joblib`). The final loop candidate is reported alongside it under a separate key labelled as unpromoted, so a model the governance gates turned down is never presented as the working defence.

### How to run it

```bash
python -m src.defend.baseline
```

Result artifact: `models/baseline_comparison.json`, with the like-for-like numbers under `_fpr_matched.models` and per-detector native numbers at the top level.

### How to read the result

Read the `_fpr_matched` block. The top-level blocks are context, not the comparison. Within the matched block, compare recall and precision at approximately equal false-positive rates, and check `threshold_selected_on` to confirm it says the validation slice.

Then apply the interpretive caveat the README states and the artifact supports: this table scores every detector on the **original** attack distribution, which is the static model's home ground, because that is exactly what it was trained on. The adaptive model is trained on that data plus several generations of evolved attacks, so on the original distribution it is at best comparable. Its advantage is robustness to attacks that have moved, and that is Experiment 7, not this one.

A bad result would be the rules baseline matching the model at a matched budget, which would mean the learned model adds nothing; or a matched false-positive rate far from the budget, which would mean the validation slice is not representative of the test period.

### What it does NOT show

It does not show which detector would perform better on tomorrow's attacks. It does not show anything about operating costs — that is Experiment 8. It does not license reading the adaptive column as a regression: an adaptive model scoring comparably on the original distribution while holding up on evolved attacks is the intended trade, and reading only this table would invert the conclusion.

---

## 5. Leave-one-attack-family-out

### What question it answers

Is a static supervised model actually weak against a fraud family it never trained on, and is that family learnable at all once the model sees it? This is the empirical justification for building an adaptive loop rather than simply retraining periodically, and it is also the evidence base that decides which families the loop is permitted to attack.

### How it works

`src/experiments/leave_one_out.py` runs, for each of the ten families in `DEFAULT_FAMILIES`, a four-step procedure on a single simulated frame of 90,000 transactions at a fraud rate of 0.035, split chronologically by the same `split_xy()` contract:

1. Train a model on the training slice with that family's fraudulent rows removed. Genuine rows and every other family's fraud stay in, so only that one family is unseen.
2. Calibrate and tune the threshold on the validation slice, then measure recall on that family's fraudulent rows in the **test** slice. This is `recall_unseen`.
3. Train a second model on the full training slice, family included, with identical calibration and threshold tuning.
4. Measure the same family's recall on the same test rows. This is `recall_after_learning`, and the difference is `gain`.

Every record carries `n_test`, a `sufficient_n` flag against `config.FAMILY_EVAL["min_n_to_report"]`, and a Wilson interval on both recall figures.

`select_hero_family()` picks the family to lead with: the largest measured gain **among families that clear the sample-size floor**. When no family clears the floor it falls back to the full set and returns `underpowered=True`, and the caller is required to refuse to headline the number. That function exists in one place on purpose. The rule was previously duplicated across the README generator, the landing page, and the hero-demo page, each with a silent fallback that would have re-admitted an under-powered family the moment the floor stopped being met — and all three would have flipped together, invisibly.

The artifact is also consumed programmatically. `unlearnable_families()` in `src/loop/redteam_loop.py` reads `models/leave_one_out.json` and excludes from red-team targeting any family whose `recall_after_learning` falls below `LEARNABILITY_FLOOR`. The reasoning is that training with the family present is the most favourable condition available; a family that still cannot be caught under it is limited by authorization-time observability, not by the decision boundary, and adversarial replay is a slower route to the same thing — more examples of the family — so replay cannot beat that ceiling either. Those families are still simulated, still scored, and still reported, as structural frontiers. This is why the pipeline runs leave-one-out at step five, before the loop at step six.

### How to run it

```bash
python -m src.experiments.leave_one_out
```

Result artifact: `models/leave_one_out.json`.

### How to read the result

A large gain means the family is genuinely hard when unseen but learnable once seen — exactly the gap adversarial replay is designed to fill, and the strongest argument in the repository for the loop existing. A small gain on an already-easy family means the static model did not need help there. A low ceiling even when the family is in training means a structural frontier, and the honest response is to name it rather than to tune it away.

**The caveat that matters most: `recall_after_learning` is an upper bound obtained by handing the model the answer.** The family is placed directly into the training set. The closed loop does not get that. The loop has to discover the weakness, generate the family itself through an evolving specification, and reach that level through replay. Reading `recall_after_learning` as "what the loop achieves" inverts the relationship — it is the ceiling the loop is aiming at, not a result the loop produced.

A bad result is a family whose gain is large but whose `n_test` is below the floor, which is not a finding; or a `recall_unseen` that is already high across the board, which would mean the families are insufficiently distinct and the experiment is not testing what it claims.

### What it does NOT show

The artifact carries its own `what_this_does_not_show` field, and it is the authoritative statement. In summary: this measures unseen-to-learned on synthetic data. It is not zero-shot detection, it is not production performance, and small-`n` families carry wide uncertainty which is why they are reported with their sample size.

---

## 6. The closed loop, one round in full

### What question it answers

When the defence is stress-tested by an attack that evolves specifically to remove the signal the defence depends on, can the defence recover through cumulative adversarial replay — without forgetting what it already knew, without spending more genuine-customer friction than agreed, and subject to a governance decision that can refuse to ship the result?

### How it works

One round is a full red-team experiment, not a retraining step. `run_loop()` in `src/loop/redteam_loop.py` implements the eleven stages named in the module docstring.

```
                 +-------------------------------------------------+
                 |                                                 |
   round 0       v                                                 |
 +----------+  (1) EVALUATE the defence in force                   |
 | base     |    on a held-out frame                               |
 | model    |          |                                           |
 +----------+          v                                           |
                 (2) LOCATE weakest families; per-family           |
                     permutation test names the relied signal      |
                          |                                        |
                          v                                        |
                 (3) PROPOSE next generation as a structured spec  |
                     (LLM agent, or deterministic heuristic)       |
                          |                                        |
                          v                                        |
                 (4) CONSTRAIN against payment-domain bounds       |
                          |                                        |
                          v                                        |
                 (5) SIMULATE  -> train frame + held-out eval frame|
                          |                                        |
                          v                                        |
                 (6) STRESS the stale defence  -> the opened gap   |
                          |                                        |
                          v                                        |
                 (7) REPLAY escaped rows + rehearsal + legit       |
                     into a bounded, stratified buffer             |
                          |                                        |
                          v                                        |
                 (8) RETRAIN a CANDIDATE on base data + buffer     |
                          |                                        |
                          v                                        |
                 (9) RE-EVALUATE: new generation, every prior      |
                     generation, guard families, genuine traffic   |
                          |                                        |
                          v                                        |
                (10) GOVERN through champion/challenger gates      |
                          |                                        |
                          v                                        |
                (11) CARRY the residual weakness forward;          |
                     retire exhausted frontiers, reallocate  ------+
```

**Round zero** trains a probe model on a base frame and uses `select_focus()` to pick the red team's targets from the families that model handles measurably worst, excluding families that `unlearnable_families()` marked structural. Nothing about which family gets attacked is decided in advance; that is what makes the loop an experiment rather than a script. Guard families are then resolved by `resolve_guards()` as the first three entries of `GUARD_CANDIDATES` that are *not* under attack — a family cannot be both a target and a control.

The frame mix is shaped by `_focus_mix()`, which up-weights the focus families by `LOOP_CONFIG["focus_mix_boost"]` so their per-family recall rests on enough rows, subject to a floor that keeps every other family present. The floor is not cosmetic: boosting the targets without one squeezes the untouched families down to a handful of rows, and the catastrophic-forgetting check — whose entire job is to watch those families — would silently report nothing.

Calibration and threshold tuning for the base model and every candidate happen on a dedicated held-out frame built the same way the evaluation frames are built, not on the base frame's validation slice. Tuning on that slice would set the threshold against a population whose cards all have mid-window history, and then measure false positives on full frames that include first-ever transactions, so the realized rate would drift above its budget round after round for reasons that have nothing to do with the attack evolving.

**Stage 2** calls `analyze_family()` in `src/loop/weakness.py`, which runs a permutation test scoped to one family: shuffle one feature across that family's fraudulent rows plus a legitimate sample and measure how much the model's ability to rank the family above genuine traffic drops. A large drop means the model leans on that feature *for this family*, which makes it the obvious thing for the next generation to remove. It also profiles what escaped versus what was caught.

**Stages 3 and 4** are the generative half. `propose_attack_spec()` asks the language-model agent for a structured `AttackSpec`; if the model is unavailable or returns something unusable, `heuristic_mutation()` produces the same kind of proposal deterministically, so the loop degrades in quality but never in reproducibility. Either proposal then passes through `validate_spec()`, which clamps numeric ranges to `config.ATTACK_SPEC_BOUNDS`, replaces out-of-vocabulary categorical values with the family's baseline, enforces family-level payment-domain requirements, and enforces that stealth is monotone within a lineage — a persistent adversary does not become louder after being caught. Every correction is recorded in a `ValidationReport` and stored in the round's `constraint_layer` field, so "the red team asked for something impossible" is a visible event rather than a silent fix.

**Stage 7** is where the continual-learning discipline lives. The buffer receives three things from each round's training frame: the focus families' adversarial rows, a **rehearsal** sample of every family that was *not* attacked, and a sample of legitimate context rows. The rehearsal set is the fix for a failure this loop actually produced. A buffer containing only the newest, hardest attacks pulls the decision boundary toward them and degrades families the red team never touched — which the forgetting gate then correctly refuses to promote. `_bounded_append()` caps the buffer at `LOOP_CONFIG["replay_buffer_max"]` with a stratified downsample that keeps an equal per-generation share, so earlier generations stay represented as the buffer fills.

**Stage 8** retrains on the base training data plus the whole buffer, emphasising replayed rows with a sample weight of `LOOP_CONFIG["replay_oversample"]` rather than by duplicating rows. The effect on the decision boundary is the same; the difference is that the emphasis stays explicit and the training frame stays small.

**Stage 9** is three checks, not one. The candidate is scored on the new generation's held-out frame; on every **prior generation's** stored evaluation frame, retained in `variant_tests`, keyed as `gen{round}:{family}`; and on a large generation-zero frame of the **guard families**, keyed as `guard:{family}`. Both prior-generation and guard checks compare the candidate against the current champion on identical rows. The guard frame is generated large and separately rather than reusing the base test slice, because that slice is small and a forgetting check that quietly reports "no data" is worse than no check at all.

**Stage 10** hands those measurements to `evaluate_candidate()` in `src/defend/governance.py`, which runs five gates from `config.CHAMPION_CHALLENGER`: a minimum recall gain on the new attack generation; an absolute false-positive ceiling; a limit on false-positive regression against the champion; no catastrophic forgetting on any prior family; and a floor on overall PR-AUC relative to the champion. A candidate failing any gate is recorded with the reason and **not** promoted.

**Stage 11** carries the residual forward. A family that stays a residual frontier — status derived from measured recall by `_status()`, never hardcoded — across two consecutive rounds is retired and replaced by the next-weakest untargeted family, with the retirement recorded in `retired_frontiers`. The reasoning is that a family sitting on the legitimate behavioural centroid will not yield to more of the same, and continuing to attack it burns replay capacity that a family with a real residual signal could have used. In the code this stage appears as the reallocation block at the end of `run_loop()`; the docstring names it CARRY, which is the same idea from the loop's point of view.

### How to run it

```bash
python -m src.loop.redteam_loop --rounds 3
```

Optional flags: `--n-base`, `--n-round`, `--n-eval`, and `--focus` to override the automatically selected weakest families. Overriding `--focus` switches `focus_selection.mode` in the artifact from `measured-weakest` to `fixed`, which is worth noting whenever a result is quoted.

Result artifacts: `models/loop_history.json` (the full round-by-round record), `models/attack_lineage.json` (how each family's specification evolved and what it cost the defence), `models/model_registry.json` (every model version with its gates and decision), `models/hero_example.json`, and three model files — `loop_base_model.joblib` (the stale defence), `loop_adapted_model.joblib` (the final candidate), and `loop_champion_model.joblib` (the last candidate that cleared every gate).

### How to read the result

For each round, read four things together. Read `families[*].stale_recall` against `adapted_recall` and the derived `status`. Read `guard_families_on_base` to confirm the untouched families held. Read `prior_generation_recall` to confirm earlier generations were not traded away. Read `champion_challenger.decision` and, if it is a rejection, `failed_gates`.

The two model artifacts exist as a deliberate pair. `loop_adapted_model.joblib` is what adaptation achieved; `loop_champion_model.joblib` is what governance would actually let into the authorization path. When no candidate clears the gates these are different models, and the difference is the point of having governance at all.

A bad result is not a rejected candidate — that is the system working. A bad result is a promoted candidate whose guard families dropped, which would mean the gates are mis-specified; a round where `constraint_layer` shows the specification being corrected on every field, which would mean the proposal stage is producing nonsense; or a status of `adapted` on a family whose `n` is small enough that the recall gain is inside sampling error.

### What it does NOT show

The loop does not demonstrate that adaptation always wins. It demonstrates a measured, governed process in which adaptation sometimes wins, sometimes trades one family for another, and sometimes fails the gates. It does not simulate a shadow-mode or controlled-rollout period, which a real deployment would require — the registry note says so explicitly. And the loop's own per-round recall figures are measured on fraud-enriched experiment frames, so they are never the source of a precision or false-positive claim.

---

## 7. Head-to-head on the final evolved generation

### What question it answers

Once the attack has moved, which defence catches it? This is the measurement the project stands on.

### How it works

Every other comparison in this repository scores detectors on the original attack distribution, which is the static model's home ground — it was trained on exactly that, so of course it does well there. The question adaptation exists to answer is only visible on the evolved generation, and that is what `head_to_head()` in `src/defend/diagnostics.py` measures.

The procedure reconstructs the final generation from what the loop recorded rather than re-deriving it. It reads `models/loop_history.json`, takes the last round, rebuilds each family's `AttackSpec` from the stored specification dictionary, and reads the final round's `focus` — which may differ from where the loop started, because exhausted frontiers are retired and replaced. It then simulates a fresh frame from those specifications at a seed of `config.GLOBAL_SEED + 606_060`, unseen by either model.

Three models are scored on that frame: the static defence loaded from `loop_base_model.joblib`, which has never seen these evolved variants; the adaptive defence from `loop_adapted_model.joblib`; and, when it exists, the promoted champion from `loop_champion_model.joblib`.

Then the part that makes the comparison legitimate. Two defences sitting at their own operating points are not comparable — whichever happens to sit stricter will catch less, and the difference says nothing about which is the better detector. So each model is **additionally re-thresholded to spend the same false-positive budget**, using `_threshold_for_fpr()` from `src/defend/baseline.py`. The threshold is chosen on a **separate generation-zero frame** at seed `config.GLOBAL_SEED + 707_070`, which neither model has seen and which is *not* the evolved frame the model is about to be scored on. Choosing it on the evolved frame would let each model tune itself against the very traffic under test.

Each model's block records the matched-threshold results at the top level and its native operating point under `native_operating_point`, with per-family recall, sample size, and a Wilson interval for every focus family.

### How to run it

It runs inside the diagnostics module:

```bash
python -m src.defend.diagnostics
```

It returns `None`, and writes nothing, unless `loop_history.json`, `loop_base_model.joblib`, and `loop_adapted_model.joblib` all exist — so run the loop first, or run `python -m src.pipeline`, which orders the steps correctly.

Result artifact: `models/head_to_head.json`.

### How to read the result

Use the **matched** numbers. The artifact's own `note` says so: "The matched numbers are the comparable ones."

Read the per-family block, not just `mean_evolved_recall`. The families the red team actually attacked are the ones that moved, and a single mean across families hides both halves of a trade — a gain on one family paid for by a loss on another. The README prints the per-family numbers next to the mean for exactly this reason, and any summary that quotes only the mean is under-reporting the result.

Read the false-positive rate alongside recall in every row. A model that catches more while spending more friction has not necessarily won.

A bad result to watch for is not a low adaptive recall; it is a `frame.n_fraud` small enough that the per-family Wilson intervals overlap heavily, in which case the comparison does not separate the models at all and should be reported as inconclusive.

### What it does NOT show

It does not show performance against attacks that evolved along a *different* axis than the loop explored, because the evolved generation is the one this loop's own specifications produced. It does not show a general claim that adaptation beats static defence; it shows what happened on this frame, for these families, at a matched budget. And because the frame is fraud-enriched, its false-positive figures are for holding the comparison fair, not for reporting.

---

## 8. Threshold sweep, calibration, and operational volumes

### What question it answers

Three related questions a payments team asks before agreeing to run a model at all. Where else could the operating point sit, and what does each position cost? Does the number on the reviewer's screen mean anything? And how much genuine traffic, review labour, and customer friction does the chosen point actually imply?

### How it works

**Threshold sweep.** `threshold_sweep()` scores the held-out test slice and evaluates recall, precision, false-positive rate, false positives per thousand genuine payments, and flagged share at candidate thresholds sampled on the score **quantiles**, so the curve is dense where decisions actually happen rather than uniformly spaced across a range where nothing changes. The artifact also records the model's operating point in both raw-score and calibrated-probability form, the resolved decision tiers, and the configured budget. There is no magic threshold; this is the curve a fraud team argues over.

**Calibration.** `calibration_report()` builds a reliability curve on quantile-edged bins, for the raw fused score and the isotonically calibrated probability, and reports the Brier score for each. Isotonic calibration is fitted on the held-out validation slice by `DefenseModel.fit_calibration()`. Because isotonic regression is monotone it cannot change the model's ranking, so ROC-AUC and PR-AUC are identical either way and only the meaning of the displayed number changes. Detection still thresholds the **raw** fused score, because a calibrated score is a step function and thresholding a step function cannot hit a false-positive budget precisely.

**Operational volumes.** `operational_metrics()` applies the measured rates to the assumptions declared in `config.OPERATIONAL_SCENARIO` and reports per-thousand rates, the distribution across the three decision tiers from `src/defend/decision_policy.py`, monthly step-up challenges, review hours if every challenge were manually reviewed, and an estimate of genuine customers abandoning after a step-up. The decision tiers apply to the **calibrated** probability so that they read as statements about risk rather than positions on an arbitrary scale, and the step-up boundary defaults to the model's own tuned operating threshold rather than a second invented cutoff — `DECISION_THRESHOLDS["step_up"]` is `None`, and `resolve_thresholds()` fills it in.

**Blind spots.** `blind_spots()` ranks families by measured recall, profiles the escaped traffic for the hardest few, and derives the next red-team target from the top entry. It is published rather than buried: a defence that cannot name its own blind spots is not being evaluated honestly.

### How to run it

```bash
python -m src.defend.diagnostics
```

Result artifacts: `models/threshold_sweep.json`, `models/calibration.json`, `models/operational_metrics.json`, `models/blind_spots.json`, plus `models/defend_demo.parquet` — precomputed scores, actions, and reason codes for the test split, so the web prototype renders from committed artifacts instead of recomputing every feature on each rerun.

### How to read the result

For the sweep, find the operating point on the curve and look at its neighbours: the honest question is not "is this threshold right" but "what would one step in either direction buy and cost". The `false_positives_per_1000_legit` column is the one to quote to a non-specialist, because it is the same statement as the false-positive rate in units a person can picture.

For calibration, read the reliability curve before the Brier score. At a one-percent base rate a low Brier score is easy to obtain by predicting near zero everywhere, so the scalar is nearly uninformative on its own; the curve shows whether predicted and observed rates track each other across the score range. `brier_improvement` is included for completeness but should not be headlined.

For operational metrics, the `scenario` block must always be read with its `label`, which states plainly that these are illustrative synthetic assumptions, and its `disclaimer`, which states that they are not a forecast for any real portfolio.

A bad result is a reliability curve that is flat or non-monotone in the upper bins, meaning the displayed probability is not trustworthy where it matters most; or an operating point sitting on a near-vertical section of the sweep, meaning small threshold changes swing the false-positive rate wildly and the point is not robust.

### What it does NOT show

The cost figures are not Mastercard figures and are not a claim about anyone's economics. They exist so a false-positive rate can be read as review volume and customer friction, which is a translation, not a forecast. The tiered decision policy is a prototype of issuer or network decisioning and is not a description of any production system.

---

## 9. Statistical practice

This section is not a separate experiment. It is the set of rules the other eight follow, gathered in one place because they are what make small-sample proportions reportable at all.

### Sample-size floors

Per-family recall is a proportion measured on however many examples of that family happen to be in the held-out frame. Three different floors exist, each with a different job, and they are deliberately not unified:

| Floor | Value | Defined in | What it governs |
|---|---|---|---|
| `min_n_to_report` | 30 | `config.FAMILY_EVAL` | Whether a per-family recall figure may be quoted at all. Sets `sufficient_n` in metrics, family recall, and leave-one-out artifacts, and gates `select_hero_family()`. |
| `MIN_FAMILY_N` | 15 | `src/generate/fidelity.py` | A simulator-side warning that a family is being generated too rarely to support any claim. |
| `min_n` | 10 (blind spots), 12 (`select_focus`), 25 (loop reallocation) | `src/defend/diagnostics.py`, `src/loop/redteam_loop.py` | Whether a family is eligible to be *ranked* as weak or chosen as a target. Lower than the reporting floor because ranking tolerates more noise than quoting does. |

A figure below the reporting floor is not suppressed; it is published with `sufficient_n` set to false. Suppression hides the fact that a family is under-sampled, which is itself a finding.

### Wilson intervals

`wilson_interval()` in `src/defend/evaluate.py` computes a 95 percent Wilson score interval and is attached to per-attack recall in `models/metrics.json`, to both recall figures in `models/leave_one_out.json`, and to every per-family recall in `models/head_to_head.json`.

The Wilson interval is used rather than the normal approximation because the normal approximation is badly behaved exactly where these measurements live: small `n`, and proportions near zero or one. Quoting a perfect recall from nine examples invites exactly one question, and the interval answers it before it is asked.

The interval is not decoration. `evaluate_candidate()` in `src/defend/governance.py` uses it to decide whether a prior-family regression is real: a drop counts as catastrophic forgetting only when the **upper** bound of the candidate's Wilson interval falls below the tolerated level. The reasoning is that a gate firing on any drop past a fixed line will fire on sampling noise, and a gate that blocks a genuinely better model on a coin flip is worse than no gate at all. When `n` is zero the code falls back to the plain difference, and the artifact's gate detail records which path was taken.

### Seeds and unseen frames

Everything reproducible flows from `config.GLOBAL_SEED`. Evaluation frames are kept unseen by adding large, distinct offsets, so the simulator draws a different portfolio of cardholders, merchants, devices, and relationships rather than a reshuffling of the same one.

| Frame | Seed | Purpose |
|---|---|---|
| Headline dataset | `GLOBAL_SEED` | Training, validation, and the chronological test slice. |
| Family recall frame | `GLOBAL_SEED + 424_242` (`FAMILY_EVAL["seed_offset"]`) | Per-family recall on an unseen portfolio. |
| Head-to-head evolved frame | `GLOBAL_SEED + 606_060` | Scoring both defences on the final evolved generation. |
| Head-to-head threshold frame | `GLOBAL_SEED + 707_070` | Selecting matched thresholds, generation-zero attacks only. |
| Loop weakness frames | `seed + 5000 + round` | Per-family weakness analysis inside a round. |
| Loop evaluation frames | `seed + 7000 + round` | Held-out scoring of stale and adapted models on the new generation. |
| Loop guard frame | `seed + 8000` | Generation-zero guard families for the forgetting check. |
| Loop calibration frame | `seed + 9000` | Calibration and threshold tuning, built the same way evaluation frames are. |

Two disciplines follow from this table and are worth stating as rules. A threshold is never selected on the frame it will be scored on — Experiment 4 selects on the validation slice, Experiment 7 selects on a separate generation-zero frame. And precision and false-positive claims come only from realistic-base-rate frames; every fraud-enriched frame in this repository carries a note in its own artifact saying it must not be used for them.

### Reproducibility

`python -m src.pipeline` regenerates every artifact in dependency order in eight steps, and `python -m src.pipeline --fast` skips the closed loop. Registry entries deliberately carry no wall-clock timestamp by default, because committed artifacts must stay byte-stable across runs; `trained_at` can be passed explicitly if a deployment wants real timestamps. `SCHEMA_VERSION` in `config.py` is stamped into every artifact and is bumped whenever the simulator schema or the feature contract changes, so a stale artifact is detectable rather than silently mixed with fresh ones. The test suite under `tests/` includes `test_artifact_consistency.py`, which checks that the committed artifacts agree with each other.

---

## Where to go next

Read [README.md](../README.md) for the results themselves — every number, with its sample size, regenerated from the artifacts described here. Read [Architecture](ARCHITECTURE.md) for how the simulator, the feature contract, the defence model, and the loop fit together as a system, and [Data model](DATA_MODEL.md) for the transaction schema, the full feature contract, and the artifact reference. Read [Design](DESIGN.md) for why the system is shaped this way and [Decisions](DECISIONS.md) for the evidence behind the choices this document takes as given — the chronological split, the matched-budget comparison, the sample-size floors. For the short-form answers to the questions this methodology most often provokes, see [Judge Q&A](JUDGE_QA.md); for driving the web prototype, whose pages render from these same artifacts, see the [Operations guide](DEMO_GUIDE.md).

To inspect a result rather than read about it, open the artifact directly: `models/metrics.json`, `models/fidelity_report.json`, `models/family_recall.json`, `models/baseline_comparison.json`, `models/leave_one_out.json`, `models/loop_history.json`, `models/head_to_head.json`, `models/threshold_sweep.json`, `models/calibration.json`, `models/operational_metrics.json`, and `models/blind_spots.json`. Each one carries its own scope note, and where a caveat in this document and a caveat in an artifact disagree, the artifact is authoritative.
