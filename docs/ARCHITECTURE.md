# System Architecture

*This document is the engineering reference for the AI Defense Lab. It is written for a reviewing engineer, a judge who wants to verify a claim against the source, or a new contributor about to make their first change. After reading it you will know what each of the four pillars owns, which module implements it, how a full pipeline run turns a seed into the committed artifacts, which code would run inside a live authorization and which code must never go near one, what every file in `models/` and `data/` is for, and where to put a new attack family, feature, catalog entry or governance gate. No measured result appears here; results live in [the README](../README.md) and in `models/*.json`, both generated from artifacts.*

---

## 1. The closed loop at a glance

The system is a cycle, not a pipeline with an end. Threat intelligence becomes a structured attack specification; the specification is constrained and executed by a deterministic simulator; the defense is trained and measured on the result; the measurement of *what the defense leaned on* becomes the input to the next attack generation; what escapes is replayed into training; and a governance gate decides whether the retrained model is allowed to exist in the authorization path at all.

```
  +--------------------------------------------------------------------------+
  |  OFFLINE ADAPTATION PATH  -  src/loop/redteam_loop.py drives one round    |
  +--------------------------------------------------------------------------+

  [1] THREAT INTELLIGENCE                     src/identify/attacks.json
      Atlas entry: observable signals,         src/identify/taxonomy.py
      auth-time observability, maps_to_injector
                 |
                 v
  [2] ATTACK SPECIFICATION                    src/generate/attack_spec.py
      BASE_SPECS (generation 0), or a          src/generate/llm_agent.py
      proposal aimed at a measured weakness    propose_attack_spec()
                 |
                 v
  [3] CONSTRAINT LAYER                        attack_spec.validate_spec()
      clamp out-of-vocabulary dials,           config.ATTACK_SPEC_BOUNDS
      correct payment-impossible combinations  FAMILY_CONSTRAINTS
                 |
                 v
  [4] DETERMINISTIC SIMULATION                src/generate/simulate.py
      cardholder/merchant profiles, legit      profiles.py, base_generator.py
      backbone, 11 attack injectors            attack_injectors.py
                 |
                 v
  [5] AUTHORIZATION-TIME DEFENSE              src/defend/features.py
      36 auth-time features, supervised +      model.py, train.py, evaluate.py
      anomaly fusion, threshold on a budget    decision_policy.py
                 |
                 v
  [6] WEAKNESS MEASUREMENT                    src/loop/weakness.py
      per-family signal attribution;           analyze_family()
      profile of the transactions that escaped weakest_families()
                 |
                 +---------------------> back to [2] as the next generation
                 |
                 v
  [7] REPLAY + RETRAIN A CANDIDATE            redteam_loop.run_loop()
      bounded, stratified, cumulative buffer;  config.LOOP_CONFIG
      rehearsal of families never attacked
                 |
                 v
  [8] PROMOTION GATE                          src/defend/governance.py
      challenger measured against champion     config.CHAMPION_CHALLENGER
           |                        |
        PROMOTE                  REJECT
           |                        |
           v                        v
   loop_champion_model      recorded in model_registry.json
   .joblib - the only       with the failed gate named, and
   model the ONLINE path    kept out of the authorization path
   is allowed to serve
```

The two things that make this a loop rather than a demo are the arrow from step 6 back to step 2, and the fork at step 8. The first means each attack generation is aimed at a signal the current model was *measured* to depend on. The second means the loop can fail: a candidate that catches more of the new attack while breaching the false-positive ceiling is written to the registry and not shipped.

---

## 2. The four pillars

### Pillar 1 — Identify

**Owns** the Threat Atlas: what attacks exist, what a defender could observe of each one, and which of them this repository actually simulates. **Implemented by** `src/identify/attacks.json` (the data) and `src/identify/taxonomy.py` (the loader and validator). **Hands to Pillar 2** the `maps_to_injector` field, which wires a catalog entry to a transaction-level injector, and `observable_signals`, which is the grounding for the features Pillar 3 computes.

The important design decision here is the `simulator_status` field, with four values: `IMPLEMENTED` (a dedicated injector reproduces the attack's authorization footprint and it is in the default mix), `PARAMETERIZED` (reachable today as a configuration of an existing injector through the specification dials, without new code), `RESEARCH_ONLY` (catalogued and characterised, not simulated) and `FUTURE` (named as planned simulator work). The catalog is deliberately wider than the simulator, and this field is what stops research breadth from being presented as simulation breadth. `taxonomy._validate` enforces the consistency that makes the claim checkable: an entry that is `IMPLEMENTED` or `PARAMETERIZED` must name a real injector, and an entry that is `RESEARCH_ONLY` or `FUTURE` must not claim one. It also rejects duplicate identifiers, unknown injector names, out-of-range novelty scores and unknown observability levels, so a malformed catalog fails at load rather than silently producing a wrong coverage map.

`load_taxonomy()` is `lru_cache`d and returns a `Taxonomy` with lookup helpers (`by_id`, `for_injector`, `by_status`, `simulatable`) and two aggregates the user interface and the documentation builders both read: `coverage_by_category()` and `summary_counts()`.

### Pillar 2 — Generate

**Owns** the production of synthetic payment traffic: a realistic legitimate backbone, eleven attack families, and a constrained specification language that lets an attack be *evolved* rather than rewritten. **Implemented by** `src/generate/`: `attack_spec.py`, `profiles.py`, `base_generator.py`, `attack_injectors.py`, `simulate.py`, `llm_agent.py`, `fidelity.py`. **Hands to Pillar 3** one homogeneous, chronologically sorted transaction table in which fraudulent and legitimate rows share a schema and, by design, overlap heavily.

`AttackSpec` is the contract between the creative half of the red team and the deterministic half of the simulator. It is a frozen dataclass of behavioural dials — `amount_profile`, `velocity_profile`, `device_behavior`, `geo_behavior`, `merchant_behavior`, `timing_profile`, plus a continuous `intensity` — with derived numeric properties (`amount_scale`, `txns_per_card`, `velocity_window_hours`, `device_trust`, `geo_km_range`) that the injectors consume. The language model never writes a transaction row; it writes a specification, and `validate_spec()` decides whether that specification is executable. Out-of-vocabulary values fall back to the family's baseline, numeric fields are clamped to `config.ATTACK_SPEC_BOUNDS`, and `FAMILY_CONSTRAINTS` corrects combinations that cannot happen on a real rail — an authorized push payment run from an attacker's device, for example, is a contradiction in terms, so `scam_transfer` is pinned to `trusted_device`. Every correction is recorded in a `ValidationReport`. Stealth is also monotone within a lineage: a proposal whose intensity is higher than the previous generation's is corrected downward, because a persistent adversary does not become louder after being caught.

Two rules keep the simulator honest, and both are implemented in `attack_injectors.py` rather than asserted in prose. First, **fraud actors have a history**: bust-out accounts, mules and front merchants emit *cover traffic* through `_cover_row()`, labelled `is_fraud=0` because at authorization time it genuinely is not fraud. Without it, "this card has no history" would become a synonym for fraud. Second, **attacks reuse things**: probing revisits the same cards and merchants, mimicry shops at the victim's own regular merchant, laundering pushes many cards through one front. Without reuse, every fraudulent row would trivially be a first-ever card/merchant pair. Supporting details follow the same logic — attack merchants are drawn by popularity through `_weighted_pick()` so that "quiet merchant" cannot become a fraud signal, fraud rides the same day-of-week and payday weighting as genuine spend, and account age is computed on the same clock for both classes via `_account_age()`.

`fidelity.py` is the automated check that these rules held. `separability()` measures univariate ranking power and marginal overlap per feature; `quality_checks()` emits explicit PASS/WARN/FAIL checks including "no single feature separates the classes", "genuine transactions do change device", "fraud sometimes uses a known device", "no history is not a fraud synonym" and "fraud actors have cover traffic". The report is committed as an artifact, so the claim that the data is not trivially separable is a measurement rather than an assertion. Its scope statement is explicit that this compares the simulator against itself and not against any real portfolio.

`llm_agent.py` holds the generative layer: `generate_attack_artifact()` for content artifacts, `propose_attack_spec()` for the structured mutation proposal that matters, and `build_text_corpus()` for the text arm's corpus. Every path degrades to a deterministic offline fallback, so the pipeline, the tests and the web application always run.

### Pillar 3 — Defend

**Owns** the authorization-time detector, its operating point, its decision tiers and its published weaknesses. **Implemented by** `src/defend/`: `features.py`, `model.py`, `train.py`, `evaluate.py`, `baseline.py`, `decision_policy.py`, `diagnostics.py`, `governance.py`, `text_model.py`. **Hands to Pillar 4** a trained model and, through `evaluate()` and the diagnostics, the per-family measurements that tell the red team where to aim.

`features.py` is where leakage is prevented structurally rather than by care. Four column classes are named explicitly: `RAW_COLUMNS` (the raw simulated fields), `FEATURE_COLUMNS` (aliased `AUTH_TIME_FEATURES`, 36 columns, the only thing the model may see), `POST_OUTCOME_COLUMNS` (`refund_flag`, `auth_result` — known only after settlement) and `ORACLE_COLUMNS` (`is_fraud`, `attack_type`, `actor_role` — simulator bookkeeping a real defender never has). `assert_auth_time_safe()` raises if any blocked column reaches the feature matrix, and `build_features()` calls it before returning. One feature deserves its own note: `card_prior_dispute_rate` is derived from `refund_flag`, a post-outcome field, but it is computed as an expanding mean over the card's *earlier* rows and then shifted, so it only ever uses outcomes already known before the current authorization. That is legitimate at authorization time, and it is what gives repeat-dispute behaviour a signal.

Six of the 36 features are `NETWORK_FEATURES`: running counters keyed on device, IP prefix and merchant (`device_card_count_prior`, `card_device_count_prior`, `ip_card_count_prior`, `merchant_card_fanin_prior`, `merchant_new_card_ratio_prior`, `merchant_txn_count_prior`). These are the signals a payment network can compute and a single issuer or merchant cannot, and they are what make card testing, mule rings and transaction laundering visible as network patterns instead of as "this card is unknown to us". They are maintained as O(1) counters over first-time pairings (`_prior_distinct`) rather than as a graph query, and the count-valued ones are log-compressed before they reach the model, because a counter that grows as the window fills would otherwise shift the score distribution over time and drift the threshold off its budget for reasons unrelated to fraud.

`DefenseModel` fuses two heads: a `HistGradientBoostingClassifier` for known patterns and an `IsolationForest` fitted on legitimate rows only, which still reacts to out-of-distribution transactions from families the supervised head never saw. There are two distinct scores and the difference is deliberate. `fused_scores()` is the *detection* score, uncalibrated and fine-grained, and it is what the operating threshold and every ranking metric use. `risk_probability()` applies isotonic calibration and is the *displayed* score, the one the decision tiers act on, so that a number on a reviewer's screen means something. Isotonic regression is monotone, so calibration cannot change the model's ranking — only what the number means. Thresholding is deliberately kept on the raw fusion because a calibrated score is a step function and cannot hit a false-positive budget precisely. `tune_threshold()` picks the lowest threshold whose *realized* false-positive rate on genuine traffic stays inside `config.TARGET_MAX_FPR`, searching distinct score values rather than taking a quantile, precisely because a quantile can land inside a tied block and admit the whole block.

`train.py` owns the split contract. `split_points()` gives one definition of the chronological train/validation/test boundaries so a split can never quietly differ between the trainer, the benchmark harness and the loop. `split_xy()` builds features **once** over the whole time-ordered frame and *then* slices, because building them per slice would restart every card's history, every velocity window and every network counter at the boundary — which no production system does. Since every feature depends only on strictly earlier rows, this introduces no leakage. Calibration and threshold tuning happen on the held-out validation slice, never on training rows and never on the test set.

`baseline.py` supplies the comparison that makes the machine learning claim falsifiable: thirteen transparent, deliberately *untuned* rules over the same authorization-time features, including the network-level rules a fraud team would write once it had the same counters. `compare()` reports every detector at its native operating point and, under `_fpr_matched`, re-thresholds each to spend the same false-positive budget with the threshold chosen on the validation slice and applied unchanged to the test slice.

`decision_policy.py` turns a calibrated probability into `APPROVE`, `STEP_UP` or `DECLINE` with human-readable reason codes drawn from which rules fired. The step-up boundary defaults to the model's own tuned operating threshold rather than a second invented cutoff — `resolve_thresholds()` implements that fallback.

`diagnostics.py` produces the operational picture: `threshold_sweep()` (the curve a fraud team actually argues over), `calibration_report()` (reliability curve and Brier score before and after), `operational_metrics()` (rates translated into review volumes under the declared synthetic scenario), `blind_spots()` (the defense's weakest families, published rather than buried, including the next red-team target), `family_recall_frame()` (per-family recall on a large, fraud-enriched frame generated from an unseen seed) and `head_to_head()` (the stale and adapted defenses scored on the *final evolved* attack generation, both re-thresholded to a matched budget on a separate generation-0 frame).

`text_model.py` is included and labelled for what it is: a trivially separable synthetic sanity check showing where content signals would attach to the architecture. Its module docstring, its `CAVEAT` constant and the pipeline key it is written under all say so. No detection claim in the project rests on it.

### Pillar 4 — Adapt

**Owns** the feedback edge: measuring what the defense depends on, proposing the next attack generation, replaying what escaped into training, and governing whether the result may ship. **Implemented by** `src/loop/weakness.py`, `src/loop/redteam_loop.py`, `src/defend/governance.py` and `src/experiments/leave_one_out.py`. **Hands back to Pillar 2** a validated `AttackSpec` for the next generation, and to Pillar 3 a candidate model plus a promotion decision.

`weakness.py` answers "what is this model actually using?" per family. `analyze_family()` runs a permutation test scoped to one family: shuffle a single feature across that family's fraudulent rows plus a legitimate sample, and measure how much the model's ability to rank that family above genuine traffic drops. A large drop means the model leans on that feature *for this family*, which makes it the obvious thing for the next generation to remove. It also profiles how transactions that escaped differ from those that were caught. `SIGNAL_TO_DIAL` is the red team's playbook: a hand-written mapping from detector signal to the specification dial that neutralises it, with a human explanation attached. `_STEALTH_RANK` guards it — a mutation is only accepted if it moves the attack *closer* to ordinary customer behaviour, so the loop cannot "evade" by handing the defense an easier signal. `heuristic_mutation()` is the deterministic offline proposal and also the reference the language model's proposal is validated against.

`redteam_loop.run_loop()` executes the eleven numbered stages listed in its module docstring. Several decisions in it are worth understanding before changing anything:

- **Targets are measured, not chosen.** `select_focus()` picks the families the defense in force handles worst. Families are excluded from targeting only on evidence: `unlearnable_families()` reads `models/leave_one_out.json` and excludes any family that cannot clear the `LEARNABILITY_FLOOR` even when it is handed to the model directly in training — the most favourable case there is. Such a family is limited by authorization-time observability, not by the decision boundary, and adversarial replay is a slower route to the same thing. Those families are still simulated, scored and reported as structural frontiers.
- **Guard families are the ones not under attack.** `resolve_guards()` resolves them at run time from `GUARD_CANDIDATES` minus the current focus, so a family can never be both the target and the control.
- **The replay buffer is bounded and stratified.** `_bounded_append()` caps total rows at `LOOP_CONFIG["replay_buffer_max"]` while keeping an equal share per generation, so earlier attack generations stay represented. Replayed rows are emphasised with a sample weight (`replay_oversample`) rather than duplicated, which keeps the training frame small and the emphasis explicit.
- **Rehearsal is not optional.** Each round adds not only the focus families' adversarial examples but a sample of every family that was *not* attacked, plus legitimate context. A buffer holding only the newest, hardest attacks pulls the boundary toward them and degrades families the red team never touched — which the forgetting gate then correctly refuses to promote.
- **Exhausted frontiers are retired, not hammered.** A family that stays a residual frontier across two consecutive rounds is dropped from the focus, replaced by the next-weakest untargeted family, and the retirement is recorded in `retired_frontiers` with its reason.
- **Family status is derived from metrics.** `_status()` returns `adapted`, `partial` or `residual_frontier` from the measured recall and gain against `LOOP_CONFIG` thresholds. Nothing is hardcoded.

`governance.py` is the gate. `evaluate_candidate()` runs five checks — attack-recall gain over the champion, an absolute false-positive ceiling, a limit on false-positive regression against the champion, no catastrophic forgetting on previously-learned families or guard families, and no material drop in overall ranking quality — and returns `PROMOTE` or `REJECT` with every gate's observed and required values. The forgetting gate is deliberately statistical: family recall is a proportion measured on a few dozen transactions, so a regression counts only when the upper bound of the candidate's 95% Wilson interval sits below the tolerated level. A gate that stops a genuinely better model on a coin flip is worse than no gate. `registry_entry()` and `write_registry()` produce a plain-JSON audit trail — model version, seed, schema, replay composition, measured metrics, gate results, decision. Entries carry no wall-clock timestamp by default so committed artifacts stay byte-stable.

`leave_one_out.py` is the experiment that motivates the whole loop: for each family, train with it removed and measure recall on it unseen, then train with it included and measure the gain. It runs *before* the loop in the pipeline for a structural reason — the loop reads its output to decide which families are worth attacking. It also owns `select_hero_family()`, the single definition of which family the README, the landing page and the hero demo lead with, so those three can never disagree.

---

## 3. Module reference

### Root

| File | Responsibility | Key public functions |
| --- | --- | --- |
| `config.py` | Every path, seed, domain constant and tuning block. The single source of configuration. | `llm_available()` |
| `src/pipeline.py` | Deterministic end-to-end artifact regeneration in eight numbered steps. | `regenerate(include_loop)`, `main()` |

### `src/identify` — Pillar 1

| File | Responsibility | Key public functions |
| --- | --- | --- |
| `attacks.json` | The Threat Atlas itself: one record per attack, with category, rails, channel, observable signals, auth-time observability, defense difficulty and `simulator_status`. | (data) |
| `taxonomy.py` | Load, validate and query the catalog. Rejects a catalog whose simulator claims do not match its injector mapping. | `load_taxonomy()`, `Taxonomy.for_injector()`, `Taxonomy.coverage_by_category()`, `Taxonomy.summary_counts()` |

### `src/generate` — Pillar 2

| File | Responsibility | Key public functions |
| --- | --- | --- |
| `attack_spec.py` | The `AttackSpec` dataclass, generation-0 `BASE_SPECS`, per-family `FAMILY_CONSTRAINTS`, and the payment-domain constraint layer. | `validate_spec()`, `as_spec()`, `spec_diff()`, `spec_distance()`, `dumps()` |
| `profiles.py` | Cardholder and merchant profile generation from behavioural archetypes; geography, calendar and day-weighting helpers. | `make_rng()`, `generate_cardholders()`, `generate_merchants()`, `assign_relationships()`, `haversine_km()`, `day_weight()` |
| `base_generator.py` | The legitimate transaction backbone: regulars, subscriptions, shopping sessions, payday and weekend clustering, mid-window card issuance and lapse. Owns `COLUMNS`, the canonical 23-column schema. | `generate_legit()` |
| `attack_injectors.py` | Eleven transaction-level fraud injectors plus the shared helpers that keep them honest. Owns `INJECTORS` and `DEFAULT_MIX`. | `card_testing()`, `bust_out()`, `account_takeover()`, `adversarial_mimicry()`, `velocity_smurfing()`, `merchant_laundering()`, `friendly_fraud()`, `otp_relay()`, `geo_anomaly()`, `scam_transfer()`, `wallet_provisioning()` |
| `simulate.py` | Assemble one labelled dataset from profiles, legitimate traffic and attack rows; compute `is_new_payee` globally; persist the dataset and the text corpus. | `simulate()`, `run_and_save()` |
| `llm_agent.py` | The GenAI red-team agent and the offline templates behind it. | `propose_attack_spec()`, `generate_attack_artifact()`, `build_text_corpus()`, `propose_evasion()`, `strip_markers()` |
| `fidelity.py` | Internal synthetic fidelity diagnostics: per-feature separability, marginal overlap, portfolio structure and PASS/WARN/FAIL guards. | `run_fidelity()`, `separability()`, `quality_checks()`, `distributions()`, `portfolio()` |
| `demo_specs.py` | Standalone demonstration that the live GenAI specification path works end to end. Refuses to write anything without an API key. | `run()`, `readiness()` |

### `src/llm`

| File | Responsibility | Key public functions |
| --- | --- | --- |
| `client.py` | Provider-agnostic language-model client with an on-disk cache keyed by a hash of model, system prompt, user prompt and schema tag. Enforces JSON object responses and repairs common wrapping. | `LLMClient.complete()`, `LLMClient.structured()`, `LLMClient.cached_only()`, `get_client()`, `LLMUnavailable` |

### `src/defend` — Pillar 3

| File | Responsibility | Key public functions |
| --- | --- | --- |
| `features.py` | Build the authorization-time feature matrix and enforce the leakage boundary. Owns `FEATURE_COLUMNS`, `NETWORK_FEATURES`, `POST_OUTCOME_COLUMNS`, `ORACLE_COLUMNS`. | `build_features()`, `build_xy()`, `assert_auth_time_safe()` |
| `model.py` | The fused supervised + anomaly detector, its isotonic calibration, its operating threshold and its persistence. | `DefenseModel.fit()`, `.fused_scores()`, `.risk_probability()`, `.fit_calibration()`, `.tune_threshold()`, `.component_scores()`, `.save()`, `.load()` |
| `train.py` | Chronological split contract and the training entry point; permutation importance on the test slice. | `load_dataset()`, `split_points()`, `time_split()`, `split_xy()`, `train_and_eval()` |
| `evaluate.py` | Headline metrics, per-attack recall with Wilson intervals, and a scored frame for the interface. | `evaluate()`, `wilson_interval()`, `scored_frame()` |
| `baseline.py` | Thirteen untuned rules as a transparent baseline, plus the native-point and matched-budget comparison harness. Owns `RULES`. | `rule_hits()`, `rules_predict()`, `rules_score()`, `compare()` |
| `decision_policy.py` | Calibrated probability to `APPROVE`/`STEP_UP`/`DECLINE`, with reason codes. | `decide()`, `resolve_thresholds()`, `reason_codes()`, `decision_frame()`, `policy_summary()` |
| `diagnostics.py` | Threshold sweep, calibration report, operational volumes, blind-spot register, unseen-frame family recall, and the static-vs-adaptive head-to-head. | `run_all()`, `threshold_sweep()`, `calibration_report()`, `operational_metrics()`, `blind_spots()`, `family_recall_frame()`, `head_to_head()` |
| `governance.py` | Champion/challenger promotion gates and the JSON model registry. | `evaluate_candidate()`, `registry_entry()`, `write_registry()`, `load_registry()` |
| `text_model.py` | The text arm, with its caveat attached to every output path. | `train_text_model()`, `score_text()`, `load_corpus()` |

### `src/loop` and `src/experiments` — Pillar 4

| File | Responsibility | Key public functions |
| --- | --- | --- |
| `loop/weakness.py` | Per-family signal attribution, escape profiling, and the signal-to-dial mutation playbook. Owns `SIGNAL_TO_DIAL` and `_STEALTH_RANK`. | `analyze_family()`, `weakest_families()`, `heuristic_mutation()`, `lineage_entry()`, `WeaknessReport` |
| `loop/redteam_loop.py` | The closed-loop engine: focus selection, evolution, replay, retraining, forgetting checks and governance. | `run_loop()`, `select_focus()`, `resolve_guards()`, `unlearnable_families()` |
| `experiments/leave_one_out.py` | Leave-one-attack-family-out evaluation and the shared hero-family selection rule. | `run_loao()`, `select_hero_family()` |

### `app` and `docs`

| File | Responsibility | Key public functions |
| --- | --- | --- |
| `app/common.py` | Path bootstrap, cached artifact loaders, palette, Demo/Live switch and small user-interface helpers. Imported by every page. | `page_setup()`, `mode_selector()`, `is_demo()`, `llm_status_badge()`, and the `load_*` family |
| `app/Home.py` | Landing page: value proposition, the loop, one proof point, the honest disclaimer. | (script) |
| `app/pages/*.py` | The seven detail pages; see section 7. | (scripts) |
| `docs/build_docs.py` | Render `README.md`, the figures and the `.docx` walkthrough **from** the committed artifacts, so no reported number is typed by hand. | `main()` |
| `docs/build_guides.py`, `docs/build_walkthrough.py` | Generate the presentation guides and the walkthrough document from the same artifacts. | called by `build_docs` |

---

## 4. Two runtime paths

The single most important structural property of this system is that the online and offline paths are separate. Conflating them is how a research loop becomes an outage.

### The online authorization path

This is what would run per transaction, inside the authorization window, if the prototype were deployed.

```
  authorization message
        |
        v
  build_features()  ---- reads: the message itself
        |            ---- reads: per-card profile store (history depth, prior
        |                  amount/distance maxima, velocity windows, previous
        |                  device and city, prior dispute rate)
        |            ---- reads: network counter service, keyed on device, IP
        |                  prefix and merchant
        v
  DefenseModel.fused_scores()      -> detection score, thresholded
  DefenseModel.risk_probability()  -> calibrated probability, displayed
        |
        v
  decision_policy.decide()  -> APPROVE | STEP_UP | DECLINE
  decision_policy.reason_codes()  -> why, from the rules that fired
```

The state it reads is entirely *prior* state: a per-card profile updated on settlement, and network counters maintained as O(1) running counts. Everything is a lookup, not a query. `app/pages/7_Deployment.py` renders exactly this mapping — for each of the 36 features, whether it comes off the authorization message, from a per-card profile store, or from a network counter service.

What this path must never do:

- **Read a post-outcome field.** `refund_flag` and `auth_result` are not known when the decision is made. `assert_auth_time_safe()` raises if either reaches the feature matrix.
- **Read a simulator oracle field.** `is_fraud`, `attack_type` and `actor_role` exist only because this is a simulation. `actor_role` in particular is analysis bookkeeping and is excluded by construction.
- **Retrain, regenerate or call a language model.** Nothing in the online path touches `src/generate`, `src/loop` or `src/llm`. The red-team agent is not in the authorization path at any latency.
- **Serve an unpromoted model.** Only `loop_champion_model.joblib` — the last candidate that cleared every gate — represents a model governance approved. `loop_adapted_model.joblib` is the final candidate the loop produced and is reported separately, labelled, precisely so a rejected challenger is never presented as the working defense.

### The offline adaptation path

This is the loop, run on a research cadence — deliberately, not per transaction. It simulates whole frames, trains models, permutes features thousands of times for attribution, and optionally calls a language model. Every one of those is orders of magnitude too slow and too stateful for an authorization window, and none of them needs to be there.

The separation matters for four reasons. **Latency**: the online path is a feature build plus two `predict` calls, and it stays that way only if nothing else is allowed in. **Blast radius**: a loop that retrained a live model would let a red-team proposal reach production traffic without a human or a gate in between. **Reproducibility**: the offline path is seeded and deterministic, so a run can be replayed byte-for-byte; the online path is stateful and continuous, and mixing the two would destroy both properties. **Governance**: the gate at the boundary only means something if there *is* a boundary. The loop produces candidates; `governance.evaluate_candidate()` decides; only then does an artifact become eligible for the online path.

---

## 5. Data and control flow through one pipeline run

`python -m src.pipeline` regenerates every derived artifact from the seed. `--fast` skips step 6. `N_STEPS` is 8; each step prints its own progress line, and the run ends by writing a summary of everything it computed.

| Step | Entry point | Consumes | Produces |
| --- | --- | --- | --- |
| 1 | `src.generate.simulate.run_and_save()` | `config` seed and simulation defaults; `BASE_SPECS`; `DEFAULT_MIX`; the Threat Atlas (for the text corpus) | `data/transactions.parquet`, `data/transactions.csv`, `data/summary.json`, `data/attack_artifacts.jsonl` |
| 2 | `src.generate.fidelity.run_fidelity()` | `data/transactions.parquet` | `models/fidelity_report.json` |
| 3 | `src.defend.train.train_and_eval()` | `data/transactions.parquet` | `models/defense_model.joblib`, `models/metrics.json` |
| 4 | `src.defend.text_model.train_text_model()` | `data/attack_artifacts.jsonl` | `models/text_model.joblib`, `models/text_metrics.json` |
| 5 | `src.experiments.leave_one_out.run_loao()` | a freshly simulated, fraud-enriched frame | `models/leave_one_out.json` |
| 6 | `src.loop.redteam_loop.run_loop()` | `models/leave_one_out.json` (via `unlearnable_families()`); `BASE_SPECS`; the simulator; optionally the language model | `models/loop_history.json`, `models/attack_lineage.json`, `models/model_registry.json`, `models/loop_base_model.joblib`, `models/loop_adapted_model.joblib`, `models/loop_champion_model.joblib`, `models/hero_example.json` |
| 7 | `src.defend.baseline.compare()` | `data/transactions.parquet`, `models/defense_model.joblib`, the loop's champion and candidate models | `models/baseline_comparison.json` |
| 8 | `src.defend.diagnostics.run_all()` | `data/transactions.parquet`, `models/defense_model.joblib`, the loop models and `models/loop_history.json` | `models/threshold_sweep.json`, `models/calibration.json`, `models/operational_metrics.json`, `models/blind_spots.json`, `models/family_recall.json`, `models/head_to_head.json`, `models/defend_demo.parquet` |
| — | `src.pipeline.regenerate()` | the return values of all eight steps | `models/pipeline_summary.json` |

Three orderings in that table are load-bearing rather than incidental.

**Step 5 runs before step 6 on purpose.** The leave-one-out experiment establishes which families are learnable at authorization time at all, and the loop reads that evidence in `unlearnable_families()` to decide which families are worth attacking. Reversing them would leave the loop targeting families that no amount of replay can help.

**Step 7 reports the promoted champion as the adaptive column.** `pipeline.regenerate()` loads `loop_champion_model.joblib` as `adaptive_ml` and, separately, `loop_adapted_model.joblib` as `adaptive_candidate_unpromoted`. A model the gates turned down is never presented as the working defense.

**Step 8 precomputes the demo frame.** `run_all()` writes `models/defend_demo.parquet` — scores, actions and reason codes for the held-out test split — so the web prototype reads a table instead of rebuilding 36 features over tens of thousands of rows on every rerun.

Step 4's result is deliberately keyed in the summary as `text_sanity_check_not_detection_evidence`, carrying its own caveat, so no consumer can quote it as a peer of the detection metrics.

---

## 6. The artifact contract

Everything below is generated. Nothing in `models/` or `data/` is hand-edited, and `docs/build_docs.py` renders the README and the walkthrough *from* these files so that documentation and artifacts cannot drift apart. Paths are declared once in `config.py`; use the constants rather than string literals.

### `data/`

| File | Written by | Read by | Purpose |
| --- | --- | --- | --- |
| `transactions.parquet` | `simulate.run_and_save()` | `train.load_dataset()`, `fidelity.run_fidelity()`, `baseline.compare()`, `diagnostics.run_all()`, `app/common.load_dataset_cached()` | The canonical labelled dataset; the input to everything in Pillar 3. |
| `transactions.csv` | `simulate.run_and_save()` | humans, external tools | A readable copy of the same frame for inspection. |
| `summary.json` | `simulate.run_and_save()` | `app/common.load_summary()`, the documentation builders | Portfolio composition: sizes, fraud rate, cover-traffic count, archetype mix, per-family counts, the specifications used, and the date range. |
| `attack_artifacts.jsonl` | `simulate.run_and_save()` via `llm_agent.build_text_corpus()` | `text_model.load_corpus()`, `app/common.load_artifacts()` | The synthetic content corpus: fraudulent artifacts and benign transactional and security messages. `text` is what the classifier sees; `display_text` keeps the provenance markers for the interface. |

### `models/`

| File | Written by | Read by | Purpose |
| --- | --- | --- | --- |
| `defense_model.joblib` | `train.train_and_eval()` | `DefenseModel.load()` in `baseline`, `diagnostics`, `decision_policy`, `app/common.get_defense_model()` | The static defense: the model trained on the known-attack distribution. |
| `metrics.json` | `train.train_and_eval()` | `app/common.load_metrics()`, documentation builders | Headline metrics, per-attack recall, split sizes, permutation importances. |
| `fidelity_report.json` | `fidelity.run_fidelity()` | `app/common.load_fidelity()` | Per-feature separability and overlap, marginal distributions, portfolio structure, and the PASS/WARN/FAIL checks. |
| `text_model.joblib`, `text_metrics.json` | `text_model.train_text_model()` | `app/common.load_text_metrics()` | The text arm and its scores, always accompanied by `caveat` and `honest_reading`. |
| `leave_one_out.json` | `leave_one_out.run_loao()` | `redteam_loop.unlearnable_families()`, `app/common.load_loao()`, `select_hero_family()` | Unseen versus learned recall per family, with intervals and sample sizes. Also the evidence that excludes structural frontiers from targeting. |
| `loop_history.json` | `redteam_loop.run_loop()` | `diagnostics.head_to_head()`, `app/common.load_loop_history()` | The full record of every round: focus selection, specifications, constraint-layer corrections, stale versus adapted recall, prior-generation and guard-family recall, replay composition, and the governance decision. |
| `attack_lineage.json` | `redteam_loop.run_loop()` | `app/common.load_lineage()` | One node per attack generation: what changed, the measured weakness that motivated it, what the constraint layer corrected, and what it cost the defense. |
| `model_registry.json` | `governance.write_registry()`, called from `run_loop()` | `app/common.load_registry()` (pages 4 and 7) | The audit trail: version, stage, seed, schema, replay composition, metrics, gate results, promote or reject. |
| `loop_base_model.joblib` | `run_loop()` | `diagnostics.head_to_head()` | The stale defense — trained before the loop ran, and the control in every adaptation comparison. |
| `loop_adapted_model.joblib` | `run_loop()` | `pipeline`, `baseline`, `diagnostics` | The final candidate the loop produced. Always labelled as a candidate. |
| `loop_champion_model.joblib` | `run_loop()` | `pipeline`, `baseline`, `diagnostics.head_to_head()` | The last candidate that cleared every promotion gate. The only artifact that represents a deployable defense. |
| `hero_example.json` | `run_loop._save_hero_example()` | `app/common.load_hero()` (page 5) | One concrete evolved transaction the stale model approves and the adapted model routes to friction, with both scores, probabilities and actions. |
| `baseline_comparison.json` | `pipeline` step 7 / `baseline.main()` | `app/common.load_baseline()` | Rules versus static versus adaptive, at native operating points and under `_fpr_matched`. |
| `threshold_sweep.json` | `diagnostics.run_all()` | `app/common.load_threshold_sweep()` (page 3) | The full recall/precision/false-positive curve, the operating point, and the decision tiers. |
| `calibration.json` | `diagnostics.run_all()` | `app/common.load_calibration()` (page 3) | Reliability curves and Brier scores before and after calibration. |
| `operational_metrics.json` | `diagnostics.run_all()` | `app/common.load_operational()` (pages 3 and 6) | Rates translated into review volumes and customer friction under `config.OPERATIONAL_SCENARIO`, with the disclaimer attached. |
| `blind_spots.json` | `diagnostics.run_all()` | `app/common.load_blind_spots()` (page 3) | Families ranked worst-first, the signals the model leans on for each, the escape profile, and the next red-team target. |
| `family_recall.json` | `diagnostics.family_recall_frame()` | `app/common.load_family_recall()` (pages 3 and 6) | Per-family recall on a large, fraud-enriched frame from an unseen seed, with the reporting floor recorded. |
| `head_to_head.json` | `diagnostics.head_to_head()` | `app/common.load_head_to_head()` (page 6) | Stale versus adapted versus promoted champion on the final evolved generation, at matched false-positive budgets. |
| `defend_demo.parquet` | `diagnostics.run_all()` | `app/common.load_demo_scores()` (page 3) | Precomputed scores, actions and reason codes for the held-out test split, so Demo mode renders instantly. |
| `pipeline_summary.json` | `pipeline.regenerate()` | `app/common.load_pipeline_summary()` | One record of what the last full run produced, including the seed and schema version. |
| `genai_spec_demo.json` | `demo_specs.run()` — only with an API key | reviewers | Evidence that the live GenAI specification path works end to end. Feeds nothing. |

`artifacts_cache/` holds language-model responses keyed by a hash of model, system prompt, user prompt and schema tag, so a live run is reproducible and an offline run can replay it.

`config.SCHEMA_VERSION` is bumped whenever the simulator schema or the feature contract changes, and it is stamped into most artifacts, so a stale artifact is detectable rather than silently wrong.

---

## 7. The Streamlit application

The prototype is a standard Streamlit multipage app. `app/Home.py` is the entry point (`streamlit run app/Home.py`); each file in `app/pages/` becomes one page, and Streamlit derives both the sidebar label and the URL path from the filename after stripping the numeric ordering prefix. The browser tab title comes from each page's own `page_setup()` call.

| File | Sidebar label | URL path | Reads |
| --- | --- | --- | --- |
| `Home.py` | (entry point) | `/` | `data/summary.json`, `metrics.json`, `loop_history.json`, `leave_one_out.json`, the taxonomy |
| `pages/1_Threat_Atlas.py` | Threat Atlas | `/Threat_Atlas` | the taxonomy only (`attacks.json`) |
| `pages/2_Generate.py` | Generate | `/Generate` | `summary.json`, `transactions.parquet`, `fidelity_report.json`, `attack_artifacts.jsonl`, `text_metrics.json` |
| `pages/3_Defend.py` | Defend | `/Defend` | `metrics.json`, `threshold_sweep.json`, `operational_metrics.json`, `calibration.json`, `family_recall.json`, `blind_spots.json`, `defend_demo.parquet` |
| `pages/4_ClosedLoop.py` | ClosedLoop | `/ClosedLoop` | `loop_history.json`, `attack_lineage.json`, `model_registry.json` |
| `pages/5_HeroDemo.py` | HeroDemo | `/HeroDemo` | `hero_example.json`, `leave_one_out.json`, `loop_history.json` |
| `pages/6_Benchmarks.py` | Benchmarks | `/Benchmarks` | `baseline_comparison.json`, `leave_one_out.json`, `head_to_head.json`, `operational_metrics.json`, `family_recall.json` |
| `pages/7_Deployment.py` | Deployment | `/Deployment` | `model_registry.json`, plus `FEATURE_COLUMNS` and `NETWORK_FEATURES` read directly from the code |

Every loader lives in `app/common.py` and is wrapped in `st.cache_data` or `st.cache_resource`, and every one returns `None` when its artifact is missing so a page can degrade to an instruction ("run this command") instead of a traceback.

### Demo mode versus Live mode

`common.mode_selector()` renders a sidebar toggle and returns `"DEMO"` or `"LIVE"`. It defaults to Demo.

**Demo mode** renders entirely from committed artifacts. It never simulates, never trains, never calls a language model and never needs an API key. Page 3 reads `defend_demo.parquet` rather than rebuilding features; page 2 reads the committed portfolio. This is the mode a demonstration is given in, because the failure mode it removes — a page that recomputes on stage and stalls — is the one that actually ruins a presentation.

**Live mode** unlocks regeneration and retraining behind explicit controls: page 2 offers a "Generate fresh portfolio" button with size, fraud-rate and seed inputs; page 4 offers a rounds slider, a confirmation checkbox acknowledging that this retrains models, and a "Run closed loop" button; page 7 offers an on-demand scoring-latency benchmark. Nothing heavy runs without a click.

Orthogonal to the mode switch, `common.llm_status_badge()` states plainly which generation path is available and reminds the reader that every committed artifact was produced by the deterministic offline path. That claim is the easiest one in the project to check, which is exactly why it is made explicitly.

---

## 8. Configuration

`config.py` is the only place a path, seed or tuning constant should be defined. The significant blocks:

| Block | Controls |
| --- | --- |
| Paths (`ROOT`, `DATA_DIR`, `MODELS_DIR`, `CACHE_DIR`, and one constant per artifact) | Every artifact location. The directories are created at import. Always reference the constants, never a literal path. |
| `GLOBAL_SEED` | The single root of reproducibility. Every generator, model and sampling step derives from it, usually as `GLOBAL_SEED + offset` so distinct frames stay distinct but deterministic. |
| Simulation defaults (`DEFAULT_N_CARDHOLDERS`, `DEFAULT_N_MERCHANTS`, `DEFAULT_N_TRANSACTIONS`, `DEFAULT_FRAUD_RATE`) | The size and base rate of the headline portfolio. The fraud rate is set to a realistic card-portfolio level, which is what makes the precision and false-positive figures meaningful. |
| `FAMILY_EVAL` | The separate, larger, fraud-enriched frame used for per-family recall, including its unseen seed offset and the minimum sample size below which a family-level number is not quoted. |
| `SCHEMA_VERSION` | Artifact and feature-contract versioning. Bump it whenever the simulator schema or `FEATURE_COLUMNS` changes. |
| Language-model block (`ANTHROPIC_API_KEY`, `LLM_MODEL`, `LLM_ENABLED`, `LLM_MAX_TOKENS`, `llm_available()`) | Whether live generation is possible. Read from the environment, with `.env` loaded automatically when `python-dotenv` is installed. When unavailable, every caller falls back to the offline path. |
| Rail and domain constants (`RAILS`, `UPI_REALISM`, `MCC_CATALOG`, `GEO_CLUSTERS`, `HIGH_RISK_COUNTRIES`) | The payment world the simulator lives in: three rails, twelve merchant category codes with ticket distributions and channel bias, ten geography clusters, and the risk-listed jurisdictions. `UPI_REALISM` carries the semantics that differ from cards — per-payment payer authentication, instant irrevocable settlement, smaller tickets, a payee address rather than a card acceptor. |
| `TARGET_MAX_FPR` | The false-positive budget the operating threshold is tuned against, and the budget every matched-rate comparison spends. |
| `LEGIT_REALISM` | The rates that make genuine customers behave with credible variation: multi-device ownership, replacement devices, travel, high-value purchases, odd hours, thin files, subscriptions, shopping bursts, payday and weekend clustering, mid-window issuance and lapse, and rare genuine high-risk-country spend. These exist so that risk indicators are indicators rather than shortcuts, and the file says explicitly that they must not be tuned to hit a target metric. |
| `CUSTOMER_ARCHETYPES` | Seven behavioural templates (weights summing to one) controlling spend level, frequency, category preference, device count, travel and night propensity, weekday shape and channel mix. A portfolio built from one behaviour is trivially separable from fraud; one built from several overlapping behaviours is not. |
| `FRAUD_REALISM` | How much cover traffic each fraud-actor type builds, how often probing and victim cards repeat, and how much geo-anomaly fraud sits in a high-risk jurisdiction. |
| `DECISION_THRESHOLDS` | The decline tier on the calibrated probability, and a step-up tier that defaults to `None`, meaning "use the model's own tuned operating threshold" rather than a second invented cutoff. |
| `LOOP_CONFIG` | Replay-buffer cap, replay sample weight, the recovery delta and residual-recall ceiling that derive family status, the tolerated false-positive regression and forgetting tolerance, and the focus-mix boost that keeps targeted families numerous enough for stable per-family recall. |
| `ATTACK_SPEC_BOUNDS` | The permitted vocabulary for every specification dial plus hard simulation limits on transactions per card per day, amount range and variants per round. Nothing outside these bounds is ever executed. |
| `CHAMPION_CHALLENGER` | The five promotion gates: minimum attack-recall gain, absolute false-positive ceiling, permitted false-positive regression, permitted prior-family recall drop, and permitted overall ranking-quality drop. |
| `OPERATIONAL_SCENARIO` | The declared, explicitly illustrative synthetic assumptions — monthly authorization volume, analyst throughput, step-up abandonment, assumed loss and review costs — that let a false-positive rate be read as review volume and customer friction. Its own `label` field states that these are not anyone's real economics. |

---

## 9. Extension points

Each of these is a short summary of *where* the change goes. The step-by-step procedure, including what to run and which tests must pass, belongs in [CONTRIBUTING.md](CONTRIBUTING.md).

**A new attack family.** Five places, in order. Add a catalog entry in `src/identify/attacks.json` with `maps_to_injector` set and `simulator_status: "IMPLEMENTED"`. Write the injector in `src/generate/attack_injectors.py`, following the two honesty rules — emit cover traffic where the actor would plausibly have history, and reuse cards and merchants — and register it in `INJECTORS` and `DEFAULT_MIX`. Add a generation-0 entry to `BASE_SPECS` in `src/generate/attack_spec.py`, and, if the family has behaviours that are impossible on a real rail, a `FAMILY_CONSTRAINTS` entry. Add the injector name to the `injectors` list in `attacks.json` so `taxonomy._validate` accepts the mapping. Finally consider whether the family belongs in `DEFAULT_FAMILIES` in `src/experiments/leave_one_out.py` and in `GUARD_CANDIDATES` in `src/loop/redteam_loop.py`.

**A new feature.** Compute it in `build_features()` in `src/defend/features.py` using only strictly-earlier rows, add its name to `FEATURE_COLUMNS` (and to `NETWORK_FEATURES` and `_COUNT_FEATURES` if it is a relational counter that grows over the window), and give it an honest fill value for rows with no history. Then extend `SIGNAL_TO_DIAL` in `src/loop/weakness.py` so the red team knows which specification dial neutralises it — a feature with no entry there is invisible to the mutation heuristic. Add it to the provenance mapping in `app/pages/7_Deployment.py` so the deployment view can say where it would be computed, and bump `SCHEMA_VERSION`.

**A new catalog entry.** Add the record to `src/identify/attacks.json` and pick the honest `simulator_status`. The validator will reject the two failure modes that matter: claiming an injector the repository does not have, and claiming `RESEARCH_ONLY` or `FUTURE` while also naming one. Every field must describe what a *defender* can observe; the catalog contains no operational guidance and must stay that way.

**A new governance gate.** Add the check inside `evaluate_candidate()` in `src/defend/governance.py`, using the `_gate()` helper so the gate records its observed and required values and its own human-readable detail, and add its threshold to `config.CHAMPION_CHALLENGER`. If the gate measures a proportion on a small sample, follow the forgetting gate's precedent and judge it against a Wilson interval rather than a point estimate. Nothing else needs to change: `run_loop()` reports whatever gates the function returns, and `write_registry()` stamps the current thresholds into the registry.

---

## 10. Where to go next

If you want the measured results, read [the README](../README.md) — every figure in it is generated from `models/*.json` by `docs/build_docs.py`, so it is the only place numbers should be quoted from. If you are about to change code, read [CONTRIBUTING.md](CONTRIBUTING.md) for the procedure and the test expectations. If you are preparing to present or evaluate the system, [DEMO_GUIDE.md](DEMO_GUIDE.md) walks the application page by page and [JUDGE_QA.md](JUDGE_QA.md) collects the questions the design anticipates, both generated from the same artifacts.
