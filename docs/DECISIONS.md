# Architecture Decision Records

*This log is for the engineer or reviewer who wants to know why this laboratory is built the way it is, rather than what it does. Each record states a problem, the evidence that revealed it, the decision taken, and what that decision bought, cost and ruled out. After reading it you should be able to open any module named here and recognise the reasoning behind the code, and you should be able to tell which choices were forced by payment-domain reality, which by measurement, and which by the need for a demonstration that reproduces byte-for-byte from a single seed.*

## How to read this log

Records are numbered by significance, not chronology: the earliest records describe decisions that shape everything downstream, the later ones describe decisions about honesty in reporting. Every record is `Accepted` — this log documents the system as built, and a decision that was tried and reversed appears as the `Context` of the record that replaced it.

**No record contains a measured number.** Recall, precision, area under a curve, false-positive rates and every other measurement live in [`README.md`](../README.md) and in `models/*.json`, both of which are generated from artifacts by `docs/build_docs.py`. That separation is itself a decision (record 18), and applying it to this document is what stops the log from ever contradicting the generated figures. Where a decision was driven by a measurement, the record describes the *nature* of the problem and names the artifact or test where the number can be read.

Structural counts — how many attack injectors exist, how many feature columns the model consumes — are stable properties of the code and are quoted here with the file that defines them.

---

## 1. Generate attack specifications with a language model, not transaction rows

### Status

Accepted

### Context

The obvious way to put generative AI into a fraud simulator is to ask it for fraudulent transactions. It is also the wrong way. A language model asked for transaction rows produces output that is neither reproducible nor verifiable: two runs disagree, the rows carry no guarantee of internal consistency with the rest of the portfolio, per-card history and network counters cannot be maintained across them, and there is no point at which a domain rule can be enforced. Worse, the resulting dataset cannot be defended to a reviewer, because nobody — including the authors — can say precisely what behaviour was generated or why.

The system also has to run on stage with no API key and no network, and every committed figure has to be reproducible from `config.GLOBAL_SEED`.

### Decision

The language model writes a *specification*, never a row. `src/generate/attack_spec.py` defines `AttackSpec`, a frozen dataclass describing one attack generation on a fixed vocabulary of behavioural dials, plus an `intensity` scalar and provenance fields (`source`, `generation`, `targets_signal`, `rationale`).

| Dial | Permitted values | Derived simulator knob |
|---|---|---|
| `amount_profile` | micro, low, moderate, high, extreme | `AMOUNT_SCALE` multiplier on the ticket |
| `velocity_profile` | single, low_and_slow, moderate, burst | `VELOCITY_TXNS`, `VELOCITY_WINDOW_H` |
| `device_behavior` | trusted_device, secondary_device, new_device, shared_device | `DEVICE_TRUST` probability |
| `geo_behavior` | home, plausible, domestic_far, foreign, high_risk | `GEO_KM` distance band |
| `merchant_behavior` | known_merchant, new_low_risk_merchant, new_high_risk_merchant, front_merchant, cash_like | merchant pool selection |
| `timing_profile` | customer_normal, business_hours, night, any | hour-of-day distribution |

The vocabularies are declared once in `config.ATTACK_SPEC_BOUNDS` and are repeated to the model verbatim in `_SPEC_VOCAB` in `src/generate/llm_agent.py`. `propose_attack_spec()` hands the model the measured weakness report and asks it to change as few dials as possible; the eleven injectors in `src/generate/attack_injectors.py` then execute the resulting specification deterministically against a seeded generator.

### Consequences

This buys reproducibility and inspectability at once. Because the simulator is deterministic given the seed, a specification produced by Claude and the same specification produced by the offline heuristic yield identical data, which is why `models/attack_lineage.json` can record exactly what changed between generations and why. It also buys a clean safety boundary: the model is only ever asked to choose values on dials a sandboxed simulator will execute, and is never in a position to emit anything operational.

The cost is expressive range. A tactic that cannot be expressed as a combination of these six dials cannot be proposed, so the red team's creativity is bounded by the simulator's behavioural surface rather than by the model. Extending the attack surface means writing an injector, not writing a prompt.

It rules out any claim that the system discovers genuinely novel attack *mechanisms*. What it discovers is which behavioural configuration defeats the current defense, which is a narrower and more defensible claim.

---

## 2. Validate every specification against payment-domain constraints before execution

### Status

Accepted

### Context

A specification that is structurally well-formed can still be nonsense on a real payment rail. A language model — or a careless heuristic — can propose an authorised push payment executed from an attacker's device, which is a contradiction in terms: the defining property of that fraud is that the genuine customer authenticates and authorises the payment themselves. It can propose an intensity outside the permitted band, a dial value outside the vocabulary, or a card-testing campaign at extreme ticket values, which no probing operation would run because the economics do not work.

If any of those were executed, the resulting data would still look plausible in aggregate, and the lineage narrative would be describing something the dataset does not contain.

### Decision

Nothing reaches the simulator without passing `validate_spec()` in `src/generate/attack_spec.py`. The constraint layer runs in three stages and records everything it did:

```
proposal (dict, from Claude or the heuristic)
      |
      |  1. numeric ranges      intensity clamped to ATTACK_SPEC_BOUNDS["intensity"];
      |                         stealth is monotone within a lineage — an attack
      |                         may not become louder after being caught
      |
      |  2. categorical vocab   _coerce_choice() replaces any out-of-vocabulary
      |                         value with the family's baseline value
      |
      |  3. family requirements FAMILY_CONSTRAINTS: per-family sets of permitted
      |                         values, corrected to the nearest legal one
      |                         (or, with strict=True, raising SpecRejected)
      v
AttackSpec  +  ValidationReport
```

`FAMILY_CONSTRAINTS` carries requirements for ten of the eleven families. `scam_transfer` and `friendly_fraud` are pinned to `trusted_device` and to home or plausible geography because the genuine customer is the one transacting. `card_testing` is restricted to micro or low amounts at moderate or burst velocity. `adversarial_mimicry` is restricted to the victim's own centroid, because an attack that left it would simply be a different family.

The `ValidationReport` is not discarded. It is carried into `models/loop_history.json` as the `constraint_layer` block for every round and family, and into `models/attack_lineage.json` for every generation, so a reviewer can see exactly which proposals needed correcting.

### Consequences

This buys the ability to let the creative half of the system be genuinely creative. Because nothing invalid can be executed, the prompt does not need defensive hedging and a surprising proposal is a feature rather than a risk. It also produces evidence: the count of corrections applied is a visible artifact rather than an assurance.

The cost is that a correction is silent from the model's point of view — the loop does not re-prompt with the rejection, it simply clamps and proceeds. That is a deliberate simplification for a prototype; a production red team would want the feedback edge closed.

It rules out ever presenting a lineage story the data does not support, because the specification that generated the data is the same object the story is told from.

---

## 3. Give fraud actors cover traffic, labelled legitimate

### Status

Accepted

### Context

An early version of the simulator emitted only the abusive transactions a fraud actor makes. The consequence showed up in the fidelity diagnostics: the share of first-ever-transaction rows among fraud was wildly higher than among genuine traffic, because mule accounts, bust-out accounts and front merchants sprang into existence at the moment of the fraud. "This card has no history" had become a near-synonym for "this card is fraudulent". A model trained on that data learns the generator's bookkeeping, not fraud behaviour, and every downstream number becomes meaningless.

The check that catches this is `no_history_is_not_a_fraud_synonym` in `src/generate/fidelity.py`, which compares the share of rows with zero card history in each class and fails when the ratio is too large; `test_absence_of_history_is_not_a_synonym_for_fraud` in `tests/test_shortcuts_and_fidelity.py` asserts the same property.

### Decision

Fraud actors build history first. `_cover_row()` in `src/generate/attack_injectors.py` emits an ordinary-looking transaction from a fraud actor's own account, through the same amount, channel and geography code paths the legitimate generator uses, and labels it `is_fraud=0` with `actor_role="fraud_actor_cover"`. The volume of cover traffic per actor is configured in `config.FRAUD_REALISM` (`bustout_cover_txns`, `mule_cover_txns`, `laundering_cover_txns`), and bust-out grooming is scheduled by picking the day of the bust first and working the ramp backwards, so the family does not pile up at the end of the window.

The label is the point. At authorisation time a mule account's grooming transaction is indistinguishable from genuine spend; calling it fraud would hand the model a label it could not possibly earn in production.

Separately, `LEGIT_REALISM["late_activation_fraction"]` gives a minority of *genuine* cards their first-ever authorisation mid-window, attacking the same shortcut from the other side.

### Consequences

This buys a portfolio in which absence of history is a weak risk indicator rather than a label. It also buys the false-positive measurement its meaning: a defense that flags cover traffic pays for it, exactly as it would in production.

The cost is that cover rows consume the requested dataset size without contributing to the fraud count — `simulate()` reports `n_cover_transactions` separately for this reason — and that the effective fraud rate is composed rather than dialled directly.

It rules out labelling by actor identity. `actor_role` documents who produced a row for analysis, and is listed in `ORACLE_COLUMNS` in `src/defend/features.py` so that it can never become a feature.

---

## 4. Make attacks reuse cards, devices and merchants

### Status

Accepted

### Context

Two shortcuts were found in the simulator and both had the same root cause: every fraudulent row was a fresh, unrelated event.

The first was `is_new_payee`. Because each fraudulent transaction happened to be the first time that card had met that merchant, the flag fired on almost all fraud while firing on a much smaller share of genuine traffic. A single boolean was close to a label.

The second was the velocity features. `velocity_1h` and `velocity_24h` sat at their floor for `card_testing` and `velocity_smurfing` — the two families whose entire definition is burst behaviour — because each probe minted a new card identifier, so no card ever accumulated a short-window count. The simulator was not producing the pattern its own family names claimed.

Both are recorded in the module docstring of `tests/test_shortcuts_and_fidelity.py` and both now have a regression test.

### Decision

Reuse is built into the injectors. `card_testing` draws a small pool of test cards and a small pool of merchants per burst and revisits both, on a shared botnet device. `account_takeover` re-hits the same victim card and, with probability `FRAUD_REALISM["victim_same_merchant_prob"]`, the same merchant. `merchant_laundering` charges the same compromised card through the front merchant more than once. `adversarial_mimicry` shops at the victim's own regular merchant, which is available because regular-merchant relationships live on the `Cardholder` profile in `src/generate/profiles.py` rather than inside the legitimate generator.

`is_new_payee` itself is computed once, globally, in `src/generate/simulate.py`, as the first chronological occurrence of a card and merchant pair — one definition shared by fraudulent and genuine rows.

The guarding tests are two-sided on purpose: `test_binary_features_are_not_one_sided_shortcuts` fails a flag that fires on nearly all fraud even if it also fires on some genuine traffic. Asserting only that a feature is non-zero on genuine traffic is what let the first shortcut survive.

### Consequences

This buys live velocity and relational features. Short-window counts, device-to-card counters and merchant fan-in only carry signal if attacks actually reuse entities, and they are the signals the network-level features in record 7 depend on.

The cost is a more entangled generator: injectors must maintain their own pools and windows, and a change to a reuse parameter moves several features at once.

It rules out treating any single categorical flag as a headline discriminator, which is the intended outcome. `no_single_feature_separates_classes` in `src/generate/fidelity.py` enforces it globally.

---

## 5. Add card issuance and attrition inside the simulation window

### Status

Accepted

### Context

Giving a minority of genuine cards their first authorisation mid-window (record 3) fixed one shortcut and created another. Cards that only start transacting partway through the window push genuine volume steadily later, while fraud stays evenly spread across it. The fraud *rate* therefore drifts downward across the window, and because the train, validation and test split is chronological, the test slice inherits a different base rate from the slice the model trained on.

That drift is a property of the generator, not of fraud. It would make every reported metric a measurement of the simulator's shape as much as of the detector, and it would do so invisibly.

### Decision

Attrition balances issuance. `Cardholder` in `src/generate/profiles.py` carries both an `activation_day` and a `lapse_day`; `LEGIT_REALISM["late_activation_fraction"]` and `LEGIT_REALISM["lapse_fraction"]` are set to the same value, and the configuration comment in `config.py` states the reason. `_day_probs()` in `src/generate/base_generator.py` zeroes a holder's per-day sampling weight before activation and after lapse, so a card genuinely enters and leaves the portfolio.

Attrition is applied only to cards that were already active at the start of the window, so a card cannot both arrive late and leave early.

`test_legit_volume_does_not_drift_across_the_window` in `tests/test_shortcuts_and_fidelity.py` splits a simulated frame into quarters and asserts the fraud rate does not vary between them beyond a stated ratio.

### Consequences

This buys a chronological split whose slices are comparable, which is what makes the headline metrics readable as a deployment story: train on earlier traffic, evaluate against later traffic.

The cost is realism spent on bookkeeping rather than on behaviour — expiry, replacement and churn are modelled as a single lapse day rather than as distinct events — and a small reduction in effective transactions per card.

It rules out a simpler generator in which every card is present for the whole window. That version is easier to reason about and produces a drifting base rate, which is the worse trade.

---

## 6. Build features once over the whole ordered frame, then split

### Status

Accepted

### Context

The natural instinct when splitting a dataset is to split first and build features per slice. For a model whose features are almost entirely historical, this is quietly destructive: every card's history depth, every velocity window, every expanding mean and every relational counter restarts at the slice boundary. The test slice then looks like a portfolio of brand-new cards, which no production system ever sees, and the historical features the defense depends on are weakest exactly where they are being measured.

The competing worry — that building features over the whole frame leaks future information into training rows — does not apply here, because every feature in `src/defend/features.py` is computed from strictly earlier rows: `expanding().shift()`, trailing rolling windows, and running counters over first-time pairings.

### Decision

`split_xy()` in `src/defend/train.py` sorts the frame by timestamp, builds the full feature matrix once, and then slices it. Boundaries come from `split_points()`, a single function shared by every caller, so a split can never quietly differ between the trainer, the benchmark harness and the closed loop. `time_split()` still exists for callers that need the raw frames, and its docstring points at `split_xy` for anything that trains or evaluates.

Every consumer uses it: `train_and_eval()`, `compare()` in `src/defend/baseline.py`, `run_loao()` in `src/experiments/leave_one_out.py`, and `_prepare_frame()` in `src/loop/redteam_loop.py`.

The no-leakage property is tested rather than asserted. `test_network_counters_only_look_backwards` builds features over a full frame and over a prefix of it, and requires the prefix rows to be identical in both — a row's features must not change when later rows are appended.

### Consequences

This buys features that behave in evaluation the way they behave in production, and it buys agreement between artifacts: because `compare()` uses the same split contract as the trainer, the static-model figures in `models/baseline_comparison.json` match those in `models/metrics.json`, which `test_baseline_agrees_with_metrics_on_the_static_model` enforces.

The cost is that the feature build is a whole-frame operation and cannot be streamed, and that adding a feature which looks forward would break the guarantee silently were it not for the test.

It rules out per-slice feature construction anywhere in the codebase, including in ad-hoc analysis, because the shared helpers make the correct path the easy one.

---

## 7. Add network-level relational counters as authorisation-time features

### Status

Accepted

### Context

Several fraud families are invisible from a single card's history and obvious from the network's vantage point. Card testing is a device carrying many cards. A mule ring is one device or one payee accumulating fan-in across accounts. Transaction laundering is a merchant whose traffic is almost entirely first-time cards. Restricted to per-card features, a detector can only express these as "this card is unknown to us", which is the shortcut record 3 exists to eliminate.

The constraint is that anything added must be computable at authorisation time, in the moment, without a graph query.

### Decision

Six relational counters are added, listed in `NETWORK_FEATURES` in `src/defend/features.py`: `device_card_count_prior`, `card_device_count_prior`, `ip_card_count_prior`, `merchant_card_fanin_prior`, `merchant_new_card_ratio_prior` and `merchant_txn_count_prior`. They are implemented by `_prior_distinct()` as running counters over first-time pairings — precisely how a network-side counter service would maintain them, and an O(1) lookup at decision time rather than a traversal. A seventh merchant-relative feature, `merchant_amount_zscore`, asks whether this ticket is unusual *for this merchant*, because a five-figure charge is ordinary at a travel agent and extraordinary at a grocer.

Two details matter. The counters that grow monotonically as the window fills are log-compressed before they reach the model, listed in `_COUNT_FEATURES`, because on a raw scale a model tuned on earlier traffic meets a systematically different distribution later and its threshold drifts off budget for reasons unrelated to fraud. And `merchant_new_card_ratio_prior` is filled with the midpoint on a merchant's first-ever transaction, which is the honest "no information" value rather than one that implies safety.

The full authorisation-time contract is 36 columns, defined by `FEATURE_COLUMNS`; post-outcome fields (`refund_flag`, `auth_result`) and simulator oracle fields (`is_fraud`, `attack_type`, `actor_role`) are hard-blocked by `assert_auth_time_safe()`, which `test_the_leakage_guard_actually_fires` verifies actually raises.

### Consequences

This buys the ability to detect ring behaviour as ring behaviour. It also buys a fair comparison, because the rules baseline is given the same counters (record 11) — the benchmark then measures what machine learning adds on top of good rules with equal information, not what it adds on top of a blindfolded one.

The cost is a vantage-point assumption. A single issuer cannot compute merchant fan-in across the ecosystem; these features presuppose a network position, and the prototype should be read as describing what that position makes possible rather than what any one participant could deploy unchanged.

It rules out framing the defense as a purely per-card model, and it commits the simulator to producing genuine entity reuse (record 4), without which the counters would be constant.

---

## 8. Threshold detection on the raw fused score while displaying a calibrated probability

### Status

Accepted

### Context

The detection score is a blend of a supervised gradient-boosting head and an isolation forest fitted on legitimate traffic only. That blend is on no meaningful scale: a value of 0.30 means nothing to a reviewer, and a decision tier expressed on it is an arbitrary cut. Isotonic calibration fixes the meaning — after fitting, 0.30 means roughly three in ten transactions scoring here were fraudulent.

Using the calibrated score for detection as well, however, breaks the operating point. Isotonic regression is a step function: large blocks of genuine transactions collapse onto identical values. A threshold that lands inside a tied block admits the entire block under the `>=` comparison, and the realised false-positive rate overshoots the agreed budget — quietly, and by a multiple rather than a margin.

### Decision

Two scores, with two jobs, both defined in `src/defend/model.py`.

`fused_scores()` is the detection score: the uncalibrated blend, continuous and fine-grained. `predict()`, `tune_threshold()`, `evaluate()` and every recall computation in the closed loop use it.

`risk_probability()` is the displayed score: the same fusion passed through the isotonic calibrator fitted by `fit_calibration()` on a held-out slice, never on training rows. It is what the tiered decision policy consumes and what a reviewer sees. `threshold_probability` expresses the operating threshold on that same scale, so the two views stay reconcilable.

Because isotonic regression is monotone, calibration cannot change the model's ranking. Area under the receiver operating characteristic and the precision-recall curve are identical either way, which is what makes the split safe.

### Consequences

This buys a score a human can act on without giving up a threshold that lands where it was asked to land. It also buys interpretability at the policy layer: `decide()` in `src/defend/decision_policy.py` takes calibrated probabilities, so its tiers are statements about how likely fraud is.

The cost is two numbers in circulation, and the discipline of knowing which one each call site wants. The code carries `raw_fused_scores` as an explicit alias and documents the distinction in both docstrings for that reason.

It rules out reporting a single "score" without saying which one, and it rules out tiering the decision policy on the raw fusion.

---

## 9. Select the threshold by searching realised false-positive rates, not by quantile

### Status

Accepted

### Context

Picking an operating point by quantile is the conventional shortcut: to spend a budget of *b*, take the *(1 − b)* quantile of genuine scores. It works when scores are continuous and distinct. It fails when they are not, for the reason described in record 8 — the quantile falls inside a tied block, the comparison admits the whole block, and the realised rate exceeds the budget.

A stated budget that the shipped model breaches is worse than having no budget at all, because it is an unverified claim printed next to verified ones.

### Decision

`tune_threshold()` in `src/defend/model.py` enumerates the distinct genuine scores, computes for each candidate the false-positive rate that would actually be realised if the threshold sat there — the share of genuine scores at or above it, via `searchsorted` — and takes the lowest candidate whose realised rate stays within `config.TARGET_MAX_FPR`. Choosing the lowest such candidate maximises recall subject to the budget. If even the top score exceeds the budget, the threshold is placed just above every observed score rather than flagging the whole book.

The identical routine appears as `_threshold_for_fpr()` in `src/defend/baseline.py` so that the matched-budget comparison (record 12) selects operating points for the rules baseline and for every model by the same rule.

Thresholds are tuned on a held-out validation slice, never on training rows, because train-tuned thresholds drift under time shift.

### Consequences

This buys a budget the artifacts can be checked against. `test_the_operating_point_respects_its_own_budget` in `tests/test_artifact_consistency.py` asserts that the realised rate in `models/metrics.json` does not exceed the budget declared in `config.py`.

The cost is a linear scan over distinct genuine scores rather than a constant-time quantile, which is irrelevant at this scale and would need revisiting at production volume.

It rules out quantile-based thresholding anywhere in the repository, including in the benchmark harness where it would have been the path of least resistance.

---

## 10. Derive the step-up decision tier from the model's tuned operating threshold

### Status

Accepted

### Context

The decision policy has three outcomes: approve, challenge with a step-up, or decline. That requires two boundaries. The decline boundary is a genuine policy choice about when an issuer is confident enough to refuse outright. The step-up boundary is not — it is the point at which a transaction is worth friction, which is exactly the question `tune_threshold()` already answers against an agreed false-positive budget.

Introducing a second, independently chosen number for it would create a magic constant with no derivation, and one that would silently stop agreeing with the budget the moment the budget changed.

### Decision

`config.DECISION_THRESHOLDS` sets `step_up` to `None`, with a comment stating that `None` means "use the model's tuned operating threshold". `resolve_thresholds()` in `src/defend/decision_policy.py` fills it in from the `step_up` argument passed by the caller, and callers pass `model.threshold_probability` — the operating threshold expressed on the calibrated scale, so the tiers are consistently probabilities (record 8).

Only the decline tier remains a declared constant.

### Consequences

This buys one fewer unjustified number, and it buys coupling in the right direction: change the false-positive budget in `config.TARGET_MAX_FPR` and the step-up boundary moves with it automatically, in every artifact and every page that reports the policy.

The cost is that the step-up tier cannot be tuned independently of the detection budget without overriding the configuration explicitly. For a prototype that is the correct default; a deployment with separate friction and review economics would want to unpick it, which the `step_up` argument allows.

It rules out reporting a policy whose step-up rate and whose detection false-positive rate disagree, since they are now the same operating point described twice.

---

## 11. Make the rules baseline competent, not a strawman

### Status

Accepted

### Context

Every fraud machine-learning demonstration includes a rules baseline, and most of them include a weak one. A baseline built from three obvious rules makes any model look transformative and tells a reviewer nothing, because the honest question is not whether machine learning beats bad rules but whether it beats the rules a competent fraud team would actually write — including, once the network counters of record 7 exist, rules over those counters.

There is a second trap. A rules baseline whose thresholds were tuned on this dataset is no longer a baseline; it is a second fitted model, and comparing a fitted model against a fitted model on the data both were fitted to proves nothing.

### Decision

`RULES` in `src/defend/baseline.py` is a list of thirteen named predicates over the same authorisation-time feature matrix the model consumes. Nine are conventional card-fraud rules — short-window velocity, high value to a new payee, new device on risky card-not-present spend, impossible travel, large risky spend on a new account, high-risk geography, sub-threshold structuring, an amount z-score spike, and a repeat-disputer rule. Four are network-level: a device shared across cards, a card hopping across devices, a merchant seeing almost only new cards, and a large transfer to a new payee. Because the counters reach the model log-compressed, those four rule thresholds are written as `log1p` of the count a fraud analyst would state, so the rule remains readable in the units a human thinks in.

The thresholds are round, intuitive, domain-chosen numbers and are never fitted. The conjunctions are deliberate: `impossible_travel` requires distance *and* foreignness *and* a short interval since the previous transaction, because distance alone just flags people who travel.

The same rule set does double duty as the reason-code vocabulary — `REASON_LABELS` in `src/defend/decision_policy.py` maps each rule to an analyst-readable phrase.

### Consequences

This buys a comparison worth printing. It also buys explanations for free: every flagged transaction can be annotated with which intuitive rules fired, which is what an analyst actually needs.

The cost is that the baseline may be genuinely competitive on some families, and the generated README says so rather than hiding it.

It rules out the framing "machine learning versus no defense". The claim the artifacts support is narrower: what a learned model adds over good rules given identical information at an identical false-positive budget.

---

## 12. Compare detectors at a matched false-positive budget, with thresholds chosen on validation

### Status

Accepted

### Context

Comparing detectors at whatever operating point each happens to occupy is close to meaningless. Any detector can buy recall by challenging more genuine customers, so a table of recall figures at unmatched false-positive rates measures where each threshold was set as much as it measures detection quality.

Re-thresholding to a common budget fixes that, but introduces a subtler problem: if each detector's matched threshold is chosen using the test set's own labels, every detector has peeked at the answer, and the comparison flatters whichever one happens to have the most exploitable score distribution on that particular slice.

### Decision

`compare()` in `src/defend/baseline.py` reports both views. Each detector appears at its native operating point, and a `_fpr_matched` block re-thresholds every detector — the rules baseline included, via its fraction-of-rules-fired score — to spend the same budget, `config.TARGET_MAX_FPR`.

Those matched thresholds are selected on the validation slice and applied unchanged to the held-out test slice. The artifact records this explicitly in `threshold_selected_on`, and its `note` field states that realised test rates will differ slightly from the budget precisely because the threshold was not fitted to the test set.

The comparison runs on the chronological test split by default, using the same `split_xy` contract as the trainer (record 6), so the static-model column reproduces the committed headline metrics rather than being a second, differently-computed estimate.

### Consequences

This buys the like-for-like comparison a fraud team would run before choosing a detector, and it buys an honest one, because the small gap between the budget and the realised rate is a symptom of the thresholds not having seen the test labels.

The cost is a more complicated artifact — two blocks per detector — and the need to explain the difference every time the table is shown. The generated README carries that explanation.

It also rules out a flattering framing of the adaptive model. This table scores every detector on the *original* attack distribution, which is the static model's home ground; the adaptive model's advantage is robustness to attacks that have moved, which is measured separately in `models/head_to_head.json` against evolved generations. The README says so directly rather than leaving the reader to infer it.

---

## 13. Rehearse un-attacked families in the replay buffer

### Status

Accepted

### Context

The first version of the closed loop replayed only what the red team had just produced: the escaped adversarial examples of the families under attack, plus legitimate context. It is the obvious design, and it failed in a specific and instructive way.

Emphasising the newest, hardest attacks pulls the decision boundary toward them, and families nobody had attacked measurably degraded as a result. The failure was not discovered by inspection — the project's own catastrophic-forgetting gate caught it and refused to promote the candidate, which is the outcome the gate exists to produce.

### Decision

The replay buffer is stratified across three things, assembled in the replay step of `run_loop()` in `src/loop/redteam_loop.py`: the focus families' adversarial examples from this generation, a rehearsal sample of every fraud family that was *not* attacked, and a sample of legitimate context rows.

Rehearsing prior tasks alongside a new one is the standard remedy for catastrophic forgetting in continual learning. Here it is also the honest one: a defense is not permitted to trade away what it already knew in exchange for the newest attack.

Two supporting choices make the rehearsal real. `_focus_mix()` applies a floor to every non-focus family's share of the simulated frame, because boosting the targets without one squeezes untouched families down to a handful of rows and the forgetting check silently reports nothing — a check that reports nothing is worse than no check. And `_bounded_append()` caps the buffer at `LOOP_CONFIG["replay_buffer_max"]` by keeping an equal share per generation, so earlier attack generations stay represented as the loop runs.

`_buffer_composition()` writes the split into `models/loop_history.json` as `evolved_attack_rows`, `rehearsal_rows` and `legit_context_rows`, so the composition is inspectable rather than described.

### Consequences

This buys candidates that can actually clear the forgetting gate, and it buys a bounded, auditable buffer.

The cost is replay capacity spent on families that are not the target, and a slower recovery on the family under attack than a focus-only buffer would give. That is the trade the gate demands.

It rules out an unbounded buffer and it rules out reporting adaptation as a pure gain. The generated README spells out per-family what adaptation bought and what it gave up, because a mean across families would hide both halves.

---

## 14. Make the catastrophic-forgetting gate sampling-error aware

### Status

Accepted

### Context

The forgetting gate compares a previously-learned family's recall under the champion against its recall under the candidate, and blocks promotion when the drop is too large. Family-level recall is a proportion measured on a few dozen transactions. A gate that fires whenever that proportion falls past a fixed line will fire on sampling noise most of the time.

That is not a conservative failure. A gate that blocks a genuinely better model on the strength of a coin flip is worse than no gate: it makes promotion arbitrary, and it teaches whoever reads the artifacts to ignore the gate.

### Decision

`evaluate_candidate()` in `src/defend/governance.py` counts a regression only when the candidate's recall is *confidently* below the tolerated level. The tolerated level is the champion's recall minus `CHAMPION_CHALLENGER["max_prior_recall_drop"]`; the judgement uses the upper bound of the candidate's 95% Wilson score interval, computed by `_wilson()` from the observed recall and the sample size. If that upper bound sits below the tolerated level, the regression is real. If it does not, the drop is inside sampling error at this sample size and is not counted.

Sample sizes are therefore carried all the way to the gate. `prior_for_gate` in `run_loop()` keys each entry by generation and family, or by guard family, and every entry ships with its `n`. Where no sample size is available the gate falls back to the plain comparison rather than assuming significance.

The gate's `detail` string states which family regressed most, on how many observations, and whether the regression counted — so a reader sees the reasoning, not only the verdict.

### Consequences

This buys a gate whose verdicts mean something, and it buys the same discipline elsewhere: the Wilson interval helper in `src/defend/evaluate.py` attaches an interval to every per-family recall in every artifact, and `docs/build_docs.py` prints intervals beside point estimates in every generated table.

The cost is reduced sensitivity. A real regression on a family with very few held-out examples can pass the gate, because at that sample size the evidence genuinely cannot distinguish it from noise. The honest response is more evaluation data, not a tighter line, which is why the guard families are measured on a large dedicated frame rather than on a small test slice.

It rules out a fixed threshold on a noisy proportion anywhere in the governance path.

---

## 15. Choose red-team targets by measured learnability, and exclude structural frontiers from targeting while still reporting them

### Status

Accepted

### Context

Some fraud is not an authorisation-time problem. First-party dispute abuse is a genuine purchase by the genuine cardholder on their own device, disputed later; the defining signal is post-outcome. Victim-authorised scams arrive with every authentication signal clean, because the customer was manipulated into authorising them. Pointing the red team at those families produces a flat recovery curve and spends replay capacity a family with a real residual signal could have used.

Asserting which families those are would be a guess dressed as a design principle. It also risks quietly excluding a family that is merely difficult, which is precisely what the loop exists to attack.

### Decision

Exclusion is measured, not declared. The leave-one-attack-family-out experiment in `src/experiments/leave_one_out.py` trains a model with each family removed and again with it present. The second case is the most favourable condition available: the family is handed to the model directly in training. A family that still cannot clear `LEARNABILITY_FLOOR` under those conditions is limited by authorisation-time observability, not by the decision boundary — and adversarial replay is a slower route to the same thing, more examples of the family, so it cannot beat that ceiling either.

`unlearnable_families()` in `src/loop/redteam_loop.py` reads `models/leave_one_out.json`, applies the floor, and returns both the excluded set and the per-family evidence for excluding it. `select_focus()` then ranks families by the current defense's measured recall via `weakest_families()` and takes the weakest that are neither excluded nor missing a base specification. Nothing about which family gets attacked is decided in advance.

The ordering in `src/pipeline.py` makes this work: the leave-one-out experiment runs at step 5, before the loop at step 6, because it is the evidence the loop reads.

Excluded families are still simulated, still scored, and still reported. `models/loop_history.json` carries a `focus_selection` block containing the exclusion list, the evidence and the floor. A small hardcoded fallback set is used only when the artifact has not been generated yet.

### Consequences

This buys an experiment rather than a script, and it buys a defensible answer to the obvious challenge — "you only attacked the families you could win" — because the exclusions carry their evidence.

The cost is a dependency between pipeline stages and a floor value that is itself a judgement call, stated in one place and reported in the artifact.

It rules out both quietly dropping hard families and pretending they are solvable by more of the same. They are reported as structural frontiers, and the generated README explains that the correct control for them is friction, payee-risk intelligence and post-transaction recall rather than a hard decline.

---

## 16. Retire attack frontiers that do not move across consecutive rounds

### Status

Accepted

### Context

Record 15 excludes families that measurement shows are unlearnable before the loop starts. A second case appears during the loop: a family that was a reasonable target, has been attacked, and is not yielding. Its recall stays below the residual ceiling and adaptation gains nothing round after round.

Continuing to attack it burns replay capacity on examples the model cannot separate — and, as this loop demonstrated, drags down families it could have improved. A real red team moves on and reports the frontier.

### Decision

The reallocation step of `run_loop()` tracks a strike count per focus family. A family earns a strike when its status is `residual_frontier` and its recall gain falls below `LOOP_CONFIG["recovery_delta"]`; any round of genuine movement resets the count. At two consecutive strikes the family is retired.

Its replacement is chosen the same way the original targets were: the weakest family by measured recall against the current candidate, excluding families already attacked in this run, families excluded as structural frontiers, and families without a base specification. Guard families and the frame mix are recomputed afterwards, because a family cannot simultaneously be a target and a control (record 13's `resolve_guards` enforces exactly that).

Retirement is recorded, never silently dropped. Each entry in `retired_frontiers` in `models/loop_history.json` names the family, the round, its final recall, what replaced it, and the reason. The generated README prints the reallocation as part of the loop narrative.

### Consequences

This buys a loop that behaves like an adversary with finite resources rather than one that grinds against a wall, and it converts a negative result into a reported finding.

The cost is a policy constant: two consecutive strikes is a choice, and a family that would have yielded on the third round will be retired. The strike rule is visible in the artifact so a reader can judge it.

It rules out the claim that the loop improves every family it touches. The claim it supports is that the loop reallocates effort toward the families where effort still pays.

---

## 17. Ship both the final candidate and the promoted champion, labelled

### Status

Accepted

### Context

An adaptive defense that deploys itself is not deployable. Every model the loop produces is a challenger measured against the model currently in force, and it ships only if it clears every gate in `config.CHAMPION_CHALLENGER`: a minimum gain on the new attack generation, an absolute false-positive ceiling, a limit on false-positive regression against the champion, no significant forgetting on a previously-learned family (record 14), and no material loss of overall ranking quality.

The consequence is that the last model the loop produces and the last model governance approved are frequently different objects. Shipping only one of them destroys information in one of two ways. Shipping only the final candidate presents a rejected model as the working defense. Shipping only the champion hides what adaptation actually achieved.

### Decision

Three artifacts are written, with distinct meanings declared in `config.py`:

| Artifact | Meaning |
|---|---|
| `models/loop_base_model.joblib` | the stale defense in force at round 0, kept for the before-and-after demonstration |
| `models/loop_adapted_model.joblib` | the final candidate the loop produced — what adaptation achieved |
| `models/loop_champion_model.joblib` | the last candidate that cleared every promotion gate — what governance would let into the authorisation path |

Downstream consumers respect the distinction. `main()` in `src/defend/baseline.py` and step 7 of `src/pipeline.py` load the champion as `adaptive_ml` and the final candidate as `adaptive_candidate_unpromoted`, and `docs/build_docs.py` labels the latter "Adaptive candidate (not promoted)" in every table and figure. `test_the_shipped_adaptive_column_is_the_promoted_model` in `tests/test_artifact_consistency.py` asserts that a promoted champion exists whenever an adaptive column is reported.

The `promotion` block in `models/loop_history.json` records which rounds were promoted and whether the final candidate was, with a note explaining that a candidate which improves attack recall but breaches the false-positive ceiling is reported, not deployed.

### Consequences

This buys a claim the artifacts can contradict, which is the only kind worth making. If no candidate clears every gate, the generated README says so in those words rather than substituting a better-looking result.

The cost is three model files instead of one and a naming discipline every consumer has to observe.

It rules out presenting adaptation as automatic improvement, and it rules out cosmetic governance — the gates change which file ships, not only what the log says.

---

## 18. Generate the README, the walkthrough and the guides from committed artifacts

### Status

Accepted

### Context

Hand-maintained numbers in documentation are the most reliable way to lose a technical reviewer's confidence. The failure is not that one figure is wrong; it is that the moment one table disagrees with another, every other number in the submission becomes suspect, including the correct ones. With a README, a Word walkthrough, a pitch script, a demonstration script and a judge question-and-answer document all quoting the same pipeline, hand-maintenance guarantees drift.

### Decision

No figure in the generated documents is typed by a human. `docs/build_docs.py` loads every committed artifact through `load_all()`, renders the figures under `docs/figures/`, and writes `README.md`; it then calls `docs/build_guides.py` for the presentation and question-and-answer documents and `docs/build_walkthrough.py` for `docs/solution_walkthrough.docx`. Regenerating the pipeline and re-running `python -m docs.build_docs` regenerates all of them from the same source.

Consistency is then tested rather than trusted. `tests/test_artifact_consistency.py` checks that the README quotes the committed recall and false-positive rate verbatim, that it agrees with the threat catalogue on attack counts, that every artifact carries the current `SCHEMA_VERSION` so stale files are detectable, that the fidelity report has no failing checks, that per-family claims carry their sample sizes, and that the documents contain no unqualified claim of validation against real or production payment data.

### Consequences

This buys documents that cannot go stale and a submission whose internal agreement is enforced by the test suite.

The cost is that prose lives in Python string literals, which is a poor editing experience, and that a document requiring genuine narrative — this one — has to be maintained by hand.

That is precisely why this log carries no measured value. A hand-written document that quotes a generated number is a drift hazard by construction; a hand-written document that explains reasoning and points at the artifact for the number cannot go stale. Where a decision needs a figure, it names the artifact.

---

## 19. Enforce a sample-size floor on any headline result, through one shared selection function

### Status

Accepted

### Context

The headline "unseen to learned" result is chosen automatically as the family with the largest measured gain in the leave-one-attack-family-out experiment. The rule for choosing it was implemented three times: once in the README generator, once on the landing page, and once on the hero-demonstration page.

Each of the three copies ended with a silent fallback that re-admitted the full family set when nothing cleared the sample-size floor. The effect would have been to headline a point estimate measured on sixteen held-out transactions — an interval far too wide to carry a headline, however good the point estimate looked — and, because all three copies shared the same defect, all three surfaces would have flipped to it together and invisibly.

Duplication was the real bug. The floor was correct in all three places and the fallback defeated it in all three places.

### Decision

The rule lives in exactly one function: `select_hero_family()` in `src/experiments/leave_one_out.py`. It returns a triple — the family, its record, and an `underpowered` flag that is true when no family cleared `config.FAMILY_EVAL["min_n_to_report"]` and the choice fell back to the full set. The caller must then refuse to headline the number. `hero_family()` in `docs/build_docs.py` does exactly that, returning nothing at all rather than an underpowered result, and every surface calls through it.

The floor is enforced from both directions. Families below it still appear in the generated tables, marked, with their sample size and their 95% interval, so nothing is hidden; they are simply never used as the headline. `wilson_interval()` in `src/defend/evaluate.py` attaches an interval to every per-family recall, and `_per_attack_recall()` stamps each family with a `sufficient_n` flag. `config.FAMILY_EVAL` also defines a larger, fraud-enriched evaluation frame generated from an unseen seed, so the rarer families have enough held-out examples for their numbers to mean anything in the first place.

Two tests guard it: `test_the_hero_family_clears_the_sample_size_floor` asserts the selection never falls back to an underpowered family, and `test_every_quoted_recall_carries_its_sample_size` checks the generated documents never quote the headline recall without its denominator.

### Consequences

This buys a headline that survives the obvious question, and it buys a general principle the project applies everywhere: a proportion is reported with its sample size and its interval, or it is not reported.

The cost is that in a run where no family clears the floor there is no headline result at all. That is the intended behaviour.

It rules out duplicated selection logic. The lesson recorded in the function's own docstring is that a rule implemented three times has three chances to be quietly wrong.

---

## 20. Treat the text arm as a trivially separable sanity check, never as evidence

### Status

Accepted

### Context

The project includes a small text classifier over the synthetic scam-artifact corpus, to show where content signals attach to the architecture. Its score is very high, and for an uninteresting reason: the corpus is composed from a fixed slot vocabulary, so the two classes separate on vocabulary alone. The number measures the corpus, not the detector, and would not survive contact with real scam messages.

A high score sitting beside the genuine detection metrics would be read as a peer of them. It is not, and no amount of small print in one place prevents it being screenshotted somewhere else.

### Decision

The caveat travels with the number, everywhere, mechanically.

`src/defend/text_model.py` defines a single `CAVEAT` string used by the artifact, the command-line output, the application and the documents, so the score cannot appear anywhere without it. The metrics file carries `is_sanity_check_not_evidence`, a `trivially_separable` flag computed from the score itself, and an `honest_reading` field that leads with the separability fact — describing the hard-negative construction first would read as "we made it hard and still scored well", which is the opposite of what the number means. The command-line entry point prints the honest reading unconditionally.

In `src/pipeline.py` the block is keyed `text_sanity_check_not_detection_evidence` rather than `text`, so it cannot be quoted bare, and `test_the_text_score_never_appears_without_its_caveat` asserts that the plain key is absent and the caveat present.

The corpus construction still avoids the crudest shortcuts, because they would make even the sanity check meaningless: provenance markers are stripped by `strip_markers()` before training, the classifier sees only the message body and never the subject line carrying the attack's catalogue name, only attacks that actually reach a victim as a message are included so the classes are not different genres, and the benign class deliberately includes genuine security notices sharing the urgency, one-time-password and payment vocabulary of scam messages. Those measures remove shortcuts; they do not make the task realistic, and the caveat says so.

### Consequences

This buys an architecture that has a place for content signals without a number that contaminates the detection claims. Every detection claim in the project rests on the transaction model alone, and the generated README lists the text arm under limitations.

The cost is an artifact that exists mainly to be disclaimed.

It rules out ever quoting the text score as a result, which is exactly the point.

---

## 21. Refuse to write a GenAI demonstration artifact when no API key is available

### Status

Accepted

### Context

Every committed artifact in the repository was produced by the deterministic offline path, because the demonstration has to run with no key and no network, and because committed figures have to reproduce from a single seed. That is the right default, and it leaves a gap: nothing in the committed evidence shows the language-model half of the design working end to end.

`src/generate/demo_specs.py` exists to close that gap. The tempting convenience is to have it fall back to the offline heuristic when no key is present, so the script always produces something. That would be worse than producing nothing, because the resulting file would be a demonstration of the GenAI path generated by the very thing it is meant to demonstrate.

### Decision

The script refuses. `run()` raises `LLMUnavailable` when `config.llm_available()` is false, with a message stating why, and writes nothing. `main()` prints a readiness report, explains that nothing was written, and exits with a non-zero status. A `--check` flag reports readiness without calling the model at all.

When a key is present, the script takes five synthetic measured-weakness seeds, sends each through the real red-team agent, puts each returned proposal through the same payment-domain constraint layer the loop uses, and writes `models/genai_spec_demo.json` including the exact prompt sent, the raw proposal, the validated specification, and what the constraint layer corrected. It touches no dataset, no model, no committed metric.

Provenance is recorded in the data, not only in prose. Every specification carries a `source` field, so heuristic and model-authored generations are distinguishable in `models/attack_lineage.json`; the generated README names that field and explains the two paths.

What can be verified without a key is verified anyway. `tests/test_genai_spec_path.py` substitutes a stub client for the network call and checks both sides of the boundary: that the prompt is constructed from the measured weakness, that a model response is parsed, that the constraint layer validates it, and that the result is a specification the simulator can execute. The stub is a test double and never writes an artifact.

### Consequences

This buys a repository whose provenance claims are all true. Nothing in it asserts a live model call that did not happen.

The cost is that a fresh clone without a key ships without `models/genai_spec_demo.json`, so a reviewer sees the live path demonstrated by tests rather than by an artifact.

It rules out the most convenient possible presentation — a committed file that looks like model output and is not — and, more generally, it rules out any fallback that changes what an artifact *is* rather than only how good it is. Elsewhere the fallback is legitimate precisely because it does not: when the model is unavailable the loop's weakness-driven heuristic reads the same measured evidence and moves the same dials, so the behaviour degrades in quality, never in reproducibility or in honesty about its source.

---

## Where to go next

Start with [`README.md`](../README.md) for the results these decisions produced — every figure in it is generated from `models/*.json` and none of it is typed by hand. Read [`JUDGE_QA.md`](JUDGE_QA.md) for the short-form answers to the questions this log addresses at length, and [`DEMO_GUIDE.md`](DEMO_GUIDE.md) for how the prototype is meant to be driven.

To see any decision in the data rather than in prose, the fastest paths are `models/attack_lineage.json` for records 1, 2 and 16, `models/loop_history.json` for records 13 to 17, `models/model_registry.json` for the governance verdicts, `models/leave_one_out.json` for record 15, and `models/fidelity_report.json` for records 3 to 5. Running `python -m pytest tests/ -q` exercises the guards described in records 3, 4, 5, 6, 9, 17, 18, 19, 20 and 21.
