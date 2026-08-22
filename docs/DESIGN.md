# Design Rationale

*This document is for a reviewing engineer, a judge, or a new contributor who wants to understand why the AI Defense Lab is built the way it is rather than how to run it. After reading it you will know the problem the system is shaped around, why generative creativity is deliberately separated from transaction generation, how each of the six core abstractions earns its place, why simulator fidelity is treated as a correctness concern rather than a presentation concern, and what the system deliberately refuses to attempt. It contains no measured results by design — every number lives in [`README.md`](../README.md) and in `models/*.json`, which are generated from artifacts, so this document cannot go stale or contradict them.*

---

## 1. The central problem

A supervised fraud model can only learn from labelled history. That is not a defect of any particular model; it is what supervision means. The labels arrive from chargebacks, disputes and confirmed fraud reports, which means they arrive weeks after the behaviour they describe, and they only describe behaviour that someone already recognised as fraud. Everything the model knows is therefore a description of an attack that has already been run, noticed, and settled.

Generative AI changed the economics on the other side of that asymmetry. Producing a new attack variant — a differently-worded lure, a differently-shaped spend pattern, a payment sequence that keeps every individual signal inside a plausible range — used to require a person with domain knowledge and time. It is now close to free, and it can be done at the rate at which defences are deployed rather than at the rate at which fraud rings learn. The defender's learning loop is bounded by dispute latency; the attacker's is not.

The design goal that follows is narrow and specific: **the defence must be able to discover its own weaknesses before an attacker does, and must be able to convert each discovery into training signal without waiting for a label.** Everything in this repository exists to serve that sentence. The Threat Atlas exists so the search space is a catalogued one rather than an improvised one. The constrained simulator exists so a discovered weakness can be turned into transactions. The closed loop exists so the discovery is driven by measurement rather than by a guess about what fraud will do next. The governance layer exists because a defence that retrains itself and deploys itself is not a defence anyone would allow into an authorization path.

The corollary is equally important: because no real fraud labels are available to this project, everything here is a laboratory result on synthetic data. That constraint is stated in the artifacts themselves, not just in the prose — see [Section 7](#7-honesty-as-a-design-constraint).

---

## 2. The core principle: generative creativity, constrained by domain rules

The single most consequential design decision in this repository is that **the language model never writes a transaction row.** It writes a specification. A payment-domain constraint layer validates that specification. A deterministic, seeded simulator executes it.

```
   Threat intelligence  +  measured model weakness
                        |
                        v
        +-------------------------------+
        |  GenAI red-team agent         |   creative
        |  src/generate/llm_agent.py    |   may propose anything
        +-------------------------------+
                        |
                 AttackSpec (JSON)
                        |
                        v
        +-------------------------------+
        |  validate_spec()              |   payment-domain rules
        |  src/generate/attack_spec.py  |   clamp, correct, or reject
        +-------------------------------+
                        |
              validated AttackSpec
                        |
                        v
        +-------------------------------+
        |  Constrained simulator        |   deterministic given the seed
        |  src/generate/*.py            |
        +-------------------------------+
                        |
                        v
           Synthetic payment transactions
```

This split is described in the module docstring of `src/generate/attack_spec.py` and implemented by `validate_spec()` in the same file.

**What the generative half buys.** The interesting question in adversarial work is not "can we make this attack quieter" — a scalar stealth dial answers that, and answers it boringly. The interesting question is *which* dimension of behaviour to change given what the defence has been measured to depend on, and that is a reasoning problem over payment semantics. A language model given the measured signal attribution for a family can reason that if the model's ranking of account takeover rests on `device_changed`, the attack that matters next is a session-hijack takeover from the victim's own device — and it can say so in one sentence that a human reviewer can argue with. The prompt in `_spec_prompt()` hands the model the family's current dial settings, its measured recall, how many of its transactions escaped, the signals the ranking depends on, and how escaped transactions differ from caught ones. It asks for the smallest change that removes the strongest dependency.

**What the deterministic half buys.** Everything downstream of the specification is reproducible, auditable and safe. A model that emitted transaction rows directly would be unreproducible (nothing guarantees the same rows twice), unauditable (there is no compact object to diff between generations), and unbounded (nothing stops it emitting a negative amount, a transaction rate no card could physically produce, or an authorized push payment made from an attacker's device, which is a contradiction in terms). Because the specification is a small frozen dataclass, `spec_diff()` and `spec_distance()` can express "how did the attack evolve" as a structured, inspectable object rather than as a claim, and `models/attack_lineage.json` records the whole chain.

**Why the pipeline stays reproducible either way.** `propose_attack_spec()` in `src/generate/llm_agent.py` computes the offline `heuristic_mutation()` proposal *before* it attempts the live call, and falls back to it on any of `LLMUnavailable`, `ValueError`, `TypeError`, `KeyError` or `SpecRejected`. Both paths produce an `AttackSpec` in the same vocabulary, both go through the same `validate_spec()`, and both are executed by the same seeded simulator. Live calls are additionally cached to `artifacts_cache/` keyed by a hash of model, system prompt, user prompt and schema tag (`_cache_key()` in `src/llm/client.py`), so a run that had an API key once can be replayed without one. The `source` field on every `AttackSpec` records which path produced it, and it is carried into the loop history and the model registry. The quality of the proposal degrades without a key; the reproducibility of the pipeline does not.

---

## 3. The key abstractions

### 3.1 AttackSpec — behavioural dials with a fixed vocabulary

**The problem.** An attack generation has to be three things at once: expressive enough that "evolve the attack" means something, small enough that a human can read the diff between two generations, and safe enough that executing it cannot produce something impossible or harmful.

**How it works.** `AttackSpec` (frozen dataclass, `src/generate/attack_spec.py`) carries six categorical behavioural dials — `amount_profile`, `velocity_profile`, `device_behavior`, `geo_behavior`, `merchant_behavior`, `timing_profile` — plus a continuous `intensity`, and provenance fields (`targets_signal`, `rationale`, `confidence`, `source`, `generation`). The permitted value list for every dial lives in `config.ATTACK_SPEC_BOUNDS`, which is also what the red-team prompt shows the model as `_SPEC_VOCAB`. Derived properties translate each categorical value into the concrete numbers the injectors consume: `AMOUNT_SCALE`, `VELOCITY_TXNS`, `VELOCITY_WINDOW_H`, `DEVICE_TRUST` and `GEO_KM` are the lookup tables, and `amount_scale`, `txns_per_card`, `velocity_window_hours`, `device_trust` and `geo_km_range` are the properties that expose them.

**Why a fixed vocabulary rather than free-form output.** Three reasons, in order of weight.

First, a fixed vocabulary makes the mutation *attributable*. When the specification can only say `device_behavior: trusted_device`, the claim "this generation removed the model's dependence on device change" is checkable against the data, because exactly one thing changed and the change has a name. Free-form output would let a generation drift on five axes at once and make any statement about cause unfalsifiable.

Second, it makes the space enumerable, which is what allows the stealth ordering in [Section 3.4](#34-the-mutation-policy--signal_to_dial-and-_stealth_rank) to exist at all. You cannot rank free text by distance from ordinary customer behaviour.

Third, it bounds the blast radius. The simulator only knows how to execute values in the vocabulary; anything else is replaced by the family baseline before execution. This is also what keeps the red-team agent's output within a defensive remit — the model chooses positions on dials that a sandboxed simulator will move, and never produces operational content, which is stated explicitly in `REDTEAM_SPEC_SYSTEM`.

`BASE_SPECS` holds the generation-0 specification for each of the eleven simulated families. These are the lineage roots: every evolved variant is diffed against its family's entry, and `spec_distance(BASE_SPECS[fam], specs[fam])` is recorded per round as `novelty_distance_from_generation_0`. Several of the baselines carry an in-code comment explaining a deliberately unobvious choice — for example, `account_takeover` is deliberately *not* a foreign-geography attack, because making generation 0 transact from another continent would hand the detector a geography tell and let the family post a recall that has nothing to do with detecting takeover.

### 3.2 The constraint layer — clamping versus rejection

**The problem.** A creative proposer will eventually propose something that cannot happen on a real payment rail. If such a proposal were executed, the resulting dataset would contain fraud that no defender could ever encounter, and every number measured on it would be measuring the simulator's imagination.

**How it works.** `validate_spec(proposed, family, previous=None, strict=False)` runs three passes.

1. **Numeric ranges.** `intensity` is coerced to a float, clamped into `config.ATTACK_SPEC_BOUNDS["intensity"]`, and — when a `previous` specification is supplied — clamped so it can never exceed the previous generation's intensity. The reason is stated in the correction message: *stealth may not regress within an attack lineage*. A persistent adversary does not become louder after being caught.
2. **Categorical vocabularies.** `_coerce_choice()` normalises case and spacing, and replaces anything outside the permitted list with the family's baseline value, recording a correction.
3. **Family-level payment-domain requirements.** `FAMILY_CONSTRAINTS` holds requirements for ten of the eleven families. Each entry maps a specification field to the set of values that family can legally take. If the current value is outside that set, the value is replaced by the family baseline (or, if the baseline itself is not permitted, the lexicographically first permitted value).

The one family with no entry in `FAMILY_CONSTRAINTS` is `account_takeover`, and that is intentional: it is the family the loop is most interested in evolving, and there is no dial position that makes an account takeover stop being an account takeover.

**Clamping versus rejection.** The default is to clamp, because the loop's job is to keep running and produce a legal generation every round; a rejected proposal that halts the round teaches nothing. But `strict=True` switches family-level violations from correction to rejection, raising `SpecRejected` with every violated requirement listed. That mode exists so the system can report "the red team asked for something impossible" as a fact rather than silently absorbing it. Every correction is recorded in a `ValidationReport`, which is serialised into each round of `models/loop_history.json` under `constraint_layer` and into every node of `models/attack_lineage.json`. The constraint layer's work is therefore visible, not implicit.

**A worked example.** A proposal for `scam_transfer` that asks for an attacker device, a foreign geography, an out-of-vocabulary amount profile and an out-of-range intensity is corrected as follows. This is the actual output of `validate_spec()`, not a sketch:

```json
{
  "family": "scam_transfer",
  "accepted": true,
  "corrections": [
    {"field": "intensity", "from": 1.8, "to": 1.0,
     "reason": "outside permitted range"},
    {"field": "amount_profile", "from": "gigantic", "to": "high",
     "reason": "value outside the permitted vocabulary"},
    {"field": "device_behavior", "from": "new_device", "to": "trusted_device",
     "reason": "scam_transfer cannot be executed with device_behavior='new_device' on a real rail"},
    {"field": "geo_behavior", "from": "foreign", "to": "home",
     "reason": "scam_transfer cannot be executed with geo_behavior='foreign' on a real rail"}
  ],
  "rejections": []
}
```

The device and geography corrections are the domain-meaningful ones. A scam transfer is an *authorized* push payment: the genuine customer, manipulated by a social engineer, authenticates on their own device and sends the money themselves. A scam transfer executed from an attacker-controlled device in another country is not a stealthier scam transfer — it is a different attack that happens to share a label. Executing that proposal would produce rows the detector could separate on geography, and the family's reported recall would then be measuring geography detection while claiming to measure scam detection. With `strict=True` the same proposal raises `SpecRejected` naming both violations instead.

### 3.3 WeaknessReport and signal attribution

**The problem.** Without measurement, "attack evolution" is a scripted walk down a stealth dial, and any improvement the defence shows afterwards is an artifact of the script. The loop needs to know what the model *actually depends on* for a specific family, on that model's own decisions, before it decides what to change.

**How it works.** `analyze_family()` in `src/loop/weakness.py` produces a `WeaknessReport` for one family against one model on one evaluation frame. It does four things.

It computes the family's recall at the model's tuned threshold, using the raw fused score. It splits the family's fraudulent rows into escaped and caught. It then runs a **permutation test scoped to that family**: it builds a sub-frame from the family's own fraudulent rows plus a bounded random sample of legitimate rows, measures the model's ability to rank that sub-frame, then shuffles one feature column at a time — repeated several times per column — and records the drop. Every column in `FEATURE_COLUMNS` is tested and the top few by mean drop are retained as `relied_signals`.

The scoping is the point. A global permutation importance would tell you what the model relies on across all fraud, which is dominated by whichever families are most numerous and most separable. Scoping to one family's own rows, against a legitimate background, answers the question the red team actually needs answered: *what is carrying this model's ability to distinguish this family from ordinary traffic?* That is precisely the thing the next generation should remove.

Alongside attribution, `escape_profile` compares the mean of each feature across escaped rows, caught rows and legitimate rows, ranked by relative gap. This is the qualitative half — it tells a reader in what respect the survivors differ from the casualties — and it is fed to the language model in the prompt.

Two boundary conditions are handled explicitly rather than hidden. If nothing escaped, the report says so and the next generation targets the strongest signal instead of the escape profile. If a family is absent from the frame, the report is returned with a note and null recall rather than a fabricated zero.

`weakest_families()` is the companion function that ranks every family by recall on a frame, and `select_focus()` in `src/loop/redteam_loop.py` uses it to choose the round's targets. This is what makes the loop an experiment rather than a demonstration: nothing about which family gets attacked is decided in advance.

### 3.4 The mutation policy — SIGNAL_TO_DIAL and _STEALTH_RANK

**The problem.** Knowing which signal the model leans on does not by itself tell you what to change. The mapping from "the model depends on `merchant_new_card_ratio_prior`" to "route the spend through established merchants instead of a front that only sees new cards" is payment domain knowledge, and it has to come from somewhere.

**How it works.** `SIGNAL_TO_DIAL` in `src/loop/weakness.py` is that mapping, written by hand: twenty detector signals, each mapped to a triple of (specification field, new value, human-readable explanation). It is deliberately *not* a fitted mapping. It is the red team's playbook — the same knowledge a human fraud strategist would apply — and it is applied only to the signal the model was actually measured to depend on. It is also the offline fallback: `heuristic_mutation()` walks the report's relied signals in order, takes the first one with an entry, and proposes that single dial move.

**Why `_STEALTH_RANK` exists.** A naive application of `SIGNAL_TO_DIAL` can propose a change that makes the attack *louder*. Consider a family whose top signal is `device_card_count_prior` — one device carrying many cards, the classic mule-ring or botnet counter. The mapping's entry says "stop sharing one device across the ring so the device-to-card counter stays low", which points at `new_device`. If the attack was already running from a trusted device, moving it to a fresh device would remove the fan-in signal and immediately hand the defence `device_changed`, which is a stronger and cheaper tell. The proposal would be an evasion in name and a gift in practice.

`_STEALTH_RANK` prevents that. It assigns every value of every categorical dial an integer distance from ordinary customer behaviour: `trusted_device` is nearest and `new_device` furthest on the device axis; `home` is nearest and `high_risk` furthest on geography; `single` is nearest and `burst` furthest on velocity; and so on across six fields. `_is_stealthier()` accepts a proposed move only when the proposed value ranks strictly lower than the current one. **A proposal must move the attack toward ordinary behaviour, never away.** That is the invariant, and it is what makes the resulting sequence a genuine evasion lineage rather than a random walk.

The amount axis is ranked differently from the others, and the reason is written into the code comment: `moderate` sits at rank 0, with `low` and `high` tied above it and `micro` and `extreme` tied above those. Distance from ordinary behaviour on amount is two-sided, because an unusually large purchase and a micro-probe are both deviations from a normal ticket.

When every dial the top signals point at is already at its quietest permitted value, `heuristic_mutation()` does not invent a new tactic the family would not plausibly adopt. It falls through a fixed ladder — shrink the residual amount deviation via `_AMOUNT_DOWN`, then slow the campaign via `_VELOCITY_DOWN`, then match the cardholder's ordinary rhythm by setting `timing_profile` to `customer_normal` — and labels the target as a residual signal rather than a named feature. The intensity multiplier applied each generation is larger when the defence is still catching most of the family and smaller when it is not, so evolution slows as the attack approaches the legitimate centroid.

### 3.5 The cumulative replay buffer

**The problem.** Retraining on the newest, hardest attack generation alone pulls the decision boundary toward it and degrades families the red team never touched. This is catastrophic forgetting, and it is not hypothetical here — the comment at the replay step in `src/loop/redteam_loop.py` records that this loop actually produced that failure, and that the forgetting gate correctly refused to promote the result.

**How it works.** The buffer holds *feature rows*, not raw transactions, because each round is simulated as a full frame so that per-card history and network counters are computed correctly over an ordered frame before anything is extracted. Three things enter the buffer each round, assembled in step 6 of `run_loop()`:

- the round's focus adversarial examples (fraudulent rows of the families under attack),
- a **rehearsal** sample of the fraudulent rows of every family that was *not* attacked,
- a matched quantity of legitimate context rows.

`_bounded_append()` then enforces the cap in `config.LOOP_CONFIG["replay_buffer_max"]` by **stratified** downsampling: it computes an equal per-generation share and samples each round's rows down to that share, so earlier attack generations stay represented as the buffer fills rather than being evicted by recency. `_buffer_composition()` records exactly what the buffer contains — total rows, fraud rows, evolved-attack rows, rehearsal rows, legitimate context rows, and breakdowns by family and by generation — and that composition is written into both the round history and the model registry entry.

**Why it rehearses families that were not attacked.** Rehearsing prior tasks alongside a new one is the standard remedy for catastrophic forgetting in continual learning, and here it is also the honest one. The system's claim is that adaptation improves the defence against evolved attacks. If that improvement is purchased by quietly losing families the loop never touched, the claim is false, and the buffer is the mechanism that makes it true rather than the gate that merely detects it being false. The gate still runs — see [Section 3.6](#36-championchallenger-governance) — but a design that relies only on a gate to catch a foreseeable failure is a design that expects to fail.

Two related details are worth noting because they are easy to get wrong. Replayed rows are emphasised with a **sample weight** (`config.LOOP_CONFIG["replay_oversample"]`, passed through `DefenseModel.fit(sample_weight=...)`) rather than by duplicating rows, which has the same effect on the decision boundary without inflating the training frame or hiding the strength of the emphasis behind a concatenation. And `_focus_mix()` applies a **floor** when it up-weights the focus families, because boosting the targets without one squeezes the untouched families down to a handful of rows and the forgetting check — whose entire job is to watch those families — then silently reports nothing.

### 3.6 Champion/challenger governance

**The problem.** An adaptive defence that deploys itself is not deployable. Every retrained model in this system is a candidate, and the question "did adaptation work" has to be answerable with "no".

**How it works.** `evaluate_candidate()` in `src/defend/governance.py` runs five gates and returns the decision together with the observed value, the required value and a human-readable detail for each:

| Gate | What it requires |
|---|---|
| `attack_recall_gain` | The candidate beats the model in force on the new attack generation by at least `CHAMPION_CHALLENGER["min_attack_recall_gain"]` |
| `absolute_false_positive_ceiling` | The candidate's legitimate false-positive rate stays at or below `CHAMPION_CHALLENGER["max_fpr"]` |
| `false_positive_regression` | The candidate does not worsen the champion's false-positive rate by more than `CHAMPION_CHALLENGER["max_fpr_regression"]` |
| `no_catastrophic_forgetting` | No previously-learned family regresses significantly (see below) |
| `overall_ranking_quality` | Overall PR-AUC does not fall more than `CHAMPION_CHALLENGER["max_overall_pr_auc_drop"]` below the champion's |

A candidate that fails any gate is recorded in `models/model_registry.json` with the failed gate names and is **not** promoted. The loop persists two distinct model artifacts for exactly this reason: `loop_adapted_model.joblib` is the final candidate the loop produced — what adaptation achieved — and `loop_champion_model.joblib` is the last candidate that cleared every gate — what governance would actually let into the authorization path. When no candidate clears the gates these are different models, and the difference is the point.

**Why the forgetting check is sampling-error aware.** Per-family recall in the loop's evaluation frames is a proportion measured on a few dozen fraudulent transactions. A gate that fires whenever the observed drop crosses a fixed line will fire on sampling noise, and a gate that blocks a genuinely better model on a coin flip is worse than no gate at all — it makes the governance layer a source of false confidence in both directions.

So the check computes a 95% Wilson score interval on the candidate's family recall using the number of observations behind it (`_wilson()`), and counts a regression only when the **upper bound** of that interval falls below the tolerated level (`champion recall − CHAMPION_CHALLENGER["max_prior_recall_drop"]`). In plain terms: the candidate must be *confidently* worse, not merely observed worse. The Wilson interval is used rather than a normal approximation because it behaves correctly at small n and at proportions near zero or one, which is exactly the regime these family counts sit in. When no observation count is available the check falls back to the raw comparison, and the gate detail records which families were compared, the largest observed regression, its sample size, and — when the drop was not counted — that it sat inside sampling error at that sample size.

The families that are watched are supplied by two independent sources in `run_loop()`: prior *generations* of the attacked families, replayed from the stored evaluation frames of earlier rounds, and **guard families** that were never attacked at all, resolved at run time by `resolve_guards()` from `GUARD_CANDIDATES` minus whatever the red team actually chose. A family cannot be both a target and a control.

---

## 4. Simulator fidelity as a first-class design concern

**Shortcut learning** is what happens when a model achieves a good score by keying on an incidental property of how the data was produced rather than on the phenomenon the data is supposed to represent. The classic symptom is a metric that looks excellent and generalises to nothing.

**A fraud simulator is unusually prone to it** because of the structure of the generation process. Legitimate rows and fraudulent rows come out of different code paths. Anything the fraud path does uniformly — always picking a merchant at random, always using a fresh card, always transacting at night, always forcing one channel, always starting from a fresh account — becomes a perfect or near-perfect label, and the detector will find it immediately. The failure is silent: nothing crashes, the numbers just become meaningless. And because the same simulator generates both the training data and the evaluation data, there is no held-out set that can reveal the problem.

The defence against this is not a check bolted on afterwards. It is a set of generation-time design choices, each of which exists to destroy a specific shortcut.

**Fraud-actor cover traffic.** Fraud rings do not spring into existence at the moment of the fraudulent authorization. Bust-out accounts groom for weeks, mule accounts carry ordinary spend, front merchants process genuine-looking sales. `_cover_row()` in `src/generate/attack_injectors.py` emits those rows with `is_fraud=0` and `actor_role=ROLE_COVER`, because at authorization time they genuinely are not fraud and labelling them as fraud would hand the model a label it could not possibly earn in production. The volumes are configured per family in `config.FRAUD_REALISM` (`bustout_cover_txns`, `bustout_ramp_days`, `mule_cover_txns`, `laundering_cover_txns`). Without this, "this card has no history" would be a synonym for "fraud".

**Entity reuse.** If every fraudulent row were a first-ever card/merchant pair, the defender would learn that artifact rather than fraud behaviour. So probing hits the same card repeatedly across a small repeated merchant pool (`testing_probes_per_card`, `testing_cover_merchants`); compromised victim cards are hit more than once and sometimes at the same merchant (`victim_repeat_prob`, `victim_repeat_extra`, `victim_same_merchant_prob`); mimicry shops at the victim's own regular merchant; laundering pushes many real compromised cards — a majority drawn from the genuine cardholder population — through one front merchant.

**Card issuance and attrition.** A minority of genuine cardholders activate mid-window (`late_activation_fraction`, materialised as `Cardholder.activation_day`), so a first-ever authorization is not a fraud signal. Crucially, a matching minority *stop* being used mid-window (`lapse_fraction`, `Cardholder.lapse_day`), and the reason is written into the config comment: without attrition, cards that only start mid-window push genuine volume steadily later while fraud stays evenly spread, the fraud rate drifts down across the window, and a chronological train/test split inherits a different base rate from the training slice — a property of the generator, not of fraud.

**Popularity-weighted merchant selection.** Genuine spend follows a heavy-tailed merchant popularity curve (`Merchant.popularity`, drawn log-normally in `generate_merchants()`). If the attack injectors sampled merchants uniformly, "quiet merchant" would become a fraud signal. `_weighted_pick()` therefore samples attack merchants in proportion to popularity, exactly as the legitimate generator does.

**Timing profiles.** Fraud rides the same weekly and payday traffic shape as genuine spend: `_DAY_P` in the injectors is built from the same `profiles.day_weight()` the legitimate generator uses, so "transacted on an ordinary Tuesday" cannot become a tell. Within the day, `_hour_for()` derives the hour from the specification's `timing_profile`, and the `customer_normal` profile draws from the same three-peak diurnal mixture the legitimate generator's `_diurnal_hour()` uses.

The same reasoning appears throughout the injectors in smaller places, each with its rationale in a comment: `_account_age()` computes account age on the same clock the legitimate generator uses; `_channel_for()` draws from the merchant's own channel mix rather than forcing every fraudulent row onto one rail; `_geo_for()` picks a city whose *true* distance from home falls inside the band the specification asked for, so a specification that says "stay plausible" cannot emit a transaction thousands of kilometres from home; and `strip_markers()` in `llm_agent.py` removes the provenance tags from generated text before it reaches the text classifier, because a model that learns "contains the word SYNTHETIC" would score perfectly and detect nothing.

**How `src/generate/fidelity.py` turns this into a standing check.** Design intentions decay. The module converts them into measurements that run as part of the pipeline and are committed as `models/fidelity_report.json`.

`separability()` computes, for every one of the thirty-six feature columns, its univariate ranking power against the fraud label and its marginal histogram overlap between the two classes. `quality_checks()` then emits explicit `PASS` / `WARN` / `FAIL` checks, each named after the property it protects. The headline check is `no_single_feature_separates_classes`, which fails when the strongest single feature exceeds `SEPARABILITY_FAIL`, with a warning tier at `SEPARABILITY_WARN` — a single feature that ranks fraud that well is a shortcut, not a signal, because at that point the model is reading the generator rather than the fraud. The remaining checks are the direct converse of each design choice above: `legit_device_changes_exist`, `legit_new_payees_exist`, `fraud_sometimes_uses_a_known_device`, `fraud_sometimes_reuses_a_merchant`, `fraud_amounts_overlap_legit_range`, `no_history_is_not_a_fraud_synonym`, `fraud_timing_is_not_uniformly_nocturnal`, `family_sample_sizes_support_their_claims`, and `fraud_actors_have_cover_traffic`.

The module's `scope` field states in the artifact itself that these are diagnostics of the simulator against itself, that they say nothing about similarity to any real payment portfolio, and that no comparison to production data of any kind has been performed. The claim "our synthetic data is not trivially separable" is therefore a measurement in a committed artifact rather than an assertion in a slide. The results are summarised in the [Is the synthetic data worth training on?](../README.md#is-the-synthetic-data-worth-training-on) section of the README.

---

## 5. The feature contract

**The problem.** A fraud dataset contains three kinds of column, and mixing them produces a model that scores beautifully and cannot be deployed.

`src/defend/features.py` makes the distinction explicit and enforces it:

| Class | Constant | Contents | Available at authorization time? |
|---|---|---|---|
| Authorization-time features | `FEATURE_COLUMNS` (36 columns) | Per-transaction attributes, per-card historical context, per-network relational counters | Yes — this is the model's entire input |
| Post-outcome | `POST_OUTCOME_COLUMNS` (2 columns: `refund_flag`, `auth_result`) | Known only after the transaction settles | No |
| Oracle | `ORACLE_COLUMNS` (3 columns: `is_fraud`, `attack_type`, `actor_role`) | Simulator bookkeeping about who produced the row | Never, in any system |

**Why the split is enforced in code rather than by convention.** `assert_auth_time_safe()` raises a `ValueError` naming the offending columns, and `build_features()` calls it on its return value before handing the matrix back. Convention fails here for a specific reason: the leaking columns are the most predictive ones in the table. `refund_flag` is the *defining* signal of friendly fraud — the injector sets it to `True` on every row — and `auth_result` carries the decline ratio that makes card testing obvious to an acquirer. A contributor adding a feature under time pressure, looking at a dataframe that contains both, will produce a model with an extraordinary score and no deployability, and nothing in a code review reliably catches it. A hard failure at the point of construction does.

The contract is not a blanket ban on outcome data, and the nuance matters. `card_prior_dispute_rate` is built from `refund_flag`, but only over the card's *earlier* transactions, via `expanding().mean().shift()`. Past outcomes known before time T are legitimately available at T, and this gives friendly fraud and refund abuse a genuine repeat-disputer signal. The same temporal discipline governs everything historical: `expanding().shift()`, trailing rolling windows and running counters throughout, so no feature can see its own row or any row after it.

**Why network-level counters matter.** Six of the thirty-six columns are relational counters listed in `NETWORK_FEATURES`: `device_card_count_prior`, `card_device_count_prior`, `ip_card_count_prior`, `merchant_card_fanin_prior`, `merchant_new_card_ratio_prior` and `merchant_txn_count_prior`. These are the signals a **payment network** can compute and a single issuer or a single merchant cannot, because computing them requires seeing traffic across the whole ecosystem rather than across one portfolio.

That distinction is the difference between two very different statements. An issuer looking at a card-testing probe sees "this card is unknown to us". A network maintaining a device-to-card counter sees "this device has carried eighty cards this hour", which is the actual pattern. The same holds for mule rings, which are visible as payee fan-in, and for transaction laundering, which is visible as a merchant whose prior traffic is almost entirely first-time cards — the purpose of `merchant_new_card_ratio_prior`, since a normal merchant converts repeat customers and a front merchant does not.

They are implemented as **running counters over first-time pairings** (`_prior_distinct()`), not as graph queries, because that is how a network-side counter service would actually maintain them and it keeps the lookup O(1) at authorization time. The count features are log-compressed before they reach the model (`_COUNT_FEATURES`), because they grow without bound as the simulation window fills; on a raw scale a model tuned on earlier traffic would see a systematically different distribution later, and the threshold would drift off its false-positive budget for reasons that have nothing to do with fraud.

One further merchant-relative feature deserves mention for the same reason. `merchant_amount_zscore` measures whether a ticket is unusual *for this merchant*, because a five-figure charge is ordinary at a travel agent and extraordinary at a grocer, and an absolute amount threshold punishes the wrong customers. Missing values are filled with honest defaults rather than convenient ones: a merchant's first-ever transaction gets `0.5` for its new-card ratio, which is the no-information midpoint rather than a value implying safety, and a card's first transaction gets `0.0` for its amount-versus-prior-maximum, meaning "no evidence this is unusual" rather than an implied alarm.

---

## 6. Scoring and decisioning

`DefenseModel` in `src/defend/model.py` fuses two heads: a `HistGradientBoostingClassifier` trained on labelled data, and an `IsolationForest` fitted on legitimate rows only. The anomaly head is fitted on legitimate traffic specifically so it retains some ability to flag out-of-distribution transactions from families the supervised head never saw, which is the property the "novel attacks" framing requires. `fusion_weight` sets the blend.

**Why detection thresholds the raw fused score.** `fused_scores()` returns the uncalibrated blend, and `predict()`, `tune_threshold()` and every recall computation in the loop compare against it. The reason is mechanical: isotonic calibration produces a **step function**, so many genuine transactions end up sharing an identical calibrated score. Thresholding a step function cannot hit a false-positive budget precisely — the first tied block the comparison admits overshoots it, potentially by a large multiple. The raw fusion is continuous and fine-grained, which is what an operating threshold needs. Because isotonic regression is monotone, calibration cannot change the model's ranking at all: ROC-AUC, PR-AUC and any quantile-selected threshold behave identically either way, so nothing is lost by thresholding the raw score.

`tune_threshold()` compounds the same reasoning. It evaluates the *realized* false-positive rate at each distinct legitimate score and takes the lowest threshold whose realized rate is within `config.TARGET_MAX_FPR`, rather than taking a quantile. A quantile can land inside a tied block, and the `>=` comparison then admits the whole block and quietly spends several times the agreed budget. Searching realized rates cannot do that.

**Why the displayed number is calibrated.** `risk_probability()` applies the isotonic calibrator fitted by `fit_calibration()` on a held-out slice. This is what a reviewer sees and what the decision policy tiers on, so the number has to mean something: a displayed score of 0.30 should mean that roughly three in ten transactions scoring there were fraudulent. Without calibration the fused blend is on no meaningful scale at all — it is an arbitrary weighted average of a classifier probability and a normalised anomaly score. The two methods therefore serve two different consumers: the raw score serves the threshold, the calibrated score serves the human.

**Why the step-up tier is derived rather than constant.** `config.DECISION_THRESHOLDS` sets `step_up` to `None`, and `resolve_thresholds()` in `src/defend/decision_policy.py` fills it from the model's own `threshold_probability` — the tuned operating threshold expressed as a fraud probability. The alternative would be a second hand-chosen constant, which would have to be justified separately, would drift out of agreement with the tuned threshold every time the model was retrained, and would silently change how much friction the system imposes without anyone deciding to change it. Tying the tier to the point already chosen against the false-positive budget means there is exactly one operating decision in the system and it is made in one place. Only the `decline` tier is an independent constant, and it should be: refusing a payment outright is a different business decision from challenging it, and it deserves its own threshold.

The tiering itself is the honest answer to modest precision at a realistic base rate. `decide()` maps the calibrated probability to `APPROVE`, `STEP_UP` or `DECLINE`, and `reason_codes()` attaches short human-readable explanations drawn from which rule-style signals fired. For families like scam transfer, where the genuine customer authenticated and authorized, friction is the only correct control — a hard decline would be refusing a payment the customer explicitly made.

---

## 7. Honesty as a design constraint

The most fragile part of any competition submission is the gap between what the artifacts show and what the prose claims. Rather than manage that gap by discipline, the repository is arranged so that overclaiming is structurally difficult.

**Documents are generated from artifacts.** `docs/build_docs.py` regenerates `README.md` and the solution walkthrough by reading `models/*.json`. No metric in the README is typed by hand, which means no metric can silently survive a change in the underlying result. This document is the mirror image of that rule: it contains no measured numbers at all, so it can never contradict a regenerated README. If you want a result, follow a link to it.

**Every quoted recall carries its sample size and a confidence interval.** `wilson_interval()` in `src/defend/evaluate.py` is applied in `_per_attack_recall()`, and each family entry carries `recall`, `n`, `ci95` and a boolean `sufficient_n` tested against `config.FAMILY_EVAL["min_n_to_report"]`. The reason is stated in the function's own docstring: a recall figure measured on a handful of transactions carries an interval no headline can support, and the interval answers that objection before it is asked. Per-family figures are measured on a dedicated large frame generated from a seed offset (`FAMILY_EVAL["seed_offset"]`) that no model was trained or tuned on — a different synthetic portfolio entirely — while overall precision and false-positive figures are reported only on the realistic-base-rate test slice. The loop's own frames carry an enriched fraud rate (`LOOP_FRAME_FRAUD_RATE`) purely so guard-family recall rests on enough rows to be worth quoting, and the code comment states that they are never used for precision or false-positive claims.

**Residual frontiers are published rather than buried.** `blind_spots()` in `src/defend/diagnostics.py` maintains a standing register of what the defence handles worst, committed as `models/blind_spots.json`. The loop classifies each family's outcome from metrics rather than by assertion — `_status()` returns `adapted`, `partial` or `residual_frontier` based on the measured gain against `LOOP_CONFIG["recovery_delta"]` and `LOOP_CONFIG["residual_recall_ceiling"]`. Families that stay a residual frontier across two consecutive rounds are retired from targeting, and the retirement is recorded in `retired_frontiers` with its reason and its replacement, never silently dropped. The exclusion of structurally unlearnable families is likewise evidence-based: `unlearnable_families()` reads `models/leave_one_out.json`, which trains a model *with* each family present — the most favourable case there is — and excludes only families that still cannot clear `LEARNABILITY_FLOOR` under those conditions. Those families are still simulated, still scored and still reported; they are just not pretended to be solvable by more of the same. The Threat Atlas applies the same discipline at the catalogue level: of the forty-five entries, thirty-one are marked simulatable and the rest are labelled research-only or future, and `_validate()` enforces that an unsimulated attack cannot claim an injector.

**Rejected challengers ship and are labelled.** Both `loop_adapted_model.joblib` and `loop_champion_model.joblib` are committed, and `models/model_registry.json` records every candidate with its gates, the observed and required value for each, and the promote/reject decision. The registry's own note states that a deployment would additionally run a shadow and controlled-rollout period, which this prototype does not simulate. The README publishes the promoted champion and the non-promoted candidate side by side. "The loop keeps improving the defence" is therefore a claim the committed artifacts are capable of contradicting, which is the only kind of claim worth making.

One deliberate omission supports all of this: registry entries carry no wall-clock timestamp by default, because committed artifacts must stay byte-stable across runs. A deployment that wants real timestamps passes `trained_at` explicitly.

---

## 8. Explicit non-goals

Stating what a system does not attempt is part of describing what it is.

**It is not validated on real payment data.** No real transactions, no real fraud labels, no comparison to any production portfolio. The fidelity diagnostics measure the simulator against itself and say so in their own scope field. Nothing here has been reviewed or validated by Mastercard.

**It is not a production fraud platform.** The decision policy is a prototype decisioning layer for simulation, not a description of any real network's system. The registry is a JSON file providing traceability, not an MLOps platform. There is no shadow-mode evaluation, no controlled rollout, no drift monitoring, no feature store, and no serving path.

**It does not produce operational attack content.** The red-team agent chooses positions on a fixed set of behavioural dials that a sandboxed simulator executes. Its system prompt forbids operational guidance, tooling, real targets, working infrastructure and real personal data, and the offline artifact vocabulary contains no brand, link, phone number, application name or procedure — only the *shapes* of scam messages, which is what a scam-text detector has to learn. Attack families are modelled purely as their behavioural consequences in the authorization stream; `wallet_provisioning`, for example, models what provisioning abuse looks like in the transaction record and says nothing about how a provisioning control would be defeated.

**It does not attempt to solve every attack family.** Some fraud is not an authorization-time problem. Friendly fraud is a genuine purchase by the genuine cardholder on the genuine device, and its defining signal is a dispute that arrives weeks later. Adversarial mimicry sits on the victim's own behavioural centroid by construction. These are simulated, scored and reported as residual frontiers precisely so the honest answer can be measured rather than asserted, and the loop deliberately declines to spend replay capacity pretending otherwise.

**It does not claim the adaptive model dominates the static one everywhere.** On the original attack distribution — the static model's home ground, since it was trained on exactly that — the adaptive model is at best comparable. Its case rests on robustness to attacks that have moved, measured against evolved generations, and the README prints the per-family trade rather than a single mean that would hide both halves of it.

**It is not a general adversarial-ML framework.** The mutation policy is payment domain knowledge encoded by hand in `SIGNAL_TO_DIAL`, not a learned or gradient-based attack. There is no attempt at gradient-based evasion, model extraction, or membership inference; those are different problems with different threat models.

---

## Where to go next

- **[Architecture](ARCHITECTURE.md)** — how the packages under `src/` are wired together, what each module is responsible for, and the order in which `src/pipeline.py` regenerates artifacts.
- **[README](../README.md)** — every measured result, regenerated from artifacts. Start with [Headline results](../README.md#headline-results), then [The closed loop](../README.md#the-closed-loop) and [Limitations](../README.md#limitations).
- **[Judge Q&A](JUDGE_QA.md)** — the direct answers to the questions this document's reasoning tends to provoke.
- **[Demo guide](DEMO_GUIDE.md)** and **[three-minute pitch](PITCH_3MIN.md)** — the presentation path through the same material.
- **The code itself** — the design decisions described here are documented at their point of use. `src/generate/attack_spec.py`, `src/loop/weakness.py` and `src/defend/features.py` carry the densest rationale.
