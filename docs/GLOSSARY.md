# Glossary

*This document is for anyone reading the repository from one side of its subject matter only: a payments or fraud specialist who does not work with machine-learning vocabulary, a machine-learning engineer who does not work with card-network vocabulary, and a reviewer who needs to know what the project's own invented terms mean. After reading it you will be able to read [ARCHITECTURE.md](ARCHITECTURE.md), [DATA_MODEL.md](DATA_MODEL.md), [EXPERIMENTS.md](EXPERIMENTS.md) and the source itself without stopping at a word. Every entry is tagged with the domain it comes from, and every project-specific entry names the module where the concept is implemented so you can check the definition against the code.*

Entries are ordered alphabetically in one sequence, not grouped by domain, because a reader who does not know a term does not know which domain it belongs to. Each entry carries one of three tags:

| Tag | Meaning |
| --- | --- |
| **[Payments]** | A term from payments, cards or fraud operations. Defined here as a payments practitioner would use it, then related to where it appears in this repository. |
| **[ML]** | A term from machine learning, statistics or model evaluation. |
| **[Project]** | A term this project coined or gave a specific local meaning. The definition names the implementing module. |

This document deliberately contains no measured results. Recall, precision, area-under-curve figures, false-positive rates and every other number produced by a run live in [README.md](../README.md) and in the generated artifacts under `models/`, so that a definition here can never drift out of agreement with a measurement there. Counts that are structural — how many attack injectors exist, how many feature columns the model consumes — are quoted, and each says where in the code it comes from.

---

## Numerals

**3-D Secure** — **[Payments]** The card-network protocol that lets an issuer challenge a card-not-present purchase before approving it, typically by asking the cardholder to confirm their identity in a banking app or with a passcode. It is the most common form of step-up authentication on cards. In this repository the fact that a transaction carried a 3-D Secure check is the `is_3ds` field on every simulated transaction and the `is_3ds` model feature listed in `src/defend/features.py`; the `STEP_UP` action in `src/defend/decision_policy.py` represents routing a transaction into such a challenge.

---

## A

**Acquirer** — **[Payments]** The bank or payment processor that holds the merchant's account and submits the merchant's card transactions into the network. The acquirer is the merchant's side of a card transaction, as the issuer is the cardholder's side. This project does not model acquirers as entities; the term appears in the simulator only as commentary on which side of the network a given signal would be observed from, for example at `src/generate/attack_injectors.py` where declined probing attempts are described as part of the acquirer-side story.

**Adapted defense** — **[Project]** The candidate model produced by one round of the closed loop: a model retrained on the base training data plus the entire adversarial replay buffer, then recalibrated and re-thresholded. It is built in `run_loop()` in `src/loop/redteam_loop.py` and the final one is persisted to `models/loop_adapted_model.joblib`. An adapted defense is explicitly *not* a deployed defense — it becomes deployable only if it clears the promotion gates and becomes a promoted champion.

**Attack generation** — **[Project]** One numbered version of an attack family's behavior. Generation 0 is the family's starting behavior, defined in `BASE_SPECS` in `src/generate/attack_spec.py`; each closed-loop round produces the next generation by mutating one behavioural dial to remove the signal the current defense was measured to depend on. The number is carried on the `generation` field of every `AttackSpec`, and `validate_spec()` increments it when a proposal does not state one.

**Attack lineage** — **[Project]** The recorded chain of generations for one attack family, where each node holds the specification that produced it, the measured weakness that motivated the change, what the constraint layer altered, and what the change cost the defense. Nodes are built by `lineage_entry()` in `src/loop/weakness.py`, using `spec_diff()` from `src/generate/attack_spec.py`, and the whole structure is written to `models/attack_lineage.json`. Its purpose is to make "the attack evolved" an inspectable record rather than an assertion.

**AttackSpec** — **[Project]** The structured, executable description of one attack generation: a frozen dataclass in `src/generate/attack_spec.py` carrying the family name, a stealth `intensity`, six categorical behavioural dials, the detector signal the generation is aiming to defeat, and provenance fields recording whether a language model, the offline heuristic, or a fixed baseline produced it. It is the contract between the creative half of the red team and the deterministic half of the simulator: the language model writes specifications, never transaction rows.

**Authorization** — **[Payments]** The real-time request, sent at the moment of purchase, asking whether a card or account payment may proceed, and the approval or refusal that answers it. Every row in this project's simulated dataset is one authorization. An authorization is a decision that must be made in milliseconds with only the information available at that instant, which is what makes it a harder detection problem than post-settlement fraud review.

**Authorization-time** — **[Payments] / [Project]** Describes information or a decision available at the moment of the authorization request, before the transaction settles and long before any dispute. The distinction is load-bearing here: `src/defend/features.py` divides transaction columns into authorization-time features that the model may use, post-outcome columns that are only knowable later, and oracle columns that no real defender ever has, and `assert_auth_time_safe()` raises if either of the latter two classes reaches the feature matrix.

**Authorized push payment scam** — **[Payments]** A fraud in which the genuine account holder is deceived into sending money themselves — the payment is correctly authenticated, correctly authorized, and irrevocable, and the only thing wrong with it is the reason it was made. It is simulated by the `scam_transfer` injector in `src/generate/attack_injectors.py`. Its baseline specification in `BASE_SPECS` records `targets_signal` as authentication itself, because the customer's own device, geography and credentials all check out.

---

## B

**Base rate** — **[ML]** The proportion of a population that belongs to the positive class — here, the share of authorizations that are fraudulent. The simulated portfolio's base rate is set by `config.DEFAULT_FRAUD_RATE`, deliberately low to match card portfolios. The base rate governs how any other metric should be read: at a low base rate a classifier can look excellent while being useless, which is why this project reports precision-recall area rather than leaning on accuracy or receiver-operating-characteristic area alone.

**Behavioural dial** — **[Project]** One field of an `AttackSpec` that controls how an attack behaves, as opposed to what family it belongs to. There are seven, defined in `src/generate/attack_spec.py`: the numeric `intensity` plus the categorical `amount_profile`, `velocity_profile`, `device_behavior`, `geo_behavior`, `merchant_behavior` and `timing_profile`. Attack evolution consists entirely of moving dials; the injectors read them and nothing about stealth is hardcoded inside an injector.

**Brier score** — **[ML]** The mean squared difference between a predicted probability and the observed outcome, used to judge calibration rather than ranking. It is computed before and after isotonic calibration in `src/defend/diagnostics.py` and reported in `models/calibration.json`. That module also records the caveat that matters at a low base rate: a low Brier score is easy to obtain by predicting near zero for everything, so it is evidence only alongside a reliability curve.

**Bust-out** — **[Payments]** A fraud in which an account is built up with ordinary, well-behaved spending — often for weeks — to earn trust and credit limit, and then drained to that limit in a short burst with no intention of repaying. It is simulated by the `bust_out` injector in `src/generate/attack_injectors.py`, whose grooming period is emitted as cover traffic. Its family constraints in `FAMILY_CONSTRAINTS` require a high or extreme amount profile and a cash-like or high-risk merchant, because a bust-out that drains nothing is not a bust-out.

---

## C

**Calibration** — **[ML]** The property that a predicted probability means what it says: among transactions scored at 0.30, roughly three in ten should actually be fraudulent. `DefenseModel` in `src/defend/model.py` keeps the uncalibrated fused score for thresholding and exposes a separately calibrated probability through `risk_probability()`, because the decision tiers and anything a reviewer reads should be statements about risk rather than positions on an arbitrary scale.

**Card-not-present** — **[Payments]** A transaction where the card is not physically presented — online, in-app, over the phone, or by stored credential. The card cannot be inspected and the cardholder cannot be observed, so card-not-present traffic carries most remote fraud. It is the `card_cnp` channel in `config.RAILS`, the `is_cnp` model feature, and a per-category propensity (`cnp_prob`) on every entry in `config.MCC_CATALOG`.

**Card-present** — **[Payments]** A transaction where the physical card or its digital equivalent is presented at a terminal — a chip dip, a contactless tap, an automated teller machine withdrawal. Fewer fraud families are viable card-present because the credential has to be physically or electronically reproduced. It is the `card_cp` channel in `config.RAILS`.

**Card testing** — **[Payments]** Validating a batch of stolen card numbers by attempting many small, cheap purchases and keeping whichever cards approve. The individual transactions are trivial in value; the pattern is the crime. It is simulated by the `card_testing` injector in `src/generate/attack_injectors.py`, and its family constraints restrict it to micro or low amounts at moderate or burst velocity, because probing at high value and low volume would not be economically sensible.

**Catastrophic forgetting** — **[ML]** The tendency of a model retrained on new data to lose performance on patterns it had previously learned. It is the central risk of an adaptive defense: a model taught to catch this month's evolved attack may quietly stop catching last month's. The closed loop measures it directly — every round re-scores all prior attack generations and the untouched guard families — and `src/defend/governance.py` refuses to promote a candidate whose regression on a previously-learned family is statistically real.

**Champion and challenger** — **[ML]** A deployment discipline in which the model currently in force is the champion, any newly trained model is a challenger, and the challenger replaces the champion only after passing explicit comparison gates. Implemented in `src/defend/governance.py`, whose `evaluate_candidate()` runs the gates configured in `config.CHAMPION_CHALLENGER` and returns `PROMOTE` or `REJECT` with the reasoning for each gate.

**Chargeback** — **[Payments]** The mechanism by which a cardholder's disputed transaction is forcibly reversed and the funds pulled back from the merchant through the network. A chargeback arrives days or weeks after authorization, which is precisely why it cannot be a model input here: reversal-related outcomes sit in `POST_OUTCOME_COLUMNS` in `src/defend/features.py` and are blocked from the feature matrix.

**Chronological split** — **[ML]** Dividing a dataset into training, validation and test portions by time rather than at random, so that the model is always trained on earlier transactions and evaluated on later ones. This mirrors deployment, where a model necessarily faces future traffic. It is implemented once, in `split_points()` and `split_xy()` in `src/defend/train.py`, so that the trainer, the benchmark harness and the closed loop cannot silently disagree about where the boundaries fall.

**Class imbalance** — **[ML]** The situation where one class is far rarer than the other, as fraud is among authorizations. Imbalance makes accuracy meaningless, makes small evaluation slices noisy, and pushes an untreated classifier toward predicting the majority class. This project addresses it with a balanced class weight in the supervised head (`src/defend/model.py`), a threshold tuned against a false-positive budget rather than left at one half, and dedicated fraud-enriched evaluation frames when per-family numbers are needed.

**Constraint layer** — **[Project]** The payment-domain validator that stands between any proposed attack specification and the simulator. It is `validate_spec()` in `src/generate/attack_spec.py`: values outside the permitted vocabulary are replaced, numbers outside `config.ATTACK_SPEC_BOUNDS` are clamped, and a proposal that contradicts a family's payment-domain requirements in `FAMILY_CONSTRAINTS` is corrected to the nearest legal value — or, under `strict=True`, rejected outright. Every correction is recorded in a `ValidationReport`, so what the layer changed is visible rather than silent.

**Cover traffic** — **[Project]** Ordinary-looking transactions generated by a fraud actor before the abuse begins: a mule account's grooming payments, a bust-out account's months of well-behaved spend, a front merchant's genuine-looking sales. Emitted by `_cover_row()` in `src/generate/attack_injectors.py` with the fraud label set to zero and the `actor_role` field set to `fraud_actor_cover`, because at authorization time those transactions genuinely are not fraud. Its purpose is to stop "this card has no history" from becoming a synonym for fraud.

---

## D

**Demo mode** — **[Project]** The default runtime path of the Streamlit application, selected by `mode_selector()` in `app/common.py`. Every page renders from committed artifacts under `models/` and `data/`: nothing simulates, nothing trains, no language model is called and no API key is needed. It exists so a demonstration cannot stall on recomputation and so every figure on screen matches the committed numbers exactly. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full comparison with Live mode.

**Dispute** — **[Payments]** A cardholder's assertion that a transaction was not legitimate, which may or may not proceed to a chargeback. In the simulator a downstream dispute or refund is the `refund_flag` column, which is post-outcome and therefore never a feature of the transaction it belongs to. A card's *prior* dispute history is a different matter and is legitimately available at authorization time: `card_prior_dispute_rate` in `src/defend/features.py` is the mean of that flag over the card's strictly earlier transactions.

---

## F

**F1** — **[ML]** The harmonic mean of precision and recall, a single number that penalizes a model that achieves one at the expense of the other. Computed in `src/defend/evaluate.py`. It is reported here for completeness rather than used as the objective, because at a low base rate the operationally meaningful question is what recall can be bought within a fixed false-positive budget.

**False-positive rate** — **[ML]** The share of genuine transactions that the model flags. In payments it is the customer-friction number: every false positive is a real customer challenged, delayed or declined. It is computed against the legitimate rows only in `src/defend/evaluate.py`, it is the quantity the operating threshold is tuned against via `config.TARGET_MAX_FPR`, and it is enforced as both an absolute ceiling and a maximum permitted regression by the promotion gates in `src/defend/governance.py`.

**Fidelity diagnostics** — **[Project]** The self-checks that ask whether the simulator is a useful training ground at all, implemented in `src/generate/fidelity.py` and written to `models/fidelity_report.json`. They measure how far each single feature can separate fraud from legitimate traffic, how far the two distributions overlap, and whether structural realism holds — that genuine customers sometimes change device, that fraud sometimes reuses a known merchant, that first-ever transactions are not a fraud synonym. Each check emits an explicit PASS, WARN or FAIL. The module states plainly that these compare the simulator against itself and involve no comparison with any real payment portfolio.

**First-party fraud** — **[Payments]** Also called friendly fraud: the genuine cardholder makes a genuine purchase on their own card and device and then disputes it, claiming they did not. There is almost nothing to see at authorization time, which is why the `friendly_fraud` entry in `BASE_SPECS` records its targeted signal as authorization-time observability itself, and why `src/loop/redteam_loop.py` treats the family as a structural frontier rather than a red-team target.

**Focus family** — **[Project]** An attack family the red team is currently attacking in the closed loop. Focus families are chosen by measurement, not in advance: `select_focus()` in `src/loop/redteam_loop.py` ranks families by how badly the defense in force handles them and takes the worst, excluding families already judged structurally unlearnable. `_focus_mix()` then up-weights them in the simulated frames so their per-family recall rests on enough rows to mean something, with a floor that keeps every other family present.

**Front merchant** — **[Payments]** A merchant account controlled by the fraudster and used as a conduit — to launder stolen card transactions, to receive scam proceeds, or to give a criminal operation a legitimate-looking acceptance surface. Simulated by the `merchant_laundering` injector and by the `front_merchant` value of the `merchant_behavior` dial. The network-level feature `merchant_new_card_ratio_prior` in `src/defend/features.py` exists largely for this case: an ordinary merchant converts repeat customers, while a front merchant sees almost only first-time cards.

---

## G

**Guard family** — **[Project]** An attack family deliberately *not* attacked in the current round, watched to prove that adaptation has not degraded what the defense already knew. Guards are resolved at run time by `resolve_guards()` in `src/loop/redteam_loop.py` from `GUARD_CANDIDATES`, excluding anything currently under attack, because a family cannot be both the target and the control. Their recall is measured on a large held-out frame of generation-0 attacks and fed into the catastrophic-forgetting promotion gate.

---

## H

**Held-out set** — **[ML]** Data that played no part in fitting a model and is therefore usable for an honest estimate of how it performs. This project holds out three distinct things: the chronological test slice used for headline metrics, a validation slice used for calibration and threshold tuning, and — inside the closed loop — a large freshly simulated evaluation frame per round, built by `_eval_frame()` in `src/loop/redteam_loop.py`, so that stale-versus-adapted comparisons are never made on rows either model was trained on.

---

## I

**Isotonic regression** — **[ML]** A non-parametric method that fits a monotone step function mapping raw scores onto observed outcome rates, used here as the calibration method in `DefenseModel.fit_calibration()` (`src/defend/model.py`). Monotonicity is the reason it was chosen: because it can never reorder two transactions, it cannot change any ranking metric, so calibration alters what the number means without altering which transactions outrank which.

**Issuer** — **[Payments]** The bank that issued the card and holds the cardholder's account, and which ultimately approves or declines an authorization. The decision policy in `src/defend/decision_policy.py` describes itself as a prototype issuer-or-network decisioning layer; the module and `config.py` both state explicitly that it is a simulation prototype and not a description of any production system.

---

## L

**Leakage** — **[ML]** Any situation where information reaches the model that would not be available at prediction time, producing results that cannot be reproduced in deployment. The classic payment case is training on a chargeback outcome to predict the authorization that caused it. This repository defends against it structurally: `src/defend/features.py` enumerates post-outcome and oracle columns and `assert_auth_time_safe()` raises if either class appears in the feature matrix, and every historical or relational feature is computed from strictly earlier rows only.

**Live mode** — **[Project]** The non-default runtime path of the Streamlit application, enabled by the sidebar toggle in `mode_selector()` (`app/common.py`). It unlocks fresh simulation, retraining and the closed loop behind explicit buttons and confirmations. Nothing heavy runs without a click, and if an Anthropic API key is present, live specification generation becomes possible; the committed artifacts were nevertheless produced by the deterministic offline path, which `llm_status_badge()` states on screen.

---

## M

**Merchant category code** — **[Payments]** The four-digit code identifying what a merchant sells — grocery, fuel, gambling, money transfer — assigned by the acquirer and carried on every card transaction. It is one of the strongest pieces of context available at authorization: a five-figure charge is ordinary at a travel agent and extraordinary at a grocer. `config.MCC_CATALOG` defines twelve simulated categories, each with a typical ticket distribution, a card-not-present propensity and a risk weight, and these feed the `mcc_risk`, `mcc_novel`, `card_mcc_share_prior` and `is_cash_like` features.

**Mule** — **[Payments]** An account, and usually a person, used to receive and forward criminal proceeds, breaking the trail between the victim and the fraudster. Mule accounts typically transact normally for a while before being used. They are simulated by the `velocity_smurfing` injector, which gives each mule cover traffic sized by `config.FRAUD_REALISM["mule_cover_txns"]` before the ring begins pushing value through it.

---

## N

**New payee** — **[Payments]** A merchant or beneficiary this card or account has never paid before. It is a genuine risk indicator — most fraud pays someone new — but it is emphatically not a fraud synonym, since genuine customers try new merchants constantly. The simulator computes it globally as the first occurrence of a card-and-merchant pair (`src/generate/simulate.py`), and `src/generate/fidelity.py` asserts as a quality check that a substantial but not overwhelming share of genuine transactions are first visits.

---

## O

**One-time passcode** — **[Payments]** A short single-use code sent to the cardholder, usually by text message or app notification, to prove they are present during a step-up challenge. Whether one was verified is the `otp_verified` field and model feature. Passcodes can be socially engineered out of a victim in real time, which is the `otp_relay` attack family: the transaction arrives with authentication satisfied, which is why its baseline specification places it in the customer's ordinary waking hours — a relay needs the victim awake and on the phone.

**Operating point** — **[Project] / [ML]** The score threshold at which the defense actually flags, together with what that choice costs. It is not a default: `DefenseModel.tune_threshold()` in `src/defend/model.py` searches distinct score values for the lowest threshold whose realized false-positive rate on genuine traffic still fits the budget in `config.TARGET_MAX_FPR`, which maximizes recall subject to that budget. The module explains why realized rates are searched rather than a quantile taken: calibrated scores are a step function, and a quantile lands inside a tied block whose whole contents the comparison then admits.

**Oracle column** — **[Project]** A column that exists only because a simulator generated the data and that no real defender would ever possess: `is_fraud`, `attack_type` and `actor_role`, enumerated as `ORACLE_COLUMNS` in `src/defend/features.py`. They are used for labelling, per-family evaluation and analysis, and are hard-blocked from the feature matrix by `assert_auth_time_safe()`. The category exists so that ground truth used for measurement can never be mistaken for a signal used for prediction.

---

## P

**Payee** — **[Payments]** The party receiving the money. On cards the payee is the merchant acceptor; on account-to-account rails such as UPI the payee is an address that resolves to an account, which is a materially different object — it can be created cheaply, changed often, and has no acceptance relationship with a network. `config.py` records this difference as one of the reasons UPI is simulated as a genuine channel rather than folded into card-not-present.

**Permutation importance** — **[ML]** A model-agnostic way to ask what a fitted model relies on: shuffle one feature's values, re-score, and see how much ranking quality falls. A large drop means the model depends on that feature. It is used twice here — globally over the test slice in `_perm_importance()` (`src/defend/train.py`) for reporting, and scoped to a single attack family in `analyze_family()` (`src/loop/weakness.py`), where it becomes the signal attribution that aims the next attack generation.

**PR-AUC** — **[ML]** The area under the precision-recall curve, a summary of ranking quality that, unlike receiver-operating-characteristic area, does not flatter a model on heavily imbalanced data — its baseline is the base rate itself. Computed as average precision in `src/defend/evaluate.py`. It is the quantity the overall-ranking-quality promotion gate protects in `src/defend/governance.py`, so that a candidate cannot buy recall on one attack family by degrading its ordering of everything else.

**Precision** — **[ML]** Of the transactions the model flags, the share that really are fraudulent. It is the analyst's number: low precision means review queues full of genuine customers. Computed in `src/defend/evaluate.py`. Precision depends heavily on the base rate of the frame it is measured on, which is why this project reports precision only from the realistic-base-rate dataset and never from the fraud-enriched closed-loop frames.

**Promoted champion** — **[Project]** The most recent candidate model that cleared every governance gate and would therefore be allowed into the authorization path. It is tracked separately from the final adapted candidate throughout `run_loop()` in `src/loop/redteam_loop.py` and saved to `models/loop_champion_model.joblib`, alongside `models/loop_adapted_model.joblib`. When no candidate passes, the two files are different models, and that difference is the intended result rather than a failure of the demonstration.

---

## R

**Reason code** — **[Project] / [Payments]** A short human-readable explanation attached to a decision, naming which rule-style signals fired — abnormal velocity, geographic anomaly, sub-threshold structuring, device shared across multiple cards. Produced by `reason_codes()` in `src/defend/decision_policy.py` from the rule hits computed in `src/defend/baseline.py`, using the labels in `REASON_LABELS`. Reason codes exist because an analyst receiving a flagged transaction needs to know why, and a bare score does not answer that.

**Recall** — **[ML]** Of the fraudulent transactions present, the share the model flags. It is the loss-prevention number, and per-attack-family recall is the primary way this project measures where a defense is weak. Computed overall and per family in `src/defend/evaluate.py`, which also attaches a Wilson interval and a sufficiency flag to every family figure so a number measured on a handful of transactions is never read as though it were solid.

**Rehearsal** — **[ML]** The standard remedy for catastrophic forgetting in continual learning: when training on a new task, mix in examples of the earlier tasks. In `src/loop/redteam_loop.py` the replay buffer receives not only the newest adversarial examples but a `rehearsal_mask` sample of every family the red team did *not* attack, plus legitimate context rows. The module records that this was added in response to an observed failure — a buffer of only the newest, hardest attacks pulled the decision boundary toward them and degraded untouched families, which the forgetting gate then correctly refused to promote.

**Replay buffer** — **[ML] / [Project]** The bounded, stratified store of adversarial feature rows accumulated across closed-loop rounds and mixed into every retraining. Managed by `_bounded_append()` in `src/loop/redteam_loop.py` with a size cap from `config.LOOP_CONFIG["replay_buffer_max"]`; when the cap is exceeded it downsamples per generation rather than globally, so early attack generations stay represented instead of being crowded out by the newest. Replayed rows are emphasized during training with a sample weight rather than by duplicating rows, which keeps the strength of the emphasis explicit.

**Residual frontier** — **[Project]** The status assigned to a focus family whose recall stays below the configured ceiling even after the defense has adapted to it — that is, an attack the loop has not managed to close. It is derived from measurements by `_status()` in `src/loop/redteam_loop.py` against `config.LOOP_CONFIG["residual_recall_ceiling"]` and `recovery_delta`, never hardcoded. A family that remains a residual frontier across two consecutive rounds is retired from the focus set with the reason recorded, and replaced by the next measurably weakest family.

**ROC-AUC** — **[ML]** The area under the receiver-operating-characteristic curve: the probability that a randomly chosen fraudulent transaction scores above a randomly chosen genuine one. Computed in `src/defend/evaluate.py`. It is a pure ranking measure and is reported here for completeness, but it is a poor primary metric under heavy class imbalance because the large genuine population makes the false-positive axis move very slowly, which is why the precision-recall area is preferred.

---

## S

**Separability** — **[Project] / [ML]** How well a single feature, on its own, can rank fraud above legitimate traffic. Measured per feature by `separability()` in `src/generate/fidelity.py`, which also reports the histogram overlap of the two class distributions. High separability on any one feature is a *bad* result here, not a good one: it means the detector could read the generator instead of the fraud, so the module defines explicit warning and failure lines (`SEPARABILITY_WARN`, `SEPARABILITY_FAIL`) and raises a check when any feature crosses them.

**Shortcut learning** — **[ML]** When a model solves a benchmark by exploiting an incidental regularity rather than the intended structure — learning that fraud is whatever happens after midnight, or whatever has no card history. It is the characteristic failure of synthetic data, and this project treats it as a first-class risk: the legitimate-behavior realism parameters in `config.LEGIT_REALISM`, the cover traffic emitted by every injector, and the fidelity checks in `src/generate/fidelity.py` all exist to remove shortcuts before the model can find them.

**Signal attribution** — **[Project]** The measurement of which features a given model actually depends on *for one attack family*, computed by `analyze_family()` in `src/loop/weakness.py` as a permutation test scoped to that family's fraudulent rows plus a legitimate sample. The ranked result is the `relied_signals` list on a weakness report, and its top entry is what the next attack generation is aimed at. This is what makes the loop adversarial rather than a scripted walk down a stealth dial: each mutation removes a dependency that was measured, not guessed.

**Stale defense** — **[Project]** The model that was in force before the current round's attack generation existed — the defense the red team is trying to get past. Held as `prev_model` in `run_loop()` (`src/loop/redteam_loop.py`) and persisted as `models/loop_base_model.joblib` for the demonstration. The gap between what the stale defense catches and what the adapted defense catches on the same held-out frame is the quantity the closed loop is built to expose.

**Step-up authentication** — **[Payments]** Adding friction to a transaction rather than refusing it: challenging the customer to prove they are present, typically through 3-D Secure or a one-time passcode. It is the middle tier of the decision policy in `src/defend/decision_policy.py`, between approval and decline. Its threshold defaults to the model's own tuned operating point rather than a second invented cutoff, and `config.OPERATIONAL_SCENARIO` carries an illustrative abandonment assumption so that step-up volume can be read as customer friction rather than as a free action.

**Structural frontier** — **[Project]** An attack family excluded from red-team targeting because evidence shows it is limited by authorization-time observability rather than by the decision boundary — the genuine customer authenticated and authorized, or the behavior sits on the legitimate centroid. Determined by `unlearnable_families()` in `src/loop/redteam_loop.py` against `LEARNABILITY_FLOOR`, using the leave-one-attack-family-out evidence in `models/leave_one_out.json`: a family trained on directly, the most favorable case available, that still cannot be caught will not yield to adversarial replay either, since replay is a slower route to the same thing. Such families are still simulated, scored and reported — they are simply not pretended to be solvable by more of the same.

**Structuring** — **[Payments]** Deliberately sizing payments to sit just below a threshold that would trigger reporting, review or a step-up challenge. It is why an absolute amount rule is weak on its own and why the `near_threshold` feature in `src/defend/features.py` looks for tickets in the band immediately below a common threshold rather than above it. It also appears as a reason code label in `src/defend/decision_policy.py`.

**Synthetic identity** — **[Payments]** A fabricated person assembled from a mixture of real and invented attributes, used to obtain credit that will never be repaid. Because the identity does not correspond to a victim, nobody reports the fraud until the account defaults. In this simulator, `_synthetic_holder()` in `src/generate/attack_injectors.py` creates a fabricated cardholder profile that behaves like a real one so that bust-out grooming produces credible history rather than an obviously empty account.

---

## T

**Threat Atlas** — **[Project]** The project's catalogue of GenAI-enabled payment fraud attacks and the single source of truth for the identification pillar. The data lives in `src/identify/attacks.json` and is loaded and validated by `src/identify/taxonomy.py`, which enforces unique identifiers, valid injector mappings and a legal status on every entry. The catalogue currently holds 45 attacks across 6 categories, and is deliberately wider than the simulator: every entry carries a `simulator_status` of `IMPLEMENTED`, `PARAMETERIZED`, `RESEARCH_ONLY` or `FUTURE`, so research breadth is never presented as simulation breadth. The validator enforces the honesty directly — a `RESEARCH_ONLY` or `FUTURE` entry that claims an injector raises, and so does an `IMPLEMENTED` entry that names none. The simulator itself has 11 injectors, listed in `INJECTORS` in `src/generate/attack_injectors.py`.

**Token provisioning** — **[Payments]** Loading a card into a digital wallet, which replaces the card number with a device-bound token. A provisioned token inherits trust: subsequent spend arrives already authenticated. If the provisioning step itself is compromised, the attacker acquires that inherited trust. This is the `wallet_provisioning` injector in `src/generate/attack_injectors.py`, whose rows arrive with 3-D Secure and passcode verification both satisfied. The module notes explicitly that it describes only the authorization footprint and no provisioning control.

**Transaction laundering** — **[Payments]** Processing payments for one business through another business's merchant account, so that the card network and the acquirer see a merchant that is not the one actually being paid. It is simulated by the `merchant_laundering` injector, whose family constraints in `src/generate/attack_spec.py` require the `front_merchant` behavior — laundering without a front is a different attack, not a stealthier version of this one.

---

## U

**UPI** — **[Payments]** The Unified Payments Interface, India's real-time account-to-account retail payment rail. It is simulated as a genuine channel in `config.RAILS`, parameterized by `config.UPI_REALISM`, and surfaced to the model as the `is_upi` feature. `config.py` records why it is not collapsed into card-not-present: the payer authenticates on every payment so a passing step-up means much less, settlement is instant and irrevocable, tickets are smaller and far more frequent, and the payee is an address rather than a card acceptor. It is also the rail on which most authorized push payment scams actually settle.

---

## V

**Velocity** — **[Payments]** The rate at which a card, account or device is transacting — how many attempts in the last hour, the last day. It is one of the oldest and most reliable fraud signals, because most attacks need volume. Represented by the `velocity_1h` and `velocity_24h` features in `src/defend/features.py`, computed as trailing rolling counts per card, and by the `velocity_profile` behavioural dial, whose values run from a single transaction to a burst.

**Virtual payment address** — **[Payments]** The human-readable address that identifies a payee on UPI and resolves to an underlying bank account, so that no account number is exchanged. Because addresses are cheap to create and easy to change, beneficiary-side patterns — how many unrelated payers a newly created address suddenly receives from — carry more signal than any individual payment does. The Threat Atlas entries in `src/identify/attacks.json` describe several attacks in exactly those terms; the simulator models the payee generically as a merchant identifier rather than reproducing address semantics.

---

## W

**Weakness report** — **[Project]** The structured output of blue-team analysis for one attack family against one model: its recall, how many of that family's transactions escaped, the ranked signal attribution, and a profile comparing the escaped transactions against the caught ones and against legitimate traffic. It is the `WeaknessReport` dataclass produced by `analyze_family()` in `src/loop/weakness.py`, and it is what the red team's next proposal is derived from — by the language model in the live path, and by `heuristic_mutation()` in the same module offline.

**Wilson interval** — **[ML]** A confidence interval for a proportion that behaves sensibly at small sample sizes and near zero or one, where the naive interval does not. Used in two places: `wilson_interval()` in `src/defend/evaluate.py` attaches one to every per-family recall figure, so a proportion measured on a handful of transactions is never quoted bare; and `_wilson()` in `src/defend/governance.py` makes the catastrophic-forgetting gate fire only when a candidate's recall is confidently below the tolerated level, because a gate that blocks a better model on sampling noise is worse than no gate.

---

## Where to go next

If a term sent you here from somewhere else, go back to it. If you are reading this document first, the useful order is [ARCHITECTURE.md](ARCHITECTURE.md) for how the four pillars fit together and which module does what, then [DATA_MODEL.md](DATA_MODEL.md) for the transaction schema and the exact column classes that the leakage rules in this glossary refer to, then [EXPERIMENTS.md](EXPERIMENTS.md) for how each measured claim was produced. [DECISIONS.md](DECISIONS.md) records why the design choices behind several of these terms were made, and the top-level [README.md](../README.md) carries every measured result.
