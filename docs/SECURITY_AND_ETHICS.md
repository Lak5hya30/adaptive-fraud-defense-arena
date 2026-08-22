# Security and Ethics

*This document is for judges, reviewing engineers, security reviewers and anyone
considering contributing to or reusing this repository. It states what the project
is for, the specific safety constraints built into the threat catalog and the
generative components, the things the project deliberately refuses to do, how
secrets and data are handled, and the commitments the project makes about how it
reports results. After reading it you will know exactly where the boundary between
defensive research and operational attack capability was drawn in this codebase,
and which file to open to check that the boundary is real.*

---

## Purpose and scope

This project is defensive fraud-detection research. It exists to answer one
question: when generative AI lets attackers iterate faster than a periodically
retrained fraud model can, what does a defense that iterates back look like, and
how would you know whether it works.

The system is a closed loop. It catalogs emerging payment-fraud attacks (the
Threat Atlas in `src/identify/attacks.json`), reproduces their authorization-time
footprint as constrained synthetic transactions (`src/generate/`), defends with an
authorization-time model (`src/defend/`), measures which signals that defense
actually depends on (`src/loop/weakness.py`), evolves the simulated attack to
remove the signal the defense leaned on (`src/loop/redteam_loop.py`), replays what
escaped into training, and then applies promotion gates that decide whether the
retrained model may ship at all (`src/defend/governance.py`).

Every part of that loop operates on data this repository generates. There is no
real cardholder data, no real merchant, no real device, no real payment service
and no real account anywhere in the system. The output of the project is a
measurement of how a defense behaves under adaptive pressure, not an attack
capability.

Scope is also worth stating negatively. This is a laboratory, not a product. It
has no shadow deployment, no drift monitoring and no analyst feedback loop, and
its results have never been validated against real payment data. Those absences
are recorded in the Limitations section of [the README](../README.md#limitations)
and are the reason several of the commitments below exist.

---

## The safety posture of the Threat Atlas

The Threat Atlas is the part of this project most likely to be misread as offensive
material, so it was written under a single editorial rule: **every field describes
what a defender can observe, and nothing describes how to carry out the attack.**
That rule is stated in the module docstring of `src/identify/taxonomy.py` and again
inside the catalog itself, in the top-level `honesty_note` field of
`src/identify/attacks.json`:

> Entries describe defender-observable consequences and signals only — no
> operational guidance, tooling, or bypass technique appears anywhere in this file.

The consequence of the rule is visible in the schema. An entry carries
`attacker_objective` (what the adversary is trying to achieve), `genai_role` and
`genai_mechanism` (what generative AI contributes at the level of "content
generation" or "identity synthesis"), `kill_chain` (the stages a defender would see
crossed), `transaction_signature` and `behavioral_signature` (what the resulting
traffic looks like), `observable_signals` and `post_transaction_signals` (the
features and downstream evidence a fraud team can act on), and
`auth_time_observability` (whether any of it is visible at the moment of
authorization at all). What the schema has no field for is procedure. There is no
tooling field, no prompt field, no infrastructure field, no step list. An attack is
characterized by its consequences because consequences are what a defense has to
detect.

### The `ethical_notes` field

Every entry carries an `ethical_notes` string, and its job is to state, per entry,
what that entry deliberately leaves out. All 45 entries in the catalog have one
populated; this was verified by loading `src/identify/attacks.json` and checking the
field on each record, and the catalog is loaded and validated by
`src.identify.taxonomy.load_taxonomy`. Two examples, quoted from the file:

| Entry | What its `ethical_notes` records as omitted |
|---|---|
| `voice_cloned_issuer_impersonation_vishing` | Names no cloning tools, sample sources, caller-identifier techniques or scripts; describes only account-behaviour and dispute consequences |
| `checkout_endpoint_merchant_enumeration` | Names no weakness class, no endpoint behaviour and no discovery technique; describes only volumetric and funnel-shape signals an acquirer or network sees |

The field is not decoration. Writing it forces the author of each entry to name the
operational detail that was withheld, which makes the omission a deliberate,
reviewable decision rather than an accident of how much the author happened to
know. A reviewer can read the `ethical_notes` of any entry and check the rest of
that entry against it.

### Status honesty is part of the safety posture

The catalog is deliberately wider than the simulator, and it would be easy to let
research breadth read as capability breadth. Every entry therefore carries a
`simulator_status` of `IMPLEMENTED`, `PARAMETERIZED`, `RESEARCH_ONLY` or `FUTURE`,
defined in the `status_definitions` block at the top of `src/identify/attacks.json`
and enforced by `_validate` in `src/identify/taxonomy.py`. The validator raises if
an entry claims a simulator status without naming an injector, and equally raises
if a `RESEARCH_ONLY` or `FUTURE` entry claims an injector it does not have. Eleven
injectors exist, listed in the `injectors` array of the catalog and registered in
the `INJECTORS` dictionary at the bottom of `src/generate/attack_injectors.py`; the
catalog describes far more attacks than that, and says so on every row.

This matters ethically as well as scientifically. A catalog that implied it could
simulate everything it describes would be overstating both the research and the
capability.

---

## The safety posture of the generative components

Generative AI appears in this system in exactly two places, and both are
constrained in structure rather than by instruction alone.

### The system prompts

Two system prompts exist, both in `src/generate/llm_agent.py`.

`ATTACKER_SYSTEM` governs the generation of synthetic content artifacts used to
populate the fidelity view and the text-classifier corpus. It frames the model as
building defensive training data and states the prohibition directly: "Never
include operational instructions, real targets, working infrastructure, or real
personal data. Focus on the observable linguistic and behavioural cues a detector
should flag."

`REDTEAM_SPEC_SYSTEM` governs the important path — proposing the next generation of
a simulated attack from measured evidence about the defense. It states the same
boundary in structural terms: "You never produce operational guidance, tooling, or
instructions anyone could act on: you only choose values on a fixed set of
behavioural dials that a sandboxed simulator will execute. Everything you describe
stays inside that simulator."

### The model chooses dial values, not attack material

A system prompt is a request, not a guarantee, so the request is backed by the
shape of the task. The model is never asked for attack content in the closed loop.
It is asked to return a JSON object choosing values on a fixed vocabulary, defined
as `_SPEC_VOCAB` in `src/generate/llm_agent.py` and bounded by
`config.ATTACK_SPEC_BOUNDS`:

| Dial | Permitted values |
|---|---|
| `amount_profile` | micro, low, moderate, high, extreme |
| `velocity_profile` | single, low_and_slow, moderate, burst |
| `device_behavior` | trusted_device, secondary_device, new_device, shared_device |
| `geo_behavior` | home, plausible, domestic_far, foreign, high_risk |
| `merchant_behavior` | known_merchant, new_low_risk_merchant, new_high_risk_merchant, front_merchant, cash_like |
| `timing_profile` | customer_normal, business_hours, night, any |
| `intensity` | a number between 0.15 and 1.0, never above the previous generation |

Everything outside that vocabulary is discarded before it can reach the simulator.
`validate_spec` in `src/generate/attack_spec.py` coerces any out-of-vocabulary
categorical value back to the family's baseline, clamps `intensity` into the
configured range, refuses to let stealth regress within an attack lineage, and then
applies the per-family payment-domain requirements in `FAMILY_CONSTRAINTS` — so a
proposal that, for example, asks for an authorized-push-payment scam executed from
an attacker device is corrected, because the genuine customer authenticating on
their own device is what defines that family. Under `strict=True` such a proposal
is rejected outright with `SpecRejected`. The free-text fields the model may return
(`strategy`, `rationale`, `targets_signal`) are length-capped and are carried as
labels for the lineage view; they are never executed.

```
measured weakness report        (numbers produced by src/loop/weakness.py)
        |
   language model               may propose anything
        |
   JSON dial selection          fixed vocabulary, nothing else survives
        |
   validate_spec()              clamp / coerce / reject  (attack_spec.py)
        |
   deterministic injector       seeded, local, no network  (attack_injectors.py)
        |
   synthetic transaction rows
```

The model never writes a transaction row. That split, documented at the top of
`src/generate/attack_spec.py`, is what makes the system both creative and
reproducible — and it is also the safety property, because the only thing the
language model can influence is which of eleven sandboxed simulators runs with
which of a few dozen enumerated settings.

### Synthetic lure text is composed from generic slots

The one place the system does produce message-shaped text is the corpus for the
text classifier, built by `build_text_corpus` in `src/generate/llm_agent.py`. When
no model is available, fraudulent samples are composed by `_template_artifact` from
a fixed slot vocabulary in the same module: eight openers, twelve pretext themes,
nine requested actions, six urgency lines and six sentence patterns, selected by a
seeded random generator and keyed to the attack's own catalog entry. The comment
above that vocabulary states the constraint that governs it — no real brand, link,
phone number, app name or step-by-step instruction appears in any slot. The
pretexts are one-clause descriptions of a situation ("your KYC record has expired
and your account will be frozen"), and the actions are one-clause descriptions of
what a victim is asked to do ("read back the code that has just been sent to your
phone"). None of it constitutes a usable lure, because a usable lure needs a
target, a channel, a brand and a destination, and the vocabulary contains none of
those.

Two further precautions apply to this corpus. Every generated fraud artifact is
tagged with `[SYNTHETIC]` and `[SIM]` markers so a human reader can never mistake
it for a real message, and `strip_markers` removes those tags before the text is
modelled so the classifier cannot learn the tag instead of the content. The benign
half of the corpus is composed from genuine-notice templates that share the same
urgency and authentication vocabulary, which is a modelling decision rather than a
safety one but has the same effect: the corpus is a study of message shape, not a
library of attack copy.

The project treats the text arm as a sanity check and not as a result at all. The
README's Limitations section states that its score measures the corpus rather than
the detector, and that every detection claim in the project rests on the
transaction model alone.

---

## What this project deliberately does not do

The following are absent by design, not by omission. Each is a capability the
project could plausibly have been extended toward and was not.

- **No credential harvesting.** Nothing in the repository collects, stores,
  requests or processes credentials of any kind. The simulated transaction schema
  has no credential field.
- **No phishing infrastructure.** No sending capability, no page hosting, no
  templating against a real brand, no recipient list, no delivery mechanism of any
  kind. The lure vocabulary described above is inert text held in a Python list.
- **No one-time-passcode interception.** The `otp_relay` family models the
  *consequence* of a relayed step-up — a transaction that arrives carrying
  `is_3ds` and `otp_verified` flags despite being fraudulent — because that is what
  a defender sees. It models nothing about how a code would be obtained.
- **No interaction with real payment services.** There is no payment integration,
  no acquirer or issuer connection, no card network client, no test-mode gateway.
  A search of `src/`, `app/` and `config.py` for HTTP clients, sockets or cloud
  SDKs returns nothing; the only outbound network call anywhere in the project is
  `LLMClient._complete_raw` in `src/llm/client.py`, which calls the Anthropic
  Messages API and nothing else.
- **No automation against real accounts.** There is no browser automation, no
  credential stuffing, no session handling, no scraping. The "accounts" in this
  system are rows produced by `src/generate/profiles.py`.
- **No bypassing of real security controls.** The adversarial families model
  evasion of *this project's own model*, measured against *this project's own
  synthetic data*. Nothing is tested against a third-party control, and no entry in
  the Threat Atlas describes a bypass technique.

Everything the system does is synthetic, local and sandboxed. A full pipeline run
(`python -m src.pipeline`) reads and writes only files inside the repository, and
completes without any network access at all.

---

## Data protection

**All data is synthetic and generated by the code in this repository.** No real
cardholder data, no real personally identifiable information and no proprietary
data of any kind is used anywhere. The cardholders, merchants, devices and
transactions are produced by `src/generate/profiles.py` and
`src/generate/base_generator.py` from `config.GLOBAL_SEED`, and the whole dataset
can be regenerated from scratch with one command. The merchant names that appear in
the benign message templates are well-known consumer brands used as filler in
plainly synthetic notification text; no data about them is used, and no
relationship with them is implied.

Secrets are handled by keeping them out of the repository entirely. `.gitignore`
excludes `.env` and `.streamlit/secrets.toml`, and the only committed
configuration example is `.env.example`, which contains a placeholder value rather
than a key. `config.py` reads `ANTHROPIC_API_KEY` from the environment, optionally
loading `.env` through `python-dotenv` if it is installed, and exposes
`config.llm_available()` so that every caller can branch on availability instead of
assuming a key exists.

The important property is that **no API key is required for any demo path.**
`src/llm/client.py` caches every response to `artifacts_cache/` keyed by a hash of
model, system prompt, prompt and schema tag, and the committed cache plus the
committed seeded dataset make the pipeline and the web application fully runnable
offline. `cached_only()` exists specifically so callers can read a cached result
with a guarantee of never touching the network. When a live call is genuinely
needed and no key is present, `_complete_raw` raises `LLMUnavailable` and every
caller in `src/generate/llm_agent.py` catches it and falls back to a deterministic
offline path. The `anthropic` package is imported lazily inside that function, so
an offline run does not even need the dependency installed.

One script deliberately refuses to degrade: `src/generate/demo_specs.py`, which
exists to demonstrate the live model path end to end, will not write its artifact
without a key. Its docstring gives the reason — a demonstration of the generative
path that was actually produced by the offline heuristic would be worse than having
no demonstration at all. That is a reporting-integrity decision as much as a
technical one, which brings us to the next section.

---

## Responsible reporting

Results from a synthetic laboratory are easy to overstate, and overstating them is
the most likely way this project could cause harm. The project therefore makes six
commitments, each of which is implemented somewhere rather than merely asserted.

**No claim of validation on real payment data.** Every number in this project is a
simulation result. The README says so in its Limitations section, and the licence
and attribution section states that nothing here has been reviewed or validated by
Mastercard.

**No implication of endorsement by any payment network.** The project was built for
a Mastercard innovation challenge and says so, and in the same breath states that
nothing in it has been reviewed, validated or endorsed by Mastercard and that no
part of it describes any real payment-network system. The decisioning layer carries
the same disclaimer in its own module docstring
(`src/defend/decision_policy.py`), because a risk-based decision policy is exactly
the component a reader might mistake for a description of a real one.

**Residual frontiers are published rather than hidden.** The attacks the defense
does not solve — the families that sit on the legitimate behavioural centroid, and
the first-party abuse that is barely observable at authorization time — are written
to `models/blind_spots.json`, ranked, with the hardest family named and carried
forward as the next red-team target. They are reported in the README rather than
being quietly excluded from the headline.

**Rejected models are labelled.** `src/defend/governance.py` evaluates every
adapted model as a challenger against the model in force and returns `PROMOTE` or
`REJECT` with the reasoning for every gate. The outcome is written to
`models/model_registry.json`, where each entry carries its stage, and the registry's
own note records that a rejected candidate stays out of the authorization path and
that a real deployment would additionally require a shadow rollout this prototype
does not simulate. Rejections are part of the record, not a failed run to be
re-rolled.

**Sample sizes and uncertainty are attached to quoted results.** Per-family recall
is only reported when enough held-out examples of that family exist:
`config.FAMILY_EVAL["min_n_to_report"]` sets the floor, `_per_attack_recall` in
`src/defend/evaluate.py` records `n` and a `sufficient_n` flag beside every figure,
and `wilson_interval` attaches a 95 per cent Wilson score interval to each. The
promotion gates use the same interval machinery in `governance._wilson`. Family
figures are measured on a dedicated fraud-enriched frame generated from a seed no
model was trained on, and `models/family_recall.json` carries a note stating that
precision and false-positive figures are never quoted from that enriched frame but
from the realistic-base-rate test slice instead.

**Numbers are generated, never typed.** The README, the figures and the submission
document are rendered from `models/*.json` and `data/summary.json` by
`python -m docs.build_docs`, and `tests/test_artifact_consistency.py` fails the
build if two documents disagree about the same number. This document contains no
metric values for the same reason: hand-written prose about measured results goes
stale, and stale prose that contradicts the generated numbers is a reporting
failure. For results, read [the README](../README.md#headline-results) and the
artifacts in `models/`.

---

## Limitations that matter ethically

**Synthetic data cannot establish real-world efficacy.** This is the limitation
that governs every other claim. The simulator was built with considerable care to
avoid handing the model artifacts to learn instead of fraud — fraud actors build
ordinary transaction history first, attacks reuse cards and merchants, fraudulent
traffic rides the same weekly and payday shape as genuine spend, all documented at
the top of `src/generate/attack_injectors.py` — and `src/generate/fidelity.py`
exists to measure how far that succeeded. None of that makes the result
transferable. A real deployment requires a labelled backtest on issuer data before
any figure in this project means anything about the world, and the project says so
in its Limitations section rather than in a footnote.

**A fraud model that challenges genuine customers has a real human cost.** A false
positive is not a rounding error in a confusion matrix. It is a card declined at a
checkout, a transfer held, a customer stepped up and, some fraction of the time, a
customer who abandons the payment entirely. This is why detection is never reported
alone in this project. `models/operational_metrics.json` reports the share of
genuine customers stepped up and declined, and translates the false-positive rate
into monthly review volume, review hours and an estimate of genuine customers
abandoning after a step-up, using the assumptions in
`config.OPERATIONAL_SCENARIO`. Those assumptions are labelled in the config and in
the artifact itself as a synthetic illustrative scenario and not production
economics, precisely so that a cost figure is never mistaken for a claim about
anyone's real business. The promotion gates in `config.CHAMPION_CHALLENGER` encode
the same priority: a candidate model may be rejected for worsening the
false-positive rate even when it catches more of the new attack, because a defense
that buys detection with customer friction has not obviously improved.

**Adaptive systems need a human in the loop.** An adaptive defense that deploys
itself is not deployable, which is the opening line of the governance module. The
loop in this project ends at a gated decision, not at a deployment. That boundary
is deliberate and should not be removed by anyone extending this work.

---

## Intended and unintended use

This project is intended for fraud-detection engineers and researchers, risk and
model-governance reviewers, judges and reviewers assessing this submission, and
students of adversarial machine learning who want a complete, runnable example of a
red-team and blue-team loop with governance attached. It is useful as a study of
architecture and evaluation methodology — how to keep a generative red team
constrained, how to measure what a defense actually depends on, how to decide
whether an adapted model may ship — and not as a source of detection performance
figures for any real portfolio.

**It must not be used to develop, test or refine attacks against systems the user
does not own.** Nothing in this repository provides that capability, and the design
decisions described above exist to keep it that way, but the intent should be
stated plainly regardless. The Threat Atlas is a defender's reference. The
simulator generates rows in a local file. Any attempt to repurpose either against a
real payment system, a real merchant, a real institution or a real person is
outside the intended use of this project and, in most jurisdictions, outside the
law.

If you extend the project, the boundary to preserve is the constraint layer. The
language model may propose; `validate_spec` disposes; only deterministic, local,
sandboxed code touches data. A contribution that lets model output reach a
simulator, a message, a network call or a customer without passing through a
validating layer is the change this document exists to prevent.

---

## Where to go next

For how the system is put together, read [Architecture](ARCHITECTURE.md) and then
[Design](DESIGN.md), which explains why each abstraction — the attack
specification, the constraint layer, the weakness report, the promotion gates —
exists at all. For the schemas referenced throughout this document, including the
full Threat Atlas entry schema with `ethical_notes` and `simulator_status`, read
[Data model](DATA_MODEL.md). For how each result is produced and how to reproduce
it, read [Experiments](EXPERIMENTS.md). For measured results, read
[the README](../README.md) and the artifacts in `models/`, which are the only place
in this project where a number is allowed to live.
