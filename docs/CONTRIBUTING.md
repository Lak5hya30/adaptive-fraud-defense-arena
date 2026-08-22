# Contributing

*This document is the developer guide for the AI Defense Lab. It is written for a new contributor about to make their first change, and for a reviewing engineer who wants to know how the repository is kept honest. After reading it you will know how to install and run the project, which command to run after which kind of change, what every test file protects and how long it takes, how to add an attack family, a feature, a Threat Atlas entry, a governance gate or an application page without breaking anything downstream, which conventions the existing code follows, which rules may never be broken, and what to do when something fails. It contains no measured results by design — every number lives in [`README.md`](../README.md) and in the generated files under `models/`, so this text cannot drift out of agreement with them.*

---

## 1. Environment setup

The project is a plain Python package with no build step, no service dependencies and no database. Everything it needs is in `requirements.txt`, and everything it produces is a file inside the repository. That is deliberate: a competition prototype that needs infrastructure to demonstrate is a prototype that will not demonstrate.

Python 3.11 or later is expected. `requirements.txt` marks `shap` as `python_version < "3.14"` because the explainability arm is optional and the code degrades gracefully when it is absent; nothing else is version-gated. The committed artefacts in this repository were produced on CPython 3.13.

```bash
python -m pip install -r requirements.txt
```

An API key is optional and is only needed for the live generative path. Copy `.env.example` to `.env` if you want it:

```bash
cp .env.example .env      # then set ANTHROPIC_API_KEY
```

Without a key everything still runs. `src/llm/client.py` caches every model response to `artifacts_cache/` keyed by a hash of model, system prompt, prompt and schema tag, and the committed cache is what makes the offline path reproducible. `config.llm_available()` is the single place that decides whether a live call is possible; it reads `ANTHROPIC_API_KEY` and the `FRAUD_LAB_LLM_ENABLED` override.

### The commands

| Command | What it does |
|---|---|
| `streamlit run app/Home.py` | Launch the prototype at `http://localhost:8501`. Demo mode is the default and renders entirely from committed artefacts — no key, no network, no training. |
| `python -m src.pipeline` | Regenerate every artefact deterministically from `config.GLOBAL_SEED`, including the closed loop. |
| `python -m src.pipeline --fast` | The same, but skips step 6 (the closed loop), which is by far the slowest stage. |
| `python -m docs.build_docs` | Regenerate `README.md`, `docs/figures/*.png`, the four generated guides and `docs/solution_walkthrough.docx` from the committed artefacts. |
| `python -m pytest tests/ -q` | Run the full test suite. |
| `./run.sh` | Pipeline, then documentation, then launch. `./run.sh --fast` passes `--fast` through to the pipeline; `./run.sh --launch` skips straight to the app. |
| `python -m src.generate.demo_specs --check` | Report whether the live generative path is available, without calling anything. |

`.streamlit/config.toml` pins the application to a dark theme with a fixed palette. This is not decoration: the custom cards defined in `app/common.py` draw light text on a translucent panel, so under a light theme they would render white-on-white. The theme is pinned so the prototype looks identical on any machine and any projector.

`.claude/launch.json` describes the same Streamlit invocation for tooling that launches the app automatically. It is gitignored and is not required to run anything.

### What a pipeline run actually does

`src/pipeline.py` runs eight numbered steps and prints each one. In order: simulate the labelled dataset and the text corpus, run the fidelity diagnostics, train the transaction defence, train the text arm, run the leave-one-attack-family-out experiment, run the closed loop, run the rules-versus-static-versus-adaptive comparison, and run the threshold, calibration, operational and blind-spot diagnostics. It finishes by writing `models/pipeline_summary.json`.

The ordering carries a decision worth knowing before you change it. Leave-one-out runs *before* the loop on purpose, because it is the experiment that establishes which families are learnable at authorisation time at all, and `src/loop/redteam_loop.unlearnable_families()` reads `models/leave_one_out.json` to decide which families the red team is allowed to target. Move the loop earlier and the exclusion set silently falls back to the hardcoded `STRUCTURAL_FRONTIERS` constant, which is the fallback for a fresh clone rather than the measured answer.

---

## 2. The development loop

The rule that matters more than any other in this repository: **anything that changes an artefact requires regenerating both the pipeline and the documentation, in that order.**

This is not bureaucracy. `README.md` and the four generated guides under `docs/` contain no hand-typed figures — `docs/build_docs.py` and `docs/build_guides.py` render them from `models/*.json` and `data/summary.json`. If you regenerate the pipeline and stop, the committed documents describe the previous run, and `tests/test_artifact_consistency.py` will fail on the mismatch. If you regenerate the documentation without first regenerating the pipeline, you have rendered the old artefacts a second time and changed nothing. Either half alone leaves the repository in a state where two committed files disagree about the same number, which is the fastest way there is to lose a technical reviewer: once one table contradicts another, every other figure becomes suspect.

| What you changed | What to run |
|---|---|
| The simulator, an injector, a spec, the default mix, or `config.py` simulation settings | `python -m src.pipeline` then `python -m docs.build_docs` |
| A feature in `src/defend/features.py`, or the model | `python -m src.pipeline` then `python -m docs.build_docs` |
| A governance gate or its thresholds | `python -m src.pipeline` then `python -m docs.build_docs` (the loop's promotion decisions change) |
| An entry in `src/identify/attacks.json` | `python -m docs.build_docs` (the catalogue counts appear in the README and the atlas figure) |
| Wording or layout in `docs/build_docs.py` or `docs/build_guides.py` | `python -m docs.build_docs` |
| A Streamlit page or `app/common.py` | Nothing to regenerate; reload the app |
| A test | `python -m pytest` on the file you touched |
| A hand-maintained document such as this one | Nothing |

`--fast` exists for iteration, not for committing. It skips the closed loop, so `models/loop_history.json`, `models/attack_lineage.json`, `models/model_registry.json`, `models/hero_example.json` and the promoted champion model are left at whatever the previous full run produced. Use it while you are working on the simulator or the features; run the full pipeline before you commit anything that lands in a document.

Individual stages can also be run on their own while iterating, each of which writes its own artefacts: `python -m src.generate.simulate`, `python -m src.generate.fidelity`, `python -m src.defend.train`, `python -m src.defend.baseline`, `python -m src.defend.diagnostics`, `python -m src.experiments.leave_one_out`, `python -m src.loop.redteam_loop`. Running stages piecemeal is fine for exploration and dangerous for committing, because the artefacts then come from different runs. `tests/test_artifact_consistency.py::test_baseline_agrees_with_metrics_on_the_static_model` exists precisely to catch that.

---

## 3. The test suite

Seven files, roughly eighty tests, about twelve minutes end to end on a laptop. The suite is offline: nothing in it touches the network, and the one file that exercises the generative path substitutes a stub for the client.

| File | Tests | Approximate time | What it protects |
|---|---|---|---|
| `tests/test_artifact_consistency.py` | 15 | under 5 seconds | Committed artefacts agreeing with each other and with the generated documents. Skips cleanly when an artefact has not been generated. |
| `tests/test_attack_spec_and_loop.py` | 17 | under 5 seconds | The attack-specification contract, the payment-domain constraint layer, weakness-driven mutation, and the promotion gates. Pure functions only. |
| `tests/test_genai_spec_path.py` | 7 | under 5 seconds | The live specification path either side of the network boundary, with a stub client. |
| `tests/test_smoke.py` | 7 | about 1.5 minutes | End to end wiring: taxonomy, generate, features, train, score, evade. Also threshold tuning against its budget and the monotonicity of calibration. |
| `tests/test_data_quality.py` | 8 | about 1.5 minutes | Authorisation-time leakage, legitimate-data realism, and the temporal causality of historical features. |
| `tests/test_shortcuts_and_fidelity.py` | 18 | about 2 minutes | The anti-shortcut guards, leakage, causality of the network counters, and reproducibility. |
| `tests/test_defense_and_loop.py` | 9 | about 6.5 minutes | The rules baseline, the tiered decision policy, and one real end-to-end run of the adaptive loop at reduced scale. |

`test_adaptive_loop_end_to_end` alone accounts for most of the total. It runs two genuine loop rounds with real simulation and real retraining at a fraction of the committed scale, because a loop test that mocks the loop tests nothing.

### The two files that carry the project's credibility

Most test suites protect behaviour. Two of these protect a claim, and they are the ones to read first and to be most careful about weakening.

**`tests/test_shortcuts_and_fidelity.py` protects the claim that the synthetic data is worth training on.** The failure mode it exists for is a detector that learns the generator instead of the fraud. Two real shortcuts existed in this simulator and were removed; both now have a test so they cannot return. `is_new_payee` once fired on almost all fraud and about half of genuine traffic, because every fraudulent row happened to be a first-ever card and merchant pairing. The velocity features were dead — identically one on the two families whose entire definition is burst behaviour — because each probing transaction minted a fresh card identifier. The tests are deliberately two-sided: `test_binary_features_are_not_one_sided_shortcuts` asserts both that a flag is not nearly always true on fraud *and* that it is not nearly always false on genuine traffic, because asserting only the second of those is exactly what let the first shortcut survive. If one of these fails, the correct response is to fix the simulator, never to relax the assertion.

**`tests/test_artifact_consistency.py` protects the claim that the documents describe this run.** It checks that `models/baseline_comparison.json` and `models/metrics.json` describe the same model on the same split, that every artefact carries the current `config.SCHEMA_VERSION`, that the README quotes the committed recall and false-positive rate, that the README's catalogue counts match `src/identify/attacks.json`, that the shipped adaptive column is the *promoted* champion and not a rejected candidate, that every per-family recall ships with its sample size, that the headline family clears the sample-size floor in `config.FAMILY_EVAL["min_n_to_report"]`, that the text arm's score never appears without its caveat, and that no generated document makes an unqualified claim of validation on real data. Every one of those corresponds to a way this project could mislead a reader without lying about a single individual number.

### Which subset to run

| Change | Run |
|---|---|
| An injector, the mix, the base generator, or `config` simulation settings | `python -m pytest tests/test_shortcuts_and_fidelity.py tests/test_data_quality.py -q` |
| A feature in `src/defend/features.py` | `python -m pytest tests/test_shortcuts_and_fidelity.py tests/test_data_quality.py tests/test_smoke.py -q` |
| An attack specification, a family constraint, or mutation logic | `python -m pytest tests/test_attack_spec_and_loop.py tests/test_genai_spec_path.py -q` |
| A governance gate | `python -m pytest tests/test_attack_spec_and_loop.py -q` |
| The loop itself | `python -m pytest tests/test_defense_and_loop.py -q` (allow six or seven minutes) |
| `src/identify/attacks.json` or the taxonomy loader | `python -m pytest tests/test_smoke.py::test_taxonomy_loads_and_validates tests/test_artifact_consistency.py -q` |
| Anything, before committing | `python -m pytest tests/ -q` |

Every test file also has a module docstring naming the single command that runs it, so the answer is always in the file itself.

---

## 4. Recipes

Each recipe lists every file that must change. Skipping one of them generally does not raise an error immediately; it produces an inconsistency that a test or a reviewer finds later.

### 4.1 Adding a new attack family

A family is not one function. It is an injector, a generation-zero specification, an optional constraint set, a place in the default mix, and at least one catalogue entry that points at it. The tests enforce the wiring in both directions, so a partial addition fails fast.

1. **`src/generate/attack_injectors.py` — the injector.** Write a function with the signature `f(holders, merchants, rng, n, spec=None)` returning a list of row dictionaries in the `COLUMNS` schema from `src/generate/base_generator.py`. Start from `_row(attack_type)`, which prefills neutral defaults, and override only the fields your family actually determines. Everything must be deterministic given the passed `rng`; never create your own generator and never read the clock. Two rules govern the content itself, both documented at the top of the module. Fraud actors must have a history, emitted through `_cover_row()` with `is_fraud=0` and `actor_role=ROLE_COVER`, because at authorisation time that traffic genuinely is not fraud and labelling it otherwise hands the model a label it could not earn in production. Attacks must reuse things — the same card, the same merchants — or every fraudulent row is trivially a first-ever pairing and the defender learns that artefact instead of fraud behaviour. Use the shared helpers (`_geo_for`, `_device_for`, `_merchant_for`, `_channel_for`, `_amount`, `_account_age`, `_pick_day`) rather than reimplementing them; each encodes a fidelity decision, and `_account_age` in particular exists so fraud and genuine traffic are aged on the same clock.
2. **`src/generate/attack_injectors.py` — register it.** Add the function to the `INJECTORS` dictionary.
3. **`src/generate/attack_injectors.py` — the default mix.** Add a weight to `DEFAULT_MIX`. The weights are relative and the simulator normalises by their sum, but they are written to total one, so adjust the neighbours rather than letting the total drift.
4. **`src/generate/attack_spec.py` — the generation-zero specification.** Add an `AttackSpec` to `BASE_SPECS` under the same key. `tests/test_attack_spec_and_loop.py::test_every_family_has_a_generation_zero_spec` fails without it. Set `source="fixed"` and write the `strategy` string as a sentence a reader can check against the injector. Think about what generation zero should *not* do: `account_takeover`, for example, is deliberately not a foreign-geography attack, because making it one would hand the detector a geography tell and let the family post a recall that has nothing to do with detecting takeover.
5. **`src/generate/attack_spec.py` — family constraints, if the family has any.** Add an entry to `FAMILY_CONSTRAINTS` naming, per dial, the set of values that are possible on a real rail. This is the layer that stops the red team executing something impossible: an authorised push payment cannot run from an attacker's device because the genuine customer is the one authenticating, so `scam_transfer` constrains `device_behavior` to `trusted_device`. Ten of the eleven current families carry constraints. A family with none is a family that any dial combination can legally describe, which is a claim worth being sure about.
6. **`src/identify/attacks.json` — at least one catalogue entry.** Give it `simulator_status` of `IMPLEMENTED` and set `maps_to_injector` to your injector's key. Add the key to the top-level `injectors` list in the same file, or the loader raises. See recipe 4.3 for the full field list.
7. **Tests.** `tests/test_smoke.py::test_simulation_is_labeled_and_covers_all_injectors` asserts that every registered injector actually appears in a simulated frame, so a family with a mix weight too small to produce rows at the test's scale will fail there. `tests/test_attack_spec_and_loop.py::test_mutation_never_makes_the_attack_louder` and `test_mutation_output_survives_the_constraint_layer` iterate over every entry in `BASE_SPECS`, so your family is covered automatically — but if it has an unusual dial combination, check it passes rather than assuming. If the family is a plausible catastrophic-forgetting control, consider adding it to `GUARD_CANDIDATES` in `src/loop/redteam_loop.py`.
8. **Regenerate.** Full pipeline, then documentation. A new family changes the fraud mix, which changes every downstream number.

Two judgement calls are worth making explicitly. If the family's fraud is authorised by the genuine customer on their own device, expect low recall and say so — `friendly_fraud` and `scam_transfer` are in the simulator precisely so that the honest answer can be measured rather than asserted. And if the family cannot be caught even when the model is trained on it directly, `unlearnable_families()` will exclude it from red-team targeting automatically once the leave-one-out artefact is regenerated; that exclusion is measured evidence, not a hardcoded opinion, and should be left to the measurement.

### 4.2 Adding a new authorisation-time feature

The feature contract in `src/defend/features.py` is the most load-bearing thing in the repository, because every detection claim depends on the model not having seen anything it could not see in production. There are currently 36 columns in `FEATURE_COLUMNS`, six of which are the network-level relational counters in `NETWORK_FEATURES`.

1. **`src/defend/features.py` — declare it.** Add the name to `FEATURE_COLUMNS`. The list is the contract: `build_features()` returns exactly these columns in exactly this order, and `tests/test_smoke.py::test_features_are_finite_and_complete` asserts it.
2. **`src/defend/features.py` — compute it.** Per-card historical features are computed in the `card_id`-sorted block using `expanding().shift()`, `cummax().shift()`, `groupby().cumcount()` or a trailing rolling window. Relational counters belong in `_network_features()`, which works on a globally time-ordered frame and reindexes back to the caller's row order. Every one of those constructions is chosen so that a row depends only on rows strictly before it.
3. **Decide whether it is causal, and prove it.** A historical or relational feature must not change when later rows are appended. `tests/test_data_quality.py::test_historical_features_use_only_the_past` checks this for the per-card features and `tests/test_shortcuts_and_fidelity.py::test_network_counters_only_look_backwards` checks it for the counters, by computing features on a prefix and on the whole frame and demanding the prefix rows are identical. If your feature needs a `.shift()` you have almost certainly got it right; if it needs a global aggregate over the whole frame, it is not causal and does not belong here.
4. **Decide whether it belongs in the network group.** A feature belongs in `NETWORK_FEATURES` if it is keyed on something other than the card — a device, an IP prefix, a merchant — and describes a relationship a payment network can see and a single issuer or merchant cannot. Those are the signals that make card testing, mule rings and transaction laundering visible as network patterns rather than as "this card is unknown to us". If it is a raw count, add it to `_COUNT_FEATURES` as well, so it is log-compressed before it reaches the model: counters grow without bound as the simulation window fills, and on a raw scale a model tuned on earlier traffic meets a systematically different distribution later, which drifts the threshold off its false-positive budget for reasons that have nothing to do with fraud.
5. **Choose an honest default for missing history.** Every fill in `build_features()` is annotated with why it is the honest value. A merchant's first transaction has no prior new-card ratio, so it is filled with the midpoint rather than a value that implies safety; a card with no prior maximum gets zero, meaning "no evidence this is unusual" rather than an implied alarm. Do not fill with a value that leans toward either class.
6. **Check it is not a shortcut.** Run the anti-shortcut file. If your feature separates the classes too cleanly, `test_no_single_feature_separates_fraud_from_genuine` fails against `SEPARABILITY_FAIL` in `src/generate/fidelity.py`, and if it is binary and one-sided, `test_binary_features_are_not_one_sided_shortcuts` fails. Both mean the same thing: you have built a label, not a signal.
7. **Consider the rules baseline.** `RULES` in `src/defend/baseline.py` is deliberately competent rather than a strawman — it includes the network-level rules a fraud team would write once it had the same counters the model gets. If your feature gives the model a capability a rule could express with a single round, intuitive threshold, add the rule too, or the comparison starts measuring access to a feature rather than what machine learning adds on top of good rules. Rule thresholds must stay round, domain-chosen numbers and must never be fitted to this dataset. Note that the counters reach the rules log-compressed, so their thresholds are written as `np.log1p` of the count an analyst would state.
8. **Regenerate.** Full pipeline, then documentation.

Never add a feature computed from `POST_OUTCOME_COLUMNS` or `ORACLE_COLUMNS`. `assert_auth_time_safe()` runs inside `build_features()` and will raise, and two separate tests confirm the guard itself still fires. The one legitimate use of a post-outcome field is `card_prior_dispute_rate`, which is the mean of `refund_flag` over the card's *earlier* transactions only, shifted — a past outcome known before the current authorisation, which is a different thing from this transaction's outcome.

### 4.3 Adding a Threat Atlas entry

`src/identify/attacks.json` is the single source of truth for the catalogue. `src/identify/taxonomy.py` loads it into frozen `Attack` dataclasses and validates it on load, so a malformed entry raises at import time rather than showing up as a wrong number in a figure.

Every entry carries: `id`, `name`, `category`, `subcategory`, `rails`, `channel`, `attacker_objective`, `genai_role`, `genai_mechanism`, `kill_chain`, `transaction_signature`, `behavioral_signature`, `observable_signals`, `auth_time_observability`, `post_transaction_signals`, `defense_difficulty`, `expected_impact`, `severity`, `novelty_score`, `real_world_grounding`, `ethical_notes`, `simulator_status`, `maps_to_injector` and `simulatable`. The loader ignores unknown keys, so a typo in a field name silently drops the value — check your entry with `python -m src.identify.taxonomy`, which prints the summary counts and the coverage table.

`simulator_status` takes exactly four values, defined in the file's own `status_definitions` block and mirrored in the loader:

| Status | Meaning | Must name an injector |
|---|---|---|
| `IMPLEMENTED` | A dedicated injector reproduces this attack's authorisation footprint, and it is in the default simulated mix | Yes |
| `PARAMETERIZED` | Reachable today by configuring an existing injector through the specification dials, without new code | Yes |
| `RESEARCH_ONLY` | Catalogued and characterised, but not simulated | No — must be `null` |
| `FUTURE` | Named as planned simulator work | No — must be `null` |

`_validate()` enforces, and will raise on: duplicate identifiers, a `maps_to_injector` that is not in the file's own `injectors` list, a `novelty_score` outside zero to one, an unrecognised `simulator_status`, an `IMPLEMENTED` or `PARAMETERIZED` entry with no injector, a `RESEARCH_ONLY` or `FUTURE` entry that claims one, and an unrecognised `auth_time_observability` (which must be one of `high`, `partial`, `low`, `none`). `tests/test_artifact_consistency.py::test_catalog_never_claims_to_simulate_what_it_does_not` checks the same invariants against the committed file, and `tests/test_smoke.py::test_taxonomy_loads_and_validates` additionally requires that at least one entry is marked research-only, on the grounds that a catalogue claiming to simulate everything is not being honest about its own breadth.

Two content rules apply to what you write. The catalogue describes defender-observable consequences only: no operational guidance, no tooling and no bypass technique appears anywhere in the file, and `ethical_notes` is where you record that boundary for a sensitive entry. And the catalogue is deliberately wider than the simulator — that is the point of the status field, so research breadth is never presented as simulation breadth. If you are not going to write an injector, `RESEARCH_ONLY` is the honest status and costs the project nothing.

After editing, run `python -m docs.build_docs`: the counts appear in the README's Threat Atlas section and in `docs/figures/atlas_coverage.png`, and `tests/test_artifact_consistency.py::test_readme_and_catalog_agree_on_attack_counts` will fail until you do. The `provenance` block at the top of the file records how many entries were authored and how many were merged as duplicates; update it if you change the entry count materially.

### 4.4 Adding or changing a governance gate

Gates live in `src/defend/governance.py` and their thresholds in `config.CHAMPION_CHALLENGER`. The design constraint is stated at the top of the module: an adaptive defence that deploys itself is not deployable. Every model the loop produces is a challenger measured against the model in force, and a candidate that fails any gate is recorded with its reason and not promoted — so "the loop keeps improving the defence" stays a claim the artefacts are able to contradict.

To add a gate, append a `_gate(name, passed, observed, required, detail)` call to the `gates` list inside `evaluate_candidate()`, and add its threshold to `config.CHAMPION_CHALLENGER` rather than writing a number inline. The decision is derived, not assigned: any failed gate makes the decision `REJECT`. Write the `detail` string as a sentence a non-specialist can read, because it is what appears in the model registry and in the application's governance view.

Three properties are worth preserving. Gates must tolerate missing inputs — every existing gate is guarded by a `None` check, so a caller that cannot measure PR-AUC still gets the gates it can evaluate. Gates on small samples must account for sampling noise: the forgetting gate judges a regression by the upper bound of a Wilson interval on the candidate's family recall rather than by the raw drop, because family recall is a proportion measured on a few dozen transactions and a gate that stops a genuinely better model for a coin-flip is worse than no gate. And a gate must be able to fail — `tests/test_attack_spec_and_loop.py` has one test per existing gate that constructs a candidate designed to trip it, and a new gate should get one too.

Changing a threshold in `config.CHAMPION_CHALLENGER` changes which rounds promote, which changes the champion model, which changes the adaptive column in the benchmark table and the promotion sentence in the README. Run the full pipeline and rebuild the documentation. `models/model_registry.json` records the gate set alongside every entry, so the registry from an old run tells you what the gates were at the time.

### 4.5 Adding a Streamlit page

Pages live in `app/pages/` and Streamlit orders them by filename, so the numeric prefix is the navigation order. Copy the shape of an existing page — `app/pages/6_Benchmarks.py` is a good short example.

1. Create `app/pages/N_Name.py`. The visible page name comes from the filename, so choose it deliberately.
2. Open with a module docstring stating what question the page answers. Every existing page does this, and it is the fastest way for a reviewer to decide whether the page is worth reading.
3. Import from `common` — not from `app.common`. `common.py` bootstraps the project root onto `sys.path` by walking up until it finds `config.py`, which is what lets a page import `config` and `src.*` while Streamlit runs it as a top-level script.
4. Call `page_setup(title, icon)` first. It sets the page config and injects the shared CSS for `.card`, `.pill` and `.kicker`.
5. Read data through the cached loaders in `common.py` — `load_metrics`, `load_baseline`, `load_loao`, `load_loop_history`, `load_family_recall`, `load_lineage`, `load_registry` and the rest. They are `st.cache_data`-wrapped readers of the committed artefacts. Never recompute features or train anything on a page: a demo that recomputes on stage is a demo that stalls, which is why `src/defend/diagnostics.py` precomputes `models/defend_demo.parquet` with scores, actions and reason codes for the held-out split.
6. Handle the missing-artefact case. If the page's artefact has not been generated, call `artifact_missing(name, command)` or `st.info(...)` and `st.stop()`, naming the exact command that produces it. A page that raises on a fresh clone is a page that makes the project look broken.
7. Use `PALETTE` for colour and `STRETCH` for full-width elements. The palette is colour-blind safe and matches the figures generated by `docs/build_docs.py`; `STRETCH` exists to avoid the deprecated width keyword.
8. Call `mode_selector()` if the page has a demo and a live mode.

Pages read artefacts and never write them, so adding one requires no regeneration.

---

## 5. Code style

The conventions here are not enforced by a linter. They are visible in every module, and the reason to follow them is that this repository is read by reviewers at least as often as it is run.

**Module docstrings explain intent, not contents.** Open `src/defend/features.py`, `src/generate/attack_spec.py` or `src/defend/governance.py` and the docstring tells you what problem the module solves and what would go wrong without it, often with a small ASCII diagram of the flow. A docstring that lists the functions below it adds nothing a reader could not get by scrolling.

**Comments state constraints, not narration.** The useful comment in this codebase is the one that says why a line is the way it is and what breaks if it changes. In `src/generate/attack_injectors.py`, the comment above the bust-out day selection explains that the day of the bust is chosen first and the grooming period worked backwards, because choosing the start day first would push nearly every bust-out into the last weeks of the window and quietly starve a chronological split of the family. That comment is worth more than the code it sits above. A comment that says what the next line does is noise.

**Write the reason for a design choice next to it.** The pattern runs throughout: the fill values in `build_features()` each carry the sentence explaining why that default is the honest one; `_weighted_pick()` explains that sampling attack merchants uniformly while genuine spend follows a heavy-tailed popularity curve would make "quiet merchant" a fraud signal; `LOOP_CONFIG["replay_oversample"]` in `config.py` notes that too high a value over-fits the newest generation and regresses overall ranking quality, which the promotion gates catch. When you make a decision that a future reader might reverse for a plausible-sounding reason, record the reason it was made.

**Naming.** Modules and functions are lowercase with underscores; a leading underscore marks a module-private helper. Module-level constants are uppercase — `FEATURE_COLUMNS`, `INJECTORS`, `DEFAULT_MIX`, `BASE_SPECS`, `RULES`, `SIGNAL_TO_DIAL`. Configuration dictionaries in `config.py` are uppercase and grouped under a banner comment describing the phase they belong to. Names are spelled out: `merchant_new_card_ratio_prior`, not `mnc_ratio`. The `_prior` suffix on a feature name is a promise that it is computed from strictly earlier rows, and the `_zscore`, `_vs_card_max_prior` and `_share_prior` suffixes describe what the value is relative to. Attack family keys, injector function names, `BASE_SPECS` keys, `DEFAULT_MIX` keys and `maps_to_injector` values are all the same string, and keeping them identical is what lets the tests check the wiring.

**Type hints and `from __future__ import annotations`.** Every module opens with the future import and uses modern annotation syntax. Public functions are annotated; short private helpers often are not, which is fine.

**Determinism.** Any function that samples takes an `rng` argument and uses it. No module reads the clock, generates a random seed, or depends on dictionary ordering that is not explicitly sorted. `src/defend/governance.py` deliberately writes no wall-clock timestamp into registry entries by default, because committed artefacts have to stay byte-stable across runs; a deployment that wants real timestamps passes `trained_at` explicitly.

---

## 6. Rules that must not be broken

These are not style preferences. Each one corresponds to a way this project could produce a number that looks good and means nothing.

**No post-outcome or oracle field may become a feature.** `refund_flag` and `auth_result` are known only after a transaction settles; `is_fraud`, `attack_type` and `actor_role` are simulator bookkeeping that no real defender has. They stay in the raw dataset for analysis, and `assert_auth_time_safe()` blocks them from the feature matrix. The prior-dispute-rate feature is the only legitimate use of a post-outcome column, and only because it is shifted to the card's earlier transactions.

**Every historical feature must be causal.** A row's features must be identical whether or not later rows exist. This is the property that makes the whole evaluation meaningful, and it is the easiest thing in the repository to break by accident — particularly in the relational counters, which are computed across cards rather than within one.

**No hand-edited numbers in generated documents.** `README.md`, `docs/JUDGE_QA.md`, `docs/DEMO_SCRIPT_90S.md`, `docs/PITCH_3MIN.md`, `docs/WHY_WE_WIN.md` and `docs/solution_walkthrough.docx` are outputs. Edit `docs/build_docs.py` or `docs/build_guides.py` and regenerate. A number typed into a generated file survives exactly until the next build, and in the meantime it is a number nothing checks.

**No result is quoted without its sample size.** Per-family recall is a proportion measured on a few dozen held-out transactions. Every table that prints one prints `n` and a 95% Wilson interval beside it, and `config.FAMILY_EVAL["min_n_to_report"]` is the floor below which a family may be shown but never used as a headline. Two tests enforce this, one on the artefacts and one on the generated documents.

**Nothing claims validation on real data.** All data here is synthetic. Nothing has been validated on real payment data, and nothing has been reviewed or validated by Mastercard. `tests/test_artifact_consistency.py::test_no_misleading_validation_language_in_the_docs` scans the generated documents for the specific phrases that would constitute such a claim and permits them only in a question or an explicit denial.

**The text arm is a sanity check, not a result.** Its corpus is composed from a fixed slot vocabulary, so the two classes separate on vocabulary alone. Its score is keyed in `models/pipeline_summary.json` under a name that cannot be mistaken for a detection metric and ships with its own caveat, and a test enforces both. Every detection claim in this project rests on the transaction model alone.

**A rejected model never appears as the working defence.** The adaptive column in the benchmark comparison is the promoted champion; the final candidate, if the gates turned it down, is reported separately and labelled as unpromoted.

The ethical boundary on catalogue and simulator content is covered in [Security and Ethics](SECURITY_AND_ETHICS.md); read it before adding anything that describes an attack.

---

## 7. Troubleshooting

**A fidelity check fails.** `run_fidelity()` prints a warning during pipeline step 2 and `tests/test_shortcuts_and_fidelity.py::test_all_fidelity_checks_pass` fails. Open `models/fidelity_report.json` and read the `detail` string of the failing check — each one states the measured quantity and the line it crossed. The failure means the simulator changed in a way that made fraud too easy to separate, so fix the generator, not the check. If `no_single_feature_separates_classes` failed, the named feature is doing something a real signal cannot; if `no_history_is_not_a_fraud_synonym` failed, a family stopped emitting cover traffic or started minting fresh cards; if `fraud_sometimes_uses_a_known_device` failed, a device dial moved. Checks marked `WARN` rather than `FAIL` do not fail the build, but a new warning is still a change in the data worth understanding. Every metric downstream of a failing dataset is untrustworthy, which is why the artefact-consistency suite refuses to accept a committed report with any failing check.

**Artefacts disagree.** `test_baseline_agrees_with_metrics_on_the_static_model` or `test_every_artifact_carries_the_same_schema_version` failing means the files in `models/` came from different runs, usually because individual stages were run piecemeal or because `--fast` was used and then something was committed. The fix is always the same: `python -m src.pipeline` followed by `python -m docs.build_docs`. If the schema version test failed, an artefact predates a bump of `config.SCHEMA_VERSION` — that constant is bumped whenever the simulator schema or the feature contract changes, precisely so stale artefacts are detectable rather than silently mixed in.

**The application shows a stale number.** The loaders in `app/common.py` are wrapped in `st.cache_data` and `st.cache_resource`, so a rerun does not re-read a file that changed on disk. Press `C` in the running app, or use the menu's "Clear cache", then rerun. If the number is still stale, the artefact itself is stale: check the file's `schema_version` and regenerate. If the number in the app disagrees with the README rather than with the artefact, the README was not rebuilt after the last pipeline run.

**A test that depends on artefacts is skipped.** This is expected on a fresh clone. `tests/test_artifact_consistency.py` calls `pytest.skip` when the artefact it needs does not exist, so the suite can run before the first pipeline run. A skip is not a pass — those tests verify the committed artefacts, and skipping them means nothing has been verified. Before committing or submitting, run the full pipeline and confirm the file reports fifteen passes and no skips. If a test skips *after* a full pipeline run, the artefact it wanted was not written: check the pipeline output for the step that should have produced it, and remember that `--fast` skips step 6 and therefore leaves every loop artefact untouched.

**The generative demonstration writes nothing.** `python -m src.generate.demo_specs` exits with status 2 and an explanatory message when no key is available, and writes no file. That is deliberate: a demonstration of the generative path produced by the offline heuristic that path is meant to demonstrate would be worthless. Run `python -m src.generate.demo_specs --check` to see what the script thinks is available, including how many cached responses are on disk.

**The loop test takes several minutes.** It is meant to. `test_adaptive_loop_end_to_end` runs two real rounds with real simulation and real retraining. Use `-k` to skip it while iterating on something else, and run it before committing.

---

## 8. Where to go next

If you are about to make a change, read [Architecture](ARCHITECTURE.md) first for what each module owns and how a pipeline run turns a seed into artefacts, then [Data Model](DATA_MODEL.md) for the exact schemas of the transaction table, the feature matrix, the attack specification and the catalogue. [Experiments](EXPERIMENTS.md) explains what each measurement means and how it can be misread, which is the document to check before you change anything that produces a number. [Decisions](DECISIONS.md) and [Design](DESIGN.md) record why the system is shaped the way it is, and are the right place to look when a piece of code seems needlessly indirect. [Security and Ethics](SECURITY_AND_ETHICS.md) states the boundaries on attack content. [Glossary](GLOSSARY.md) is there if a payments or machine-learning term is unfamiliar. Results live in [the README](../README.md) and in `models/*.json`, both generated from artefacts.
