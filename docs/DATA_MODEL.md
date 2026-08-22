# Data Model

*This document is the schema reference for the laboratory. It is written for a reviewing engineer, a competition judge, or a new contributor who needs to answer a precise question about the data — what a column means, whether the model is allowed to see it, how a feature is computed, what values a dial may take — without reading the source. After reading it you will know the transaction schema and its three-way column classification, the full feature matrix and how each feature is derived, the temporal-causality rule and the mechanisms that enforce it, the cardholder, merchant, attack-specification and Threat Atlas schemas, and which module writes each generated artifact. It contains no measured results; those live in [the README](../README.md) and in the generated files under `models/`.*

## Scope and conventions

Every fact in this document is verifiable in the repository. Where a structural count appears (number of transaction columns, number of features, number of injectors) the defining symbol is named so the count can be checked directly. All data described here is synthetic. Nothing in this system has been validated on real payment data, and no figure in this document is a measurement.

Two version markers tie the schema together. `config.SCHEMA_VERSION` is the string `"2.0"`, and it is stamped into every generated artifact so a stale file is detectable. The Threat Atlas carries its own matching `schema_version` field at the top of `src/identify/attacks.json`. Both are bumped when the simulator schema or the feature contract changes.

Paths in this document are relative to the repository root.

---

## 1. The transaction schema

The transaction table is the single stream that both the legitimate generator and every attack injector emit into. That is deliberate: the defender must see one homogeneous authorization stream, not a legitimate table and a fraud table it could tell apart by their shape. The canonical column order is the list `COLUMNS` in `src/generate/base_generator.py`, which has 23 entries; both `generate_legit` and every injector construct rows against it, and `src/generate/simulate.py` concatenates them with `columns=BG.COLUMNS`.

Each column carries a classification that determines whether the authorization-time model may see it:

- **Authorization-time input** — known to the authorizing system at the instant the decision must be made. Eligible to inform a feature.
- **Post-outcome** — knowable only after the transaction settles or is disputed. Retained in the raw dataset for analysis, never fed to the model.
- **Simulator oracle** — the simulator's own bookkeeping about who really produced a row. A real defender has none of this, so it is hard-blocked.

| Column | Type | Meaning | Classification |
| --- | --- | --- | --- |
| `txn_id` | string | Sequential identifier `T00000000`, reassigned in `simulate()` after the frame is sorted chronologically, so identifiers run in time order. | Authorization-time (identifier, not a feature) |
| `timestamp` | datetime64[ns] | Moment of authorization within the fixed 90-day window that starts at `profiles.SIM_START_ISO`. | Authorization-time |
| `cardholder_id` | string | Owner of the card, `CH######`. | Authorization-time (identifier) |
| `card_id` | string | The instrument, `CARD######`. Grouping key for all per-card history features. | Authorization-time (identifier) |
| `merchant_id` | string | Acceptor, `M#####`. Grouping key for merchant-relative and fan-in features. | Authorization-time (identifier) |
| `mcc` | string | Merchant Category Code, drawn from the keys of `config.MCC_CATALOG` (for example `5411_grocery`, `6011_atm`). | Authorization-time |
| `mcc_risk` | float | Per-merchant risk level. Sampled once per merchant in `generate_merchants` as the category baseline `MCC_CATALOG[mcc]["risk"]` plus Gaussian noise, clipped to `[0.02, 0.99]`, so a grocer can still be a bad actor. | Authorization-time |
| `amount` | float | Transaction value in Indian rupees, rounded to two decimals. | Authorization-time |
| `channel` | string | One of `card_cp`, `card_cnp`, `upi` — the three rails in `config.RAILS`. | Authorization-time |
| `country` | string | ISO country of the transaction. `IN` domestically; `AE`, `SG`, `GB` for the simulated foreign cities; and the members of `config.HIGH_RISK_COUNTRIES` (`XX`, `RU`, `NG`, `KP`) for high-risk geography. | Authorization-time |
| `geo_city` | string | City key from `config.GEO_CLUSTERS`. | Authorization-time |
| `distance_from_home_km` | float | Haversine distance from the cardholder's home city to `geo_city`, plus local jitter, rounded to one decimal. | Authorization-time |
| `device_id` | string | Device fingerprint. Genuine holders draw from their own `device_pool`; replacement devices appear as `NEWLEG#######`. | Authorization-time |
| `ip_prefix` | string | First two octets of the source address, as a network-side counter service would key on. | Authorization-time |
| `is_3ds` | bool | A step-up authentication was performed. On UPI this is always true, because the payer enters a PIN on every payment. | Authorization-time |
| `otp_verified` | bool | The step-up challenge was answered correctly. Tracks `is_3ds` for genuine traffic and is set explicitly by injectors that model relay and scam attacks. | Authorization-time |
| `is_new_payee` | bool | First time this card has transacted with this merchant. Computed once globally in `simulate()` as `~df.duplicated(subset=["card_id", "merchant_id"], keep="first")` on the chronologically sorted frame, so legitimate and fraudulent rows share one definition. | Authorization-time (historical fact) |
| `refund_flag` | bool | The transaction was later refunded, disputed or charged back. | **Post-outcome** |
| `account_age_days` | int | Age of the account in days at the moment of the transaction. | Authorization-time |
| `auth_result` | string | `approved` or `declined`. Only the card-testing injector emits declines, because a high decline ratio is the classic probing tell and belongs on the acquirer's side of the timeline. | **Post-outcome** |
| `is_fraud` | int | Ground-truth label, 0 or 1. | **Simulator oracle** |
| `attack_type` | string | `legit` for genuine traffic, otherwise the family name (one of the eleven injector keys). | **Simulator oracle** |
| `actor_role` | string | Who produced the row: `genuine_customer`, `fraud_actor_cover`, or `fraud` — the constants `ROLE_GENUINE`, `ROLE_COVER`, `ROLE_FRAUD` in `base_generator.py`. | **Simulator oracle** |

### Why `actor_role` exists

Fraud rings do not appear at the instant of the fraudulent authorization. Mule accounts, bust-out accounts and front merchants build ordinary-looking history first. Those rows are emitted with `is_fraud = 0`, because at authorization time they genuinely are not fraud, and are tagged `fraud_actor_cover` so an analyst can still count them. Without cover traffic, "this card has no history" would become a synonym for "fraud" and the defense would learn a simulator artifact instead of fraud behaviour. `tests/test_shortcuts_and_fidelity.py::test_fraud_actors_generate_cover_traffic` asserts both that cover rows exist and that every one of them is labelled legitimate.

### Post-outcome fields that carry real signal

`refund_flag` is post-outcome, but the *history* of a card's earlier refunds is not: it is a settled fact by the time the next authorization arrives. The feature `card_prior_dispute_rate` therefore uses `refund_flag` over the card's strictly earlier rows only. That is the one legitimate path from a post-outcome column into the model, and it is legitimate precisely because the shift makes it a statement about the past. The friendly-fraud injector sets `refund_flag = True` on every row it produces — that is the defining signal of the family and it is invisible at authorization time, which is why the family is included at all: so the honest answer can be measured rather than asserted.

---

## 2. The feature matrix

The feature matrix is defined by the list `FEATURE_COLUMNS` in `src/defend/features.py`, which has 36 entries. `AUTH_TIME_FEATURES` is an alias for the same list, used at call sites where the intent is clearer. `build_features(df)` returns exactly those columns, as floats, indexed like its input; `DefenseModel` stores `feature_columns` as a copy of the list and reindexes to it before fitting or scoring, so column order can never drift between training and inference.

Features are grouped below by what they are computed *from*. The groups are documentary; the model consumes one flat matrix.

### 2.1 Per-transaction features (17)

These depend only on the current row. They need no history and are defined on a card's first ever transaction.

| Feature | How it is computed |
| --- | --- |
| `log_amount` | `log1p(amount)` with the amount clipped at zero. |
| `mcc_risk` | The raw `mcc_risk` column, as a float. |
| `is_cnp` | 1 when `channel == "card_cnp"`. |
| `is_upi` | 1 when `channel == "upi"`. Kept separate from card-not-present because UPI authenticates on every payment, settles instantly and irrevocably, and addresses a payee rather than a card acceptor. Collapsing the two would hide all three differences. |
| `is_3ds` | The `is_3ds` column as an integer. |
| `otp_verified` | The `otp_verified` column as an integer. |
| `is_new_payee` | The `is_new_payee` column as an integer (first card-merchant pairing, computed globally in the simulator). |
| `log_distance` | `log1p(distance_from_home_km)` clipped at zero. |
| `hour` | Hour of day from the timestamp. |
| `is_night` | 1 when the hour is before 6 or after 22. |
| `is_weekend` | 1 when the timestamp's day of week is Saturday or Sunday. |
| `account_age_days` | The `account_age_days` column as a float. |
| `is_new_account` | 1 when `account_age_days < 90`. |
| `is_foreign` | 1 when `country != "IN"`. |
| `is_high_risk_country` | 1 when `country` is in `config.HIGH_RISK_COUNTRIES`. |
| `near_threshold` | 1 when the amount lies strictly between 4,000 and 5,000 rupees — proximity to a common structuring threshold. |
| `is_cash_like` | 1 when the MCC is in `features.CASH_LIKE_MCCS`, which is `{"6011_atm", "4829_money_transfer"}`. |

### 2.2 Per-card history features (9)

Computed inside a `groupby("card_id")` on a frame sorted by card and timestamp. Every one of them uses only rows strictly earlier than the current row.

| Feature | How it is computed |
| --- | --- |
| `amount_zscore` | `(amount - amt_mean_prev) / (amt_std_prev + 1)`, where the mean and standard deviation come from `expanding().mean().shift()` and `expanding().std().shift()` over the card's own amounts. The `+ 1` denominator keeps a card with near-constant spend from producing an unbounded score. |
| `time_since_last_hours` | Difference between consecutive timestamps on the card, in hours. |
| `velocity_1h` | Count of the card's transactions in a trailing one-hour window, from a timestamp-indexed `rolling("1h").count()` in `_rolling_velocity`. |
| `velocity_24h` | The same over a trailing 24-hour window. |
| `device_changed` | 1 when `device_id` differs from the card's previous `device_id` and a previous one exists. |
| `geo_changed` | 1 when `geo_city` differs from the card's previous `geo_city` and a previous one exists. |
| `card_history_depth` | `log1p` of the card's `cumcount()`, that is, the number of strictly prior transactions on the card. |
| `mcc_novel` | 1 the first time this card transacts at this MCC, from `~duplicated(subset=["card_id", "mcc"])`. |
| `card_prior_dispute_rate` | Mean of `refund_flag` over the card's earlier rows only, via `expanding().mean().shift()`. The shift is what makes a post-outcome column legal here. |

### 2.3 Behaviour-relative features (4)

These are the list `RELATIVE_FEATURES`. They ask whether a transaction is unusual *for this cardholder* or *for this merchant*, rather than unusual against a portfolio average — which is the difference between flagging an affluent customer's ordinary purchase and flagging a genuine outlier.

| Feature | How it is computed |
| --- | --- |
| `amount_vs_card_max_prior` | `log1p(amount) - log1p(amt_max_prev)`, where `amt_max_prev` is the running `cummax().shift()` of the card's own amounts. Positive means this is the largest charge the card has ever carried. |
| `distance_vs_card_max_prior` | The same construction on `distance_from_home_km` against the card's running maximum distance. |
| `card_mcc_share_prior` | Prior count of this card's transactions at this MCC, from `groupby(["card_id", "mcc"]).cumcount()`, divided by `card_history_depth + 1`. A first grocery charge on a grocery-only card is not the same event as a first money transfer on it. |
| `merchant_amount_zscore` | `(amount - merchant_mean_prev) / (merchant_std_prev + 1)`, with the mean and standard deviation taken from `expanding()...shift()` over that *merchant's* prior tickets. Computed in `_network_features`. A five-figure charge is ordinary at a travel agent and extraordinary at a grocer. |

### 2.4 Network relational features (6)

These are the list `NETWORK_FEATURES`, computed by `_network_features` on a globally time-ordered copy of the frame. They are what a payment network can compute and a single issuer or merchant cannot, and they are maintained as running counters keyed on device, address prefix and merchant, so each is an O(1) lookup at authorization time rather than a graph query. The helper `_prior_distinct(df, key, other)` implements "distinct `other` values seen against `key` strictly before now" as a cumulative sum over first-time pairings, minus the current row's own contribution.

| Feature | How it is computed |
| --- | --- |
| `device_card_count_prior` | Distinct cards previously seen on this `device_id`. |
| `card_device_count_prior` | Distinct devices previously seen on this `card_id`. |
| `ip_card_count_prior` | Distinct cards previously seen behind this `ip_prefix`. |
| `merchant_card_fanin_prior` | Distinct cards previously seen at this `merchant_id`. |
| `merchant_txn_count_prior` | Prior transaction count at this merchant, from `groupby("merchant_id").cumcount()`. |
| `merchant_new_card_ratio_prior` | `merchant_card_fanin_prior / max(merchant_txn_count_prior, 1)`, and undefined (later filled) when the merchant has no prior traffic. A normal merchant converts repeat customers; a front merchant sees almost only first-time cards. |

Five of these are counters that grow without bound as the simulation window fills — the set `_COUNT_FEATURES`, which is `NETWORK_FEATURES` minus `merchant_new_card_ratio_prior`. They are passed through `log1p` before reaching the model. On a raw scale, a model tuned on earlier traffic would meet a systematically different distribution later, and its threshold would drift off its false-positive budget for reasons that have nothing to do with fraud. `card_history_depth` is log-compressed for the same reason.

### 2.5 Defaults for undefined history

A card's first transaction has no prior mean, no prior maximum and no prior gap. Rather than dropping those rows or filling them with values that imply a verdict, `build_features` fills each with the honest neutral value:

| Feature | Fill | Why this value |
| --- | --- | --- |
| `amount_zscore` | `0.0` | No evidence the amount is unusual. |
| `time_since_last_hours` | `720.0` (30 days) | Stands for "no recent prior activity" rather than "instantaneous". |
| `card_prior_dispute_rate` | `0.0` | No prior disputes observed. |
| `merchant_new_card_ratio_prior` | `0.5` | The midpoint means "no information", not "safe". |
| `amount_vs_card_max_prior` | `0.0` | No prior maximum, so no evidence of excess. |
| `distance_vs_card_max_prior` | `0.0` | Same reasoning on distance. |
| `merchant_amount_zscore` | `0.0` | The merchant's first ticket has no prior distribution. |

`tests/test_data_quality.py::test_first_transaction_history_is_neutral` and its counterpart in `tests/test_shortcuts_and_fidelity.py` assert these defaults hold on every card's first row.

---

## 3. The three-way column classification and how it is enforced

The classification is declared as data, in `src/defend/features.py`, so it is one fact rather than a convention repeated at each call site:

```python
POST_OUTCOME_COLUMNS = ["refund_flag", "auth_result"]
ORACLE_COLUMNS = ["is_fraud", "attack_type", "actor_role"]
_BLOCKED = set(POST_OUTCOME_COLUMNS) | set(ORACLE_COLUMNS)
```

Everything else in `COLUMNS` is authorization-time input. The guard is `assert_auth_time_safe(feature_df)`, which raises `ValueError` naming the offending columns if any blocked name appears among the feature matrix's columns. It is called at the end of `build_features`, immediately before the matrix is returned, so no caller can obtain a feature frame that has not been checked.

### What would happen without the guard

Two of the blocked columns are close to perfect predictors, and both would be silently catastrophic.

`refund_flag` is set to `True` on every row the friendly-fraud injector produces and on a share of merchant-laundering rows. A model given it would learn to read the dispute rather than predict it, and would report near-perfect performance on a family that is, by construction, invisible at authorization time. The result would be a system that appears to solve the hardest case and in deployment solves nothing, because at the moment of the authorization decision the refund has not happened yet.

`auth_result` is worse in a subtler way. Only the card-testing injector emits declines, so the value `declined` is close to a family label. A model using it would be reading a downstream consequence of its own decision.

The oracle columns are more obvious and are blocked for completeness: `is_fraud` is the target, `attack_type` names the family, and `actor_role` states outright whether the row came from a fraud actor. None of them exists in production.

### Test coverage

| Test | File | What it establishes |
| --- | --- | --- |
| `test_post_outcome_fields_never_reach_model` | `tests/test_data_quality.py` | No name in `POST_OUTCOME_COLUMNS` appears in a real feature matrix. |
| `test_assert_guard_catches_injected_leakage` | `tests/test_data_quality.py` | Adding `refund_flag` to a valid matrix makes the guard raise. |
| `test_refund_flag_still_present_in_raw_dataset` | `tests/test_data_quality.py` | The column survives in the raw frame for analysis; blocking it from the model must not delete it from the data. |
| `test_post_outcome_and_oracle_columns_never_reach_the_model` | `tests/test_shortcuts_and_fidelity.py` | Extends the first check to `ORACLE_COLUMNS` as well. |
| `test_the_leakage_guard_actually_fires` | `tests/test_shortcuts_and_fidelity.py` | Confirms the guard raises for both a post-outcome name (`refund_flag`) and an oracle name (`attack_type`), so the check is not passing vacuously. |

The last test matters more than it first appears. A guard that never fires is indistinguishable from a guard that is broken, so the suite deliberately constructs the leak it is defending against.

---

## 4. Temporal causality

**The rule: a feature for the transaction at time T may depend only on rows strictly earlier than T.** No aggregate, no counter and no z-score may include the current row or any row after it. This is not a modelling preference. A feature that peeks forward produces a model that cannot exist at authorization time, and every measurement taken from it is meaningless.

Three mechanisms enforce the rule, and each is chosen because it makes the forward-looking version awkward to write by accident.

**Shifted expanding aggregates.** Anything of the form "this card's history so far" is written `expanding().<agg>().shift()`. The `shift()` is what excludes the current row. This covers `amt_mean_prev`, `amt_std_prev`, `card_prior_dispute_rate`, the running maxima `amt_max_prev` and `dist_max_prev` (via `cummax().shift()`), and the merchant-side mean and standard deviation behind `merchant_amount_zscore`.

**Trailing rolling windows.** Velocity is computed in `_rolling_velocity` by setting a timestamp index per card and calling `rolling("1h").count()` and `rolling("24h").count()`. A pandas time-based rolling window is trailing by definition — it spans `(T - window, T]` — so it can only look backwards. Note that these counts include the current transaction, which is the intended semantic: "how many authorizations has this card made in the last hour, including this one".

**Running counters over first-time pairings.** The relational features cannot use a per-card `groupby`, because they aggregate across cards. `_prior_distinct` therefore marks each row where a `(key, other)` pair is seen for the first time, takes a cumulative sum of those marks within the key, and subtracts the current row's own mark. The result is the number of distinct partners seen strictly before now, computed exactly the way a network-side counter service would maintain it.

### How the training split preserves it

`split_xy` in `src/defend/train.py` builds features **once** over the whole chronologically ordered frame and then slices into train, validation and test by row position, with the boundaries defined in `split_points` (default 15 percent validation, 25 percent test, taken from the end of the window). Building features per slice would restart every card's history, every velocity window and every network counter at the slice boundary, which no production system does and which would quietly weaken exactly the historical features the defense relies on. Because every feature already depends only on strictly earlier rows, computing globally and splitting afterwards introduces no leakage: a test-set row still sees only its own past.

### How it is tested

The tests do not inspect the implementation; they assert the property directly, by checking that a row's features are unchanged when future rows are added or removed.

| Test | File | Mechanism it covers |
| --- | --- | --- |
| `test_historical_features_use_only_the_past` | `tests/test_data_quality.py` | Computes the per-card history features on one card's first four rows and on all its rows, and asserts the first four are identical. Covers `amount_zscore`, both velocities, `device_changed`, `card_history_depth`, `card_prior_dispute_rate` and `time_since_last_hours`. |
| `test_network_counters_only_look_backwards` | `tests/test_shortcuts_and_fidelity.py` | The same prefix-invariance argument applied to `NETWORK_FEATURES` over a 4,000-row prefix of a 9,000-row frame. The relational counters are the easy place to get this wrong, because they aggregate across cards rather than within one. The test reindexes to the original chronological order before comparing, since `build_features` returns rows in card-sorted order. |
| `test_first_transaction_history_is_neutral` | both files | Every card's first row carries the neutral defaults, so "no history" is represented honestly rather than as an implied signal. |

---

## 5. Cardholder and merchant profiles

Profiles exist so that every entity has stable behaviour a deviation can be measured against. A portfolio built from one behaviour is trivially separable from fraud; a portfolio built from several overlapping behaviours is not. Both dataclasses live in `src/generate/profiles.py`, and both are constructed once per simulation from a seeded generator, so a given seed always yields the same portfolio.

### 5.1 `Cardholder`

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `cardholder_id` | str | required | `CH######`. |
| `card_id` | str | required | `CARD######`. |
| `home_city` | str | required | Key from `config.GEO_CLUSTERS`, drawn with the metro-weighted distribution in `_home_city_probs`. |
| `home_country` | str | required | Derived from `home_city`. |
| `spend_factor` | float | required | Multiplicative wealth scaler applied to every ticket. Drawn from the archetype's `spend` range with lognormal jitter, clipped to `[0.2, 6.0]`. |
| `preferred_mccs` | list[str] | required | The categories this customer buys in: mostly the archetype's `mcc_bias`, plus two idiosyncratic picks so no archetype is a fixed basket. |
| `usual_device_id` | str | required | Primary device, `DEV######`. |
| `usual_ip_prefix` | str | required | Two-octet address prefix. |
| `account_age_days` | int | required | Age at the *start* of the window. Thin-file holders draw below `LEGIT_REALISM["thin_file_max_age_days"]`; others draw from 90 to 3,000 days. |
| `daily_rate` | float | required | Expected transactions per day, from the archetype's `rate` range with lognormal jitter, clipped to `[0.05, 8.0]`. Also the weight used to sample which holder emits the next legitimate row. |
| `risk_tier` | float | required | Latent riskiness in `[0, 1]`, drawn `Beta(2, 8)`. |
| `device_pool` | list[str] | `[]` | Owned devices, primary at index 0. Size follows the archetype's `devices`. |
| `is_thin_file` | bool | `False` | New or low-history genuine customer. |
| `archetype` | str | `"frequent_urban"` | Key into `config.CUSTOMER_ARCHETYPES`. |
| `travel_prob` | float | `0.06` | Per-transaction chance of transacting away from home. Base rate times the archetype's `travel_mult`, clipped to `[0, 0.6]`. |
| `foreign_given_travel` | float | `0.30` | Of those trips, the share that are international. Base rate times `foreign_mult`, clipped to `[0, 0.9]`. |
| `odd_hour_prob` | float | `0.05` | Chance of a genuine late-night purchase. Base rate times `night_mult`, clipped to `[0, 0.5]`. |
| `weekend_mult` | float | `1.25` | Weekend spend lift, straight from the archetype. A business card is quieter at the weekend; a retail card busier. |
| `cnp_shift` | float | `0.0` | Additive shift on the merchant's card-not-present probability. |
| `activation_day` | int | `0` | First day of the window on which this card may transact. |
| `lapse_day` | int | `SIM_DAYS` (90) | Day after which the card stops being used. |
| `regular_merchants` | list[str] | `[]` | Merchants this customer revisits, assigned by `assign_relationships`. |
| `recurring_merchants` | list[str] | `[]` | Subscription merchants. |
| `recurring_amount` | dict[str, float] | `{}` | Stable price per subscription, so a recurring charge repeats at nearly the same value. |

Three of these fields deserve their reasoning stated. `regular_merchants` lives on the profile rather than inside the legitimate generator because the attack injectors need it too: mimicry that shops at the victim's own regular merchant is the entire point of mimicry. `activation_day` exists because a minority of genuine cards make their first-ever authorization mid-window, with no history at all — exactly like a fraudulently opened account — which is what stops "no prior history" from meaning "fraud". `lapse_day` balances that: without attrition, cards that start mid-window would push genuine volume steadily later while fraud stayed evenly spread, so the fraud rate would drift down across the window and a chronological test split would inherit a different base rate from the training slice. That would be a property of the generator, not of fraud.

### 5.2 What the customer archetypes control

`config.CUSTOMER_ARCHETYPES` holds seven behavioural templates whose `weight` values sum to 1.0 and determine how the portfolio is composed. Each template supplies the same set of dials:

| Archetype key | What it represents |
| --- | --- |
| `low_frequency_retail` | Occasional in-store shopper with a tight merchant set. |
| `frequent_urban` | High-frequency metro spender across many categories. |
| `traveller` | Frequently transacts away from home, often abroad. |
| `business_user` | Weekday-skewed corporate spend, high tickets, multi-device. |
| `digital_first` | Almost entirely card-not-present, late-night skew. |
| `thin_file_new` | Newly issued card, little or no history at authorization. |
| `high_value` | Affluent portfolio where large genuine tickets are normal. |

| Archetype key | Controls |
| --- | --- |
| `weight` | Share of the portfolio drawn from this archetype. |
| `rate` | Range for `daily_rate` — transaction frequency. |
| `spend` | Range for `spend_factor` — ticket size. |
| `n_mcc` | How many categories this customer buys in. |
| `mcc_bias` | Which categories those tend to be. |
| `devices` | Device-pool size. |
| `travel_mult`, `foreign_mult` | Multipliers on the base travel and foreign-travel rates. |
| `night_mult` | Multiplier on the base odd-hour rate. |
| `weekend_mult` | Weekend spend lift, used directly. |
| `cnp_shift` | Additive shift on card-not-present likelihood. |
| `desc` | Human-readable description surfaced in the web prototype. |

The archetypes are the reason genuine behaviour overlaps fraud. A traveller transacting abroad, a `digital_first` customer buying at 2 a.m., a `high_value` customer making a five-figure purchase and a `thin_file_new` customer with no history are all ordinary events in this portfolio, so none of the corresponding features can act as a fraud shortcut.

### 5.3 `Merchant`

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `merchant_id` | str | required | `M#####`. |
| `mcc` | str | required | Category key from `config.MCC_CATALOG`. |
| `city` | str | required | Key from `config.GEO_CLUSTERS`. |
| `country` | str | required | Derived from `city`. |
| `cnp_prob` | float | required | Baseline probability this merchant's sales are card-not-present, taken from the category. |
| `ticket_mu` | float | required | Lognormal location of the ticket distribution, category baseline plus noise. |
| `ticket_sigma` | float | required | Lognormal scale, from the category. |
| `risk` | float | required | Merchant risk, category baseline plus noise, clipped to `[0.02, 0.99]`. Written to the `mcc_risk` transaction column. |
| `is_synthetic` | bool | `False` | Reserved marker for a fabricated merchant. |
| `risk_tier` | str | `"standard"` | Banded from `risk` by `_risk_tier`: `low` below 0.25, `standard` below 0.5, `elevated` below 0.75, `high` above that. |
| `is_international` | bool | `False` | True when `country != "IN"`. |
| `supports_recurring` | bool | `False` | True for `4900_utilities`, `5816_digital_goods` and `5967_direct_mktg` — the categories that carry subscriptions. |
| `accepts_upi` | bool | `False` | Domestic merchants only, excluding `6011_atm`, at the rate `UPI_REALISM["merchant_acceptance"]`. UPI is a domestic rail and cash withdrawal is not on it. |
| `popularity` | float | `1.0` | Relative traffic weight, drawn lognormal and clipped to `[0.05, 25.0]`, so merchant volume is heavy-tailed: a few merchants take most of the traffic. |

`config.MCC_CATALOG` holds twelve categories, each with a lognormal ticket location and scale in rupees, a card-not-present probability, and a risk baseline. It is the source of `ticket_mu`, `ticket_sigma`, `cnp_prob` and the `risk` baseline above.

---

## 6. The `AttackSpec` schema

`AttackSpec`, in `src/generate/attack_spec.py`, is the contract between the creative half of the red team and the deterministic half of the simulator. The language model never writes transaction rows; it writes a specification saying which behavioural dial to move, in which direction, and why. Everything that touches the dataset is deterministic given the seed, so results are reproducible whether the specification came from a model or from the offline heuristic. The dataclass is frozen, so a spec cannot be mutated after validation.

### 6.1 Fields

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `attack_family` | str | required | One of the eleven keys in `BASE_SPECS`, which are the same eleven names as `INJECTORS`. |
| `strategy` | str | `"baseline behaviour of the family"` | Free-text description of the approach, truncated to 400 characters by the validator. |
| `intensity` | float | `1.0` | 1.0 is blatant, 0.15 is near-legitimate. Bounded by `ATTACK_SPEC_BOUNDS["intensity"]`. |
| `amount_profile` | str | `"moderate"` | Ticket size dial. |
| `velocity_profile` | str | `"single"` | Transactions per card and the window they fall in. |
| `device_behavior` | str | `"new_device"` | How much device history the attack inherits. |
| `geo_behavior` | str | `"plausible"` | Distance band from the victim's home. |
| `merchant_behavior` | str | `"new_high_risk_merchant"` | What kind of acceptor the attack transacts at. |
| `timing_profile` | str | `"any"` | Time-of-day shape. |
| `targets_signal` | str | `""` | The detector signal this generation is trying to defeat. Truncated to 200 characters. |
| `rationale` | str | `""` | Why the red team believes this will work. Truncated to 600 characters. |
| `confidence` | float | `0.5` | The proposer's own confidence, clamped to `[0, 1]`. |
| `source` | str | `"default"` | Provenance: `llm`, `heuristic`, `default` or `fixed`. |
| `generation` | int | `0` | Position in the family's lineage. Generation 0 specs are the roots in `BASE_SPECS`. |

### 6.2 Permitted vocabulary

Every categorical dial's vocabulary is `config.ATTACK_SPEC_BOUNDS`. A value outside it is not an error to be argued about; `_coerce_choice` replaces it with the family's baseline value and records a correction.

| Dial | Permitted values |
| --- | --- |
| `intensity` | Any float in `[0.15, 1.0]` |
| `amount_profile` | `micro`, `low`, `moderate`, `high`, `extreme` |
| `velocity_profile` | `single`, `low_and_slow`, `moderate`, `burst` |
| `device_behavior` | `trusted_device`, `secondary_device`, `new_device`, `shared_device` |
| `geo_behavior` | `home`, `plausible`, `domestic_far`, `foreign`, `high_risk` |
| `merchant_behavior` | `known_merchant`, `new_low_risk_merchant`, `new_high_risk_merchant`, `front_merchant`, `cash_like` |
| `timing_profile` | `customer_normal`, `business_hours`, `night`, `any` |

`ATTACK_SPEC_BOUNDS` also carries four hard simulation limits that no specification may exceed: `max_txns_per_card_per_day` (40), `max_amount_inr` (2,000,000), `min_amount_inr` (1.0) and `max_variants_per_round` (4,000). The amount bounds are applied in the injectors' `_amount` helper, which clips every generated value into that range.

### 6.3 Derived numeric knobs

The dials above are vocabulary. What the injectors actually read are the derived properties, which map each categorical choice to concrete numbers. This indirection is what lets the red team reason in behavioural language while the simulator stays deterministic.

| Property | Derived from | Value |
| --- | --- | --- |
| `amount_scale` | `AMOUNT_SCALE[amount_profile]` | `micro` 0.05, `low` 0.45, `moderate` 1.0, `high` 1.9, `extreme` 3.4. Multiplies the merchant's lognormal ticket. |
| `txns_per_card` | `VELOCITY_TXNS[velocity_profile]` | `single` (1, 1), `low_and_slow` (1, 2), `moderate` (2, 4), `burst` (4, 9). An inclusive draw range. |
| `velocity_window_hours` | `VELOCITY_WINDOW_H[velocity_profile]` | `single` 0.0, `low_and_slow` 72.0, `moderate` 24.0, `burst` 1.5. |
| `device_trust` | `DEVICE_TRUST[device_behavior]` | Probability the attack runs from a device the card has used before: `trusted_device` 1.0, `secondary_device` 0.6, `shared_device` 0.0, `new_device` 0.0. |
| `geo_km_range` | `GEO_KM[geo_behavior]` | Distance band in kilometres: `home` (0, 12), `plausible` (5, 60), `domestic_far` (400, 1800), `foreign` (1800, 7000), `high_risk` (2500, 8000). |
| `reuse_known_merchant` | `merchant_behavior == "known_merchant"` | Boolean. |

`to_dict()` serialises the fields plus a `derived` block containing all of the above, which is what appears inside `models/attack_lineage.json` and in the `attack_specs` block of `data/summary.json`.

The geography dial is worth one further note. `_geo_for` in `attack_injectors.py` chooses the destination city so that the *realised* haversine distance falls inside the requested band, rather than picking a city and reporting whatever distance results. Otherwise a specification that says "stay plausible" could emit a transaction 5,000 kilometres from home, the attack would be caught by a geography signal the red team had explicitly decided to stop using, and the lineage narrative would describe something the data does not contain.

### 6.4 The constraint layer

`validate_spec(proposed, family, previous=None, strict=False)` turns an arbitrary proposal into an executable, payment-legal specification, returning the spec and a `ValidationReport` recording every change. It applies four kinds of correction, in order: numeric range clamping; a monotone-stealth rule that prevents `intensity` from rising above the previous generation, because a persistent adversary does not become louder after being caught; vocabulary coercion; and family-level payment-domain requirements from `FAMILY_CONSTRAINTS`.

The family constraints encode statements about what can happen on a real rail. A `scam_transfer` must run from a trusted device, because the genuine customer authenticates on their own device and an "attacker device" authorized push payment is a contradiction in terms. `adversarial_mimicry` is confined to home or plausible geography and low or moderate amounts, because mimicry is *defined* by staying on the victim's centroid — a foreign high-value burst would simply be a different attack. `card_testing` is confined to micro or low amounts at moderate or burst velocity, because probing is only economically sensible at trivial value and real volume.

With `strict=False` a violating field is corrected to the nearest permitted value and the change is recorded. With `strict=True` the proposal is rejected and `SpecRejected` is raised, which is how the closed loop can report that the red team asked for something impossible. `ValidationReport` carries `family`, `accepted`, a list of `corrections` (each a `field`/`from`/`to`/`reason` record) and a list of `rejections`.

Two comparison helpers support the lineage view. `spec_diff(a, b)` returns the structured difference across the seven mutable dials, with `n_changed_dials`, so "how did the attack evolve" is an inspectable answer rather than a claim. `spec_distance(a, b)` returns a blunt 0-to-1 behavioural distance: one point per differing categorical dial plus the absolute intensity change, divided by seven. It is deliberately simple and is a readable summary, not a claim about a metric space.

---

## 7. The Threat Atlas entry schema

`src/identify/attacks.json` is the single source of truth for the attack catalog. `src/identify/taxonomy.py` loads it into frozen `Attack` dataclasses through `load_taxonomy()`, which is cached with `lru_cache` and runs `_validate` on every load. Loading is field-filtered — only keys that are `Attack` dataclass fields are passed to the constructor — so an extra key in the JSON is ignored rather than crashing the loader.

### 7.1 Top-level file structure

| Key | Type | Meaning |
| --- | --- | --- |
| `schema_version` | str | Matches `config.SCHEMA_VERSION`. |
| `focus` | str | The rail scope the catalog is written for. |
| `injectors` | list[str] | The names of the transaction-level injectors. Every `maps_to_injector` value must appear here, and the list is validated against `src/generate/attack_injectors.py` in practice by both naming the same eleven families. |
| `status_definitions` | dict | The four simulator-status definitions, carried in the data so the file explains itself. |
| `honesty_note` | str | The statement that the catalog is deliberately wider than the simulator. |
| `provenance` | dict | `authored_entries`, `merged_as_duplicates`, `distinct_entries` — how the catalog was assembled. |
| `attacks` | list[object] | The entries. |

### 7.2 Entry fields

Every entry carries every field below; none is optional in the committed file, though the dataclass supplies defaults for the fields marked as such.

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | str | Stable snake-case identifier. Validated unique across the catalog. |
| `name` | str | Human-readable title. |
| `category` | str | One of six: Social Engineering, Account & Identity Compromise, Payment Instrument Attacks, Merchant & Ecosystem Abuse, Authorized Push Payment Scams, Adversarial ML & Model Evasion. |
| `subcategory` | str | Finer classification within the category. |
| `rails` | list[str] | Payment rails the attack touches: `card_cnp`, `card_cp`, `upi`, `a2a`, `wallet`, `token`. |
| `channel` | str | Where the attack reaches the victim or the system: `ONLINE`, `MESSAGING`, `VOICE`, `IN_APP`, `ONBOARDING`, `CNP`, `CP`, `BACKOFFICE`. |
| `attacker_objective` | str | What the attacker is trying to achieve, in one sentence. |
| `genai_role` | str | How generative AI participates: `content_generation`, `automation_scale`, `multimodal_impersonation`, `identity_synthesis`, `strategy_planning`, `evasion_optimization`. |
| `genai_mechanism` | str | Prose explanation of what the technology changes for the *defender* — which detection assumption it breaks. |
| `kill_chain` | list[str] | Ordered stages from targeting to cash-out. |
| `transaction_signature` | str | What the attack looks like in the payment stream. |
| `behavioral_signature` | str | What it looks like in account and session behaviour. |
| `observable_signals` | list[str] | Concrete, defender-measurable indicators. These ground the features in `src/defend/features.py`. |
| `auth_time_observability` | str | How much of the attack is visible at authorization: `high`, `partial`, `low`, `none`. Ranked by `OBSERVABILITY_ORDER` and validated against it. |
| `post_transaction_signals` | list[str] | What shows up only after settlement — disputes, chargeback clustering, recall failures. |
| `defense_difficulty` | str | `low`, `medium`, `high`, `extreme`. Ranked by `DIFFICULTY_ORDER`. |
| `expected_impact` | str | `medium`, `high`, `critical`. |
| `severity` | str | `medium`, `high`, `critical`. Ranked by `severity_rank`, which also recognises `low`. |
| `novelty_score` | float | How much of this attack is genuinely new versus a known pattern at new scale. Validated to lie in `[0, 1]`. |
| `real_world_grounding` | str | The evidence base — what is documented, and what specifically is new. |
| `ethical_notes` | str | A statement of what the entry deliberately does not contain. Every entry describes defender-visible consequences only; no lure content, infrastructure detail, tooling or bypass technique appears anywhere in the file. |
| `simulator_status` | str | One of the four values below. |
| `maps_to_injector` | str or null | The injector that reproduces this attack, or null. |
| `simulatable` | bool | Convenience mirror of whether the status is one of the two simulated values. |

The `Attack` dataclass exposes four derived properties: `severity_rank`, `observability_rank` and `difficulty_rank` for ordering, and `is_simulated`, which is true when `simulator_status` is `IMPLEMENTED` or `PARAMETERIZED`.

### 7.3 The four simulator-status values

The catalog is deliberately wider than the simulator. Every entry carries its status so that breadth of research is never presented as breadth of simulation. These are the exact definitions, from `status_definitions` in the JSON and the module docstring of `taxonomy.py`:

| Status | Definition |
| --- | --- |
| `IMPLEMENTED` | A dedicated transaction-level injector reproduces this attack's authorization footprint, and it is part of the default simulated mix. |
| `PARAMETERIZED` | Reachable today as a configuration of an existing injector through the attack-specification dials (amount, velocity, device, geography, merchant, timing) without new code, but not part of the default mix. |
| `RESEARCH_ONLY` | Catalogued and characterized — objective, GenAI role, kill chain, transaction and behavioural signature, authorization-time observability — but not simulated. Listed so the research surface is visible and not overstated. |
| `FUTURE` | Named as planned simulator work. Not simulated today. |

### 7.4 What `_validate` enforces at load time

The status field would be decorative if nothing checked it, so `_validate` raises `ValueError` on any of the following, and the exception propagates out of `load_taxonomy()`:

- Duplicate `id` values anywhere in the catalog.
- A `maps_to_injector` naming an injector that is not in the file's `injectors` list.
- A `novelty_score` outside `[0, 1]`.
- A `simulator_status` outside `VALID_STATUS`.
- An entry that is `IMPLEMENTED` or `PARAMETERIZED` but maps to no injector — a claimed simulator mapping must actually name one.
- An entry that is `RESEARCH_ONLY` or `FUTURE` but claims an injector — an unsimulated attack must not pretend to have one.
- An `auth_time_observability` outside the four known levels.

The last two rules are the ones that make the honesty claim structural rather than editorial. It is not possible to mark an entry as simulated without wiring it to a real injector, and it is not possible to leave a stale injector reference on an entry that has been downgraded to research.

`Taxonomy` provides the query surface over the loaded catalog: `by_id`, `by_status`, `for_injector`, `simulatable()`, the `categories`/`all_rails`/`channels`/`genai_roles` vocabularies, `coverage_by_category()` for the coverage map, and `summary_counts()` for the headline structure. Running `python -m src.identify.taxonomy` prints the summary and the per-category coverage table, which is the quickest way to check the catalog's shape after an edit.

---

## 8. Artifact files

Artifacts are generated, not authored. Nothing under `data/` or `models/` should be edited by hand; every file is rewritten by the module named below, and `python -m src.pipeline` regenerates the whole set in order. Files carrying `schema_version` can be checked against `config.SCHEMA_VERSION` to detect staleness. Every path constant is declared in `config.py`.

### 8.1 `data/`

| File | Contains | Written by |
| --- | --- | --- |
| `transactions.parquet` | The full labeled transaction table, all 23 columns, sorted chronologically. The canonical dataset. | `src/generate/simulate.py` (`run_and_save`) |
| `transactions.csv` | The same table in CSV, for inspection outside Python. | `src/generate/simulate.py` (`run_and_save`) |
| `summary.json` | Dataset manifest: schema version, row and fraud counts, cover-traffic count, cardholder and merchant counts, the archetype histogram, per-family transaction counts, the serialised `AttackSpec` used for each family, the date range, and the text-corpus counts. | `src/generate/simulate.py` (`_summarize` and `run_and_save`) |
| `attack_artifacts.jsonl` | The text corpus for the artifact classifier, one JSON object per line with keys `text` (what the classifier sees, provenance markers stripped), `display_text` (markers kept, for the interface), `label`, `attack_id`, `artifact_type`, `red_flags` and `source`. The subject line is excluded from `text` because it carries the attack's catalog name, and leaving it in would let the model read the label off the string. | `src/generate/llm_agent.py` (`build_text_corpus`), written by `simulate.run_and_save` |

### 8.2 `models/` — JSON reports

| File | Contains | Written by |
| --- | --- | --- |
| `metrics.json` | Headline evaluation of the static model on the held-out chronological test split: operating threshold, the standard classification and ranking measures, confusion matrix, evaluation sizes, per-family recall, schema version, seed and feature count. | `src/defend/train.py` |
| `fidelity_report.json` | Data-realism audit: separability of the feature space, the strongest single-feature signal, distribution summaries, portfolio composition including `actor_role` counts, and the pass/warn/fail checks with their counts. | `src/generate/fidelity.py` |
| `leave_one_out.json` | Leave-one-attack-family-out results — what the defense retains for a family it has never been trained on — with an `interpretation` block and an explicit `what_this_does_not_show`. | `src/experiments/leave_one_out.py` |
| `loop_history.json` | The full closed-loop record: per-round history, the focus families and how they were selected, retired frontiers, guard families, the loop configuration, the base model's starting position, and the promotion outcome. | `src/loop/redteam_loop.py` |
| `attack_lineage.json` | One node per attack generation: the specification that produced it, the measured weakness that motivated it, what the constraint layer corrected, and what it cost the defense. | `src/loop/redteam_loop.py` |
| `model_registry.json` | Governance ledger: the promotion gates from `config.CHAMPION_CHALLENGER`, and one `registry_entry` per trained model recording version, stage (`champion`, `challenger` or `baseline`), schema version, data seed, training time, metrics, replay composition, the attack generations used, and the acceptance decision. | `src/defend/governance.py` (`write_registry`), called from `src/loop/redteam_loop.py` |
| `hero_example.json` | One end-to-end worked case: the family, the narrative, the evolved specification and the signal it targets, the transaction itself, and how the stale and adapted models each scored it. | `src/loop/redteam_loop.py` (`_save_hero_example`) |
| `baseline_comparison.json` | Rules baseline against static machine learning against the promoted adaptive champion, plus the unpromoted final candidate reported separately and labelled, and a false-positive-matched comparison block. | `src/defend/baseline.py` (`compare`), rewritten by `src/pipeline.py` with the full model set |
| `threshold_sweep.json` | The operating-point sweep: points along the threshold curve, the selected operating point in both score and probability space, the decision-policy thresholds and the false-positive budget. | `src/defend/diagnostics.py` |
| `calibration.json` | Whether the score is calibrated, Brier scores before and after isotonic calibration, and the reliability curves for each. | `src/defend/diagnostics.py` |
| `operational_metrics.json` | The static and adaptive models translated into review volume and customer friction under the illustrative assumptions in `config.OPERATIONAL_SCENARIO`. Those assumptions are synthetic and are labelled as such in the config. | `src/defend/diagnostics.py` |
| `blind_spots.json` | Attack families ranked by how poorly the defense handles them, the hardest set, and the next red-team target the loop would choose. | `src/defend/diagnostics.py` |
| `family_recall.json` | Per-family recall measured on the dedicated fraud-enriched frame described by `config.FAMILY_EVAL`, generated from an unseen seed, with the reporting floor `min_n_to_report`. | `src/defend/diagnostics.py` |
| `head_to_head.json` | Stale model against adapted model on the same evolved-attack frame at a matched false-positive rate, with the specifications that produced the frame. | `src/defend/diagnostics.py` |
| `text_metrics.json` | Evaluation of the artifact text classifier, together with `caveat`, `honest_reading` and `is_sanity_check_not_evidence` fields stating plainly that this is a sanity check on a synthetic corpus. | `src/defend/text_model.py` |
| `pipeline_summary.json` | The one-file record of a full pipeline run: seed, dataset summary, fidelity, headline metrics, leave-one-out results, loop rounds and outcomes, baseline comparison, and calibration. | `src/pipeline.py` |
| `genai_spec_demo.json` | The specification-generation demonstration: what a proposal looked like, what the constraint layer did to it. Written only when the demo is run. | `src/generate/demo_specs.py` |

### 8.3 `models/` — binary and tabular artifacts

| File | Contains | Written by |
| --- | --- | --- |
| `defense_model.joblib` | The trained static defense model — both heads, the scaler, the calibrator, the threshold and the feature-column list. | `src/defend/train.py` |
| `text_model.joblib` | The trained artifact text classifier. | `src/defend/text_model.py` |
| `loop_base_model.joblib` | The stale model as it stood at round zero, kept so the before-and-after comparison uses the actual earlier model rather than a reconstruction. | `src/loop/redteam_loop.py` |
| `loop_adapted_model.joblib` | The final candidate the loop produced — what adaptation achieved. | `src/loop/redteam_loop.py` |
| `loop_champion_model.joblib` | The last candidate that cleared every promotion gate — what governance would actually allow into the authorization path. When no candidate clears the gates, this differs from the adapted model, and that difference is the point. | `src/loop/redteam_loop.py` |
| `defend_demo.parquet` | Precomputed scores, actions and reason codes for the held-out test split, with the identifying and display columns listed in `diagnostics.run_all`. The web prototype reads this instead of rebuilding every feature on each rerun. | `src/defend/diagnostics.py` |

### 8.4 `artifacts_cache/`

`config.CACHE_DIR` holds the language-model response cache written by `src/llm/client.py`. Each file is named by a hash of model, system prompt, user prompt and response kind, and stores the response alongside the prompt that produced it. The cache is what makes the system fully runnable without an API key: when `config.llm_available()` is false, the generation paths fall back to the cached artifacts and to offline templates, and the pipeline still completes end to end.

---

## Where to go next

For how these schemas are wired together — which module calls which, and in what order a pipeline run produces the artifacts in section 8 — read [Architecture](ARCHITECTURE.md). For the reasoning behind the design choices this document only states, particularly the anti-shortcut work in the generator and the promotion-gate policy, read [Design](DESIGN.md). For the measured results, which are deliberately absent here so this reference can never contradict them, read the evaluation sections of [the README](../README.md) and the generated files under `models/`. For the questions reviewers ask most often about the data and its limits, read [Judge Q&A](JUDGE_QA.md).
