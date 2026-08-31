"""Central configuration: paths, seeds, model params, rail/domain constants.

Everything reproducible flows from GLOBAL_SEED. Live GenAI generation reads
ANTHROPIC_API_KEY from the environment (.env is auto-loaded if python-dotenv is
installed); when absent the system falls back to the committed artifact cache.
"""
from __future__ import annotations

import os
from pathlib import Path

try:  # optional convenience
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass

# Paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
CACHE_DIR = ROOT / "artifacts_cache"
IDENTIFY_DIR = ROOT / "src" / "identify"
ATTACKS_JSON = IDENTIFY_DIR / "attacks.json"

for _d in (DATA_DIR, MODELS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Canonical dataset / model artifact names
DATASET_PARQUET = DATA_DIR / "transactions.parquet"
DATASET_CSV = DATA_DIR / "transactions.csv"
ARTIFACTS_JSONL = DATA_DIR / "attack_artifacts.jsonl"
MODEL_PATH = MODELS_DIR / "defense_model.joblib"
TEXT_MODEL_PATH = MODELS_DIR / "text_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
LOOP_HISTORY_PATH = MODELS_DIR / "loop_history.json"
LOOP_BASE_MODEL_PATH = MODELS_DIR / "loop_base_model.joblib"      # stale model (round 0)
LOOP_ADAPTED_MODEL_PATH = MODELS_DIR / "loop_adapted_model.joblib"  # final adapted candidate
LOOP_CHAMPION_MODEL_PATH = MODELS_DIR / "loop_champion_model.joblib"  # last PROMOTED model
HERO_EXAMPLE_PATH = MODELS_DIR / "hero_example.json"
BASELINE_COMPARISON_PATH = MODELS_DIR / "baseline_comparison.json"
LEAVE_ONE_OUT_PATH = MODELS_DIR / "leave_one_out.json"
FIDELITY_REPORT_PATH = MODELS_DIR / "fidelity_report.json"
OPERATIONAL_PATH = MODELS_DIR / "operational_metrics.json"
THRESHOLD_SWEEP_PATH = MODELS_DIR / "threshold_sweep.json"
BLIND_SPOTS_PATH = MODELS_DIR / "blind_spots.json"
MODEL_REGISTRY_PATH = MODELS_DIR / "model_registry.json"
ATTACK_LINEAGE_PATH = MODELS_DIR / "attack_lineage.json"
CALIBRATION_PATH = MODELS_DIR / "calibration.json"
FAMILY_RECALL_PATH = MODELS_DIR / "family_recall.json"
HEAD_TO_HEAD_PATH = MODELS_DIR / "head_to_head.json"

# Reproducibility
GLOBAL_SEED = 42

# Simulation defaults
DEFAULT_N_CARDHOLDERS = 6000
DEFAULT_N_MERCHANTS = 1200
DEFAULT_N_TRANSACTIONS = 150_000
DEFAULT_FRAUD_RATE = 0.012  # ~1.2% base rate, realistic for card portfolios

# A per-family recall figure is only worth quoting if enough held-out examples of
# that family exist. At a 1.2% base rate the headline test slice contains only a
# few dozen of the rarer families, so family-level numbers are measured on a
# dedicated, larger, fraud-enriched frame generated from an UNSEEN seed — a
# different synthetic portfolio the model has never met.
FAMILY_EVAL = {
    "n_transactions": 90_000,
    "fraud_rate": 0.030,
    "seed_offset": 424_242,
    "min_n_to_report": 30,
}

# Model / artifact versioning (bumped whenever the simulator schema or the
# feature contract changes, so stale artifacts are detectable).
SCHEMA_VERSION = "2.0"

# LLM (Anthropic) config
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
LLM_MODEL = os.getenv("FRAUD_LAB_MODEL", "claude-sonnet-5").strip()
LLM_ENABLED = os.getenv("FRAUD_LAB_LLM_ENABLED", "1").strip() not in {"0", "false", "False"}
LLM_MAX_TOKENS = 1500

def llm_available() -> bool:
    """True when live generation is possible (key present and not disabled)."""
    return bool(ANTHROPIC_API_KEY) and LLM_ENABLED

# Domain constants — cards-first (Mastercard core), light multi-rail for breadth
RAILS = ["card_cnp", "card_cp", "upi"]

# UPI is the dominant retail rail in India and the one most authorized-push-payment
# scams actually settle on, so it is simulated as a real channel rather than
# declared and left empty. Its semantics differ from cards in ways a detector
# cares about: the payer authenticates with a UPI PIN (so a step-up "passing"
# means much less), settlement is instant and irrevocable, tickets are smaller and
# far more frequent, and the payee is an address rather than a card acceptor.
UPI_REALISM = {
    "merchant_acceptance": 0.62,       # share of merchants that accept UPI at all
    "legit_share": 0.34,               # share of eligible genuine transactions on UPI
    "ticket_scale": 0.45,              # UPI tickets run smaller than card tickets
    "p2p_share": 0.30,                 # share of UPI volume that is person-to-person
    "max_ticket_inr": 100_000.0,       # per-transaction cap
}

# Merchant Category Codes we simulate, with typical ticket (log-normal mu/sigma
# on INR) and channel bias. Grounded in real MCC spend behaviour.
MCC_CATALOG = {
    "5411_grocery": {"mu": 6.6, "sigma": 0.5, "cnp_prob": 0.15, "risk": 0.2},
    "5812_restaurant": {"mu": 6.4, "sigma": 0.6, "cnp_prob": 0.25, "risk": 0.3},
    "5541_fuel": {"mu": 7.4, "sigma": 0.4, "cnp_prob": 0.05, "risk": 0.3},
    "5732_electronics": {"mu": 9.2, "sigma": 0.8, "cnp_prob": 0.70, "risk": 0.7},
    "5999_retail": {"mu": 7.0, "sigma": 0.9, "cnp_prob": 0.55, "risk": 0.5},
    "4829_money_transfer": {"mu": 8.8, "sigma": 1.0, "cnp_prob": 0.95, "risk": 0.9},
    "5967_direct_mktg": {"mu": 6.8, "sigma": 0.7, "cnp_prob": 0.98, "risk": 0.8},
    "7995_gambling": {"mu": 8.0, "sigma": 1.0, "cnp_prob": 0.98, "risk": 0.95},
    "4900_utilities": {"mu": 7.6, "sigma": 0.5, "cnp_prob": 0.80, "risk": 0.2},
    "4722_travel": {"mu": 9.6, "sigma": 0.7, "cnp_prob": 0.90, "risk": 0.6},
    "5816_digital_goods": {"mu": 6.2, "sigma": 0.8, "cnp_prob": 0.99, "risk": 0.7},
    "6011_atm": {"mu": 8.2, "sigma": 0.5, "cnp_prob": 0.0, "risk": 0.5},
}

# Home geography clusters (lat, lon) — Indian metros + a few intl for travel.
GEO_CLUSTERS = {
    "mumbai": (19.076, 72.877),
    "delhi": (28.704, 77.102),
    "bengaluru": (12.972, 77.595),
    "hyderabad": (17.385, 78.487),
    "chennai": (13.083, 80.271),
    "kolkata": (22.573, 88.364),
    "pune": (18.520, 73.856),
    "dubai": (25.205, 55.271),
    "singapore": (1.352, 103.820),
    "london": (51.507, -0.128),
}

HIGH_RISK_COUNTRIES = {"XX", "RU", "NG", "KP"}

# Detection threshold target: keep false-positive rate on legit low.
TARGET_MAX_FPR = 0.02

# Legitimate-behaviour realism (Phase 1)
# These rates make genuine customers behave with realistic variation so that
# `device_changed`, `is_new_payee`, foreign spend, high value, etc. are genuine
# *risk indicators* rather than perfect fraud shortcuts. They are deliberately
# uncommon-but-not-rare, producing credible overlap with fraud. All reproducible
# via GLOBAL_SEED. Do NOT tune these to hit a target metric.
LEGIT_REALISM = {
    "multi_device_fraction": 0.28,     # share of cardholders who own >1 device
    "max_devices": 3,                  # device-pool size for multi-device holders
    "secondary_device_prob": 0.18,     # chance a multi-device holder uses a non-primary device
    "new_device_prob": 0.015,          # chance a legit txn is from a brand-new (replacement) device
    "travel_prob": 0.06,               # base chance a legit txn is outside the home city
    "foreign_given_travel": 0.30,      # of travel txns, share that are international
    "high_value_prob": 0.02,           # chance of a genuine unusually large purchase
    "high_value_mult": (4.0, 9.0),     # multiplier range for a high-value purchase
    "odd_hour_prob": 0.05,             # base chance of a genuine late-night purchase
    "thin_file_fraction": 0.10,        # share of cardholders who are new/thin-file
    "thin_file_max_age_days": 75,      # account age ceiling for thin-file customers
    "legit_dispute_prob": 0.008,       # genuine downstream refund/dispute on a legit txn (post-outcome only)
    # --- Phase 4 realism (temporal + relational) ---------------------------
    "weekend_lift": 1.25,              # baseline weekend spend multiplier (archetypes modulate it)
    "payday_days": (1, 2, 3, 15, 16, 17),   # day-of-month payday-like spend clustering
    "payday_lift": 1.45,               # relative weight of payday days
    "burst_prob": 0.10,                # chance a legit txn starts a short shopping session
    "burst_extra": (1, 3),             # extra transactions in that session
    "burst_window_hours": 2.0,         # session length
    "subscription_fraction": 0.45,     # share of holders with >=1 recurring merchant
    "subscription_max": 2,             # recurring merchants per subscribing holder
    "subscription_share": 0.07,        # share of a subscriber's transactions that are the recurring charge
    "late_activation_fraction": 0.16,  # holders whose card is issued DURING the window
    # Attrition balances issuance. Without it, cards that only start mid-window
    # push genuine volume steadily later while fraud stays evenly spread, so the
    # fraud RATE drifts down across the window and a chronological test split
    # inherits a different base rate from the training slice — a property of the
    # generator, not of fraud.
    "lapse_fraction": 0.16,            # holders whose card stops being used mid-window
    "regulars_repeat_prob": 0.78,      # chance a legit txn goes to an already-known merchant
    "high_risk_country_prob": 0.0015,  # rare genuine spend in a high-risk geography
}

# Customer archetypes — behavioural templates that shape amount, frequency,
# device count, geography, timing and channel mix. A portfolio made of one
# behaviour is trivially separable from fraud; a portfolio made of several
# overlapping behaviours is not. Weights sum to 1.0.
CUSTOMER_ARCHETYPES = {
    "low_frequency_retail": {
        "weight": 0.24, "rate": (0.08, 0.5), "spend": (0.4, 1.0), "n_mcc": (2, 4),
        "mcc_bias": ["5411_grocery", "5541_fuel", "5812_restaurant", "4900_utilities"],
        "devices": 1, "travel_mult": 0.5, "foreign_mult": 0.3, "night_mult": 0.5,
        "weekend_mult": 1.35, "cnp_shift": -0.10, "desc": "Occasional in-store shopper, tight merchant set.",
    },
    "frequent_urban": {
        "weight": 0.22, "rate": (1.2, 4.0), "spend": (0.7, 1.6), "n_mcc": (5, 8),
        "mcc_bias": ["5812_restaurant", "5411_grocery", "5999_retail", "5816_digital_goods"],
        "devices": 2, "travel_mult": 1.0, "foreign_mult": 0.6, "night_mult": 1.2,
        "weekend_mult": 1.30, "cnp_shift": 0.0, "desc": "High-frequency metro spender across many categories.",
    },
    "traveller": {
        "weight": 0.10, "rate": (0.6, 2.0), "spend": (0.9, 2.2), "n_mcc": (4, 7),
        "mcc_bias": ["4722_travel", "5812_restaurant", "5541_fuel", "4900_utilities"],
        "devices": 2, "travel_mult": 5.0, "foreign_mult": 2.2, "night_mult": 1.6,
        "weekend_mult": 1.10, "cnp_shift": 0.05, "desc": "Frequently transacts away from home, often abroad.",
    },
    "business_user": {
        "weight": 0.12, "rate": (0.9, 3.0), "spend": (1.4, 3.2), "n_mcc": (4, 7),
        "mcc_bias": ["4722_travel", "5732_electronics", "4829_money_transfer", "5999_retail"],
        "devices": 3, "travel_mult": 2.5, "foreign_mult": 1.5, "night_mult": 0.8,
        "weekend_mult": 0.55, "cnp_shift": 0.10, "desc": "Weekday-skewed corporate spend, high tickets, multi-device.",
    },
    "digital_first": {
        "weight": 0.18, "rate": (0.8, 3.0), "spend": (0.5, 1.4), "n_mcc": (4, 7),
        "mcc_bias": ["5816_digital_goods", "5967_direct_mktg", "5999_retail", "7995_gambling"],
        "devices": 2, "travel_mult": 0.6, "foreign_mult": 0.9, "night_mult": 2.4,
        "weekend_mult": 1.15, "cnp_shift": 0.25, "desc": "Almost entirely card-not-present, late-night skew.",
    },
    "thin_file_new": {
        "weight": 0.09, "rate": (0.08, 0.8), "spend": (0.3, 0.9), "n_mcc": (2, 4),
        "mcc_bias": ["5816_digital_goods", "5812_restaurant", "5411_grocery"],
        "devices": 1, "travel_mult": 0.7, "foreign_mult": 0.4, "night_mult": 1.3,
        "weekend_mult": 1.20, "cnp_shift": 0.10, "desc": "Newly issued card, little or no history at authorization.",
    },
    "high_value": {
        "weight": 0.05, "rate": (0.3, 1.6), "spend": (2.2, 4.0), "n_mcc": (4, 7),
        "mcc_bias": ["5732_electronics", "4722_travel", "5999_retail", "4829_money_transfer"],
        "devices": 2, "travel_mult": 2.0, "foreign_mult": 1.8, "night_mult": 0.9,
        "weekend_mult": 1.15, "cnp_shift": 0.05, "desc": "Affluent portfolio; large genuine tickets are normal.",
    },
}

# Fraud-actor realism. Fraud rings do not spring into existence at the moment of
# the fraudulent authorization: mule and bust-out accounts build ordinary-looking
# history first ("cover traffic"), and probing runs hit the same card repeatedly.
# Cover traffic is labelled LEGITIMATE, because at authorization time it is.
FRAUD_REALISM = {
    "bustout_cover_txns": (8, 26),     # normal-looking transactions before the bust-out
    "bustout_ramp_days": (30, 70),     # days of grooming before the spike
    "mule_cover_txns": (3, 12),        # cover traffic per smurfing mule account
    "laundering_cover_txns": (2, 9),   # genuine-looking sales through the front merchant
    "testing_probes_per_card": (2, 6), # repeat probes on the same test card (creates velocity)
    "testing_cover_merchants": (2, 5), # merchants revisited within a probing burst
    "victim_repeat_prob": 0.35,        # chance a compromised victim card is hit more than once
    "victim_repeat_extra": (1, 3),     # additional transactions on that victim card
    "victim_same_merchant_prob": 0.40, # chance a repeat hit reuses the same merchant
    "geo_high_risk_prob": 0.22,        # share of geo-anomaly fraud in a high-risk jurisdiction
}

# Risk-based decision policy (Phase 3) — prototype issuer/network decisioning.
# NOT a claim about any production system. Thresholds are on the fused risk score.
# Tiers are applied to the CALIBRATED fraud probability, so they are readable as
# statements about risk rather than as positions on an arbitrary score.
#   >= decline : confident enough to refuse outright
#   >= step_up : worth friction (3DS challenge / review) but not a refusal
#   otherwise  : approve
# step_up defaults to the model's own operating threshold — the point already
# tuned to the agreed false-positive budget — rather than a second magic number.
DECISION_THRESHOLDS = {
    "decline": 0.60,
    "step_up": None,     # None => use the model's tuned operating threshold
}

# Adaptive closed-loop (Phase 2) — cumulative adversarial replay.
LOOP_CONFIG = {
    "replay_buffer_max": 6000,         # cap on retained adversarial examples (bounded replay)
    # Emphasis on replayed adversarial examples during retraining, applied as a
    # sample weight. Too high and the model over-fits the newest generation and
    # its overall ranking quality regresses — which the promotion gates catch.
    "replay_oversample": 4,
    "recovery_delta": 0.05,            # min recall gain (adapted-stale) to call it "adapted"
    "residual_recall_ceiling": 0.60,   # below this after adaptation => "residual frontier"
    "max_fpr_regression": 0.015,       # adaptation may not worsen legit FPR by more than this
    "forgetting_tolerance": 0.10,      # max allowed recall drop on a previously-learned family
    "focus_mix_boost": 3.5,            # up-weight focus families in loop frames for stable n
}

# Attack-specification bounds (Phase 4).
# The red-team agent proposes a STRUCTURED attack specification; the simulator
# only ever executes specifications that satisfy these payment-domain bounds.
# LLM creativity is constrained by simulation rules — an invalid or impossible
# specification is normalized (clamped) or rejected, never silently executed.
ATTACK_SPEC_BOUNDS = {
    "intensity": (0.15, 1.0),
    "amount_profile": ["micro", "low", "moderate", "high", "extreme"],
    "velocity_profile": ["single", "low_and_slow", "moderate", "burst"],
    "device_behavior": ["trusted_device", "secondary_device", "new_device", "shared_device"],
    "geo_behavior": ["home", "plausible", "domestic_far", "foreign", "high_risk"],
    "merchant_behavior": ["known_merchant", "new_low_risk_merchant", "new_high_risk_merchant",
                          "front_merchant", "cash_like"],
    "timing_profile": ["customer_normal", "business_hours", "night", "any"],
    # Hard simulation limits — a specification may never exceed these.
    "max_txns_per_card_per_day": 40,
    "max_amount_inr": 2_000_000.0,
    "min_amount_inr": 1.0,
    "max_variants_per_round": 4000,
}

# Champion / challenger governance (Phase 4).
# An adapted model is a CANDIDATE. It is promoted only if it clears every gate.
CHAMPION_CHALLENGER = {
    "min_attack_recall_gain": 0.05,    # candidate must beat champion on the new attack by this
    "max_fpr": 0.025,                  # absolute ceiling on legitimate false-positive rate
    "max_fpr_regression": 0.005,       # candidate may not worsen FPR beyond this vs champion
    "max_prior_recall_drop": 0.10,     # no previously-learned family may fall by more than this
    "max_overall_pr_auc_drop": 0.03,   # overall ranking quality must not materially regress
}

# Operational scenario for cost-sensitive reporting (Phase 4).
# These are ILLUSTRATIVE SYNTHETIC ASSUMPTIONS for a prototype, not Mastercard
# figures and not a claim about anyone's real economics. They exist so that a
# false-positive rate can be read as review volume and customer friction.
OPERATIONAL_SCENARIO = {
    "label": "synthetic illustrative scenario — not production economics",
    "monthly_authorizations": 10_000_000,
    "analyst_reviews_per_hour": 25,
    "step_up_abandonment_rate": 0.08,   # share of stepped-up genuine customers who drop off
    "avg_fraud_loss_inr": 18_500.0,     # assumed average loss per undetected fraudulent auth
    "avg_review_cost_inr": 55.0,        # assumed fully-loaded cost of one manual review
}
