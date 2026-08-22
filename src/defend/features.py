"""Feature engineering for the defense model.

Turns the raw transaction table into a numeric feature matrix built ONLY from
information available at the moment of authorization. Four column classes are
made explicit so leakage cannot creep in:

  RAW_COLUMNS          - the raw synthetic transaction fields.
  AUTH_TIME_FEATURES   - what the real-time model may use (== FEATURE_COLUMNS):
                         per-transaction attributes, per-card *historical*
                         context (velocity, device/geo change, amount z-score vs
                         the card's own past, MCC novelty, prior dispute rate),
                         and per-network *relational* counters (how many cards
                         this device has carried, how many cards this merchant
                         has seen, payee fan-in, new-card ratio).
  POST_OUTCOME_COLUMNS - known only AFTER the transaction settles (refund /
                         chargeback / auth result). Kept in the raw dataset for
                         analysis, but NEVER fed to the authorization-time model.
  ORACLE_COLUMNS       - simulator bookkeeping (who really produced the row).
                         A real defender does not have these; hard-blocked.

The relational counters are what a payment NETWORK can compute and a single
issuer or merchant cannot: they are maintained as running counters keyed on
device, IP and merchant, so they are O(1) lookups at authorization time rather
than a graph query. They are the signals that make card testing, mule rings and
transaction laundering visible as *network* patterns instead of as "this card is
unknown to us".

All historical and relational features depend only on rows strictly before the
current one (expanding().shift(), trailing rolling windows, running counters),
preserving temporal causality. Deterministic and stateless: same input rows ->
same features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

# Post-outcome fields — NOT knowable at authorization time. Hard-blocked from the
# feature matrix by assert_auth_time_safe(). Retained in the raw dataset only.
POST_OUTCOME_COLUMNS = ["refund_flag", "auth_result"]

# Simulator ground truth about who generated a row. Never available in
# production, so never a feature.
ORACLE_COLUMNS = ["is_fraud", "attack_type", "actor_role"]

FEATURE_COLUMNS = [
    "log_amount", "mcc_risk", "is_cnp", "is_upi", "is_3ds", "otp_verified", "is_new_payee",
    "log_distance", "hour", "is_night", "is_weekend", "account_age_days",
    "is_new_account", "is_foreign", "is_high_risk_country", "amount_zscore",
    "time_since_last_hours", "velocity_1h", "velocity_24h", "device_changed",
    "geo_changed", "card_history_depth", "near_threshold", "mcc_novel",
    "card_prior_dispute_rate",
    # --- behaviour-relative features (this card vs its own history) -------
    "amount_vs_card_max_prior", "distance_vs_card_max_prior",
    "card_mcc_share_prior", "is_cash_like",
    # --- merchant-relative features (this ticket vs this merchant's norm) -
    "merchant_amount_zscore",
    # --- network-level relational counters (auth-time computable) ---------
    "device_card_count_prior", "card_device_count_prior", "ip_card_count_prior",
    "merchant_card_fanin_prior", "merchant_new_card_ratio_prior",
    "merchant_txn_count_prior",
]

RELATIVE_FEATURES = [
    "amount_vs_card_max_prior", "distance_vs_card_max_prior",
    "card_mcc_share_prior", "merchant_amount_zscore",
]

CASH_LIKE_MCCS = {"6011_atm", "4829_money_transfer"}

# Counters that grow monotonically as the simulation window fills. Log-compressed
# before they reach the model so their distribution is stable over time.
_COUNT_FEATURES = {"device_card_count_prior", "card_device_count_prior",
                   "ip_card_count_prior", "merchant_card_fanin_prior",
                   "merchant_txn_count_prior"}
# Alias for readability at call sites.
AUTH_TIME_FEATURES = FEATURE_COLUMNS

NETWORK_FEATURES = [
    "device_card_count_prior", "card_device_count_prior", "ip_card_count_prior",
    "merchant_card_fanin_prior", "merchant_new_card_ratio_prior",
    "merchant_txn_count_prior",
]

_BLOCKED = set(POST_OUTCOME_COLUMNS) | set(ORACLE_COLUMNS)


def assert_auth_time_safe(feature_df: pd.DataFrame) -> None:
    """Raise if any post-outcome or oracle column leaked into the feature matrix."""
    leaked = [c for c in feature_df.columns if c in _BLOCKED]
    if leaked:
        raise ValueError(
            f"Authorization-time leakage: {leaked} are post-outcome or simulator-oracle "
            "fields and must not be used as model features.")


def _rolling_velocity(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Trailing 1h / 24h transaction counts per card, aligned to df row order."""
    v1 = np.ones(len(df))
    v24 = np.ones(len(df))
    pos = {idx: i for i, idx in enumerate(df.index)}
    for _, grp in df.groupby("card_id", sort=False):
        s = grp.set_index("timestamp")["amount"]
        c1 = s.rolling("1h").count().to_numpy()
        c24 = s.rolling("24h").count().to_numpy()
        for j, idx in enumerate(grp.index):
            v1[pos[idx]] = c1[j]
            v24[pos[idx]] = c24[j]
    return v1, v24


def _prior_distinct(df: pd.DataFrame, key: str, other: str) -> pd.Series:
    """Number of distinct `other` values seen against `key` strictly before now.

    Implemented as a running counter over first-time pairings, which is exactly
    how a network-side counter service would maintain it.
    """
    first_pair = (~df.duplicated(subset=[key, other])).astype(int)
    return first_pair.groupby(df[key]).cumsum() - first_pair


def _network_features(df: pd.DataFrame) -> pd.DataFrame:
    """Relational counters, computed on a globally time-ordered frame."""
    d = df.sort_values("timestamp", kind="mergesort")
    out = pd.DataFrame(index=d.index)

    out["device_card_count_prior"] = _prior_distinct(d, "device_id", "card_id")
    out["card_device_count_prior"] = _prior_distinct(d, "card_id", "device_id")
    out["ip_card_count_prior"] = _prior_distinct(d, "ip_prefix", "card_id")

    fanin = _prior_distinct(d, "merchant_id", "card_id")
    prior_txns = d.groupby("merchant_id").cumcount()
    out["merchant_card_fanin_prior"] = fanin
    out["merchant_txn_count_prior"] = prior_txns
    # Share of this merchant's prior traffic that was a first-time card. A normal
    # merchant converts repeat customers; a front merchant sees almost only new
    # cards, and a probing target sees a spike of them.
    out["merchant_new_card_ratio_prior"] = np.where(
        prior_txns.to_numpy() > 0,
        fanin.to_numpy() / np.maximum(prior_txns.to_numpy(), 1),
        np.nan)

    # Is this ticket unusual FOR THIS MERCHANT? A five-figure charge is ordinary
    # at a travel agent and extraordinary at a grocer, so an absolute amount
    # threshold punishes the wrong customers.
    mg = d.groupby("merchant_id")["amount"]
    m_mean = mg.transform(lambda s: s.expanding().mean().shift())
    m_std = mg.transform(lambda s: s.expanding().std().shift())
    out["merchant_amount_zscore"] = (d["amount"] - m_mean) / (m_std + 1.0)
    return out.reindex(df.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric feature DataFrame indexed like `df` (order preserved)."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    net = _network_features(df)

    df = df.sort_values(["card_id", "timestamp"])
    g = df.groupby("card_id", sort=False)
    df["card_history_depth"] = g.cumcount()
    df["time_since_last_hours"] = g["timestamp"].diff().dt.total_seconds() / 3600.0
    df["amt_mean_prev"] = g["amount"].transform(lambda s: s.expanding().mean().shift())
    df["amt_std_prev"] = g["amount"].transform(lambda s: s.expanding().std().shift())
    df["prev_device"] = g["device_id"].shift()
    df["prev_geo"] = g["geo_city"].shift()
    # Prior dispute rate: mean of refund_flag over the card's EARLIER transactions
    # only (shifted). Uses past outcomes known before T -> auth-time legal, and
    # gives friendly-fraud / refund-abuse a legitimate repeat-disputer signal.
    df["card_prior_dispute_rate"] = g["refund_flag"].transform(
        lambda s: s.astype(float).expanding().mean().shift())

    # MCC novelty: first time this card transacts at this MCC.
    df["mcc_novel"] = (~df.duplicated(subset=["card_id", "mcc"])).astype(int)
    # How much of this card's own past sits in this category? A first grocery
    # charge on a card that only ever buys groceries is not the same event as a
    # first money-transfer on that card.
    same_mcc = df.groupby(["card_id", "mcc"], sort=False).cumcount()
    df["card_mcc_share_prior"] = same_mcc / (df["card_history_depth"] + 1.0)
    # Running maxima of the card's own past, so "unusual for this cardholder"
    # is measured against that cardholder rather than a portfolio average.
    df["amt_max_prev"] = g["amount"].transform(lambda s: s.cummax().shift())
    df["dist_max_prev"] = g["distance_from_home_km"].transform(lambda s: s.cummax().shift())

    v1, v24 = _rolling_velocity(df)
    df["velocity_1h"] = v1
    df["velocity_24h"] = v24

    feats = pd.DataFrame(index=df.index)
    feats["log_amount"] = np.log1p(df["amount"].clip(lower=0))
    feats["mcc_risk"] = df["mcc_risk"].astype(float)
    feats["is_cnp"] = (df["channel"] == "card_cnp").astype(int)
    # UPI carries different risk semantics from a card authorization: the payer
    # authenticates on every payment, settlement is instant and irrevocable, and
    # the payee is an address rather than a card acceptor. Collapsing it into
    # "card-not-present" would hide all three.
    feats["is_upi"] = (df["channel"] == "upi").astype(int)
    feats["is_3ds"] = df["is_3ds"].astype(int)
    feats["otp_verified"] = df["otp_verified"].astype(int)
    feats["is_new_payee"] = df["is_new_payee"].astype(int)
    feats["log_distance"] = np.log1p(df["distance_from_home_km"].clip(lower=0))
    feats["hour"] = df["timestamp"].dt.hour
    feats["is_night"] = ((df["timestamp"].dt.hour < 6) | (df["timestamp"].dt.hour > 22)).astype(int)
    feats["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(int)
    feats["account_age_days"] = df["account_age_days"].astype(float)
    feats["is_new_account"] = (df["account_age_days"] < 90).astype(int)
    feats["is_foreign"] = (df["country"] != "IN").astype(int)
    feats["is_high_risk_country"] = df["country"].isin(config.HIGH_RISK_COUNTRIES).astype(int)
    feats["amount_zscore"] = ((df["amount"] - df["amt_mean_prev"]) /
                              (df["amt_std_prev"] + 1.0))
    feats["time_since_last_hours"] = df["time_since_last_hours"]
    feats["velocity_1h"] = df["velocity_1h"]
    feats["velocity_24h"] = df["velocity_24h"]
    feats["device_changed"] = ((df["device_id"] != df["prev_device"]) &
                               df["prev_device"].notna()).astype(int)
    feats["geo_changed"] = ((df["geo_city"] != df["prev_geo"]) &
                            df["prev_geo"].notna()).astype(int)
    feats["card_history_depth"] = np.log1p(df["card_history_depth"].astype(float))
    # proximity to a common velocity/structuring threshold (INR 5000)
    feats["near_threshold"] = ((df["amount"] > 4000) & (df["amount"] < 5000)).astype(int)
    feats["mcc_novel"] = df["mcc_novel"].astype(int)
    feats["card_prior_dispute_rate"] = df["card_prior_dispute_rate"]

    feats["amount_vs_card_max_prior"] = (np.log1p(df["amount"].clip(lower=0)) -
                                         np.log1p(df["amt_max_prev"].clip(lower=0)))
    feats["distance_vs_card_max_prior"] = (
        np.log1p(df["distance_from_home_km"].clip(lower=0)) -
        np.log1p(df["dist_max_prev"].clip(lower=0)))
    feats["card_mcc_share_prior"] = df["card_mcc_share_prior"].astype(float)
    feats["is_cash_like"] = df["mcc"].isin(CASH_LIKE_MCCS).astype(int)

    # Counters are log-compressed. They grow without bound as the window fills,
    # so on raw scale a model tuned on earlier traffic sees a systematically
    # different distribution later — the threshold then drifts off its
    # false-positive budget for reasons that have nothing to do with fraud.
    for c in NETWORK_FEATURES + ["merchant_amount_zscore"]:
        v = net[c].reindex(feats.index).astype(float)
        feats[c] = np.log1p(v.clip(lower=0)) if c in _COUNT_FEATURES else v

    # First-txn rows have undefined history; fill with neutral-but-honest defaults.
    feats["amount_zscore"] = feats["amount_zscore"].fillna(0.0)
    feats["time_since_last_hours"] = feats["time_since_last_hours"].fillna(24 * 30.0)
    feats["card_prior_dispute_rate"] = feats["card_prior_dispute_rate"].fillna(0.0)
    # A merchant's first-ever transaction has no prior new-card ratio; 0.5 is the
    # honest "no information" midpoint rather than a value that implies safety.
    feats["merchant_new_card_ratio_prior"] = feats["merchant_new_card_ratio_prior"].fillna(0.5)
    # No prior maximum on a card's first transaction: 0.0 means "no evidence this
    # is unusual", which is the honest default rather than an implied alarm.
    feats["amount_vs_card_max_prior"] = feats["amount_vs_card_max_prior"].fillna(0.0)
    feats["distance_vs_card_max_prior"] = feats["distance_vs_card_max_prior"].fillna(0.0)
    feats["merchant_amount_zscore"] = feats["merchant_amount_zscore"].fillna(0.0)

    out = feats[FEATURE_COLUMNS].astype(float)
    assert_auth_time_safe(out)  # guarantee no post-outcome field slipped in
    # out carries the original row labels; callers reindex to their desired order.
    return out


def build_xy(df: pd.DataFrame):
    """Return (X, y, attack_type) aligned. Preserves the input's row order."""
    feats = build_features(df)
    feats = feats.reindex(df.index)
    y = df.loc[feats.index, "is_fraud"].astype(int).to_numpy()
    attack = df.loc[feats.index, "attack_type"].to_numpy()
    return feats, y, attack
