"""Realistic *legitimate* card-transaction stream.

Each cardholder emits a stream over the simulation window shaped by their
behavioural archetype: category preference, spend level, device pool, travel and
night-time propensity, weekday/weekend shape and channel mix. On top of that:

  * REGULAR merchants — customers revisit the same places, so ``is_new_payee``
    is a genuine minority signal rather than a fraud shortcut.
  * RECURRING charges — subscriptions land repeatedly at a stable amount.
  * SHOPPING SESSIONS — a purchase sometimes triggers 1-3 more within a couple
    of hours, which is what makes short-window velocity a real (noisy) signal.
  * PAYDAY and WEEKEND clustering — volume is not uniform across the window.
  * CARD ISSUANCE DURING THE WINDOW — a minority of genuine cards make their
    first-ever authorization mid-window, so "no prior history" does not mean
    "fraud".

This legit backbone is what keeps the false-positive rate measurable and what
stops the defender from learning simulator artifacts instead of fraud behaviour.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import config
from src.generate import profiles as P

# Canonical column order for the transaction table.
COLUMNS = [
    "txn_id", "timestamp", "cardholder_id", "card_id", "merchant_id", "mcc",
    "mcc_risk", "amount", "channel", "country", "geo_city", "distance_from_home_km",
    "device_id", "ip_prefix", "is_3ds", "otp_verified", "is_new_payee",
    "refund_flag", "account_age_days", "auth_result", "is_fraud", "attack_type",
    "actor_role",
]

# actor_role documents WHO produced the row, for analysis only. It is never a
# model feature (a real defender does not know it) and is excluded from the
# feature matrix by construction.
ROLE_GENUINE = "genuine_customer"
ROLE_COVER = "fraud_actor_cover"      # legitimate-looking traffic from a fraud actor
ROLE_FRAUD = "fraud"


def _diurnal_hour(rng: np.random.Generator, night_prone: bool = False) -> float:
    """Mixture of morning, lunch and evening peaks; night-prone customers skew late."""
    p = [0.22, 0.24, 0.54] if night_prone else [0.30, 0.30, 0.40]
    comp = rng.choice([9, 13, 20], p=p)
    return float(np.clip(rng.normal(comp, 2.0), 0, 23.999))


def _day_probs(holder: P.Cardholder) -> np.ndarray:
    """Per-day sampling weights for one holder: weekday/weekend shape, payday
    clustering, and zero weight before the card is activated."""
    w = np.array([P.day_weight(d, holder.weekend_mult) for d in range(P.SIM_DAYS)],
                 dtype=float)
    if holder.activation_day > 0:
        w[: holder.activation_day] = 0.0
    if holder.lapse_day < P.SIM_DAYS:
        w[holder.lapse_day:] = 0.0
    if w.sum() <= 0:  # pragma: no cover - defensive
        w = np.ones(P.SIM_DAYS)
    return w / w.sum()


def generate_legit(
    holders: list[P.Cardholder],
    merchants: list[P.Merchant],
    n_transactions: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    sim_start = datetime.fromisoformat(P.SIM_START_ISO)
    R = config.LEGIT_REALISM
    all_cities = list(config.GEO_CLUSTERS.keys())
    foreign_cities = ["dubai", "singapore", "london"]
    by_id = {m.merchant_id: m for m in merchants}
    merch_by_mcc: dict[str, list[P.Merchant]] = {}
    for m in merchants:
        merch_by_mcc.setdefault(m.mcc, []).append(m)

    # Weight cardholders by their daily rate so busier holders get more rows.
    rates = np.array([h.daily_rate for h in holders], dtype=float)
    holder_probs = rates / rates.sum()
    day_probs = {h.card_id: _day_probs(h) for h in holders}

    rows: list[dict] = []
    k = 0
    while k < n_transactions:
        h = holders[int(rng.choice(len(holders), p=holder_probs))]
        day = int(rng.choice(P.SIM_DAYS, p=day_probs[h.card_id]))
        hour = (float(rng.uniform(0, 5)) if rng.random() < h.odd_hour_prob
                else _diurnal_hour(rng, night_prone=h.odd_hour_prob > 0.09))
        ts = sim_start + timedelta(days=day, hours=hour, minutes=int(rng.integers(0, 60)))

        # A purchase sometimes starts a short shopping session (1-3 more auths
        # inside a couple of hours) -> short-window velocity is a real signal.
        session = 1
        if rng.random() < R["burst_prob"]:
            lo, hi = R["burst_extra"]
            session += int(rng.integers(lo, hi + 1))

        for s in range(session):
            if k >= n_transactions:
                break
            t = ts if s == 0 else ts + timedelta(
                hours=float(rng.uniform(0.05, R["burst_window_hours"])))

            m, is_recurring = _pick_merchant(h, by_id, merch_by_mcc, merchants, rng, R)
            amount = _pick_amount(h, m, is_recurring, rng, R)
            geo_city, country = _pick_geo(h, all_cities, foreign_cities, rng, R, is_recurring)
            channel = _pick_channel(h, m, country, is_recurring, rng, R)
            if channel == "upi":
                amount = _upi_amount(amount, rng)
            dist = P.haversine_km(P.city_latlon(h.home_city), P.city_latlon(geo_city))
            dist += float(abs(rng.normal(0, 6)))  # local jitter (km)
            device, ip = _pick_device(h, rng, R, is_recurring)

            # On UPI the payer authenticates with a UPI PIN on every payment, so
            # "authenticated" is the norm rather than a risk-reducing signal — the
            # same reason authorized-push-payment scams settle so easily on it.
            is_3ds = ((channel == "card_cnp" and rng.random() < 0.6)
                      or channel == "upi")
            rows.append({
                "txn_id": f"T{k:08d}",
                "timestamp": t,
                "cardholder_id": h.cardholder_id,
                "card_id": h.card_id,
                "merchant_id": m.merchant_id,
                "mcc": m.mcc,
                "mcc_risk": m.risk,
                "amount": amount,
                "channel": channel,
                "country": country,
                "geo_city": geo_city,
                "distance_from_home_km": round(dist, 1),
                "device_id": device,
                "ip_prefix": ip,
                "is_3ds": bool(is_3ds),
                "otp_verified": bool(is_3ds),
                "is_new_payee": False,   # computed globally (first-visit) in simulate.py
                # refund_flag is a POST-OUTCOME field (not known at auth time); kept in
                # the raw dataset for analysis only, never used by the auth-time model.
                "refund_flag": bool(rng.random() < R["legit_dispute_prob"]),
                "account_age_days": max(1, h.account_age_days + day - h.activation_day),
                "auth_result": "approved",
                "is_fraud": 0,
                "attack_type": "legit",
                "actor_role": ROLE_GENUINE,
            })
            k += 1
    return pd.DataFrame(rows, columns=COLUMNS)


def _pick_merchant(h, by_id, merch_by_mcc, merchants, rng, R):
    """Recurring charge, a regular merchant, or a genuinely new one."""
    if h.recurring_merchants and rng.random() < R["subscription_share"]:
        mid = str(rng.choice(h.recurring_merchants))
        return by_id.get(mid, merchants[0]), True
    if h.regular_merchants and rng.random() < R["regulars_repeat_prob"]:
        mid = str(rng.choice(h.regular_merchants))
        return by_id.get(mid, merchants[0]), False
    mcc = str(rng.choice(h.preferred_mccs))
    pool = merch_by_mcc.get(mcc) or merchants
    w = np.array([m.popularity for m in pool], dtype=float)
    return pool[int(rng.choice(len(pool), p=w / w.sum()))], False


def _pick_channel(h, m, country, is_recurring, rng, R) -> str:
    """Card-present, card-not-present, or UPI.

    UPI is domestic-only and merchant-dependent, and recurring card-on-file
    charges stay on cards.
    """
    U = config.UPI_REALISM
    if (not is_recurring and country == "IN" and getattr(m, "accepts_upi", False)
            and rng.random() < U["legit_share"]):
        return "upi"
    return ("card_cnp" if rng.random() < np.clip(m.cnp_prob + h.cnp_shift, 0.0, 1.0)
            else "card_cp")


def _upi_amount(amount: float, rng: np.random.Generator) -> float:
    """UPI tickets are smaller and capped, and cluster on round rupee values."""
    U = config.UPI_REALISM
    v = amount * U["ticket_scale"] * float(rng.uniform(0.8, 1.25))
    v = min(v, U["max_ticket_inr"])
    if rng.random() < 0.35:                       # people type round numbers
        v = float(round(v / 10.0) * 10)
    return float(np.round(max(v, 1.0), 2))


def _pick_amount(h, m, is_recurring, rng, R) -> float:
    if is_recurring and m.merchant_id in h.recurring_amount:
        # Subscriptions repeat at a stable price (small plan/tax jitter only).
        return float(np.round(h.recurring_amount[m.merchant_id] * rng.uniform(0.99, 1.01), 2))
    amount = float(np.exp(rng.normal(m.ticket_mu, m.ticket_sigma)) * h.spend_factor)
    if rng.random() < R["high_value_prob"]:            # genuine large purchase
        lo, hi = R["high_value_mult"]
        amount *= float(rng.uniform(lo, hi))
    return float(np.round(amount, 2))


def _pick_geo(h, all_cities, foreign_cities, rng, R, is_recurring):
    if is_recurring:                       # a subscription bills from home
        return h.home_city, h.home_country
    if rng.random() < h.travel_prob:
        if rng.random() < h.foreign_given_travel:
            city = str(rng.choice(foreign_cities))
        else:
            city = str(rng.choice(all_cities))
    else:
        city = h.home_city
    country = _country_for_city(city)
    # Rare genuine spend in a high-risk jurisdiction: keeps that feature from
    # being a fraud-only flag (which would make the matching rule a shortcut).
    if city != h.home_city and rng.random() < R["high_risk_country_prob"] * 40:
        country = str(rng.choice(sorted(config.HIGH_RISK_COUNTRIES)))
    return city, country


def _pick_device(h, rng, R, is_recurring):
    """Primary most of the time; a secondary owned device sometimes; rarely a
    genuine brand-new (replacement) device. Recurring charges are card-on-file
    and carry the device of record."""
    if is_recurring:
        return h.usual_device_id, h.usual_ip_prefix
    if rng.random() < R["new_device_prob"]:
        return (f"NEWLEG{int(rng.integers(0, 1_000_000)):07d}",
                f"{int(rng.integers(2,223))}.{int(rng.integers(0,255))}")
    if len(h.device_pool) > 1 and rng.random() < R["secondary_device_prob"]:
        return str(rng.choice(h.device_pool[1:])), h.usual_ip_prefix
    return h.usual_device_id, h.usual_ip_prefix


def _country_for_city(city: str) -> str:
    return {"dubai": "AE", "singapore": "SG", "london": "GB"}.get(city, "IN")
