"""Transaction-level fraud injectors — one per simulatable attack family.

Each injector consumes a validated :class:`~src.generate.attack_spec.AttackSpec`
and returns transaction dicts in the same schema as the legitimate generator, so
the defender sees one homogeneous authorization stream. Every function is
deterministic given the passed RNG.

Two design rules keep this simulator honest:

1. **Fraud actors have a history.** Mule accounts, bust-out accounts and front
   merchants build ordinary-looking traffic before they are used. Those *cover*
   rows are emitted with ``is_fraud=0`` — because at authorization time they are
   genuinely not fraud — and only the abusive transactions carry the label. This
   is what stops "this card has no history" from being a synonym for "fraud".
2. **Attacks reuse things.** Probing hits the same card and the same merchants
   repeatedly; mimicry shops at the victim's own regular merchant; laundering
   pushes many cards through one front merchant. Without reuse, every fraudulent
   row would trivially be a first-ever card/merchant pair, and the defender would
   learn that artifact instead of fraud behaviour.

The spec's dials (amount, velocity, device, geography, merchant, timing) are what
the closed-loop red team moves when it evolves a family. Nothing here is
hardcoded to a "stealth" number.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

import config
from src.generate import profiles as P
from src.generate.attack_spec import AttackSpec, as_spec
from src.generate.base_generator import (COLUMNS, ROLE_COVER, ROLE_FRAUD,
                                         _country_for_city)

SIM_START = datetime.fromisoformat(P.SIM_START_ISO)
RISKY_MCCS = ["5816_digital_goods", "5967_direct_mktg", "7995_gambling", "4829_money_transfer"]
CASH_MCCS = ["6011_atm", "4829_money_transfer"]
LOW_RISK_MCCS = ["5411_grocery", "5812_restaurant", "5541_fuel", "4900_utilities"]
FR = config.FRAUD_REALISM

# Day-of-window sampling weights. Fraud rides the same weekly/payday traffic
# shape as genuine spend — if it did not, "transacted on an ordinary Tuesday"
# would itself become a fraud signal, which is a simulator artifact, not fraud.
_DAY_W = np.array([P.day_weight(d) for d in range(P.SIM_DAYS)], dtype=float)
_DAY_P = _DAY_W / _DAY_W.sum()


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _pick_day(rng: np.random.Generator) -> int:
    return int(rng.choice(P.SIM_DAYS, p=_DAY_P))


def _hour_for(spec: AttackSpec, rng: np.random.Generator) -> float:
    """Time-of-day consistent with the specification's timing profile."""
    if spec.timing_profile == "night":
        return float(rng.uniform(0.5, 5.0))
    if spec.timing_profile == "business_hours":
        return float(np.clip(rng.normal(14.0, 2.2), 8.0, 19.5))
    if spec.timing_profile == "customer_normal":
        return float(np.clip(rng.normal(rng.choice([9, 13, 20], p=[0.3, 0.3, 0.4]), 2.0),
                             0, 23.999))
    return float(rng.uniform(0, 24))


def _ts(rng: np.random.Generator, spec: AttackSpec, day: int | None = None) -> datetime:
    day = _pick_day(rng) if day is None else int(np.clip(day, 0, P.SIM_DAYS - 1))
    return SIM_START + timedelta(days=day, hours=_hour_for(spec, rng),
                                 minutes=int(rng.integers(0, 60)))


def _row(attack_type: str, role: str = ROLE_FRAUD, fraud: int = 1) -> dict:
    """Row prefilled with neutral defaults; injectors override the signal fields."""
    return {c: None for c in COLUMNS} | {
        "mcc_risk": 0.5, "channel": "card_cnp", "country": "IN",
        "is_3ds": False, "otp_verified": False, "is_new_payee": False,
        "refund_flag": False, "auth_result": "approved",
        "is_fraud": fraud, "attack_type": attack_type, "actor_role": role,
    }


def _amount(mcc: str, rng: np.random.Generator, scale: float = 1.0) -> float:
    spec_mcc = config.MCC_CATALOG[mcc]
    v = float(np.exp(rng.normal(spec_mcc["mu"], spec_mcc["sigma"])) * scale)
    return float(np.round(np.clip(v, config.ATTACK_SPEC_BOUNDS["min_amount_inr"],
                                  config.ATTACK_SPEC_BOUNDS["max_amount_inr"]), 2))


def _n_hits(spec: AttackSpec, rng: np.random.Generator) -> int:
    lo, hi = spec.txns_per_card
    return int(rng.integers(lo, hi + 1))


def _geo_for(spec: AttackSpec, home_city: str, rng: np.random.Generator):
    """(city, country, distance_km) consistent with the geography dial.

    The city is chosen so that the realized distance from home falls inside the
    band the specification asked for. Picking a city first and reporting whatever
    distance results would let a spec that says "stay plausible" emit a
    transaction 5,000 km from home — the attack would then be caught by a
    geography signal the red team had explicitly decided to stop using, and the
    lineage story would be describing something the data does not contain.
    """
    cities = list(config.GEO_CLUSTERS)
    foreign = ["dubai", "singapore", "london"]
    g = spec.geo_behavior
    lo, hi = spec.geo_km_range

    if g == "home":
        city = home_city
    else:
        # Candidates whose true distance from home lands inside the requested band.
        banded = [c for c in cities
                  if lo <= P.haversine_km(P.city_latlon(home_city), P.city_latlon(c)) <= hi]
        if g == "plausible":
            # "Plausible" means the cardholder could credibly be there: usually
            # home, occasionally somewhere close by.
            city = home_city if (rng.random() < 0.7 or not banded) else str(rng.choice(banded))
        elif g == "domestic_far":
            pool = [c for c in banded if c not in foreign] or \
                   [c for c in cities if c not in foreign and c != home_city]
            city = str(rng.choice(pool))
        else:                               # foreign / high_risk
            pool = [c for c in banded if c in foreign] or \
                   [c for c in foreign if c != home_city] or foreign
            city = str(rng.choice(pool))

    country = _country_for_city(city)
    if g == "high_risk" or (city != home_city and rng.random() < FR["geo_high_risk_prob"]
                            and g in ("foreign", "domestic_far")):
        country = str(rng.choice(sorted(config.HIGH_RISK_COUNTRIES)))

    if city == home_city:
        # Local movement only: keep it inside the band's lower reaches.
        dist = float(abs(rng.normal(0, max(1.0, min(hi, 25.0) * 0.4))))
    else:
        dist = P.haversine_km(P.city_latlon(home_city), P.city_latlon(city))
        dist += float(abs(rng.normal(0, 8)))
    return city, country, round(float(min(dist, hi if hi > 0 else dist)), 1)


def _device_for(spec: AttackSpec, h: P.Cardholder, rng: np.random.Generator,
                shared: str | None = None):
    """(device_id, ip_prefix) consistent with the device dial."""
    if rng.random() < spec.device_trust:
        if spec.device_behavior == "secondary_device" and len(h.device_pool) > 1:
            return str(rng.choice(h.device_pool[1:])), h.usual_ip_prefix
        return h.usual_device_id, h.usual_ip_prefix
    if spec.device_behavior == "shared_device" and shared is not None:
        return shared, f"45.{int(rng.integers(0, 255))}"
    return (f"NEWDEV{int(rng.integers(0, 1_000_000)):07d}",
            f"{int(rng.integers(2,223))}.{int(rng.integers(0,255))}")


def _weighted_pick(pool, rng: np.random.Generator):
    """Pick a merchant in proportion to its traffic, exactly as the legitimate
    generator does. Sampling attack merchants uniformly while genuine spend
    follows a heavy-tailed popularity curve would make "quiet merchant" a fraud
    signal — a property of the simulator, not of fraud."""
    if not pool:
        return None
    w = np.array([m.popularity for m in pool], dtype=float)
    return pool[int(rng.choice(len(pool), p=w / w.sum()))]


def _merchant_for(spec: AttackSpec, h: P.Cardholder | None, merchants, by_id,
                  rng: np.random.Generator, front=None):
    """A merchant consistent with the merchant dial."""
    b = spec.merchant_behavior
    if b == "front_merchant" and front is not None:
        return front
    if b == "known_merchant" and h is not None and h.regular_merchants:
        mid = str(rng.choice(h.regular_merchants))
        if mid in by_id:
            return by_id[mid]
    if b == "cash_like":
        pool = [m for m in merchants if m.mcc in CASH_MCCS] or merchants
    elif b == "new_high_risk_merchant":
        pool = [m for m in merchants if m.mcc in RISKY_MCCS] or merchants
    elif b == "new_low_risk_merchant":
        pool = [m for m in merchants if m.mcc in LOW_RISK_MCCS] or merchants
    else:
        pool = merchants
    return _weighted_pick(pool, rng)


def _channel_for(m: P.Merchant, rng: np.random.Generator, force_cnp: bool = False,
                 upi_bias: float = 0.0) -> str:
    """Channel drawn from the merchant's own channel mix.

    Forcing every fraudulent row onto one channel would make the channel flag a
    near-perfect label; real fraud follows the merchant's channel reality.
    ``upi_bias`` raises the chance of settling on UPI for the families that
    genuinely do — authorized push payments and mule layering in India mostly run
    on UPI, not on cards.
    """
    if upi_bias and getattr(m, "accepts_upi", False) and rng.random() < upi_bias:
        return "upi"
    if force_cnp:
        return "card_cnp"
    return "card_cnp" if rng.random() < m.cnp_prob else "card_cp"


def _account_age(h: P.Cardholder, day: int) -> int:
    """Account age at the moment of the transaction, on the SAME clock the
    legitimate generator uses. Using a different clock for fraud would make
    account age an artificial giveaway."""
    return max(1, h.account_age_days + day - h.activation_day)


def _synthetic_holder(rng: np.random.Generator, prefix: str, age_days: int,
                      city: str | None = None) -> P.Cardholder:
    """A fabricated identity that behaves like a cardholder profile, so cover
    traffic for a fraud actor goes through exactly the same code paths."""
    i = int(rng.integers(0, 1_000_000))
    home = city or str(rng.choice(list(config.GEO_CLUSTERS)[:7]))
    return P.Cardholder(
        cardholder_id=f"{prefix}ID{i:06d}", card_id=f"{prefix}C{i:06d}",
        home_city=home, home_country=_country_for_city(home),
        spend_factor=float(np.clip(rng.lognormal(0.0, 0.3), 0.4, 2.0)),
        preferred_mccs=list(rng.choice(list(config.MCC_CATALOG), size=3, replace=False)),
        usual_device_id=f"DEV{prefix}{i:06d}",
        usual_ip_prefix=f"{int(rng.integers(10, 223))}.{int(rng.integers(0,255))}",
        account_age_days=age_days, daily_rate=0.4, risk_tier=0.5,
        device_pool=[f"DEV{prefix}{i:06d}"], archetype="thin_file_new",
    )


def _cover_row(actor: P.Cardholder, merchants, rng: np.random.Generator, day: int,
               attack_type: str, merchant=None) -> dict:
    """One ordinary-looking transaction by a fraud actor's account.

    Labelled ``is_fraud=0`` on purpose: at authorization time this transaction is
    indistinguishable from genuine spend, and calling it fraud would hand the
    model a label it could not possibly earn in production.
    """
    m = merchant if merchant is not None else merchants[int(rng.integers(len(merchants)))]
    r = _row("legit", role=ROLE_COVER, fraud=0)
    upi = getattr(m, "accepts_upi", False) and rng.random() < config.UPI_REALISM["legit_share"]
    ts = SIM_START + timedelta(days=day,
                               hours=float(np.clip(rng.normal(
                                   rng.choice([9, 13, 20], p=[0.3, 0.3, 0.4]), 2.0), 0, 23.99)),
                               minutes=int(rng.integers(0, 60)))
    cnp = rng.random() < m.cnp_prob
    r.update({
        "timestamp": ts,
        "cardholder_id": actor.cardholder_id, "card_id": actor.card_id,
        "merchant_id": m.merchant_id, "mcc": m.mcc, "mcc_risk": m.risk,
        "amount": _amount(m.mcc, rng,
                          scale=actor.spend_factor * (config.UPI_REALISM["ticket_scale"]
                                                      if upi else 1.0)),
        "channel": "upi" if upi else ("card_cnp" if cnp else "card_cp"),
        "geo_city": actor.home_city, "country": actor.home_country,
        "distance_from_home_km": round(float(abs(rng.normal(0, 7))), 1),
        "device_id": actor.usual_device_id, "ip_prefix": actor.usual_ip_prefix,
        "is_3ds": bool(upi or (cnp and rng.random() < 0.6)),
        "otp_verified": bool(upi or (cnp and rng.random() < 0.6)),
        "account_age_days": max(1, actor.account_age_days + day),
    })
    return r


def _by_id(merchants) -> dict:
    return {m.merchant_id: m for m in merchants}


# --------------------------------------------------------------------------- #
# Injectors
# --------------------------------------------------------------------------- #
def card_testing(holders, merchants, rng, n, spec=None):
    """Distributed validation of stolen card numbers.

    Probing bursts reuse a small pool of test cards across a small pool of
    merchants inside a short window, so the pattern shows up as genuine
    short-window velocity and merchant repetition rather than as a crowd of
    unrelated one-off rows.
    """
    spec = as_spec(spec, "card_testing")
    rows = []
    digital = [m for m in merchants if m.mcc in ("5816_digital_goods", "5967_direct_mktg",
                                                 "5999_retail")] or merchants
    remaining = n
    while remaining > 0:
        shared_dev = f"BOTNET{int(rng.integers(0, 60)):03d}"
        shared_ip = f"45.{int(rng.integers(0, 255))}"
        day = _pick_day(rng)
        base = _ts(rng, spec, day)
        n_merch = int(rng.integers(*FR["testing_cover_merchants"]))
        burst_merchants = [_weighted_pick(digital, rng) for _ in range(max(2, n_merch))]
        # Deliberately small bursts. A handful of very large bursts would put the
        # whole family on two or three days of the window, which starves a
        # chronological split and makes per-family recall a lottery.
        n_cards = int(rng.integers(3, 7))
        for _ in range(n_cards):
            if remaining <= 0:
                break
            card = f"TESTC{int(rng.integers(0, 1_000_000)):06d}"
            holder_id = f"SYN{int(rng.integers(0, 1_000_000)):06d}"
            probes = min(remaining, int(rng.integers(*FR["testing_probes_per_card"])))
            remaining -= probes
            for j in range(probes):
                m = burst_merchants[j % len(burst_merchants)]
                r = _row("card_testing")
                r.update({
                    "timestamp": base + timedelta(minutes=int(rng.integers(0, 90))),
                    "cardholder_id": holder_id, "card_id": card,
                    "merchant_id": m.merchant_id, "mcc": m.mcc, "mcc_risk": m.risk,
                    "amount": _amount(m.mcc, rng, scale=spec.amount_scale),
                    "channel": "card_cnp", "country": "IN", "geo_city": "mumbai",
                    "distance_from_home_km": round(float(rng.uniform(0, 20)), 1),
                    "device_id": shared_dev, "ip_prefix": shared_ip,
                    "account_age_days": int(rng.integers(1, 40)),
                    # A high decline ratio is the classic tell. It is a POST-outcome
                    # field, so the auth-time model never sees it; it stays in the
                    # raw data for analysis and for the acquirer-side story.
                    "auth_result": ("declined" if rng.random() < 0.35 + 0.3 * spec.intensity
                                    else "approved"),
                })
                rows.append(r)
    return rows


def bust_out(holders, merchants, rng, n, spec=None):
    """Aged synthetic account: months of ordinary spend, then a drain to the limit.

    The grooming period is emitted as cover traffic (label 0). Without it the
    account would have no history at all, and the defender would be learning
    "unknown card" rather than "an account whose behaviour just broke".
    """
    spec = as_spec(spec, "bust_out")
    rows = []
    cash = [m for m in merchants if m.mcc in CASH_MCCS] or merchants
    ordinary = [m for m in merchants if m.mcc in LOW_RISK_MCCS] or merchants
    remaining = n
    while remaining > 0:
        actor = _synthetic_holder(rng, "BUST", int(rng.integers(45, 900)))
        ramp = int(rng.integers(*FR["bustout_ramp_days"]))
        # Pick the DAY OF THE BUST first, then work the grooming period backwards.
        # Choosing the start day first and adding the ramp would push almost every
        # bust-out into the last weeks of the window — a property of the loop that
        # generated it, not of bust-out fraud, and one that would quietly starve a
        # chronological train/test split of this family.
        bust_day = _pick_day(rng)
        start_day = max(0, bust_day - ramp)
        span = max(1, bust_day - start_day)
        # --- grooming: ordinary-looking spend on the actor's own device -----
        n_cover = int(rng.integers(*FR["bustout_cover_txns"]))
        for _ in range(n_cover):
            d = start_day + int(rng.integers(0, span))
            rows.append(_cover_row(actor, ordinary, rng, min(d, P.SIM_DAYS - 1), "bust_out"))
        # --- the bust-out itself -------------------------------------------
        hits = min(remaining, _n_hits(spec, rng) + 1)
        remaining -= hits
        for _ in range(hits):
            m = _weighted_pick(cash, rng)
            city, country, dist = _geo_for(spec, actor.home_city, rng)
            device, ip = _device_for(spec, actor, rng)
            r = _row("bust_out")
            r.update({
                "timestamp": _ts(rng, spec, bust_day),
                "cardholder_id": actor.cardholder_id, "card_id": actor.card_id,
                "merchant_id": m.merchant_id, "mcc": m.mcc, "mcc_risk": m.risk,
                "amount": _amount(m.mcc, rng, scale=spec.amount_scale * actor.spend_factor),
                "channel": _channel_for(m, rng), "geo_city": city, "country": country,
                "distance_from_home_km": dist,
                "device_id": device, "ip_prefix": ip,
                "account_age_days": max(1, actor.account_age_days + bust_day),
            })
            rows.append(r)
    return rows


def account_takeover(holders, merchants, rng, n, spec=None):
    """Compromised genuine account spent by someone else.

    The lineage this family follows is the heart of the closed loop: generation 0
    runs from an attacker device in a far geography, so ``device_changed`` fires.
    As the red team evolves it toward a session-hijack / on-device-malware
    takeover, the spend comes from the victim's OWN device and a plausible
    geography, and that top signal stops firing. What survives is the residual
    behavioural break — amount, merchant novelty, category, timing — which is
    what the adapted defense has to learn.
    """
    spec = as_spec(spec, "account_takeover")
    by_id = _by_id(merchants)
    rows = []
    remaining = n
    while remaining > 0:
        h = holders[int(rng.integers(len(holders)))]
        hits = min(remaining, _n_hits(spec, rng))
        remaining -= hits
        day = _pick_day(rng)
        city, country, dist = _geo_for(spec, h.home_city, rng)
        device, ip = _device_for(spec, h, rng)
        sticky_merchant = _merchant_for(spec, h, merchants, by_id, rng)
        for j in range(hits):
            same_merchant = j > 0 and rng.random() < FR["victim_same_merchant_prob"]
            m = sticky_merchant if same_merchant else _merchant_for(spec, h, merchants, by_id, rng)
            t_day = min(P.SIM_DAYS - 1,
                        day + int(spec.velocity_window_hours // 24) * j
                        if spec.velocity_profile == "low_and_slow" else day)
            r = _row("account_takeover")
            r.update({
                "timestamp": _ts(rng, spec, t_day),
                "cardholder_id": h.cardholder_id, "card_id": h.card_id,
                "merchant_id": m.merchant_id, "mcc": m.mcc, "mcc_risk": m.risk,
                "amount": _amount(m.mcc, rng, scale=spec.amount_scale * h.spend_factor),
                "channel": _channel_for(m, rng), "geo_city": city, "country": country,
                "distance_from_home_km": dist,
                "device_id": device, "ip_prefix": ip,
                "is_3ds": rng.random() < 0.3,
                "account_age_days": _account_age(h, t_day),
            })
            rows.append(r)
    return rows


def adversarial_mimicry(holders, merchants, rng, n, spec=None):
    """Stolen-card spend deliberately shaped onto the victim's own centroid.

    Shops at the victim's REGULAR merchant, at the victim's usual ticket size, in
    the victim's city, at the victim's usual hours. The only reliable deviation
    is the device — and even that is a device the card has sometimes used. This
    family is expected to stay hard; it is reported as a residual frontier rather
    than solved.
    """
    spec = as_spec(spec, "adversarial_mimicry")
    by_id = _by_id(merchants)
    rows = []
    for _ in range(n):
        h = holders[int(rng.integers(len(holders)))]
        m = _merchant_for(spec, h, merchants, by_id, rng)
        city, country, dist = _geo_for(spec, h.home_city, rng)
        device, ip = _device_for(spec, h, rng)
        day = _pick_day(rng)
        jitter = 1.0 + rng.normal(0, 0.08 + 0.10 * spec.intensity)
        r = _row("adversarial_mimicry")
        r.update({
            "timestamp": _ts(rng, spec, day),
            "cardholder_id": h.cardholder_id, "card_id": h.card_id,
            "merchant_id": m.merchant_id, "mcc": m.mcc, "mcc_risk": m.risk,
            "amount": _amount(m.mcc, rng, scale=spec.amount_scale * h.spend_factor) * abs(jitter),
            # Mimicry copies the victim's channel mix, UPI included — otherwise the
            # rail itself would become a tell.
            "channel": _channel_for(m, rng, upi_bias=config.UPI_REALISM["legit_share"]),
            "geo_city": city, "country": country,
            "distance_from_home_km": dist,
            "device_id": device, "ip_prefix": ip,
            "is_3ds": True, "otp_verified": True,
            "account_age_days": _account_age(h, day),
        })
        rows.append(r)
    return rows


def velocity_smurfing(holders, merchants, rng, n, spec=None):
    """A mule ring splitting value into sub-threshold amounts.

    Each mule account carries cover traffic and then pushes several
    just-under-the-limit transactions through one front merchant inside a day, so
    the pattern is visible as real velocity and real payee fan-in rather than as
    a scatter of unrelated single rows.
    """
    spec = as_spec(spec, "velocity_smurfing")
    rows = []
    threshold = 5000.0
    ordinary = [m for m in merchants if m.mcc in LOW_RISK_MCCS] or merchants
    remaining = n
    while remaining > 0:
        ring_merchant = _weighted_pick(merchants, rng)
        ring_dev = f"MULE{int(rng.integers(0, 300)):03d}"
        ring_size = int(rng.integers(2, 5))
        day = _pick_day(rng)
        for _ in range(ring_size):
            if remaining <= 0:
                break
            # Mule accounts are frequently long-lived recruited accounts, not
            # freshly minted ones, so their age spans the genuine range.
            actor = _synthetic_holder(rng, "MULE", int(rng.integers(20, 1500)),
                                      city="hyderabad")
            for _ in range(int(rng.integers(*FR["mule_cover_txns"]))):
                d = int(np.clip(day - int(rng.integers(1, 40)), 0, P.SIM_DAYS - 1))
                rows.append(_cover_row(actor, ordinary, rng, d, "velocity_smurfing"))
            hits = min(remaining, max(2, _n_hits(spec, rng)))
            remaining -= hits
            base = _ts(rng, spec, day)
            for _ in range(hits):
                device, ip = _device_for(spec, actor, rng, shared=ring_dev)
                r = _row("velocity_smurfing")
                r.update({
                    "timestamp": base + timedelta(
                        hours=float(rng.uniform(0, spec.velocity_window_hours or 6.0))),
                    "cardholder_id": actor.cardholder_id, "card_id": actor.card_id,
                    "merchant_id": ring_merchant.merchant_id, "mcc": ring_merchant.mcc,
                    "mcc_risk": ring_merchant.risk,
                    "amount": float(np.round(threshold * rng.uniform(0.85, 0.99)
                                             * min(1.0, spec.amount_scale), 2)),
                    # Mule layering in India runs largely on UPI.
                    "channel": _channel_for(ring_merchant, rng, upi_bias=0.55),
                    "geo_city": actor.home_city, "country": "IN",
                    "distance_from_home_km": round(float(rng.uniform(0, 30)), 1),
                    "device_id": device, "ip_prefix": ip,
                    "account_age_days": max(1, actor.account_age_days + day),
                })
                rows.append(r)
    return rows


def merchant_laundering(holders, merchants, rng, n, spec=None):
    """Stolen cards pushed through a controlled front merchant.

    The front merchant also processes a little genuine-looking business (cover
    traffic), and the same stolen card is charged more than once, so the
    detectable pattern is the merchant's abnormal new-card ratio and payee
    fan-in — a network-level signal — rather than "this card is unknown".
    """
    spec = as_spec(spec, "merchant_laundering")
    rows = []
    mcc = str(rng.choice(["5999_retail", "5732_electronics", "5816_digital_goods"]))
    round_amounts = [999.0, 1999.0, 4999.0, 9999.0]
    remaining = n
    while remaining > 0:
        front_id = f"MFAKE{int(rng.integers(0, 999)):03d}"
        front = P.Merchant(merchant_id=front_id, mcc=mcc, city="mumbai", country="IN",
                           cnp_prob=0.95, ticket_mu=config.MCC_CATALOG[mcc]["mu"],
                           ticket_sigma=config.MCC_CATALOG[mcc]["sigma"], risk=0.8,
                           is_synthetic=True, risk_tier="high", accepts_upi=True,
                           popularity=1.0)
        for _ in range(int(rng.integers(*FR["laundering_cover_txns"]))):
            actor = holders[int(rng.integers(len(holders)))]
            rows.append(_cover_row(actor, [front], rng, _pick_day(rng),
                                   "merchant_laundering", merchant=front))
        n_cards = int(rng.integers(3, 8))
        for _ in range(n_cards):
            if remaining <= 0:
                break
            # Laundering runs REAL compromised cards through the front merchant;
            # a minority are fabricated harvest accounts.
            if rng.random() < 0.7:
                h = holders[int(rng.integers(len(holders)))]
            else:
                h = _synthetic_holder(rng, "HARV", int(rng.integers(5, 1200)), city="mumbai")
            hits = min(remaining, max(1, _n_hits(spec, rng)))
            remaining -= hits
            day = _pick_day(rng)
            device, ip = _device_for(spec, h, rng, shared=f"DEVH{int(rng.integers(0, 400)):04d}")
            for _ in range(hits):
                r = _row("merchant_laundering")
                r.update({
                    "timestamp": _ts(rng, spec, day),
                    "cardholder_id": h.cardholder_id, "card_id": h.card_id,
                    "merchant_id": front_id, "mcc": mcc, "mcc_risk": front.risk,
                    "amount": (float(rng.choice(round_amounts)) * min(1.0, spec.amount_scale)
                               if rng.random() < 0.55
                               else _amount(mcc, rng, scale=spec.amount_scale)),
                    "channel": _channel_for(front, rng, upi_bias=0.45),
                    "geo_city": "mumbai", "country": "IN",
                    "distance_from_home_km": round(float(rng.uniform(0, 50)), 1),
                    "device_id": device, "ip_prefix": ip,
                    "account_age_days": _account_age(h, day),
                    "refund_flag": rng.random() < 0.15,  # chargeback lag (post-outcome)
                })
                rows.append(r)
    return rows


def friendly_fraud(holders, merchants, rng, n, spec=None):
    """A genuine purchase by the genuine cardholder, disputed afterwards.

    There is essentially nothing to see at authorization time — the customer, the
    device, the merchant and the amount are all normal. It is included precisely
    so the honest answer ("this is not an authorization-time problem") can be
    measured rather than asserted.
    """
    spec = as_spec(spec, "friendly_fraud")
    by_id = _by_id(merchants)
    rows = []
    for _ in range(n):
        h = holders[int(rng.integers(len(holders)))]
        m = _merchant_for(spec, h, merchants, by_id, rng)
        day = _pick_day(rng)
        r = _row("friendly_fraud")
        r.update({
            "timestamp": _ts(rng, spec, day),
            "cardholder_id": h.cardholder_id, "card_id": h.card_id,
            "merchant_id": m.merchant_id, "mcc": m.mcc, "mcc_risk": m.risk,
            "amount": _amount(m.mcc, rng, scale=spec.amount_scale * h.spend_factor),
            "channel": _channel_for(m, rng, upi_bias=0.30), "geo_city": h.home_city,
            "country": h.home_country,
            "distance_from_home_km": round(float(abs(rng.normal(0, 10))), 1),
            "device_id": h.usual_device_id, "ip_prefix": h.usual_ip_prefix,
            "is_3ds": True, "otp_verified": True,
            "account_age_days": _account_age(h, day),
            "refund_flag": True,  # the defining signal — and post-outcome only
        })
        rows.append(r)
    return rows


def otp_relay(holders, merchants, rng, n, spec=None):
    """A step-up challenge that was passed by relaying the code from the victim.

    The transaction therefore arrives fully authenticated. Generation 0 still
    runs on the attacker's device; as the family evolves toward an on-device
    relay, that tell disappears and only the authenticated-high-value-on-a-risky-
    category pattern remains.
    """
    spec = as_spec(spec, "otp_relay")
    by_id = _by_id(merchants)
    rows = []
    remaining = n
    while remaining > 0:
        h = holders[int(rng.integers(len(holders)))]
        hits = min(remaining, _n_hits(spec, rng))
        remaining -= hits
        day = _pick_day(rng)
        device, ip = _device_for(spec, h, rng)
        for j in range(hits):
            m = _merchant_for(spec, h, merchants, by_id, rng)
            t_day = min(P.SIM_DAYS - 1, day + (j if spec.velocity_profile == "low_and_slow" else 0))
            r = _row("otp_relay")
            r.update({
                "timestamp": _ts(rng, spec, t_day),
                "cardholder_id": h.cardholder_id, "card_id": h.card_id,
                "merchant_id": m.merchant_id, "mcc": m.mcc, "mcc_risk": m.risk,
                "amount": _amount(m.mcc, rng, scale=spec.amount_scale * h.spend_factor),
                # A relayed UPI PIN is the same attack on a different rail, and it
                # is common in India — so this family is not card-only.
                "channel": _channel_for(m, rng, force_cnp=True, upi_bias=0.35),
                "geo_city": h.home_city, "country": h.home_country,
                "distance_from_home_km": round(float(abs(rng.normal(0, 12))), 1),
                "device_id": device, "ip_prefix": ip,
                "is_3ds": True, "otp_verified": True,  # relayed factor => looks authenticated
                "account_age_days": _account_age(h, t_day),
            })
            rows.append(r)
    return rows


def geo_anomaly(holders, merchants, rng, n, spec=None):
    """Card-present spend far from home, sometimes in a high-risk jurisdiction."""
    spec = as_spec(spec, "geo_anomaly")
    rows = []
    cp = [m for m in merchants if m.cnp_prob < 0.3] or merchants
    remaining = n
    while remaining > 0:
        h = holders[int(rng.integers(len(holders)))]
        hits = min(remaining, _n_hits(spec, rng))
        remaining -= hits
        day = _pick_day(rng)
        city, country, dist = _geo_for(spec, h.home_city, rng)
        terminal = f"POS{int(rng.integers(0, 1_000_000)):06d}"
        for _ in range(hits):
            m = _weighted_pick(cp, rng)
            r = _row("geo_anomaly")
            r.update({
                "timestamp": _ts(rng, spec, day),
                "cardholder_id": h.cardholder_id, "card_id": h.card_id,
                "merchant_id": m.merchant_id, "mcc": m.mcc, "mcc_risk": m.risk,
                "amount": _amount(m.mcc, rng, scale=spec.amount_scale * h.spend_factor),
                "channel": "card_cp", "geo_city": city, "country": country,
                "distance_from_home_km": dist,
                "device_id": terminal,
                "ip_prefix": f"{int(rng.integers(2,223))}.{int(rng.integers(0,255))}",
                "account_age_days": _account_age(h, day),
            })
            rows.append(r)
    return rows


def scam_transfer(holders, merchants, rng, n, spec=None):
    """An authorized push payment the genuine customer was manipulated into making.

    Every authentication signal is clean: the customer's own device, the
    customer's own city, a passed step-up. The only authorization-time signal is
    that the payee is new and the value is out of character — which is why the
    correct control is friction, not a hard decline.
    """
    spec = as_spec(spec, "scam_transfer")
    rows = []
    xfer = [m for m in merchants if m.mcc == "4829_money_transfer"] or merchants
    remaining = n
    while remaining > 0:
        h = holders[int(rng.integers(len(holders)))]
        hits = min(remaining, _n_hits(spec, rng))
        remaining -= hits
        day = _pick_day(rng)
        payee = _weighted_pick(xfer, rng)
        for j in range(hits):
            # Scams escalate: the second and third payments are larger.
            escalate = 1.0 + 0.6 * j
            r = _row("scam_transfer")
            # In India these scams overwhelmingly settle on UPI: the victim pushes
            # the money themselves, authenticated with their own UPI PIN, and
            # settlement is instant and irrevocable.
            channel = _channel_for(payee, rng, force_cnp=True, upi_bias=0.65)
            r.update({
                "timestamp": _ts(rng, spec, min(P.SIM_DAYS - 1, day + j)),
                "cardholder_id": h.cardholder_id, "card_id": h.card_id,
                "merchant_id": payee.merchant_id, "mcc": "4829_money_transfer",
                "mcc_risk": payee.risk,
                "amount": _amount("4829_money_transfer", rng,
                                  scale=spec.amount_scale * escalate * h.spend_factor),
                "channel": channel, "geo_city": h.home_city, "country": h.home_country,
                "distance_from_home_km": round(float(abs(rng.normal(0, 10))), 1),
                "device_id": h.usual_device_id,   # victim's own device (authorized push)
                "ip_prefix": h.usual_ip_prefix,
                "is_3ds": True, "otp_verified": True,
                "account_age_days": _account_age(h, min(P.SIM_DAYS - 1, day + j)),
            })
            rows.append(r)
    return rows


def wallet_provisioning(holders, merchants, rng, n, spec=None):
    """Digital-wallet / token provisioning abuse.

    A compromised card is provisioned onto a wallet on an attacker-controlled
    device. The provisioning check itself looks like a trivial verification
    authorization; afterwards the spend arrives as a fully authenticated
    wallet token, inheriting trust the underlying card never granted.

    This is modelled purely as its *behavioural consequence* in the
    authorization stream. Nothing here describes how a provisioning control
    would be defeated.
    """
    spec = as_spec(spec, "wallet_provisioning")
    by_id = _by_id(merchants)
    rows = []
    remaining = n
    while remaining > 0:
        h = holders[int(rng.integers(len(holders)))]
        wallet_device = f"WALLET{int(rng.integers(0, 1_000_000)):06d}"
        wallet_ip = f"{int(rng.integers(2,223))}.{int(rng.integers(0,255))}"
        day = _pick_day(rng)
        # 1) the provisioning verification: a tiny authenticated auth on a new device
        verify_m = _merchant_for(spec, h, merchants, by_id, rng)
        rows.append(_row("wallet_provisioning") | {
            "timestamp": _ts(rng, spec, day),
            "cardholder_id": h.cardholder_id, "card_id": h.card_id,
            "merchant_id": verify_m.merchant_id, "mcc": verify_m.mcc,
            "mcc_risk": verify_m.risk,
            "amount": float(np.round(rng.uniform(1, 60), 2)),
            "channel": "card_cnp", "geo_city": h.home_city, "country": h.home_country,
            "distance_from_home_km": round(float(abs(rng.normal(0, 15))), 1),
            "device_id": wallet_device, "ip_prefix": wallet_ip,
            "is_3ds": True, "otp_verified": True,
            "account_age_days": _account_age(h, day),
        })
        remaining -= 1
        # 2) subsequent spend on the provisioned token, inheriting its trust
        hits = min(max(0, remaining), max(1, _n_hits(spec, rng)))
        remaining -= hits
        for j in range(hits):
            m = _merchant_for(spec, h, merchants, by_id, rng)
            t_day = min(P.SIM_DAYS - 1, day + int(rng.integers(0, 3)))
            city, country, dist = _geo_for(spec, h.home_city, rng)
            r = _row("wallet_provisioning")
            r.update({
                "timestamp": _ts(rng, spec, t_day),
                "cardholder_id": h.cardholder_id, "card_id": h.card_id,
                "merchant_id": m.merchant_id, "mcc": m.mcc, "mcc_risk": m.risk,
                "amount": _amount(m.mcc, rng, scale=spec.amount_scale * h.spend_factor),
                "channel": "card_cnp", "geo_city": city, "country": country,
                "distance_from_home_km": dist,
                "device_id": wallet_device, "ip_prefix": wallet_ip,
                "is_3ds": True, "otp_verified": True,   # token-inherited authentication
                "account_age_days": _account_age(h, t_day),
            })
            rows.append(r)
    return rows


INJECTORS = {
    "card_testing": card_testing,
    "bust_out": bust_out,
    "account_takeover": account_takeover,
    "adversarial_mimicry": adversarial_mimicry,
    "velocity_smurfing": velocity_smurfing,
    "merchant_laundering": merchant_laundering,
    "friendly_fraud": friendly_fraud,
    "otp_relay": otp_relay,
    "geo_anomaly": geo_anomaly,
    "scam_transfer": scam_transfer,
    "wallet_provisioning": wallet_provisioning,
}

# Default relative mix of fraud across injectors (sums to ~1).
DEFAULT_MIX = {
    "card_testing": 0.18, "account_takeover": 0.13, "adversarial_mimicry": 0.10,
    "otp_relay": 0.11, "scam_transfer": 0.10, "velocity_smurfing": 0.08,
    "merchant_laundering": 0.08, "bust_out": 0.06, "friendly_fraud": 0.06,
    "geo_anomaly": 0.04, "wallet_provisioning": 0.06,
}
