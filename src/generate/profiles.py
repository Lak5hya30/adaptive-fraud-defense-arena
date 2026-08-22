"""Cardholder and merchant profile generation + geo helpers.

Profiles give the base generator realistic, *stable* per-entity behaviour so that
fraud injectors can create deviations (new device, new geo, off-pattern MCC) that
a defender can actually learn. All randomness is seeded for reproducibility.

Every cardholder is drawn from a behavioural ARCHETYPE (config.CUSTOMER_ARCHETYPES).
A portfolio built from a single behaviour is trivially separable from fraud; a
portfolio built from overlapping archetypes is not. The archetype controls spend
level, frequency, category preference, device count, travel and night-time
propensity, weekday/weekend shape and channel mix.

Merchants carry their own profile (risk tier, domestic/international, recurring,
popularity) so that "new merchant" and "risky merchant" mean something specific.
"""
from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field

import numpy as np

import config

# Fixed simulation window so datasets are deterministic regardless of run date.
SIM_START_ISO = "2026-05-01T00:00:00"
SIM_DAYS = 90


def make_rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(config.GLOBAL_SEED if seed is None else seed)


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    (lat1, lon1), (lat2, lon2) = a, b
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


@dataclass
class Cardholder:
    cardholder_id: str
    card_id: str
    home_city: str
    home_country: str
    spend_factor: float          # multiplicative wealth/spend scaler
    preferred_mccs: list[str]
    usual_device_id: str
    usual_ip_prefix: str
    account_age_days: int        # age at the START of the simulation window
    daily_rate: float            # expected transactions/day
    risk_tier: float             # 0..1 latent riskiness
    device_pool: list[str] = field(default_factory=list)  # owned devices; [0] is primary
    is_thin_file: bool = False   # new/low-history genuine customer
    archetype: str = "frequent_urban"
    travel_prob: float = 0.06        # per-transaction chance of being away from home
    foreign_given_travel: float = 0.30
    odd_hour_prob: float = 0.05
    weekend_mult: float = 1.25
    cnp_shift: float = 0.0           # additive shift on the merchant's CNP probability
    activation_day: int = 0          # first day of the window this card can transact
    lapse_day: int = SIM_DAYS        # day after which this card stops being used
    regular_merchants: list[str] = field(default_factory=list)   # merchant_ids revisited
    recurring_merchants: list[str] = field(default_factory=list)  # subscription merchants
    recurring_amount: dict[str, float] = field(default_factory=dict)


@dataclass
class Merchant:
    merchant_id: str
    mcc: str
    city: str
    country: str
    cnp_prob: float
    ticket_mu: float
    ticket_sigma: float
    risk: float
    is_synthetic: bool = False
    risk_tier: str = "standard"      # low | standard | elevated | high
    is_international: bool = False
    supports_recurring: bool = False
    accepts_upi: bool = False        # UPI acceptance (domestic merchants only)
    popularity: float = 1.0          # relative traffic weight


def _archetype_names_weights() -> tuple[list[str], np.ndarray]:
    names = list(config.CUSTOMER_ARCHETYPES)
    w = np.array([config.CUSTOMER_ARCHETYPES[n]["weight"] for n in names], dtype=float)
    return names, w / w.sum()


def generate_cardholders(n: int, rng: np.random.Generator) -> list[Cardholder]:
    cities = list(config.GEO_CLUSTERS.keys())
    mccs = list(config.MCC_CATALOG.keys())
    R = config.LEGIT_REALISM
    arch_names, arch_probs = _archetype_names_weights()
    holders: list[Cardholder] = []
    for i in range(n):
        arch = str(rng.choice(arch_names, p=arch_probs))
        A = config.CUSTOMER_ARCHETYPES[arch]

        # Home skews to Indian metros (first 7 cities) for the GFF context.
        home = str(rng.choice(cities, p=_home_city_probs(len(cities))))

        # Category preference: mostly the archetype's characteristic categories,
        # plus a couple of idiosyncratic ones so no archetype is a fixed basket.
        n_pref = int(rng.integers(A["n_mcc"][0], A["n_mcc"][1] + 1))
        bias = [m for m in A["mcc_bias"] if m in mccs]
        pref = list(dict.fromkeys(
            list(rng.permutation(bias))[: max(1, n_pref - 2)]
            + list(rng.choice(mccs, size=min(2, len(mccs)), replace=False))))[:n_pref]

        # Device ownership follows the archetype, with a genuine multi-device tail.
        primary = f"DEV{i:06d}"
        pool = [primary]
        n_dev = int(A["devices"])
        if n_dev > 1 and rng.random() < R["multi_device_fraction"] + 0.25:
            for k in range(n_dev - 1):
                pool.append(f"DEV{i:06d}-{k + 2}")

        thin = arch == "thin_file_new" or rng.random() < R["thin_file_fraction"] * 0.4
        age = (int(rng.integers(3, R["thin_file_max_age_days"])) if thin
               else int(rng.integers(90, 3000)))

        lo, hi = A["rate"]
        rate = float(np.clip(rng.uniform(lo, hi) * rng.lognormal(0.0, 0.25), 0.05, 8.0))
        s_lo, s_hi = A["spend"]
        spend = float(np.clip(rng.uniform(s_lo, s_hi) * rng.lognormal(0.0, 0.22), 0.2, 6.0))

        # A minority of cards are ISSUED during the window: they have no history at
        # all on their first authorization, exactly like a fraudulently opened
        # account. This is what stops "no prior history" from meaning "fraud".
        activation = 0
        if arch == "thin_file_new" or rng.random() < R["late_activation_fraction"]:
            activation = int(rng.integers(0, max(1, SIM_DAYS - 5)))
            age = min(age, int(rng.integers(1, 40)))
        # Cards also leave the portfolio: expiry, replacement, closure, churn.
        lapse = SIM_DAYS
        if activation == 0 and rng.random() < R["lapse_fraction"]:
            lapse = int(rng.integers(5, SIM_DAYS))

        holders.append(Cardholder(
            cardholder_id=f"CH{i:06d}",
            card_id=f"CARD{i:06d}",
            home_city=home,
            home_country=_country_for_city(home),
            spend_factor=spend,
            preferred_mccs=pref,
            usual_device_id=primary,
            usual_ip_prefix=f"{int(rng.integers(10, 223))}.{int(rng.integers(0,255))}",
            account_age_days=age,
            daily_rate=rate,
            risk_tier=float(rng.beta(2, 8)),
            device_pool=pool,
            is_thin_file=thin,
            archetype=arch,
            travel_prob=float(np.clip(R["travel_prob"] * A["travel_mult"], 0.0, 0.6)),
            foreign_given_travel=float(np.clip(R["foreign_given_travel"] * A["foreign_mult"], 0.0, 0.9)),
            odd_hour_prob=float(np.clip(R["odd_hour_prob"] * A["night_mult"], 0.0, 0.5)),
            weekend_mult=float(A["weekend_mult"]),
            cnp_shift=float(A["cnp_shift"]),
            activation_day=activation,
            lapse_day=lapse,
        ))
    return holders


_RISK_TIERS = (("low", 0.25), ("standard", 0.5), ("elevated", 0.75), ("high", 1.01))


def _risk_tier(risk: float) -> str:
    for name, ceiling in _RISK_TIERS:
        if risk < ceiling:
            return name
    return "high"


def generate_merchants(n: int, rng: np.random.Generator) -> list[Merchant]:
    cities = list(config.GEO_CLUSTERS.keys())
    mccs = list(config.MCC_CATALOG.keys())
    merchants: list[Merchant] = []
    for i in range(n):
        mcc = str(rng.choice(mccs))
        spec = config.MCC_CATALOG[mcc]
        city = str(rng.choice(cities, p=_home_city_probs(len(cities))))
        country = _country_for_city(city)
        # Merchant-level risk varies around the category baseline: a grocer can
        # still be a bad actor, a gambling site can be well-controlled.
        risk = float(np.clip(spec["risk"] + rng.normal(0, 0.10), 0.02, 0.99))
        merchants.append(Merchant(
            merchant_id=f"M{i:05d}",
            mcc=mcc,
            city=city,
            country=country,
            cnp_prob=float(spec["cnp_prob"]),
            ticket_mu=float(spec["mu"] + rng.normal(0, 0.1)),
            ticket_sigma=float(spec["sigma"]),
            risk=risk,
            risk_tier=_risk_tier(risk),
            is_international=country != "IN",
            supports_recurring=mcc in ("4900_utilities", "5816_digital_goods", "5967_direct_mktg"),
            # UPI is a domestic rail, and ATM withdrawals are not on it.
            accepts_upi=bool(country == "IN" and mcc != "6011_atm"
                             and rng.random() < config.UPI_REALISM["merchant_acceptance"]),
            # Merchant traffic is heavy-tailed: a few merchants take most volume.
            popularity=float(np.clip(rng.lognormal(0.0, 1.0), 0.05, 25.0)),
        ))
    return merchants


def assign_relationships(holders: list[Cardholder], merchants: list[Merchant],
                         rng: np.random.Generator) -> None:
    """Attach each cardholder's REGULAR and RECURRING merchants, in place.

    Regulars are the merchants a customer revisits, which is what makes
    ``is_new_payee`` a genuine minority signal instead of a fraud shortcut. They
    live on the profile (not inside the legit generator) so the attack injectors
    can use them too — mimicry that shops at the victim's own regular merchant is
    the whole point of mimicry.
    """
    R = config.LEGIT_REALISM
    by_mcc: dict[str, list[Merchant]] = {}
    for m in merchants:
        by_mcc.setdefault(m.mcc, []).append(m)

    for h in holders:
        picks: list[str] = []
        for mcc in h.preferred_mccs:
            pool = by_mcc.get(mcc) or merchants
            w = np.array([m.popularity for m in pool], dtype=float)
            idx = int(rng.choice(len(pool), p=w / w.sum()))
            picks.append(pool[idx].merchant_id)
            # Busier customers keep two regulars in their main categories.
            if h.daily_rate > 1.0 and len(pool) > 1 and rng.random() < 0.45:
                idx2 = int(rng.choice(len(pool), p=w / w.sum()))
                picks.append(pool[idx2].merchant_id)
        h.regular_merchants = list(dict.fromkeys(picks)) or [merchants[0].merchant_id]

        h.recurring_merchants = []
        h.recurring_amount = {}
        if rng.random() < R["subscription_fraction"]:
            recur_pool = [m for m in merchants if m.supports_recurring] or merchants
            k = int(rng.integers(1, R["subscription_max"] + 1))
            for _ in range(k):
                m = recur_pool[int(rng.integers(len(recur_pool)))]
                if m.merchant_id in h.recurring_amount:
                    continue
                h.recurring_merchants.append(m.merchant_id)
                h.recurring_amount[m.merchant_id] = float(np.round(
                    np.exp(rng.normal(m.ticket_mu, m.ticket_sigma * 0.4)) * h.spend_factor, 2))


def _home_city_probs(n_cities: int) -> np.ndarray:
    # Weight the 7 Indian metros heavily, international lightly.
    w = np.array([6, 6, 6, 4, 4, 3, 3, 1, 1, 1], dtype=float)[:n_cities]
    if len(w) < n_cities:  # pragma: no cover - defensive
        w = np.pad(w, (0, n_cities - len(w)), constant_values=1.0)
    return w / w.sum()


def city_latlon(city: str) -> tuple[float, float]:
    return config.GEO_CLUSTERS[city]


def _country_for_city(city: str) -> str:
    return {"dubai": "AE", "singapore": "SG", "london": "GB"}.get(city, "IN")


_SIM_START_WEEKDAY = _dt.datetime.fromisoformat(SIM_START_ISO).weekday()  # Mon=0 .. Sun=6


def is_weekend(day: int) -> bool:
    """True when this day offset in the simulation window falls on a weekend."""
    return (_SIM_START_WEEKDAY + day) % 7 >= 5


def day_of_month(day: int) -> int:
    """Calendar day-of-month for this day offset (payday clustering)."""
    return (_dt.datetime.fromisoformat(SIM_START_ISO) + _dt.timedelta(days=day)).day


def day_weight(day: int, weekend_mult: float | None = None) -> float:
    """Relative spend weight for a day offset in the simulation window.

    Combines a weekend lift (archetype-specific: a business card is quieter at
    the weekend, a retail card busier) and a payday-like salary-credit cluster
    at the start and middle of the calendar month.
    """
    R = config.LEGIT_REALISM
    w = 1.0
    if is_weekend(day):
        w *= R["weekend_lift"] if weekend_mult is None else weekend_mult
    if day_of_month(day) in R["payday_days"]:
        w *= R["payday_lift"]
    return float(w)
