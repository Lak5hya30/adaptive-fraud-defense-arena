"""Internal synthetic fidelity diagnostics.

A simulator is only useful as a training ground if fraudulent and legitimate
behaviour genuinely overlap. If any single field cleanly separates the classes,
the detector learns that artifact and every downstream number becomes meaningless
— it would look excellent and generalize to nothing.

This module measures the overlap and emits explicit PASS / WARN / FAIL checks, so
the claim "our synthetic data is not trivially separable" is a measurement in a
committed artifact rather than an assertion in a slide.

These are diagnostics of the SIMULATOR against itself. They say nothing about
similarity to any real payment portfolio, and no comparison to production data of
any kind has been performed.

CLI:  python -m src.generate.fidelity
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
from src.defend.features import FEATURE_COLUMNS, build_xy

# A single feature that ranks fraud above legit this well is a shortcut, not a
# signal: at that point the model is reading the generator, not the fraud.
SEPARABILITY_FAIL = 0.90
SEPARABILITY_WARN = 0.82

# Minimum family sample size before a per-family recall number is worth quoting.
MIN_FAMILY_N = 15


def _hist(series: pd.Series, bins) -> dict:
    counts, edges = np.histogram(series.dropna(), bins=bins)
    total = max(1, counts.sum())
    return {"edges": [round(float(e), 3) for e in edges],
            "share": [round(float(c) / total, 5) for c in counts]}


def _overlap_coefficient(a: pd.Series, b: pd.Series, bins: int = 40) -> float:
    """Histogram intersection of two distributions on a shared grid: 1.0 means
    indistinguishable marginals, 0.0 means completely disjoint."""
    a = a.dropna().to_numpy()
    b = b.dropna().to_numpy()
    if len(a) == 0 or len(b) == 0:
        return 0.0
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if hi <= lo:
        return 1.0
    grid = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a, bins=grid)
    hb, _ = np.histogram(b, bins=grid)
    ha = ha / max(1, ha.sum())
    hb = hb / max(1, hb.sum())
    return round(float(np.minimum(ha, hb).sum()), 4)


def separability(X: pd.DataFrame, y: np.ndarray) -> list[dict]:
    """Per-feature univariate ranking power and marginal overlap."""
    out = []
    for c in FEATURE_COLUMNS:
        col = X[c].fillna(0.0)
        auc = float(roc_auc_score(y, col))
        out.append({
            "feature": c,
            "auc": round(max(auc, 1 - auc), 4),
            "legit_mean": round(float(col[y == 0].mean()), 4),
            "fraud_mean": round(float(col[y == 1].mean()), 4),
            "overlap": _overlap_coefficient(col[y == 0], col[y == 1]),
        })
    return sorted(out, key=lambda d: d["auc"], reverse=True)


def distributions(df: pd.DataFrame) -> dict:
    """Side-by-side marginals a reviewer can eyeball."""
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    legit, fraud = d[d.is_fraud == 0], d[d.is_fraud == 1]
    amt_bins = np.linspace(0, np.log1p(d.amount.quantile(0.999)), 41)
    return {
        "log_amount": {
            "legit": _hist(np.log1p(legit.amount), amt_bins),
            "fraud": _hist(np.log1p(fraud.amount), amt_bins),
        },
        "hour_of_day": {
            "legit": _hist(legit.timestamp.dt.hour, np.arange(0, 25)),
            "fraud": _hist(fraud.timestamp.dt.hour, np.arange(0, 25)),
        },
        "day_of_week": {
            "legit": _hist(legit.timestamp.dt.dayofweek, np.arange(0, 8)),
            "fraud": _hist(fraud.timestamp.dt.dayofweek, np.arange(0, 8)),
        },
        "distance_km": {
            "legit": _hist(np.log1p(legit.distance_from_home_km), np.linspace(0, 10, 41)),
            "fraud": _hist(np.log1p(fraud.distance_from_home_km), np.linspace(0, 10, 41)),
        },
    }


def portfolio(df: pd.DataFrame) -> dict:
    """Structural realism of the synthetic portfolio itself."""
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    legit = d[d.is_fraud == 0]
    per_card = legit.groupby("card_id").size()
    per_merchant = d.groupby("merchant_id").size()
    repeat = legit.duplicated(subset=["card_id", "merchant_id"]).mean()
    return {
        "transactions_per_card": {
            "median": float(per_card.median()), "p90": float(per_card.quantile(0.9)),
            "max": int(per_card.max()), "n_cards": int(per_card.size),
        },
        "transactions_per_merchant": {
            "median": float(per_merchant.median()), "p90": float(per_merchant.quantile(0.9)),
            "max": int(per_merchant.max()), "n_merchants": int(per_merchant.size),
        },
        "legit_repeat_merchant_share": round(float(repeat), 4),
        "legit_distinct_merchants_per_card_median":
            float(legit.groupby("card_id").merchant_id.nunique().median()),
        "legit_distinct_devices_per_card_median":
            float(legit.groupby("card_id").device_id.nunique().median()),
        "weekend_share": round(float((d.timestamp.dt.dayofweek >= 5).mean()), 4),
        "cnp_share": round(float((d.channel == "card_cnp").mean()), 4),
        "foreign_share": round(float((d.country != "IN").mean()), 4),
        "actor_role_counts": {k: int(v) for k, v in d.actor_role.value_counts().items()}
            if "actor_role" in d.columns else {},
    }


def _check(name: str, ok: bool, detail: str, level: str = "FAIL") -> dict:
    return {"check": name, "status": "PASS" if ok else level, "detail": detail}


def quality_checks(df: pd.DataFrame, X: pd.DataFrame, y: np.ndarray,
                   sep: list[dict]) -> list[dict]:
    """Automated guards against a degenerate or shortcut-ridden simulator."""
    legit, fraud = df[df.is_fraud == 0], df[df.is_fraud == 1]
    checks: list[dict] = []

    worst = sep[0]
    checks.append(_check(
        "no_single_feature_separates_classes",
        worst["auc"] < SEPARABILITY_FAIL,
        f"strongest single feature is {worst['feature']} at AUC {worst['auc']:.3f} "
        f"(fail at {SEPARABILITY_FAIL})"))
    near = [s for s in sep if s["auc"] >= SEPARABILITY_WARN]
    checks.append(_check(
        "no_feature_near_perfect_separation", not near,
        "none above the warning line" if not near
        else "above warning line: " + ", ".join(f"{s['feature']}={s['auc']:.3f}" for s in near),
        level="WARN"))

    dev = float(X.loc[y == 0, "device_changed"].mean())
    checks.append(_check("legit_device_changes_exist", dev > 0.02,
                         f"{dev:.3%} of genuine transactions change device"))
    npay = float(X.loc[y == 0, "is_new_payee"].mean())
    checks.append(_check("legit_new_payees_exist", 0.05 < npay < 0.95,
                         f"{npay:.1%} of genuine transactions are a first visit to that merchant"))

    known_dev = float((X.loc[y == 1, "device_changed"] == 0).mean())
    checks.append(_check("fraud_sometimes_uses_a_known_device", known_dev > 0.10,
                         f"{known_dev:.1%} of fraud runs from a device the card had used before"))
    repeat_payee = float((X.loc[y == 1, "is_new_payee"] == 0).mean())
    checks.append(_check("fraud_sometimes_reuses_a_merchant", repeat_payee > 0.05,
                         f"{repeat_payee:.1%} of fraud is a repeat visit to the same merchant"))

    lo, hi = np.quantile(legit.amount, [0.25, 0.75])
    in_iqr = float(((fraud.amount >= lo) & (fraud.amount <= hi)).mean())
    checks.append(_check("fraud_amounts_overlap_legit_range", in_iqr > 0.10,
                         f"{in_iqr:.1%} of fraud sits inside the genuine inter-quartile amount range "
                         f"(INR {lo:,.0f}-{hi:,.0f})"))

    d0_l = float((X.loc[y == 0, "card_history_depth"] == 0).mean())
    d0_f = float((X.loc[y == 1, "card_history_depth"] == 0).mean())
    ratio = d0_f / max(d0_l, 1e-6)
    checks.append(_check("no_history_is_not_a_fraud_synonym", d0_l > 0.02 and ratio < 6.0,
                         f"first-ever transaction: {d0_l:.1%} of genuine vs {d0_f:.1%} of fraud "
                         f"(ratio {ratio:.1f}x)"))

    night_l = float(X.loc[y == 0, "is_night"].mean())
    night_f = float(X.loc[y == 1, "is_night"].mean())
    checks.append(_check("fraud_timing_is_not_uniformly_nocturnal",
                         night_f < 0.60 and night_l > 0.02,
                         f"night-time share: {night_l:.1%} genuine vs {night_f:.1%} fraud",
                         level="WARN"))

    small = {a: int((df.attack_type == a).sum()) for a in sorted(set(df.attack_type))
             if a != "legit" and int((df.attack_type == a).sum()) < MIN_FAMILY_N}
    checks.append(_check("family_sample_sizes_support_their_claims", not small,
                         "all families at or above the reporting floor" if not small
                         else f"below n={MIN_FAMILY_N}: {small}", level="WARN"))

    cover = int((df.get("actor_role", pd.Series(dtype=str)) == "fraud_actor_cover").sum())
    checks.append(_check("fraud_actors_have_cover_traffic", cover > 0,
                         f"{cover} legitimate-looking transactions were generated by fraud actors "
                         "(mule grooming, bust-out ramp-up, front-merchant sales)"))
    return checks


def run_fidelity(df: pd.DataFrame | None = None, save: bool = True) -> dict:
    if df is None:
        df = pd.read_parquet(config.DATASET_PARQUET)
    X, y, _ = build_xy(df)
    sep = separability(X, y)
    checks = quality_checks(df, X, y, sep)
    report = {
        "schema_version": config.SCHEMA_VERSION,
        "scope": ("Internal synthetic fidelity diagnostics. Measures overlap between simulated "
                  "legitimate and fraudulent behaviour within this simulator. It is NOT a "
                  "comparison against any real payment portfolio; none has been performed."),
        "n_transactions": int(len(df)),
        "n_fraud": int(df.is_fraud.sum()),
        "separability": sep,
        "max_single_feature_auc": sep[0]["auc"],
        "distributions": distributions(df),
        "portfolio": portfolio(df),
        "checks": checks,
        "n_fail": sum(1 for c in checks if c["status"] == "FAIL"),
        "n_warn": sum(1 for c in checks if c["status"] == "WARN"),
    }
    if save:
        config.FIDELITY_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    rep = run_fidelity()
    print(f"Fidelity diagnostics over {rep['n_transactions']:,} transactions "
          f"({rep['n_fraud']} fraudulent)\n")
    print("Strongest single features (a high value here means a shortcut):")
    for s in rep["separability"][:8]:
        print(f"  {s['feature']:32s} AUC={s['auc']:.3f}  overlap={s['overlap']:.3f}")
    print("\nChecks:")
    for c in rep["checks"]:
        print(f"  [{c['status']:4s}] {c['check']:44s} {c['detail']}")
    print(f"\n{rep['n_fail']} failing, {rep['n_warn']} warning. "
          f"Saved {config.FIDELITY_REPORT_PATH}")


if __name__ == "__main__":
    main()
