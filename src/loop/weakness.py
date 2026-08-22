"""Blue-team weakness analysis: what is the defense actually relying on, and
what does the traffic that escapes it look like?

This is the feedback edge of the closed loop. Without it, "attack evolution"
would just be a scripted walk down a stealth dial. With it, each new generation
is aimed at the signal the *current* model demonstrably depends on, measured on
that model's own decisions:

    escaped transactions (false negatives)
              +
    signal attribution on this family
              |
       WeaknessReport
              |
    mutation proposal  ->  GenAI red-team agent  ->  AttackSpec  ->  simulator

Everything here operates on the local simulated environment and a locally trained
model. Nothing probes an external service.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
from src.defend.features import FEATURE_COLUMNS
from src.generate.attack_spec import AttackSpec, spec_diff

# Which specification dial neutralizes reliance on which detector signal. This is
# the red team's playbook: it is domain knowledge about payment behaviour, not a
# fitted mapping, and it is applied only to the signal the model was measured to
# depend on.
SIGNAL_TO_DIAL: dict[str, tuple[str, str, str]] = {
    # signal            -> (spec field,        new value,           human explanation)
    "device_changed": ("device_behavior", "trusted_device",
                       "run from a device the card already trusts, so the device-change flag never fires"),
    "device_card_count_prior": ("device_behavior", "new_device",
                                "stop sharing one device across the ring so the device-to-card counter stays low"),
    "card_device_count_prior": ("device_behavior", "trusted_device",
                                "stay on the card's existing device instead of hopping"),
    "log_amount": ("amount_profile", "moderate",
                   "bring the ticket down into the cardholder's ordinary range"),
    "amount_zscore": ("amount_profile", "moderate",
                      "keep the amount close to the card's own historical average"),
    "near_threshold": ("amount_profile", "low",
                       "move off the structuring threshold the rules watch"),
    "velocity_1h": ("velocity_profile", "low_and_slow",
                    "spread the attempts out so no short window looks busy"),
    "velocity_24h": ("velocity_profile", "low_and_slow",
                     "spread the attempts across days rather than one session"),
    "log_distance": ("geo_behavior", "plausible",
                     "transact from a location consistent with the cardholder's geography"),
    "is_foreign": ("geo_behavior", "plausible",
                   "stay domestic so the cross-border flag never fires"),
    "is_high_risk_country": ("geo_behavior", "plausible",
                             "avoid jurisdictions that are risk-listed outright"),
    "geo_changed": ("geo_behavior", "home",
                    "transact from the cardholder's home city"),
    "mcc_risk": ("merchant_behavior", "new_low_risk_merchant",
                 "move the spend into a low-risk merchant category"),
    "mcc_novel": ("merchant_behavior", "known_merchant",
                  "buy in a category this card already uses"),
    "is_new_payee": ("merchant_behavior", "known_merchant",
                     "route the value through a merchant the card has paid before"),
    "merchant_new_card_ratio_prior": ("merchant_behavior", "new_low_risk_merchant",
                                      "use established merchants instead of a front that only sees new cards"),
    "merchant_card_fanin_prior": ("merchant_behavior", "new_low_risk_merchant",
                                  "spread across merchants so no single payee accumulates fan-in"),
    "is_night": ("timing_profile", "customer_normal",
                 "transact during the cardholder's ordinary waking hours"),
    "hour": ("timing_profile", "customer_normal",
             "match the cardholder's normal time-of-day pattern"),
    "is_cnp": ("merchant_behavior", "known_merchant",
               "present the transaction through a channel the card ordinarily uses"),
}

# One step quieter, used when the targeted dial is already at its quietest value.
_AMOUNT_DOWN = {"extreme": "high", "high": "moderate", "moderate": "low",
                "low": "micro", "micro": "micro"}
_VELOCITY_DOWN = {"burst": "moderate", "moderate": "low_and_slow",
                  "low_and_slow": "single", "single": "single"}

# How far each dial value sits from ordinary customer behaviour. A mutation is
# only worth making if it moves the attack CLOSER to that behaviour: proposing
# "stop using the victim's own device" because the model watches device sharing
# would hand the defense an easier signal, not a harder one. Amount is scored by
# distance from a normal ticket in either direction, because both an unusually
# large purchase and a micro-probe are deviations.
_STEALTH_RANK = {
    "device_behavior": {"trusted_device": 0, "secondary_device": 1, "shared_device": 2,
                        "new_device": 3},
    "geo_behavior": {"home": 0, "plausible": 1, "domestic_far": 2, "foreign": 3,
                     "high_risk": 4},
    "velocity_profile": {"single": 0, "low_and_slow": 1, "moderate": 2, "burst": 3},
    "merchant_behavior": {"known_merchant": 0, "new_low_risk_merchant": 1,
                          "new_high_risk_merchant": 2, "cash_like": 3, "front_merchant": 3},
    "timing_profile": {"customer_normal": 0, "business_hours": 1, "any": 2, "night": 3},
    "amount_profile": {"moderate": 0, "low": 1, "high": 1, "micro": 2, "extreme": 2},
}


def _is_stealthier(field_name: str, current: str, proposed: str) -> bool:
    ranks = _STEALTH_RANK.get(field_name)
    if ranks is None:
        return current != proposed
    return ranks.get(proposed, 99) < ranks.get(current, 99)


@dataclass
class WeaknessReport:
    """What the defense leans on for one family, and what got past it."""

    family: str
    recall: float | None
    n_fraud: int
    n_escaped: int
    relied_signals: list[dict] = field(default_factory=list)   # signal attribution
    escape_profile: list[dict] = field(default_factory=list)   # escaped vs caught
    note: str = ""

    @property
    def top_signal(self) -> str | None:
        return self.relied_signals[0]["feature"] if self.relied_signals else None

    def to_dict(self) -> dict:
        return {
            "family": self.family, "recall": self.recall, "n_fraud": self.n_fraud,
            "n_escaped": self.n_escaped, "relied_signals": self.relied_signals,
            "escape_profile": self.escape_profile, "top_signal": self.top_signal,
            "note": self.note,
        }


def analyze_family(model, X: pd.DataFrame, y: np.ndarray, attack: np.ndarray,
                   family: str, n_repeats: int = 3, top_k: int = 6,
                   legit_sample: int = 4000, seed: int | None = None) -> WeaknessReport:
    """Measure which signals this model uses to catch `family`, and profile the
    transactions of that family which still got through.

    Signal attribution is a permutation test scoped to the family: shuffle one
    feature across this family's fraud rows plus a legitimate sample, and see how
    much the model's ability to rank that family above legitimate traffic drops.
    A large drop means the model is leaning on that feature *for this family* —
    which makes it the obvious thing for the next attack generation to remove.
    """
    rng = np.random.default_rng(config.GLOBAL_SEED if seed is None else seed)
    fam_mask = (attack == family) & (y == 1)
    n_fraud = int(fam_mask.sum())
    if n_fraud == 0:
        return WeaknessReport(family, None, 0, 0, note="family absent from this evaluation frame")

    scores = model.fused_scores(X)
    flagged = scores >= model.threshold
    recall = float(flagged[fam_mask].mean())
    escaped = fam_mask & ~flagged
    caught = fam_mask & flagged
    n_escaped = int(escaped.sum())

    legit_idx = np.where(y == 0)[0]
    if len(legit_idx) > legit_sample:
        legit_idx = rng.choice(legit_idx, legit_sample, replace=False)
    sub_idx = np.concatenate([np.where(fam_mask)[0], legit_idx])
    X_sub = X.iloc[sub_idx]
    y_sub = np.concatenate([np.ones(n_fraud, dtype=int), np.zeros(len(legit_idx), dtype=int)])

    relied: list[dict] = []
    if len(np.unique(y_sub)) > 1:
        base = roc_auc_score(y_sub, model.fused_scores(X_sub))
        for col in FEATURE_COLUMNS:
            drops = []
            for _ in range(n_repeats):
                Xp = X_sub.copy()
                Xp[col] = rng.permutation(Xp[col].to_numpy())
                drops.append(base - roc_auc_score(y_sub, model.fused_scores(Xp)))
            relied.append({"feature": col, "auc_drop": round(float(np.mean(drops)), 5)})
        relied = sorted(relied, key=lambda d: d["auc_drop"], reverse=True)[:top_k]

    profile: list[dict] = []
    if n_escaped >= 3 and int(caught.sum()) >= 3:
        for col in FEATURE_COLUMNS:
            v = X[col].to_numpy()
            esc_mean = float(v[escaped].mean())
            caught_mean = float(v[caught].mean())
            legit_mean = float(v[y == 0].mean())
            spread = abs(esc_mean - caught_mean) / (abs(caught_mean) + abs(esc_mean) + 1e-9)
            profile.append({"feature": col, "escaped_mean": round(esc_mean, 4),
                            "caught_mean": round(caught_mean, 4),
                            "legit_mean": round(legit_mean, 4),
                            "relative_gap": round(spread, 4)})
        profile = sorted(profile, key=lambda d: d["relative_gap"], reverse=True)[:top_k]

    note = ""
    if n_escaped == 0:
        note = "nothing escaped in this frame; the next generation targets the strongest signal instead"
    elif recall is not None and recall < 0.25:
        note = "most of this family already evades the current defense"
    return WeaknessReport(family, round(recall, 4), n_fraud, n_escaped, relied, profile, note)


def weakest_families(model, X: pd.DataFrame, y: np.ndarray, attack: np.ndarray,
                     min_n: int = 10) -> list[dict]:
    """Rank families by how badly the current defense handles them.

    This is what picks the next red-team target: the loop attacks where the
    defense is measurably weakest, not a family chosen in advance.
    """
    scores = model.fused_scores(X)
    flagged = scores >= model.threshold
    out = []
    for fam in sorted({a for a in attack if a != "legit"}):
        m = (attack == fam) & (y == 1)
        n = int(m.sum())
        if n < min_n:
            continue
        out.append({"family": fam, "recall": round(float(flagged[m].mean()), 4), "n": n})
    return sorted(out, key=lambda d: d["recall"])


def heuristic_mutation(report: WeaknessReport, previous: AttackSpec) -> dict:
    """Deterministic, offline mutation proposal derived from the weakness report.

    Used as the fallback whenever the language model is unavailable, and as the
    reference the model's proposal is validated against. The rule is simple and
    defensible: find the signal the defense leans on hardest for this family, and
    move the one specification dial that removes it.
    """
    proposed = {"source": "heuristic", "generation": previous.generation + 1}
    target = None
    for s in report.relied_signals:
        cand = SIGNAL_TO_DIAL.get(s["feature"])
        if not cand:
            continue
        field_name, new_value, why = cand
        # Only accept a move that takes the attack CLOSER to ordinary behaviour.
        # Otherwise the "evasion" would hand the defense a stronger signal.
        if _is_stealthier(field_name, getattr(previous, field_name), new_value):
            target = (s["feature"], field_name, new_value, why)
            break

    if target is None:
        # Every dial the top signals point at is already as quiet as it gets.
        # Fall back to shrinking the residual deviation rather than inventing a
        # new tactic the family would not plausibly adopt.
        if previous.amount_profile not in ("moderate", "micro"):
            field_name, new_value = "amount_profile", _AMOUNT_DOWN[previous.amount_profile]
            why = "shrink the remaining amount deviation further toward the cardholder's norm"
            sig = "residual amount signal"
        elif previous.velocity_profile != "single":
            field_name, new_value = "velocity_profile", _VELOCITY_DOWN[previous.velocity_profile]
            why = "slow the campaign down further"
            sig = "residual velocity signal"
        else:
            # Nothing structural is left to change: the attack is already sitting
            # on the legitimate centroid. Only the magnitude dial remains.
            field_name, new_value = "timing_profile", "customer_normal"
            why = "match the cardholder's ordinary rhythm exactly"
            sig = "residual timing signal"
        target = (sig, field_name, new_value, why)

    sig, field_name, new_value, why = target
    proposed[field_name] = new_value
    # Stealth also creeps up each generation; the size of the step is larger when
    # the defense is still catching most of the family.
    r = report.recall if report.recall is not None else 1.0
    step = 0.65 if r >= 0.85 else (0.8 if r >= 0.5 else 0.9)
    proposed["intensity"] = round(previous.intensity * step, 3)
    proposed["targets_signal"] = str(sig)
    proposed["strategy"] = f"{why} (defeats: {sig})"
    proposed["rationale"] = (
        f"The defense catches {r:.0%} of {report.family} and its ranking of this family depends "
        f"most on {sig}. Moving {field_name} to {new_value} removes that dependency; whatever "
        "the defense still detects afterwards is a genuinely different signal.")
    proposed["confidence"] = 0.6
    return proposed


def lineage_entry(family: str, previous: AttackSpec, current: AttackSpec,
                  report: WeaknessReport, validation: dict,
                  stale_recall=None, adapted_recall=None) -> dict:
    """One node of the attack lineage: what changed, why, and what it cost the
    defense. This is what makes the evolution story inspectable rather than
    asserted."""
    d = spec_diff(previous, current)
    d.update({
        "family": family,
        "generation": current.generation,
        "spec": current.to_dict(),
        "driven_by": {
            "measured_recall_before": report.recall,
            "n_escaped_before": report.n_escaped,
            "top_relied_signal": report.top_signal,
            "relied_signals": report.relied_signals,
        },
        "constraint_layer": validation,
        "stale_recall": stale_recall,
        "adapted_recall": adapted_recall,
        "spec_source": current.source,
    })
    return d
