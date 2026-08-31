"""Regenerate the 2-Minute Judge Demo hero transaction.

The Judge Demo leads with one CLOSED-LOOP family — the attack the red team
actually evolved, the defense actually re-learned, and governance actually voted
on. That is a different experiment from the leave-one-attack-family-out hero on
the landing page (``models/hero_example.json``), so it gets its own artifact and
its own regeneration command; neither disturbs the other.

Nothing is retrained here. This loads the two models the closed loop already
produced and committed — the stale base defense (round 0) and the final adapted
candidate — plus the committed final attack specifications, and persists ONE
concrete evolved transaction the stale model waved through but the adapted model
routes to friction. Fully deterministic from ``config.GLOBAL_SEED``.

    python -m src.experiments.judge_hero                 # otp_relay (default)
    python -m src.experiments.judge_hero --family scam_transfer

Output: models/judge_hero.json
"""
from __future__ import annotations

import argparse
import json

import config
from src.defend.model import DefenseModel
from src.loop.redteam_loop import _save_hero_example, _spec_from_dict

# The closed-loop hero. otp_relay is the focus family with the largest, best-
# powered adaptive recovery in the committed run (round 1: 16% -> 51%, promoted;
# head-to-head on the final generation: static 33% -> adaptive 68%, n=531), so it
# carries the "detector learns a new scam" story most honestly.
DEFAULT_FAMILY = "otp_relay"
JUDGE_HERO_PATH = config.MODELS_DIR / "judge_hero.json"


def regenerate(family: str = DEFAULT_FAMILY, out_path=None):
    """Rebuild the Judge Demo hero transaction for ``family`` and return its path."""
    if not config.LOOP_HISTORY_PATH.exists():
        raise FileNotFoundError(
            "models/loop_history.json is missing — run `python -m src.pipeline` first.")
    for p in (config.LOOP_BASE_MODEL_PATH, config.LOOP_ADAPTED_MODEL_PATH):
        if not p.exists():
            raise FileNotFoundError(
                f"{p.name} is missing — run `python -m src.pipeline` first.")

    hist = json.loads(config.LOOP_HISTORY_PATH.read_text(encoding="utf-8"))
    final_specs = hist["history"][-1]["specs"]
    if family not in final_specs:
        raise ValueError(
            f"{family!r} was not a closed-loop focus family; choose one of "
            f"{sorted(final_specs)}.")
    specs = {fam: _spec_from_dict(d, fam) for fam, d in final_specs.items()}

    stale = DefenseModel.load(config.LOOP_BASE_MODEL_PATH)      # defense in force, round 0
    adapted = DefenseModel.load(config.LOOP_ADAPTED_MODEL_PATH)  # final adapted candidate

    out = out_path or JUDGE_HERO_PATH
    _save_hero_example(stale, adapted, specs, config.GLOBAL_SEED,
                       hero_fam=family, out_path=out)
    return out


def main():
    ap = argparse.ArgumentParser(description="Regenerate the Judge Demo hero transaction.")
    ap.add_argument("--family", default=DEFAULT_FAMILY,
                    help="closed-loop focus family to feature (default: otp_relay)")
    args = ap.parse_args()
    out = regenerate(args.family)
    data = json.loads(out.read_text(encoding="utf-8"))
    s, a = data["stale"], data["adapted"]
    print(f"Wrote {out}")
    print(f"  family        : {data['family']}")
    print(f"  txn           : {data['transaction'].get('txn_id')} "
          f"amount={data['transaction'].get('amount')}")
    print(f"  stale defense : {s['action']:8s} (fraud prob {s['probability']:.3f})")
    print(f"  adapted       : {a['action']:8s} (fraud prob {a['probability']:.3f})")
    hr = data.get("hero_family_recall", {})
    print(f"  family recall : stale {hr.get('stale')} -> adapted {hr.get('adapted')} "
          f"(n={hr.get('n')})")


if __name__ == "__main__":
    main()
