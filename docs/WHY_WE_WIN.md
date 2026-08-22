# Positioning against the judging criteria

Internal. Maps what exists in the repository to each official criterion, with literal
counts. Regenerated from artifacts.

## 1. Diversity of attacks identified

- **45 distinct attacks** across **6 fraud
  surfaces** and 6 payment rails, each with attacker objective, GenAI
  role, kill chain, transaction signature, behavioural signature, authorization-time
  observability, and post-settlement signals.
- Coverage is stated, not implied: 12 IMPLEMENTED,
  19 PARAMETERIZED, 8 RESEARCH_ONLY,
  6 FUTURE.
- 23 entries are flagged as low or no visibility at
  authorization time — volunteering the limit of our own control surface.

## 2. Fidelity of simulated attacks

- Customer archetypes, merchant profiles, weekday/weekend and payday rhythms,
  shopping sessions, subscriptions, card issuance **and attrition** inside the window.
- Fraud actors carry cover traffic; attacks reuse cards, devices and merchants.
- 11 automated fidelity checks; strongest single feature at
  AUC 0.722, 0 failing.
- Temporal causality enforced and unit-tested; post-outcome fields hard-blocked.

## 3. Detection efficacy

- Recall 67.6%, precision 0.242,
  PR-AUC 0.500, false-positive rate
  1.97% against a
  2% budget.
- Rules / static / adaptive compared **at a matched false-positive budget**, with a
  competent rule set including network-level rules.
- Family recall on an unseen fraud-enriched frame, with sample sizes and 95% Wilson
  intervals.
- Leave-one-attack-family-out across 10 families.
- Head-to-head on the evolved attacks: 49% for the static defense versus 48% for the adaptive one, on 2703 fraudulent transactions of the final evolved generation from a seed neither model has seen.

## 4. Novelty

- Weakness-driven attack evolution: the family attacked and the dial moved are both
  derived from measured feature attribution on the current model.
- Structured attack specifications with a payment-domain constraint layer.
- Inspectable attack lineage with per-generation dial deltas and recall.
- Bounded, stratified cumulative replay across every generation, with a rehearsal
  sample of every family NOT under attack — the fix for a catastrophic-forgetting
  failure this loop actually produced and its own gate caught.
- Effort reallocation: frontiers that do not move across two rounds are retired and
  the red team retargets.

## 5. Real-world feasibility

- Authorization-time features only; network counters are constant-time lookups.
- Online inference and offline adaptation separated explicitly.
- Calibrated probabilities driving APPROVE / STEP-UP / DECLINE with reason codes.
- Champion/challenger gates that can and do refuse a candidate.
- Operational volumes translated into review hours and customer friction.

## The thing to say out loud

Every number is synthetic, none is validated on real payment data, and the residual
frontiers are published rather than tuned away. That posture is not a weakness in the
submission — under technical questioning it is the strongest thing in it.
