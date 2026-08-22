"""Generate the presentation and Q&A guides from the committed artifacts.

The judge Q&A in particular must never contain a stale number: being caught with
a figure that contradicts the app is worse than not having prepared an answer at
all. So the numbers are injected here, at generation time, from the same
artifacts the app and the README read.

Called by ``docs/build_docs.py``.
"""
from __future__ import annotations

from pathlib import Path

import config

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def pct(x, dp=0):
    return "—" if x is None else f"{x*100:.{dp}f}%"


def num(x, dp=3):
    return "—" if x is None else f"{x:.{dp}f}"


def build_guides(A: dict) -> list[Path]:
    from docs.build_docs import hero_family

    tax = A["tax"]
    counts = tax.summary_counts()
    m = A.get("metrics") or {}
    b = A.get("baseline") or {}
    loao = A.get("loao") or {}
    loop = A.get("loop") or {}
    fid = A.get("fidelity") or {}
    fam = A.get("family") or {}
    ops = (A.get("ops") or {}).get("static_ml", {})
    text = A.get("text") or {}
    hf, hero = hero_family(A)

    matched = b.get("_fpr_matched", {}).get("models", {})
    rules = matched.get("rules_baseline", {})
    static = matched.get("static_ml", {})
    n_injectors = len(tax.injectors)
    n_param = counts["parameterized"]
    n_impl = counts["implemented"]
    n_total = counts["total_attacks"]
    n_research = counts["research_only"] + counts["future"]
    budget_pct = f"{b.get('_fpr_matched', {}).get('target_false_positive_rate', 0)*100:.0f}%"
    strongest = (fid.get("separability") or [{}])[0].get("feature", "—")
    strongest_auc = f"{fid.get('max_single_feature_auc', 0):.3f}"
    n_checks = len(fid.get("checks", []))
    text_auc = num(text.get("roc_auc_cv5") or text.get("roc_auc"))
    promo = loop.get("promotion", {})
    promoted = promo.get("promoted_rounds", [])

    # One sample-size-and-interval string, used everywhere the hero pair is quoted,
    # so no surface can quote the headline without its denominator.
    hero_ci = ""
    if hero:
        ci = hero.get("recall_after_learning_ci95")
        hero_ci = (f" (n={hero['n_test']} held-out"
                   + (f", 95% interval {ci[0]*100:.0f}–{ci[1]*100:.0f}%" if ci else "")
                   + ")")

    h2h = A.get("h2h") or {}
    h2h_static = h2h.get("models", {}).get("static_defense", {})
    h2h_adaptive = h2h.get("models", {}).get("adaptive_defense", {})
    h2h_line = ""
    if h2h_static and h2h_adaptive:
        h2h_line = (
            f"{pct(h2h_static.get('mean_evolved_recall'))} for the static defense versus "
            f"{pct(h2h_adaptive.get('mean_evolved_recall'))} for the adaptive one, on "
            f"{h2h.get('frame', {}).get('n_fraud', '—')} fraudulent transactions of the final "
            "evolved generation from a seed neither model has seen")

    fam_block = (fam.get("models", {}).get("static_ml", {}).get("per_family_recall", {}))
    hardest = sorted(fam_block.items(), key=lambda kv: kv[1]["recall"])[:3]
    easiest = sorted(fam_block.items(), key=lambda kv: -kv[1]["recall"])[:3]

    written = []

    # ----------------------------------------------------------------- Q&A --
    qa = f"""# Judge Q&A

Prepared answers, with the numbers taken from the committed artifacts. Regenerated
by `python -m docs.build_docs`, so nothing here can drift away from what the app
shows.

---

### "Where is generative AI actually used? Isn't this just a classifier?"

In three places, and only one of them is cosmetic.

1. **Structured attack generation.** The red-team agent does not write transactions.
   It writes an `AttackSpec` — which behavioural dial to move (amount, velocity,
   device, geography, merchant, timing), in which direction, and which detector
   signal that is meant to defeat. A deterministic simulator executes it.
2. **Weakness-driven mutation.** Each round we measure which signals the current
   model relies on for a family, by permuting features on that family's own rows,
   and the proposal is aimed at removing the strongest one.
3. **Content artifacts.** Synthetic scam text, plus a classifier over it. Both are
   demonstrations: the corpus is trivially separable by vocabulary, so the classifier
   is a sanity check on where content signals would attach — never evidence of
   detection efficacy. Every detection claim here rests on the transaction model.

The important design point is the split: **LLM creativity constrained by
payment-domain simulation rules.** Everything that touches the dataset is
deterministic and seeded, so the pipeline reproduces byte-for-byte whether the
specification came from Claude or from the offline heuristic.

**Say this before you are asked.** Demo mode uses **deterministic, committed
specifications** — every lineage node carries `spec_source: "heuristic"` and every
content artifact `source: "template"`. That is a reliability decision: the demo has
to run with no key and no network, and every committed figure has to reproduce from
one seed. So what you are watching is the loop's behaviour, not a single model call.

The **optional GenAI red team** generates the specification instead when
`ANTHROPIC_API_KEY` is set. It receives the same measured weakness and its output
goes through the same payment-domain constraint layer, stamped `spec_source: "llm"`
so the two are never confused. To show it live:

```bash
python -m src.generate.demo_specs --check   # readiness
python -m src.generate.demo_specs           # 5 live specs -> models/genai_spec_demo.json
```

With no key that script writes nothing and says so, rather than substituting
heuristic output for a model response.

### "Is the attacker scripted?"

Partly, and we are explicit about which part.

- **Predefined:** the {n_injectors} transaction injectors — the mechanics of how a family
  writes rows into the payment stream.
- **Specification-driven:** the dials. Any combination the constraint layer accepts is
  executable without new code, which is what the {n_param} PARAMETERIZED catalog entries mean.
- **Specification-generated:** which dial to move, and the strategy narrative. In Demo
  mode these come from the deterministic weakness-driven heuristic
  (`spec_source: "heuristic"`); the optional GenAI red team produces the same structure
  and stamps `spec_source: "llm"`.
- **Feedback-driven:** which family is attacked and which signal is targeted — both
  derived from measurement, not chosen in advance. Change the defense and the loop
  attacks something else.

### "Why not just use rules?"

At a matched {budget_pct} false-positive budget on the same held-out split:

| Detector | Recall | Precision | False positives / 1,000 genuine |
|---|---|---|---|
| Rules baseline | {pct(rules.get('recall'))} | {num(rules.get('precision'))} | {rules.get('false_positives_per_1000_legit', 0):.1f} |
| Static ML | {pct(static.get('recall'))} | {num(static.get('precision'))} | {static.get('false_positives_per_1000_legit', 0):.1f} |

And the rules here are not a strawman: they include the network-level rules a real
fraud team would write once it had the same counters the model gets — device shared
across cards, card seen on many devices, merchants seeing almost only first-time
cards. The thresholds are round, domain-chosen numbers, never fitted to this data.

The deeper answer is the one the leave-one-out experiment gives: rules and static ML
both fail the same way against a family nobody has written a rule for yet.

### "Your adaptive model scores worse than your static model. Doesn't that sink the whole idea?"

It would, if that table were the relevant one. It is not — it scores every detector on
the **original** attack distribution, which is the static model's home ground: that is
literally what it was trained on. The adaptive model carries several generations of
evolved attacks in its training set, which costs it a little there.

The comparison that matters is on the attacks that **moved**:
{h2h_line if h2h_line else "see the head-to-head table on the Benchmarks page"}.

We ship both tables. Showing only whichever one flattered us would be the easy version
of this submission.

### "Why should I trust synthetic data?"

You should not, unconditionally — so we measure it rather than asserting it. If any
single field separated fraud from genuine traffic cleanly, the model would be reading
the generator and every number would be meaningless.

- Strongest single feature: **{strongest}** at AUC **{strongest_auc}**.
- {fid.get('n_fail', 0)} failing checks and {fid.get('n_warn', 0)} warnings across {n_checks} automated fidelity checks.
- Fraud actors build ordinary-looking history before they are used, and that traffic
  is labelled **legitimate** — so "no history" is not a synonym for fraud.
- Attacks reuse cards, devices and merchants, so a fraudulent row is not trivially a
  first-ever card/merchant pair.

Two specific shortcuts were found and removed during development: `is_new_payee` once
fired on 99.9% of fraud versus 46% of genuine traffic, and the velocity features were
dead because every probing transaction used a fresh card ID. Both are in the fidelity
checks now precisely so they cannot come back.

### "Have you validated on real payment data?"

No. Every number is a simulation result. The necessary next step is a labelled
backtest on issuer data, followed by champion/challenger rollout — and until that
happens these figures say something about the *method*, not about production
performance.

### "Does this retrain during authorization?"

No. The online path reads precomputed counters, scores, calibrates, applies a
versioned policy and logs reason codes. It never trains and never calls a language
model. Retraining is an offline research loop on a cadence of days, and its output is
a candidate model.

### "What stops a bad adapted model from being deployed?"

Champion/challenger gates. A candidate is promoted only if attack recall improves by
at least {config.CHAMPION_CHALLENGER['min_attack_recall_gain']:.0%}, the false-positive
rate stays under {config.CHAMPION_CHALLENGER['max_fpr']:.1%} absolute and within
{config.CHAMPION_CHALLENGER['max_fpr_regression']:.1%} of the champion, no
previously-learned family drops by more than
{config.CHAMPION_CHALLENGER['max_prior_recall_drop']:.0%}, and overall PR-AUC does not
regress by more than {config.CHAMPION_CHALLENGER['max_overall_pr_auc_drop']:.2f}.

{"In the committed run, round(s) " + str(promoted) + " were promoted."
 if promoted else
 "**In the committed run, no candidate cleared every gate.** The adapted models learned "
 "the evolved attack and breached another constraint, so they would not be deployed. We "
 "ship both artifacts — the candidate and the promoted champion — and the benchmark "
 "tables label which is which. A gate that never fires is not a gate."}

### "Why is recall on {hardest[0][0] if hardest else 'the hardest family'} so low?"

Because of limited authorization-time observability, not because the model is
undertrained. {"The three hardest families are " + ", ".join(f"{k} ({v['recall']*100:.0f}%, n={v['n']})" for k, v in hardest) + "." if hardest else ""}
In first-party abuse and victim-authorized scams the genuine customer authenticates on
their own device and authorizes the payment. Every signal the authorization carries
says legitimate. The right control is friction, payee-risk intelligence and
post-transaction recall — not a hard decline — and reporting a low number here is more
honest than tuning until it looks better.

### "Why does leave-one-out improve so dramatically?"

Because the family is *intentionally absent* during the first training run. The
experiment measures unseen → learned adaptation; it does not measure zero-shot
detection. And `after learning` is an upper bound obtained by putting the family
directly into training — the closed loop has to *reach* that bound by generating the
family itself, and on the hardest families it does not get all the way.

### "Does {pct(hero['recall_after_learning']) if hero else 'a high recall'} mean production-level performance?"

No. It is performance on a held-out synthetic frame for that family, at
n={hero['n_test'] if hero else '—'}. Every family-level number is reported with its
sample size and a 95% Wilson interval for exactly this reason.

### "What happens if the attacker evolves again?"

That is the loop. Each round the attack moves to defeat whatever the model now relies
on, and some families keep recovering while others do not — those are reported as
residual frontiers. An attack that ends up sitting on the legitimate behavioural
centroid is not solved by more replay; it is solved by a different control surface.

The loop also knows when to stop. A family that stays a residual frontier across two
consecutive rounds is **retired** and the red team moves to the next-weakest target.
That is not giving up: continuing to hammer an unlearnable family burns replay
capacity on examples the model cannot separate, and we measured it dragging down
families it could. {("In the committed run it retired " + ", ".join(r["family"] for r in (loop.get("retired_frontiers") or [])) + ".") if loop.get("retired_frontiers") else ""}

### "Your text detector scores {text_auc}. Isn't that suspicious?"

Yes, and we say so on the page. The corpus is composed from a fixed slot vocabulary,
so the two classes are almost perfectly separable by vocabulary alone. **That number
measures the corpus, not the detector**, and it would not survive contact with real
scam messages. The text arm is included to show where content signals attach to the
architecture, and its score is published precisely so it is not mistaken for a
headline result.

### "What is the commercial value?"

Continuous adversarial testing *before* an attack becomes common in production. The
expensive part of fraud is not the attack you have a rule for; it is the weeks between
a new family appearing and enough labelled examples accumulating to retrain on. This
compresses that window by generating the family instead of waiting for it — and the
governance gates make the output safe to act on.

### "How much of the catalog do you actually simulate?"

{n_impl} of {n_total} have a dedicated injector, {n_param} more are reachable by
configuring an existing one, and {n_research} are characterized but **not** simulated.
Every entry carries that status in the data file and in the UI. We would rather
volunteer the ratio than have it counted during questioning.

Note the fan-in: {n_total} catalogued attacks map onto {n_injectors} transaction
injectors, because many attacks differ upstream of the payment and converge on the
same authorization footprint. That is a real property of the problem, not a gap we
are papering over — and it is why the atlas records the transaction signature
separately from the behavioural one.
"""
    p = DOCS / "JUDGE_QA.md"
    p.write_text(qa, encoding="utf-8")
    written.append(p)

    # ------------------------------------------------------------ 90-second --
    script = f"""# 90-second demo script

One browser tab, Demo mode, no network needed. Numbers below are the committed ones;
if the app shows something different, the app is right and this file is stale — run
`python -m docs.build_docs`.

---

**0:00–0:15 — the problem** *(Home)*

> "A fraud model can only learn from fraud that already happened. Generative AI made
> producing a *new* attack almost free — so new families now arrive faster than
> labelled history accumulates. This is a lab that goes looking for what our defense
> doesn't know yet."

Point at: `DISCOVER → SIMULATE → ATTACK → DETECT → ADAPT`.

**0:15–0:30 — breadth and generation** *(Threat Atlas → Generate)*

> "We catalogued {counts['total_attacks']} attacks across {counts['categories']} fraud
> surfaces. {counts['implemented']} are simulated end to end,
> {counts['parameterized']} more are configurable, and the rest we label research-only
> — we don't claim to simulate what we don't."

Switch to Generate, point at the specification box.

> "The red team never writes transactions. It writes a *specification* — move the device
> dial to trusted, because that's the signal the defense is leaning on. What you're seeing
> is Demo mode, which uses deterministic committed specifications so this runs the same way
> every time. Optionally the GenAI red team generates that specification instead. Either
> way a constraint layer refuses anything impossible on a real rail, and a deterministic
> simulator executes what survives."

**0:30–0:50 — the blind spot** *(Hero Demo, beats ② and ③)*

> "Here's {hf.replace('_', ' ') if hf else 'a fraud family'} — removed from training
> entirely. The defense is competent, correctly tuned, inside its false-positive
> budget. It catches **{pct(hero['recall_unseen']) if hero else '—'}** of it — on
> {hero['n_test'] if hero else '—'} held-out transactions."

Show the single transaction the stale defense approved.

**0:50–1:10 — the attack becomes training data** *(Hero Demo, beat ④)*

> "The lab discovers it, generates variants from a constrained specification, replays
> them into training, and retrains. No waiting for chargebacks. No waiting for labels."

**1:10–1:30 — the defense learns, and the guardrails hold** *(beats ⑤ and ⑥)*

> "**{pct(hero['recall_unseen']) if hero else '—'} → {pct(hero['recall_after_learning']) if hero else '—'}**{hero_ci}. And critically —"

Switch to beat ⑥.

> "— the false-positive rate, the families we never attacked, and the promotion gate.
> {"A candidate that improves detection but breaks something else does not ship, and in this run none of them did. That's the governance layer working, and we show it rather than hiding it." if not promoted else "A candidate ships only if it beats the model in force without breaking anything else."}"

**Closing line**

> "We don't wait for tomorrow's fraud to appear in historical data. We generate it
> today, and use it to train tomorrow's defense."

---

## If something breaks

- The app reads only committed artifacts — no network, no API key, no training.
- Every page has a static fallback in `docs/figures/` and in the walkthrough.
- If Streamlit will not start: open `docs/solution_walkthrough.docx`, which contains
  the same figures and the same numbers.
"""
    p = DOCS / "DEMO_SCRIPT_90S.md"
    p.write_text(script, encoding="utf-8")
    written.append(p)

    # ---------------------------------------------------------------- pitch --
    pitch = f"""# 3-minute pitch

Conversational. Do not read it out.

---

## The problem (35s)

Fraud detection is trained on labelled history: fraud happens, customers dispute,
chargebacks land, someone labels them, the model retrains. That loop takes weeks.

Generative AI broke the economics on the other side. Producing a new, plausible,
well-targeted attack — a fresh lure in fluent Hinglish, a synthetic identity with a
coherent digital footprint, a variant that sidesteps the control that just started
biting — now costs almost nothing. Attacks iterate in hours; defenses iterate in
weeks.

So the question is not "how accurate is your model on last year's fraud". It is "what
does it do the first time it meets something new, and how fast can it learn".

## The insight (25s)

If the problem is that we lack examples of attacks that haven't happened yet — then
generate them. Use the attacker's own tool against them: have an AI red team invent
the next attack, run it against our defense in a sandbox, and turn every failure into
training data.

The trick is that the generative part has to be *constrained*. A language model
inventing raw transaction rows produces nonsense that a fraud analyst spots in
seconds. So the model writes a *specification* — which behaviour to change and why —
and a deterministic simulator, bound by payment-domain rules, executes it.

## The system (40s)

Five stages. **Discover**: {counts['total_attacks']} attacks catalogued across
{counts['categories']} fraud surfaces, each labelled with whether we actually simulate
it. **Simulate**: a constrained engine producing a realistic portfolio — customer
archetypes, merchant profiles, weekend and payday rhythms, and fraud actors who build
ordinary-looking history before they strike. **Attack**: the red team measures which
signal the defense relies on and aims the next generation at removing it.
**Detect**: gradient boosting fused with an anomaly detector over authorization-time
features only — including network counters an issuer can't compute alone but a
network can. **Adapt**: whatever escapes goes into a replay buffer and trains the next
model, which then has to pass promotion gates.

## The proof (40s)

Take {hf.replace('_', ' ') if hf else 'an attack family'} out of training entirely.
The defense catches **{pct(hero['recall_unseen']) if hero else '—'}** of it — a real
blind spot, not a tuning problem. Let the lab generate that family and replay it:
**{pct(hero['recall_after_learning']) if hero else '—'}**{hero_ci}.

Quote the sample size out loud. It is the first thing a technical judge will ask for,
and every number in this project ships with it.

And against a *competent* rule set — one with the same network counters the model gets
— at a matched false-positive budget: rules {pct(rules.get('recall'))} recall, the
model {pct(static.get('recall'))}, at {static.get('false_positives_per_1000_legit', 0):.1f}
false positives per thousand genuine payments.

Then the one that matters. On the attacks that actually moved —
{h2h_line if h2h_line else "the final evolved generation"} — which is the whole point:
the static model is excellent on the fraud it was trained on and blind to the fraud
that came next.

## Feasibility (25s)

Online and offline are separate. Authorization reads precomputed counters, scores,
applies a versioned policy, logs reason codes — it never trains. Adaptation runs
offline, and every model it produces is a *challenger* that must clear explicit gates:
recall gain, a false-positive ceiling, no forgetting, no ranking regression.

{"In our committed run none of them cleared every gate. We ship that result rather than a nicer one — a gate that never fires isn't a gate." if not promoted else "In our committed run the gates passed, and we show the arithmetic."}

## The close (15s)

Some fraud stays hard at authorization time — when the genuine customer authenticates
and authorizes, there is very little to see. We report those as residual frontiers
instead of tuning until they look solved.

Everything here is synthetic and none of it is validated on real payment data. What we
are claiming is a method: we don't wait for tomorrow's fraud to appear in historical
data — we generate it today and use it to train tomorrow's defense.
"""
    p = DOCS / "PITCH_3MIN.md"
    p.write_text(pitch, encoding="utf-8")
    written.append(p)

    # ------------------------------------------------------------ why we win --
    why = f"""# Positioning against the judging criteria

Internal. Maps what exists in the repository to each official criterion, with literal
counts. Regenerated from artifacts.

## 1. Diversity of attacks identified

- **{counts['total_attacks']} distinct attacks** across **{counts['categories']} fraud
  surfaces** and {counts['rails']} payment rails, each with attacker objective, GenAI
  role, kill chain, transaction signature, behavioural signature, authorization-time
  observability, and post-settlement signals.
- Coverage is stated, not implied: {counts['implemented']} IMPLEMENTED,
  {counts['parameterized']} PARAMETERIZED, {counts['research_only']} RESEARCH_ONLY,
  {counts['future']} FUTURE.
- {counts['auth_time_hard']} entries are flagged as low or no visibility at
  authorization time — volunteering the limit of our own control surface.

## 2. Fidelity of simulated attacks

- Customer archetypes, merchant profiles, weekday/weekend and payday rhythms,
  shopping sessions, subscriptions, card issuance **and attrition** inside the window.
- Fraud actors carry cover traffic; attacks reuse cards, devices and merchants.
- {len(fid.get('checks', []))} automated fidelity checks; strongest single feature at
  AUC {fid.get('max_single_feature_auc', 0):.3f}, {fid.get('n_fail', 0)} failing.
- Temporal causality enforced and unit-tested; post-outcome fields hard-blocked.

## 3. Detection efficacy

- Recall {pct(m.get('recall'), 1)}, precision {num(m.get('precision'))},
  PR-AUC {num(m.get('pr_auc'))}, false-positive rate
  {pct(m.get('false_positive_rate'), 2)} against a
  {pct(config.TARGET_MAX_FPR, 0)} budget.
- Rules / static / adaptive compared **at a matched false-positive budget**, with a
  competent rule set including network-level rules.
- Family recall on an unseen fraud-enriched frame, with sample sizes and 95% Wilson
  intervals.
- Leave-one-attack-family-out across {len(loao.get('families', {}))} families.
- Head-to-head on the evolved attacks: {h2h_line if h2h_line else "see head_to_head.json"}.

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
"""
    p = DOCS / "WHY_WE_WIN.md"
    p.write_text(why, encoding="utf-8")
    written.append(p)

    return written
