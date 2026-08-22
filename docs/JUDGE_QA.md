# Judge Q&A

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

- **Predefined:** the 11 transaction injectors — the mechanics of how a family
  writes rows into the payment stream.
- **Specification-driven:** the dials. Any combination the constraint layer accepts is
  executable without new code, which is what the 19 PARAMETERIZED catalog entries mean.
- **Specification-generated:** which dial to move, and the strategy narrative. In Demo
  mode these come from the deterministic weakness-driven heuristic
  (`spec_source: "heuristic"`); the optional GenAI red team produces the same structure
  and stamps `spec_source: "llm"`.
- **Feedback-driven:** which family is attacked and which signal is targeted — both
  derived from measurement, not chosen in advance. Change the defense and the loop
  attacks something else.

### "Why not just use rules?"

At a matched 2% false-positive budget on the same held-out split:

| Detector | Recall | Precision | False positives / 1,000 genuine |
|---|---|---|---|
| Rules baseline | 10% | 0.068 | 12.9 |
| Static ML | 68% | 0.242 | 19.7 |

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
49% for the static defense versus 48% for the adaptive one, on 2703 fraudulent transactions of the final evolved generation from a seed neither model has seen.

We ship both tables. Showing only whichever one flattered us would be the easy version
of this submission.

### "Why should I trust synthetic data?"

You should not, unconditionally — so we measure it rather than asserting it. If any
single field separated fraud from genuine traffic cleanly, the model would be reading
the generator and every number would be meaningless.

- Strongest single feature: **amount_vs_card_max_prior** at AUC **0.722**.
- 0 failing checks and 0 warnings across 11 automated fidelity checks.
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
at least 5%, the false-positive
rate stays under 2.5% absolute and within
0.5% of the champion, no
previously-learned family drops by more than
10%, and overall PR-AUC does not
regress by more than 0.03.

In the committed run, round(s) [1] were promoted.

### "Why is recall on friendly_fraud so low?"

Because of limited authorization-time observability, not because the model is
undertrained. The three hardest families are friendly_fraud (24%, n=162), adversarial_mimicry (53%, n=270), wallet_provisioning (62%, n=162).
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

### "Does 86% mean production-level performance?"

No. It is performance on a held-out synthetic frame for that family, at
n=44. Every family-level number is reported with its
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
families it could. 

### "Your text detector scores 1.000. Isn't that suspicious?"

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

12 of 45 have a dedicated injector, 19 more are reachable by
configuring an existing one, and 14 are characterized but **not** simulated.
Every entry carries that status in the data file and in the UI. We would rather
volunteer the ratio than have it counted during questioning.

Note the fan-in: 45 catalogued attacks map onto 11 transaction
injectors, because many attacks differ upstream of the payment and converge on the
same authorization footprint. That is a real property of the problem, not a gap we
are papering over — and it is why the atlas records the transaction signature
separately from the behavioural one.
