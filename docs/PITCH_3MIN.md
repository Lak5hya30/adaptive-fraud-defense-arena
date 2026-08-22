# 3-minute pitch

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

Five stages. **Discover**: 45 attacks catalogued across
6 fraud surfaces, each labelled with whether we actually simulate
it. **Simulate**: a constrained engine producing a realistic portfolio — customer
archetypes, merchant profiles, weekend and payday rhythms, and fraud actors who build
ordinary-looking history before they strike. **Attack**: the red team measures which
signal the defense relies on and aims the next generation at removing it.
**Detect**: gradient boosting fused with an anomaly detector over authorization-time
features only — including network counters an issuer can't compute alone but a
network can. **Adapt**: whatever escapes goes into a replay buffer and trains the next
model, which then has to pass promotion gates.

## The proof (40s)

Take bust out out of training entirely.
The defense catches **16%** of it — a real
blind spot, not a tuning problem. Let the lab generate that family and replay it:
**86%** (n=44 held-out, 95% interval 73–94%).

Quote the sample size out loud. It is the first thing a technical judge will ask for,
and every number in this project ships with it.

And against a *competent* rule set — one with the same network counters the model gets
— at a matched false-positive budget: rules 10% recall, the
model 68%, at 19.7
false positives per thousand genuine payments.

Then the one that matters. On the attacks that actually moved —
49% for the static defense versus 48% for the adaptive one, on 2703 fraudulent transactions of the final evolved generation from a seed neither model has seen — which is the whole point:
the static model is excellent on the fraud it was trained on and blind to the fraud
that came next.

## Feasibility (25s)

Online and offline are separate. Authorization reads precomputed counters, scores,
applies a versioned policy, logs reason codes — it never trains. Adaptation runs
offline, and every model it produces is a *challenger* that must clear explicit gates:
recall gain, a false-positive ceiling, no forgetting, no ranking regression.

In our committed run the gates passed, and we show the arithmetic.

## The close (15s)

Some fraud stays hard at authorization time — when the genuine customer authenticates
and authorizes, there is very little to see. We report those as residual frontiers
instead of tuning until they look solved.

Everything here is synthetic and none of it is validated on real payment data. What we
are claiming is a method: we don't wait for tomorrow's fraud to appear in historical
data — we generate it today and use it to train tomorrow's defense.
