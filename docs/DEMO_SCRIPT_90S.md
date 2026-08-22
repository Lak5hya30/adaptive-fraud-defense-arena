# 90-second demo script

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

> "We catalogued 45 attacks across 6 fraud
> surfaces. 12 are simulated end to end,
> 19 more are configurable, and the rest we label research-only
> — we don't claim to simulate what we don't."

Switch to Generate, point at the specification box.

> "The red team never writes transactions. It writes a *specification* — move the device
> dial to trusted, because that's the signal the defense is leaning on. What you're seeing
> is Demo mode, which uses deterministic committed specifications so this runs the same way
> every time. Optionally the GenAI red team generates that specification instead. Either
> way a constraint layer refuses anything impossible on a real rail, and a deterministic
> simulator executes what survives."

**0:30–0:50 — the blind spot** *(Hero Demo, beats ② and ③)*

> "Here's bust out — removed from training
> entirely. The defense is competent, correctly tuned, inside its false-positive
> budget. It catches **16%** of it — on
> 44 held-out transactions."

Show the single transaction the stale defense approved.

**0:50–1:10 — the attack becomes training data** *(Hero Demo, beat ④)*

> "The lab discovers it, generates variants from a constrained specification, replays
> them into training, and retrains. No waiting for chargebacks. No waiting for labels."

**1:10–1:30 — the defense learns, and the guardrails hold** *(beats ⑤ and ⑥)*

> "**16% → 86%** (n=44 held-out, 95% interval 73–94%). And critically —"

Switch to beat ⑥.

> "— the false-positive rate, the families we never attacked, and the promotion gate.
> A candidate ships only if it beats the model in force without breaking anything else."

**Closing line**

> "We don't wait for tomorrow's fraud to appear in historical data. We generate it
> today, and use it to train tomorrow's defense."

---

## If something breaks

- The app reads only committed artifacts — no network, no API key, no training.
- Every page has a static fallback in `docs/figures/` and in the walkthrough.
- If Streamlit will not start: open `docs/solution_walkthrough.docx`, which contains
  the same figures and the same numbers.
