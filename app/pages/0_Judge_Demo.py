"""2-MINUTE JUDGE DEMO — the whole story on one page, in plain language.

The question a judge should leave with an answer to:

    Can a fraud detector learn a new scam before criminals exploit it?

One attack carries the whole arc — otp_relay, the closed-loop family with the
largest, best-powered adaptive recovery in the committed run. Every number on
this page is read from a committed artifact produced by `python -m src.pipeline`;
nothing is recomputed live, nothing is hardcoded, and the app needs no API key.

Human name for the hero attack (from the Threat Atlas):
    "Hyper-Personalized Cardholder Spear Phishing With Live OTP Relay"

Artifacts used
  models/metrics.json        portfolio detector at its operating point
  models/loop_history.json   the closed loop: round-1 promotion, weakness, gates
  models/head_to_head.json   static vs adaptive on the final evolved generation
  models/judge_hero.json     one concrete otp_relay transaction, before/after
  data/summary.json          the synthetic portfolio
  src/identify/attacks.json  the human-readable attack name and its fraud surface
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from common import (PALETTE, STRETCH, get_taxonomy, load_head_to_head,
                    load_json, load_loop_history, load_metrics, load_summary,
                    page_setup)

import config

page_setup("2-Minute Judge Demo", "🎯")

# --- semantic colours (section 14) ------------------------------------------
C_ATTACK = PALETTE["danger"]   # red   — the attack
C_CURRENT = PALETTE["info"]    # blue  — the current / static detector
C_CAND = "#7C3AED"             # purple— the candidate / adapted detector
C_PASS = PALETTE["safe"]       # green — passed / approved
C_STEP = PALETTE["accent"]     # amber — step-up / friction
C_GREY = PALETTE["muted"]

HERO_FAMILY = "otp_relay"
HERO_NAME = "Hyper-Personalized Cardholder Spear Phishing With Live OTP Relay"
REGEN = "python -m src.pipeline   ·   hero transaction: python -m src.experiments.judge_hero"

# The six-phase spine shown as a progress bar, and which of the 8 stages sits
# in each phase.
PHASES = ["Attack", "Evade", "Analyse", "Evolve", "Retrain", "Govern"]
STAGE_TO_PHASE = [0, 1, 1, 2, 3, 4, 5, 5]
STAGE_TITLES = [
    "① Meet the attack", "② Test the current detector", "③ Inspect what escaped",
    "④ Discover the weakness", "⑤ Evolve the attack", "⑥ Retrain the defence",
    "⑦ Run the safety gate", "⑧ The result",
]


# --- small render helpers ----------------------------------------------------
def pct(x, digits: int = 0) -> str:
    return "—" if x is None else f"{x * 100:.{digits}f}%"


def badge(text: str, color: str) -> str:
    return (f'<span style="display:inline-block;padding:3px 12px;border-radius:14px;'
            f'font-size:.74rem;font-weight:700;letter-spacing:.03em;'
            f'background:{color}22;color:{color};border:1px solid {color}55">{text}</span>')


def progress_bar(active_phase: int):
    cells = st.columns(len(PHASES))
    for i, (cell, name) in enumerate(zip(cells, PHASES)):
        on = i == active_phase
        done = i < active_phase
        col = PALETTE["primary"] if on else (C_PASS if done else C_GREY)
        weight = "800" if on else "600"
        mark = "▸ " if on else ("✓ " if done else "")
        cell.markdown(
            f'<div style="text-align:center;font-size:.80rem;font-weight:{weight};'
            f'color:{col};border-top:3px solid {col};padding-top:6px">{mark}{name}</div>',
            unsafe_allow_html=True)


def agent_mode_badge():
    if config.llm_available():
        st.markdown(badge("🤖 Agent Mode: Claude-Assisted", C_CAND), unsafe_allow_html=True)
        st.caption("An API key is present, so the red-team **strategist** may be Claude. It only "
                   "proposes a structured attack *recipe*; the same constraint validator, the "
                   "same simulator, the same ML detector and the same deterministic governance "
                   "run regardless. The committed numbers on this page were produced offline.")
    else:
        st.markdown(badge("⚪ Agent Mode: Deterministic Offline", C_CURRENT),
                    unsafe_allow_html=True)
        st.caption("No API key present — the attack strategy comes from a weakness-driven "
                   "heuristic. Claude is an **optional** strategist; when present it only proposes "
                   "a structured attack recipe. The constrained simulator builds the "
                   "transactions, the ML detector learns, and deterministic governance decides "
                   "promotion — never Claude.")


# --- load artifacts ----------------------------------------------------------
metrics = load_metrics()
loop = load_loop_history()
h2h = load_head_to_head()
summary = load_summary()
hero = load_json(str(config.MODELS_DIR / "judge_hero.json"))
tax = get_taxonomy()

# round 1 is the genuine PROMOTE; it carries the clean end-to-end story.
round1 = loop["history"][0] if (loop and loop.get("history")) else None
otp1 = (round1 or {}).get("families", {}).get(HERO_FAMILY, {})


# =============================================================================
# HEADER — always visible
# =============================================================================
st.markdown('<div class="kicker">2-Minute Judge Demo · synthetic data only</div>',
            unsafe_allow_html=True)
st.title("Can a fraud detector learn a new scam before criminals exploit it?")
st.markdown(
    "#### Watch a synthetic fraud attack evade the current detector, see the system learn from "
    "it, and verify that genuine customers remain protected.")

cols = st.columns([1, 1])
with cols[0]:
    agent_mode_badge()
with cols[1]:
    st.markdown(badge("🔒 Synthetic payment data only — no real cards or customers", C_PASS),
                unsafe_allow_html=True)
    st.caption("A reproducible simulation. Nothing here is validated on real payment data or "
               "endorsed by any payment network.")

st.divider()

# --- simplified metric cards (section 8) ------------------------------------
if metrics and otp1 and loop:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Frauds caught, out of 100", f"{round(metrics['recall'] * 100)}",
              help="Recall of the current detector across the whole synthetic portfolio, at its "
                   "tuned operating point. Technical term: recall.")
    m1.caption(f"current detector · recall {pct(metrics['recall'], 1)}")

    fp_per_1000 = metrics["false_positive_rate"] * 1000
    m2.metric("Genuine flagged, per 1,000", f"{fp_per_1000:.0f}",
              help="How many genuine payments out of 1,000 the detector challenges or declines. "
                   "Technical term: false-positive rate. Budget is 20 per 1,000 (2%).")
    m2.caption(f"false-positive rate {pct(metrics['false_positive_rate'], 2)} · budget ≤ 2%")

    sr, ar = otp1.get("stale_recall"), otp1.get("adapted_recall")
    m3.metric("Hero attack: before → after", f"{pct(sr)} → {pct(ar)}",
              delta=f"+{round((ar - sr) * 100)} pts" if (sr is not None and ar is not None) else None,
              help="The evolved OTP-relay scam: what the current detector catches, versus what "
                   "the detector catches after learning from it. Closed-loop round 1.")
    m3.caption(f"evolved OTP-relay · n = {otp1.get('n', '—')}")

    promoted = loop.get("promotion", {}).get("promoted_rounds", [])
    m4.metric("Candidate models approved", f"{len(promoted)} of {loop.get('rounds', '—')}",
              help="Each round the blue team trains a candidate; governance promotes it only if "
                   "it clears every safety gate. The rest are held back — governance has teeth.")
    m4.caption(f"{loop.get('rounds', 0) - len(promoted)} held back by the safety gate")

with st.expander("How the synthetic data works", expanded=False):
    if summary:
        d1, d2, d3 = st.columns(3)
        d1.metric("Synthetic transactions", f"{summary['n_transactions']:,}")
        d2.metric("Of which fraud", f"{summary['n_fraud']:,}",
                  f"{summary['fraud_rate'] * 100:.1f}% base rate")
        d3.metric("Cardholders · merchants",
                  f"{summary['n_cardholders']:,} · {summary['n_merchants']:,}")
    st.markdown(
        "The simulator creates **one** synthetic payment ecosystem of genuine and fraudulent "
        "transactions. Fraud generators inject different **attack families** into it. During "
        "adaptation the simulated attacker generates additional **evolved attack batches**, and "
        "the transactions that escape the detector are added to a **practice dataset** "
        "(replay buffer) used to train the next candidate.\n\n"
        "- **Portfolio** — the whole synthetic stream above.\n"
        "- **Genuine transactions** — the majority; they set the false-positive budget.\n"
        "- **Fraud transactions** — a small labelled minority across many families.\n"
        "- **Attack families** — distinct fraud behaviours (OTP relay, card testing, bust-out …).\n"
        "- **Evolved attack batches** — new variants the attacker generates during the loop.\n"
        "- **Practice dataset** — escaped fraud, retained cumulatively so nothing is forgotten.\n\n"
        "The 150,000-transaction portfolio is generated **once** — not separately for every "
        "attack. Evolved batches are additional, smaller frames layered on during adaptation.")
    st.caption("**Fraud surface** — the area of the payment system where fraud occurs "
               "(onboarding, account access, card payments, transfers, wallets, merchants). "
               f"This lab catalogues {tax.summary_counts()['total_attacks']} attacks across "
               f"{tax.summary_counts()['categories']} such surfaces.")

st.divider()


# =============================================================================
# STAGE CONTENT
# =============================================================================
def stage_meet():
    a = next((x for x in tax.attacks if x.maps_to_injector == HERO_FAMILY
              and x.simulator_status == "IMPLEMENTED"), None)
    surface = a.category if a else "Social Engineering"
    st.markdown(badge(f"Fraud surface — {surface}", C_ATTACK), unsafe_allow_html=True)
    st.subheader(HERO_NAME)
    st.markdown(
        "The attacker spear-phishes a genuine cardholder, walks them through a real bank "
        "step-up challenge, and **relays the one-time passcode live** — so the payment passes "
        "3-D Secure and the OTP check *looks* completely legitimate. Then they spend from their "
        "own device.")
    if summary:
        spec0 = summary.get("attack_specs", {}).get(HERO_FAMILY, {})
        if spec0:
            st.caption(f"Baseline strategy (generation 0): *{spec0.get('strategy', '')}*.")
    st.info("Why it is hard: the strongest fraud signals — a failed step-up, an unverified OTP — "
            "never fire, because the real customer really did authenticate. That is the gap a "
            "static model cannot close on its own.", icon="🎣")


def stage_test():
    rank = (loop.get("focus_selection", {}).get("initial_ranking", []) if loop else [])
    row = next((r for r in rank if r["family"] == HERO_FAMILY), None)
    c1, c2 = st.columns([1, 1.3])
    with c1:
        if row:
            st.metric("Current detector catches", pct(row["recall"]),
                      f"n = {row['n']}", delta_color="off")
        st.markdown(badge("current detector", C_CURRENT), unsafe_allow_html=True)
    with c2:
        st.markdown(
            "Run the current, competent, correctly-tuned detector against OTP-relay fraud and it "
            f"catches only about **{pct(row['recall']) if row else '—'}** of it — one of the "
            "three learnable families it handles **worst**. Across the whole portfolio it catches "
            f"**{pct(metrics['recall']) if metrics else '—'}**, so this family is a genuine soft "
            "spot, not general weakness.")
        st.caption("Targets are chosen from what the detector is *measured* to be weakest at — "
                   "nothing is decided in advance.")


def stage_escaped():
    if not hero:
        st.info("Run `python -m src.experiments.judge_hero` to generate the concrete example.",
                icon="ℹ️")
        return
    txn, s = hero["transaction"], hero["stale"]
    st.markdown("**One fraudulent transaction the current detector waved through**")
    st.caption(f"Transaction {txn.get('txn_id')} · synthetic cardholder "
               f"{txn.get('cardholder_id')} · {str(txn.get('timestamp', ''))[:16]}")
    f = st.columns(4)
    f[0].metric("Amount", f"₹{txn['amount']:,.0f}")
    f[1].metric("Merchant category", txn["mcc"].split("_", 1)[-1].replace("_", " "))
    f[2].metric("Device", "new to this card" if str(txn.get("device_id", "")).startswith("NEW")
                else "known device")
    f[3].metric("OTP / 3-D Secure", "passed" if txn.get("otp_verified") else "not done")
    g = st.columns(4)
    g[0].metric("Home city?", "yes" if txn.get("distance_from_home_km", 99) < 25 else "no",
                f"{txn.get('distance_from_home_km', 0):.0f} km from home", delta_color="off")
    g[1].metric("New payee?", "yes" if txn.get("is_new_payee") else "no")
    g[2].metric("Account age", f"{txn.get('account_age_days', 0):,} days")
    g[3].metric("Actual label", "FRAUD", "synthetic ground truth", delta_color="off")

    d1, d2 = st.columns(2)
    with d1:
        st.markdown(badge("current detector · APPROVE", C_STEP if s["action"] != "APPROVE"
                          else C_ATTACK), unsafe_allow_html=True)
        st.metric("Decision", s["action"], f"fraud probability {s['probability']:.3f}",
                  delta_color="off")
    with d2:
        codes = hero.get("reason_codes") or []
        st.markdown("**Legacy risk rules that fired:** " +
                    ("; ".join(codes) if codes else "**none**"))
        st.caption("The evolved variant trips *zero* hand-written rules — normal amount, "
                   "ordinary merchant, home city, a single payment. That is exactly why a "
                   "rules-and-static-model stack lets it through.")


def stage_weakness():
    w = otp1.get("weakness", {})
    signals = w.get("relied_signals", [])[:5]
    st.markdown("The blue team measures **which signals the detector's ranking of this family "
                "actually depends on** — a permutation test on the model's own decisions.")
    if signals:
        labels = {"is_3ds": "3-D Secure passed", "mcc_risk": "merchant risk level",
                  "time_since_last_hours": "time since last payment",
                  "log_distance": "distance from home", "ip_card_count_prior": "cards seen on IP",
                  "card_history_depth": "card history depth", "log_amount": "amount",
                  "amount_zscore": "amount vs card norm"}
        fig = go.Figure(go.Bar(
            x=[s["auc_drop"] for s in signals][::-1],
            y=[labels.get(s["feature"], s["feature"]) for s in signals][::-1],
            orientation="h", marker_color=C_CURRENT))
        fig.update_layout(height=240, margin=dict(t=8, l=8, r=8, b=8),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          xaxis_title="how much the detector leans on it (AUC drop when removed)")
        st.plotly_chart(fig, width=STRETCH)
    st.info("The detector leans hardest on **3-D Secure** — but this attack *passes* 3-D Secure "
            "by relaying the OTP, so that signal is useless here. The next strongest signal the "
            "attacker can cheaply remove is the **merchant's risk level**. That becomes the "
            "target.", icon="🔍")


def stage_evolve():
    spec = otp1.get("spec", {})
    cl = otp1.get("constraint_layer", {})
    spec0 = (summary or {}).get("attack_specs", {}).get(HERO_FAMILY, {}) if summary else {}
    st.markdown(badge("Mutation selected by deterministic weakness-driven rules", C_GREY)
                if not config.llm_available()
                else badge("Mutation proposed by the strategist, then validated", C_CAND),
                unsafe_allow_html=True)
    st.markdown(f"**Strategy:** *{spec.get('strategy', '')}*")

    st.markdown("**What changed, and why**")
    rows = [("Merchant", spec0.get("merchant_behavior", "—"),
             spec.get("merchant_behavior", "—"),
             "move off high-risk merchants so the 'merchant risk' signal stops firing"),
            ("Attack intensity", f"{spec0.get('intensity', 1.0):.2f}",
             f"{spec.get('intensity', '—'):.2f}", "dial the campaign down to stay quiet")]
    import pandas as pd
    st.dataframe(pd.DataFrame(rows, columns=["Signal", "Before (gen 0)", "Evolved", "Why it changed"]),
                 width=STRETCH, hide_index=True)

    accepted = cl.get("accepted", True)
    corr = cl.get("corrections", [])
    if accepted and not corr:
        st.markdown(badge("✓ constraint validator: accepted, no corrections", C_PASS),
                    unsafe_allow_html=True)
    elif accepted:
        st.markdown(badge(f"✓ accepted after {len(corr)} correction(s)", C_STEP),
                    unsafe_allow_html=True)
    st.caption("Every proposal — heuristic or Claude — passes the **same** payment-domain "
               "validator, which clamps or rejects anything impossible before a single "
               "transaction is generated. The constrained simulator then builds the evolved "
               "batch deterministically.")
    sr = otp1.get("stale_recall")
    st.error(f"The evolution works: faced with the evolved variant, the current detector's catch "
             f"rate collapses to **{pct(sr)}**. The attack has slipped its leash.", icon="📉")


def stage_retrain():
    comp = round1.get("replay_composition", {}) if round1 else {}
    ar, sr = otp1.get("adapted_recall"), otp1.get("stale_recall")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown(
            "The escaped variants are added to the cumulative **practice dataset** — every past "
            "generation is retained, so the detector cannot fix the new scam by forgetting an old "
            "one. The blue team trains a **candidate** on the original data plus the whole "
            "buffer.")
        if comp:
            st.caption(f"Practice dataset this round: {comp.get('total_rows', 0):,} rows "
                       f"({comp.get('by_family', {}).get(HERO_FAMILY, 0)} of them evolved "
                       f"OTP-relay), spanning every family so nothing is forgotten.")
    with c2:
        st.metric("OTP-relay caught after learning", pct(ar),
                  f"+{round((ar - sr) * 100)} pts vs {pct(sr)}"
                  if (ar is not None and sr is not None) else None)
        st.markdown(badge("candidate detector", C_CAND), unsafe_allow_html=True)
    st.success("The candidate re-learns the evolved attack from replay alone — no waiting for it "
               "to show up in labelled production history.", icon="🧠")


def stage_gate():
    cc = round1.get("champion_challenger", {}) if round1 else {}
    gates = cc.get("gates", [])
    friendly = {
        "attack_recall_gain": "Caught more of the new attack than the model in force",
        "absolute_false_positive_ceiling": "Genuine-customer friction stayed within budget",
        "false_positive_regression": "Did not flag more genuine customers than before",
        "no_catastrophic_forgetting": "Did not forget previously-learned attacks",
        "overall_ranking_quality": "Overall detection quality held or improved",
    }
    st.markdown("A candidate is promoted **only** if it clears every gate. These are computed "
                "from measured values, never hand-set.")
    for g in gates:
        ok = g["passed"]
        icon = "✅" if ok else "⛔"
        color = C_PASS if ok else C_ATTACK
        st.markdown(
            f'<div style="padding:8px 12px;margin-bottom:6px;border-left:4px solid {color};'
            f'background:{color}11;border-radius:6px">{icon} <b>{friendly.get(g["gate"], g["gate"])}</b>'
            f'<br><span style="color:{C_GREY};font-size:.82rem">{g["detail"]}</span></div>',
            unsafe_allow_html=True)
    decision = cc.get("decision", "—")
    if decision == "PROMOTE":
        st.markdown(badge("✅ APPROVED FOR SHADOW EVALUATION", C_PASS), unsafe_allow_html=True)
        st.caption("Never 'deployed to production' — a promoted candidate would still enter a "
                   "shadow / controlled-rollout period this prototype does not simulate.")
    else:
        st.markdown(badge(f"⛔ CANDIDATE REJECTED — {cc.get('summary', '')}", C_ATTACK),
                    unsafe_allow_html=True)
    st.info("A candidate is not accepted merely because it catches more fraud. It must also "
            "protect genuine customers and keep what it already knew.", icon="⚖️")


def stage_result():
    st.subheader("Did adaptation catch more evolved fraud without punishing genuine customers?")
    if h2h:
        models = h2h.get("models", {})
        static = models.get("static_defense", {})
        champ = models.get("promoted_champion", {})
        cand = models.get("adaptive_defense", {})

        def rec(m):
            return m.get("evolved_family_recall", {}).get(HERO_FAMILY, {}).get("recall")
        s_r, c_r, a_r = rec(static), rec(champ), rec(cand)
        n = static.get("evolved_family_recall", {}).get(HERO_FAMILY, {}).get("n")
        fig = go.Figure()
        fig.add_bar(x=["Current<br>(static)", "Approved<br>candidate", "Final candidate<br>(held back)"],
                    y=[s_r, c_r, a_r], marker_color=[C_CURRENT, C_PASS, C_CAND],
                    text=[pct(s_r), pct(c_r), pct(a_r)], textposition="outside")
        fig.update_layout(height=320, yaxis_range=[0, 0.85], yaxis_title="OTP-relay caught",
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          margin=dict(t=20, b=10),
                          title=f"On the FINAL evolved generation · matched false-positive budget "
                                f"· n = {n}")
        st.plotly_chart(fig, width=STRETCH)
        sfpr = static.get("false_positive_rate")
        afpr = cand.get("false_positive_rate")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Current detector", pct(s_r), "static, pre-loop", delta_color="off")
        cc2.metric("Governance-approved", pct(c_r),
                   f"+{round((c_r - s_r) * 100)} pts" if (c_r and s_r) else None)
        cc3.metric("Genuine-customer friction", pct(afpr, 2),
                   f"{(afpr - sfpr) * 1000:+.0f} per 1,000 vs current"
                   if (afpr is not None and sfpr is not None) else None,
                   delta_color="inverse")
        st.caption("Both detectors re-thresholded to spend the **same** false-positive budget, so "
                   "the recall numbers are like-for-like. The adapted detector catches far more "
                   "evolved OTP-relay fraud **and** flags fewer genuine customers.")
    if hero:
        a, s = hero["adapted"], hero["stale"]
        st.markdown(f"On the concrete transaction from stage ③, the adapted detector now returns "
                    f"**{a['action']}** (fraud probability {s['probability']:.3f} → "
                    f"{a['probability']:.3f}) instead of waving it through — it recognised the "
                    "residual pattern the evolved attack could not hide: a brand-new device "
                    "relaying a live OTP to a new payee.")
    st.warning("Honest limits: this is one round's promotion. Across the full run only "
               f"{len(loop.get('promotion', {}).get('promoted_rounds', []))} of "
               f"{loop.get('rounds', '—')} candidates cleared every gate — later, stealthier "
               "generations were held back, and some families stay hard at authorization time. "
               "All results are on synthetic held-out data and are not production-performance "
               "claims.", icon="⚠️")


STAGES = [stage_meet, stage_test, stage_escaped, stage_weakness, stage_evolve,
          stage_retrain, stage_gate, stage_result]

# =============================================================================
# STEPPER
# =============================================================================
if "judge_started" not in st.session_state:
    st.session_state.judge_started = False
if "judge_stage" not in st.session_state:
    st.session_state.judge_stage = 0

if not (metrics and loop and round1):
    st.info("Core artifacts are missing. Run `python -m src.pipeline` to generate them, then "
            "reload.", icon="ℹ️")
    st.stop()

if not st.session_state.judge_started:
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
        if st.button("▶  Run 2-Minute Demo", type="primary", width=STRETCH):
            st.session_state.judge_started = True
            st.session_state.judge_stage = 0
            st.rerun()
        st.caption("Eight stages, about fifteen seconds each — Attack → Evade → Analyse → "
                   "Evolve → Retrain → Govern.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

stage = st.session_state.judge_stage
progress_bar(STAGE_TO_PHASE[stage])
st.markdown(f"### {STAGE_TITLES[stage]}")
STAGES[stage]()

st.divider()
nav = st.columns([1, 1, 2, 1])
if nav[0].button("⟵ Back", disabled=stage == 0, width=STRETCH):
    st.session_state.judge_stage = max(0, stage - 1)
    st.rerun()
if nav[1].button("Restart", width=STRETCH):
    st.session_state.judge_started = False
    st.session_state.judge_stage = 0
    st.rerun()
nav[2].markdown(f"<div style='text-align:center;color:{C_GREY};padding-top:8px'>"
                f"Stage {stage + 1} of {len(STAGES)}</div>", unsafe_allow_html=True)
if stage < len(STAGES) - 1:
    if nav[3].button("Next ⟶", type="primary", width=STRETCH):
        st.session_state.judge_stage = stage + 1
        st.rerun()
else:
    nav[3].markdown(badge("✓ complete", C_PASS), unsafe_allow_html=True)

# --- technical evidence + glossary ------------------------------------------
st.divider()
with st.expander("Technical evidence (exact figures, sample sizes, reproduction)"):
    if metrics:
        t = st.columns(3)
        t[0].metric("Recall", pct(metrics["recall"], 1))
        t[1].metric("Precision", pct(metrics["precision"], 1))
        t[2].metric("F1", f"{metrics['f1']:.3f}")
        t = st.columns(3)
        t[0].metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
        t[1].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        t[2].metric("False-positive rate", pct(metrics["false_positive_rate"], 2))
        st.caption(f"Portfolio detector measured on a held-out, time-split test set of "
                   f"{metrics.get('n_eval', 0):,} transactions ({metrics.get('n_fraud_eval', 0)} "
                   f"fraud). Train/validation/test are separated in time; the model uses "
                   "authorization-time features only, with no refund/dispute/chargeback outcomes.")
    if h2h and otp1:
        st.markdown("**Hero attack (otp_relay), with 95% confidence intervals**")
        import pandas as pd
        models = h2h.get("models", {})
        ev = lambda k: models.get(k, {}).get("evolved_family_recall", {}).get(HERO_FAMILY, {})
        rows = []
        for key, label in [("static_defense", "current / static"),
                           ("promoted_champion", "approved candidate"),
                           ("adaptive_defense", "final candidate")]:
            r = ev(key)
            if r:
                ci = r.get("ci95", [None, None])
                rows.append({"model": label, "otp_relay recall": pct(r.get("recall"), 1),
                             "95% CI": f"{ci[0]*100:.0f}–{ci[1]*100:.0f}%" if ci[0] is not None else "—",
                             "n": r.get("n")})
        st.dataframe(pd.DataFrame(rows), width=STRETCH, hide_index=True)
        st.caption(f"Closed-loop round 1 (promoted): otp_relay {pct(otp1.get('stale_recall'))} → "
                   f"{pct(otp1.get('adapted_recall'))} on n = {otp1.get('n')}. Head-to-head frame "
                   f"above is the final evolved generation on n = {ev('static_defense').get('n')}.")
    st.code(REGEN, language="bash")

with st.expander("Plain-language glossary"):
    st.markdown(
        "| On this page | Technical term |\n|---|---|\n"
        "| Simulated attacker | Red team |\n"
        "| Fraud detector | Blue team |\n"
        "| Attack recipe | Attack specification |\n"
        "| Practice dataset | Replay buffer |\n"
        "| Current approved model | Champion |\n"
        "| Candidate model | Challenger |\n"
        "| Safety check / gate | Promotion gate |\n"
        "| Frauds caught | Recall |\n"
        "| Genuine payments incorrectly flagged | False-positive rate |\n"
        "| Accidentally using future information | Temporal leakage |\n")

st.caption("Every figure on this page is read from a committed artifact produced by "
           "`python -m src.pipeline`. Reproducible experiment artifacts — nothing is shown as "
           "live that was not computed offline and committed.")
