"""Pillar 2 — GENERATE: structured attack specifications, the payment-domain
constraint layer, the constrained simulator, and the fidelity diagnostics that
say whether any of it is worth training on.

Demo mode (default): renders instantly from the committed seeded dataset.
Live mode: regenerate a fresh portfolio on demand (behind an explicit button).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (PALETTE, STRETCH, is_demo, llm_status_badge, load_artifacts,
                    load_dataset_cached, load_fidelity, load_summary, mode_selector,
                    page_setup)

import config
from src.generate.attack_spec import BASE_SPECS, FAMILY_CONSTRAINTS, validate_spec

page_setup("Generate", "⚗️")
st.markdown('<div class="kicker">Pillar 2 · Generate</div>', unsafe_allow_html=True)
st.title("⚗️ Constrained Attack Simulator")
st.markdown(
    "The red team writes a **specification**, not transactions — which behavioural dial to "
    "move and which detector signal that defeats.\n\n"
    "**Demo mode uses the deterministic committed specifications**, so the prototype runs "
    "identically every time with no API key and no network. **The optional GenAI red team can "
    "generate that specification instead.** Either way it passes through the same "
    "payment-domain constraint layer, which clamps or refuses anything that could not happen "
    "on a real rail, and the same deterministic simulator turns what survives into "
    "transactions. That split is what makes the attack generation both creative and "
    "reproducible."
)
llm_status_badge()

mode = mode_selector()

st.code("""Threat intelligence  /  measured model weakness
                 |
   +-------------+-------------+
   |                           |
Weakness-driven heuristic   GenAI red-team agent    <- optional, needs an API key
   (DEMO MODE default)      (spec_source: "llm")
   (spec_source:"heuristic")
   |                           |
   +-------------+-------------+
                 |
        AttackSpec (structured)       amount · velocity · device · geo · merchant · timing
                 |
        validate_spec()               payment-domain constraints: clamp or reject
                 |
        Constrained simulator         deterministic, seeded, reproducible
                 |
        Synthetic payment stream""", language=None)
st.caption("Every specification behind the committed numbers carries "
           "`spec_source: \"heuristic\"` — the deterministic path. Run "
           "`python -m src.generate.demo_specs` with an API key set to see the GenAI path "
           "produce specifications through the same constraint layer.")

tab_spec, tab_data, tab_fidelity, tab_text = st.tabs(
    ["🧬 Attack specifications", "📊 The portfolio", "🔬 Fidelity diagnostics",
     "✉️ GenAI content layer"])

# --------------------------------------------------------------------------- #
# 1. Attack specifications + constraint layer
# --------------------------------------------------------------------------- #
with tab_spec:
    st.subheader("Generation-0 specifications")
    st.caption("Every simulated family starts from an explicit specification. These are the "
               "lineage roots the closed loop evolves away from.")
    rows = []
    for fam, s in BASE_SPECS.items():
        rows.append({"family": fam, "amount": s.amount_profile, "velocity": s.velocity_profile,
                     "device": s.device_behavior, "geography": s.geo_behavior,
                     "merchant": s.merchant_behavior, "timing": s.timing_profile,
                     "strategy": s.strategy})
    st.dataframe(pd.DataFrame(rows), width=STRETCH, hide_index=True, height=420)

    st.subheader("The constraint layer, live")
    st.caption("A specification that contradicts payment reality is normalized or refused "
               "before anything is generated. Try it: ask for an authorized push payment "
               "scam that runs from an attacker's device — a contradiction in terms, because "
               "the genuine customer is the one authenticating.")
    c1, c2 = st.columns([1, 1])
    with c1:
        fam = st.selectbox("Attack family", sorted(BASE_SPECS), index=sorted(BASE_SPECS).index(
            "scam_transfer") if "scam_transfer" in BASE_SPECS else 0)
        dev = st.selectbox("device_behavior",
                           config.ATTACK_SPEC_BOUNDS["device_behavior"], index=2)
        geo = st.selectbox("geo_behavior", config.ATTACK_SPEC_BOUNDS["geo_behavior"], index=4)
        amt = st.selectbox("amount_profile", config.ATTACK_SPEC_BOUNDS["amount_profile"], index=4)
        inten = st.slider("intensity", 0.0, 2.0, 1.6, 0.05,
                          help="Outside the permitted 0.15–1.0 band on purpose, to show clamping.")
    proposed = {"device_behavior": dev, "geo_behavior": geo, "amount_profile": amt,
                "intensity": inten, "source": "manual"}
    spec, report = validate_spec(proposed, fam)
    with c2:
        st.markdown("**What the simulator will actually execute**")
        st.json(spec.to_dict(), expanded=False)
        if report.corrections:
            st.warning(f"{len(report.corrections)} correction(s) applied by the constraint layer.")
            for corr in report.corrections:
                st.markdown(f"- `{corr['field']}`: **{corr['from']}** → **{corr['to']}** "
                            f"— {corr['reason']}")
        else:
            st.success("Specification accepted unchanged — every value is legal for this family.")
    reqs = FAMILY_CONSTRAINTS.get(fam)
    if reqs:
        st.caption("Payment-domain requirements for this family: " + "; ".join(
            f"`{k}` must be one of {sorted(v)}" for k, v in reqs.items()))

    summary = load_summary()
    if summary and summary.get("attack_specs"):
        with st.expander("Specifications used to build the committed dataset"):
            st.json(summary["attack_specs"], expanded=False)

# --------------------------------------------------------------------------- #
# 2. The portfolio
# --------------------------------------------------------------------------- #
with tab_data:
    if is_demo(mode):
        df = load_dataset_cached()
        if df is None:
            st.error("No committed dataset. Run `python -m src.generate.simulate`.")
            st.stop()
        st.caption("🟢 Showing the committed seeded portfolio.")
    else:
        with st.sidebar:
            st.markdown("### Regenerate (Live)")
            n_tx = st.select_slider("Transactions", [5000, 10000, 20000, 40000, 80000],
                                    value=20000)
            fraud_rate = st.slider("Fraud rate", 0.005, 0.08, config.DEFAULT_FRAUD_RATE, 0.005)
            seed = st.number_input("Seed", value=config.GLOBAL_SEED, step=1)
            run = st.button("▶ Generate fresh portfolio", type="primary")
        if run or "gen_df" not in st.session_state:
            from src.generate.simulate import simulate
            with st.spinner("Simulating…"):
                gdf, _ = simulate(n_transactions=n_tx, fraud_rate=fraud_rate, seed=int(seed))
            st.session_state["gen_df"] = gdf
        df = st.session_state["gen_df"]

    legit = df[df.is_fraud == 0]
    fraud = df[df.is_fraud == 1]
    cover = (df[df.actor_role == "fraud_actor_cover"] if "actor_role" in df.columns
             else df.iloc[0:0])

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Transactions", f"{len(df):,}")
    m2.metric("Fraud", f"{int(df.is_fraud.sum()):,}", f"{df.is_fraud.mean()*100:.2f}%")
    m3.metric("Cardholders", f"{df.cardholder_id.nunique():,}")
    m4.metric("Merchants", f"{df.merchant_id.nunique():,}")
    m5.metric("Fraud-actor cover traffic", f"{len(cover):,}", "labelled legitimate")
    st.caption("Cover traffic is the ordinary-looking spend a mule, bust-out account or front "
               "merchant produces before it is used. It is labelled **legitimate**, because at "
               "authorization time it is — which is what stops \"this card has no history\" "
               "from becoming a synonym for fraud.")

    if summary := load_summary():
        arch = summary.get("customer_archetypes", {})
        if arch:
            st.subheader("Customer archetypes")
            adf = pd.DataFrame([{"archetype": k.replace("_", " "), "cardholders": v}
                                for k, v in arch.items()]).sort_values("cardholders")
            fig = px.bar(adf, x="cardholders", y="archetype", orientation="h",
                         color_discrete_sequence=[PALETTE["info"]])
            fig.update_layout(height=260, yaxis_title="", plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=10))
            st.plotly_chart(fig, width=STRETCH)
            st.caption("Amount, frequency, device count, travel, night-time propensity, "
                       "weekday shape and channel mix all follow the archetype. A portfolio "
                       "built from one behaviour would be trivially separable from fraud.")

    st.subheader("Attack composition")
    comp = fraud.attack_type.value_counts().reset_index()
    comp.columns = ["attack", "count"]
    fig = px.bar(comp.sort_values("count"), x="count", y="attack", orientation="h",
                 color="count", color_continuous_scale="Oranges")
    fig.update_layout(height=360, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      yaxis_title="", xaxis_title="", coloraxis_showscale=False)
    st.plotly_chart(fig, width=STRETCH)

    st.subheader("Sample transactions")
    show_fraud = st.toggle("Show fraud only", value=False)
    src = fraud if show_fraud else df
    sample = src.sample(min(200, len(src)), random_state=42)
    cols = ["timestamp", "card_id", "merchant_id", "mcc", "amount", "channel", "geo_city",
            "distance_from_home_km", "device_id", "otp_verified", "is_new_payee",
            "attack_type"]
    st.dataframe(sample[cols].sort_values("timestamp"), width=STRETCH, hide_index=True,
                 height=300)

# --------------------------------------------------------------------------- #
# 3. Fidelity diagnostics
# --------------------------------------------------------------------------- #
with tab_fidelity:
    fid = load_fidelity()
    st.subheader("Internal synthetic fidelity diagnostics")
    if not fid:
        st.info("Run `python -m src.generate.fidelity` to generate the report.", icon="ℹ️")
    else:
        st.caption(fid["scope"])
        a, b, c = st.columns(3)
        a.metric("Strongest single feature", f"{fid['max_single_feature_auc']:.3f} AUC",
                 fid["separability"][0]["feature"])
        b.metric("Failing checks", fid["n_fail"], delta_color="inverse")
        c.metric("Warnings", fid["n_warn"], delta_color="off")
        st.caption("If any one field ranked fraud above genuine traffic near-perfectly, the "
                   "detector would be reading the generator rather than the fraud, and every "
                   "downstream number would be meaningless. This is the check for that.")

        st.markdown("**Automated quality checks**")
        for ck in fid["checks"]:
            icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(ck["status"], "•")
            st.markdown(f"{icon} **{ck['check'].replace('_', ' ')}** — {ck['detail']}")

        st.markdown("**Per-feature separability and overlap**")
        sdf = pd.DataFrame(fid["separability"])
        fig = go.Figure()
        fig.add_bar(x=sdf["auc"], y=sdf["feature"], orientation="h", name="ranking power (AUC)",
                    marker_color=PALETTE["primary"])
        fig.add_bar(x=sdf["overlap"], y=sdf["feature"], orientation="h",
                    name="distribution overlap", marker_color=PALETTE["safe"], opacity=0.65)
        fig.update_layout(barmode="group", height=760, yaxis=dict(autorange="reversed"),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          legend=dict(orientation="h"), margin=dict(t=10), yaxis_title="")
        st.plotly_chart(fig, width=STRETCH)
        st.caption("High AUC with low overlap on any single feature would be a shortcut. "
                   "The strongest feature here is distance from home, which is a genuine "
                   "fraud signal rather than an artifact of how rows are produced.")

        port = fid.get("portfolio", {})
        if port:
            with st.expander("Portfolio structure"):
                st.json(port, expanded=False)

# --------------------------------------------------------------------------- #
# 4. GenAI content layer
# --------------------------------------------------------------------------- #
with tab_text:
    st.subheader("Attacker-agent content artifacts")
    st.caption("Synthetic scam messages used to train the text arm. All are clearly marked "
               "synthetic, contain no brand, link, number or procedure, and describe only the "
               "shape of an approach a detector needs to recognize.")
    arts = [a for a in load_artifacts() if a.get("label") == 1]
    if arts:
        cols3 = st.columns(3)
        for i, art in enumerate(arts[:6]):
            with cols3[i % 3]:
                st.markdown(
                    f'<div class="card"><span class="kicker">{art.get("artifact_type","")}</span>'
                    f'<br><b>{art.get("attack_id","")}</b><br><br>'
                    f'{art.get("display_text", art.get("text",""))[:280]}…'
                    f'<br><br><span class="kicker">source: {art.get("source","")}</span></div>',
                    unsafe_allow_html=True)
    from common import load_text_metrics
    tm = load_text_metrics()
    if tm:
        st.markdown("**Text arm — synthetic sanity check, not a detection result**")
        st.warning(
            "This is not evidence of detection efficacy and is never used as such. The corpus "
            "is composed from a fixed slot vocabulary, so the two classes separate on "
            "vocabulary alone — the score measures the corpus, not the detector, and would not "
            "survive contact with real scam messages. **Every detection claim in this project "
            "rests on the transaction model alone.**", icon="⚠️")
        with st.expander("Show the score anyway, with its caveats"):
            t1, t2, t3 = st.columns(3)
            t1.metric("Corpus", f"{tm['n_corpus']}",
                      f"{tm['n_fraud']} scam / {tm['n_benign']} genuine")
            t2.metric("ROC-AUC (5-fold)", f"{tm.get('roc_auc_cv5') or tm['roc_auc']:.3f}",
                      "measures the corpus", delta_color="off")
            t3.metric("Trivially separable?", "yes" if tm.get("trivially_separable") else "no",
                      delta_color="off")
            st.caption(tm.get("honest_reading", ""))
            st.caption(tm.get("caveat", ""))
