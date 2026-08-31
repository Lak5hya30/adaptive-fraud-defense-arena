"""AI Defense Lab — home page.

A judge has about ten seconds here. One sentence of value proposition, the loop,
three pillars, one proof point, and an honest disclaimer. Everything else lives
one click away.

Run from the project root:  streamlit run app/Home.py
"""
from __future__ import annotations

import streamlit as st

from common import (PALETTE, STRETCH, get_taxonomy, llm_status_badge, load_loao,
                    load_loop_history, load_metrics, load_summary, page_setup)

import config

page_setup("Overview", "🛡️")

st.markdown('<div class="kicker">Mastercard Innovation Challenge 2026 · GFF Mumbai</div>',
            unsafe_allow_html=True)
st.title("🛡️ AI Defense Lab for Adaptive Payment Fraud")
st.markdown(
    "#### A closed-loop GenAI red team / blue team for payment security.\n"
    "Traditional fraud systems learn from fraud that has already happened. This lab actively "
    "searches for what the current defense **does not know**, generates realistic adversarial "
    "payment behaviour, stress-tests the model against it, and turns every discovered weakness "
    "into training data for the next defense."
)

with st.container(border=True):
    jc1, jc2 = st.columns([3, 1])
    with jc1:
        st.markdown("#### 🎯 New here? Start with the 2-Minute Judge Demo")
        st.markdown("One attack, end to end — a scam evades the detector, the system learns it, "
                    "and genuine customers stay protected. Plain language, every number from a "
                    "committed artifact, no API key required.")
    with jc2:
        st.page_link("pages/0_Judge_Demo.py", label="Run the 2-Minute Demo", icon="▶️")

st.markdown(
    f'<div style="font-size:1.05rem;letter-spacing:.02em;margin:.4rem 0 1rem 0">'
    f'<b style="color:{PALETTE["primary"]}">DISCOVER</b> → '
    f'<b style="color:{PALETTE["accent"]}">SIMULATE</b> → '
    f'<b style="color:{PALETTE["danger"]}">ATTACK</b> → '
    f'<b style="color:{PALETTE["info"]}">DETECT</b> → '
    f'<b style="color:{PALETTE["safe"]}">ADAPT</b></div>',
    unsafe_allow_html=True)

llm_status_badge()

tax = get_taxonomy()
counts = tax.summary_counts()
summary = load_summary()
metrics = load_metrics()
loop = load_loop_history()
loao = load_loao()

# --- one compact proof point ------------------------------------------------
# Selection rule lives in src/experiments/leave_one_out.py so the landing page,
# the hero demo and the README can never disagree about which family leads.
from src.experiments.leave_one_out import select_hero_family  # noqa: E402

hero_family, hero, underpowered = select_hero_family(loao)

if hero and not underpowered:
    ci = hero.get("recall_after_learning_ci95")
    ci_txt = (f", 95% interval {ci[0]*100:.0f}–{ci[1]*100:.0f}%" if ci else "")
    st.markdown(
        f'<div class="card" style="border-color:{PALETTE["primary"]}">'
        f'<span class="kicker">The proof point</span><br>'
        f'<span style="font-size:1.15rem">Against an attack family it had '
        f'<b>never seen</b> — {hero_family.replace("_", " ")} — the defense caught '
        f'<b style="color:{PALETTE["danger"]}">{hero["recall_unseen"]*100:.0f}%</b>. '
        f'After the lab generated that family and replayed it into training, '
        f'<b style="color:{PALETTE["safe"]}">{hero["recall_after_learning"]*100:.0f}%</b>.'
        f'</span><br><span style="color:#8B93A7;font-size:.82rem">'
        f'Measured on {hero["n_test"]} held-out synthetic transactions{ci_txt}. This is '
        f'unseen → learned adaptation, not zero-shot detection.</span></div>',
        unsafe_allow_html=True)
elif underpowered:
    st.info("No attack family currently clears the sample-size floor needed to headline a "
            "result, so no proof point is shown. The per-family figures are on the Benchmarks "
            "page with their sample sizes and confidence intervals.", icon="ℹ️")

# --- headline metrics -------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Attacks catalogued", counts["total_attacks"],
          f'{counts["implemented"]} simulated end-to-end')
c2.metric("Transactions simulated",
          f'{summary["n_transactions"]:,}' if summary else "—",
          f'{summary["fraud_rate"]*100:.1f}% fraud' if summary else None)
c3.metric("Recall", f'{metrics["recall"]*100:.1f}%' if metrics else "—",
          f'PR-AUC {metrics["pr_auc"]:.3f}' if metrics else None)
c4.metric("False positives",
          f'{metrics["false_positive_rate"]*1000:.1f}' if metrics else "—",
          "per 1,000 genuine payments" if metrics else None)
c5.metric("Red-team rounds", loop["rounds"] if loop else "—",
          "weakness-driven" if loop else None)

st.divider()

# --- the loop ---------------------------------------------------------------
left, right = st.columns([1.1, 1])
with left:
    st.subheader("The closed loop")
    st.graphviz_chart(
        f"""
        digraph G {{
          rankdir=TB; bgcolor="transparent"; nodesep=0.28; ranksep=0.32;
          node [style="filled,rounded", shape=box, fontname="Helvetica",
                fontsize=10, color="#33384a", fontcolor="#e8eaf0"];
          edge [color="#8B93A7", fontname="Helvetica", fontsize=8, fontcolor="#b9c0d0"];
          atlas   [label="Threat Atlas\\n{counts['total_attacks']} attacks catalogued",
                   fillcolor="#7c3aed"];
          spec    [label="Red-team agent\\nstructured attack specification",
                   fillcolor="#EB6C1E"];
          sim     [label="Constrained simulator\\nsynthetic payment stream",
                   fillcolor="#F79E1B", fontcolor="#1a1d26"];
          defend  [label="Defense model\\nauthorization-time features",
                   fillcolor="#3E7BFA"];
          escaped [label="Escaped\\ntransactions", fillcolor="#E5484D"];
          weak    [label="Weakness analysis\\nwhich signal did it lean on?",
                   fillcolor="#E5484D"];
          replay  [label="Adversarial replay\\n+ retrain", fillcolor="#30A46C"];
          gate    [label="Champion / challenger\\npromotion gates", fillcolor="#33384a"];
          atlas -> spec -> sim -> defend;
          defend -> escaped [label=" missed "];
          escaped -> weak -> replay -> gate;
          gate -> defend [label=" promoted "];
          weak -> spec [label=" evolve the attack ", constraint=false, style=dashed];
        }}
        """,
        width="stretch",
    )
with right:
    st.subheader("What each page shows")
    st.markdown(
        """
        - **🗺️ Threat Atlas** — the research surface, and honestly how much of it the
          simulator reproduces.
        - **⚗️ Generate** — structured attack specifications, the payment-domain constraint
          layer, and the fidelity diagnostics that say whether the synthetic data is worth
          training on.
        - **🎯 Defend** — the operating point and what it costs, recall by family with its
          uncertainty, tiered decisions with reason codes, and the model's own blind spots.
        - **🔄 Closed Loop** — weakness-driven attack evolution, attack lineage, and the
          governance gate that can refuse a model.
        - **🎬 Hero Demo** — the 90-second story: unseen → learned.
        - **📊 Benchmarks** — rules vs static vs adaptive, compared at a matched
          false-positive budget.
        - **🏗️ Deployment** — what would run online, what stays offline, and what this
          prototype does not do.
        """
    )
    st.info("Demo mode is the default: every page renders from committed artifacts. No API "
            "key, no network, no training on stage.", icon="ℹ️")

st.divider()

# --- pillars ----------------------------------------------------------------
p1, p2, p3 = st.columns(3)
with p1:
    st.markdown(f"""<div class="card"><div class="kicker">Identify</div>
    <h4>{counts['total_attacks']} attacks across {counts['categories']} fraud surfaces</h4>
    Map how generative AI changes payment fraud — social engineering, identity compromise,
    payment instruments, authorized push payment scams, merchant abuse, and attacks aimed at
    fraud models themselves. {counts['implemented']} are simulated end-to-end,
    {counts['parameterized']} more are configurable, and the rest are labelled research-only.
    </div>""", unsafe_allow_html=True)
with p2:
    n_cover = summary.get("n_cover_transactions") if summary else None
    st.markdown(f"""<div class="card"><div class="kicker">Generate</div>
    <h4>Creativity, constrained</h4>
    The red team proposes an attack <i>specification</i> — deterministic and committed in Demo
    mode, or generated by the optional GenAI red team. Either way a payment-domain constraint
    layer clamps or refuses anything impossible, and a deterministic simulator executes what
    survives. Fraud actors build ordinary-looking history first
    {f'({n_cover:,} cover transactions)' if n_cover else ''}, so "no history" never becomes a
    synonym for fraud.</div>""", unsafe_allow_html=True)
with p3:
    st.markdown(f"""<div class="card"><div class="kicker">Defend</div>
    <h4>{'PR-AUC ' + format(metrics['pr_auc'], '.3f') if metrics else 'Detector'}</h4>
    Gradient boosting fused with an isolation forest over authorization-time features only,
    including network-level counters an issuer cannot compute alone. Calibrated scores, tiered
    approve / step-up / decline with reason codes, and a published list of its own blind spots.
    </div>""", unsafe_allow_html=True)

st.divider()
st.caption(
    "**All data is synthetic.** Every figure is a simulation result produced by "
    "`python -m src.pipeline` from a single seed. No real cardholder data is used anywhere in "
    "this system, no result has been validated against real payment data, and nothing here has "
    "been reviewed or validated by Mastercard. Built for the Mastercard Innovation Challenge "
    "2026."
)
