"""AI Defense Lab — home page.

A judge has about ten seconds here. One sentence of value proposition, the loop,
three pillars, one proof point, and an honest disclaimer. Everything else lives
one click away.

Run from the project root:  streamlit run app/Home.py
"""
from __future__ import annotations

import streamlit as st

from ui import Metric, card, columns, metric_grid, page_header

from common import (PALETTE, STRETCH, get_taxonomy, llm_status_badge, load_loao,
                    load_loop_history, load_metrics, load_summary, page_setup)

import config

page_setup("Overview", ":material/shield:")

page_header("A learning defense for payment fraud.", "Mastercard Innovation Challenge 2026 · GFF Mumbai")
st.markdown(
    "#### A closed-loop GenAI red team / blue team for payment security.\n"
    "Traditional fraud systems learn from fraud that has already happened. This lab actively "
    "searches for what the current defense **does not know**, generates realistic adversarial "
    "payment behaviour, stress-tests the model against it, and turns every discovered weakness "
    "into training data for the next defense."
)

with st.container(border=True, key="demo-entry"):
    jc1, jc2 = columns([3, 1])
    with jc1:
        st.markdown("#### Start with the 2-Minute Judge Demo")
        st.markdown("One attack, end to end — a scam evades the detector, the system learns it, "
                    "and genuine customers stay protected. Plain language, every number from a "
                    "committed artifact, no API key required.")
    with jc2:
        st.page_link("pages/0_Judge_Demo.py", label="Run the demo", icon=":material/play_circle:")

st.markdown(
    '<div class="workflow"><b>Discover</b><span class="material-symbols-rounded" aria-hidden="true">arrow_forward</span>'
    '<b>Simulate</b><span class="material-symbols-rounded" aria-hidden="true">arrow_forward</span><b>Attack</b>'
    '<span class="material-symbols-rounded" aria-hidden="true">arrow_forward</span><b>Detect</b><span class="material-symbols-rounded" aria-hidden="true">arrow_forward</span>'
    '<b>Adapt</b></div>',
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
        f'</span><br><span style="color:#596579;font-size:.82rem">'
        f'Measured on {hero["n_test"]} held-out synthetic transactions{ci_txt}. This is '
        f'unseen → learned adaptation, not zero-shot detection.</span></div>',
        unsafe_allow_html=True)
elif underpowered:
    st.info("No attack family currently clears the sample-size floor needed to headline a "
            "result, so no proof point is shown. The per-family figures are on the Benchmarks "
            "page with their sample sizes and confidence intervals.", icon=":material/info:")

# --- headline metrics -------------------------------------------------------
metric_grid([
    Metric("Attacks catalogued", counts["total_attacks"], f'{counts["implemented"]} simulated end-to-end'),
    Metric("Transactions simulated", f'{summary["n_transactions"]:,}' if summary else "—",
           f'{summary["fraud_rate"]*100:.1f}% fraud' if summary else ""),
    Metric("Recall", f'{metrics["recall"]*100:.1f}%' if metrics else "—",
           f'PR-AUC {metrics["pr_auc"]:.3f}' if metrics else ""),
    Metric("False positives / 1,000", f'{metrics["false_positive_rate"]*1000:.1f}' if metrics else "—",
           "Per 1,000 genuine payments"),
    Metric("Red-team rounds", loop["rounds"] if loop else "—", "Weakness-driven"),
])

st.divider()

# --- the loop ---------------------------------------------------------------
left, right = columns([1.1, 1])
with left:
    st.subheader("The closed loop")
    st.graphviz_chart(
        f"""
        digraph G {{
          rankdir=TB; bgcolor="transparent"; nodesep=0.28; ranksep=0.32;
          node [style="filled,rounded", shape=box, fontname="Helvetica",
                fontsize=10, color="#dce3ec", fontcolor="#ffffff"];
          edge [color="#596579", fontname="Helvetica", fontsize=8, fontcolor="#596579"];
          atlas   [label="Threat Atlas\\n{counts['total_attacks']} attacks catalogued",
                   fillcolor="#7c3aed"];
          spec    [label="Red-team agent\\nstructured attack specification",
                   fillcolor="#A44D16"];
          sim     [label="Constrained simulator\\nsynthetic payment stream",
                   fillcolor="#976000"];
          defend  [label="Defense model\\nauthorization-time features",
                   fillcolor="#007D91"];
          escaped [label="Escaped\\ntransactions", fillcolor="#BB3444"];
          weak    [label="Weakness analysis\\nwhich signal did it lean on?",
                   fillcolor="#BB3444"];
          replay  [label="Adversarial replay\\n+ retrain", fillcolor="#16734D"];
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
        - **:material/travel_explore: Threat Atlas** — the research surface, and honestly how much of it the
          simulator reproduces.
        - **Attack Simulator** — structured attack specifications, the payment-domain constraint
          layer, and the fidelity diagnostics that say whether the synthetic data is worth
          training on.
        - **Detection & Decisions** — the operating point and what it costs, recall by family with its
          uncertainty, tiered decisions with reason codes, and the model's own blind spots.
        - **:material/sync: Closed Loop** — weakness-driven attack evolution, attack lineage, and the
          governance gate that can refuse a model.
        - **Unseen Attack Demo** — the 90-second story: unseen → learned.
        - **:material/bar_chart: Benchmarks** — rules vs static vs adaptive, compared at a matched
          false-positive budget.
        - **:material/account_tree: Deployment** — what would run online, what stays offline, and what this
          prototype does not do.
        """
    )
    st.info("Demo mode is the default: every page renders from committed artifacts. No API "
            "key, no network, no training on stage.", icon=":material/info:")

st.divider()

# --- pillars ----------------------------------------------------------------
p1, p2, p3 = columns(3)
with p1:
    card(f"{counts['total_attacks']} attacks across {counts['categories']} fraud surfaces",
         "Map how generative AI changes payment fraud across social engineering, identity, "
         "payment instruments, scams, merchant abuse, and attacks on fraud models.",
         eyebrow="Identify", caption=f"{counts['implemented']} simulated end-to-end; "
         f"{counts['parameterized']} configurable. The rest are labelled research-only.")
with p2:
    n_cover = summary.get("n_cover_transactions") if summary else None
    card("Creativity, constrained", "The red team proposes an attack specification. "
         "Payment-domain constraints clamp or refuse impossible behavior before a seeded "
         "simulator executes it. Fraud actors build ordinary-looking history first.",
         eyebrow="Generate", caption=f"{n_cover:,} cover transactions." if n_cover else "")
with p3:
    card('PR-AUC ' + format(metrics['pr_auc'], '.3f') if metrics else 'Detector',
         "Gradient boosting and an isolation forest use authorization-time features, "
         "including network-level counters. Decisions carry calibrated scores and reason codes.",
         eyebrow="Defend", caption="Approve, step up, or decline — with published blind spots.")

st.divider()
st.caption(
    "**All data is synthetic.** Every figure is a simulation result produced by "
    "`python -m src.pipeline` from a single seed. No real cardholder data is used anywhere in "
    "this system, no result has been validated against real payment data, and nothing here has "
    "been reviewed or validated by Mastercard. Built for the Mastercard Innovation Challenge "
    "2026."
)
