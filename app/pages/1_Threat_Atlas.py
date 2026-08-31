"""Pillar 1 — IDENTIFY: the GenAI Payment Fraud Threat Atlas.

Shows the breadth of the research surface and, in the same view, exactly how much
of it the simulator actually reproduces. The coverage map is the honest part: a
judge should be able to see the gap between what we understand and what we
simulate without having to ask for it.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from common import (PALETTE, STRETCH, coverage_bar, get_taxonomy, mode_selector,
                    observability_pill, page_setup, severity_pill, status_pill)

page_setup("Threat Atlas", "🗺️")
st.markdown('<div class="kicker">Pillar 1 · Identify</div>', unsafe_allow_html=True)
st.title("🗺️ GenAI Payment Fraud Threat Atlas")
mode_selector()

tax = get_taxonomy()
counts = tax.summary_counts()
prov = tax.provenance

st.markdown(
    "A structured catalog of how generative AI is changing payment fraud — across "
    "social engineering, account and identity compromise, payment instruments, "
    "authorized push payment scams, merchant and ecosystem abuse, and attacks aimed "
    "at fraud models themselves. Every entry carries what a **defender** could "
    "observe: the transaction signature, the behavioural signature, how visible it "
    "is at authorization time, and what only shows up after settlement."
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Attacks catalogued", counts["total_attacks"], f'{counts["categories"]} surfaces')
c2.metric("Simulated end-to-end", counts["implemented"], "dedicated injectors")
c3.metric("Configurable today", counts["parameterized"], "via attack-spec dials")
c4.metric("Research only", counts["research_only"] + counts["future"], "not simulated")
c5.metric("Hard at auth time", counts["auth_time_hard"],
          "low or no visibility", delta_color="off")

st.caption(
    f"Deliberately wider than the simulator. {prov.get('authored_entries', '—')} entries were "
    f"drafted across six fraud surfaces, {prov.get('merged_as_duplicates', '—')} merged as "
    f"duplicates, leaving {counts['total_attacks']} distinct attacks. "
    f"{counts['implemented']} have a dedicated transaction-level injector; "
    f"{counts['parameterized']} more are reachable by configuring an existing injector through "
    f"the attack-specification dials; the rest are characterized but **not** simulated, and are "
    "labelled that way everywhere they appear."
)

st.divider()

# Coverage map
st.subheader("Coverage map — research surface vs simulator")
cov = tax.coverage_by_category()
left, right = st.columns([1.15, 1])
with left:
    lines = []
    for cat, d in sorted(cov.items(), key=lambda kv: -kv[1]["total"]):
        simulated = d["IMPLEMENTED"] + d["PARAMETERIZED"]
        lines.append(f"{cat:34s} {coverage_bar(simulated, d['total'])}  "
                     f"{simulated}/{d['total']} simulated")
    st.code("\n".join(lines), language=None)
    st.caption("Filled blocks are attacks the simulator can produce as transactions. "
               "Empty blocks are attacks we have characterized but do not claim to simulate.")
with right:
    rows = []
    for cat, d in cov.items():
        for status in ("IMPLEMENTED", "PARAMETERIZED", "RESEARCH_ONLY", "FUTURE"):
            if d[status]:
                rows.append({"category": cat, "status": status, "count": d[status]})
    fig = px.bar(pd.DataFrame(rows), x="count", y="category", color="status",
                 orientation="h",
                 color_discrete_map={"IMPLEMENTED": PALETTE["safe"],
                                     "PARAMETERIZED": PALETTE["info"],
                                     "RESEARCH_ONLY": PALETTE["muted"],
                                     "FUTURE": PALETTE["accent"]})
    fig.update_layout(height=330, barmode="stack", yaxis_title="", xaxis_title="attacks",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.25), margin=dict(t=10))
    st.plotly_chart(fig, width=STRETCH)

st.divider()

# Filters
st.subheader("Browse the atlas")
f1, f2, f3 = st.columns(3)
f4, f5, f6 = st.columns(3)
sel_cat = f1.multiselect("Fraud surface", tax.categories, default=[])
sel_status = f2.multiselect("Simulator status",
                            ["IMPLEMENTED", "PARAMETERIZED", "RESEARCH_ONLY", "FUTURE"],
                            default=[])
sel_rail = f3.multiselect("Payment rail", tax.all_rails, default=[])
sel_channel = f4.multiselect("Channel", tax.channels, default=[])
sel_role = f5.multiselect("Role of generative AI", tax.genai_roles, default=[])
sel_obs = f6.multiselect("Authorization-time visibility",
                         ["high", "partial", "low", "none"], default=[])

rows = tax.attacks
if sel_cat:
    rows = [a for a in rows if a.category in sel_cat]
if sel_status:
    rows = [a for a in rows if a.simulator_status in sel_status]
if sel_rail:
    rows = [a for a in rows if any(r in sel_rail for r in a.rails)]
if sel_channel:
    rows = [a for a in rows if a.channel in sel_channel]
if sel_role:
    rows = [a for a in rows if a.genai_role in sel_role]
if sel_obs:
    rows = [a for a in rows if a.auth_time_observability in sel_obs]

sort_by = st.radio("Order by", ["Novelty", "Defense difficulty", "Expected impact", "Name"],
                   horizontal=True, index=0)
keys = {"Novelty": lambda a: -a.novelty_score,
        "Defense difficulty": lambda a: -a.difficulty_rank,
        "Expected impact": lambda a: -a.severity_rank,
        "Name": lambda a: a.name}
rows = sorted(rows, key=keys[sort_by])

st.caption(f"{len(rows)} of {len(tax.attacks)} attacks shown.")

for a in rows:
    header = (f"{a.name}  ·  {a.category}"
              f"{'  ·  ' + a.subcategory if a.subcategory else ''}")
    with st.expander(header):
        st.markdown(
            status_pill(a.simulator_status) + observability_pill(a.auth_time_observability)
            + severity_pill(a.severity)
            + f'<span class="pill" style="background:#33384a">NOVELTY {a.novelty_score:.1f}</span>'
            + f'<span class="pill" style="background:#33384a">{a.channel}</span>'
            + "".join(f'<span class="pill" style="background:#2a2f3d">{r}</span>' for r in a.rails),
            unsafe_allow_html=True)
        if a.attacker_objective:
            st.markdown(f"**What the attacker wants** — {a.attacker_objective}")
        st.markdown(f"**How generative AI changes it** ({a.genai_role.replace('_', ' ')}) — "
                    f"{a.genai_mechanism}")
        if a.transaction_signature:
            st.markdown(f"**In the authorization stream** — {a.transaction_signature}")
        if a.behavioral_signature:
            st.markdown(f"**In account behaviour** — {a.behavioral_signature}")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Signals a detector can watch**")
            for s in a.observable_signals:
                st.markdown(f"- {s}")
        with cc2:
            if a.post_transaction_signals:
                st.markdown("**Only visible after settlement**")
                for s in a.post_transaction_signals:
                    st.markdown(f"- {s}")
        st.markdown("**Kill chain (defender view)** — " + " → ".join(a.kill_chain))
        st.caption(f"Grounding: {a.real_world_grounding}")
        if a.maps_to_injector:
            st.caption(f"Simulator: `{a.maps_to_injector}` injector · "
                       f"status {a.simulator_status}")
        else:
            st.caption(f"Not simulated ({a.simulator_status}). Characterized for coverage; "
                       "no transaction-level claim is made for it anywhere in this project.")

st.divider()
st.caption(tax.honesty_note)
