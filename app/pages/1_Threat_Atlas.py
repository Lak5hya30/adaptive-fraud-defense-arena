"""Pillar 1 — IDENTIFY: the GenAI Payment Fraud Threat Atlas.

Shows the breadth of the research surface and, in the same view, exactly how much
of it the simulator actually reproduces. The coverage map is the honest part: a
judge should be able to see the gap between what we understand and what we
simulate without having to ask for it.
"""
from __future__ import annotations

from html import escape

import pandas as pd
import plotly.express as px
import streamlit as st

from ui import Metric, columns, metric_grid, page_header, plot

from common import (PALETTE, STATUS_LABEL, STRETCH, get_taxonomy, mode_selector,
                    observability_pill, page_setup, pill, severity_pill, status_pill)

page_setup("Threat Atlas", "🗺️")
page_header("Threat Atlas", "Pillar 1 · Identify")
mode_selector()

tax = get_taxonomy()
counts = tax.summary_counts()
prov = tax.provenance

st.markdown(
    "Explore how generative AI changes payment fraud. Find the signals a defender can "
    "observe, understand each attack, and see exactly what the simulator can reproduce."
)

metric_grid([
    Metric("Attacks catalogued", counts["total_attacks"], f'{counts["categories"]} fraud surfaces'),
    Metric("Simulated end-to-end", counts["implemented"], "Dedicated injectors"),
    Metric("Configurable today", counts["parameterized"], "Via attack-specification dials"),
    Metric("Research only", counts["research_only"] + counts["future"], "Not simulated"),
    Metric("Hard at auth time", counts["auth_time_hard"], "Low or no visibility"),
])

provenance_note = (
    f"Deliberately wider than the simulator. {prov.get('authored_entries', '—')} entries were "
    f"drafted across six fraud surfaces, {prov.get('merged_as_duplicates', '—')} merged as "
    f"duplicates, leaving {counts['total_attacks']} distinct attacks. "
    f"{counts['implemented']} have a dedicated transaction-level injector; "
    f"{counts['parameterized']} more are reachable by configuring an existing injector through "
    f"the attack-specification dials; the rest are characterized but **not** simulated, and are "
    "labelled that way everywhere they appear."
)

with st.expander("Simulator coverage · what is and isn’t reproduced"):
    st.caption(provenance_note)
    cov = tax.coverage_by_category()
    left, right = columns([1, 1.3])
    with left:
        for cat, d in sorted(cov.items(), key=lambda kv: -kv[1]["total"]):
            simulated = d["IMPLEMENTED"] + d["PARAMETERIZED"]
            st.markdown(
                f'<div class="coverage-row"><div class="coverage-label">'
                f'<span>{escape(cat)}</span><span>{simulated} / {d["total"]}</span></div>'
                f'<progress value="{simulated}" max="{d["total"]}" '
                f'aria-label="{escape(cat)}: {simulated} of {d["total"]} simulated">'
                f'{simulated} of {d["total"]}</progress></div>', unsafe_allow_html=True)
        st.caption("Simulated includes dedicated injectors and configurable variants. "
                   "The remaining entries are characterized, but not simulated.")
    with right:
        coverage_rows = []
        for cat, d in cov.items():
            for status in ("IMPLEMENTED", "PARAMETERIZED", "RESEARCH_ONLY", "FUTURE"):
                if d[status]:
                    coverage_rows.append({"category": cat, "status": STATUS_LABEL[status],
                                          "count": d[status]})
        fig = px.bar(pd.DataFrame(coverage_rows), x="count", y="category", color="status",
                     orientation="h",
                     color_discrete_map={"SIMULATED": PALETTE["safe"],
                                         "CONFIGURABLE": PALETTE["info"],
                                         "RESEARCH ONLY": PALETTE["muted"],
                                         "ROADMAP": PALETTE["accent"]})
        fig.update_layout(height=360, barmode="stack", yaxis_title="", xaxis_title="attacks",
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          legend=dict(orientation="h", y=-0.25, title_text=""),
                          margin=dict(t=10, l=0, r=0))
        plot(fig, width=STRETCH)
st.divider()

# Keep the common path visible; specialist filters are one disclosure away.
FILTER_KEYS = ("atlas_category", "atlas_status", "atlas_rail", "atlas_channel",
               "atlas_role", "atlas_visibility")


def clear_filters():
    for key in FILTER_KEYS:
        st.session_state[key] = []
    st.session_state["atlas_search"] = ""
    st.session_state["atlas_sort"] = "Novelty"


st.subheader("Explore attacks")
search_col, reset_col = columns([4, 1], vertical_alignment="bottom")
query = search_col.text_input("Search the atlas", placeholder="Search attacks, signals, or objectives…",
                             key="atlas_search")
reset_col.button("Clear filters", on_click=clear_filters, width=STRETCH)
f1, f2, f3 = columns(3)
sel_cat = f1.multiselect("Fraud surface", tax.categories, key="atlas_category")
sel_status = f2.multiselect("Simulator status",
                            ["IMPLEMENTED", "PARAMETERIZED", "RESEARCH_ONLY", "FUTURE"],
                            format_func=lambda s: STATUS_LABEL[s].capitalize(),
                            key="atlas_status")
sel_rail = f3.multiselect("Payment rail", tax.all_rails, key="atlas_rail")
with st.expander("More filters · channel, AI role, and visibility"):
    f4, f5, f6 = columns(3)
    sel_channel = f4.multiselect("Channel", tax.channels, key="atlas_channel")
    sel_role = f5.multiselect("Role of generative AI", tax.genai_roles, key="atlas_role")
    sel_obs = f6.multiselect("Authorization-time visibility",
                             ["high", "partial", "low", "none"], key="atlas_visibility")

rows = tax.attacks
if query.strip():
    words = query.casefold().split()
    rows = [a for a in rows if all(word in " ".join([
        a.name, a.category, a.attacker_objective, a.genai_mechanism,
        a.transaction_signature, a.behavioral_signature, *a.observable_signals,
    ]).casefold() for word in words)]
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
                   horizontal=True, index=0, key="atlas_sort")
keys = {"Novelty": lambda a: -a.novelty_score,
        "Defense difficulty": lambda a: -a.difficulty_rank,
        "Expected impact": lambda a: -a.severity_rank,
        "Name": lambda a: a.name}
rows = sorted(rows, key=keys[sort_by])

st.caption(f"{len(rows)} of {len(tax.attacks)} attacks shown.")
if not rows:
    st.info("No attacks match these filters. Try fewer search terms or choose Clear filters "
            "to return to the full atlas.", icon=":material/search:")

for a in rows:
    header = (f"{a.name}  ·  {a.category}"
              f"{'  ·  ' + a.subcategory if a.subcategory else ''}")
    with st.expander(header):
        st.markdown(
            status_pill(a.simulator_status) + observability_pill(a.auth_time_observability)
            + severity_pill(a.severity)
            + pill(f"NOVELTY {a.novelty_score:.1f}", PALETTE["muted"])
            + pill(a.channel, PALETTE["muted"])
            + "".join(pill(r, PALETTE["muted"]) for r in a.rails),
            unsafe_allow_html=True)
        if a.attacker_objective:
            st.markdown(f"**What the attacker wants** — {a.attacker_objective}")
        st.markdown(f"**How generative AI changes it** ({a.genai_role.replace('_', ' ')}) — "
                    f"{a.genai_mechanism}")
        if a.transaction_signature:
            st.markdown(f"**In the authorization stream** — {a.transaction_signature}")
        if a.behavioral_signature:
            st.markdown(f"**In account behaviour** — {a.behavioral_signature}")
        cc1, cc2 = columns(2)
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
