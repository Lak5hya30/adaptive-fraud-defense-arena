"""Pillar 4 — CLOSED LOOP: weakness-driven attack evolution, cumulative
adversarial replay, attack lineage, and the governance gate that decides whether
an adapted model is allowed anywhere near an authorization path.

Every label here is derived from measured metrics. Nothing is hardcoded, and a
round that failed its gates says so.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (PALETTE, STRETCH, is_demo, llm_status_badge, load_lineage,
                    load_loop_history, load_registry, mode_selector, page_setup)

import config

page_setup("Closed Loop", "🔄")
st.markdown('<div class="kicker">Pillar 4 · Closed Loop</div>', unsafe_allow_html=True)
st.title("Red Team ⇄ Blue Team")
st.markdown(
    "Each round measures **which signals the current defense actually depends on** for a "
    "family, aims the next attack generation at removing exactly that signal, folds what "
    "escapes into a cumulative replay buffer, retrains, and then puts the result through "
    "promotion gates. The red team learns from the blue team's blind spots — and a candidate "
    "that improves detection while breaking something else does not ship."
)
llm_status_badge()
st.caption("Every specification on this page carries `spec_source: \"heuristic\"` — Demo mode "
           "uses the deterministic path so these numbers reproduce from the seed. The optional "
           "GenAI red team produces the same structure, stamped `spec_source: \"llm\"`, through "
           "the same constraint layer.")

mode = mode_selector()
STATUS_STYLE = {
    "adapted": ("#30A46C", "✅ Defense adapted"),
    "partial": ("#F79E1B", "◐ Partial recovery"),
    "residual_frontier": ("#E5484D", "⚠ Residual frontier"),
    "n/a": ("#8B93A7", "—"),
}

if mode == "LIVE":
    st.sidebar.markdown("### Run the loop")
    rounds = st.sidebar.slider("Rounds", 2, 4, 3)
    confirm = st.sidebar.checkbox("I understand this retrains models (several minutes)")
    if st.sidebar.button("▶ Run closed loop", type="primary", disabled=not confirm):
        from src.loop.redteam_loop import run_loop
        with st.spinner("Running the loop (measure → propose → constrain → simulate → "
                        "retrain → govern)…"):
            run_loop(rounds=rounds, n_base=24000, n_round=12000, n_eval=14000)
        st.cache_data.clear()

loop = load_loop_history()
if not loop:
    st.info("Run `python -m src.loop.redteam_loop` to generate the loop history.", icon="ℹ️")
    st.stop()

sel = loop.get("focus_selection", {})
promo = loop.get("promotion", {})

attacked = loop.get("families_attacked", loop["focus"])
retired = loop.get("retired_frontiers", [])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rounds", loop["rounds"])
c2.metric("Families attacked", len(attacked), ", ".join(attacked))
c3.metric("Control families", len(loop.get("guard_families", [])),
          "never attacked")
c4.metric("Candidates promoted", len(promo.get("promoted_rounds", [])),
          f"of {loop['rounds']}", delta_color="off")

if retired:
    st.warning(
        "**The red team reallocated mid-run.** " + " ".join(
            f"After round {r['round']} it retired **{r['family']}** "
            f"(recall {(r['final_recall'] or 0)*100:.0f}%)"
            + (f" and moved to **{r['replaced_by']}**. " if r["replaced_by"] else ". ")
            for r in retired)
        + "A family that stays a residual frontier under repeated attack is not going to "
          "yield to more of the same — it sits on the legitimate behavioural centroid, or the "
          "genuine customer authorized it. Continuing to hammer it burns replay capacity on "
          "examples the model cannot separate, and drags down families it could. The frontier "
          "is reported, not hidden.", icon="🔀")

if sel.get("initial_ranking"):
    with st.expander("How the targets were chosen — measured, not decided in advance"):
        st.caption(sel.get("why", ""))
        rank = pd.DataFrame(sel["initial_ranking"])
        fig = px.bar(rank.sort_values("recall"), x="recall", y="family", orientation="h",
                     range_x=[0, 1], color="recall", color_continuous_scale="RdYlGn",
                     hover_data=["n"])
        fig.update_layout(height=340, yaxis_title="", coloraxis_showscale=False,
                          margin=dict(t=10), plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width=STRETCH)
        st.caption("Excluded from targeting as authorization-time invisible: "
                   + ", ".join(sel.get("excluded_as_auth_invisible", [])) +
                   " — their limit is observability, not the decision boundary.")

tab_rounds, tab_lineage, tab_gov, tab_buffer = st.tabs(
    ["🔁 Rounds", "🧬 Attack lineage", "🛡️ Promotion gates", "🗃️ Replay buffer"])

with tab_rounds:
    for h in loop["history"]:
        st.subheader(f"Round {h['round']}")
        cols = st.columns(len(h["families"]))
        for col, (fam, d) in zip(cols, h["families"].items()):
            color, label = STATUS_STYLE.get(d["status"], STATUS_STYLE["n/a"])
            stale = d["stale_recall"]
            adapted = d["adapted_recall"]
            with col:
                st.markdown(
                    f'<div class="card"><span class="kicker">{fam}</span><br>'
                    f'<b style="color:{color}">{label}</b><br><br>'
                    f'stale defense <b>{(stale or 0)*100:.0f}%</b> → '
                    f'adapted <b>{(adapted or 0)*100:.0f}%</b><br>'
                    f'<span style="color:#8B93A7;font-size:.8rem">n={d["n"]} · '
                    f'targets: {d.get("targets_signal") or "—"}</span></div>',
                    unsafe_allow_html=True)
                st.caption(d.get("strategy", ""))
                corr = d.get("constraint_layer", {}).get("corrections", [])
                if corr:
                    st.caption(f"⚙️ constraint layer corrected {len(corr)} value(s): " +
                               ", ".join(f"{c['field']} {c['from']}→{c['to']}" for c in corr))

        li = h["legit_impact"]
        gr = h.get("guard_families_on_base", {})
        cc = h.get("champion_challenger", {})
        g1, g2, g3 = st.columns(3)
        g1.metric("Legitimate false positives",
                  f"{li['adapted_fpr']*100:.2f}%",
                  f"{li['fpr_regression']*100:+.2f} pts vs the base defense",
                  delta_color="inverse")
        g2.metric("Overall PR-AUC", f"{h['adapted_overall']['pr_auc']:.3f}",
                  f"stale {h['stale_overall']['pr_auc']:.3f}")
        g3.metric("Promotion decision", cc.get("decision", "—"),
                  cc.get("summary", "")[:60], delta_color="off")
        if gr:
            kept = ", ".join(f"{k} {v['recall']:.2f} (n={v['n']})" if v["recall"] is not None
                             else f"{k} n/a" for k, v in gr.items())
            st.caption(f"Families never attacked, re-measured after retraining: {kept}. "
                       "This is the catastrophic-forgetting check.")
        prior = h.get("prior_generation_recall", {})
        if prior:
            st.caption("Recall on earlier attack generations after this retrain: "
                       + "; ".join(f"gen {k}: " + ", ".join(f"{fam} {v:.2f}" for fam, v
                                                            in d.items() if v is not None)
                                   for k, d in prior.items()))
        st.divider()

with tab_lineage:
    lin = load_lineage()
    st.subheader("How each attack evolved, and what it cost the defense")
    if not lin:
        st.info("Run `python -m src.loop.redteam_loop` to generate the lineage.", icon="ℹ️")
    else:
        st.caption(lin.get("note", ""))
        for fam, nodes in lin["lineage"].items():
            st.markdown(f"### {fam}")
            spec0 = nodes[0]["spec"] if nodes else {}
            chain = []
            base_dials = {k: v for k, v in spec0.items()
                          if k in ("amount_profile", "velocity_profile", "device_behavior",
                                   "geo_behavior", "merchant_behavior", "timing_profile")}
            for n in nodes:
                changed = n.get("changes", {})
                chain.append({
                    "generation": n["generation"],
                    "changed": ", ".join(f"{k}: {v['from']} → {v['to']}"
                                         for k, v in changed.items()) or "—",
                    "targets": n.get("targets_signal") or "—",
                    "spec source": n.get("spec_source", ""),
                    "distance from gen 0": n.get("n_changed_dials", 0),
                    "stale recall": n.get("stale_recall"),
                    "adapted recall": n.get("adapted_recall"),
                })
            st.dataframe(pd.DataFrame(chain), width=STRETCH, hide_index=True)
            rec = [c for c in chain if c["stale recall"] is not None]
            if rec:
                fig = go.Figure()
                fig.add_scatter(x=[c["generation"] for c in rec],
                                y=[c["stale recall"] for c in rec], mode="lines+markers",
                                name="stale defense", line=dict(color=PALETTE["danger"]))
                fig.add_scatter(x=[c["generation"] for c in rec],
                                y=[c["adapted recall"] for c in rec], mode="lines+markers",
                                name="after replay", line=dict(color=PALETTE["safe"]))
                fig.update_layout(height=260, yaxis_range=[0, 1.05], xaxis_title="generation",
                                  yaxis_title="recall", plot_bgcolor="rgba(0,0,0,0)",
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  legend=dict(orientation="h"), margin=dict(t=10))
                st.plotly_chart(fig, width=STRETCH)
            with st.expander(f"Evidence behind each mutation — {fam}"):
                for n in nodes:
                    drv = n.get("driven_by", {})
                    st.markdown(
                        f"**Generation {n['generation']}** — recall before "
                        f"{drv.get('measured_recall_before')}, "
                        f"{drv.get('n_escaped_before')} transactions escaped. "
                        f"Strongest signal the defense relied on: "
                        f"`{drv.get('top_relied_signal')}`.")
                    st.caption(n.get("strategy", ""))

with tab_gov:
    reg = load_registry()
    st.subheader("Champion / challenger")
    st.markdown(
        "An adaptive defense that deploys itself is not deployable. Every model the loop "
        "produces is a **challenger**, measured against the model in force, and it ships only "
        "if it clears every gate."
    )
    if not reg:
        st.info("Run `python -m src.loop.redteam_loop` to generate the model registry.",
                icon="ℹ️")
    else:
        st.caption(reg.get("note", ""))
        st.markdown("**Gates**")
        st.json(reg.get("gates", {}), expanded=False)
        for e in reg["entries"]:
            acc = e.get("acceptance", {})
            decision = acc.get("decision", "—")
            icon = "✅" if decision == "PROMOTE" else ("⛔" if decision == "REJECT" else "•")
            with st.expander(f"{icon} {e['model_version']} · {e['stage']} · {decision}"):
                st.caption(acc.get("summary", ""))
                m = e.get("metrics", {})
                cols = st.columns(len(m) or 1)
                for col, (k, v) in zip(cols, m.items()):
                    col.metric(k.replace("_", " "),
                               f"{v:.3f}" if isinstance(v, (int, float)) and v is not None
                               else "—")
                gates = acc.get("gates", [])
                if gates:
                    gdf = pd.DataFrame([{
                        "gate": g["gate"].replace("_", " "),
                        "observed": g["observed"], "required": g["required"],
                        "passed": g["passed"], "detail": g["detail"]} for g in gates])
                    st.dataframe(gdf, width=STRETCH, hide_index=True)
                comp = e.get("replay_composition", {})
                if comp:
                    st.caption(f"Trained with a replay buffer of {comp.get('total_rows', 0)} "
                               f"rows ({comp.get('fraud_rows', 0)} adversarial).")
        if not promo.get("promoted_rounds"):
            st.warning(
                "No candidate cleared every gate in this run. That is the governance layer "
                "doing its job, and it is reported rather than hidden: the adapted model "
                "improved detection on the evolved attack but breached another constraint, so "
                "it stays out of the authorization path. " + promo.get("note", ""), icon="⛔")
        else:
            st.success(f"Promoted in round(s): {promo['promoted_rounds']}. "
                       + promo.get("note", ""), icon="✅")

with tab_buffer:
    st.subheader("Cumulative replay buffer")
    st.caption("Bounded and stratified: every attack generation keeps a share, so learning the "
               "newest variant does not mean forgetting the first one. The buffer also carries "
               "a **rehearsal** sample of every family the red team did *not* attack — without "
               "it, retraining pulls the boundary toward the newest, hardest attacks and "
               "quietly degrades families nobody touched, which is exactly what the "
               "forgetting gate refuses to promote.")
    last_comp = loop["history"][-1].get("replay_composition", {})
    if last_comp.get("evolved_attack_rows") is not None:
        b1, b2, b3 = st.columns(3)
        b1.metric("Evolved attack rows", f"{last_comp['evolved_attack_rows']:,}")
        b2.metric("Rehearsal rows", f"{last_comp.get('rehearsal_rows', 0):,}",
                  "families never attacked")
        b3.metric("Legitimate context rows", f"{last_comp.get('legit_context_rows', 0):,}")
    rows = []
    for h in loop["history"]:
        comp = h.get("replay_composition", {})
        for fam, n in comp.get("by_family", {}).items():
            rows.append({"round": h["round"], "family": fam, "rows": n})
    if rows:
        bdf = pd.DataFrame(rows)
        fig = px.bar(bdf, x="round", y="rows", color="family", barmode="stack")
        fig.update_layout(height=340, plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=10),
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width=STRETCH)
        last = loop["history"][-1].get("replay_composition", {})
        st.json(last, expanded=False)
    cfg = loop.get("config", {})
    st.caption(f"Buffer cap {cfg.get('replay_buffer_max')} rows; replayed examples carry a "
               f"training weight of {cfg.get('replay_oversample')}×; a family counts as "
               f"'adapted' only above {cfg.get('residual_recall_ceiling')} recall with a gain "
               f"of at least {cfg.get('recovery_delta')}.")
