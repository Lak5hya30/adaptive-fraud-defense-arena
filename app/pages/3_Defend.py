"""Pillar 3 — DEFEND: detection efficacy, the operating point and what it costs,
tiered decisions with reason codes, per-family recall with its uncertainty, and
the model's own blind spots.

Demo mode (default): reads precomputed scores for the held-out test split.
Live mode: score a freshly simulated batch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (PALETTE, STRETCH, fmt_ci, is_demo, load_blind_spots,
                    load_calibration, load_demo_scores, load_family_recall,
                    load_metrics, load_operational, load_threshold_sweep,
                    mode_selector, page_setup)

import config

page_setup("Defend", "🎯")
st.markdown('<div class="kicker">Pillar 3 · Defend</div>', unsafe_allow_html=True)
st.title("Detection, Decisions & Blind Spots")

metrics = load_metrics()
if metrics is None:
    st.error("No trained model. Run `python -m src.defend.train`.")
    st.stop()
mode = mode_selector()

st.caption("Gradient boosting fused with an isolation forest over "
           "**authorization-time features only** — per-transaction attributes, the card's own "
           "history, and network-level counters. No post-outcome field ever reaches the model. "
           "The operating threshold is tuned on a held-out validation slice against an agreed "
           "false-positive budget, and scores are isotonically calibrated so the number on "
           "screen is a probability rather than an arbitrary blend.")

fpr = metrics["false_positive_rate"]
within = fpr <= config.TARGET_MAX_FPR
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Recall", f"{metrics['recall']*100:.1f}%")
m2.metric("Precision", f"{metrics['precision']:.3f}")
m3.metric("PR-AUC", f"{metrics['pr_auc']:.3f}", f"ROC-AUC {metrics['roc_auc']:.3f}")
m4.metric("False-positive rate", f"{fpr*100:.2f}%",
          f"budget ≤ {config.TARGET_MAX_FPR*100:.0f}%" if within
          else f"OVER budget of {config.TARGET_MAX_FPR*100:.0f}%",
          delta_color="normal" if within else "inverse")
m5.metric("False positives", f"{fpr*1000:.1f}", "per 1,000 genuine payments")
st.caption(
    f"Held-out time-split test set: {metrics.get('test_size', 0):,} transactions, "
    f"{metrics.get('n_fraud_eval', 0)} of them fraudulent at a "
    f"{config.DEFAULT_FRAUD_RATE*100:.1f}% base rate. PR-AUC leads because ROC-AUC flatters "
    "problems this imbalanced. Precision is naturally modest at a ~1% base rate — the answer "
    "is tiered decisioning, not a bigger number.")

tab_ops, tab_family, tab_decisions, tab_blind = st.tabs(
    ["⚖️ Operating point", "🎯 Recall by family", "🚦 Decisions & reason codes",
     "🕳️ Blind spots"])

# Operating point: threshold sweep, calibration, operational volumes
with tab_ops:
    sweep = load_threshold_sweep()
    st.subheader("There is no magic threshold")
    if not sweep:
        st.info("Run `python -m src.defend.diagnostics` to generate the sweep.", icon="ℹ️")
    else:
        pts = pd.DataFrame(sweep["points"])
        fig = go.Figure()
        fig.add_scatter(x=pts["false_positive_rate"], y=pts["recall"], mode="lines",
                        name="recall", line=dict(color=PALETTE["primary"], width=3))
        fig.add_scatter(x=pts["false_positive_rate"], y=pts["precision"], mode="lines",
                        name="precision", line=dict(color=PALETTE["info"], width=2))
        fig.add_vline(x=sweep["fpr_budget"], line_dash="dash", line_color=PALETTE["accent"],
                      annotation_text=f"budget {sweep['fpr_budget']*100:.0f}%")
        fig.update_layout(height=380, xaxis_title="false-positive rate on genuine traffic",
                          yaxis_title="", xaxis_range=[0, 0.12],
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          legend=dict(orientation="h"), margin=dict(t=10))
        st.plotly_chart(fig, width=STRETCH)
        st.caption(sweep["note"] + " Every point on this curve is a different answer to "
                   "'how much genuine friction are we willing to buy detection with'.")

    ops = load_operational()
    if ops and "static_ml" in ops:
        o = ops["static_ml"]
        st.subheader("What that costs to run")
        d = o["action_distribution"]
        sc = o["scenario"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Approved outright", f"{d['approve']*100:.1f}%")
        k2.metric("Sent to step-up", f"{d['step_up']*100:.2f}%",
                  f"{d['genuine_customers_stepped_up']*100:.2f}% of genuine customers")
        k3.metric("Declined", f"{d['decline']*100:.2f}%",
                  f"{d['genuine_customers_declined']*100:.3f}% of genuine customers")
        k4.metric("Fraud sent to friction",
                  f"{o['policy']['fraud_to_friction']*100:.0f}%")
        st.markdown(
            f"Applied to the declared scenario of **{sc['monthly_authorizations']:,} monthly "
            f"authorizations**, that is **{sc['monthly_step_up_challenges']:,} step-up "
            f"challenges a month**, roughly "
            f"**{sc['monthly_review_hours_if_all_manually_reviewed']:,} analyst-hours** if every "
            f"one were reviewed by hand, and an estimated "
            f"**{sc['estimated_genuine_customers_abandoning_after_step_up']:,} genuine customers "
            f"abandoning** at the challenge.")
        st.warning(sc["label"] + " — " + o["disclaimer"], icon="⚠️")

    cal = load_calibration()
    if cal:
        st.subheader("Does the probability mean anything?")
        c1, c2 = st.columns([1, 1.3])
        with c1:
            st.metric("Brier score", f"{cal['brier_calibrated']:.5f}",
                      f"{cal['brier_improvement']:+.5f} vs uncalibrated")
            st.caption(cal["note"])
        with c2:
            rel = pd.DataFrame(cal["reliability_calibrated"])
            if len(rel):
                fig = go.Figure()
                fig.add_scatter(x=[0, rel["predicted"].max()], y=[0, rel["predicted"].max()],
                                mode="lines", name="perfect", line=dict(dash="dot",
                                                                        color=PALETTE["muted"]))
                fig.add_scatter(x=rel["predicted"], y=rel["observed"], mode="markers+lines",
                                name="calibrated", marker=dict(size=9,
                                                               color=PALETTE["primary"]))
                fig.update_layout(height=300, xaxis_title="predicted fraud probability",
                                  yaxis_title="observed fraud rate",
                                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  legend=dict(orientation="h"), margin=dict(t=10))
                st.plotly_chart(fig, width=STRETCH)

# Per-family recall with intervals
with tab_family:
    fam = load_family_recall()
    st.subheader("Recall by attack family")
    if not fam:
        st.info("Run `python -m src.defend.diagnostics` to generate family recall.", icon="ℹ️")
    else:
        st.caption(fam["note"])
        which = st.radio("Detector", list(fam["models"]), horizontal=True)
        block = fam["models"][which]["per_family_recall"]
        rows = []
        for a, d in block.items():
            from src.defend.evaluate import wilson_interval
            ci = wilson_interval(d["recall"], d["n"])
            rows.append({"family": a, "recall": d["recall"], "n": d["n"],
                         "lo": ci[0], "hi": ci[1], "enough": d["sufficient_n"]})
        fdf = pd.DataFrame(rows).sort_values("recall")
        fig = go.Figure()
        fig.add_bar(x=fdf["recall"], y=fdf["family"], orientation="h",
                    marker_color=[PALETTE["safe"] if r >= 0.7 else
                                  PALETTE["accent"] if r >= 0.4 else PALETTE["danger"]
                                  for r in fdf["recall"]],
                    error_x=dict(type="data", symmetric=False,
                                 array=(fdf["hi"] - fdf["recall"]).tolist(),
                                 arrayminus=(fdf["recall"] - fdf["lo"]).tolist(),
                                 color=PALETTE["muted"]),
                    hovertext=[f"n={n}" for n in fdf["n"]])
        fig.update_layout(height=420, xaxis_range=[0, 1.05], yaxis_title="",
                          xaxis_title="recall (95% Wilson interval)",
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          margin=dict(t=10))
        st.plotly_chart(fig, width=STRETCH)
        st.dataframe(fdf[["family", "recall", "n", "lo", "hi", "enough"]],
                     width=STRETCH, hide_index=True)
        st.info(
            "Families whose fraud is **authorized by the genuine customer on their own device** "
            "— first-party dispute abuse, and scams the victim was talked into — sit at the "
            "bottom by construction. There is very little to see at authorization time. The "
            "correct control for those is friction, payee-risk intelligence and "
            "post-transaction recall, not a hard decline, and they are reported here rather "
            "than quietly excluded.", icon="ℹ️")

# Decisions and reason codes
with tab_decisions:
    if is_demo(mode):
        dframe = load_demo_scores()
        if dframe is None:
            st.info("Run `python -m src.defend.diagnostics` to precompute demo scores.",
                    icon="ℹ️")
            st.stop()
        st.caption("🟢 Precomputed scores for the committed held-out test split.")
        y = dframe["is_fraud"].to_numpy()
        scores = dframe["risk_score"].to_numpy()
    else:
        from src.defend.decision_policy import decision_frame
        from src.defend.features import build_xy
        from src.defend.model import DefenseModel
        lc = st.columns([1, 1, 1, 2])
        n = lc[0].select_slider("Batch", [4000, 8000, 15000], value=8000)
        fr = lc[1].slider("Fraud rate", 0.005, 0.05, config.DEFAULT_FRAUD_RATE, 0.005)
        seed = lc[2].number_input("Seed", value=777, step=1)
        # Simulate + score only on an explicit click — never automatically on first
        # render, so entering Live mode never triggers heavy compute unasked.
        if lc[3].button("▶ Score fresh batch", type="primary"):
            from src.generate.simulate import simulate
            with st.spinner("Simulating and scoring…"):
                fresh = simulate(n_transactions=n, fraud_rate=fr, seed=int(seed))[0]
                fresh = fresh.sort_values("timestamp").reset_index(drop=True)
                Xf, yf, _ = build_xy(fresh)
                mdl = DefenseModel.load()
                pf = mdl.risk_probability(Xf)
                st.session_state["def_df"] = decision_frame(
                    fresh, Xf, pf, step_up=mdl.threshold_probability)
        if "def_df" not in st.session_state:
            st.info("Click **▶ Score fresh batch** to simulate and score a fresh batch live. "
                    "(Demo mode shows the committed held-out split with no compute.)", icon="▶️")
            st.stop()
        dframe = st.session_state["def_df"]
        y = dframe["is_fraud"].to_numpy()
        scores = dframe["risk_score"].to_numpy()

    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Where genuine and fraudulent traffic sit")
        fig = go.Figure()
        fig.add_histogram(x=scores[y == 0], name="genuine", nbinsx=50,
                          marker_color=PALETTE["safe"], opacity=0.7)
        fig.add_histogram(x=scores[y == 1], name="fraud", nbinsx=50,
                          marker_color=PALETTE["danger"], opacity=0.7)
        sweep = load_threshold_sweep()
        if sweep:
            dt = sweep.get("decision_thresholds", {})
            for key, lab in (("step_up", "step-up"), ("decline", "decline")):
                if dt.get(key) is not None:
                    fig.add_vline(x=dt[key], line_dash="dash", line_color=PALETTE["accent"],
                                  annotation_text=lab)
        fig.update_layout(barmode="overlay", height=340, plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h"),
                          yaxis_type="log", margin=dict(t=10),
                          xaxis_title="calibrated fraud probability")
        st.plotly_chart(fig, width=STRETCH)
    with right:
        st.subheader("Tiered decisions")
        counts = dframe["action"].value_counts()
        total = max(1, len(dframe))
        legit_mask = y == 0
        st.markdown(
            f'<div class="card">'
            f'<b style="color:{PALETTE["safe"]}">APPROVE</b> — '
            f'{counts.get("APPROVE", 0)/total*100:.1f}% of traffic<br>'
            f'<b style="color:{PALETTE["accent"]}">STEP-UP</b> — '
            f'{counts.get("STEP_UP", 0)/total*100:.2f}% of traffic<br>'
            f'<b style="color:{PALETTE["danger"]}">DECLINE</b> — '
            f'{counts.get("DECLINE", 0)/total*100:.2f}% of traffic'
            f'</div>', unsafe_allow_html=True)
        friction = dframe["action"].isin(["STEP_UP", "DECLINE"]).to_numpy()
        st.metric("Fraud sent to friction", f"{friction[y == 1].mean()*100:.0f}%")
        st.metric("Genuine customers left alone",
                  f"{(~friction[legit_mask]).mean()*100:.2f}%")
        st.caption("Issuers route to a challenge, not a hard decline. That is what turns a "
                   "modest precision into a manageable review queue instead of mass false "
                   "declines — and it is the only realistic control for fraud the customer "
                   "themselves authorized.")

    st.subheader("Decisions with reason codes")
    show = [c for c in ["timestamp", "card_id", "amount", "mcc", "geo_city", "device_id",
                        "attack_type", "risk_score", "action", "reason_codes"]
            if c in dframe.columns]
    t1, t2 = st.tabs(["🚨 Challenged or declined", "⚠️ Genuine customers we inconvenienced"])
    with t1:
        flagged = dframe[dframe.action != "APPROVE"].sort_values("risk_score", ascending=False)
        st.dataframe(flagged[show].head(40), width=STRETCH, hide_index=True,
                     column_config={"risk_score": st.column_config.ProgressColumn(
                         "fraud probability", min_value=0.0, max_value=1.0, format="%.3f")})
    with t2:
        fps = dframe[(dframe.action != "APPROVE") & (dframe.is_fraud == 0)].sort_values(
            "risk_score", ascending=False)
        st.caption(f"{len(fps)} genuine transactions received friction "
                   f"({len(fps)/max(1, int(legit_mask.sum()))*100:.2f}% of genuine traffic). "
                   "Shown deliberately: these are real customers, and the cost of the model "
                   "is measured in them.")
        st.dataframe(fps[show].head(40), width=STRETCH, hide_index=True)

# Blind spots
with tab_blind:
    bs = load_blind_spots()
    st.subheader("Where this defense is still weakest")
    if not bs:
        st.info("Run `python -m src.defend.diagnostics` to generate blind spots.", icon="ℹ️")
    else:
        st.caption(bs["note"])
        for d in bs["hardest"]:
            with st.expander(f"{d['family']} — recall {d['recall']:.2f} "
                             f"(n={d['n']}, {d['n_escaped']} escaped)"):
                if d["relied_signals"]:
                    st.markdown("**What the model leans on for this family**")
                    rdf = pd.DataFrame(d["relied_signals"])
                    fig = px.bar(rdf, x="auc_drop", y="feature", orientation="h",
                                 color_discrete_sequence=[PALETTE["info"]])
                    fig.update_layout(height=200, yaxis_title="", margin=dict(t=10),
                                      xaxis_title="AUC drop when this feature is shuffled",
                                      plot_bgcolor="rgba(0,0,0,0)",
                                      paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, width=STRETCH)
                if d["escape_profile"]:
                    st.markdown("**How the transactions that got through differ from the "
                                "ones that were caught**")
                    st.dataframe(pd.DataFrame(d["escape_profile"]), width=STRETCH,
                                 hide_index=True)
        nxt = bs.get("next_red_team_target")
        if nxt:
            st.success(
                f"**Next red-team target: {nxt['family']}** — {nxt['strategy']}\n\n"
                f"Proposed specification change: `{nxt['proposed_change']}`", icon="🎯")
            st.caption("Chosen from measured weakness, not decided in advance. This is the "
                       "input to the next round of the closed loop.")
