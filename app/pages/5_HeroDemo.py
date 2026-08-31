"""HERO DEMO — the 90-second story, one beat at a time.

Beat 1  A static model can only learn from attacks its training data contains.
Beat 2  Here is a fraud family it has never seen.
Beat 3  The stale defense misses it. This is the blind spot.
Beat 4  The lab discovers it, simulates variants, replays them, retrains.
Beat 5  The defense now catches it.
Beat 6  And the guardrails still hold — or, honestly, where they do not.

Every number on this page is read from a committed artifact. Nothing is
recomputed, nothing is hardcoded, and nothing requires a network connection.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import Metric, columns, metric_grid, page_header, plot

from common import (PALETTE, STRETCH, load_hero, load_loao, load_loop_history,
                    mode_selector, page_setup)

import config

page_setup("Hero Demo", "🎬")
page_header("What happens when an unseen attack arrives?", "The 90-second story")
mode_selector()

loao = load_loao()
loop = load_loop_history()
hero = load_hero()

if not loao:
    st.info("Run `python -m src.experiments.leave_one_out` to generate the hero result.",
            icon="ℹ️")
    st.stop()

# Chosen from measured evidence: the largest unseen→learned gain among families
# with enough held-out examples to support the claim. One shared rule, so this
# page, the landing page and the README always lead with the same family.
from src.experiments.leave_one_out import select_hero_family

hero_family, H, underpowered = select_hero_family(loao)
if H is None:
    st.info("Leave-one-attack-family-out has not produced a usable result yet.", icon="ℹ️")
    st.stop()
if underpowered:
    st.warning(
        f"No family clears the {config.FAMILY_EVAL['min_n_to_report']}-transaction floor this "
        f"run, so **{hero_family}** is shown for illustration only and its number must not be "
        "quoted as a headline — the interval is far too wide to carry one.", icon="⚠️")

beat = st.radio("Walk the story", [
    "① The problem", "② An unseen attack", "③ The blind spot",
    "④ The lab responds", "⑤ The defense learns", "⑥ The guardrails"],
    horizontal=True, label_visibility="collapsed")

st.divider()

if beat.startswith("①"):
    st.header("A model can only learn from fraud that already happened")
    st.markdown(
        "Supervised fraud detection is trained on labelled history. It is strong on the attack "
        "families that history contains, and it has no representation at all of the ones it "
        "does not. Generative AI has made the gap worse: producing a new, plausible, "
        "well-targeted attack variant now costs almost nothing, so new families arrive faster "
        "than labelled history accumulates."
    )
    st.info("So the question is not 'how accurate is the model on last year's fraud'. It is "
            "**'what does it do the first time it meets something new, and how quickly can it "
            "learn?'** That is what this demo measures.", icon="💡")

elif beat.startswith("②"):
    st.header(f"An attack family the defense has never seen: {hero_family.replace('_', ' ')}")
    st.markdown(
        f"We remove **{hero_family}** entirely from the training data — every example, not a "
        "held-out sample — and train the defense on everything else. The simulator then "
        f"produces {H['n_test']} genuinely unseen {hero_family.replace('_', ' ')} transactions "
        "in held-out traffic."
    )
    from common import get_taxonomy
    tax = get_taxonomy()
    linked = [a for a in tax.attacks if a.maps_to_injector == hero_family]
    if linked:
        a = linked[0]
        st.markdown(f"**{a.name}** — {a.attacker_objective}")
        st.caption(f"In the authorization stream: {a.transaction_signature}")
    st.caption("This is a leave-one-attack-family-out experiment. It measures unseen → learned "
               "adaptation on synthetic data. It is **not** a zero-shot detection claim, and "
               "it is not a production result.")

elif beat.startswith("③"):
    st.header("The blind spot")
    c1, c2 = columns([1, 1.4])
    with c1:
        st.metric(f"Recall on unseen {hero_family}",
                  f"{H['recall_unseen']*100:.0f}%",
                  f"n = {H['n_test']}", delta_color="off")
        if H.get("recall_unseen_ci95"):
            lo, hi = H["recall_unseen_ci95"]
            st.caption(f"95% interval {lo*100:.0f}–{hi*100:.0f}%.")
    with c2:
        st.markdown(
            f"The defense is competent, correctly tuned, and inside its false-positive budget. "
            f"It still misses **{(1-H['recall_unseen'])*100:.0f}%** of this family, because "
            "nothing in its training data looks like it. No amount of tuning fixes that: the "
            "information simply is not there."
        )
    if hero:
        st.markdown("**One concrete transaction the stale defense waved through**")
        txn = hero["transaction"]
        show = {k: txn.get(k) for k in
                ("timestamp", "amount", "mcc", "channel", "geo_city",
                 "distance_from_home_km", "device_id", "otp_verified", "is_new_payee")}
        st.dataframe(pd.DataFrame([show]), width=STRETCH, hide_index=True)
        s, a = hero["stale"], hero["adapted"]
        k1, k2 = columns(2)
        k1.metric("Stale defense", s.get("action", "—"),
                  f"fraud probability {s.get('probability', s['score']):.3f}")
        k2.metric("Adapted defense", a.get("action", "—"),
                  f"fraud probability {a.get('probability', a['score']):.3f}")

elif beat.startswith("④"):
    st.header("The lab responds")
    st.code("""     attack discovered
              |
     variants simulated        from a constrained attack specification
              |
     replay buffer             bounded, stratified across every generation
              |
     defense retrained         candidate, not yet deployed
              |
     promotion gates           recall gain · false positives · no forgetting""",
            language=None)
    st.markdown(
        "Nothing here waits for the attack to appear in labelled production history. The lab "
        "generates it, measures what the current defense does with it, and turns the failure "
        "into training data."
    )
    if hero and hero.get("evolved_spec"):
        st.markdown("**The specification that produced the evolved attack**")
        st.json({k: v for k, v in hero["evolved_spec"].items()
                 if k not in ("derived",)}, expanded=False)

elif beat.startswith("⑤"):
    st.header("The defense learns")
    metric_grid([
        Metric("Before — never seen", f"{H['recall_unseen']*100:.0f}%", f"n = {H['n_test']}"),
        Metric("After — learned", f"{H['recall_after_learning']*100:.0f}%",
               f"+{H['gain']*100:.0f} points", "safe"),
    ])
    if H.get("recall_after_learning_ci95"):
        lo, hi = H["recall_after_learning_ci95"]
        st.caption(f"95% interval {lo*100:.0f}–{hi*100:.0f}% on {H['n_test']} held-out "
                   f"{hero_family} transactions.")

    st.subheader("Every family, unseen versus learned")
    rows = []
    for fam, r in loao["families"].items():
        if r["recall_unseen"] is None:
            continue
        rows.append({"family": fam, "unseen": r["recall_unseen"],
                     "after learning": r["recall_after_learning"], "n": r["n_test"]})
    ldf = pd.DataFrame(rows).sort_values("unseen")
    fig = go.Figure()
    fig.add_bar(x=ldf["unseen"], y=ldf["family"], orientation="h", name="never seen",
                marker_color=PALETTE["danger"])
    fig.add_bar(x=ldf["after learning"], y=ldf["family"], orientation="h",
                name="after learning", marker_color=PALETTE["safe"])
    fig.update_layout(barmode="group", height=440, xaxis_range=[0, 1.02], yaxis_title="",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h"), margin=dict(t=10))
    plot(fig, width=STRETCH)
    st.warning(
        "Read this correctly. `after learning` is what the same pipeline achieves once the "
        "family is in training — an upper bound obtained by handing the model the answer. The "
        "closed loop has to *reach* that bound by generating the family itself, and on the "
        "hardest families it does not get all the way. Some families barely move even when "
        "learned; those are reported as residual frontiers rather than smoothed over.",
        icon="⚠️")

else:
    st.header("The guardrails")
    st.markdown(
        "Catching the new attack is only half the result. A defense that learns a new family "
        "by challenging far more genuine customers, or by forgetting what it already knew, has "
        "not improved — it has moved the damage somewhere less visible."
    )
    if loop:
        last = loop["history"][-1]
        li = last["legit_impact"]
        cc = last.get("champion_challenger", {})
        g1, g2, g3 = columns(3)
        g1.metric("Genuine false positives after adapting",
                  f"{li['adapted_fpr']*100:.2f}%",
                  f"{li['fpr_regression']*100:+.2f} pts", delta_color="inverse")
        guards = last.get("guard_families_on_base", {})
        kept = [v["recall"] for v in guards.values() if v["recall"] is not None]
        g2.metric("Families never attacked, still caught",
                  f"{(sum(kept)/len(kept)*100):.0f}%" if kept else "—",
                  f"{len(kept)} control families")
        g3.metric("Promotion decision", cc.get("decision", "—"), delta_color="off")
        st.caption(cc.get("summary", ""))

        promo = loop.get("promotion", {})
        if not promo.get("promoted_rounds"):
            st.error(
                "In this run **no candidate cleared every promotion gate**. The adapted model "
                "learned the evolved attack, and it also breached a constraint the gates "
                "protect — so it would not be deployed. That result is shown here rather than "
                "quietly replaced with a better-looking one.", icon="⛔")
        else:
            st.success(f"Promoted in round(s) {promo['promoted_rounds']} — the candidate beat "
                       "the model in force on the new attack without breaching the "
                       "false-positive ceiling or regressing a previously-learned family.",
                       icon="✅")

    st.divider()
    st.subheader("Continuous evolution is harder than a single unseen family")
    if loop:
        rows = []
        for h in loop["history"]:
            for fam, d in h["families"].items():
                rows.append({"round": h["round"], "family": fam,
                             "stale": d["stale_recall"], "adapted": d["adapted_recall"],
                             "status": d["status"], "n": d["n"]})
        st.dataframe(pd.DataFrame(rows), width=STRETCH, hide_index=True)
    st.markdown(
        "> Some attacks remain structurally difficult at authorization time — that is exactly "
        "why continuous red teaming is required, and why we report the frontier instead of "
        "claiming it away."
    )
    st.caption("All results are simulation results on synthetic data. Nothing here has been "
               "validated on real payment data, and nothing here has been reviewed or "
               "validated by Mastercard.")

st.divider()
st.markdown("**The attack becomes the training ground for the next defense.**")
