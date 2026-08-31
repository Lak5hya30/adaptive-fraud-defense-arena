"""BENCHMARKS — does the machine learning earn its place?

Rules versus static ML versus the adaptive defense, compared both at each
detector's own operating point and at a matched false-positive budget, plus the
leave-one-attack-family-out evidence for why adaptation is needed at all.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui import Metric, metric_grid, page_header, plot

from common import (PALETTE, STRETCH, load_baseline, load_family_recall,
                    load_head_to_head, load_loao, load_operational, mode_selector,
                    page_setup)

import config

page_setup("Benchmarks", "📊")
page_header("Rules vs Static ML vs Adaptive Defense", "Evidence")
mode_selector()

comp = load_baseline()
loao = load_loao()

if not comp:
    st.info("Run `python -m src.defend.baseline` to generate the comparison.", icon="ℹ️")
    st.stop()

LABELS = {
    "rules_baseline": "Rules baseline",
    "static_ml": "Static ML",
    "adaptive_ml": "Adaptive (promoted)",
    "adaptive_candidate_unpromoted": "Adaptive candidate (NOT promoted)",
}
meta = comp.get("_meta", {})

st.markdown(
    "The rule set is deliberately **competent, not a strawman**: it includes the "
    "network-level rules a real fraud team would write once it had the same counters the "
    "model gets — device shared across cards, card seen on many devices, merchants seeing "
    "almost only first-time cards. Its thresholds are round, domain-chosen numbers, never "
    "fitted to this dataset."
)
st.caption(f"Evaluated on the {meta.get('evaluated_on', 'held-out test set')}: "
           f"{meta.get('n_eval', 0):,} transactions, {meta.get('n_fraud_eval', 0)} fraudulent.")

st.subheader("At each detector's own operating point")
rows = []
for key, m in comp.items():
    if key.startswith("_"):
        continue
    rows.append({"model": LABELS.get(key, key), "recall": m["recall"],
                 "precision": m["precision"], "F1": m["f1"],
                 "FPR": m["false_positive_rate"],
                 "FP per 1,000 genuine": m.get("false_positives_per_1000_legit"),
                 "PR-AUC": m.get("pr_auc")})
native = pd.DataFrame(rows)
st.dataframe(native, width=STRETCH, hide_index=True)

st.warning(
    "Comparing detectors at whatever operating point each happens to sit on is close to "
    "meaningless — any of them can buy recall by challenging more genuine customers. The "
    "comparison below re-thresholds every detector to spend the **same** false-positive "
    "budget on the **same** split. That is the comparison worth arguing about.", icon="⚖️")

matched = comp.get("_fpr_matched")
if matched:
    st.subheader(f"At a matched {matched['target_false_positive_rate']*100:.0f}% "
                 "false-positive budget")
    mrows = []
    for key, m in matched["models"].items():
        mrows.append({"model": LABELS.get(key, key), "recall": m["recall"],
                      "precision": m["precision"], "F1": m["f1"],
                      "realized FPR": m["false_positive_rate"],
                      "FP per 1,000 genuine": m.get("false_positives_per_1000_legit")})
    mdf = pd.DataFrame(mrows)
    metric_grid([Metric(r["model"], f'{r["recall"]*100:.0f}% recall',
                        f'Precision {r["precision"]:.3f} · '
                        f'{r["FP per 1,000 genuine"]:.1f} false positives / 1,000 genuine') for r in mrows])
    fig = go.Figure()
    for _, r in mdf.iterrows():
        fig.add_bar(name=r["model"], x=["recall", "precision", "F1"],
                    y=[r["recall"], r["precision"], r["F1"]])
    fig.update_layout(barmode="group", height=340, plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=-0.2),
                      margin=dict(t=10), yaxis_title="")
    plot(fig, width=STRETCH)
    st.caption(matched["note"])

h2h = load_head_to_head()
if h2h:
    st.divider()
    st.subheader("What adaptation actually buys — the evolved attacks")
    st.markdown(
        "Everything above scores detectors on the **original** attack distribution, which is "
        "the static model's home ground: it was trained on exactly that, so it should win "
        "there. The question adaptation exists to answer is what happens once the attack has "
        f"**moved**. Both models below are scored on the same unseen frame of the final "
        f"evolved generation — {h2h['frame']['n_fraud']} fraudulent transactions, from a seed "
        "neither model has seen."
    )
    H_LABELS = {"static_defense": "Static defense<br>(never saw the evolved attack)",
                "adaptive_defense": "Adaptive defense<br>(trained through the loop)",
                "promoted_champion": "Promoted champion"}
    metric_grid([Metric(H_LABELS.get(name, name).replace("<br>", " "),
                        f'{(blk["mean_evolved_recall"] or 0)*100:.0f}%',
                        f'Mean evolved recall · {blk["false_positive_rate"]*100:.2f}% false positives')
                 for name, blk in h2h["models"].items()])
    rows = []
    for fam in h2h["focus"]:
        row = {"evolved family": fam}
        for name, blk in h2h["models"].items():
            d = blk["evolved_family_recall"].get(fam)
            row[H_LABELS.get(name, name).replace("<br>", " ")] = (
                f"{d['recall']*100:.0f}% (n={d['n']})" if d else "—")
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width=STRETCH, hide_index=True)
    st.caption(h2h["note"])
    with st.expander("The specifications that produced these evolved attacks"):
        st.json(h2h["specs"], expanded=False)

ops = load_operational()
if ops:
    st.subheader("What the false-positive rate means in practice")
    orows = []
    for name, o in ops.items():
        orows.append({
            "model": LABELS.get(name, name),
            "false positives per 1,000 genuine payments":
                o["per_1000"]["false_positives_per_1000_legit"],
            "fraud caught per 1,000 fraud attempts":
                o["per_1000"]["fraud_caught_per_1000_fraud"],
            "step-up rate": o["action_distribution"]["step_up"],
            "decline rate": o["action_distribution"]["decline"],
        })
    st.dataframe(pd.DataFrame(orows), width=STRETCH, hide_index=True)
    st.caption("Per-thousand figures are the honest way to read a false-positive rate: they "
               "are customers, and someone has to answer for each one.")

st.subheader("Where the lift actually is — recall by attack family")
fam = load_family_recall()
if fam:
    st.caption(fam["note"])
    frows = []
    for model_name, block in fam["models"].items():
        for a, d in block["per_family_recall"].items():
            frows.append({"family": a, "model": LABELS.get(model_name, model_name),
                          "recall": d["recall"], "n": d["n"]})
    fdf = pd.DataFrame(frows)
    fig = px.bar(fdf, x="recall", y="family", color="model", orientation="h",
                 barmode="group", range_x=[0, 1], hover_data=["n"])
    fig.update_layout(height=520, plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=10), yaxis_title="",
                      legend=dict(orientation="h", y=-0.15))
    plot(fig, width=STRETCH)
elif "static_ml" in comp:
    st.caption("Run `python -m src.defend.diagnostics` for family recall at a reportable "
               "sample size.")

st.divider()
st.subheader("Why adaptation is needed — leave-one-attack-family-out")
st.caption("Each family is removed from training entirely, then scored as a genuinely unseen "
           "attack, then re-added. A large gain means the family is unseen-hard but learnable "
           "— which is precisely the gap the adversarial replay loop exists to close.")
if loao:
    lr = []
    for fam_name, r in loao["families"].items():
        if r["recall_unseen"] is None:
            continue
        lr.append({"family": fam_name, "unseen": r["recall_unseen"],
                   "after learning": r["recall_after_learning"], "n": r["n_test"],
                   "enough n": r.get("sufficient_n", True)})
    ldf = pd.DataFrame(lr).sort_values("unseen")
    melt = ldf.melt(["family", "n", "enough n"], var_name="stage", value_name="recall")
    fig = px.bar(melt, x="recall", y="family", color="stage", orientation="h",
                 barmode="group", range_x=[0, 1], hover_data=["n"],
                 color_discrete_map={"unseen": PALETTE["danger"],
                                     "after learning": PALETTE["safe"]})
    fig.update_layout(height=420, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=10), yaxis_title="", legend=dict(orientation="h"))
    plot(fig, width=STRETCH)
    st.dataframe(ldf, width=STRETCH, hide_index=True)
    st.info(loao.get("what_this_does_not_show", ""), icon="⚠️")
else:
    st.info("Run `python -m src.experiments.leave_one_out` to generate this.", icon="ℹ️")

st.divider()
st.caption("All figures are simulation results on synthetic data, regenerated by "
           "`python -m src.pipeline` from a single seed. No real cardholder data is used "
           "anywhere in this project, and nothing here has been validated against Mastercard "
           "production data.")
