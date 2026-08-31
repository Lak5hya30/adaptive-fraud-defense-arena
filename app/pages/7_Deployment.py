"""REAL-WORLD FEASIBILITY — what would actually run where.

The two paths are separated on purpose. Authorization is a synchronous, latency-
bound decision that reads precomputed state and never trains. Adaptation is an
offline research loop that produces candidate models, which only reach the
authorization path after passing governance.

Latency figures on this page are measured locally, on this machine, on the
prototype. They are a sanity check that the feature set is cheap, not a claim
about anyone's production infrastructure.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st

from ui import columns, page_header

from common import (PALETTE, STRETCH, load_registry, mode_selector, page_setup)

import config
from src.defend.features import FEATURE_COLUMNS, NETWORK_FEATURES

page_setup("Deployment", "🏗️")
page_header("How this would run in live payments", "Real-world feasibility")
mode_selector()

st.markdown(
    "Two paths, deliberately separate. **Nothing retrains during an authorization.** The "
    "online path reads precomputed state, scores, applies a policy and logs reason codes. "
    "The offline path is where red teaming, replay and retraining happen, and its output is a "
    "*candidate* that has to pass governance before it is allowed anywhere near a live "
    "decision."
)

c1, c2 = columns(2)
with c1:
    st.subheader("Online — the authorization path")
    st.code("""Payment authorization
        |
Real-time feature service          precomputed per-card and network counters
        |
Fraud risk model                   gradient boosting + isolation forest, no training
        |
Isotonic calibration               score becomes a probability
        |
Decision policy                    thresholds fixed, versioned, auditable
   /        |        \\
APPROVE  STEP-UP   DECLINE
        |
Reason codes logged                for the analyst, the customer and the regulator""",
            language=None)
    st.markdown(
        "- Only **authorization-time** features. Post-outcome fields are hard-blocked in code "
        "and the block is unit-tested.\n"
        "- The per-card and network features are **running counters**, not graph queries: a "
        "counter service keyed on card, device, IP and merchant answers in constant time.\n"
        "- No model update, no retraining, no network call to a language model.\n"
        "- Thresholds are configuration, not model internals, so a policy change is a "
        "reviewable change."
    )
with c2:
    st.subheader("Offline — the adaptation path")
    st.code("""New fraud intelligence
        |
Red-team agent                     proposes a structured attack specification
        |
Constraint layer                   payment-domain rules: clamp or reject
        |
Constrained simulator              deterministic, seeded
        |
Stress test                        against the model currently in force
        |
Replay dataset                     bounded, stratified across generations
        |
Retraining                         produces a CANDIDATE
        |
Evaluation + champion/challenger   gates below
        |
Shadow / controlled rollout        NOT simulated in this prototype""",
            language=None)
    st.markdown(
        "- Runs on a research cadence — days or weeks — not inside a payment.\n"
        "- The language model is used **here only**, and only to write specifications. If it "
        "is unavailable the loop still runs from a deterministic weakness-driven heuristic.\n"
        "- Being offline is what makes the reproducibility claim possible: the whole path is "
        "seeded and re-runnable."
    )

st.divider()

st.subheader("Feature cost at authorization time")
st.caption("Every feature the online path uses, and where its state would live.")
rows = []
for f in FEATURE_COLUMNS:
    if f in NETWORK_FEATURES:
        where = "network counter service (keyed on device / IP / merchant)"
    elif f.endswith("_prior") or f in ("amount_zscore", "velocity_1h", "velocity_24h",
                                       "device_changed", "geo_changed",
                                       "card_history_depth", "time_since_last_hours",
                                       "card_prior_dispute_rate", "mcc_novel",
                                       "amount_vs_card_max_prior",
                                       "distance_vs_card_max_prior",
                                       "card_mcc_share_prior"):
        where = "per-card profile store (updated on settlement)"
    else:
        where = "on the authorization message itself"
    rows.append({"feature": f, "computed from": where})
fdf = pd.DataFrame(rows)
st.dataframe(fdf, width=STRETCH, hide_index=True, height=320)
st.caption(f"{len(FEATURE_COLUMNS)} features: "
           f"{sum(1 for r in rows if 'authorization message' in r['computed from'])} read "
           f"straight off the message, "
           f"{sum(1 for r in rows if 'per-card' in r['computed from'])} from a per-card "
           f"profile, and {len(NETWORK_FEATURES)} from network-level counters — the signals "
           "an issuer or a single merchant cannot compute alone, and a payment network can.")

st.subheader("Prototype scoring latency")
st.caption("Measured here, now, on this machine, on the committed model. This is a sanity "
           "check that the feature set and model are cheap to evaluate — not a production "
           "latency claim, and not a measurement of any real authorization path.")
if st.button("▶ Benchmark scoring on this machine"):
    from src.defend.model import DefenseModel
    if not config.MODEL_PATH.exists():
        st.error("No trained model available to benchmark.")
    else:
        model = DefenseModel.load()
        rng = np.random.default_rng(config.GLOBAL_SEED)
        with st.spinner("Timing…"):
            X = pd.DataFrame(rng.normal(size=(2048, len(model.feature_columns))),
                             columns=model.feature_columns)
            model.risk_probability(X.iloc[:8])          # warm up
            singles = []
            for i in range(200):
                row = X.iloc[[i]]
                t0 = time.perf_counter()
                model.risk_probability(row)
                singles.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            model.risk_probability(X)
            batch_ms = (time.perf_counter() - t0) * 1000
        b1, b2, b3 = columns(3)
        b1.metric("Median, one transaction", f"{np.median(singles):.2f} ms")
        b2.metric("95th percentile", f"{np.percentile(singles, 95):.2f} ms")
        b3.metric("Batch of 2,048", f"{batch_ms:.0f} ms",
                  f"{batch_ms/2048*1000:.0f} µs per transaction")
        st.caption("Single-row timings are dominated by Python call overhead; a production "
                   "path would score in a compiled runtime. Feature assembly, network calls "
                   "and the counter service are not included here — those, not the model, "
                   "would dominate a real latency budget.")

st.divider()

st.subheader("Model governance — champion / challenger")
reg = load_registry()
if not reg:
    st.info("Run `python -m src.loop.redteam_loop` to generate the model registry.", icon="ℹ️")
else:
    st.markdown("A candidate is promoted only if **every** gate passes:")
    g = reg.get("gates", {})
    st.markdown(
        f"- recall on the new attack ≥ champion **+ {g.get('min_attack_recall_gain', 0):.0%}**\n"
        f"- false-positive rate on genuine traffic ≤ **{g.get('max_fpr', 0):.1%}** absolute\n"
        f"- false-positive rate ≤ champion **+ {g.get('max_fpr_regression', 0):.1%}**\n"
        f"- no previously-learned family drops by more than "
        f"**{g.get('max_prior_recall_drop', 0):.0%}**\n"
        f"- overall PR-AUC ≥ champion **− {g.get('max_overall_pr_auc_drop', 0):.2f}**"
    )
    st.caption(reg.get("note", ""))
    entries = []
    for e in reg["entries"]:
        acc = e.get("acceptance", {})
        entries.append({
            "model": e["model_version"], "stage": e["stage"],
            "decision": acc.get("decision", "—"),
            "failed gates": ", ".join(acc.get("failed_gates", [])) or "—",
            "recall": e["metrics"].get("recall"),
            "FPR": e["metrics"].get("false_positive_rate"),
            "PR-AUC": e["metrics"].get("pr_auc"),
            "data seed": e.get("data_seed"),
            "schema": e.get("schema_version"),
        })
    st.dataframe(pd.DataFrame(entries), width=STRETCH, hide_index=True)

st.divider()
st.subheader("What this prototype does not do")
st.markdown(
    "- **No validation on real payment data.** Every number in this project is a simulation "
    "result. A real deployment needs a labelled backtest on issuer data before any of it means "
    "anything.\n"
    "- **No shadow-mode or controlled rollout.** Governance here stops at the promotion "
    "decision; a real rollout would run the challenger in shadow against live traffic first.\n"
    "- **No drift monitoring or feedback loop from analysts.** Both are required in production "
    "and neither is simulated.\n"
    "- **No production latency, throughput or availability engineering.**\n"
    "- **Nothing here has been reviewed or validated by Mastercard**, and no part of this "
    "describes any real payment-network system."
)
