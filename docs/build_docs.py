"""Render README.md, the figures, and the .docx solution walkthrough FROM the
committed artifacts.

Hand-maintained numbers in documentation are the single most reliable way to
lose a technical reviewer's confidence: the moment one table disagrees with
another, every other number becomes suspect too. So no figure in the README or
the walkthrough is typed by a human. Both are generated here, from
``models/*.json`` and ``data/summary.json``, and regenerating the pipeline
regenerates the documents.

    python -m docs.build_docs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402
from src.identify.taxonomy import load_taxonomy  # noqa: E402

FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

INK = "#1b1f2a"
ORANGE = "#EB6C1E"
BLUE = "#3E7BFA"
GREEN = "#30A46C"
RED = "#E5484D"
GREY = "#8B93A7"
AMBER = "#F79E1B"


def _load(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_all() -> dict:
    return {
        "summary": _load(config.DATA_DIR / "summary.json"),
        "metrics": _load(config.METRICS_PATH),
        "baseline": _load(config.BASELINE_COMPARISON_PATH),
        "loao": _load(config.LEAVE_ONE_OUT_PATH),
        "loop": _load(config.LOOP_HISTORY_PATH),
        "fidelity": _load(config.FIDELITY_REPORT_PATH),
        "family": _load(config.FAMILY_RECALL_PATH),
        "ops": _load(config.OPERATIONAL_PATH),
        "text": _load(config.MODELS_DIR / "text_metrics.json"),
        "registry": _load(config.MODEL_REGISTRY_PATH),
        "lineage": _load(config.ATTACK_LINEAGE_PATH),
        "sweep": _load(config.THRESHOLD_SWEEP_PATH),
        "calibration": _load(config.CALIBRATION_PATH),
        "hero": _load(config.HERO_EXAMPLE_PATH),
        "h2h": _load(config.HEAD_TO_HEAD_PATH),
        "tax": load_taxonomy(),
    }


def pct(x, dp=1):
    return "—" if x is None else f"{x*100:.{dp}f}%"


def num(x, dp=3):
    return "—" if x is None else f"{x:.{dp}f}"


# Below this many held-out examples a per-family recall is too thin to lead with.
MIN_N_TO_HEADLINE = 30


def ci_text(recall, n, precomputed=None) -> str:
    """A recall is a proportion measured on a handful of transactions. Printing it
    without its interval invites exactly one question, so every table prints both."""
    lo_hi = precomputed
    if lo_hi is None:
        from src.defend.evaluate import wilson_interval
        lo_hi = wilson_interval(recall, n)
    if not lo_hi:
        return "—"
    return f"{lo_hi[0]*100:.0f}–{lo_hi[1]*100:.0f}%"


def hero_family(A) -> tuple[str | None, dict | None]:
    """The family to lead with. Selection rule lives in one place; see
    :func:`src.experiments.leave_one_out.select_hero_family`.

    If nothing clears the sample-size floor this returns nothing at all rather
    than headlining an underpowered result.
    """
    from src.experiments.leave_one_out import select_hero_family
    fam, rec, underpowered = select_hero_family(A.get("loao"))
    if underpowered:
        return None, None
    return fam, rec


# Figures
def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, color=INK, fontsize=11, weight="bold", loc="left")
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    ax.tick_params(colors=INK, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9cedb")
    ax.grid(axis="x", color="#e7eaf1", linewidth=0.8)
    ax.set_axisbelow(True)


def _legend_below(ax, ncol=3, fontsize=8):
    """Legends sit under the axes, never on top of the data.

    A legend box floating over a bar is the difference between a figure a
    reviewer reads and one they skip.
    """
    ax.legend(frameon=False, fontsize=fontsize, ncol=ncol,
              loc="upper center", bbox_to_anchor=(0.5, -0.14))


def fig_baseline(A) -> Path | None:
    b = A.get("baseline")
    if not b or "_fpr_matched" not in b:
        return None
    models = b["_fpr_matched"]["models"]
    labels = {"rules_baseline": "Rules", "static_ml": "Static ML",
              "adaptive_ml": "Adaptive (promoted)",
              "adaptive_candidate_unpromoted": "Adaptive candidate\n(not promoted)"}
    names = [k for k in models if k in labels]
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=200)
    width = 0.26
    xs = range(len(names))
    ax.bar([x - width for x in xs], [models[n]["recall"] for n in names], width,
           label="recall", color=ORANGE)
    ax.bar(list(xs), [models[n]["precision"] for n in names], width,
           label="precision", color=BLUE)
    ax.bar([x + width for x in xs], [models[n]["f1"] for n in names], width,
           label="F1", color=GREEN)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([labels[n] for n in names], fontsize=8)
    _style(ax, f"At a matched {b['_fpr_matched']['target_false_positive_rate']*100:.0f}% "
               "false-positive budget")
    _legend_below(ax, ncol=3)
    fig.tight_layout()
    p = FIG_DIR / "benchmark_matched_fpr.png"
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


def fig_loao(A) -> Path | None:
    loao = A.get("loao")
    if not loao:
        return None
    rows = [(f, r["recall_unseen"], r["recall_after_learning"], r["n_test"])
            for f, r in loao["families"].items() if r["recall_unseen"] is not None]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 1.2), dpi=200)
    ys = range(len(rows))
    ax.barh([y + 0.2 for y in ys], [r[1] for r in rows], 0.38, color=RED, label="never seen")
    ax.barh([y - 0.2 for y in ys], [r[2] for r in rows], 0.38, color=GREEN,
            label="after learning")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{r[0]}  (n={r[3]})" for r in rows], fontsize=8)
    ax.set_xlim(0, 1.02)
    _style(ax, "Leave-one-attack-family-out: unseen vs learned", xlabel="recall")
    _legend_below(ax, ncol=2)
    fig.tight_layout()
    p = FIG_DIR / "leave_one_out.png"
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


def fig_family(A) -> Path | None:
    fam = A.get("family")
    if not fam or "static_ml" not in fam.get("models", {}):
        return None
    block = fam["models"]["static_ml"]["per_family_recall"]
    from src.defend.evaluate import wilson_interval
    rows = []
    for a, d in block.items():
        ci = wilson_interval(d["recall"], d["n"]) or [d["recall"], d["recall"]]
        rows.append((a, d["recall"], d["n"], ci))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.2, 0.4 * len(rows) + 1.2), dpi=200)
    ys = list(range(len(rows)))
    colors = [GREEN if r[1] >= 0.7 else (AMBER if r[1] >= 0.4 else RED) for r in rows]
    err_lo = [r[1] - r[3][0] for r in rows]
    err_hi = [r[3][1] - r[1] for r in rows]
    ax.barh(ys, [r[1] for r in rows], 0.6, color=colors,
            xerr=[err_lo, err_hi], error_kw=dict(ecolor=GREY, lw=1, capsize=2))
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r[0]}  (n={r[2]})" for r in rows], fontsize=8)
    ax.set_xlim(0, 1.05)
    _style(ax, "Recall by attack family, with 95% Wilson intervals", xlabel="recall")
    fig.tight_layout()
    p = FIG_DIR / "family_recall.png"
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


def fig_sweep(A) -> Path | None:
    s = A.get("sweep")
    if not s:
        return None
    pts = s["points"]
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=200)
    ax.plot([p["false_positive_rate"] for p in pts], [p["recall"] for p in pts],
            color=ORANGE, lw=2.2, label="recall")
    ax.plot([p["false_positive_rate"] for p in pts], [p["precision"] for p in pts],
            color=BLUE, lw=1.6, label="precision")
    ax.axvline(s["fpr_budget"], color=AMBER, ls="--", lw=1.2,
               label=f"budget {s['fpr_budget']*100:.0f}%")
    ax.set_xlim(0, 0.12)
    _style(ax, "Detection / friction trade-off across the threshold range",
           xlabel="false-positive rate on genuine traffic")
    _legend_below(ax, ncol=3)
    fig.tight_layout()
    p = FIG_DIR / "threshold_sweep.png"
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


def fig_separability(A) -> Path | None:
    fid = A.get("fidelity")
    if not fid:
        return None
    rows = fid["separability"][:16]
    fig, ax = plt.subplots(figsize=(7.2, 0.34 * len(rows) + 1.2), dpi=200)
    ys = list(range(len(rows)))[::-1]
    ax.barh(ys, [r["auc"] for r in rows], 0.45, color=ORANGE, label="ranking power (AUC)")
    ax.barh([y - 0.42 for y in ys], [r["overlap"] for r in rows], 0.4, color=GREEN,
            alpha=0.75, label="distribution overlap")
    ax.set_yticks(ys)
    ax.set_yticklabels([r["feature"] for r in rows], fontsize=7.5)
    ax.axvline(0.9, color=RED, ls="--", lw=1, label="shortcut threshold")
    ax.set_xlim(0, 1.02)
    _style(ax, "Fidelity: no single feature separates fraud from genuine traffic")
    _legend_below(ax, ncol=3, fontsize=7.5)
    fig.tight_layout()
    p = FIG_DIR / "fidelity_separability.png"
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


def fig_coverage(A) -> Path | None:
    tax = A["tax"]
    cov = tax.coverage_by_category()
    cats = sorted(cov, key=lambda c: -cov[c]["total"])
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(cats) + 1.2), dpi=200)
    ys = list(range(len(cats)))[::-1]
    left = [0] * len(cats)
    for status, color, label in (("IMPLEMENTED", GREEN, "simulated"),
                                 ("PARAMETERIZED", BLUE, "configurable"),
                                 ("RESEARCH_ONLY", GREY, "research only"),
                                 ("FUTURE", AMBER, "roadmap")):
        vals = [cov[c][status] for c in cats]
        ax.barh(ys, vals, 0.55, left=left, color=color, label=label)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(ys)
    ax.set_yticklabels(cats, fontsize=8)
    _style(ax, "Threat Atlas coverage — research surface vs simulator",
           xlabel="attacks catalogued")
    _legend_below(ax, ncol=4)
    fig.tight_layout()
    p = FIG_DIR / "atlas_coverage.png"
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


def fig_lineage(A) -> Path | None:
    lin = A.get("lineage")
    if not lin or not lin.get("lineage"):
        return None
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=200)
    palette = [ORANGE, BLUE, GREEN, RED, AMBER]
    plotted = False
    for i, (fam, nodes) in enumerate(lin["lineage"].items()):
        gens = [n["generation"] for n in nodes if n.get("stale_recall") is not None]
        stale = [n["stale_recall"] for n in nodes if n.get("stale_recall") is not None]
        adapt = [n["adapted_recall"] for n in nodes if n.get("stale_recall") is not None]
        if not gens:
            continue
        plotted = True
        c = palette[i % len(palette)]
        ax.plot(gens, stale, "o--", color=c, lw=1.4, alpha=0.6,
                label=f"{fam} — stale defense")
        ax.plot(gens, adapt, "o-", color=c, lw=2.2, label=f"{fam} — after replay")
    if not plotted:
        plt.close(fig)
        return None
    ax.set_ylim(0, 1.05)
    ax.set_xticks(sorted({g for nodes in lin["lineage"].values()
                          for g in [n["generation"] for n in nodes]}))
    _style(ax, "Attack lineage: each generation targets the signal the defense relied on",
           xlabel="attack generation", ylabel="recall")
    _legend_below(ax, ncol=2, fontsize=7)
    fig.tight_layout()
    p = FIG_DIR / "attack_lineage.png"
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


def build_figures(A) -> dict:
    return {
        "coverage": fig_coverage(A),
        "separability": fig_separability(A),
        "baseline": fig_baseline(A),
        "family": fig_family(A),
        "loao": fig_loao(A),
        "sweep": fig_sweep(A),
        "lineage": fig_lineage(A),
    }


# README
def _md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def build_readme(A, figs) -> str:
    tax = A["tax"]
    counts = tax.summary_counts()
    s, m = A["summary"], A["metrics"]
    b, loao, loop = A["baseline"], A["loao"], A["loop"]
    fid, fam, ops = A["fidelity"], A["family"], A["ops"]
    hf, hero = hero_family(A)

    parts = []
    parts.append(
        "# AI Defense Lab for Adaptive Payment Fraud\n\n"
        "**A closed-loop GenAI red team / blue team laboratory for discovering, simulating "
        "and defending against emerging payment fraud.**\n\n"
        "*Mastercard Innovation Challenge 2026 · GFF Mumbai*\n\n"
        "```\nDISCOVER  ->  SIMULATE  ->  ATTACK  ->  DETECT  ->  ADAPT\n```\n\n"
        "> Traditional fraud systems mostly learn from fraud that has already happened. This "
        "lab actively searches for what the current defense does not know, generates realistic "
        "adversarial payment behaviour, stress-tests the model against it, and converts every "
        "discovered weakness into training data for the next defense.\n\n"
        "> **All data is synthetic.** Every number in this document is a simulation result, "
        "regenerated by `python -m src.pipeline` from a single seed. Nothing here has been "
        "validated on real payment data, and nothing here has been reviewed or endorsed by "
        "Mastercard.\n"
    )

    # ---- table of contents -------------------------------------------------
    parts.append(
        "\n---\n\n"
        "## Contents\n\n"
        "| Section | What it answers |\n|---|---|\n"
        "| [Quick start](#quick-start) | How do I run it? |\n"
        "| [The problem](#the-problem) | Why does this need to exist? |\n"
        "| [How it works](#how-it-works) | What did you actually build? |\n"
        "| [Headline results](#headline-results) | Does it work? |\n"
        "| [Architecture](#architecture) | How is it put together? |\n"
        "| [Where generative AI contributes](#where-generative-ai-actually-contributes) | "
        "Is the GenAI real or decorative? |\n"
        "| [Is the synthetic data trustworthy?](#is-the-synthetic-data-worth-training-on) | "
        "Why should I believe any of these numbers? |\n"
        "| [The Threat Atlas](#the-threat-atlas) | How much of the fraud landscape is covered? |\n"
        "| [What it costs to run](#what-it-costs-to-run) | What would this mean operationally? |\n"
        "| [Repository map](#repository-map) | Where is everything? |\n"
        "| [Documentation](#documentation) | Where do I read more? |\n"
        "| [Limitations](#limitations) | What is it not? |\n"
        "| [Reproducibility](#reproducibility) | Can I regenerate all of this myself? |\n"
    )

    # ---- quick start -------------------------------------------------------
    parts.append(
        "\n---\n\n## Quick start\n\n"
        "Python 3.11 or newer. No API key, no network access and no training are required to "
        "run the prototype.\n\n"
        "```bash\n"
        "git clone https://github.com/Lak5hya30/adaptive-fraud-defense-arena.git\n"
        "cd adaptive-fraud-defense-arena\n"
        "python -m pip install -r requirements.txt\n"
        "streamlit run app/Home.py\n"
        "```\n\n"
        "The app opens on `http://localhost:8501` in **Demo mode**, which renders every page "
        "from committed artifacts. Nothing trains, nothing calls out to a network, and every "
        "figure matches this README because both are produced from the same files.\n\n"
        "### Other commands\n\n"
        "| Command | What it does | Roughly |\n|---|---|---|\n"
        "| `python -m src.pipeline` | Regenerate every artifact from the seed | 45 min |\n"
        "| `python -m src.pipeline --fast` | Everything except the closed loop | 12 min |\n"
        "| `python -m docs.build_docs` | Regenerate this README, the figures and the "
        "walkthrough from those artifacts | seconds |\n"
        "| `python -m pytest tests/ -q` | Full test suite | 11 min |\n"
        "| `python -m src.generate.fidelity` | Simulator shortcut diagnostics | 1 min |\n"
        "| `python -m src.generate.demo_specs` | Live GenAI specification demo (needs a key) | "
        "seconds |\n\n"
        "Run `python -m docs.build_docs` after any pipeline run. The documentation is generated "
        "from artifacts, so skipping it is the one way to reintroduce a number that disagrees "
        "with the app.\n")

    # ---- the problem -------------------------------------------------------
    parts.append(
        "\n---\n\n## The problem\n\n"
        "Supervised fraud detection is trained on labelled history: fraud happens, customers "
        "dispute it, chargebacks land, someone labels them, the model retrains. That loop takes "
        "weeks.\n\n"
        "Generative AI broke the economics on the other side of it. Producing a new, plausible, "
        "well-targeted attack — a fresh lure in fluent local language, a synthetic identity with "
        "a coherent digital footprint, a variant that sidesteps whichever control just started "
        "biting — now costs almost nothing. Attacks iterate in hours; defenses iterate in weeks."
        "\n\nSo the useful question is not *how accurate is the model on last year's fraud*. It "
        "is **what does it do the first time it meets something new, and how quickly can it "
        "learn**. That is what this project measures, and the "
        "[leave-one-attack-family-out experiment](#the-hero-result-unseen-to-learned) is the "
        "direct answer.\n")

    # ---- how it works ------------------------------------------------------
    parts.append(
        "\n---\n\n## How it works\n\n"
        "Five stages, each implemented by a package under `src/`.\n\n"
        "| Stage | What happens | Where |\n|---|---|---|\n"
        "| **Discover** | A structured catalog of GenAI-enabled payment fraud, each entry "
        "labelled with whether the simulator actually reproduces it | `src/identify/` |\n"
        "| **Simulate** | A constrained engine turns an attack *specification* into transactions "
        "inside a realistic synthetic portfolio | `src/generate/` |\n"
        "| **Attack** | The red team measures which signal the defense depends on and aims the "
        "next attack generation at removing it | `src/loop/` |\n"
        "| **Detect** | Gradient boosting fused with an anomaly detector over authorization-time "
        "features only | `src/defend/` |\n"
        "| **Adapt** | What escapes goes into a replay buffer, the defense retrains, and "
        "governance decides whether the result may ship | `src/loop/`, `src/defend/` |\n\n"
        "The distinguishing property is that stages three and five are driven by *measurement*. "
        "Nothing about which family is attacked, or which behaviour is changed, is decided in "
        "advance — see [Design](docs/DESIGN.md) for why that matters and "
        "[Architecture](docs/ARCHITECTURE.md) for how it is wired.\n")

    if m:
        parts.append("\n---\n\n## Headline results\n\n")
        rows = [
            ("Recall", pct(m["recall"])),
            ("Precision", num(m["precision"])),
            ("F1", num(m["f1"])),
            ("PR-AUC", num(m["pr_auc"])),
            ("ROC-AUC", num(m["roc_auc"])),
            ("False-positive rate", f"{pct(m['false_positive_rate'], 2)} "
                                    f"(budget ≤ {pct(config.TARGET_MAX_FPR, 0)})"),
            ("False positives per 1,000 genuine payments",
             f"{m['false_positive_rate']*1000:.1f}"),
            ("Held-out test set", f"{m.get('test_size', 0):,} transactions, "
                                  f"{m.get('n_fraud_eval', 0)} fraudulent"),
        ]
        parts.append(_md_table(["Metric", "Value"], rows))
        parts.append(
            "\n\nPrecision is naturally modest at a "
            f"{config.DEFAULT_FRAUD_RATE*100:.1f}% base rate. The answer is tiered "
            "decisioning — challenge, do not decline — not a bigger number. PR-AUC leads "
            "because ROC-AUC flatters problems this imbalanced.\n")

    if b and "_fpr_matched" in b:
        parts.append("\n### Rules vs static ML vs adaptive defense\n\n"
                     "Compared at a **matched false-positive budget** on the same held-out "
                     "split. Comparing detectors at whatever operating point each happens to "
                     "sit on is close to meaningless — any of them can buy recall by "
                     "challenging more genuine customers.\n\n")
        labels = {"rules_baseline": "Rules baseline", "static_ml": "Static ML",
                  "adaptive_ml": "Adaptive (promoted)",
                  "adaptive_candidate_unpromoted": "Adaptive candidate (not promoted)"}
        rows = []
        for k, v in b["_fpr_matched"]["models"].items():
            rows.append((labels.get(k, k), pct(v["recall"]), num(v["precision"]),
                         num(v["f1"]), pct(v["false_positive_rate"], 2),
                         f"{v.get('false_positives_per_1000_legit', 0):.1f}"))
        parts.append(_md_table(
            ["Detector", "Recall", "Precision", "F1", "FPR", "FP / 1,000 genuine"], rows))
        parts.append("\n\nThe rule set is deliberately competent, not a strawman: it includes "
                     "the network-level rules a real fraud team would write once it had the "
                     "same counters the model gets. Its thresholds are round, domain-chosen "
                     "numbers, never fitted to this dataset. Matched thresholds are selected "
                     "on the validation slice and applied unchanged to the test slice, so no "
                     "detector gets to see the answer before choosing where to sit.\n\n"
                     "**On the adaptive model's numbers here:** this table scores every "
                     "detector on the *original* attack distribution, which is the static "
                     "model's home ground — it was trained on exactly this. The adaptive model "
                     "is trained on that data plus several generations of evolved attacks, so "
                     "on the original distribution it is at best comparable. Its advantage is "
                     "robustness to attacks that have moved, which is measured on the Closed "
                     "Loop page against each evolved generation, not here.\n")
        if figs.get("baseline"):
            parts.append(f"\n![Benchmark at matched FPR](docs/figures/"
                         f"{figs['baseline'].name})\n")

    h2h = A.get("h2h")
    if h2h:
        labels = {"static_defense": "Static defense (never saw the evolved attack)",
                  "adaptive_defense": "Adaptive defense (trained through the loop)",
                  "promoted_champion": "Promoted champion"}
        parts.append(
            "\n### What adaptation actually buys — the evolved attacks\n\n"
            "Every comparison above scores detectors on the *original* attack distribution, "
            "which is the static model's home ground. The question adaptation exists to answer "
            "is what happens once the attack has moved. Both models below are scored on the "
            f"same unseen frame of the **final evolved generation** "
            f"({h2h['frame']['n_fraud']} fraudulent transactions, seed unseen by either "
            "model):\n\n")
        rows = []
        for k, blk in h2h["models"].items():
            per = "  ·  ".join(f"{f} {v['recall']*100:.0f}% (n={v['n']})"
                               for f, v in blk["evolved_family_recall"].items())
            rows.append((labels.get(k, k), pct(blk["mean_evolved_recall"], 0),
                         pct(blk["false_positive_rate"], 2), per))
        parts.append(_md_table(
            ["Defense", "Mean recall on evolved attacks", "FPR", "By family"], rows))
        parts.append("\n\n" + h2h["note"] + "\n")
        # Spell out the per-family trade. A mean across families hides the thing a
        # reviewer most wants to know: which attacks the adaptation actually
        # bought, and what it gave up to buy them.
        base_fam = h2h["models"].get("static_defense", {}).get("evolved_family_recall", {})
        best = h2h["models"].get("promoted_champion") or h2h["models"].get("adaptive_defense")
        if base_fam and best:
            gains, losses = [], []
            for f, v in best["evolved_family_recall"].items():
                if f not in base_fam:
                    continue
                delta = v["recall"] - base_fam[f]["recall"]
                entry = f"**{f}** {base_fam[f]['recall']*100:.0f}% → {v['recall']*100:.0f}%"
                (gains if delta > 0.02 else losses if delta < -0.02 else []).append(entry)
            if gains or losses:
                parts.append(
                    "\n**Read the trade, not just the mean.** "
                    + (("Adaptation bought " + "; ".join(gains) + ". ") if gains else "")
                    + (("It gave up " + "; ".join(losses) + ". ") if losses else "")
                    + "The families the red team actually attacked are the ones that moved; "
                      "a single mean across families would hide both halves of that, which is "
                      "why the per-family numbers are printed next to it.\n")

    if hero and hf:
        parts.append(
            "\n### The hero result: unseen to learned\n\n"
            f"With **{hf.replace('_', ' ')}** removed from training entirely, the defense "
            f"caught **{pct(hero['recall_unseen'], 0)}** of it. After the lab generated that "
            f"family and replayed it into training: **{pct(hero['recall_after_learning'], 0)}**"
            f" (n={hero['n_test']} held-out transactions"
            + (f", 95% interval {hero['recall_after_learning_ci95'][0]*100:.0f}–"
               f"{hero['recall_after_learning_ci95'][1]*100:.0f}%"
               if hero.get("recall_after_learning_ci95") else "") + ").\n\n"
            f"This family is chosen automatically as the largest measured gain among families "
            f"with at least {MIN_N_TO_HEADLINE} held-out examples. Families below that floor "
            "appear in the table below, marked, and are never used as the headline — a wide "
            "interval cannot carry one, however good the point estimate looks.\n\n"
            "This measures **unseen → learned adaptation on synthetic data**. It is not a "
            "zero-shot detection claim, and `after learning` is an upper bound obtained by "
            "putting the family directly into training — the closed loop has to *reach* that "
            "bound by generating the family itself.\n")
        if figs.get("loao"):
            parts.append(f"\n![Leave-one-attack-out](docs/figures/{figs['loao'].name})\n")
        rows, thin = [], []
        for f, r in sorted(A["loao"]["families"].items(),
                           key=lambda kv: -(kv[1]["gain"] or 0)):
            if r["recall_unseen"] is None:
                continue
            small = r["n_test"] < MIN_N_TO_HEADLINE
            if small:
                thin.append(f)
            rows.append((
                f + (" *" if small else ""), pct(r["recall_unseen"], 0),
                ci_text(r["recall_unseen"], r["n_test"], r.get("recall_unseen_ci95")),
                pct(r["recall_after_learning"], 0),
                ci_text(r["recall_after_learning"], r["n_test"],
                        r.get("recall_after_learning_ci95")),
                r["n_test"]))
        parts.append("\n" + _md_table(
            ["Family", "Never seen", "95% interval", "After learning", "95% interval", "n"],
            rows))
        if thin:
            parts.append(
                f"\n\n`*` {', '.join(thin)} — fewer than {MIN_N_TO_HEADLINE} held-out "
                "examples. Shown for completeness and deliberately **not** used as the hero "
                "result: the interval is far too wide to carry a headline, however good the "
                "point estimate looks. The hero above is chosen automatically from the "
                "families that clear the sample-size floor.\n")

    if fam and "static_ml" in fam.get("models", {}):
        parts.append("\n### Recall by attack family\n\n"
                     f"Measured on an unseen, fraud-enriched frame "
                     f"({fam['frame']['n_transactions']:,} transactions, "
                     f"{fam['frame']['n_fraud']} fraudulent) generated from a seed no model "
                     "was trained or tuned on, so every family has enough held-out examples "
                     "for its number to mean something.\n\n")
        block = fam["models"]["static_ml"]["per_family_recall"]
        rows = [(k, pct(v["recall"], 0), v["n"],
                 ci_text(v["recall"], v["n"], v.get("ci95")))
                for k, v in sorted(block.items(), key=lambda kv: -kv[1]["recall"])]
        parts.append(_md_table(["Family", "Recall", "n", "95% interval"], rows))
        parts.append(
            "\n\nFamilies whose fraud is **authorized by the genuine customer on their own "
            "device** — first-party dispute abuse, and scams the victim was talked into — sit "
            "at the bottom by construction. There is very little to see at authorization "
            "time. The correct control for those is friction, payee-risk intelligence and "
            "post-transaction recall, not a hard decline. They are reported rather than "
            "quietly excluded.\n")
        if figs.get("family"):
            parts.append(f"\n![Family recall](docs/figures/{figs['family'].name})\n")

    if loop:
        promo = loop.get("promotion", {})
        parts.append("\n### The closed loop\n\n")
        parts.append(
            f"Targets are chosen from the defense's **measured** weakest families "
            f"({', '.join(loop['focus'])}), not decided in advance. Families whose fraud is "
            "authorized by the genuine customer are measured but not attacked further: their "
            "limit is observability, not the decision boundary.\n\n")
        rows = []
        for h in loop["history"]:
            for famname, d in h["families"].items():
                rows.append((h["round"], famname, d.get("targets_signal") or "—",
                             pct(d["stale_recall"], 0), pct(d["adapted_recall"], 0),
                             d["status"], d["n"]))
        parts.append(_md_table(
            ["Round", "Family", "Signal targeted", "Stale defense", "After replay",
             "Status", "n"], rows))
        retired = loop.get("retired_frontiers", [])
        if retired:
            parts.append("\n\n**The red team reallocates.** " + " ".join(
                f"After round {r['round']} it retired `{r['family']}`"
                + (f" and moved to `{r['replaced_by']}`." if r["replaced_by"] else ".")
                for r in retired)
                + " A family that stays a residual frontier under repeated attack will not "
                  "yield to more of the same — it sits on the legitimate behavioural centroid, "
                  "or the genuine customer authorized it. Continuing to hammer it burns replay "
                  "capacity on examples the model cannot separate, and drags down families it "
                  "could. The frontier is reported, not hidden.\n")
        promoted = promo.get("promoted_rounds", [])
        parts.append(
            f"\n\n**Governance:** every adapted model is a challenger measured against the "
            f"model in force, and ships only if it clears every promotion gate. In this run "
            + (f"round(s) {promoted} were promoted."
               if promoted else
               "**no candidate cleared every gate** — the adapted model improved detection on "
               "the evolved attack but breached another constraint, so it would not be "
               "deployed. That result is reported rather than replaced with a "
               "better-looking one.") + "\n")
        if figs.get("lineage"):
            parts.append(f"\n![Attack lineage](docs/figures/{figs['lineage'].name})\n")

    parts.append("\n---\n\n## Architecture\n\n"
                 "```\n"
                 "  Emerging fraud intelligence\n"
                 "             |\n"
                 "  +----------v-----------+\n"
                 "  |     THREAT ATLAS     |  src/identify/    catalogued attacks, each\n"
                 "  |                      |                   labelled by simulator status\n"
                 "  +----------+-----------+\n"
                 "             |\n"
                 "  +----------v-----------+\n"
                 "  |   RED-TEAM AGENT     |  src/generate/llm_agent.py\n"
                 "  |                      |  src/loop/weakness.py\n"
                 "  |  structured spec     |  deterministic heuristic (Demo mode)\n"
                 "  |                      |  or language model (optional)\n"
                 "  +----------+-----------+\n"
                 "             |\n"
                 "  +----------v-----------+\n"
                 "  |  CONSTRAINT LAYER    |  src/generate/attack_spec.py\n"
                 "  |  clamp or reject     |  payment-domain rules\n"
                 "  +----------+-----------+\n"
                 "             |\n"
                 "  +----------v-----------+\n"
                 "  | CONSTRAINED SIMULATOR|  src/generate/    deterministic, seeded\n"
                 "  +----------+-----------+\n"
                 "             |\n"
                 "  +----------v-----------+\n"
                 "  |    DEFENSE MODEL     |  src/defend/      authorization-time\n"
                 "  |                      |                   features only\n"
                 "  +----+------------+----+\n"
                 "       |            |\n"
                 "   detected      escaped\n"
                 "                    |\n"
                 "         +----------v-----------+\n"
                 "         |  WEAKNESS ANALYSIS   |  src/loop/weakness.py\n"
                 "         |  which signal did    |  permutation attribution,\n"
                 "         |  the model rely on?  |  per family\n"
                 "         +----------+-----------+\n"
                 "                    |\n"
                 "         +----------v-----------+\n"
                 "         |   ATTACK EVOLUTION   |  aim the next generation at\n"
                 "         |                      |  removing that signal\n"
                 "         +----------+-----------+\n"
                 "                    |\n"
                 "         +----------v-----------+\n"
                 "         |  ADVERSARIAL REPLAY  |  bounded, stratified, with\n"
                 "         |  + RETRAIN           |  rehearsal of untouched families\n"
                 "         +----------+-----------+\n"
                 "                    |\n"
                 "         +----------v-----------+\n"
                 "         | CHAMPION / CHALLENGER|  src/defend/governance.py\n"
                 "         |  promotion gates     |  may refuse the candidate\n"
                 "         +----------+-----------+\n"
                 "                    |\n"
                 "                    +--> promoted model returns to DEFENSE, loop repeats\n"
                 "```\n\n"
                 "Two runtime paths, kept deliberately separate. The **online** authorization "
                 "path reads precomputed counters, scores, calibrates, applies a versioned "
                 "policy and logs reason codes — it never trains and never calls a language "
                 "model. The **offline** adaptation path is the loop above, running on a "
                 "research cadence, and its output is a *candidate* that must clear governance "
                 "before it could reach a live decision. "
                 "[Architecture](docs/ARCHITECTURE.md) has the module-level detail.\n")

    parts.append("\n### Where generative AI actually contributes\n\n"
                 "The language model **never writes transactions**. It writes a *specification*:\n\n"
                 "```json\n" + json.dumps({
                     "attack_family": "account_takeover",
                     "strategy": "reuse the victim's trusted device, change merchant behaviour",
                     "amount_profile": "moderate",
                     "velocity_profile": "low_and_slow",
                     "device_behavior": "trusted_device",
                     "merchant_behavior": "new_high_risk_merchant",
                     "geo_behavior": "plausible",
                     "targets_signal": "device_changed",
                     "confidence": 0.82}, indent=2) + "\n```\n\n"
                 "That specification is validated against payment-domain constraints — an "
                 "authorized push payment scam cannot run from an attacker's device, because "
                 "the genuine customer is the one authenticating — and anything impossible is "
                 "clamped or refused before a single row is generated. A deterministic, seeded "
                 "simulator then executes what survives.\n\n"
                 "### Two paths, and which one produced the committed numbers\n\n"
                 "**Demo mode uses deterministic, committed specifications.** That is a "
                 "reliability decision, not a limitation: the prototype has to run on stage "
                 "with no API key, no network and no training, and every committed figure has "
                 "to be reproducible from a single seed. So every specification in "
                 "`models/attack_lineage.json` carries `spec_source: \"heuristic\"` and every "
                 "content artifact carries `source: \"template\"`. The weakness-driven "
                 "heuristic reads the same measured evidence and moves the same dials, which "
                 "is why the behaviour being demonstrated is the loop's rather than a single "
                 "model call's.\n\n"
                 "**The optional GenAI red team generates the specification instead.** With "
                 "`ANTHROPIC_API_KEY` set, the language model receives the measured weakness "
                 "and returns the structured specification, which then goes through exactly "
                 "the same payment-domain constraint layer before anything is simulated — same "
                 "validation, same clamping, same refusals. Specifications produced that way "
                 "are stamped `spec_source: \"llm\"`, so the two are never confused in the "
                 "artifacts.\n\n"
                 "```bash\n"
                 "python -m src.generate.demo_specs --check   # report readiness\n"
                 "python -m src.generate.demo_specs           # 5 live specs -> "
                 "models/genai_spec_demo.json\n"
                 "```\n\n"
                 "That script demonstrates the live path on five safe synthetic seeds without "
                 "touching the dataset, the models or any committed metric. With no key it "
                 "writes nothing and says so, rather than quietly substituting heuristic "
                 "output for a model response.\n")

    if fid:
        parts.append("\n---\n\n## Is the synthetic data worth training on?\n\n"
                     "If any single field separated fraud from genuine traffic cleanly, the "
                     "detector would be learning the generator rather than fraud, and every "
                     "number above would be meaningless. So it is measured, not asserted:\n\n")
        rows = [(c["check"].replace("_", " "), c["status"], c["detail"])
                for c in fid["checks"]]
        parts.append(_md_table(["Check", "Result", "Detail"], rows))
        parts.append(f"\n\nStrongest single feature: **{fid['separability'][0]['feature']}** "
                     f"at AUC {fid['max_single_feature_auc']:.3f}. "
                     f"{fid['n_fail']} failing checks, {fid['n_warn']} warnings.\n")
        if figs.get("separability"):
            parts.append(f"\n![Fidelity](docs/figures/{figs['separability'].name})\n")

    counts = tax.summary_counts()
    parts.append(f"\n---\n\n## The Threat Atlas\n\n"
                 f"**{counts['total_attacks']} attacks** across **{counts['categories']} fraud "
                 f"surfaces** and {counts['rails']} payment rails. Deliberately wider than the "
                 f"simulator, and labelled so the difference is visible:\n\n")
    parts.append(_md_table(
        ["Status", "Count", "Meaning"],
        [("IMPLEMENTED", counts["implemented"],
          "a dedicated injector reproduces its authorization footprint"),
         ("PARAMETERIZED", counts["parameterized"],
          "reachable today by configuring an existing injector, no new code"),
         ("RESEARCH_ONLY", counts["research_only"],
          "characterized but **not** simulated"),
         ("FUTURE", counts["future"], "named as planned simulator work")]))
    parts.append("\n\n" + _md_table(
        ["Fraud surface", "Total", "Simulated"],
        [(c, d["total"], d["IMPLEMENTED"] + d["PARAMETERIZED"])
         for c, d in sorted(tax.coverage_by_category().items(), key=lambda kv: -kv[1]["total"])]))
    parts.append(f"\n\n{counts['auth_time_hard']} of the {counts['total_attacks']} catalogued "
                 "attacks have low or no visibility at authorization time. Saying so is the "
                 "point: an authorization-time model is the wrong control for most of them.\n")
    if figs.get("coverage"):
        parts.append(f"\n![Atlas coverage](docs/figures/{figs['coverage'].name})\n")

    if ops and "static_ml" in ops:
        o = ops["static_ml"]
        sc = o["scenario"]
        parts.append("\n---\n\n## What it costs to run\n\n"
                     f"Applied to a declared scenario of {sc['monthly_authorizations']:,} "
                     f"monthly authorizations:\n\n")
        parts.append(_md_table(["Measure", "Value"], [
            ("False positives per 1,000 genuine payments",
             f"{o['per_1000']['false_positives_per_1000_legit']:.1f}"),
            ("Genuine customers sent to step-up",
             pct(o["action_distribution"]["genuine_customers_stepped_up"], 2)),
            ("Genuine customers declined",
             pct(o["action_distribution"]["genuine_customers_declined"], 3)),
            ("Fraud sent to friction", pct(o["policy"]["fraud_to_friction"], 0)),
            ("Monthly step-up challenges", f"{sc['monthly_step_up_challenges']:,}"),
            ("Analyst-hours if every challenge were reviewed by hand",
             f"{sc['monthly_review_hours_if_all_manually_reviewed']:,}"),
        ]))
        parts.append(f"\n\n*{sc['label']}.* {o['disclaimer']}\n")

    parts.append(
        "\n---\n\n## Repository map\n\n```\n"
        "config.py                          seeds, archetypes, spec bounds, gates, paths\n"
        "\n"
        "src/identify/    IDENTIFY\n"
        "  attacks.json                     the Threat Atlas catalog\n"
        "  taxonomy.py                      loader + schema validation\n"
        "\n"
        "src/generate/    GENERATE\n"
        "  attack_spec.py                   AttackSpec + payment-domain constraint layer\n"
        "  profiles.py                      customer archetypes, merchant profiles\n"
        "  base_generator.py                the legitimate transaction stream\n"
        "  attack_injectors.py              one injector per simulated family\n"
        "  simulate.py                      orchestration; assembles the dataset\n"
        "  fidelity.py                      shortcut-learning diagnostics\n"
        "  llm_agent.py                     red-team agent: specifications + content\n"
        "  demo_specs.py                    live GenAI demonstration (needs a key)\n"
        "\n"
        "src/defend/      DEFEND\n"
        "  features.py                      authorization-time feature contract\n"
        "  model.py                         gradient boosting + isolation forest + calibration\n"
        "  train.py                         chronological split and training\n"
        "  evaluate.py                      metrics, per-family recall, Wilson intervals\n"
        "  baseline.py                      rules baseline + matched-budget comparison\n"
        "  decision_policy.py               approve / step-up / decline + reason codes\n"
        "  diagnostics.py                   thresholds, calibration, operations, blind spots\n"
        "  governance.py                    champion/challenger gates + model registry\n"
        "\n"
        "src/loop/        ADAPT\n"
        "  weakness.py                      signal attribution + mutation proposals\n"
        "  redteam_loop.py                  the closed loop\n"
        "\n"
        "src/experiments/\n"
        "  leave_one_out.py                 unseen -> learned experiment\n"
        "\n"
        "src/llm/client.py                  provider-agnostic client with a prompt-hash cache\n"
        "src/pipeline.py                    end-to-end regeneration, eight steps\n"
        "\n"
        "app/                               Streamlit prototype, eight pages\n"
        "docs/build_docs.py                 regenerates this README, figures and guides\n"
        "docs/build_walkthrough.py          regenerates the .docx submission document\n"
        "tests/                             leakage, shortcut, behaviour and consistency tests\n"
        "models/  data/                     committed artifacts (see Data model)\n"
        "```\n")

    # ---- documentation index ----------------------------------------------
    parts.append(
        "\n---\n\n## Documentation\n\n"
        "| Document | Read it for |\n|---|---|\n"
        "| [Architecture](docs/ARCHITECTURE.md) | Module-by-module structure, runtime paths, "
        "the artifact contract |\n"
        "| [Design](docs/DESIGN.md) | Why the system is shaped this way; the key abstractions "
        "and what each solves |\n"
        "| [Decisions](docs/DECISIONS.md) | The significant engineering decisions, the evidence "
        "behind them, and their consequences |\n"
        "| [Data model](docs/DATA_MODEL.md) | Transaction schema, feature contract, catalog "
        "schema, artifact reference |\n"
        "| [Experiments](docs/EXPERIMENTS.md) | How each experiment works, how to run it, and "
        "how to read the result |\n"
        "| [Contributing](docs/CONTRIBUTING.md) | Setup, the development loop, and recipes for "
        "extending the system |\n"
        "| [Security and ethics](docs/SECURITY_AND_ETHICS.md) | Responsible-use posture and what "
        "this project refuses to do |\n"
        "| [Glossary](docs/GLOSSARY.md) | Payments and machine-learning terms, and the "
        "project's own vocabulary |\n\n"
        "**For the demo:** [90-second script](docs/DEMO_SCRIPT_90S.md) · "
        "[3-minute pitch](docs/PITCH_3MIN.md) · [Judge Q&A](docs/JUDGE_QA.md) · "
        "[Operations guide](docs/DEMO_GUIDE.md) · "
        "[Screenshots](docs/screenshots/README.md) · "
        "[Submission checklist](docs/SUBMISSION_CHECKLIST.md)\n\n"
        "**The submission document:** `docs/solution_walkthrough.docx`, generated from the same "
        "artifacts as this README.\n")

    parts.append("\n---\n\n## Limitations\n\n"
                 "- **Synthetic only.** No real cardholder data. Every number is a simulation "
                 "result, and a real deployment needs a labelled backtest on issuer data "
                 "before any of it means anything.\n"
                 "- **Some fraud is not observable at authorization time.** First-party abuse "
                 "and victim-authorized scams are intentionally hard here and reported with "
                 "low recall; they belong to behavioural monitoring and post-transaction "
                 "intervention.\n"
                 "- **Residual frontiers remain.** Attacks that sit on the legitimate "
                 "behavioural centroid are not solved; the loop surfaces them rather than "
                 "hiding them.\n"
                 "- **The text arm is a sanity check, not a result, and is never used as "
                 "evidence of detection efficacy.** Its corpus is composed from a fixed slot "
                 "vocabulary, so the two classes separate on vocabulary alone and the score is "
                 "trivially high. It measures the corpus, not the detector, and would not "
                 "survive contact with real scam messages. Every detection claim in this "
                 "project rests on the transaction model alone.\n"
                 "- **No shadow rollout, drift monitoring or analyst feedback loop** — all "
                 "three are required in production and none is simulated.\n"
                 "- **Nothing here has been reviewed or validated by Mastercard.**\n")

    parts.append(
        "\n---\n\n## Reproducibility\n\n"
        f"Everything flows from `config.GLOBAL_SEED = {config.GLOBAL_SEED}` (schema version "
        f"{config.SCHEMA_VERSION}). The simulator, the split, the model, the loop and every "
        "evaluation frame are seeded, so `python -m src.pipeline` reproduces every artifact "
        "from scratch. Language-model calls are cached by prompt hash, and the default path "
        "needs no key at all, so a committed run is stable and the application runs "
        "network-free.\n\n"
        "Documentation is generated, not written. `python -m docs.build_docs` regenerates this "
        "README, the figures, the guides and the `.docx` walkthrough directly from "
        "`models/*.json` and `data/summary.json`. **No figure in any of them is typed by "
        "hand**, which is what makes it structurally impossible for two documents to disagree "
        "about the same number. `tests/test_artifact_consistency.py` fails the build if they "
        "ever do.\n")

    parts.append(
        "\n---\n\n## Licence and attribution\n\n"
        "Built for the Mastercard Innovation Challenge 2026 at the Global Fintech Fest, "
        "Mumbai. All data is synthetic and generated by the code in this repository. No real "
        "cardholder data is used anywhere. Nothing in this project has been reviewed, validated "
        "or endorsed by Mastercard, and no part of it describes any real payment-network "
        "system.\n\n"
        "See [Security and ethics](docs/SECURITY_AND_ETHICS.md) for the responsible-use "
        "position, including what this project deliberately does not do.\n")
    return "".join(parts)


def main():
    A = load_all()
    figs = build_figures(A)
    readme = build_readme(A, figs)
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    print(f"README.md written ({len(readme):,} chars)")
    for k, v in figs.items():
        print(f"  figure {k}: {'—' if v is None else v.name}")

    from docs.build_guides import build_guides
    for p in build_guides(A):
        print(f"  guide: {p.name}")

    from docs.build_walkthrough import build_walkthrough
    out = build_walkthrough(A, figs)
    print(f"Walkthrough written: {out}")


if __name__ == "__main__":
    main()
