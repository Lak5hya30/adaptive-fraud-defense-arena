# Screenshots

Six images for the submission form, and the fallback if the live app fails on stage.

**These are the only artifact in this project not produced by a command** — they need
a real browser window, so they have to be captured by hand.

## Setup

```bash
streamlit run app/Home.py
```

Serves on `http://localhost:8501` by default. Browser at **1440×900**, zoom **90%**
so each metric row fits on one line, sidebar collapsed, **Demo mode** (the default —
do not touch the Live toggle).

The artifacts are already current; do not regenerate the pipeline before capturing.

## The six captures

Streamlit derives each route from the page filename, dropping the numeric prefix.

| # | File to save | Route | Page file | Frame it on |
|---|---|---|---|---|
| 1 | `01_home.png` | `/` | `app/Home.py` | The proof-point card and the closed-loop diagram below it. Scroll so both are in frame. |
| 2 | `02_threat_atlas.png` | `/Threat_Atlas` | `app/pages/1_Threat_Atlas.py` | The five metric tiles and the **Coverage map — research surface vs simulator** text bars directly under them. |
| 3 | `03_constraint_layer.png` | `/Generate` | `app/pages/2_Generate.py` | Tab **🧬 Attack specifications**, scrolled to *The constraint layer, live*. Set family `scam_transfer`, device `new_device`, geography `high_risk`, intensity `1.60` — capture with the three corrections visible. |
| 4 | `04_hero_blind_spot.png` | `/HeroDemo` | `app/pages/5_HeroDemo.py` | Beat **③ The blind spot**. Shows the unseen-family recall with its confidence interval, and the transaction the stale defense approved. |
| 5 | `05_hero_learned.png` | `/HeroDemo` | `app/pages/5_HeroDemo.py` | Beat **⑤ The defense learns**. The before/after metric pair with its interval, plus the per-family chart underneath. |
| 6 | `06_closed_loop.png` | `/ClosedLoop` | `app/pages/4_ClosedLoop.py` | Tab **🔁 Rounds**, scrolled to **Round 1** — the three family cards plus the `PROMOTE` decision and the forgetting-check line. |

Two routes are not in the list but are worth having if the submission form allows more:
`/Defend` (tab *🎯 Recall by family*, for the confidence-interval bars) and `/Benchmarks`
(the matched-false-positive-budget table).

Remaining routes, for reference: `/Deployment` → `app/pages/7_Deployment.py`.

## Already generated — use these for charts

`docs/figures/*.png` are regenerated from the committed artifacts by
`python -m docs.build_docs` and are embedded in the walkthrough:

| File | Shows |
|---|---|
| `atlas_coverage.png` | research surface vs simulator, by fraud category |
| `fidelity_separability.png` | per-feature separability and overlap |
| `benchmark_matched_fpr.png` | rules vs static vs adaptive at a matched budget |
| `family_recall.png` | recall by family with 95% Wilson intervals |
| `leave_one_out.png` | unseen vs learned, with n per family |
| `threshold_sweep.png` | the detection / friction trade-off curve |
| `attack_lineage.png` | recall per attack generation, stale vs after replay |

Prefer these over screenshots wherever a chart will do — they are higher resolution
and they cannot go stale, because regenerating the pipeline regenerates them.
