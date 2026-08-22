# Demo operations guide

Everything needed to run the prototype in front of people. For the words to say,
see [`DEMO_SCRIPT_90S.md`](DEMO_SCRIPT_90S.md); for the hard questions, see
[`JUDGE_QA.md`](JUDGE_QA.md).

---

## Before you leave for the venue

```bash
python -m pip install -r requirements.txt
python -m pytest tests/ -q            # everything green
streamlit run app/Home.py             # opens on http://localhost:8501
```

Click through all seven pages once with the laptop **in aeroplane mode**. Demo mode
reads only committed artifacts, so if a page needs the network something has been
mis-wired and you want to find that at home.

Confirm the theme is dark (`.streamlit/config.toml` pins it). The cards use light
text on a translucent panel; under a light theme they render white-on-white.

## On stage

1. Start the app **before** you are introduced. First load compiles the page cache.
2. Leave the sidebar toggle on **Demo mode**. Live mode retrains and takes minutes.
3. Full screen, browser zoom around 90% so the metric rows fit on one line.
4. Follow the 90-second script. If you have longer, the deep-tech route is:
   Threat Atlas coverage map → Generate constraint layer → Defend blind spots →
   Closed Loop lineage and promotion gates → Benchmarks matched-FPR table →
   Deployment.

## If something breaks

| Symptom | Do this |
|---|---|
| A page shows "run `python -m ...`" | An artifact is missing. Switch to the walkthrough; do not regenerate on stage. |
| Streamlit will not start | Open `docs/solution_walkthrough.docx` — same figures, same numbers. |
| A chart renders blank | Screenshots in `docs/screenshots/` cover every page. |
| Asked for something not on screen | Every number lives in `models/*.json`; open the file rather than guessing. |

Nothing in the demo path needs an API key, a network connection, or training.

## Regenerating everything

```bash
python -m src.pipeline          # dataset → fidelity → model → text → loop → baseline → diagnostics → LOAO
python -m src.pipeline --fast   # everything except the closed loop
python -m docs.build_docs       # README, figures, guides and walkthrough, from those artifacts
```

The full pipeline takes tens of minutes. `--fast` is enough for everything except
the Closed Loop page.

**Always run `python -m docs.build_docs` after regenerating.** The README, the
walkthrough and the judge Q&A are generated from the artifacts; skipping this step
is the one way to reintroduce a number that disagrees with the app.

## Pre-flight checks

```bash
python -m pytest tests/ -q                                  # behaviour and anti-shortcut tests
python -m src.generate.fidelity                             # simulator diagnostics
python -m src.identify.taxonomy                             # catalog counts and coverage
python -m src.defend.baseline                               # rules vs static vs adaptive
python -m src.defend.diagnostics                            # thresholds, calibration, blind spots
```

If `src.generate.fidelity` reports any FAIL, the synthetic data has developed a
shortcut and **no metric downstream of it is trustworthy** — fix that before
presenting anything.

## Screenshots

`docs/screenshots/` holds one image per page, captured from the running prototype
in Demo mode. Refresh them by hand whenever the artifacts change materially — start
the app, visit each page, and capture at 1440×900 with the browser at 90% zoom so
the metric rows sit on one line.

They exist for two reasons: the submission form wants images, and they are the
fallback if the live app fails on stage.
