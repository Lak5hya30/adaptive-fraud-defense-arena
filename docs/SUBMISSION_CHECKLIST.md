# Submission checklist

Tick these against the repository, not against memory.

## Repository

- [ ] `python -m pip install -r requirements.txt` succeeds on a clean environment
- [ ] `python -m pytest tests/ -q` passes
- [ ] `python -m src.pipeline` regenerates every artifact from the seed
- [ ] `python -m docs.build_docs` regenerates README, figures and the walkthrough
- [ ] `streamlit run app/Home.py` opens in Demo mode with no API key and no network
- [ ] Committed artifacts are newer than the code that produced them
- [ ] No secrets, no API keys, no absolute local paths (`python -m pytest tests/ -q -k secret`
      and the scan in `docs/DEMO_GUIDE.md`)
- [ ] `.env` is git-ignored; only `.env.example` is committed
- [ ] No stray scratch files, notebooks, or `.log` files

## Numbers agree everywhere

The README, the walkthrough, the judge Q&A and the app all read the same
`models/*.json`. After any regeneration:

- [ ] `python -m docs.build_docs` has been re-run
- [ ] The README results table matches `models/metrics.json`
- [ ] `models/baseline_comparison.json` and `models/metrics.json` agree on static ML
- [ ] The adaptive column is the **promoted champion**, with the unpromoted candidate
      shown separately and labelled
- [ ] Attack counts in the README, the atlas page and `attacks.json` agree
- [ ] Every per-family recall quoted anywhere carries its sample size

## Honesty review

- [ ] No claim of validation on real payment data
- [ ] No implication of Mastercard review, endorsement or data
- [ ] Leave-one-out labelled as unseen → learned, never as zero-shot
- [ ] Residual frontiers visible in the app and in the walkthrough
- [ ] Rejected challengers reported as rejected
- [ ] The text detector's near-perfect score carries its "this measures the corpus,
      not the detector" caveat
- [ ] Catalog entries labelled RESEARCH_ONLY are never counted as simulated

## Walkthrough (.docx)

- [ ] Regenerated from artifacts, not hand-edited
- [ ] Cover, executive summary, problem, solution, three pillars, experiments,
      hero result, residual frontiers, feasibility, limitations, future work
- [ ] Figures render; captions present; page numbers in the footer
- [ ] Document properties set (title, subject, category)

## Web prototype

- [ ] Every page loads in Demo mode without touching the model
- [ ] No page triggers training or simulation on load
- [ ] Theme pinned (`.streamlit/config.toml`) so cards render on any machine
- [ ] Hero Demo readable in under two minutes
- [ ] Live mode clearly gated behind a confirmation

## Demo readiness

- [ ] `docs/DEMO_SCRIPT_90S.md` rehearsed end to end
- [ ] `docs/JUDGE_QA.md` read once immediately before presenting
- [ ] Screenshots in `docs/screenshots/` current
- [ ] Fallback path confirmed: the walkthrough contains the same figures and numbers

## Submission form

- [ ] GitHub repository URL
- [ ] Prototype URL (or the local run command, if not hosted)
- [ ] `docs/solution_walkthrough.docx` uploaded
- [ ] Screenshots attached
- [ ] Team details and deadline confirmed
