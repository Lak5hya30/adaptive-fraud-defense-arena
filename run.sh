#!/usr/bin/env bash
# Build all artifacts, then launch the demo.
#   ./run.sh            # regenerate everything (incl. closed loop), rebuild docs, launch
#   ./run.sh --fast     # regenerate everything except the slow closed loop
#   ./run.sh --launch   # just launch the app from the committed artifacts
set -e
cd "$(dirname "$0")"

if [ "$1" == "--launch" ]; then
  exec streamlit run app/Home.py
fi

echo "==> Regenerating artifacts (deterministic, seeded)"
python -m src.pipeline "$@"

echo "==> Regenerating README, figures and the walkthrough from those artifacts"
python -m docs.build_docs

echo "==> Launching the prototype at http://localhost:8501 (Demo mode)"
streamlit run app/Home.py
