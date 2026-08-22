"""Text arm — a trivially separable synthetic sanity check, NOT a detection result.

A lightweight TF-IDF + logistic-regression model over the attack-artifact corpus,
included to show where content signals would attach to the architecture. Its
corpus is composed from a fixed slot vocabulary, so the two classes separate on
vocabulary alone and the score is high for that reason and no other. It measures
the corpus, not the detector, and would not survive contact with real scam
messages.

**Nothing in this module is evidence of detection efficacy, and its score is
never used as such. Every detection claim in this project rests on the
transaction model alone.**

CLI:  python -m src.defend.text_model
"""
from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import Pipeline

import config

# One wording, used by the artifact, the CLI, the app and the docs, so the score
# can never appear anywhere without it. It leads with the separability fact:
# describing the hard-negative construction first would read as "we made it hard
# and still scored 1.0", which is the opposite of what the number means.
CAVEAT = (
    "A trivially separable synthetic sanity check, not evidence of detection capability. "
    "The corpus is composed from a fixed slot vocabulary, so the two classes separate on "
    "vocabulary alone. The benign class does deliberately include genuine security notices "
    "sharing the urgency, OTP and payment vocabulary of scam messages, and provenance markers "
    "and the attack's catalog name are stripped before training — but those measures only "
    "remove the crudest shortcuts; they do not make the task realistic. Every detection claim "
    "in this project rests on the transaction model alone.")

TRIVIAL_READING = (
    "This corpus is composed from a fixed slot vocabulary, so the two classes are almost "
    "perfectly separable by vocabulary alone. The near-perfect score measures the CORPUS, not "
    "the detector: it would not survive contact with real scam messages. The text arm is "
    "included to show where content signals attach to the architecture, and its score is "
    "reported here precisely so it is never mistaken for a headline result.")

INDICATIVE_READING = (
    "Scores are cross-validated over the whole corpus; the corpus is small and synthetic, so "
    "treat the estimate as indicative only and never as evidence of detection efficacy.")


def load_corpus(path=config.ARTIFACTS_JSONL) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.generate.simulate` first.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def train_text_model(save: bool = True) -> tuple[Pipeline, dict]:
    corpus = load_corpus()
    texts = [c["text"] for c in corpus]
    labels = np.array([c["label"] for c in corpus])

    stratify = labels if len(set(labels)) > 1 and min(np.bincount(labels)) >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(
        texts, labels, test_size=0.3, random_state=config.GLOBAL_SEED, stratify=stratify)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000,
                                  sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   random_state=config.GLOBAL_SEED)),
    ]).fit(Xtr, ytr)

    proba = pipe.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # A single 30% split of a corpus this small is a very noisy estimate, so the
    # headline number is a cross-validated one over the whole corpus.
    cv_auc = None
    if stratify is not None and min(np.bincount(labels)) >= 5:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.GLOBAL_SEED)
        cv_proba = cross_val_predict(
            Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                                                max_features=5000, sublinear_tf=True)),
                      ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                                 random_state=config.GLOBAL_SEED))]),
            texts, labels, cv=cv, method="predict_proba")[:, 1]
        cv_auc = float(roc_auc_score(labels, cv_proba))

    metrics = {
        "n_corpus": len(corpus),
        "n_fraud": int(labels.sum()),
        "n_benign": int((labels == 0).sum()),
        "roc_auc": float(roc_auc_score(yte, proba)) if len(set(yte)) > 1 else None,
        "roc_auc_cv5": cv_auc,
        "report": classification_report(yte, pred, output_dict=True, zero_division=0),
        "caveat": CAVEAT,
    }
    headline = cv_auc if cv_auc is not None else metrics["roc_auc"]
    metrics["is_sanity_check_not_evidence"] = True
    metrics["trivially_separable"] = bool(headline is not None and headline > 0.99)
    metrics["honest_reading"] = (TRIVIAL_READING if metrics["trivially_separable"]
                                 else INDICATIVE_READING)
    if save:
        joblib.dump(pipe, config.TEXT_MODEL_PATH)
        (config.MODELS_DIR / "text_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8")
    return pipe, metrics


def score_text(text: str, pipe: Pipeline | None = None) -> float:
    pipe = pipe or joblib.load(config.TEXT_MODEL_PATH)
    return float(pipe.predict_proba([text])[:, 1][0])


def main():
    _, metrics = train_text_model()
    print(json.dumps({k: metrics[k] for k in
                      ("n_corpus", "n_fraud", "n_benign", "roc_auc", "roc_auc_cv5",
                       "trivially_separable", "is_sanity_check_not_evidence")}, indent=2))
    # Printed unconditionally: the terminal output must not be screenshottable
    # without the caveat that explains what the score does and does not mean.
    print("\n" + metrics["honest_reading"])
    print(f"\nText model saved to {config.TEXT_MODEL_PATH}")


if __name__ == "__main__":
    main()
