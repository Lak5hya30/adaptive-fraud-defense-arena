"""Train the defense model from the simulated dataset and persist model + metrics.

Uses a time-ordered split (train on earlier transactions, test on later ones) to
avoid leakage and mimic how a fraud model is deployed against future traffic.

CLI:  python -m src.defend.train
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config
from src.defend.evaluate import evaluate
from src.defend.features import build_xy
from src.defend.model import DefenseModel


def load_dataset(path=config.DATASET_PARQUET) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.generate.simulate` first.")
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def split_points(n: int, val_frac: float = 0.15, test_frac: float = 0.25):
    """Row indices of the chronological train / validation / test boundaries.

    One definition, used by every caller, so a split can never quietly differ
    between the trainer, the benchmark harness and the loop.
    """
    test_cut = int(n * (1 - test_frac))
    val_cut = int(n * (1 - test_frac - val_frac))
    return val_cut, test_cut


def time_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.25):
    """Chronological train / validation / test split of the raw frame (no shuffling).

    Prefer :func:`split_xy` for anything that trains or evaluates: it builds
    features once over the whole ordered frame before splitting, which is what a
    production system does and what keeps history features intact.
    """
    val_cut, test_cut = split_points(len(df), val_frac, test_frac)
    return (df.iloc[:val_cut].copy(), df.iloc[val_cut:test_cut].copy(),
            df.iloc[test_cut:].copy())


def split_xy(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.25):
    """Build features ONCE over the whole ordered frame, then split by time.

    Building features per slice would restart every card's history, every
    velocity window and every network counter at the slice boundary — which no
    production system does, and which quietly weakens exactly the historical
    features this defense relies on. Because every feature is computed from
    strictly-earlier rows, building globally and then splitting introduces no
    leakage: a test-set row still only ever sees its own past.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    X, y, atk = build_xy(df)
    n = len(df)
    val_cut, test_cut = split_points(n, val_frac, test_frac)
    sl = {
        "train": slice(0, val_cut),
        "val": slice(val_cut, test_cut),
        "test": slice(test_cut, n),
    }
    out = {}
    for name, s in sl.items():
        out[name] = (X.iloc[s], y[s], atk[s])
    out["frames"] = {"train": df.iloc[sl["train"]], "val": df.iloc[sl["val"]],
                     "test": df.iloc[sl["test"]]}
    return out


def train_and_eval(df: pd.DataFrame | None = None, val_frac: float = 0.15,
                   test_frac: float = 0.25, save: bool = True
                   ) -> tuple[DefenseModel, dict]:
    df = load_dataset() if df is None else df
    parts = split_xy(df, val_frac, test_frac)
    X_tr, y_tr, _ = parts["train"]
    X_val, y_val, _ = parts["val"]
    X_te, y_te, atk_te = parts["test"]
    train_df, val_df, test_df = (parts["frames"]["train"], parts["frames"]["val"],
                                 parts["frames"]["test"])

    model = DefenseModel().fit(X_tr, y_tr)
    # Calibrate and tune the threshold on a held-out validation slice: train-tuned
    # thresholds drift under time shift, and calibrating on training rows would
    # simply memorize them. Calibration is monotone, so it changes what the score
    # means, not which transactions outrank which.
    model.fit_calibration(X_val, y_val)
    model.tune_threshold(X_val, y_val, target_fpr=config.TARGET_MAX_FPR)

    metrics = evaluate(model, X_te, y_te, atk_te)
    metrics["schema_version"] = config.SCHEMA_VERSION
    metrics["seed"] = config.GLOBAL_SEED
    metrics["n_features"] = len(model.feature_columns)
    metrics["train_size"] = int(len(train_df))
    metrics["val_size"] = int(len(val_df))
    metrics["test_size"] = int(len(test_df))
    metrics["train_fraud"] = int(y_tr.sum())
    metrics["feature_importances"] = _perm_importance(model, X_te, y_te)

    if save:
        model.save()
        config.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return model, metrics


def _perm_importance(model: DefenseModel, X, y, n_repeats: int = 3) -> dict:
    """Lightweight permutation importance on fused scores (AUC drop)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(config.GLOBAL_SEED)
    base = roc_auc_score(y, model.fused_scores(X)) if len(np.unique(y)) > 1 else 0.5
    imp = {}
    for col in model.feature_columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            auc = roc_auc_score(y, model.fused_scores(Xp)) if len(np.unique(y)) > 1 else 0.5
            drops.append(base - auc)
        imp[col] = float(np.mean(drops))
    return dict(sorted(imp.items(), key=lambda kv: kv[1], reverse=True))


def main():
    model, metrics = train_and_eval()
    summary = {k: metrics[k] for k in
               ("precision", "recall", "f1", "roc_auc", "pr_auc",
                "false_positive_rate", "threshold")}
    print(json.dumps(summary, indent=2))
    print("\nPer-attack recall:")
    for atk, v in metrics["per_attack_recall"].items():
        print(f"  {atk:22s} recall={v['recall']:.3f}  (n={v['n']})")
    print(f"\nModel saved to {config.MODEL_PATH}")
    print(f"Metrics saved to {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
