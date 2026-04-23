#!/usr/bin/env python3
"""
train.py
Trains a random forest on dataset.csv and saves dex.joblib.
Run after collate.py: python3 train.py
"""

import pathlib
import sys
import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

DATASET_CSV  = pathlib.Path("dataset.csv")
MODEL_OUT    = pathlib.Path("dex.joblib")
FEATURES     = ["amplitude", "attack_s", "release_s", "baseline_uS"]
MIN_SAMPLES  = 10   # warn if below this

def main():
    if not DATASET_CSV.exists():
        print("[error] dataset.csv not found — run collate.py first")
        sys.exit(1)

    df = pd.read_csv(DATASET_CSV)
    print(f"[train] {len(df)} samples loaded")

    reads     = (df["label"] == 1).sum()
    artifacts = (df["label"] == 0).sum()
    print(f"        reads: {reads}  |  artifacts/other: {artifacts}")

    if len(df) < MIN_SAMPLES:
        print(f"[warn]  only {len(df)} samples — model will be very rough, keep labeling")

    if reads == 0 or artifacts == 0:
        print("[error] need at least one example of each class")
        sys.exit(1)

    X = df[FEATURES].values
    y = df["label"].values

    # Pipeline: scale → forest
    # StandardScaler helps when features have very different ranges
    # (baseline_uS in µS, attack_s in seconds, etc.)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("forest", RandomForestClassifier(
            n_estimators=200,
            max_depth=4,          # shallow — prevents overfitting on small data
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        ))
    ])

    # Leave-one-out CV — appropriate for small datasets
    if len(df) >= 6:
        loo    = LeaveOneOut()
        scores = cross_val_score(model, X, y, cv=loo, scoring="accuracy")
        print(f"[train] LOO accuracy: {scores.mean():.2f} ± {scores.std():.2f}  (n={len(scores)})")
    else:
        print("[train] too few samples for CV — skipping")

    # Fit on full dataset
    model.fit(X, y)

    # Feature importances
    forest     = model.named_steps["forest"]
    importances = zip(FEATURES, forest.feature_importances_)
    print("[train] feature importances:")
    for name, imp in sorted(importances, key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"        {name:<15} {imp:.3f}  {bar}")

    # Save
    meta = {
        "n_samples":  len(df),
        "n_reads":    int(reads),
        "n_artifacts": int(artifacts),
        "features":   FEATURES,
        "loo_accuracy": float(scores.mean()) if len(df) >= 6 else None,
    }
    joblib.dump({"model": model, "meta": meta}, MODEL_OUT)
    print(f"[train] saved → {MODEL_OUT}")
    print(f"[train] meta: {json.dumps(meta, indent=2)}")

if __name__ == "__main__":
    main()
