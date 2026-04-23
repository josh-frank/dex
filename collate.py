#!/usr/bin/env python3
"""
collate.py
Merges all annotation JSONL files in ./annotations/ into dataset.csv.
Run after every session: python3 collate.py
"""

import json
import csv
import pathlib
import sys

ANNOTATIONS_DIR = pathlib.Path("annotations")
OUTPUT_CSV      = pathlib.Path("dataset.csv")

FIELDS = ["session", "t0", "t1", "amplitude", "attack_s", "release_s", "baseline_uS", "label"]

def load_all():
    rows = []
    skipped = 0
    for path in sorted(ANNOTATIONS_DIR.glob("*.jsonl")):
        text = path.read_text().strip()
        if not text:
            continue
        try:
            rec = json.loads(text)
        except json.JSONDecodeError:
            print(f"  [skip] bad JSON: {path.name}")
            skipped += 1
            continue

        features = rec.get("features") or {}
        amplitude   = features.get("amplitude")
        attack_s    = features.get("attack_s")
        release_s   = features.get("release_s")
        baseline_uS = features.get("baseline_uS")

        # skip records with missing features
        if any(v is None for v in [amplitude, attack_s, release_s, baseline_uS]):
            print(f"  [skip] missing features: {path.name}")
            skipped += 1
            continue

        rows.append({
            "session":      rec.get("session", ""),
            "t0":           rec.get("t0", ""),
            "t1":           rec.get("t1", ""),
            "amplitude":    amplitude,
            "attack_s":     attack_s,
            "release_s":    release_s,
            "baseline_uS":  baseline_uS,
            "label":        1 if rec.get("read") else 0,
        })

    return rows, skipped

def main():
    if not ANNOTATIONS_DIR.exists():
        print(f"[error] {ANNOTATIONS_DIR}/ not found — point this script at your annotations folder")
        sys.exit(1)

    rows, skipped = load_all()

    if not rows:
        print("[error] no valid records found")
        sys.exit(1)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    reads     = sum(r["label"] == 1 for r in rows)
    artifacts = sum(r["label"] == 0 for r in rows)

    print(f"[collate] {len(rows)} records → {OUTPUT_CSV}")
    print(f"          reads: {reads}  |  artifacts/other: {artifacts}  |  skipped: {skipped}")

if __name__ == "__main__":
    main()
