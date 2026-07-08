"""
Stratified temporal 80/10/10 split.

Per `Type of Breach` stratum, sort by `Breach Submission Date` ascending and
cut: oldest TRAIN_FRAC -> train, next VAL_FRAC -> val, newest remainder ->
test. Strata smaller than MIN_GROUP_SIZE_TO_SPLIT go entirely to train (too
few rows to split reliably).

Run:
    python -m pipeline.data_prep.split_data
"""

import json
from collections import defaultdict

from .. import config
from .load_data import load_reports


def make_split(reports):
    by_type = defaultdict(list)
    for r in reports:
        by_type[r["type"]].append(r)

    train_ids, val_ids, test_ids = [], [], []
    per_type_summary = {}

    for t, group in by_type.items():
        group = sorted(group, key=lambda r: r["breach_id"])  # already date-sorted
        n = len(group)
        if n < config.MIN_GROUP_SIZE_TO_SPLIT:
            train_ids.extend(r["breach_id"] for r in group)
            per_type_summary[t] = {"n": n, "train": n, "val": 0, "test": 0, "note": "too small to split"}
            continue

        n_train = round(n * config.TRAIN_FRAC)
        n_val = round(n * config.VAL_FRAC)
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        n_test = n - n_train - n_val

        train_ids.extend(r["breach_id"] for r in group[:n_train])
        val_ids.extend(r["breach_id"] for r in group[n_train : n_train + n_val])
        test_ids.extend(r["breach_id"] for r in group[n_train + n_val :])
        per_type_summary[t] = {"n": n, "train": n_train, "val": n_val, "test": n_test}

    return train_ids, val_ids, test_ids, per_type_summary


def run():
    reports = load_reports()
    train_ids, val_ids, test_ids, summary = make_split(reports)

    with open(config.TRAIN_IDS_JSON, "w") as fh:
        json.dump(sorted(train_ids), fh)
    with open(config.VAL_IDS_JSON, "w") as fh:
        json.dump(sorted(val_ids), fh)
    with open(config.TEST_IDS_JSON, "w") as fh:
        json.dump(sorted(test_ids), fh)
    with open(config.SPLIT_SUMMARY_JSON, "w") as fh:
        json.dump(
            {
                "total_reports": len(reports),
                "train": len(train_ids),
                "val": len(val_ids),
                "test": len(test_ids),
                "per_type": summary,
            },
            fh,
            indent=2,
        )

    print(f"Total non-empty reports : {len(reports)}")
    print(f"train : {len(train_ids)}")
    print(f"val   : {len(val_ids)}")
    print(f"test  : {len(test_ids)}")
    print(f"Saved -> {config.SPLITS_DIR}")
    return train_ids, val_ids, test_ids


if __name__ == "__main__":
    run()
