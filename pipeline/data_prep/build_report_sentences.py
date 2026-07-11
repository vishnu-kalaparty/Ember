"""
Build the numbered-sentence view of every val+test report. This is the input
handed to all 3 labeling LLMs (they return sentence numbers, 1-indexed).

Run:
    python -m pipeline.data_prep.build_report_sentences
"""

import json
import re

from .. import config
from .load_data import load_reports

_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text):
    """EMBER sentence-split rules (pre-split, curly quotes, merge rules)."""
    if not text or not text.strip():
        return []

    for marker in config.PRESPLIT_BREAK_BEFORE:
        text = text.replace(marker, "\n" + marker.strip())
    if config.NORMALIZE_CURLY_QUOTES:
        text = text.replace("\u2019", "'")
    text = text.encode().decode("utf-8", "ignore")

    raw_parts = [p.strip() for p in _SPLIT_RE.split(text.strip()) if p.strip()]

    if not config.APPLY_MERGE_RULES:
        return raw_parts

    sentences = []
    for part in raw_parts:
        part = part.replace("\\", "").replace("\r\n", "").replace("\n", "").strip()
        if not part:
            continue
        if sentences and (
            sentences[-1].endswith(",") or part.startswith(",") or part[0].islower()
        ):
            sentences[-1] = sentences[-1] + " " + part
        else:
            sentences.append(part)
    return sentences


def run():
    reports = load_reports()
    by_id = {r["breach_id"]: r for r in reports}

    val_ids = json.load(open(config.VAL_IDS_JSON))
    test_ids = json.load(open(config.TEST_IDS_JSON))
    target_ids = sorted(set(val_ids) | set(test_ids))

    out = {}
    skipped = 0
    for bid in target_ids:
        r = by_id.get(bid)
        if not r:
            continue
        sents = split_sentences(r["text"])
        if not sents:
            skipped += 1
            continue
        out[str(bid)] = {
            "sentences": sents,
            "type": r["type"],
            "date": r["date"],
        }

    with open(config.REPORT_SENTENCES_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    n_sent = sum(len(v["sentences"]) for v in out.values())
    print(f"val+test reports        : {len(target_ids)}")
    print(f"skipped (no sentences)  : {skipped}")
    print(f"reports with sentences  : {len(out)}")
    print(f"total sentences         : {n_sent}")
    print(f"Saved -> {config.REPORT_SENTENCES_JSON}")


if __name__ == "__main__":
    run()
