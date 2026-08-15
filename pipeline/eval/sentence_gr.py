"""
Sentence-level ground truth G_r and report-level scoring against it.

G_r for a report = the full texts of its 3-LLM-agreed remediation sentences
(``agreed_action_sentences.json`` IDs looked up in ``report_sentences.json``).
``score_system`` scores each report's generated actions against its own G_r
with BERTScore P/R/F1 and MoverScore, then averages over reports.

Build G_r once from repo root:
    .venv/bin/python -m pipeline.eval.sentence_gr
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pipeline import config
from pipeline.eval import bertscore_utils
from pipeline.labeling import common

TOP_N = 5


def _load_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def build_sentence_gr(out_path: Path | None = None) -> dict:
    """Map breach_id -> list of agreed remediation sentence texts (no spaCy)."""
    out_path = out_path or config.SENTENCE_GR_JSON
    agreed = _load_json(config.AGREED_ACTIONS_JSON)
    report_sentences = common.load_report_sentences()

    sentence_gr = {}
    total_sents = 0
    for bid, sentence_ids in agreed.items():
        if not sentence_ids:
            continue
        sentences = report_sentences[bid]["sentences"]
        texts = []
        for sid in sentence_ids:
            if not (1 <= sid <= len(sentences)):
                continue
            text = (sentences[sid - 1] or "").strip()
            if text:
                texts.append(text)
        if texts:
            sentence_gr[bid] = texts
            total_sents += len(texts)

    _save_json(out_path, sentence_gr)
    print(f"Reports with agreed action sentences : {sum(1 for v in agreed.values() if v)}")
    print(f"Reports with >=1 sentence G_r        : {len(sentence_gr)}")
    print(f"Total remediation sentences (G_r)    : {total_sents}")
    print(f"Saved -> {out_path}")
    return sentence_gr


def aggregate(per_report):
    scored = [v for v in per_report.values() if v.get("scored")]
    if not scored:
        return {"n_scored": 0}
    return {
        "n_scored": len(scored),
        "bertscore_precision": float(np.mean([x["precision"] for x in scored])),
        "bertscore_recall": float(np.mean([x["recall"] for x in scored])),
        "bertscore_f1": float(np.mean([x["f1"] for x in scored])),
        "moverscore": float(np.mean([x["moverscore"] for x in scored])),
        "n_empty": sum(1 for v in per_report.values() if not v.get("scored")),
        "mean_n_preds": float(np.mean([len(x.get("actions") or []) for x in scored])),
        "mean_n_g_r": float(np.mean([x.get("n_g_r", 0) for x in scored])),
    }


def score_system(name, actions_by_bid, sentence_gr):
    print(f"Scoring {name} on {len(actions_by_bid)} reports...")
    per_report = {}
    for i, (bid_s, actions) in enumerate(actions_by_bid.items()):
        if (i + 1) % 50 == 0:
            print(f"  {name} {i + 1}/{len(actions_by_bid)}")
        to_score = [a for a in (actions or []) if a][:TOP_N]
        g_r = sentence_gr.get(bid_s) or []
        if not to_score:
            per_report[bid_s] = {"scored": False, "reason": "no_actions"}
            continue
        if not g_r:
            per_report[bid_s] = {"scored": False, "reason": "empty_sentence_g_r"}
            continue
        scored = bertscore_utils.score_actions_vs_report_local_g(to_score, g_r)
        if scored is None:
            per_report[bid_s] = {"scored": False, "reason": "unscoreable"}
            continue
        per_report[bid_s] = {
            "scored": True,
            "actions": to_score,
            "n_g_r": scored["n_g_r"],
            "precision": scored["precision"],
            "recall": scored["recall"],
            "f1": scored["f1"],
            "moverscore": scored["moverscore"],
        }
    return per_report, aggregate(per_report)


if __name__ == "__main__":
    build_sentence_gr()
