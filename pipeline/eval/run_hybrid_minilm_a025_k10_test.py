"""
Test-set scores at the frozen hyperparameters: Hybrid MiniLM α=0.25, k=10.

Scores the RAG actions for every test report with a nonempty sentence-level
G_r (BERTScore P/R/F1 + MoverScore, see ``sentence_gr.score_system``).

Run from repo root (after the frozen hybrid test batch is in cache):
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
      .venv/bin/python -m pipeline.eval.run_hybrid_minilm_a025_k10_test
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pipeline import config
from pipeline.eval import batch_rag, run_rag_actions
from pipeline.eval.sentence_gr import score_system

EMBEDDER = "minilm"
ALPHA = 0.25
TOPK = 10
SENTENCE_GR_JSON = config.SENTENCE_GR_JSON


def _load_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def _fmt_row(name, m):
    n = m.get("n_scored", 0)
    return (
        f"| {name} | {m['bertscore_precision']:.4f} | {m['bertscore_recall']:.4f} | "
        f"{m['bertscore_f1']:.4f} | {m['moverscore']:.4f} | {n} |"
    )


def _eligible():
    eligible, _ = batch_rag.sentence_gr_eligible_ids("test")
    sentence_gr = _load_json(SENTENCE_GR_JSON)
    return [str(bid) for bid in eligible], sentence_gr


def collect_preds(eligible, rag_cache, top_n):
    rag_actions = {}
    missing_rag = []
    empty_rag = []
    for bid_s in eligible:
        key = run_rag_actions.hp_key(EMBEDDER, ALPHA, TOPK, bid_s)
        entry = rag_cache.get(key)
        if entry is None:
            missing_rag.append(bid_s)
            continue
        acts = [a for a in (entry.get("actions") or []) if a][:top_n]
        if not acts:
            empty_rag.append(bid_s)
        else:
            rag_actions[bid_s] = acts
    return {"rag": rag_actions, "missing_rag": missing_rag, "empty_rag": empty_rag}


def _print_table(rag_agg):
    print("\n## Test vs sentence G_r (frozen HP: Hybrid MiniLM α=0.25, k=10)\n")
    print("| System | P | R | F1 | MoverScore | n |")
    print("|---|---:|---:|---:|---:|---:|")
    print(_fmt_row("RAG Hybrid MiniLM α=0.25 k=10", rag_agg))


def score_and_write():
    top_n = config.SUGGESTED_ACTION_COUNT
    out_json = config.TEST_HYBRID_JSON

    eligible, sentence_gr = _eligible()
    print(f"Test eligible (nonempty sentence G_r): {len(eligible)}")
    print(f"G_r path: {SENTENCE_GR_JSON}")
    print(
        f"RAG keys: {EMBEDDER}_{run_rag_actions.format_alpha(ALPHA)}_{TOPK}_{{bid}}"
    )

    rag_cache = run_rag_actions.load_cache("test")
    preds = collect_preds(eligible, rag_cache, top_n)

    if preds["missing_rag"]:
        raise RuntimeError(
            f"{len(preds['missing_rag'])} eligible reports missing RAG cache "
            f"keys (do not resubmit; fetch first): {preds['missing_rag'][:10]}"
        )
    if preds["empty_rag"]:
        print(f"WARNING: {len(preds['empty_rag'])} RAG empty-action reports dropped")

    rag_label = "RAG Hybrid MiniLM α=0.25 k=10"
    rag_per, rag_agg = score_system(rag_label, dict(preds["rag"]), sentence_gr)
    n_scored = sum(1 for v in rag_per.values() if v.get("scored"))

    out = {
        "methodology": {
            "retrieval": "hybrid MiniLM + BM25",
            "alpha": ALPHA,
            "embedder": EMBEDDER,
            "topk": TOPK,
            "top_n": top_n,
            "hybrid_score": "0.25 * cosine + 0.75 * BM25",
            "n_test_eligible_sentence_gr": len(eligible),
            "n_scored": n_scored,
            "n_rag_missing": len(preds["missing_rag"]),
            "n_rag_empty": len(preds["empty_rag"]),
            "g_r_path": str(SENTENCE_GR_JSON),
            "g_r_definition": (
                "Full texts of 3-LLM-intersection agreed positive remediation sentences"
            ),
            "metrics": "report-local G_r BERTScore P/R/F1 + MoverScore",
            "predictions": (
                "raw LLM action strings, cache key minilm_0.25_10_{bid}"
            ),
        },
        "comparison_vs_sentence_gr": {"rag_hybrid_minilm_a025_k10": rag_agg},
        "per_report": {"rag_hybrid_minilm_a025_k10": rag_per},
    }
    _save_json(out_json, out)
    _print_table(rag_agg)
    print(f"\nWrote {out_json}")
    return out


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    score_and_write()


if __name__ == "__main__":
    main()
