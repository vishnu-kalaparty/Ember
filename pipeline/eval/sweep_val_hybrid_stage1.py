"""
Stage 1 hybrid HP sweep vs **sentence-level G_r** (no spaCy VPs).

Grid: embedder × α at k=10 on validation.
Selection: mean BERTScore F1. MoverScore is computed on the same encodings.

Run (after batch_rag fetch --stage 1):
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
      .venv/bin/python -m pipeline.eval.sweep_val_hybrid_stage1
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from .. import config
from ..labeling import common
from . import run_rag_actions
from . import scoring

SENTENCE_GR_JSON = config.SENTENCE_GR_JSON

_GR_TOK: dict[str, np.ndarray] = {}
_ACT_TOK: dict[str, np.ndarray] = {}


def _encode(texts, cache):
    missing = [t for t in dict.fromkeys(texts) if t not in cache]
    if missing:
        for t, arr in zip(missing, scoring.encode_tokens(missing)):
            cache[t] = arr
    return [cache[t] for t in texts]


def _score(actions, g_phrases):
    a_toks = _encode(actions, _ACT_TOK)
    g_toks = _encode(g_phrases, _GR_TOK)
    ps, rs, f1s, movers = [], [], [], []
    for at in a_toks:
        p, r, f1 = scoring.best_bertscore_prf(at, g_toks)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        movers.append(scoring.best_moverscore(at, g_toks))
    return (
        float(np.mean(ps)),
        float(np.mean(rs)),
        float(np.mean(f1s)),
        float(np.mean(movers)),
    )


def _load_sentence_gr():
    if SENTENCE_GR_JSON.exists():
        return common.load_json_or_empty(SENTENCE_GR_JSON)
    raise RuntimeError(f"Missing sentence G_r at {SENTENCE_GR_JSON}")


def _eligible():
    val_ids = json.load(open(config.VAL_IDS_JSON))
    sentence_gr = _load_sentence_gr()
    report_sentences = common.load_report_sentences()
    eligible = [
        bid
        for bid in val_ids
        if str(bid) in sentence_gr
        and sentence_gr[str(bid)]
        and str(bid) in report_sentences
    ]
    return eligible, sentence_gr


def _markdown_table(grid):
    lines = [
        "| Embedder | α | k | n | P | R | F1 | MoverScore |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in grid:
        lines.append(
            f"| {r['embedder']} | {r['alpha']:.2g} | {r['topk']} | "
            f"{r['n_reports_scored']} | "
            f"{r['mean_bertscore_precision']:.4f} | "
            f"{r['mean_bertscore_recall']:.4f} | "
            f"{r['mean_bertscore_f1']:.4f} | "
            f"{r['mean_moverscore']:.4f} |"
        )
    return "\n".join(lines)


def run():
    out_json = config.HP_STAGE1_SWEEP_JSON
    eligible, sentence_gr = _eligible()
    rag_cache = run_rag_actions.load_cache("val")
    grid = [
        (embedder, alpha, config.SWEEP_STAGE1_TOPK)
        for embedder in config.SWEEP_EMBEDDERS
        for alpha in config.SWEEP_ALPHAS
    ]

    print(f"eligible val reports (non-empty sentence G_r): {len(eligible)}")
    print(f"reference: {SENTENCE_GR_JSON}")
    print(f"cells: {len(grid)} (embedder × α at k={config.SWEEP_STAGE1_TOPK})")

    grid_results = []
    per_report_all = {}
    t0 = time.time()

    for embedder, alpha, tk in grid:
        _ACT_TOK.clear()
        ps, rs, f1s, movers = [], [], [], []
        n_empty = 0
        empty_ids = []
        cell = {}
        for bid in eligible:
            key = run_rag_actions.hp_key(embedder, alpha, tk, bid)
            entry = rag_cache.get(key) or {}
            actions = entry.get("actions") or []
            if not actions:
                n_empty += 1
                empty_ids.append(int(bid))
                cell[str(bid)] = {"scored": False, "empty": True}
                continue
            p, r, f1, mover = _score(actions, sentence_gr[str(bid)])
            ps.append(p)
            rs.append(r)
            f1s.append(f1)
            movers.append(mover)
            cell[str(bid)] = {
                "scored": True,
                "precision": p,
                "recall": r,
                "f1": f1,
                "moverscore": mover,
                "n_actions": len(actions),
                "n_g_r": len(sentence_gr[str(bid)]),
            }

        if n_empty:
            raise RuntimeError(
                f"Incomplete cell {embedder} α={alpha} k={tk}: "
                f"{n_empty} empty-action reports (e.g. {empty_ids[:5]}). "
                "Fetch the missing generations before scoring."
            )

        rec = {
            "embedder": embedder,
            "alpha": float(alpha),
            "topk": tk,
            "n_reports_scored": len(f1s),
            "n_empty": n_empty,
            "mean_bertscore_precision": float(np.mean(ps)),
            "mean_bertscore_recall": float(np.mean(rs)),
            "mean_bertscore_f1": float(np.mean(f1s)),
            "mean_moverscore": float(np.mean(movers)),
        }
        grid_results.append(rec)
        per_report_all[f"{embedder}_{run_rag_actions.format_alpha(alpha)}_{tk}"] = cell
        print(
            f"{embedder:13s} α={run_rag_actions.format_alpha(alpha):4s} k={tk:>2} "
            f"n={rec['n_reports_scored']:>4} "
            f"F1={rec['mean_bertscore_f1']:.6f} "
            f"P={rec['mean_bertscore_precision']:.6f} "
            f"R={rec['mean_bertscore_recall']:.6f} "
            f"Mover={rec['mean_moverscore']:.6f}  [{time.time() - t0:.0f}s]",
            flush=True,
        )

    ranked = sorted(grid_results, key=lambda r: r["mean_bertscore_f1"], reverse=True)
    best = dict(ranked[0])
    best["selection_metric"] = "bertscore_f1"
    f1_values = [r["mean_bertscore_f1"] for r in grid_results]
    payload = {
        "n_val_reports_eligible": len(eligible),
        "selection_metric": "bertscore_f1",
        "g_r_definition": (
            "Full texts of 3-LLM-intersection agreed positive remediation "
            "sentences (no spaCy VP extraction)"
        ),
        "g_r_path": str(SENTENCE_GR_JSON),
        "k": config.SWEEP_STAGE1_TOPK,
        "grid_f1_spread": [min(f1_values), max(f1_values)],
        "best": best,
        "grid": grid_results,
        "per_report": per_report_all,
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2)

    print("\n## Stage 1 (sentence G_r, k=10)\n")
    print(_markdown_table(grid_results))
    print(
        f"\n**Winner:** `{best['embedder']}` α={best['alpha']} k={best['topk']}  "
        f"F1={best['mean_bertscore_f1']:.4f}  "
        f"P={best['mean_bertscore_precision']:.4f}  "
        f"R={best['mean_bertscore_recall']:.4f}  "
        f"Mover={best['mean_moverscore']:.4f}"
    )
    print(f"Spread F1: {max(f1_values) - min(f1_values):.6f}")
    print(f"Saved -> {out_json}")
    return payload


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()


if __name__ == "__main__":
    main()
