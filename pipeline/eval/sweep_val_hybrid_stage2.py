"""
Stage 2 hybrid HP sweep vs **sentence-level G_r** (no spaCy VPs).

Grid: the Stage 1 setting (hybrid MiniLM, α=0.25) × k ∈ {5, 10, 15, 20} on
validation. Selection: mean BERTScore F1.

Run (after batch_rag fetch --stage 2):
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
      .venv/bin/python -m pipeline.eval.sweep_val_hybrid_stage2
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
from . import run_rag_actions
from .sweep_val_hybrid_stage1 import (
    _ACT_TOK,
    _GR_TOK,
    _eligible,
    _markdown_table,
    _score,
)

SENTENCE_GR_JSON = config.SENTENCE_GR_JSON
FLAT_F1_SPREAD = 0.002


def _score_cell(eligible, g_r, rag_cache, topk, embedder, alpha, require_full=True):
    _ACT_TOK.clear()
    ps, rs, f1s, movers = [], [], [], []
    n_empty = 0
    empty_ids = []
    cell = {}
    for bid in eligible:
        key = run_rag_actions.hp_key(embedder, alpha, topk, bid)
        entry = rag_cache.get(key) or {}
        actions = entry.get("actions") or []
        if not actions:
            n_empty += 1
            empty_ids.append(int(bid))
            cell[str(bid)] = {"scored": False, "empty": True}
            continue
        p, r, f1, mover = _score(actions, g_r[str(bid)])
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
            "n_g_r": len(g_r[str(bid)]),
        }

    if require_full and n_empty:
        raise RuntimeError(
            f"Incomplete cell {embedder} α={alpha} k={topk}: "
            f"{n_empty} empty-action reports (e.g. {empty_ids[:5]})."
        )

    rec = {
        "embedder": embedder,
        "alpha": float(alpha),
        "topk": topk,
        "n_reports_scored": len(f1s),
        "n_empty": n_empty,
        "mean_bertscore_precision": float(np.mean(ps)) if f1s else None,
        "mean_bertscore_recall": float(np.mean(rs)) if f1s else None,
        "mean_bertscore_f1": float(np.mean(f1s)) if f1s else None,
        "mean_moverscore": float(np.mean(movers)) if f1s else None,
    }
    return rec, cell



def run():
    out_json = config.HP_STAGE2_SWEEP_JSON
    embedder = config.STAGE2_EMBEDDER
    alpha = config.STAGE2_ALPHA
    ks = list(config.SWEEP_STAGE2_TOPKS)
    retrieval = "hybrid MiniLM + BM25"

    eligible, sentence_gr = _eligible()
    rag_cache = run_rag_actions.load_cache("val")

    print(f"eligible val reports (non-empty sentence G_r): {len(eligible)}")
    print(f"reference: {SENTENCE_GR_JSON}")
    print(f"cells: {len(ks)} ({retrieval} α={alpha} × k={ks})")

    grid_results = []
    per_report_all = {}
    t0 = time.time()

    for tk in ks:
        rec, cell = _score_cell(
            eligible, sentence_gr, rag_cache, tk, embedder, alpha
        )
        rec["retrieval"] = retrieval
        grid_results.append(rec)
        per_report_all[f"{embedder}_{run_rag_actions.format_alpha(alpha)}_{tk}"] = cell
        print(
            f"{retrieval} α={alpha} k={tk:>2} "
            f"n={rec['n_reports_scored']:>4} "
            f"F1={rec['mean_bertscore_f1']:.6f} "
            f"P={rec['mean_bertscore_precision']:.6f} "
            f"R={rec['mean_bertscore_recall']:.6f} "
            f"Mover={rec['mean_moverscore']:.6f}  [{time.time() - t0:.0f}s]",
            flush=True,
        )

    ranked = sorted(grid_results, key=lambda r: r["mean_bertscore_f1"], reverse=True)
    numerical_best = dict(ranked[0])
    f1_values = [r["mean_bertscore_f1"] for r in grid_results]
    spread = max(f1_values) - min(f1_values)
    is_flat = spread < FLAT_F1_SPREAD
    k10 = next(r for r in grid_results if r["topk"] == 10)
    chosen = dict(k10) if is_flat else dict(numerical_best)
    chosen["selection_metric"] = "bertscore_f1"
    chosen["flat_k_sweep"] = is_flat
    chosen["numerical_winner_k"] = numerical_best["topk"]
    if is_flat:
        chosen["selection_note"] = (
            f"Stage 2 F1 spread {spread:.6f} < {FLAT_F1_SPREAD}; "
            "freeze default depth k=10 (Stage 1 k) rather than the numerical winner"
        )
    else:
        chosen["selection_note"] = (
            f"Stage 2 F1 spread {spread:.6f}; freeze numerical winner k={chosen['topk']}"
        )

    frozen = {
        "retrieval": retrieval,
        "alpha": float(alpha),
        "embedder": embedder,
        "topk": chosen["topk"],
        "selection_split": "val",
        "selection_metric": "bertscore_f1",
        "g_r": "sentence",
        "mean_bertscore_f1": chosen["mean_bertscore_f1"],
        "flat_k_sweep": is_flat,
        "numerical_winner_k": numerical_best["topk"],
        "note": chosen["selection_note"],
    }


    payload = {
        "n_val_reports_eligible": len(eligible),
        "selection_metric": "bertscore_f1",
        "g_r_definition": (
            "Full texts of 3-LLM-intersection agreed positive remediation "
            "sentences (no spaCy VP extraction)"
        ),
        "g_r_path": str(SENTENCE_GR_JSON),
        "retrieval": retrieval,
        "alpha": float(alpha),
        "embedder": embedder,
        "ks": ks,
        "grid_f1_spread": [min(f1_values), max(f1_values)],
        "flat_k_sweep": is_flat,
        "flat_threshold": FLAT_F1_SPREAD,
        "best": chosen,
        "numerical_best": numerical_best,
        "frozen": frozen,
        "grid": grid_results,
        "per_report": per_report_all,
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\n## Stage 2 (sentence G_r, {retrieval})\n")
    print(_markdown_table(grid_results))
    print(
        f"\n**Frozen:** {retrieval}, k={frozen['topk']}, "
        f"α={frozen['alpha']}  "
        f"F1={frozen['mean_bertscore_f1']:.4f}"
    )
    print(
        f"Numerical winner k={numerical_best['topk']}  "
        f"F1={numerical_best['mean_bertscore_f1']:.4f}  "
        f"spread={spread:.6f}  flat={is_flat}"
    )
    print(chosen["selection_note"])
    print(f"Saved -> {out_json}")
    return payload


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()


if __name__ == "__main__":
    main()
