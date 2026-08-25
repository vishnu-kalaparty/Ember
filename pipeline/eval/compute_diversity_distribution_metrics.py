#!/usr/bin/env python3
"""Compute Wasserstein distance and Jensen-Shannon divergence for RAG vs G_r diversity metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon

ROOT = Path(__file__).resolve().parents[1]
SRC_JSON = ROOT / "results" / "test_diversity_rag_vs_gr.json"
OUT_JSON = ROOT / "results" / "test_diversity_distribution_metrics.json"

METRIC_FAMILY = [
    ("cr", "CR (gzip)", False),
    ("cr_pos", "CR:POS", False),
    ("ngd", "NGD (n=1..4)", True),
    ("srs", "SRS (n=4)", True),
    ("srs_normalized", "SRS / (L-3)", True),
    ("self_bleu_loo", "Self-BLEU, LOO", True),
    ("self_bleu_intra_set", "Self-BLEU, intra-set", True),
    ("words_per_block", "Words / block", None),
    ("words_per_action", "Words / action", None),
    ("vp_count", "Verb phrases / report", None),
]

PAPER_KEYS = {"srs", "self_bleu_loo", "self_bleu_intra_set"}


def _is_valid(v) -> bool:
    if v is None:
        return False
    try:
        return not math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def _clean(values: list) -> np.ndarray:
    return np.array([float(v) for v in values if _is_valid(v)], dtype=float)


def _paired(values_a: list, values_b: list) -> tuple[np.ndarray, np.ndarray]:
    a_out, b_out = [], []
    for a, b in zip(values_a, values_b):
        if _is_valid(a) and _is_valid(b):
            a_out.append(float(a))
            b_out.append(float(b))
    return np.array(a_out, dtype=float), np.array(b_out, dtype=float)


def _js_divergence(a: np.ndarray, b: np.ndarray, n_bins: int = 50) -> dict:
    """JS divergence on normalized histograms over shared bin edges."""
    if len(a) < 2 or len(b) < 2:
        return {"js_divergence": None, "n_bins": n_bins, "method": "histogram"}

    combined = np.concatenate([a, b])
    lo, hi = float(combined.min()), float(combined.max())
    if lo == hi:
        return {
            "js_divergence": 0.0,
            "n_bins": 1,
            "method": "histogram (degenerate: all values equal)",
        }

    edges = np.histogram_bin_edges(combined, bins=n_bins)
    if len(edges) < 2:
        edges = np.linspace(lo, hi, n_bins + 1)

    hist_a, _ = np.histogram(a, bins=edges, density=False)
    hist_b, _ = np.histogram(b, bins=edges, density=False)

    # Laplace smoothing avoids log(0) in JS when a bin is empty in one group.
    hist_a = hist_a.astype(float) + 1e-12
    hist_b = hist_b.astype(float) + 1e-12
    p = hist_a / hist_a.sum()
    q = hist_b / hist_b.sum()

    js = float(jensenshannon(p, q, base=2) ** 2)
    return {
        "js_divergence": js,
        "n_bins": len(edges) - 1,
        "method": "normalized histogram on shared Freedman-Diaconis bin edges (numpy.histogram_bin_edges); JS = jensenshannon(p, q, base=2)**2 in [0, 1]",
    }


def main() -> None:
    data = json.loads(SRC_JSON.read_text(encoding="utf-8"))
    rag_per = data["per_report"]["rag"]
    gr_per = data["per_report"]["g_r"]

    results = []
    for key, label, _ in METRIC_FAMILY:
        rag_vals = rag_per[key]
        gr_vals = gr_per[key]

        rag_all = _clean(rag_vals)
        gr_all = _clean(gr_vals)
        rag_paired, gr_paired = _paired(rag_vals, gr_vals)

        # Paired t-test (report-aligned).
        if len(rag_paired) >= 2:
            t_stat, p_val = stats.ttest_rel(rag_paired, gr_paired)
            t_stat, p_val = float(t_stat), float(p_val)
        else:
            t_stat, p_val = None, None

        # Wasserstein on all valid per-report values (marginal distributions).
        if len(rag_all) >= 1 and len(gr_all) >= 1:
            w_dist = float(stats.wasserstein_distance(rag_all, gr_all))
        else:
            w_dist = None

        js_info = _js_divergence(rag_all, gr_all)
        js_div = js_info.pop("js_divergence")

        results.append(
            {
                "key": key,
                "label": label,
                "in_paper": key in PAPER_KEYS,
                "n_rag": int(len(rag_all)),
                "n_gr": int(len(gr_all)),
                "n_pairs": int(len(rag_paired)),
                "mean_rag": float(rag_all.mean()) if len(rag_all) else None,
                "mean_gr": float(gr_all.mean()) if len(gr_all) else None,
                "mean_diff_rag_minus_gr": (
                    float(rag_paired.mean() - gr_paired.mean()) if len(rag_paired) else None
                ),
                "t_paired": t_stat,
                "p_paired": p_val,
                "wasserstein_distance": w_dist,
                "js_divergence": js_div,
                "js_method": js_info,
            }
        )

    out = {
        "comparison": "Human (G_r) vs RAG",
        "source": str(SRC_JSON.relative_to(ROOT.parent)),
        "n_reports": data.get("n_reports", len(data.get("bids", []))),
        "methodology": {
            "paired_t_test": "scipy.stats.ttest_rel on report-aligned pairs; None/NaN dropped pairwise",
            "wasserstein_distance": "scipy.stats.wasserstein_distance on all valid per-report values per group (marginal 1D empirical distributions; unequal n allowed)",
            "js_divergence": js_info.get("method") if results else None,
            "interpretation": {
                "wasserstein_distance": "Earth Mover's Distance in the metric's native units. Higher = more mass must be transported to align RAG and Human histograms (larger distributional shift).",
                "js_divergence": "Symmetric divergence in [0, 1] (base-2 JS). 0 = identical binned distributions; 1 = maximally different. Higher = greater distributional dissimilarity regardless of location shift vs shape difference.",
                "paired_t_test": "Tests whether mean per-report difference (RAG − Human) differs from 0. Significant p (< 0.05) = systematic shift; does not measure effect size of distributional overlap.",
            },
        },
        "paper_metrics": sorted(PAPER_KEYS),
        "results": results,
    }

    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
