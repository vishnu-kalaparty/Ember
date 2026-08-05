"""Per-query min-max fusion of dense cosine and BM25 report scores.

    hybrid = α * cosine' + (1-α) * BM25'

Missing channel values are 0 *before* min-max. If max==min, that channel
is all zeros. α=1 is dense-only (callers should skip this module);
α=0 is BM25-only.
"""

from __future__ import annotations


def minmax(values: dict) -> dict:
    if not values:
        return {}
    xs = list(values.values())
    lo, hi = min(xs), max(xs)
    if hi == lo:
        return {k: 0.0 for k in values}
    scale = hi - lo
    return {k: (v - lo) / scale for k, v in values.items()}


def fuse_report_scores(cosine_by_bid: dict, bm25_by_bid: dict, alpha: float) -> dict:
    """Return {breach_id: hybrid} over the union of both channels."""
    alpha = float(alpha)
    union = set(cosine_by_bid) | set(bm25_by_bid)
    cosine = {b: float(cosine_by_bid.get(b, 0.0)) for b in union}
    bm25 = {b: float(bm25_by_bid.get(b, 0.0)) for b in union}
    cosine_n = minmax(cosine)
    bm25_n = minmax(bm25)
    return {
        b: alpha * cosine_n[b] + (1.0 - alpha) * bm25_n[b] for b in union
    }
