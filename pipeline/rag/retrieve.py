"""
Retrieval against the train-only indexes.

  - alpha=1.0: dense cosine only, top-k reports after max-pooling sentences
  - alpha=0.0: BM25 only (embedder ignored)
  - else:      hybrid alpha * cosine_minmax + (1 - alpha) * BM25_minmax
"""

from __future__ import annotations

from .. import config
from . import hybrid
from .index import get_collection

_COLLECTIONS = {}
_REPORTS_BY_ID = None


def _get_cached_collection(embedder="minilm"):
    if embedder not in _COLLECTIONS:
        _COLLECTIONS[embedder] = get_collection(create=False, embedder=embedder)
    return _COLLECTIONS[embedder]


def _reports_by_id():
    global _REPORTS_BY_ID
    if _REPORTS_BY_ID is None:
        from ..data_prep.load_data import load_reports

        _REPORTS_BY_ID = {int(r["breach_id"]): r for r in load_reports()}
    return _REPORTS_BY_ID


def _as_bid(x):
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return x


def _select_top_reports(sentence_results, topk):
    by_id = {}
    report_order = []
    for r in sentence_results:
        bid = r["breach_id"]
        if bid not in by_id:
            by_id[bid] = {"best_score": r["score"], "sentences": []}
            report_order.append(bid)
        by_id[bid]["sentences"].append(r)
        by_id[bid]["best_score"] = max(by_id[bid]["best_score"], r["score"])

    report_order.sort(key=lambda b: by_id[b]["best_score"], reverse=True)
    if len(report_order) > topk:
        report_order = report_order[:topk]

    selected = []
    for bid in report_order:
        sents = sorted(by_id[bid]["sentences"], key=lambda r: r["score"], reverse=True)
        selected.extend(sents)
    return selected


def _dense_query(
    query_text,
    embedder,
    collection,
    query_embedding=None,
    breach_id=None,
):
    n_candidates = min(config.RETRIEVE_N_CANDIDATES, max(collection.count(), 1))
    kwargs = {"n_results": n_candidates, "include": ["documents", "metadatas", "distances"]}
    if query_embedding is not None:
        vec = query_embedding.tolist() if hasattr(query_embedding, "tolist") else list(query_embedding)
        kwargs["query_embeddings"] = [vec]
    elif embedder == "openai3large":
        from . import openai_embed

        vec = openai_embed.get_val_query_embedding(breach_id, query_text=query_text)
        kwargs["query_embeddings"] = [vec.tolist()]
    else:
        kwargs["query_texts"] = [query_text]

    raw = collection.query(**kwargs)
    ids = raw["ids"][0]
    docs = raw["documents"][0]
    metas = raw["metadatas"][0]
    dists = raw["distances"][0]

    results = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        similarity = 1.0 - float(dist)
        meta = dict(meta or {})
        full_report = meta.pop("full_report", "")
        bid = _as_bid(meta.pop("breach_id", None))
        sentence_index = meta.pop("sentence_index", None)
        results.append(
            {
                "chunk_id": cid,
                "breach_id": bid,
                "sentence_index": sentence_index,
                "score": round(similarity, 4),
                "text": doc,
                "full_report": full_report,
                "metadata": meta,
            }
        )
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def _attach_report_fields(rows, fused_score):
    lookup = _reports_by_id()
    out = []
    for r in rows:
        item = dict(r)
        bid = _as_bid(item["breach_id"])
        item["breach_id"] = bid
        item["score"] = round(float(fused_score), 4)
        report = lookup.get(bid)
        if report is not None:
            item["full_report"] = item.get("full_report") or report["text"]
            item["metadata"] = item.get("metadata") or report["metadata"]
        out.append(item)
    return out


def retrieve(
    query_text,
    topk,
    collection=None,
    embedder="minilm",
    alpha=1.0,
    query_embedding=None,
    breach_id=None,
):
    """Return (selected_for_llm, dense_pool)."""
    alpha = float(alpha)
    embedder = embedder or "minilm"

    # Pure BM25: embedder does not affect ranking or context.
    if abs(alpha) < 1e-12:
        from . import bm25_index

        bm25_by_bid, bm25_sents = bm25_index.report_and_sentence_scores(query_text)
        ranked = sorted(bm25_by_bid, key=bm25_by_bid.get, reverse=True)[:topk]
        selected = []
        for bid in ranked:
            rows = _attach_report_fields(bm25_sents.get(bid, []), bm25_by_bid[bid])
            selected.extend(rows)
        return selected, []

    collection = collection or _get_cached_collection(embedder)
    dense_pool = _dense_query(
        query_text,
        embedder,
        collection,
        query_embedding=query_embedding,
        breach_id=breach_id,
    )

    # Dense only: cosine scores.
    if abs(alpha - 1.0) < 1e-12:
        selected = _select_top_reports(dense_pool, topk)
        return selected, dense_pool

    from . import bm25_index

    cosine_by_bid = {}
    dense_sents_by_bid = {}
    for r in dense_pool:
        bid = r["breach_id"]
        dense_sents_by_bid.setdefault(bid, []).append(r)
        cosine_by_bid[bid] = max(cosine_by_bid.get(bid, -1.0), r["score"])

    bm25_by_bid, bm25_sents = bm25_index.report_and_sentence_scores(query_text)
    fused = hybrid.fuse_report_scores(cosine_by_bid, bm25_by_bid, alpha)
    ranked = sorted(fused, key=fused.get, reverse=True)[:topk]

    selected = []
    for bid in ranked:
        if bid in dense_sents_by_bid:
            rows = dense_sents_by_bid[bid]
        else:
            rows = bm25_sents.get(bid, [])
        selected.extend(_attach_report_fields(rows, fused[bid]))
    return selected, dense_pool

