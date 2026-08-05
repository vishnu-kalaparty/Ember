"""
Sentence-level BM25 over the same train chunks as the dense indexes,
then max-pooled to reports (mirrors cosine aggregation).

Run:
    python -m pipeline.rag.bm25_index
"""

from __future__ import annotations

import pickle
import re

from rank_bm25 import BM25Okapi

from .. import config
from .index import iter_train_chunks

TOKEN_RE = re.compile(r"[a-z0-9]+")

_INDEX = None
_REPORTS_BY_ID = None


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def _payload_path():
    return config.BM25_INDEX_PATH


def build():
    chunks, _, _ = iter_train_chunks()
    tokenized = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    payload = {
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "breach_ids": [int(c["breach_id"]) for c in chunks],
        "texts": [c["text"] for c in chunks],
        "sentence_indexes": [c["sentence_index"] for c in chunks],
        "tokenized": tokenized,
        "bm25": bm25,
    }
    config.BM25_DIR.mkdir(parents=True, exist_ok=True)
    with open(_payload_path(), "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    n_reports = len(set(payload["breach_ids"]))
    print(
        f"BM25 index: {len(chunks)} sentences, {n_reports} reports -> {_payload_path()}"
    )
    return payload


def load(force=False):
    global _INDEX
    if _INDEX is not None and not force:
        return _INDEX
    path = _payload_path()
    if not path.exists():
        print("BM25 pickle missing — building…")
        _INDEX = build()
        return _INDEX
    with open(path, "rb") as fh:
        _INDEX = pickle.load(fh)
    return _INDEX


def reports_by_id():
    global _REPORTS_BY_ID
    if _REPORTS_BY_ID is None:
        from ..data_prep.load_data import load_reports

        _REPORTS_BY_ID = {int(r["breach_id"]): r for r in load_reports()}
    return _REPORTS_BY_ID


def report_and_sentence_scores(query_text, max_sents_per_report=5):
    """Return (report_max_scores, top sentences per report)."""
    idx = load()
    tokens = tokenize(query_text)
    if not tokens:
        scores = [0.0] * len(idx["chunk_ids"])
    else:
        scores = idx["bm25"].get_scores(tokens)

    report_max = {}
    sents_by_bid = {}
    for i, raw in enumerate(scores):
        bid = int(idx["breach_ids"][i])
        s = float(raw)
        if bid not in report_max or s > report_max[bid]:
            report_max[bid] = s
        sents_by_bid.setdefault(bid, []).append(
            {
                "chunk_id": idx["chunk_ids"][i],
                "breach_id": bid,
                "sentence_index": idx["sentence_indexes"][i],
                "score": round(s, 4),
                "text": idx["texts"][i],
            }
        )

    top_sents = {}
    for bid, rows in sents_by_bid.items():
        rows.sort(key=lambda r: r["score"], reverse=True)
        top_sents[bid] = rows[:max_sents_per_report]
    return report_max, top_sents


if __name__ == "__main__":
    build()
