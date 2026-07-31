"""
OpenAI text-embedding-3-large helpers with on-disk caches.

Train sentence embeddings are stored once (npz) and reused to build the
Chroma collection without re-calling the API. Val first-sentence query
embeddings are cached separately so retrieve() never re-embeds a grid cell.
"""

from __future__ import annotations

import json
import time

import numpy as np

from .. import config
from ..labeling import common

MODEL = "text-embedding-3-large"
EMBED_BATCH = 64
MAX_RETRIES = 6


def _client():
    if not config.OPENAI_API_KEY:
        raise RuntimeError(
            "OpenAI API key is not set. Set OPENAI_API_KEY in .env "
            "or export OPENAI_API_KEY."
        )
    from openai import OpenAI

    return OpenAI(api_key=config.OPENAI_API_KEY)


def embed_texts(texts, batch_size=EMBED_BATCH, label="texts"):
    """Return list[list[float]] aligned with `texts`."""
    if not texts:
        return []
    client = _client()
    out = []
    n = len(texts)
    for start in range(0, n, batch_size):
        batch = [t if (t and t.strip()) else " " for t in texts[start : start + batch_size]]
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.embeddings.create(model=MODEL, input=batch)
                ordered = sorted(resp.data, key=lambda d: d.index)
                out.extend([d.embedding for d in ordered])
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                wait = min(2 ** attempt, 30)
                print(
                    f"  embed retry {attempt + 1}/{MAX_RETRIES} after error: {exc} "
                    f"(sleep {wait}s)"
                )
                time.sleep(wait)
        if last_err is not None:
            raise RuntimeError(
                f"OpenAI embeddings failed at {label} {start}:{start + len(batch)}: {last_err}"
            )
        print(f"  embedded {min(start + batch_size, n)}/{n} {label}", flush=True)
    return out


def _load_npz_map(path, id_key, emb_key="embeddings"):
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    ids = [str(x) for x in data[id_key].tolist()]
    embs = data[emb_key]
    return {i: embs[j] for j, i in enumerate(ids)}


def _save_npz_map(path, mapping, id_key):
    if not mapping:
        return
    ids = list(mapping.keys())
    embs = np.stack([np.asarray(mapping[i], dtype=np.float32) for i in ids])
    np.savez_compressed(
        path,
        **{id_key: np.array(ids, dtype=object), "embeddings": embs},
    )


def ensure_train_embeddings(chunks, save_every=2048):
    """Return list of embeddings aligned with `chunks`. API only for missing ids."""
    cache = _load_npz_map(config.OPENAI_TRAIN_EMB_CACHE, "chunk_ids")
    missing_idx = [i for i, c in enumerate(chunks) if str(c["chunk_id"]) not in cache]
    print(
        f"OpenAI train embeddings: cached={len(cache)} missing={len(missing_idx)} "
        f"total={len(chunks)}"
    )
    if missing_idx:
        texts = [chunks[i]["text"] for i in missing_idx]
        vecs = embed_texts(texts, label="train sentences")
        for i, vec in zip(missing_idx, vecs):
            cache[str(chunks[i]["chunk_id"])] = np.asarray(vec, dtype=np.float32)
            if len(cache) % save_every == 0:
                _save_npz_map(config.OPENAI_TRAIN_EMB_CACHE, cache, "chunk_ids")
                print(f"  saved train embedding cache ({len(cache)} vectors)")
        _save_npz_map(config.OPENAI_TRAIN_EMB_CACHE, cache, "chunk_ids")
        print(f"Saved train embedding cache -> {config.OPENAI_TRAIN_EMB_CACHE}")
    return [np.asarray(cache[str(c["chunk_id"])], dtype=np.float32) for c in chunks]


def ensure_val_query_embeddings(eligible_ids, report_sentences):
    """Cache first-sentence embeddings for val eligible reports. Return {bid: vec}."""
    cache = _load_npz_map(config.OPENAI_VAL_QUERY_EMB_CACHE, "breach_ids")
    missing = [bid for bid in eligible_ids if str(bid) not in cache]
    print(
        f"OpenAI val query embeddings: cached={len(cache)} missing={len(missing)} "
        f"eligible={len(eligible_ids)}"
    )
    if missing:
        texts = [report_sentences[str(bid)]["sentences"][0] for bid in missing]
        vecs = embed_texts(texts, label="val queries")
        for bid, vec in zip(missing, vecs):
            cache[str(bid)] = np.asarray(vec, dtype=np.float32)
        _save_npz_map(config.OPENAI_VAL_QUERY_EMB_CACHE, cache, "breach_ids")
        print(f"Saved val query embedding cache -> {config.OPENAI_VAL_QUERY_EMB_CACHE}")
    return {int(k): np.asarray(v, dtype=np.float32) for k, v in cache.items()}


def get_val_query_embedding(breach_id, query_text=None):
    cache = _load_npz_map(config.OPENAI_VAL_QUERY_EMB_CACHE, "breach_ids")
    key = str(breach_id)
    if key in cache:
        return np.asarray(cache[key], dtype=np.float32)
    if not query_text:
        raise KeyError(f"no cached OpenAI query embedding for breach_id={breach_id}")
    vec = embed_texts([query_text], label=f"query {breach_id}")[0]
    cache[key] = np.asarray(vec, dtype=np.float32)
    _save_npz_map(config.OPENAI_VAL_QUERY_EMB_CACHE, cache, "breach_ids")
    return cache[key]


if __name__ == "__main__":
    import argparse

    from .index import iter_train_chunks

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["train", "val-queries", "both"],
        help="train=embed train chunks; val-queries=embed val first sentences",
    )
    args = parser.parse_args()
    if args.command in ("train", "both"):
        chunks, _, _ = iter_train_chunks()
        ensure_train_embeddings(chunks)
    if args.command in ("val-queries", "both"):
        ids_path = config.VAL_IDS_JSON
        ids = json.load(open(ids_path))
        report_vps = common.load_json_or_empty(config.REPORT_VPS_JSON)
        report_sentences = common.load_report_sentences()
        eligible = [
            bid
            for bid in ids
            if str(bid) in report_vps
            and report_vps[str(bid)]
            and str(bid) in report_sentences
        ]
        ensure_val_query_embeddings(eligible, report_sentences)
