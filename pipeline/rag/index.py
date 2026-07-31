"""
Build a dedicated ChromaDB vector store from TRAIN-split sentence chunks only
(val + test breach_ids are never indexed — enforced by only ever loading
config.TRAIN_IDS_JSON here).

Run:
    python -m pipeline.rag.index --embedder minilm     # existing MiniLM (skip if present)
    python -m pipeline.rag.index --embedder openai3large
    python -m pipeline.rag.index --embedder minilm --rebuild  # DANGER: wipes MiniLM
"""

from __future__ import annotations

import argparse
import json

import chromadb
from chromadb.utils import embedding_functions

from .. import config
from ..data_prep.build_report_sentences import split_sentences
from ..data_prep.load_data import load_reports


def _client():
    config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.VECTORSTORE_DIR))


def _minilm_embedding_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL
    )


def _spec(embedder: str) -> dict:
    if embedder not in config.EMBEDDERS:
        raise ValueError(f"unknown embedder {embedder!r}; choose from {list(config.EMBEDDERS)}")
    return config.EMBEDDERS[embedder]


def get_collection(create=False, embedder="minilm"):
    """Return the Chroma collection for `embedder`.

    create=True deletes *that embedder's* collection only. MiniLM
    (`ember_train_minilm`) is never deleted when building OpenAI.
    """
    spec = _spec(embedder)
    client = _client()
    name = spec["collection"]
    if embedder == "minilm":
        ef = _minilm_embedding_fn()
        if create:
            try:
                client.delete_collection(name)
            except Exception:
                pass
            return client.create_collection(
                name=name,
                embedding_function=ef,
                metadata={"hnsw:space": config.CHROMA_DISTANCE_SPACE},
            )
        return client.get_collection(name=name, embedding_function=ef)

    # OpenAI (and any future precomputed-embedding collections): no EF on the
    # collection so query_texts cannot silently re-embed.
    if create:
        if name == config.COLLECTION_NAME:
            raise RuntimeError("refusing to delete the MiniLM collection")
        try:
            client.delete_collection(name)
        except Exception:
            pass
        return client.create_collection(
            name=name,
            metadata={"hnsw:space": config.CHROMA_DISTANCE_SPACE},
        )
    return client.get_collection(name=name)


def collection_exists(embedder: str) -> bool:
    spec = _spec(embedder)
    client = _client()
    try:
        names = {c.name for c in client.list_collections()}
    except Exception:
        names = set()
    return spec["collection"] in names


def _is_good(text, sentences):
    if config.DUPLICATE_MARKER in text.lower():
        return False
    return config.MIN_SENTENCES <= len(sentences) <= config.MAX_SENTENCES


def _chunk_report(report):
    sentences = split_sentences(report["text"])
    if config.INDEX_APPLY_GOOD_FILTER and not _is_good(report["text"], sentences):
        return []
    chunks = []
    for sent_idx, sent in enumerate(sentences):
        if len(sent.split()) < config.MIN_CHUNK_WORDS:
            continue
        chunks.append(
            {
                "chunk_id": f"{report['breach_id']}.{sent_idx}",
                "breach_id": report["breach_id"],
                "sentence_index": sent_idx,
                "text": sent,
                "full_report": report["text"],
                "metadata": report["metadata"],
            }
        )
    return chunks


def iter_train_chunks():
    train_ids = set(json.load(open(config.TRAIN_IDS_JSON)))
    reports = load_reports()
    chunks = []
    indexed_reports = 0
    for report in reports:
        if report["breach_id"] not in train_ids:
            continue
        rc = _chunk_report(report)
        if rc:
            indexed_reports += 1
            chunks.extend(rc)
    return chunks, indexed_reports, len(train_ids)


def _ingest(collection, chunks, embeddings=None, batch_size=500):
    n = len(chunks)
    for start in range(0, n, batch_size):
        batch = chunks[start : start + batch_size]
        kwargs = {
            "ids": [c["chunk_id"] for c in batch],
            "documents": [c["text"] for c in batch],
            "metadatas": [
                {
                    "breach_id": c["breach_id"],
                    "sentence_index": c["sentence_index"],
                    "full_report": c["full_report"],
                    **c["metadata"],
                }
                for c in batch
            ],
        }
        if embeddings is not None:
            kwargs["embeddings"] = [
                e.tolist() if hasattr(e, "tolist") else list(e)
                for e in embeddings[start : start + batch_size]
            ]
        collection.add(**kwargs)
        print(f"Ingested {min(start + batch_size, n)}/{n}")


def build(embedder="minilm", batch_size=500, rebuild=False):
    chunks, indexed_reports, n_train_ids = iter_train_chunks()
    print(f"Embedder                : {embedder}")
    print(f"Train reports available : {n_train_ids}")
    print(f"Train reports indexed   : {indexed_reports}")
    print(f"Sentence chunks         : {len(chunks)}")

    if embedder == "minilm":
        if collection_exists(embedder) and not rebuild:
            col = get_collection(create=False, embedder=embedder)
            print(
                f"MiniLM collection '{config.EMBEDDERS[embedder]['collection']}' "
                f"already exists (count={col.count()}). Skipping rebuild. "
                "Pass --rebuild to wipe and recreate (not recommended)."
            )
            return col
        if rebuild:
            print("WARNING: --rebuild will DELETE the MiniLM collection.")
        collection = get_collection(create=True, embedder=embedder)
        _ingest(collection, chunks, embeddings=None, batch_size=batch_size)
        print(
            f"Done. Collection '{config.EMBEDDERS[embedder]['collection']}' "
            f"count = {collection.count()}"
        )
        return collection

    if embedder == "openai3large":
        from . import openai_embed

        if collection_exists(embedder) and not rebuild:
            col = get_collection(create=False, embedder=embedder)
            if col.count() == len(chunks):
                print(
                    f"OpenAI collection already complete (count={col.count()}). "
                    "Skipping rebuild. Pass --rebuild to recreate (MiniLM is never deleted)."
                )
                return col
            print(
                f"OpenAI collection exists but count={col.count()} != {len(chunks)}; rebuilding it."
            )

        print("Computing / loading cached OpenAI train embeddings…")
        embs = openai_embed.ensure_train_embeddings(chunks)
        collection = get_collection(create=True, embedder=embedder)
        _ingest(collection, chunks, embeddings=embs, batch_size=batch_size)
        print(
            f"Done. Collection '{config.EMBEDDERS[embedder]['collection']}' "
            f"count = {collection.count()}  (MiniLM collection untouched)"
        )

        from ..labeling import common
        from . import openai_embed as _oe

        val_ids = json.load(open(config.VAL_IDS_JSON))
        report_vps = common.load_json_or_empty(config.REPORT_VPS_JSON)
        report_sentences = common.load_report_sentences()
        eligible = [
            bid
            for bid in val_ids
            if str(bid) in report_vps
            and report_vps[str(bid)]
            and str(bid) in report_sentences
        ]
        print("Caching OpenAI embeddings for val first-sentence queries…")
        _oe.ensure_val_query_embeddings(eligible, report_sentences)
        return collection

    raise ValueError(f"no build path for embedder={embedder!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a train-only Chroma collection for one embedder."
    )
    parser.add_argument(
        "--embedder",
        choices=list(config.EMBEDDERS),
        required=True,
        help="Which collection to build. openai3large never deletes MiniLM.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and recreate THIS embedder's collection only.",
    )
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    build(embedder=args.embedder, batch_size=args.batch_size, rebuild=args.rebuild)
