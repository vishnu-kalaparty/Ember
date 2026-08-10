"""
Cache-aware helper: run the train-indexed RAG system for one report at one
hyperparameter setting and return its generated actions.

Cache keys: `{embedder}_{alpha}_{topk}_{breach_id}`, e.g. `minilm_0.25_10_952`,
stored in `data/cache/{split}_actions_cache.json`.
"""

from __future__ import annotations

from .. import config
from ..labeling import common
from ..rag import generate, retrieve


def _cache_path(split_name):
    return config.CACHE_DIR / f"{split_name}_actions_cache.json"


def format_alpha(alpha) -> str:
    """0.0 / 0.25 / 0.5 / 0.75 / 1.0 (never 0.50 or 1)."""
    a = float(alpha)
    s = f"{a:.2f}".rstrip("0").rstrip(".")
    if "." not in s:
        s += ".0"
    return s


def hp_key(embedder, alpha, topk, breach_id) -> str:
    return f"{embedder}_{format_alpha(alpha)}_{topk}_{breach_id}"


def get_actions(
    split_name,
    breach_id,
    query_text,
    topk,
    cache=None,
    persist=True,
    cache_only=False,
    embedder="minilm",
    alpha=1.0,
    query_embedding=None,
):
    """Return (actions, cache). `actions` is [] on a retrieval miss."""
    cache_path = _cache_path(split_name)
    if cache is None:
        cache = common.load_json_or_empty(cache_path)

    key = hp_key(embedder, alpha, topk, breach_id)
    if key in cache:
        return cache[key]["actions"], cache

    if cache_only:
        raise RuntimeError(
            f"cache miss for {key!r} (split={split_name}). Fill it with "
            f"`python -m pipeline.eval.batch_rag submit --split {split_name} ...`."
        )

    selected, pool = retrieve.retrieve(
        query_text,
        topk,
        embedder=embedder,
        alpha=alpha,
        query_embedding=query_embedding,
        breach_id=breach_id,
    )
    if not selected:
        actions = []
    else:
        raw = generate.generate_actions(query_text, selected, topk=topk)
        actions = generate.split_action_lines(raw)

    cache[key] = {
        "actions": actions,
        "n_sentences_in_pool": len(pool),
        "n_reports_retrieved": len({r["breach_id"] for r in selected}),
        "embedder": embedder,
        "alpha": float(alpha),
    }

    if persist:
        common.save_json_atomic(cache_path, cache)

    return actions, cache


def flush_cache(split_name, cache):
    common.save_json_atomic(_cache_path(split_name), cache)


def load_cache(split_name):
    return common.load_json_or_empty(_cache_path(split_name))
