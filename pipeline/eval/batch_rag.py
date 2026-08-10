"""
Fill val/test RAG action caches via the OpenAI Batch API (async, ~50% cheaper,
separate from sync rate limits).

Hyperparameter search on validation:
    python -m pipeline.eval.batch_rag submit --split val --stage 1   # embedder x alpha at k=10
    python -m pipeline.eval.batch_rag status --split val --stage 1
    python -m pipeline.eval.batch_rag fetch  --split val --stage 1
    python -m pipeline.eval.batch_rag submit --split val --stage 2   # k on the chosen setting

Test set at the chosen setting (hybrid MiniLM, alpha=0.25, k=10):
    python -m pipeline.eval.batch_rag submit --split test --frozen
    python -m pipeline.eval.batch_rag status --split test --frozen
    python -m pipeline.eval.batch_rag fetch  --split test --frozen

Retrieval is local (train Chroma + BM25). Only chat completions are batched.
Already-cached keys are skipped. Empty retrievals are written to the cache
immediately (no LLM request).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .. import config
from ..labeling import common
from ..rag import generate, retrieve
from . import run_rag_actions

BATCH_DIR = config.CACHE_DIR / "openai_batch"
BATCH_DIR.mkdir(parents=True, exist_ok=True)
MAX_BATCH_BYTES = 180 * 1024 * 1024


def _client():
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OpenAI API key is not set. Set OPENAI_API_KEY in .env.")
    from openai import OpenAI

    return OpenAI(api_key=config.OPENAI_API_KEY)


def _eligible_ids(split_name: str):
    ids_path = config.VAL_IDS_JSON if split_name == "val" else config.TEST_IDS_JSON
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
    return eligible, report_sentences


def sentence_gr_eligible_ids(split_name: str):
    """Reports with nonempty sentence G_r and a first-sentence query."""
    ids_path = config.VAL_IDS_JSON if split_name == "val" else config.TEST_IDS_JSON
    ids = json.load(open(ids_path))
    sentence_gr = common.load_json_or_empty(config.SENTENCE_GR_JSON)
    report_sentences = common.load_report_sentences()
    eligible = [
        bid
        for bid in ids
        if sentence_gr.get(str(bid))
        and str(bid) in report_sentences
        and (report_sentences[str(bid)].get("sentences") or [None])[0]
    ]
    return eligible, report_sentences


# --------------------------------------------------------------------------- #
# Shared batch plumbing
# --------------------------------------------------------------------------- #

def _paths(split_name: str, tag: str):
    return {
        "batches": BATCH_DIR / f"batches_{split_name}_{tag}.json",
        "errors": BATCH_DIR / f"errors_{split_name}_{tag}.json",
        "jsonl": BATCH_DIR / f"pending_{split_name}_{tag}.jsonl",
        "meta": BATCH_DIR / f"pending_meta_{split_name}_{tag}.json",
    }


def _empty_entry(pool, embedder, alpha):
    return {
        "actions": [],
        "n_sentences_in_pool": len(pool),
        "n_reports_retrieved": 0,
        "embedder": embedder,
        "alpha": float(alpha),
    }


def _request(key, query_text, selected, topk):
    if len(key) > 64:
        raise RuntimeError(f"custom_id exceeds 64 chars: {key!r}")
    return {
        "custom_id": key,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": config.GEN_LLM_MODEL,
            "messages": generate.build_messages(
                query_text, selected, max_full_reports=topk
            ),
            "temperature": config.GEN_LLM_TEMPERATURE,
            "max_tokens": config.GEN_LLM_MAX_TOKENS,
        },
    }


def _shards(jsonl_path: Path, tasks):
    """Split into <180MB files if needed (OpenAI batch input limit)."""
    if jsonl_path.stat().st_size <= MAX_BATCH_BYTES:
        return [jsonl_path]
    print("JSONL exceeds 180MB — splitting into shards…")
    shard_tasks, current, current_bytes = [], [], 0
    for obj in tasks:
        line = json.dumps(obj) + "\n"
        if current and current_bytes + len(line.encode()) > MAX_BATCH_BYTES:
            shard_tasks.append(current)
            current, current_bytes = [], 0
        current.append(obj)
        current_bytes += len(line.encode())
    if current:
        shard_tasks.append(current)
    shards = []
    for i, chunk in enumerate(shard_tasks):
        p = jsonl_path.with_name(jsonl_path.stem + f"_part{i + 1}.jsonl")
        with open(p, "w", encoding="utf-8") as fh:
            for obj in chunk:
                fh.write(json.dumps(obj) + "\n")
        shards.append(p)
        print(f"  shard {i + 1}: {len(chunk)} requests -> {p}")
    return shards


def _submit_files(paths, files, metadata, extra_rec):
    """Upload each JSONL as one OpenAI batch and record it in the tracker."""
    client = _client()
    batches_list = common.load_json_or_empty(paths["batches"]).get("batches", [])
    submitted_ids = []
    for shard in files:
        with open(shard, "rb") as fh:
            batch_file = client.files.create(file=fh, purpose="batch")
        with open(shard, encoding="utf-8") as fh:
            custom_ids = [json.loads(line)["custom_id"] for line in fh]
        batch = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata=metadata,
        )
        batches_list.append(
            {
                "batch_id": batch.id,
                "input_file_id": batch_file.id,
                "n_requests": len(custom_ids),
                "custom_ids": custom_ids,
                "submitted_at": time.time(),
                "fetched": False,
                "status": batch.status,
                "shard": str(shard),
                **extra_rec,
            }
        )
        submitted_ids.append(batch.id)
        print(
            f"Submitted OpenAI batch {batch.id} ({len(custom_ids)} requests). "
            f"status={batch.status}"
        )
    common.save_json_atomic(paths["batches"], {"batches": batches_list})
    return submitted_ids


def _write_and_submit(paths, tasks, meta, inv_path, metadata, extra_rec, track_flags):
    with open(paths["jsonl"], "w", encoding="utf-8") as fh:
        for obj in tasks:
            fh.write(json.dumps(obj) + "\n")
    common.save_json_atomic(paths["meta"], meta)
    size_mb = paths["jsonl"].stat().st_size / (1024 * 1024)
    print(f"Wrote {len(tasks)} requests → {paths['jsonl']} ({size_mb:.1f} MB)")

    submitted_ids = _submit_files(
        paths, _shards(paths["jsonl"], tasks), metadata, extra_rec
    )
    inv = common.load_json_or_empty(inv_path)
    inv["batch_ids"] = submitted_ids
    common.save_json_atomic(inv_path, inv)
    print(f"Track: python -m pipeline.eval.batch_rag status {track_flags}")
    print(f"Fetch later: python -m pipeline.eval.batch_rag fetch {track_flags}")
    print(f"Inventory -> {inv_path}")
    return submitted_ids


def _status(paths, label):
    batches = common.load_json_or_empty(paths["batches"]).get("batches", [])
    if not batches:
        print(f"No {label} batches submitted.")
        return
    client = _client()
    for b in batches:
        info = client.batches.retrieve(b["batch_id"])
        counts = getattr(info, "request_counts", None)
        print(
            f"{info.id}  status={info.status}  fetched={b.get('fetched')}  "
            f"n={b.get('n_requests')}  counts={counts}"
        )


def _read_file(client, file_id):
    content = client.files.content(file_id)
    return content.text if hasattr(content, "text") else content.read().decode("utf-8")


def _fetch(split_name, paths, label, on_ok=None):
    """Pull finished batches into the cache. Returns (n_ok, n_still, errors)."""
    batches_list = common.load_json_or_empty(paths["batches"]).get("batches", [])
    if not batches_list:
        print(f"No {label} batches submitted.")
        return 0, 0, {}

    client = _client()
    cache = run_rag_actions.load_cache(split_name)
    errors = common.load_json_or_empty(paths["errors"])
    meta = common.load_json_or_empty(paths["meta"])

    n_ok = 0
    n_still = 0
    for b in batches_list:
        if b.get("fetched"):
            continue
        info = client.batches.retrieve(b["batch_id"])
        if info.status not in ("completed", "expired", "cancelled", "failed"):
            print(f"{info.id}: still {info.status} — skip fetch")
            n_still += 1
            continue
        if info.status != "completed" or not info.output_file_id:
            print(f"{info.id}: status={info.status}, no output to fetch")
            if info.error_file_id:
                print(f"  error_file_id={info.error_file_id}")
            b["fetched"] = True
            b["status"] = info.status
            continue

        for line in _read_file(client, info.output_file_id).splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["custom_id"]
            resp = row.get("response") or {}
            body = resp.get("body") or {}
            status_code = resp.get("status_code")
            if status_code != 200:
                errors[key] = f"status={status_code} error={row.get('error') or body}"
                continue
            try:
                raw = body["choices"][0]["message"]["content"].strip()
                actions = generate.split_action_lines(
                    generate.normalize_actions(raw)
                )
            except Exception as exc:
                errors[key] = f"parse_error: {exc}"
                continue

            m = meta.get(key, {})
            cache[key] = {
                "actions": actions,
                "n_sentences_in_pool": m.get("n_sentences_in_pool", 0),
                "n_reports_retrieved": m.get("n_reports_retrieved", 0),
                "embedder": m.get("embedder"),
                "alpha": m.get("alpha"),
            }
            errors.pop(key, None)
            n_ok += 1
            if on_ok:
                on_ok(cache, key)

        if info.error_file_id:
            for line in _read_file(client, info.error_file_id).splitlines():
                if line.strip():
                    row = json.loads(line)
                    errors[row.get("custom_id", "?")] = str(row.get("error") or row)

        b["fetched"] = True
        b["status"] = info.status
        print(f"Fetched {info.id}: +{n_ok} ok so far")

    run_rag_actions.flush_cache(split_name, cache)
    common.save_json_atomic(paths["errors"], errors)
    common.save_json_atomic(paths["batches"], {"batches": batches_list})
    print(f"Cache now has {len(cache)} entries → {run_rag_actions._cache_path(split_name)}")
    print(f"ok={n_ok} still_in_progress={n_still} error_records={len(errors)}")
    return n_ok, n_still, errors


def _drop_errored(split_name, paths, cache):
    errors = common.load_json_or_empty(paths["errors"])
    for key in list(errors.keys()):
        cache.pop(key, None)
    common.save_json_atomic(paths["errors"], {})
    run_rag_actions.flush_cache(split_name, cache)


# --------------------------------------------------------------------------- #
# Hyperparameter search on validation
# --------------------------------------------------------------------------- #

def _hp_tag(stage: int) -> str:
    return f"hp_stage{stage}"


def _hp_inventory_json(stage: int) -> Path:
    if stage == 1:
        return config.HP_STAGE1_INVENTORY_JSON
    if stage == 2:
        return config.HP_STAGE2_INVENTORY_JSON
    raise RuntimeError(f"Unknown HP stage {stage}")


def _hp_grid(stage: int):
    if stage == 1:
        return [
            (embedder, alpha, config.SWEEP_STAGE1_TOPK)
            for embedder in config.SWEEP_EMBEDDERS
            for alpha in config.SWEEP_ALPHAS
        ]
    if stage == 2:
        return [
            (config.STAGE2_EMBEDDER, config.STAGE2_ALPHA, int(k))
            for k in config.SWEEP_STAGE2_TOPKS
        ]
    raise RuntimeError(f"Unknown HP stage {stage}")


def _minilm_query_embeddings(eligible, report_sentences):
    """Encode val first sentences once with MiniLM (local)."""
    col = retrieve._get_cached_collection("minilm")
    ef = getattr(col, "_embedding_function", None)
    texts = [report_sentences[str(bid)]["sentences"][0] for bid in eligible]
    if ef is None:
        return {}
    print(f"Encoding {len(texts)} MiniLM val queries locally…")
    vecs = list(ef(texts))
    return {int(bid): vecs[i] for i, bid in enumerate(eligible)}


def _copy_alpha0_openai_from_minilm(cache: dict, topk: int, bid) -> bool:
    """alpha=0 is pure BM25, so the generation is embedder-independent."""
    src = run_rag_actions.hp_key("minilm", 0.0, topk, bid)
    dst = run_rag_actions.hp_key("openai3large", 0.0, topk, bid)
    if dst in cache or src not in cache:
        return False
    entry = dict(cache[src])
    entry["copied_from"] = src
    entry["embedder"] = "openai3large"
    entry["alpha"] = 0.0
    cache[dst] = entry
    return True


def prepare_and_submit_hp(split_name: str, stage: int, limit=None, retry_errored=False):
    """Stage 1: embedder x alpha at k=10. Stage 2: k in {5, 10, 15, 20}."""
    if split_name != "val":
        raise RuntimeError("Hyperparameter search is validation-only.")
    from ..rag import openai_embed

    paths = _paths(split_name, _hp_tag(stage))
    eligible, report_sentences = _eligible_ids(split_name)
    grid = _hp_grid(stage)
    cache = run_rag_actions.load_cache(split_name)
    inv_path = _hp_inventory_json(stage)
    if retry_errored:
        _drop_errored(split_name, paths, cache)

    print(f"[HP stage {stage} {split_name}] eligible={len(eligible)} cells={len(grid)}")

    need_openai = any(e == "openai3large" and abs(float(a)) > 1e-12 for e, a, _ in grid)
    need_minilm = any(e == "minilm" and abs(float(a)) > 1e-12 for e, a, _ in grid)
    openai_qemb = (
        openai_embed.ensure_val_query_embeddings(eligible, report_sentences)
        if need_openai
        else {}
    )
    minilm_qemb = _minilm_query_embeddings(eligible, report_sentences) if need_minilm else {}

    tasks, meta, inventory = [], {}, []
    n_skipped = n_empty = n_todo = n_copy_on_fetch = 0
    for embedder, alpha, tk in grid:
        cell = {"n_submit": 0, "n_cached": 0, "n_copied": 0, "n_empty": 0}
        for bid in eligible:
            key = run_rag_actions.hp_key(embedder, alpha, tk, bid)
            if key in cache:
                n_skipped += 1
                cell["n_cached"] += 1
                continue
            # alpha=0 is embedder-independent: generate for MiniLM only and copy
            # the result onto openai3large.
            if embedder == "openai3large" and abs(float(alpha)) < 1e-12:
                if _copy_alpha0_openai_from_minilm(cache, tk, bid):
                    n_skipped += 1
                    cell["n_copied"] += 1
                    continue
                if stage == 1:
                    n_copy_on_fetch += 1
                    cell["n_copied"] += 1
                    continue

            query_text = report_sentences[str(bid)]["sentences"][0]
            qemb = (openai_qemb if embedder == "openai3large" else minilm_qemb).get(int(bid))
            selected, pool = retrieve.retrieve(
                query_text,
                tk,
                embedder=embedder,
                alpha=alpha,
                query_embedding=qemb,
                breach_id=bid,
            )
            n_todo += 1
            if limit is not None and n_todo > limit:
                break
            if not selected:
                cache[key] = _empty_entry(pool, embedder, alpha)
                n_empty += 1
                cell["n_empty"] += 1
                continue
            tasks.append(_request(key, query_text, selected, tk))
            meta[key] = {
                "n_sentences_in_pool": len(pool),
                "n_reports_retrieved": len({r["breach_id"] for r in selected}),
                "embedder": embedder,
                "alpha": float(alpha),
                "topk": tk,
                "breach_id": str(bid),
            }
            cell["n_submit"] += 1
            if len(tasks) % 100 == 0:
                print(f"  prepared {len(tasks)} LLM requests…", flush=True)
        else:
            inventory.append(
                {"embedder": embedder, "alpha": float(alpha), "topk": tk,
                 "n_eligible": len(eligible), **cell}
            )
            continue
        break  # limit hit

    run_rag_actions.flush_cache(split_name, cache)
    common.save_json_atomic(
        inv_path,
        {
            "split": split_name,
            "stage": stage,
            "n_actions_requested": config.SUGGESTED_ACTION_COUNT,
            "eligible": len(eligible),
            "grid": [{"embedder": e, "alpha": a, "topk": k} for e, a, k in grid],
            "cells": inventory,
            "n_llm_requests": len(tasks),
            "n_empty": n_empty,
            "n_skipped_cached": n_skipped,
            "openai_alpha0_copy_on_fetch": n_copy_on_fetch,
        },
    )
    if not tasks:
        print(f"Nothing to submit (empty retrievals {n_empty}, cached {n_skipped}).")
        print(f"Inventory -> {inv_path}")
        return []

    return _write_and_submit(
        paths,
        tasks,
        meta,
        inv_path,
        metadata={"split": split_name, "project": "ember", "stage": str(stage)},
        extra_rec={"stage": stage},
        track_flags=f"--split {split_name} --stage {stage}",
    )


def upload_hp_prepared(split_name: str, stage: int):
    """Upload an already-written pending JSONL (retry after a failed upload)."""
    paths = _paths(split_name, _hp_tag(stage))
    if not paths["jsonl"].exists():
        raise RuntimeError(f"No prepared JSONL at {paths['jsonl']}")
    ids = _submit_files(
        paths,
        [paths["jsonl"]],
        metadata={"split": split_name, "project": "ember", "stage": str(stage)},
        extra_rec={"stage": stage},
    )
    inv_path = _hp_inventory_json(stage)
    inv = common.load_json_or_empty(inv_path)
    inv["batch_ids"] = ids
    common.save_json_atomic(inv_path, inv)
    return ids


def fetch_hp(split_name: str, stage: int):
    def mirror_alpha0(cache, key):
        if key.startswith("minilm_0.0_"):
            dst = "openai3large" + key[len("minilm"):]
            if dst not in cache:
                entry = dict(cache[key])
                entry["copied_from"] = key
                entry["embedder"] = "openai3large"
                cache[dst] = entry

    paths = _paths(split_name, _hp_tag(stage))
    _, _, errors = _fetch(split_name, paths, f"HP stage-{stage}", on_ok=mirror_alpha0)

    # alpha=0 MiniLM entries cached by an earlier fetch.
    eligible, _ = _eligible_ids(split_name)
    cache = run_rag_actions.load_cache(split_name)
    topks = {k for _, _, k in _hp_grid(stage)}
    n_copied = sum(
        _copy_alpha0_openai_from_minilm(cache, tk, bid) for tk in topks for bid in eligible
    )
    if n_copied:
        run_rag_actions.flush_cache(split_name, cache)
        print(f"alpha0_copied={n_copied}")
    if errors:
        print(
            "Retry failed keys: python -m pipeline.eval.batch_rag submit "
            f"--split {split_name} --stage {stage} --retry-errored"
        )


def coverage_hp(split_name: str, stage: int):
    eligible, _ = _eligible_ids(split_name)
    cache = run_rag_actions.load_cache(split_name)
    print(f"[HP stage {stage} {split_name}] eligible={len(eligible)}")
    for embedder, alpha, tk in _hp_grid(stage):
        missing = sum(
            run_rag_actions.hp_key(embedder, alpha, tk, bid) not in cache for bid in eligible
        )
        print(
            f"  {embedder:13s} α={run_rag_actions.format_alpha(alpha):4s} k={tk} "
            f"cached={len(eligible) - missing}/{len(eligible)} missing={missing}"
        )


# --------------------------------------------------------------------------- #
# Test set at the chosen setting
# --------------------------------------------------------------------------- #

def prepare_and_submit_frozen(
    split_name: str,
    embedder=config.FROZEN_EMBEDDER,
    alpha=config.FROZEN_ALPHA,
    topk=config.FROZEN_TOPK,
    limit=None,
    retry_errored=False,
):
    """Retrieve locally and batch-generate GPT-4o actions for every report with
    a nonempty sentence G_r. Empty retrievals are cached with actions=[]."""
    paths = _paths(split_name, "hp_frozen")
    eligible, report_sentences = sentence_gr_eligible_ids(split_name)
    cache = run_rag_actions.load_cache(split_name)
    inv_path = config.HP_FROZEN_TEST_INVENTORY_JSON
    if retry_errored:
        _drop_errored(split_name, paths, cache)

    print(
        f"[frozen {split_name}] eligible={len(eligible)} embedder={embedder} "
        f"α={run_rag_actions.format_alpha(alpha)} k={topk} "
        f"n_actions={config.SUGGESTED_ACTION_COUNT}"
    )

    tasks, meta = [], {}
    n_skipped = n_empty = n_todo = 0
    for bid in eligible:
        key = run_rag_actions.hp_key(embedder, alpha, topk, bid)
        if key in cache:
            n_skipped += 1
            continue
        query_text = report_sentences[str(bid)]["sentences"][0]
        selected, pool = retrieve.retrieve(
            query_text, topk, embedder=embedder, alpha=alpha, breach_id=bid
        )
        n_todo += 1
        if limit is not None and n_todo > limit:
            break
        if not selected:
            cache[key] = _empty_entry(pool, embedder, alpha)
            n_empty += 1
            continue
        tasks.append(_request(key, query_text, selected, topk))
        meta[key] = {
            "n_sentences_in_pool": len(pool),
            "n_reports_retrieved": len({r["breach_id"] for r in selected}),
            "embedder": embedder,
            "alpha": float(alpha),
            "topk": topk,
            "breach_id": str(bid),
        }

    run_rag_actions.flush_cache(split_name, cache)
    common.save_json_atomic(
        inv_path,
        {
            "split": split_name,
            "n_actions_requested": config.SUGGESTED_ACTION_COUNT,
            "retrieval": "BM25 only" if abs(alpha) < 1e-12 else "hybrid",
            "embedder": None if abs(alpha) < 1e-12 else embedder,
            "alpha": float(alpha),
            "topk": topk,
            "eligible": len(eligible),
            "n_llm_requests": len(tasks),
            "n_empty": n_empty,
            "n_skipped_cached": n_skipped,
            "g_r": "sentence",
        },
    )
    if not tasks:
        print(f"Nothing to submit (empty retrievals {n_empty}, cached {n_skipped}).")
        print(f"Inventory -> {inv_path}")
        return []

    return _write_and_submit(
        paths,
        tasks,
        meta,
        inv_path,
        metadata={
            "split": split_name,
            "project": "ember",
            "embedder": embedder,
            "alpha": run_rag_actions.format_alpha(alpha),
            "topk": str(topk),
        },
        extra_rec={"embedder": embedder, "alpha": float(alpha), "topk": topk},
        track_flags=f"--split {split_name} --frozen",
    )


def fetch_frozen(split_name: str):
    paths = _paths(split_name, "hp_frozen")
    result = _fetch(split_name, paths, "frozen")
    if result[2]:
        print(
            "Retry failed keys: python -m pipeline.eval.batch_rag submit "
            f"--split {split_name} --frozen --retry-errored"
        )
    return result


def coverage_frozen(
    split_name: str,
    embedder=config.FROZEN_EMBEDDER,
    alpha=config.FROZEN_ALPHA,
    topk=config.FROZEN_TOPK,
):
    eligible, _ = sentence_gr_eligible_ids(split_name)
    cache = run_rag_actions.load_cache(split_name)
    missing = empty = 0
    for bid in eligible:
        key = run_rag_actions.hp_key(embedder, alpha, topk, bid)
        if key not in cache:
            missing += 1
        elif not (cache[key].get("actions") or []):
            empty += 1
    print(
        f"[frozen {split_name}] eligible={len(eligible)} "
        f"{embedder} α={run_rag_actions.format_alpha(alpha)} k={topk} "
        f"cached={len(eligible) - missing}/{len(eligible)} "
        f"missing={missing} empty_actions={empty}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OpenAI Batch API fill for the RAG action caches."
    )
    parser.add_argument(
        "command", choices=["submit", "status", "fetch", "coverage", "upload-prepared"]
    )
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--stage", type=int, choices=[1, 2], default=None,
                        help="hyperparameter search stage (validation)")
    parser.add_argument("--frozen", action="store_true",
                        help="run the chosen setting (default: test set)")
    parser.add_argument("--embedder", default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap new LLM requests this submit")
    parser.add_argument("--retry-errored", action="store_true",
                        help="drop errored keys from cache/errors and resubmit them")
    args = parser.parse_args()

    if args.frozen == (args.stage is not None):
        raise SystemExit("Use exactly one of --stage or --frozen.")

    if args.frozen:
        kw = dict(
            embedder=args.embedder or config.FROZEN_EMBEDDER,
            alpha=config.FROZEN_ALPHA if args.alpha is None else args.alpha,
            topk=config.FROZEN_TOPK if args.topk is None else args.topk,
        )
        if args.command == "submit":
            prepare_and_submit_frozen(
                args.split, limit=args.limit, retry_errored=args.retry_errored, **kw
            )
        elif args.command == "status":
            _status(_paths(args.split, "hp_frozen"), "frozen")
        elif args.command == "fetch":
            fetch_frozen(args.split)
        elif args.command == "coverage":
            coverage_frozen(args.split, **kw)
        else:
            raise SystemExit("upload-prepared is only available with --stage.")
    else:
        if args.command == "submit":
            prepare_and_submit_hp(
                args.split, args.stage, limit=args.limit, retry_errored=args.retry_errored
            )
        elif args.command == "upload-prepared":
            upload_hp_prepared(args.split, args.stage)
        elif args.command == "status":
            _status(_paths(args.split, _hp_tag(args.stage)), f"HP stage-{args.stage}")
        elif args.command == "fetch":
            fetch_hp(args.split, args.stage)
        elif args.command == "coverage":
            coverage_hp(args.split, args.stage)
