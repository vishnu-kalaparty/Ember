"""
Claude action-sentence labeling via the Anthropic Message Batches API.

Usage:
    python -m pipeline.labeling.claude_batch submit                 # submit pending reports
    python -m pipeline.labeling.claude_batch submit --retry-errored # + previously-errored reports
    python -m pipeline.labeling.claude_batch status                 # poll all open batches
    python -m pipeline.labeling.claude_batch fetch                  # pull results of ended batches
    python -m pipeline.labeling.claude_batch cancel                 # cancel all unfetched batches
    python -m pipeline.labeling.claude_batch cancel --batch-id ID   # cancel one batch

Writes:
    pipeline/data/labels/claude/batches.json   -- submitted batch ids (bookkeeping)
    pipeline/data/labels/claude/actions_claude.json -- {breach_id: {sentence_ids, raw, status}}
    pipeline/data/labels/claude/errors.json    -- {breach_id: reason}
"""

import argparse
import time

from .. import config
from ..keys import CLAUDE_API_KEY
from . import common

BATCHES_JSON = config.CLAUDE_DIR / "batches.json"
RESULTS_JSON = config.CLAUDE_DIR / "actions_claude.json"
ERRORS_JSON = config.CLAUDE_DIR / "errors.json"


def _client():
    import anthropic

    if not CLAUDE_API_KEY:
        raise RuntimeError(
            "Set ANTHROPIC_API_KEY in .env (see .env.example)."
        )
    return anthropic.Anthropic(api_key=CLAUDE_API_KEY)


def _target_ids(report_sentences):
    return sorted(int(k) for k in report_sentences.keys())


def submit(retry_errored=False):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    report_sentences = common.load_report_sentences()
    all_ids = _target_ids(report_sentences)
    results = common.load_json_or_empty(RESULTS_JSON)
    errors = common.load_json_or_empty(ERRORS_JSON)

    todo = common.pending_ids(all_ids, results, errors, retry_errored)
    if not todo:
        print("Nothing to submit — all reports already labeled (ok) or errored (use --retry-errored).")
        return

    requests = []
    for bid in todo:
        sentences = report_sentences[str(bid)]["sentences"]
        user_prompt = common.build_user_prompt(sentences)
        requests.append(
            Request(
                custom_id=str(bid),
                params=MessageCreateParamsNonStreaming(
                    model=config.CLAUDE_MODEL,
                    max_tokens=config.LABEL_LLM_MAX_TOKENS,
                    system=config.ACTION_LABEL_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                ),
            )
        )

    client = _client()
    batch = client.messages.batches.create(requests=requests)

    batches = common.load_json_or_empty(BATCHES_JSON)
    batches_list = batches.get("batches", [])
    batches_list.append(
        {
            "batch_id": batch.id,
            "custom_ids": [str(b) for b in todo],
            "submitted_at": time.time(),
            "fetched": False,
        }
    )
    common.save_json_atomic(BATCHES_JSON, {"batches": batches_list})

    print(f"Submitted batch {batch.id} with {len(todo)} requests.")
    print(f"Processing status: {batch.processing_status}")


def status():
    batches = common.load_json_or_empty(BATCHES_JSON).get("batches", [])
    if not batches:
        print("No batches submitted yet.")
        return
    client = _client()
    for b in batches:
        if b.get("fetched"):
            print(f"{b['batch_id']}: fetched ({len(b['custom_ids'])} requests)")
            continue
        info = client.messages.batches.retrieve(b["batch_id"])
        print(
            f"{b['batch_id']}: {info.processing_status} "
            f"({len(b['custom_ids'])} requests) counts={info.request_counts}"
        )


def _extract_text(message):
    for block in message.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


def fetch():
    batches = common.load_json_or_empty(BATCHES_JSON).get("batches", [])
    if not batches:
        print("No batches submitted yet.")
        return

    client = _client()
    results = common.load_json_or_empty(RESULTS_JSON)
    errors = common.load_json_or_empty(ERRORS_JSON)
    report_sentences = common.load_report_sentences()

    any_fetched = False
    for b in batches:
        if b.get("fetched"):
            continue
        info = client.messages.batches.retrieve(b["batch_id"])
        if info.processing_status != "ended":
            print(f"{b['batch_id']}: still {info.processing_status}, skipping.")
            continue

        n_ok, n_err = 0, 0
        for entry in client.messages.batches.results(b["batch_id"]):
            cid = entry.custom_id
            n_sentences = len(report_sentences.get(cid, {}).get("sentences", []))
            if entry.result.type == "succeeded":
                raw = _extract_text(entry.result.message)
                try:
                    ids = common.parse_sentence_ids(raw, n_sentences)
                    results[cid] = {"sentence_ids": ids, "raw": raw, "status": "ok"}
                    errors.pop(cid, None)
                    n_ok += 1
                except ValueError as e:
                    errors[cid] = f"parse_error: {e}"
                    n_err += 1
            else:
                errors[cid] = f"{entry.result.type}"
                n_err += 1

        b["fetched"] = True
        any_fetched = True
        print(f"{b['batch_id']}: fetched {n_ok} ok, {n_err} errored.")

    if any_fetched:
        common.save_json_atomic(BATCHES_JSON, {"batches": batches})
        common.save_json_atomic(RESULTS_JSON, results)
        common.save_json_atomic(ERRORS_JSON, errors)
        print(f"Saved -> {RESULTS_JSON}")
        if errors:
            print(f"{len(errors)} errored entries -> {ERRORS_JSON} (use `submit --retry-errored`)")
    else:
        print("Nothing new to fetch.")


def cancel(batch_id=None):
    """Cancel one batch (by id) or every unfetched batch recorded locally."""
    batches = common.load_json_or_empty(BATCHES_JSON).get("batches", [])
    if not batches:
        print("No batches recorded locally.")
        return

    client = _client()
    targets = []
    if batch_id:
        targets = [b for b in batches if b["batch_id"] == batch_id]
        if not targets:
            # Still try canceling by id even if not in local bookkeeping.
            info = client.messages.batches.cancel(batch_id)
            print(f"{batch_id}: cancel requested -> {info.processing_status}")
            return
    else:
        targets = [b for b in batches if not b.get("fetched") and not b.get("cancelled")]

    if not targets:
        print("No open (unfetched) batches to cancel.")
        return

    for b in targets:
        info = client.messages.batches.cancel(b["batch_id"])
        b["cancelled"] = True
        print(f"{b['batch_id']}: cancel requested -> {info.processing_status}")

    common.save_json_atomic(BATCHES_JSON, {"batches": batches})
    print(
        "Local bookkeeping updated. Re-submit later with "
        "`python -m pipeline.labeling.claude_batch submit` "
        "(pending reports are still unlabeled)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Claude batch action-sentence labeling.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_submit = sub.add_parser("submit")
    p_submit.add_argument("--retry-errored", action="store_true")
    sub.add_parser("status")
    sub.add_parser("fetch")
    p_cancel = sub.add_parser("cancel")
    p_cancel.add_argument("--batch-id", default=None, help="Cancel this batch id only.")
    args = parser.parse_args()

    if args.cmd == "submit":
        submit(retry_errored=args.retry_errored)
    elif args.cmd == "status":
        status()
    elif args.cmd == "fetch":
        fetch()
    elif args.cmd == "cancel":
        cancel(batch_id=args.batch_id)
