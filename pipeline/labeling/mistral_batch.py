"""
Mistral action-sentence labeling via the Mistral batch API (Mistral Large 3).

Usage:
    python -m pipeline.labeling.mistral_batch submit
    python -m pipeline.labeling.mistral_batch submit --retry-errored
    python -m pipeline.labeling.mistral_batch status
    python -m pipeline.labeling.mistral_batch fetch

Writes:
    pipeline/data/labels/mistral/batches.json
    pipeline/data/labels/mistral/actions_mistral.json -- {breach_id: {sentence_ids, raw, status}}
    pipeline/data/labels/mistral/errors.json
"""

import argparse
import io
import json
import time

from .. import config
from ..keys import MISTRAL_API_KEY
from . import common

BATCHES_JSON = config.MISTRAL_DIR / "batches.json"
RESULTS_JSON = config.MISTRAL_DIR / "actions_mistral.json"
ERRORS_JSON = config.MISTRAL_DIR / "errors.json"


def _client():
    from mistralai.client import Mistral

    if not MISTRAL_API_KEY:
        raise RuntimeError(
            "Set MISTRAL_API_KEY in .env (see .env.example)."
        )
    return Mistral(api_key=MISTRAL_API_KEY)


def _target_ids(report_sentences):
    return sorted(int(k) for k in report_sentences.keys())


def _build_jsonl(todo, report_sentences):
    buf = io.BytesIO()
    for bid in todo:
        sentences = report_sentences[str(bid)]["sentences"]
        user_prompt = common.build_user_prompt(sentences)
        row = {
            "custom_id": str(bid),
            "body": {
                "max_tokens": config.LABEL_LLM_MAX_TOKENS,
                "temperature": config.LABEL_LLM_TEMPERATURE,
                "messages": [
                    {"role": "system", "content": config.ACTION_LABEL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            },
        }
        buf.write(json.dumps(row).encode("utf-8"))
        buf.write(b"\n")
    return buf.getvalue()


def submit(retry_errored=False):
    from mistralai.client.models import File

    report_sentences = common.load_report_sentences()
    all_ids = _target_ids(report_sentences)
    results = common.load_json_or_empty(RESULTS_JSON)
    errors = common.load_json_or_empty(ERRORS_JSON)

    todo = common.pending_ids(all_ids, results, errors, retry_errored)
    if not todo:
        print("Nothing to submit — all reports already labeled (ok) or errored (use --retry-errored).")
        return

    client = _client()
    jsonl_bytes = _build_jsonl(todo, report_sentences)
    uploaded = client.files.upload(
        file=File(file_name="bertscore_actions_batch.jsonl", content=jsonl_bytes),
        purpose="batch",
    )
    job = client.batch.jobs.create(
        input_files=[uploaded.id],
        endpoint="/v1/chat/completions",
        model=config.MISTRAL_MODEL,
    )

    batches = common.load_json_or_empty(BATCHES_JSON)
    batches_list = batches.get("batches", [])
    batches_list.append(
        {
            "job_id": job.id,
            "input_file_id": uploaded.id,
            "custom_ids": [str(b) for b in todo],
            "submitted_at": time.time(),
            "fetched": False,
        }
    )
    common.save_json_atomic(BATCHES_JSON, {"batches": batches_list})

    print(f"Submitted job {job.id} with {len(todo)} requests (input file {uploaded.id}).")
    print(f"Status: {job.status}")


def status():
    batches = common.load_json_or_empty(BATCHES_JSON).get("batches", [])
    if not batches:
        print("No jobs submitted yet.")
        return
    client = _client()
    for b in batches:
        if b.get("fetched"):
            print(f"{b['job_id']}: fetched ({len(b['custom_ids'])} requests)")
            continue
        job = client.batch.jobs.get(job_id=b["job_id"])
        print(
            f"{b['job_id']}: {job.status} ({len(b['custom_ids'])} requests) "
            f"succeeded={getattr(job, 'succeeded_requests', '?')} "
            f"failed={getattr(job, 'failed_requests', '?')}"
        )


def fetch():
    batches = common.load_json_or_empty(BATCHES_JSON).get("batches", [])
    if not batches:
        print("No jobs submitted yet.")
        return

    client = _client()
    results = common.load_json_or_empty(RESULTS_JSON)
    errors = common.load_json_or_empty(ERRORS_JSON)
    report_sentences = common.load_report_sentences()

    any_fetched = False
    for b in batches:
        if b.get("fetched"):
            continue
        job = client.batch.jobs.get(job_id=b["job_id"])
        if job.status not in ("SUCCESS", "FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"):
            print(f"{b['job_id']}: still {job.status}, skipping.")
            continue

        n_ok, n_err = 0, 0
        if job.output_file:
            resp = client.files.download(file_id=job.output_file)
            content = resp.read().decode("utf-8")
            for line in content.splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                cid = row.get("custom_id")
                n_sentences = len(report_sentences.get(cid, {}).get("sentences", []))
                resp = row.get("response") or {}
                if row.get("error") or resp.get("status_code") != 200:
                    errors[cid] = f"api_error: {row.get('error') or resp}"
                    n_err += 1
                    continue
                try:
                    raw = resp["body"]["choices"][0]["message"]["content"]
                    ids = common.parse_sentence_ids(raw, n_sentences)
                    results[cid] = {"sentence_ids": ids, "raw": raw, "status": "ok"}
                    errors.pop(cid, None)
                    n_ok += 1
                except (KeyError, IndexError, ValueError) as e:
                    errors[cid] = f"parse_error: {e}"
                    n_err += 1

        if job.error_file:
            resp = client.files.download(file_id=job.error_file)
            content = resp.read().decode("utf-8")
            for line in content.splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                cid = row.get("custom_id")
                if cid and cid not in results:
                    errors[cid] = f"job_error_file: {row.get('error')}"
                    n_err += 1

        b["fetched"] = True
        any_fetched = True
        print(f"{b['job_id']}: fetched {n_ok} ok, {n_err} errored.")

    if any_fetched:
        common.save_json_atomic(BATCHES_JSON, {"batches": batches})
        common.save_json_atomic(RESULTS_JSON, results)
        common.save_json_atomic(ERRORS_JSON, errors)
        print(f"Saved -> {RESULTS_JSON}")
        if errors:
            print(f"{len(errors)} errored entries -> {ERRORS_JSON} (use `submit --retry-errored`)")
    else:
        print("Nothing new to fetch.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mistral batch action-sentence labeling.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_submit = sub.add_parser("submit")
    p_submit.add_argument("--retry-errored", action="store_true")
    sub.add_parser("status")
    sub.add_parser("fetch")
    args = parser.parse_args()

    if args.cmd == "submit":
        submit(retry_errored=args.retry_errored)
    elif args.cmd == "status":
        status()
    elif args.cmd == "fetch":
        fetch()
