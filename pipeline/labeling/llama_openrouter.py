"""
Llama 3.3 70B action-sentence labeling via OpenRouter (free-tier key), one
request per report (OpenRouter's free tier does not offer async batch).

Sequential with checkpointing: every row is saved to disk immediately, so a
crash or quota cutoff never loses progress. When a quota/rate-limit error is
detected, the run STOPS immediately (does not burn through remaining rows
failing one-by-one) and records exactly which breach_id it stopped at. Get a
fresh free-tier key, set OPENROUTER_API_KEY in .env, and re-run —
it resumes automatically from the first not-yet-completed report.

Usage:
    python -m pipeline.labeling.llama_openrouter run
    python -m pipeline.labeling.llama_openrouter run --retry-errored
    python -m pipeline.labeling.llama_openrouter run --limit 50   # cap this run
    python -m pipeline.labeling.llama_openrouter status

Writes:
    pipeline/data/labels/llama/actions_llama.json -- {breach_id: {sentence_ids, raw, status}}
    pipeline/data/labels/llama/errors.json
    pipeline/data/labels/llama/stopped_at.json    -- last quota-stop checkpoint
"""

import argparse
import time

from .. import config
from ..keys import OPENROUTER_API_KEY
from . import common

RESULTS_JSON = config.LLAMA_DIR / "actions_llama.json"
ERRORS_JSON = config.LLAMA_DIR / "errors.json"
STOPPED_JSON = config.LLAMA_DIR / "stopped_at.json"

# Substrings that mark a response as a quota/rate-limit cutoff rather than a
# one-off transient error worth skipping and continuing past.
_QUOTA_MARKERS = (
    "429",
    "rate limit",
    "rate-limit",
    "ratelimit",
    "quota",
    "insufficient_quota",
    "payment required",
    "402",
    "free-tier",
    "temporarily rate-limited",
)


def _client():
    from openai import OpenAI

    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "Set OPENROUTER_API_KEY in .env (see .env.example)."
        )
    return OpenAI(api_key=OPENROUTER_API_KEY, base_url=config.OPENROUTER_BASE_URL)


def _is_quota_error(exc):
    msg = str(exc).lower()
    return any(marker in msg for marker in _QUOTA_MARKERS)


def _target_ids(report_sentences):
    return sorted(int(k) for k in report_sentences.keys())


def run(retry_errored=False, limit=None, sleep_s=1.0):
    report_sentences = common.load_report_sentences()
    all_ids = _target_ids(report_sentences)
    results = common.load_json_or_empty(RESULTS_JSON)
    errors = common.load_json_or_empty(ERRORS_JSON)

    todo = common.pending_ids(all_ids, results, errors, retry_errored)
    if limit:
        todo = todo[:limit]
    if not todo:
        print("Nothing to do — all reports already labeled (ok) or errored (use --retry-errored).")
        return

    client = _client()
    print(f"{len(todo)} reports to label via OpenRouter ({config.LLAMA_MODEL}).")

    n_ok, n_err = 0, 0
    for i, bid in enumerate(todo):
        sentences = report_sentences[str(bid)]["sentences"]
        user_prompt = common.build_user_prompt(sentences)
        try:
            resp = client.chat.completions.create(
                model=config.LLAMA_MODEL,
                temperature=config.LABEL_LLM_TEMPERATURE,
                max_tokens=config.LABEL_LLM_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": config.ACTION_LABEL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = resp.choices[0].message.content
            ids = common.parse_sentence_ids(raw, len(sentences))
            results[str(bid)] = {"sentence_ids": ids, "raw": raw, "status": "ok"}
            errors.pop(str(bid), None)
            n_ok += 1
        except ValueError as e:
            # Parsed response but couldn't extract a JSON int array -> skip, keep going.
            errors[str(bid)] = f"parse_error: {e}"
            n_err += 1
        except Exception as e:  # noqa: BLE001 - need to inspect any SDK/HTTP error
            if _is_quota_error(e):
                common.save_json_atomic(RESULTS_JSON, results)
                common.save_json_atomic(ERRORS_JSON, errors)
                common.save_json_atomic(
                    STOPPED_JSON,
                    {
                        "stopped_at_breach_id": bid,
                        "remaining": todo[i:],
                        "reason": str(e),
                        "stopped_at_ts": time.time(),
                    },
                )
                print(f"\nQUOTA/RATE-LIMIT hit at breach_id={bid} ({i}/{len(todo)} done this run).")
                print(f"Reason: {e}")
                print(
                    f"Progress saved. {len(todo) - i} reports remain.\n"
                    "Get a fresh OpenRouter key, update OPENROUTER_API_KEY in .env, "
                    "then re-run `python -m pipeline.labeling.llama_openrouter run` to resume."
                )
                return
            errors[str(bid)] = f"api_error: {e}"
            n_err += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(todo):
            common.save_json_atomic(RESULTS_JSON, results)
            common.save_json_atomic(ERRORS_JSON, errors)
            print(f"  {i + 1}/{len(todo)} processed (ok={n_ok}, err={n_err})")

        time.sleep(sleep_s)

    common.save_json_atomic(RESULTS_JSON, results)
    common.save_json_atomic(ERRORS_JSON, errors)
    if STOPPED_JSON.exists():
        STOPPED_JSON.unlink()
    print(f"Done. ok={n_ok}, err={n_err}. Saved -> {RESULTS_JSON}")
    if errors:
        print(f"{len(errors)} errored entries -> {ERRORS_JSON} (use `run --retry-errored`)")


def status():
    report_sentences = common.load_report_sentences()
    all_ids = _target_ids(report_sentences)
    results = common.load_json_or_empty(RESULTS_JSON)
    errors = common.load_json_or_empty(ERRORS_JSON)
    n_ok = sum(1 for k in results if results[k].get("status") == "ok")
    print(f"target reports : {len(all_ids)}")
    print(f"labeled ok     : {n_ok}")
    print(f"errored        : {len(errors)}")
    print(f"pending        : {len(all_ids) - n_ok - len(errors)}")
    if STOPPED_JSON.exists():
        stopped = common.load_json_or_empty(STOPPED_JSON)
        print(f"STOPPED at breach_id={stopped.get('stopped_at_breach_id')} — swap key and re-run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama (OpenRouter) action-sentence labeling.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--retry-errored", action="store_true")
    p_run.add_argument("--limit", type=int, default=None)
    p_run.add_argument("--sleep", type=float, default=1.0)
    sub.add_parser("status")
    args = parser.parse_args()

    if args.cmd == "run":
        run(retry_errored=args.retry_errored, limit=args.limit, sleep_s=args.sleep)
    elif args.cmd == "status":
        status()
