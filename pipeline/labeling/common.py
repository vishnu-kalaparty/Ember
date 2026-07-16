"""
Shared helpers used by claude_batch.py, mistral_batch.py, llama_openrouter.py:
prompt building, response parsing, and the on-disk result-file format.

Result file format (actions_<model>.json), one dict for the whole run:
    {
      "<breach_id>": {"sentence_ids": [2, 4], "raw": "...", "status": "ok"},
      ...
    }
Errors are tracked separately in errors_<model>.json:
    {"<breach_id>": "reason string"}
so `--retry-errored` can resubmit only the failed rows.
"""

import json
import re

from .. import config


def load_report_sentences():
    with open(config.REPORT_SENTENCES_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def build_user_prompt(sentences):
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
    return config.ACTION_LABEL_USER_TEMPLATE.format(numbered_sentences=numbered)


_JSON_ARRAY_RE = re.compile(r"\[[^\[\]]*\]", re.DOTALL)


def parse_sentence_ids(raw_text, n_sentences):
    """Parse a JSON int array from the LLM reply, clamped to valid sentence IDs.

    Raises ValueError if no valid JSON array of ints can be found.
    """
    if raw_text is None:
        raise ValueError("empty response")
    text = raw_text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    candidates = [text]
    m = _JSON_ARRAY_RE.search(text)
    if m:
        candidates.append(m.group(0))

    for cand in candidates:
        try:
            parsed = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, list):
            ids = sorted({int(x) for x in parsed if isinstance(x, (int, float))})
            ids = [i for i in ids if 1 <= i <= n_sentences]
            return ids
    raise ValueError(f"could not parse a JSON int array from: {raw_text[:200]!r}")


def load_json_or_empty(path):
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_json_atomic(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    tmp.replace(path)


def pending_ids(all_ids, results, errors, retry_errored):
    """IDs still needing a call: never attempted, or errored (if retrying)."""
    todo = []
    for bid in all_ids:
        key = str(bid)
        if key in results and results[key].get("status") == "ok":
            continue
        if key in errors and not retry_errored:
            continue
        todo.append(bid)
    return todo
