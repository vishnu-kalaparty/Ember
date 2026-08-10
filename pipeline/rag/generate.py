"""
LLM generation: format retrieved context and call OpenAI. `max_full_reports`
mirrors the `topk` used for retrieval (retrieval depth k == number of
full reports in context).
"""

import re

from .. import config


def normalize_actions(text, count=None):
    count = count or config.SUGGESTED_ACTION_COUNT
    if not text or not text.strip():
        return text

    items = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\d+[\.\):\-]\s+(.+)$", line)
        if m:
            items.append(m.group(1).strip())
            continue
        if items and re.match(r"^[-*\u2022]\s+", line):
            continue
        if not items and len(line) > 10:
            continue

    if not items:
        items = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]

    items = items[:count]
    return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))


def split_action_lines(normalized_text):
    """Turn the "1. foo\\n2. bar" normalized string back into a plain list."""
    if not normalized_text:
        return []
    out = []
    for line in normalized_text.strip().splitlines():
        m = re.match(r"^\d+[\.\):\-]\s+(.+)$", line.strip())
        if m:
            out.append(m.group(1).strip())
    return out


def format_context(selected, max_full_reports):
    by_breach = {}
    order = []
    for r in selected:
        bid = r["breach_id"]
        if bid not in by_breach:
            by_breach[bid] = {
                "best_score": r["score"],
                "metadata": r["metadata"],
                "full_report": r["full_report"],
                "sentences": [],
            }
            order.append(bid)
        by_breach[bid]["sentences"].append((r["score"], r["text"]))
        by_breach[bid]["best_score"] = max(by_breach[bid]["best_score"], r["score"])

    order.sort(key=lambda b: by_breach[b]["best_score"], reverse=True)

    blocks = []
    for rank, bid in enumerate(order):
        info = by_breach[bid]
        m = info["metadata"]
        header = (
            f"[Breach {bid}] (similarity: {info['best_score']})\n"
            f"Entity: {m.get('Name of Covered Entity', 'N/A')} | "
            f"Type: {m.get('Type of Breach', 'N/A')} | "
            f"Location: {m.get('Location of Breached Information', 'N/A')} | "
            f"Date: {m.get('Breach Submission Date', 'N/A')}"
        )
        matched = "\n".join(
            f'  - "{txt}" ({score})' for score, txt in sorted(info["sentences"], reverse=True)
        )
        block = f"{header}\nMatched sentences:\n{matched}"
        if config.INCLUDE_FULL_REPORT_IN_CONTEXT and rank < max_full_reports:
            block += f"\nFull report: {info['full_report']}"
        blocks.append(block)

    return "\n---\n".join(blocks)


def build_messages(query_text, selected, max_full_reports):
    context = format_context(selected, max_full_reports)
    user_msg = config.USER_PROMPT_TEMPLATE.format(query=query_text, context=context)
    return [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def _openai_complete(messages, model=None):
    if not config.OPENAI_API_KEY:
        raise RuntimeError(
            "OpenAI API key is not set. Set OPENAI_API_KEY in .env."
        )
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=model or config.GEN_LLM_MODEL,
        messages=messages,
        temperature=config.GEN_LLM_TEMPERATURE,
        max_tokens=config.GEN_LLM_MAX_TOKENS,
    )
    raw = resp.choices[0].message.content.strip()
    return normalize_actions(raw)


def generate_actions(query_text, selected, topk, model=None):
    """Call OpenAI (RAG) and return the generated action list as a string."""
    messages = build_messages(query_text, selected, max_full_reports=topk)
    return _openai_complete(messages, model=model)
