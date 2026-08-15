"""
Report-local BERTScore P/R/F1 (+ MoverScore): score RAG-generated actions
against that SAME report's G_r (its own 3-LLM-agreed remediation text).

Both metrics use the RoBERTa-large token embeddings in `scoring.py` and are
always computed together (they share the same token encodings).
"""

import numpy as np

from . import scoring


def score_actions_vs_report_local_g(actions, g_r_phrases):
    """actions: list[str] (5 RAG actions). g_r_phrases: list[str] (report G_r).

    Returns dict with report-level mean BERTScore P/R/F1 and mean MoverScore
    (each meaned over actions), plus the per-action breakdown. Returns None
    if either list is empty (unscoreable).
    """
    if not actions or not g_r_phrases:
        return None

    action_tokens = scoring.encode_tokens(actions)
    g_tokens = scoring.encode_tokens(g_r_phrases)

    per_action = []
    for a, toks in zip(actions, action_tokens):
        p, r, f1 = scoring.best_bertscore_prf(toks, g_tokens)
        mover = scoring.best_moverscore(toks, g_tokens)
        per_action.append(
            {"action": a, "precision": p, "recall": r, "f1": f1, "moverscore": mover}
        )

    return {
        "n_actions": len(actions),
        "n_g_r": len(g_r_phrases),
        "precision": float(np.mean([x["precision"] for x in per_action])),
        "recall": float(np.mean([x["recall"] for x in per_action])),
        "f1": float(np.mean([x["f1"] for x in per_action])),
        "moverscore": float(np.mean([x["moverscore"] for x in per_action])),
        "per_action": per_action,
    }
