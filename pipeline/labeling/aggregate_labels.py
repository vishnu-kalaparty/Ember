"""
Combine actions_claude.json / actions_mistral.json / actions_llama.json into
one agreed action-sentence set per report, using config.AGREEMENT_RULE
("intersection" = all 3 models agree, "majority" = >=2 of 3).

Only reports labeled "ok" by all 3 models are scored (reports missing a
model's label are reported separately so you know to finish labeling first).

Run:
    python -m pipeline.labeling.aggregate_labels
"""

import json

from .. import config
from . import common
from .claude_batch import RESULTS_JSON as CLAUDE_RESULTS
from .mistral_batch import RESULTS_JSON as MISTRAL_RESULTS
from .llama_openrouter import RESULTS_JSON as LLAMA_RESULTS

_MODEL_FILES = {
    "claude": CLAUDE_RESULTS,
    "mistral": MISTRAL_RESULTS,
    "llama": LLAMA_RESULTS,
}


def _votes_needed():
    if config.AGREEMENT_RULE == "intersection":
        return 3
    if config.AGREEMENT_RULE == "majority":
        return 2
    raise ValueError(f"Unknown AGREEMENT_RULE: {config.AGREEMENT_RULE}")


def run():
    per_model = {}
    for name, path in _MODEL_FILES.items():
        data = common.load_json_or_empty(path)
        per_model[name] = {
            bid: set(v["sentence_ids"])
            for bid, v in data.items()
            if v.get("status") == "ok"
        }
        print(f"{name}: {len(per_model[name])} reports labeled ok")

    report_sentences = common.load_report_sentences()
    all_ids = sorted(int(k) for k in report_sentences.keys())

    complete_ids = [
        bid
        for bid in all_ids
        if all(str(bid) in per_model[name] for name in _MODEL_FILES)
    ]
    incomplete_ids = [bid for bid in all_ids if bid not in complete_ids]

    votes_needed = _votes_needed()
    agreed = {}
    n_with_actions = 0
    for bid in complete_ids:
        key = str(bid)
        n_sentences = len(report_sentences[key]["sentences"])
        vote_count = {}
        for name in _MODEL_FILES:
            for sid in per_model[name][key]:
                vote_count[sid] = vote_count.get(sid, 0) + 1
        kept = sorted(
            sid
            for sid in range(1, n_sentences + 1)
            if vote_count.get(sid, 0) >= votes_needed
        )
        agreed[key] = kept
        if kept:
            n_with_actions += 1

    with open(config.AGREED_ACTIONS_JSON, "w", encoding="utf-8") as fh:
        json.dump(agreed, fh, indent=2)

    print(f"\nAgreement rule       : {config.AGREEMENT_RULE} (>= {votes_needed}/3 votes)")
    print(f"Reports w/ all 3 labels : {len(complete_ids)}")
    print(f"Reports missing a label : {len(incomplete_ids)} (finish labeling these first)")
    print(f"Reports w/ >=1 agreed action sentence : {n_with_actions}")
    print(f"Saved -> {config.AGREED_ACTIONS_JSON}")

    if incomplete_ids:
        preview = incomplete_ids[:20]
        print(f"Incomplete breach_ids (first 20): {preview}")


if __name__ == "__main__":
    run()
