"""
RAG vs ground-truth G_r text-diversity comparison (Table II) with per-report
significance testing.

The question is whether the *generated* language is as varied as the
ground-truth language.

  1. RAG actions per report are capped at ``--depth`` (default 3, the number
     generated); G_r is never truncated, since it is the reference.
  2. Every metric is computed twice: pooled at corpus level,
     and again *within* each report, giving one value per report per system.
     The per-report arrays are what make significance testing possible at all —
     a corpus-pooled number is a single point with no sampling distribution.
  3. Two-sided Mann-Whitney U on each per-report metric (RAG vs G_r), with
     rank-biserial effect size, Holm-Bonferroni correction across the metric
     family, and a paired Wilcoxon signed-rank secondary (the same 599 reports
     appear on both sides, so the paired test is better powered).
  4. Verb-phrase count per report is reported for both corpora, using the
     project's own spaCy extractor.

All inputs are read from cached artifacts; no API calls are made.

Run from repo root:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
      .venv/bin/python -m pipeline.eval.run_diversity_rag_vs_gr --depth 3

To point at a different generation run (e.g. the native n=3 batch output once
it lands) pass ``--rag-json`` and, if its cache key differs, ``--rag-key``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from .. import config
from . import diversity_utils as du

DEFAULT_RAG_JSON = config.TEST_HYBRID_JSON
DEFAULT_RAG_KEY = "rag_hybrid_minilm_a025_k10"
DEFAULT_DEPTH = 3

# Verb-phrase counts are keyed by a hash of the action text itself, so runs
# that repeat a string share work.
VP_COUNT_CACHE = config.CACHE_DIR / "action_vp_counts.json"

# Regenerated G_r verb phrases for any test id missing from REPORT_VPS_JSON.
# Written to its own file; REPORT_VPS_JSON is never overwritten.
GR_VP_BACKFILL_JSON = config.VPS_DIR / "report_vps_gr_backfill.json"

# Order fixes the Holm family. Every entry is (json_key, display label,
# "higher means more diverse"?) — the last flag is only used for the printout.
METRIC_FAMILY = [
    ("cr", "CR (gzip)", False),
    ("cr_pos", "CR:POS", False),
    ("ngd", "NGD (n=1..4)", True),
    ("srs", "SRS (n=4)", False),
    ("srs_normalized", "SRS / (L-3)", False),
    ("self_bleu_loo", "Self-BLEU, LOO", False),
    ("self_bleu_intra_set", "Self-BLEU, intra-set", False),
    ("words_per_block", "Words / block", None),
    ("words_per_action", "Words / action", None),
    ("vp_count", "Verb phrases / report", None),
]


def _load(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _get_tagger():
    """Fast tagger-only pipeline for the POS sequences (no parser/NER)."""
    import spacy

    return spacy.load(
        config.SPACY_MODEL, disable=["parser", "ner", "lemmatizer", "attribute_ruler"]
    )


def _pos_sequences(nlp, texts, batch_size=128):
    return [du.pos_tag_sequence(doc) for doc in nlp.pipe(texts, batch_size=batch_size)]


# --------------------------------------------------------------------------- #
# Verb-phrase counting
# --------------------------------------------------------------------------- #
def _vp_cache_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _load_vp_cache() -> dict:
    if VP_COUNT_CACHE.exists():
        try:
            return _load(VP_COUNT_CACHE)
        except json.JSONDecodeError:
            return {}
    return {}


def _save_vp_cache(cache: dict) -> None:
    VP_COUNT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(VP_COUNT_CACHE, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2)


def count_verb_phrases_per_report(per_item_lists: list[list[str]], label: str) -> list[int]:
    """Verb phrases per report = sum over that report's items of the phrases
    the project's spaCy extractor finds in each. Uses the FULL spaCy pipeline
    (the extractor needs the dependency parse and NER), so it is the slow part
    of this script on a cold cache; results are memoised per action string.
    """
    from ..vp_extraction import spacy_extract

    cache = _load_vp_cache()
    nlp = None
    bad_verbs = None
    n_new = 0

    counts = []
    for items in per_item_lists:
        total = 0
        for text in items:
            if not text or not text.strip():
                continue
            key = _vp_cache_key(text)
            if key not in cache:
                if nlp is None:
                    nlp = spacy_extract._load_nlp()
                    bad_verbs = spacy_extract._load_bad_verbs()
                cache[key] = len(
                    spacy_extract.extract_verb_phrases(text, nlp=nlp, bad_verbs=bad_verbs)
                )
                n_new += 1
                if n_new % 500 == 0:
                    print(f"    [{label}] extracted VPs for {n_new} new strings ...")
            total += cache[key]
        counts.append(total)

    if n_new:
        _save_vp_cache(cache)
        print(f"    [{label}] {n_new} new strings extracted, cache -> {VP_COUNT_CACHE}")
    else:
        print(f"    [{label}] all strings served from VP cache")
    return counts


def gr_verb_phrase_counts(bids: list[str], gr_items: list[list[str]]) -> tuple[list[int], dict]:
    """Prefer the pre-extracted G_r phrases at REPORT_VPS_JSON. Verify coverage
    over the test ids first; anything missing is re-extracted from that
    report's G_r sentences and written to a separate backfill file."""
    pre = _load(config.REPORT_VPS_JSON)
    missing = [b for b in bids if b not in pre]
    provenance = {
        "source": str(config.REPORT_VPS_JSON),
        "n_ids_covered_by_pre_extracted": len(bids) - len(missing),
        "n_ids_backfilled": len(missing),
        "backfilled_ids": missing,
    }

    backfill = {}
    if missing:
        print(f"  [G_r] {len(missing)} test ids missing from REPORT_VPS_JSON; re-extracting")
        from ..vp_extraction import spacy_extract

        nlp = spacy_extract._load_nlp()
        bad_verbs = spacy_extract._load_bad_verbs()
        by_bid = dict(zip(bids, gr_items))
        for b in missing:
            vps = []
            for sent in by_bid[b]:
                vps.extend(
                    r["verb_phrase"]
                    for r in spacy_extract.extract_verb_phrases(
                        sent, nlp=nlp, bad_verbs=bad_verbs
                    )
                )
            backfill[b] = vps
        with open(GR_VP_BACKFILL_JSON, "w", encoding="utf-8") as fh:
            json.dump(backfill, fh, indent=2)
        provenance["backfill_file"] = str(GR_VP_BACKFILL_JSON)
        print(f"  [G_r] backfill saved -> {GR_VP_BACKFILL_JSON}")
    else:
        print(f"  [G_r] REPORT_VPS_JSON covers all {len(bids)} test ids")

    counts = [len(pre.get(b, backfill.get(b, []))) for b in bids]
    return counts, provenance


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_corpus_and_per_report(nlp, block_texts, per_item_lists, label, vp_counts):
    """Corpus-pooled metrics (identical definitions to the 3-corpus driver)
    plus the per-report arrays used for significance testing."""
    token_lists = [du.tokenize_words(t) for t in block_texts]
    pos_seqs = _pos_sequences(nlp, block_texts)

    srs = du.self_repetition_score(token_lists, n=config.DIVERSITY_SELF_REP_N)

    print(f"  [{label}] leave-one-out Self-BLEU over {len(block_texts)} blocks ...")
    t0 = time.perf_counter()
    loo = du.leave_one_out_self_bleu(
        token_lists,
        max_n=config.DIVERSITY_SELF_BLEU_MAX_N,
        sample_size=None,
    )
    loo_secs = time.perf_counter() - t0
    n_pairs = len(block_texts) * (len(block_texts) - 1)
    print(f"  [{label}] LOO Self-BLEU done in {loo_secs:.1f}s ({n_pairs:,} exact BLEU evals)")

    intra = [
        du.intra_set_self_bleu(
            [du.tokenize_words(a) for a in items if a and a.strip()],
            max_n=config.DIVERSITY_SELF_BLEU_MAX_N,
        )
        for items in per_item_lists
    ]
    intra_present = [v for v in intra if v is not None]

    all_items = [a for items in per_item_lists for a in items if a and a.strip()]

    corpus = {
        "label": label,
        "n_docs": len(block_texts),
        "cr": du.compression_ratio(block_texts),
        "cr_pos": du.compression_ratio(pos_seqs),
        "ngd": du.ngram_diversity_score(token_lists, max_n=config.DIVERSITY_NGD_MAX_N),
        "self_repetition": {k: v for k, v in srs.items() if k != "per_doc_srs"},
        "self_bleu_corpus": du.pairwise_self_bleu(
            token_lists,
            max_n=config.DIVERSITY_SELF_BLEU_MAX_N,
            max_pairs=config.DIVERSITY_SELF_BLEU_MAX_PAIRS,
        ),
        "self_bleu_intra_set": {
            "mean": sum(intra_present) / len(intra_present) if intra_present else None,
            "n_reports": len(intra_present),
        },
        "length_words_per_action": du.word_count_stats(all_items),
        "length_words_per_block": du.word_count_stats(block_texts),
    }

    per_report = {
        "cr": du.per_doc_compression_ratio(block_texts),
        "cr_pos": du.per_doc_compression_ratio(pos_seqs),
        "ngd": du.per_doc_ngram_diversity(token_lists, max_n=config.DIVERSITY_NGD_MAX_N),
        "srs": srs["per_doc_srs"],
        "srs_normalized": du.normalized_self_repetition(
            srs["per_doc_srs"], token_lists, n=config.DIVERSITY_SELF_REP_N
        ),
        "self_bleu_loo": loo["per_doc"],
        "self_bleu_intra_set": intra,
        "words_per_block": du.per_doc_word_counts(block_texts),
        "words_per_action": du.per_doc_words_per_item(per_item_lists),
        "vp_count": vp_counts,
    }

    runtime = {
        "self_bleu_loo_seconds": loo_secs,
        "self_bleu_loo_pairs": n_pairs,
        "self_bleu_loo_sampled": loo["sampled"],
        "self_bleu_loo_comparisons_per_doc": loo["comparisons_per_doc"],
    }
    return corpus, per_report, runtime


def run_statistics(rag_per_report: dict, gr_per_report: dict) -> dict:
    """Mann-Whitney U (primary) + Wilcoxon (paired secondary) per metric, then
    Holm-Bonferroni over the whole metric family."""
    tests = {}
    for key, label, _ in METRIC_FAMILY:
        res = du.mann_whitney_compare(rag_per_report[key], gr_per_report[key], paired=True)
        res["label"] = label
        tests[key] = res

    keys = [k for k, _, _ in METRIC_FAMILY]
    mw_adj = du.holm_bonferroni([tests[k]["p_value"] for k in keys])
    wx_adj = du.holm_bonferroni([tests[k]["wilcoxon_p"] for k in keys])
    for k, pa, wa in zip(keys, mw_adj, wx_adj):
        tests[k]["p_value_holm"] = pa
        tests[k]["wilcoxon_p_holm"] = wa

    return {
        "test": "two-sided Mann-Whitney U, RAG vs G_r, on per-report values",
        "effect_size": "rank-biserial r = 1 - 2U/(n1*n2); r>0 means RAG ranks above G_r",
        "correction": f"Holm-Bonferroni across the {len(keys)}-metric family",
        "secondary": (
            "Wilcoxon signed-rank on the report-aligned differences (same 599 "
            "reports on both sides), Holm-corrected over the same family"
        ),
        "family_size": len(keys),
        "metrics": tests,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(
    depth: int = DEFAULT_DEPTH,
    rag_json: Path = DEFAULT_RAG_JSON,
    rag_key: str = DEFAULT_RAG_KEY,
    out_json: Path | None = None,
    tag: str = "",
    provenance_note: str = "",
):
    rag_json = Path(rag_json)
    if out_json is None:
        out_json = config.RESULTS_DIR / f"test_diversity_rag_vs_gr{tag}.json"
    out_json = Path(out_json)

    print(f"RAG source : {rag_json}")
    print(f"RAG key    : {rag_key}")
    print(f"Depth      : top-{depth} actions per report (G_r is never truncated)")
    print(f"Output     : {out_json}")

    rag_data = _load(rag_json)
    sentence_gr = _load(config.SENTENCE_GR_JSON)

    rag_per = rag_data["per_report"][rag_key]
    bids = sorted((b for b, v in rag_per.items() if v.get("scored")), key=int)
    print(f"n reports (RAG-scored): {len(bids)}")

    rag_items, gr_items = [], []
    truncated_reports = 0
    short_reports = 0
    for b in bids:
        acts = [a for a in (rag_per[b].get("actions") or []) if a and a.strip()]
        if len(acts) > depth:
            truncated_reports += 1
        elif len(acts) < depth:
            short_reports += 1
        rag_items.append(acts[:depth])
        gr_items.append([s for s in (sentence_gr.get(b) or []) if s and s.strip()])

    print(
        f"  {truncated_reports} reports truncated to {depth}; "
        f"{short_reports} had fewer than {depth} actions to begin with"
    )
    missing_gr = [b for b, g in zip(bids, gr_items) if not g]
    if missing_gr:
        raise RuntimeError(f"{len(missing_gr)} bids have empty G_r: {missing_gr[:10]}")

    rag_block = [du.build_block_text(items) for items in rag_items]
    gr_block = [du.build_block_text(items) for items in gr_items]

    print("\nCounting verb phrases (full spaCy pipeline, cached)")
    rag_vp = count_verb_phrases_per_report(rag_items, "RAG")
    gr_vp, gr_vp_provenance = gr_verb_phrase_counts(bids, gr_items)

    nlp = _get_tagger()
    print(f"\nspaCy tagger loaded ({config.SPACY_MODEL}, parser/ner/lemmatizer disabled)")

    print("\n[RAG] corpus + per-report metrics")
    rag_corpus, rag_pr, rag_runtime = compute_corpus_and_per_report(
        nlp, rag_block, rag_items, "rag", rag_vp
    )
    print("\n[G_r] corpus + per-report metrics")
    gr_corpus, gr_pr, gr_runtime = compute_corpus_and_per_report(
        nlp, gr_block, gr_items, "g_r", gr_vp
    )

    print("\nSignificance testing")
    stats_block = run_statistics(rag_pr, gr_pr)

    result = {
        "n_reports": len(bids),
        "depth": depth,
        "bids": bids,
        "methodology": {
            "corpora": (
                f"Two corpora over the same {len(bids)} test reports: RAG actions "
                f"truncated to top-{depth}, and human G_r sentences (3-LLM-"
                "intersection agreed remediation sentences, never truncated)."
            ),
            "provenance_note": provenance_note,
            "document_definition": (
                "One document = one report's items joined by a space "
                "(build_block_text). Corpus metrics pool across documents; "
                "per-report metrics recompute inside a single document."
            ),
            "ngd_max_n": config.DIVERSITY_NGD_MAX_N,
            "self_repetition_n": config.DIVERSITY_SELF_REP_N,
            "self_bleu_max_n": config.DIVERSITY_SELF_BLEU_MAX_N,
            "spacy_model": config.SPACY_MODEL,
            "min_vp_words": config.MIN_VP_WORDS,
            "sources": {
                "rag_json": str(rag_json),
                "rag_key": rag_key,
                "sentence_gr_json": str(config.SENTENCE_GR_JSON),
                "g_r_verb_phrases": gr_vp_provenance,
                "rag_vp_count_cache": str(VP_COUNT_CACHE),
            },
            "caveats": {
                "per_report_cr": (
                    "Per-report gzip CR is length-confounded: gzip's fixed "
                    "header and 32 KiB window are not amortised by a "
                    "~50-token block, so shorter blocks score lower CR "
                    "independently of redundancy. Read with words_per_block."
                ),
                "self_bleu_intra_set_n": (
                    "A report with a single item has no within-report pair, so "
                    "its intra-set Self-BLEU is undefined and it is dropped. "
                    "Effective n is below the report count for G_r."
                ),
                "vp_count_asymmetry": (
                    "RAG verb phrases are counted over the generated "
                    "actions; G_r verb phrases are counted over all of that "
                    "report's agreed sentences."
                ),
            },
        },
        "runtime": {"rag": rag_runtime, "g_r": gr_runtime},
        "corpus_level": {"rag": rag_corpus, "g_r": gr_corpus},
        "per_report": {"rag": rag_pr, "g_r": gr_pr},
        "statistics": stats_block,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    _print_report(result)
    print(f"\nSaved -> {out_json}")
    return result


def _g(d, path):
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _fmt(v, width=12, prec=4):
    if v is None:
        return f"{'n/a':>{width}}"
    if isinstance(v, float):
        return f"{v:>{width}.{prec}f}"
    return f"{v:>{width}}"


def _fmt_p(p, width=11):
    if p is None:
        return f"{'n/a':>{width}}"
    if p < 1e-4:
        return f"{p:>{width}.2e}"
    return f"{p:>{width}.4f}"


def _print_report(result):
    depth = result["depth"]
    print("\n" + "=" * 100)
    print(f"CORPUS-LEVEL (pooled)   RAG top-{depth} vs human G_r   n={result['n_reports']} reports")
    print("=" * 100)
    print(f"{'Metric':<26}{'RAG':>14}{'G_r':>14}")
    rows = [
        ("CR", ["cr", "cr"]),
        ("CR:POS", ["cr_pos", "cr"]),
        ("NGD", ["ngd", "ngd"]),
        ("SRS (n=4)", ["self_repetition", "mean_srs"]),
        ("Self-BLEU, corpus", ["self_bleu_corpus", "self_bleu"]),
        ("Self-BLEU, intra-set", ["self_bleu_intra_set", "mean"]),
        ("Words/action, mean", ["length_words_per_action", "mean"]),
        ("Words/block, mean", ["length_words_per_block", "mean"]),
    ]
    for label, path in rows:
        cells = "".join(
            _fmt(_g(result["corpus_level"][s], path), 14) for s in ("rag", "g_r")
        )
        print(f"{label:<26}{cells}")
    n_intra_rag = _g(result["corpus_level"]["rag"], ["self_bleu_intra_set", "n_reports"])
    n_intra_gr = _g(result["corpus_level"]["g_r"], ["self_bleu_intra_set", "n_reports"])
    print(f"{'  (intra-set n reports)':<26}{n_intra_rag:>14}{n_intra_gr:>14}")

    print("\n" + "=" * 100)
    print("PER-REPORT + SIGNIFICANCE   two-sided Mann-Whitney U, RAG vs G_r, Holm-corrected")
    print("=" * 100)
    hdr = (
        f"{'Metric':<24}{'n':>5}{'RAG med':>11}{'G_r med':>11}"
        f"{'U':>12}{'p':>11}{'p(Holm)':>11}{'r':>8}{'Wilcox p':>11}{'W p(Holm)':>11}"
    )
    print(hdr)
    print("-" * len(hdr))
    for key, label, _ in METRIC_FAMILY:
        t = result["statistics"]["metrics"][key]
        n_disp = f"{t['n_a']}/{t['n_b']}" if t["n_a"] != t["n_b"] else str(t["n_a"])
        print(
            f"{label:<24}{n_disp:>5}"
            f"{_fmt(t['median_a'], 11)}{_fmt(t['median_b'], 11)}"
            f"{_fmt(t['u_statistic'], 12, 0)}"
            f"{_fmt_p(t['p_value'])}{_fmt_p(t['p_value_holm'])}"
            f"{_fmt(t['rank_biserial_r'], 8, 3)}"
            f"{_fmt_p(t['wilcoxon_p'])}{_fmt_p(t['wilcoxon_p_holm'])}"
        )
    print("-" * len(hdr))
    print("r>0: RAG ranks above G_r.  n shown as RAG/G_r when the two differ")
    print("(a single-item report has no within-report pair, so intra-set drops it).")

    rt = result["runtime"]
    print(
        f"\nLOO Self-BLEU runtime: RAG {rt['rag']['self_bleu_loo_seconds']:.1f}s "
        f"({rt['rag']['self_bleu_loo_pairs']:,} pairs), "
        f"G_r {rt['g_r']['self_bleu_loo_seconds']:.1f}s "
        f"({rt['g_r']['self_bleu_loo_pairs']:,} pairs); "
        f"sampled={rt['rag']['self_bleu_loo_sampled']}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    ap.add_argument("--rag-json", type=Path, default=DEFAULT_RAG_JSON)
    ap.add_argument("--rag-key", default=DEFAULT_RAG_KEY)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--tag",
        default="",
        help="suffix appended to the default output filename, e.g. _trunc_from_5action",
    )
    ap.add_argument(
        "--note",
        default="",
        help="free-text provenance note stored in the output JSON",
    )
    args = ap.parse_args()
    run(
        depth=args.depth,
        rag_json=args.rag_json,
        rag_key=args.rag_key,
        out_json=args.out,
        tag=args.tag,
        provenance_note=args.note,
    )


if __name__ == "__main__":
    main()
