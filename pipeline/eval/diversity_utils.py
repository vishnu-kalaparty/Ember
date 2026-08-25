"""
Text diversity metrics from Shaib et al. 2024 ("Standardizing the Measurement
of Text Diversity", arXiv:2403.00553), adapted for EMBER's "k actions per
report" generation format.

"Document" for every corpus-level metric = one report's generated actions (or
G_r phrases) concatenated into a single string (`build_block_text`). All
functions here are metric-only (no I/O, no config); `run_diversity_rag_vs_gr.py`
wires them to the RAG and G_r caches.

Metrics implemented:
  - compression_ratio       : gzip CR on raw text or a POS-tag sequence
  - pos_tag_sequence        : spaCy tagger only (fast: tok2vec+tagger, no parser/NER)
  - ngram_diversity_score   : NGD, sum_{n=1..N} unique/total n-grams, corpus-pooled
  - self_repetition_score   : SRS(d) = log(1 + #other-docs containing each of d's n-grams)
  - bleu_sentence           : single-reference BLEU-N with Chen&Cherry-style smoothing
  - pairwise_self_bleu      : mean single-reference BLEU over all ordered pairs in a set
                              (Shaib et al.'s "homogenization score" with BLEU)
  - word_count_stats        : length confound bookkeeping

NOTE on Self-Repetition's n: our "documents" are 5 short actions (mean
6-17 words each, `MIN_VP_WORDS = 3` floor on individual G_r phrases)
concatenated with a plain space into a ~19-84 word block, so the paper's
default n=4 word-gram can, for a 3-word item, only ever match via a
cross-item boundary artifact rather than real within-item repetition. We
checked n=3 vs n=4 directly: both give the identical RAG vs G_r
ranking with comparable gaps, so we keep the paper's n=4 default (avoids
the appearance of parameter-shopping) with this as a verified robustness
check rather than a live concern.
"""

from __future__ import annotations

import gzip
import math
import random
import re
from collections import Counter

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize_words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def build_block_text(items: list[str]) -> str:
    """Concatenate a report's action/G_r strings into one 'document'."""
    return " ".join(s.strip() for s in items if s and s.strip())


# --------------------------------------------------------------------------- #
# Compression Ratio (CR) and CR:POS
# --------------------------------------------------------------------------- #
def compression_ratio(docs: list[str]) -> dict:
    """gzip CR over the concatenation of `docs`. Higher = more redundant.

    CR = uncompressed_bytes / compressed_bytes (gzip level 9, UTF-8).
    """
    blob = "\n".join(docs).encode("utf-8")
    if not blob:
        return {"cr": None, "raw_bytes": 0, "compressed_bytes": 0, "n_docs": len(docs)}
    compressed = gzip.compress(blob, compresslevel=9)
    return {
        "cr": len(blob) / len(compressed),
        "raw_bytes": len(blob),
        "compressed_bytes": len(compressed),
        "n_docs": len(docs),
    }


def pos_tag_sequence(doc) -> str:
    """`doc` is a spaCy Doc (tagger-only pipeline). Returns a space-joined
    Penn-Treebank tag sequence, e.g. 'VB NN IN DT NN'."""
    return " ".join(t.tag_ for t in doc if not t.is_space)


# --------------------------------------------------------------------------- #
# N-Gram Diversity Score (corpus-pooled)
# --------------------------------------------------------------------------- #
def _ngrams(tokens: list[str], n: int):
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def ngram_diversity_score(token_lists: list[list[str]], max_n: int = 4) -> dict:
    """NGD = sum_{n=1..max_n} (unique n-grams / total n-grams), pooled over the
    whole corpus (all docs' n-grams combined into one bag per n)."""
    per_n = {}
    total = 0.0
    for n in range(1, max_n + 1):
        bag = []
        for toks in token_lists:
            bag.extend(_ngrams(toks, n))
        n_total = len(bag)
        n_unique = len(set(bag))
        ratio = (n_unique / n_total) if n_total else None
        per_n[n] = {"unique": n_unique, "total": n_total, "ratio": ratio}
        if ratio is not None:
            total += ratio
    return {"ngd": total, "per_n": per_n, "max_n": max_n, "n_docs": len(token_lists)}


# --------------------------------------------------------------------------- #
# Self-Repetition Score (verbatim cross-document n-gram reuse)
# --------------------------------------------------------------------------- #
def self_repetition_score(token_lists: list[list[str]], n: int = 3) -> dict:
    """SRS(d) = log(1 + sum_i N_i) where N_i = # OTHER docs containing d's
    i-th *unique* n-gram. Returns the corpus mean plus per-doc scores (so the
    highest-SRS docs can be pulled for qualitative "most-repeated phrase"
    examples in the paper appendix)."""
    doc_ngram_sets = [set(_ngrams(toks, n)) for toks in token_lists]

    inverted: dict[tuple, set[int]] = {}
    for doc_idx, grams in enumerate(doc_ngram_sets):
        for g in grams:
            inverted.setdefault(g, set()).add(doc_idx)

    per_doc_scores = []
    for doc_idx, grams in enumerate(doc_ngram_sets):
        if not grams:
            per_doc_scores.append(0.0)
            continue
        total_other = sum(len(inverted[g]) - 1 for g in grams)
        per_doc_scores.append(math.log(1 + total_other))

    mean_score = sum(per_doc_scores) / len(per_doc_scores) if per_doc_scores else None
    top_idx = sorted(range(len(per_doc_scores)), key=lambda i: per_doc_scores[i], reverse=True)[:10]
    return {
        "mean_srs": mean_score,
        "n": n,
        "n_docs": len(token_lists),
        "per_doc_srs": per_doc_scores,
        "top_doc_indices": top_idx,
    }


# --------------------------------------------------------------------------- #
# BLEU (single-reference) and Self-BLEU / homogenization
# --------------------------------------------------------------------------- #
def bleu_sentence(hyp_tokens: list[str], ref_tokens: list[str], max_n: int = 4) -> float:
    """Single-reference BLEU-max_n with add-epsilon smoothing (Chen & Cherry
    method-1 style: avoids a hard 0 from one empty higher-order n-gram match,
    which is common at our short (3-10 word) action-phrase lengths)."""
    if not hyp_tokens or not ref_tokens:
        return 0.0

    log_precisions = []
    for n in range(1, max_n + 1):
        hyp_ngrams = Counter(_ngrams(hyp_tokens, n))
        ref_ngrams = Counter(_ngrams(ref_tokens, n))
        if not hyp_ngrams:
            log_precisions.append(math.log(1e-9))
            continue
        overlap = sum(min(c, ref_ngrams[g]) for g, c in hyp_ngrams.items())
        total = sum(hyp_ngrams.values())
        precision = overlap / total if total else 0.0
        if precision <= 0.0:
            precision = 1.0 / (2 * total)  # smoothing floor, avoids log(0)
        log_precisions.append(math.log(precision))

    geo_mean = math.exp(sum(log_precisions) / max_n)

    hyp_len, ref_len = len(hyp_tokens), len(ref_tokens)
    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / hyp_len)
    return bp * geo_mean


def pairwise_self_bleu(
    token_lists: list[list[str]],
    max_n: int = 4,
    max_pairs: int | None = None,
    seed: int = 0,
) -> dict:
    """Mean single-reference BLEU over all ordered pairs (i != j) in the set.
    This is Shaib et al.'s "homogenization score" with BLEU as the pairwise
    metric M — the doc-to-doc similarity view, distinct from CR's aggregate
    compressibility view.

    If max_pairs is set and the full O(n^2) pair count exceeds it, a random
    sample of that many ordered pairs is used instead (reproducible via seed).
    """
    n_docs = len(token_lists)
    if n_docs < 2:
        return {"self_bleu": None, "n_docs": n_docs, "n_pairs": 0, "sampled": False}

    all_pairs = [(i, j) for i in range(n_docs) for j in range(n_docs) if i != j]
    sampled = False
    if max_pairs is not None and len(all_pairs) > max_pairs:
        rng = random.Random(seed)
        all_pairs = rng.sample(all_pairs, max_pairs)
        sampled = True

    scores = [bleu_sentence(token_lists[i], token_lists[j], max_n=max_n) for i, j in all_pairs]
    return {
        "self_bleu": sum(scores) / len(scores) if scores else None,
        "n_docs": n_docs,
        "n_pairs": len(all_pairs),
        "sampled": sampled,
    }


def stratified_pairwise_bleu(
    token_lists: list[list[str]],
    same_group_fn,
    max_n: int = 4,
    max_pairs_per_bucket: int | None = None,
    seed: int = 0,
) -> dict:
    """Split all ordered pairs (i != j) into "same-group" vs "different-group"
    by `same_group_fn(i, j) -> bool`, and report mean BLEU separately for each
    bucket. Used to test *mechanisms* behind an aggregate Self-BLEU number:
      - same_group_fn = "same breach type" -> is cross-report similarity
        concentrated within genuinely similar contexts (appropriate) or
        spread indiscriminately across unrelated ones (not)?
      - same_group_fn = "retrieved neighbor sets overlap" -> is RAG's
        elevated corpus Self-BLEU explained by shared retrieval context?
    Each bucket is capped/sampled to `max_pairs_per_bucket` for tractability.
    """
    n = len(token_lists)
    same_pairs, diff_pairs = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            (same_pairs if same_group_fn(i, j) else diff_pairs).append((i, j))

    rng = random.Random(seed)
    same_sampled = len(same_pairs) > (max_pairs_per_bucket or float("inf"))
    diff_sampled = len(diff_pairs) > (max_pairs_per_bucket or float("inf"))
    if max_pairs_per_bucket is not None:
        if same_sampled:
            same_pairs = rng.sample(same_pairs, max_pairs_per_bucket)
        if diff_sampled:
            diff_pairs = rng.sample(diff_pairs, max_pairs_per_bucket)

    def _mean_bleu(pairs):
        if not pairs:
            return None
        return sum(bleu_sentence(token_lists[i], token_lists[j], max_n=max_n) for i, j in pairs) / len(
            pairs
        )

    return {
        "same_group_mean_bleu": _mean_bleu(same_pairs),
        "same_group_n_pairs": len(same_pairs),
        "same_group_sampled": same_sampled,
        "diff_group_mean_bleu": _mean_bleu(diff_pairs),
        "diff_group_n_pairs": len(diff_pairs),
        "diff_group_sampled": diff_sampled,
    }


def intra_set_self_bleu(items_tokens: list[list[str]], max_n: int = 4) -> float | None:
    """Mean pairwise BLEU among the (typically 5) items of ONE report's own
    action/G_r list. None if fewer than 2 items."""
    n = len(items_tokens)
    if n < 2:
        return None
    scores = [
        bleu_sentence(items_tokens[i], items_tokens[j], max_n=max_n)
        for i in range(n)
        for j in range(n)
        if i != j
    ]
    return sum(scores) / len(scores) if scores else None


# --------------------------------------------------------------------------- #
# Length bookkeeping (confound control, reported alongside every metric above)
# --------------------------------------------------------------------------- #
def word_count_stats(texts: list[str]) -> dict:
    counts = [len(tokenize_words(t)) for t in texts if t and t.strip()]
    if not counts:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    counts_sorted = sorted(counts)
    n = len(counts_sorted)
    median = (
        counts_sorted[n // 2]
        if n % 2
        else (counts_sorted[n // 2 - 1] + counts_sorted[n // 2]) / 2
    )
    return {
        "n": n,
        "mean": sum(counts) / n,
        "median": median,
        "min": counts_sorted[0],
        "max": counts_sorted[-1],
    }


# =========================================================================== #
# PER-REPORT variants (used for the RAG-vs-G_r significance tests)
#
# Everything above is corpus-pooled: one number per corpus, so there is no
# sampling distribution to test against. The functions below recompute the
# same quantities *within* each report's block, yielding one value per report
# and therefore a paired sample of 599 that Mann-Whitney / Wilcoxon can act on.
# =========================================================================== #
def per_doc_compression_ratio(block_texts: list[str]) -> list[float | None]:
    """gzip CR computed *within* each block independently.

    Note this is strongly length-confounded: gzip's ~18-byte header plus its
    32 KiB window mean a short block cannot amortise the format overhead, so
    short blocks score low CR regardless of how repetitive they are. Always
    read alongside `per_doc_word_counts`.
    """
    return [compression_ratio([t])["cr"] if t and t.strip() else None for t in block_texts]


def per_doc_ngram_diversity(token_lists: list[list[str]], max_n: int = 4) -> list[float | None]:
    """NGD restricted to a single block's own n-grams (sum_{n=1..max_n} of
    unique/total inside that one block)."""
    out: list[float | None] = []
    for toks in token_lists:
        if not toks:
            out.append(None)
            continue
        out.append(ngram_diversity_score([toks], max_n=max_n)["ngd"])
    return out


def normalized_self_repetition(
    per_doc_srs: list[float], token_lists: list[list[str]], n: int = 4
) -> list[float | None]:
    """SRS(d) divided by the number of n-gram slots in d, i.e. (L - n + 1).

    Raw SRS is log(1 + <count of other docs sharing each of d's n-grams>), and
    that inner count scales with how many n-grams d has, so longer blocks score
    higher for free. Dividing by (L - n + 1) removes the mechanical part of
    that. None when the block is too short to contain a single n-gram.
    """
    out: list[float | None] = []
    for srs, toks in zip(per_doc_srs, token_lists):
        slots = len(toks) - n + 1
        out.append(srs / slots if slots > 0 else None)
    return out


def leave_one_out_self_bleu(
    token_lists: list[list[str]],
    max_n: int = 4,
    sample_size: int | None = None,
    seed: int = 0,
) -> dict:
    """Per-document Self-BLEU: block i scored against every *other* block in
    the same corpus, mean-reduced.

    Numerically identical to
    ``mean_j!=i bleu_sentence(token_lists[i], token_lists[j])`` but computed
    with the n-gram Counters hoisted out of the pair loop and the per-pair
    overlap driven by a C-level set intersection, which takes the full 599x598
    grid from ~24 s to ~3 s. The mean of the returned per-doc list equals
    `pairwise_self_bleu`'s corpus number exactly.

    `sample_size`, if set and smaller than n_docs-1, restricts each document's
    comparison set to that many other documents drawn with a fixed seed. The
    returned dict always states whether sampling was applied.
    """
    n_docs = len(token_lists)
    if n_docs < 2:
        return {
            "per_doc": [None] * n_docs,
            "mean": None,
            "n_docs": n_docs,
            "sampled": False,
            "sample_size": None,
            "comparisons_per_doc": 0,
        }

    counters = [[Counter(_ngrams(toks, n)) for n in range(1, max_n + 1)] for toks in token_lists]
    totals = [[sum(c.values()) for c in doc_c] for doc_c in counters]
    keysets = [[set(c.keys()) for c in doc_c] for doc_c in counters]
    lengths = [len(toks) for toks in token_lists]

    sampled = sample_size is not None and sample_size < n_docs - 1
    rng = random.Random(seed) if sampled else None
    log, exp = math.log, math.exp
    log_eps = log(1e-9)

    per_doc: list[float | None] = []
    for i in range(n_docs):
        if not token_lists[i]:
            per_doc.append(0.0)
            continue
        if sampled:
            others = rng.sample([j for j in range(n_docs) if j != i], sample_size)
        else:
            others = [j for j in range(n_docs) if j != i]

        hyp_counters, hyp_totals, hyp_keys, li = counters[i], totals[i], keysets[i], lengths[i]
        acc = 0.0
        for j in others:
            if not token_lists[j]:
                continue  # bleu_sentence returns 0.0 against an empty reference
            log_p = 0.0
            for k in range(max_n):
                total = hyp_totals[k]
                if not total:
                    log_p += log_eps
                    continue
                hc, rc = hyp_counters[k], counters[j][k]
                overlap = 0
                for g in hyp_keys[k] & keysets[j][k]:
                    a, b = hc[g], rc[g]
                    overlap += a if a < b else b
                precision = overlap / total
                if precision <= 0.0:
                    precision = 1.0 / (2 * total)
                log_p += log(precision)
            geo_mean = exp(log_p / max_n)
            lj = lengths[j]
            acc += geo_mean if li >= lj else exp(1 - lj / li) * geo_mean
        per_doc.append(acc / len(others) if others else None)

    present = [v for v in per_doc if v is not None]
    return {
        "per_doc": per_doc,
        "mean": sum(present) / len(present) if present else None,
        "n_docs": n_docs,
        "sampled": sampled,
        "sample_size": sample_size if sampled else None,
        "comparisons_per_doc": sample_size if sampled else n_docs - 1,
    }


def per_doc_word_counts(block_texts: list[str]) -> list[int]:
    return [len(tokenize_words(t)) for t in block_texts]


def per_doc_words_per_item(per_item_lists: list[list[str]]) -> list[float | None]:
    """Mean words per action/phrase inside each report."""
    out: list[float | None] = []
    for items in per_item_lists:
        kept = [s for s in items if s and s.strip()]
        if not kept:
            out.append(None)
            continue
        out.append(sum(len(tokenize_words(s)) for s in kept) / len(kept))
    return out


# --------------------------------------------------------------------------- #
# Significance testing over the per-report samples
# --------------------------------------------------------------------------- #
def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def mann_whitney_compare(a: list, b: list, paired: bool = True) -> dict:
    """Two-sided Mann-Whitney U of sample `a` (RAG) against `b` (G_r), with
    rank-biserial effect size and a paired Wilcoxon signed-rank secondary.

    r = 1 - 2U/(n1*n2), with U taken as U(a). r > 0 means `a` tends to sit
    above `b`. The two samples are the same 599 reports measured under two
    systems, so Wilcoxon on the report-aligned differences is the better-
    powered test; MWU is reported as the assumption-light primary. `None`
    entries are dropped independently for MWU and pairwise for Wilcoxon, so
    the two tests can legitimately report different n.
    """
    from scipy import stats

    a_clean = [float(v) for v in a if v is not None and not math.isnan(float(v))]
    b_clean = [float(v) for v in b if v is not None and not math.isnan(float(v))]

    out = {
        "n_a": len(a_clean),
        "n_b": len(b_clean),
        "median_a": _median(a_clean),
        "median_b": _median(b_clean),
        "mean_a": sum(a_clean) / len(a_clean) if a_clean else None,
        "mean_b": sum(b_clean) / len(b_clean) if b_clean else None,
        "u_statistic": None,
        "p_value": None,
        "rank_biserial_r": None,
        "wilcoxon_statistic": None,
        "wilcoxon_p": None,
        "wilcoxon_n_pairs": 0,
    }
    if len(a_clean) < 2 or len(b_clean) < 2:
        return out

    u, p = stats.mannwhitneyu(a_clean, b_clean, alternative="two-sided")
    out["u_statistic"] = float(u)
    out["p_value"] = float(p)
    out["rank_biserial_r"] = 1.0 - (2.0 * float(u)) / (len(a_clean) * len(b_clean))

    if paired and len(a) == len(b):
        pairs = [
            (float(x), float(y))
            for x, y in zip(a, b)
            if x is not None
            and y is not None
            and not math.isnan(float(x))
            and not math.isnan(float(y))
        ]
        diffs = [x - y for x, y in pairs]
        out["wilcoxon_n_pairs"] = len(pairs)
        if any(d != 0 for d in diffs) and len(diffs) >= 2:
            w, wp = stats.wilcoxon(
                [x for x, _ in pairs], [y for _, y in pairs], alternative="two-sided"
            )
            out["wilcoxon_statistic"] = float(w)
            out["wilcoxon_p"] = float(wp)
    return out


def holm_bonferroni(p_values: list[float | None]) -> list[float | None]:
    """Holm-Bonferroni step-down adjusted p-values, order-preserving.

    Positions holding None (test not run) are skipped entirely and do not
    count toward the family size m.
    """
    idx = [i for i, p in enumerate(p_values) if p is not None]
    m = len(idx)
    adjusted: list[float | None] = [None] * len(p_values)
    if m == 0:
        return adjusted

    order = sorted(idx, key=lambda i: p_values[i])
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p_values[i]
        running = max(running, val)  # enforce monotonicity of the step-down
        adjusted[i] = min(1.0, running)
    return adjusted
