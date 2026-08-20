"""
Corpus statistics for the scored test reports, plus a shortlist of
near-median reports to use as a worked example.

Prints the median report length and the median number of verb phrases per
G_r (3), which sets the output depth of 3 actions.

Run from repo root after the Table I scoring run:
    .venv/bin/python -m pipeline.eval.corpus_stats
"""

import argparse
import json
import re
import statistics as st
from pathlib import Path

from pipeline import config

NAME_RE = re.compile(r"\b(Mr|Mrs|Ms|Dr)\.\s+[A-Z]")


def pct(sorted_vals, x):
    lo = sum(1 for v in sorted_vals if v < x)
    eq = sum(1 for v in sorted_vals if v == x)
    return 100.0 * (lo + 0.5 * eq) / len(sorted_vals)


def q(v, p):
    return st.quantiles(v, n=100)[p - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=config.TEST_HYBRID_JSON)
    ap.add_argument(
        "--out", type=Path, default=config.RESULTS_DIR / "worked_example_candidates.json"
    )
    args = ap.parse_args()

    res = json.load(open(args.results))
    per = res["per_report"]["rag_hybrid_minilm_a025_k10"]
    gr = json.load(open(config.SENTENCE_GR_JSON))
    vps = json.load(open(config.REPORT_VPS_JSON))
    reports = {str(r["breach_id"]): r for r in json.load(open(config.REPORTS_CACHE))}

    scored = {b: v for b, v in per.items() if v.get("scored")}
    print("n scored:", len(scored))

    f1s = sorted(v["f1"] for v in scored.values())
    ps = sorted(v["precision"] for v in scored.values())
    rs = sorted(v["recall"] for v in scored.values())

    print("F1  median %.4f  d1 %.4f d9 %.4f" % (st.median(f1s), q(f1s, 10), q(f1s, 90)))
    print("P   median %.4f" % st.median(ps))
    print("R   median %.4f" % st.median(rs))

    wordcounts = [len(reports[b]["text"].split()) for b in scored]
    print(
        "words: median %d  iqr %d-%d"
        % (st.median(wordcounts), q(wordcounts, 25), q(wordcounts, 75))
    )
    nvp = [len(vps[b]) for b in scored]
    ngr = [len(gr[b]) for b in scored]
    print(
        "n_vp median %s mean %.2f | n_gr median %s mean %.2f"
        % (st.median(nvp), st.mean(nvp), st.median(ngr), st.mean(ngr))
    )

    cands = []
    for b, v in scored.items():
        rep = reports[b]
        txt = rep["text"]
        w = len(txt.split())
        pf = pct(f1s, v["f1"])
        if not (35 <= pf <= 65):
            continue
        if len(gr[b]) not in (2, 3):
            continue
        if len(vps[b]) != 3:
            continue
        if not (60 <= w <= 130):
            continue
        if NAME_RE.search(txt):
            continue
        cands.append(
            {
                "bid": b,
                "words": w,
                "n_gr": len(gr[b]),
                "n_vp": len(vps[b]),
                "f1": v["f1"],
                "p": v["precision"],
                "r": v["recall"],
                "pct_f1": pf,
                "pct_p": pct(ps, v["precision"]),
                "pct_r": pct(rs, v["recall"]),
                "type": rep["type"],
                "loc": rep["metadata"]["Location of Breached Information"],
                "cet": rep["metadata"]["Covered Entity Type"],
                "affected": rep["metadata"]["Individuals Affected"],
                "ba": rep["metadata"]["Business Associate Present"],
                "text": txt,
                "gr": gr[b],
                "vps": vps[b],
                "actions": v["actions"],
            }
        )

    cands.sort(key=lambda c: abs(c["pct_f1"] - 50))
    print("\ncandidates:", len(cands))
    for c in cands[:25]:
        print(
            "bid %5s w=%3d gr=%d vp=%d F1=%.4f(p%.1f) P=%.4f(p%.1f) R=%.4f(p%.1f) %s | %s | %s"
            % (
                c["bid"], c["words"], c["n_gr"], c["n_vp"], c["f1"], c["pct_f1"],
                c["p"], c["pct_p"], c["r"], c["pct_r"], c["type"], c["loc"], c["affected"],
            )
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(cands, open(args.out, "w"), indent=1)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
