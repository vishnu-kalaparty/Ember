"""
VerbPhrase extraction on the 3-LLM-agreed action sentences, via spaCy
dependency parsing, following the original EMBER verb-phrase algorithm: find
qualifying verbs (kept_verb_deps, not in bad_verbs), walk each verb's subtree,
strip the leading subject/aux/connective "bridge", render the remaining
tokens (lemmatizing the head verb, collapsing named entities to tags), trim
trailing junk words, and keep phrases with >= MIN_VP_WORDS words.

Input sentences are the report-local, 3-LLM-agreed action sentences (val +
test).

Run:
    python -m pipeline.vp_extraction.spacy_extract
"""

import json

from .. import config
from ..labeling import common

_ENTITY_TAGS = {"ORG", "GPE", "CARDINAL", "DATE", "PERSON"}
_ENTITY_WHITELIST = {"CE", "BA", "BAA", "HHS", "OCR", "HIPAA"}
_TRAILING_JUNK = {",", "and", "had", "that", "(", "who", "may", "or"}

_nlp = None
_bad_verbs = None


def _load_nlp():
    global _nlp
    if _nlp is None:
        import spacy

        _nlp = spacy.load(config.SPACY_MODEL)
    return _nlp


def _load_bad_verbs():
    global _bad_verbs
    if _bad_verbs is None:
        with open(config.BAD_VERBS_CSV, encoding="utf-8") as fh:
            _bad_verbs = {line.strip().lower() for line in fh if line.strip()}
        # pipeline override: allow HIPAA remediation lemmas blocked in the
        # shared Ember list so notify/provide ∩ sentences still extract.
        for lemma in getattr(config, "BAD_VERBS_ALLOWLIST", []) or []:
            _bad_verbs.discard(lemma.lower())
    return _bad_verbs


def _render_entity_aware(tokens):
    words = []
    prev = ""
    for t in tokens:
        if t.ent_type_ in _ENTITY_TAGS and t.text not in _ENTITY_WHITELIST:
            if t.ent_type_ != prev:
                words.append(f"[{t.ent_type_}]")
        else:
            words.append(t.text)
        prev = t.ent_type_
    return " ".join(words)


def _is_candidate_verb(token):
    """True if token is a verb we should try to extract a VerbPhrase from."""
    if not token.tag_.startswith("V"):
        return False
    if token.dep_ in config.KEPT_VERB_DEPS:
        return True
    # Mistagged remediation verbs (often VBN marked amod instead of ROOT/conj).
    amod_tags = getattr(config, "ALLOW_AMOD_VERB_TAGS", ()) or ()
    return token.dep_ == "amod" and token.tag_ in amod_tags


def extract_verb_phrases(sentence_text, nlp=None, bad_verbs=None):
    """Return a list of dicts: verb, verb_phrase, subject, subject_type."""
    nlp = nlp or _load_nlp()
    bad_verbs = bad_verbs if bad_verbs is not None else _load_bad_verbs()

    doc = nlp(sentence_text.encode().decode("utf-8", "ignore"))
    tokens = list(doc)

    children = {}
    verbs = []
    bridge = {}
    for token in doc:
        if not _is_candidate_verb(token):
            continue
        if token.lemma_.lower() in bad_verbs:
            continue

        chl = list(token.subtree)
        # Mistagged amod verbs have unreliable/tiny subtrees (often just the
        # verb). Expand from the verb to sentence end so objects get included.
        if token.dep_ == "amod":
            chl = [t for t in tokens if t.i >= token.i]

        # A bare VBN needs an aux. Skip that gate for amod
        # mistags (they almost never have aux and would otherwise be dropped).
        if token.tag_ == "VBN" and token.dep_ != "amod":
            good = any(
                t.tag_.startswith("V") and t.dep_.startswith("aux") and t.head == token
                for t in chl
            )
            if not good:
                continue

        children[token.i] = chl
        verbs.append(token)

        bridge[token.i] = []
        anchor = chl[0].i - 1
        while anchor >= 0:
            t = tokens[anchor]
            if t.dep_ in ("aux", "prep", "mark", "cc", "punct") or t.tag_ == "WDT":
                bridge[token.i].append(t)
            else:
                break
            anchor -= 1
        bridge[token.i].reverse()

    actions = {}
    subject = {}
    subject_type = {}
    flags = [True] * len(tokens)

    order = sorted(children.keys(), key=lambda idx: len(children[idx]))

    for idx in order:
        verb = tokens[idx]
        chl = children[idx]

        for t in chl:
            if t.dep_ in ("aux", "neg"):
                bridge[idx].append(t)
            else:
                break

        sub_tokens = []
        sub_text = ""
        for t in chl:
            if t.dep_.startswith("nsubj") and t.head == verb and t.tag_ != "WDT":
                sub_tokens = list(t.subtree)
                sub_text = _render_entity_aware(sub_tokens)

        subtype = []
        low = sub_text.lower()
        if "BA" in sub_text or "business associate" in low:
            subtype.append("BA")
        if "CE" in sub_text or "covered entity" in low:
            subtype.append("CE")
        else:
            for t in sub_tokens:
                if t.ent_type_ in ("GPE", "ORG", "PERSON"):
                    subtype.append("CE")
                    break
        if "employee" in low or "member" in low:
            subtype.append("employee")
        if "OCR" in sub_text:
            subtype = []

        subject[idx] = sub_text
        subject_type[idx] = ",".join(subtype)

        first = verb
        for i, t1 in enumerate(chl):
            if t1 == verb:
                j = i
                while j > 0:
                    j -= 1
                    t = chl[j]
                    if (
                        verb.tag_ == "VBN"
                        and t.lemma_.lower() == "be"
                        and t.dep_.startswith("aux")
                        and t.head == verb
                    ):
                        first = t
                    elif t.tag_ == "RB" and t.dep_ != "neg":
                        first = t
                    else:
                        break
                break

        started = False
        action = []
        for t in chl:
            if not flags[t.i]:
                if not started:
                    continue
                else:
                    break
            if started:
                action.append(t)
                continue
            if t == first:
                started = True
                action.append(t)

        words = []
        first_word = True
        prev = ""
        for t in action:
            if t.tag_.startswith("V") and first_word:
                words.append(t.lemma_.lower())
                first_word = False
            elif t.ent_type_ in _ENTITY_TAGS and t.text not in _ENTITY_WHITELIST:
                if t.ent_type_ != prev:
                    words.append(f"[{t.ent_type_}]")
            else:
                words.append(t.text)
            prev = t.ent_type_

        while words and words[-1].lower() in _TRAILING_JUNK:
            words = words[:-1]

        if len(words) < config.MIN_VP_WORDS:
            continue

        if verb.dep_ not in ("ccomp", "xcomp", "acl"):
            for t in action:
                flags[t.i] = False
            for t in bridge[idx]:
                flags[t.i] = False

        actions[idx] = " ".join(" ".join(words).split())

    results = []
    for verb in verbs:
        if verb.i not in actions:
            continue
        results.append(
            {
                "verb": verb.lemma_.lower(),
                "verb_phrase": actions[verb.i],
                "subject": subject[verb.i],
                "subject_type": subject_type[verb.i],
            }
        )
    return results


def run():
    nlp = _load_nlp()
    bad_verbs = _load_bad_verbs()

    with open(config.AGREED_ACTIONS_JSON, encoding="utf-8") as fh:
        agreed = json.load(fh)
    report_sentences = common.load_report_sentences()

    report_vps = {}
    total_vps = 0
    reports_with_vps = 0
    for bid, sentence_ids in agreed.items():
        if not sentence_ids:
            continue
        sentences = report_sentences[bid]["sentences"]
        vps = []
        for sid in sentence_ids:
            text = sentences[sid - 1]
            for r in extract_verb_phrases(text, nlp=nlp, bad_verbs=bad_verbs):
                vps.append(r["verb_phrase"])
        if vps:
            report_vps[bid] = vps
            total_vps += len(vps)
            reports_with_vps += 1

    with open(config.REPORT_VPS_JSON, "w", encoding="utf-8") as fh:
        json.dump(report_vps, fh, indent=2)

    print(f"Reports with agreed action sentences : {sum(1 for v in agreed.values() if v)}")
    print(f"Reports with >=1 extracted VerbPhrase : {reports_with_vps}")
    print(f"Total VerbPhrases (G_r, all reports)  : {total_vps}")
    print(f"Saved -> {config.REPORT_VPS_JSON}")


if __name__ == "__main__":
    run()
