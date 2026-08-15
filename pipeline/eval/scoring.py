"""BERTScore and MoverScore between short phrases.

RoBERTa-large (layer 17, as recommended by the BERTScore paper) gives
contextual, L2-normalized per-token embeddings. BERTScore is the greedy
token-matching P/R/F1 of Zhang et al. (2020); MoverScore is 1 - Earth Mover's
Distance over the same token embeddings (Zhao et al., 2019), solved with
`pyemd`.
"""

import os
from functools import lru_cache

import numpy as np
import pyemd
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BERT_MODEL_NAME = "roberta-large"
# Layer index recommended by the BERTScore paper for roberta-large.
BERT_LAYER = 17

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_EPS = 1e-8


@lru_cache(maxsize=1)
def get_bert():
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModel.from_pretrained(BERT_MODEL_NAME)
    model.eval()
    model.to(DEVICE)
    return tokenizer, model


@torch.no_grad()
def encode_tokens(texts, batch_size=32, layer=BERT_LAYER, max_length=80):
    """Contextual, L2-normalized, per-token embeddings for each string.

    Returns a list (len == len(texts)) of float32 (L_i, D) numpy arrays, one
    row per real subword token (special tokens / padding stripped).
    """
    tokenizer, model = get_bert()
    out = [None] * len(texts)

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        outputs = model(**enc, output_hidden_states=True)
        hidden = outputs.hidden_states[layer]  # (B, L, D)

        special_mask = torch.tensor(
            [
                tokenizer.get_special_tokens_mask(
                    ids.tolist(), already_has_special_tokens=True
                )
                for ids in enc["input_ids"].cpu()
            ],
            dtype=torch.bool,
        ).to(DEVICE)
        keep_mask = enc["attention_mask"].bool() & (~special_mask)

        hidden = torch.nn.functional.normalize(hidden, dim=-1)

        for i in range(len(batch)):
            mask_i = keep_mask[i]
            toks = hidden[i][mask_i]  # (L_i, D)
            if toks.shape[0] == 0:
                # Degenerate (e.g. empty string) -> keep at least one vector
                # so downstream max/mean ops don't divide by zero.
                toks = hidden[i][enc["attention_mask"][i].bool()][:1]
            out[start + i] = toks.cpu().numpy().astype(np.float32)

    return out


def bertscore_prf(tokens_a, tokens_b):
    """BERTScore (precision, recall, F1) between two (L, D) L2-normalized
    token-embedding arrays. `a` is the candidate, `b` is the reference."""
    sims = tokens_a.astype(np.float32) @ tokens_b.astype(np.float32).T
    p = float(sims.max(axis=1).mean())
    r = float(sims.max(axis=0).mean())
    f1 = 0.0 if (p + r) < _EPS else 2 * p * r / (p + r)
    return p, r, f1


def best_bertscore_prf(tokens_a, ref_tokens_list):
    """Score `a` against every reference in ref_tokens_list, keep the one
    with the best F1, and return its (precision, recall, f1)."""
    best = (0.0, 0.0, -1.0)
    for ref_toks in ref_tokens_list:
        p, r, f1 = bertscore_prf(tokens_a, ref_toks)
        if f1 > best[2]:
            best = (p, r, f1)
    return best


def moverscore(tokens_a, tokens_b):
    """MoverScore (1 - Earth Mover's Distance) between two (L, D) L2-normalized
    token-embedding arrays: each phrase's tokens are a uniform-mass
    distribution and the cost is 1 - cosine similarity."""
    la, lb = tokens_a.shape[0], tokens_b.shape[0]
    sims = tokens_a.astype(np.float64) @ tokens_b.astype(np.float64).T
    dist = 1.0 - sims

    n = la + lb
    joint = np.zeros((n, n), dtype=np.float64)
    joint[:la, la:] = dist
    joint[la:, :la] = dist.T
    hist_a = np.concatenate([np.full(la, 1.0 / la), np.zeros(lb)])
    hist_b = np.concatenate([np.zeros(la), np.full(lb, 1.0 / lb)])
    wmd = pyemd.emd(hist_a, hist_b, joint)
    return 1.0 - wmd


def best_moverscore(tokens_a, ref_tokens_list):
    """Score `a` against every reference in ref_tokens_list and return the
    highest MoverScore (it has no precision/recall split)."""
    best = -1.0
    for ref_toks in ref_tokens_list:
        m = moverscore(tokens_a, ref_toks)
        if m > best:
            best = m
    return best
