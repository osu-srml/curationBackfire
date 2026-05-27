\
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict
import math
import random
import re
from collections import Counter

def _lcs_len(a: str, b: str) -> int:
    """Longest Common Subsequence length (character-level). O(len(a)*len(b)) DP.
    For Chinese short texts this is fine; for long texts consider trimming.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    # Use rolling DP to save memory.
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = cur[j - 1] if cur[j - 1] >= prev[j] else prev[j]
        prev = cur
    return prev[m]

def rouge_l_f1(pred: str, ref: str) -> float:
    """ROUGE-L F1 at character-level."""
    pred = (pred or "").strip()
    ref = (ref or "").strip()
    if not pred or not ref:
        return 0.0
    lcs = _lcs_len(pred, ref)
    prec = lcs / max(1, len(pred))
    rec = lcs / max(1, len(ref))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def _word_bigrams(s: str) -> set:
    """Word-level bigrams for English proxy rewards."""
    _ws_re = re.compile(r"\s+")
    s = _ws_re.sub(" ", (s or "").strip().lower())
    if not s:
        return set()
    toks = s.split(" ")
    if len(toks) < 2:
        return {(toks[0], toks[0])} if toks else set()
    return set(zip(toks, toks[1:]))

def _jaccard(A: set, B: set) -> float:
    if not A and not B:
        return 1.0
    inter = len(A & B)
    union = max(len(A | B), 1)
    return inter / union

def length_ratio(pred: str, ref: str) -> float:
    pred = (pred or "").strip()
    ref = (ref or "").strip()
    return len(pred) / max(1, len(ref))

def summarize_eval_reward(pred: str, ref: str, alpha_len: float = 0.35) -> float:
    """Eval reward for summarization: ROUGE-L - alpha*|len_ratio-1|."""
    r = rouge_l_f1(pred, ref)
    lr = length_ratio(pred, ref)
    return r - alpha_len * abs(lr - 1.0)

def summarize_proxy_reward(pred: str, src: str, target_len: int = 25) -> float:
    """No-ref proxy reward used for *selection* (curation).
    Encourages:
      - short length near target_len
      - overlap with src (rough faithfulness proxy)
    """
    pred = (pred or "").strip()
    src = (src or "").strip()
    if not pred:
        return -1e9
    # Length shaping
    len_pen = -abs(len(pred) - target_len) / max(1, target_len)

    # overlap proxy: word-level bigram Jaccard (better for English)
    A = _word_bigrams(pred)
    B = _word_bigrams(src[:2000])  # cap src for speed
    jacc = _jaccard(A, B)
    return 0.6 * jacc + 0.4 * len_pen

def paraphrase_proxy_reward(pred: str, src: str) -> float:
    """No-ref proxy reward for paraphrase curation. 
    Goals:
      - preserve meaning (similarity proxy high)
      - change wording (ngram overlap not too high)
      - avoid degenerate length drift (len_ration near 1.0)"""
    pred = (pred or "").strip()
    src = (src or "").strip()
    if not pred:
        return -1e9
    sim = rouge_l_f1(pred, src)  # rough semantic proxy
    novelty = 1.0 - _jaccard(_word_bigrams(pred), _word_bigrams(src))
    lr = length_ratio(pred, src)
    len_pen = -abs(lr - 1.0)
    # prevent empty/garbage that trivially maximizes novelty
    if sim < 0.15:
        return -1e9 + sim
    return 0.5 * sim + 0.2 * novelty + 0.3 * len_pen


def paraphrase_eval_reward(pred: str, ref: str) -> float:
    """Eval reward for paraphrase: similarity to reference + novelty."""
    sim = rouge_l_f1(pred, ref)
    novelty = 1.0 - _jaccard(_word_bigrams(pred), _word_bigrams(ref))
    return 0.7 * sim + 0.3 * novelty


def pick_best(cands: List[str], scores: List[float]) -> Tuple[str, float, int]:
    assert len(cands) == len(scores) and len(cands) > 0
    best_i = max(range(len(cands)), key=lambda i: scores[i])
    return cands[best_i], float(scores[best_i]), int(best_i)

_CONTRACTIONS = {
    "can't","won't","n't","i'm","you're","we're","they're","it's","that's","there's",
    "isn't","aren't","wasn't","weren't","don't","doesn't","didn't","haven't","hasn't",
    "wouldn't","shouldn't","couldn't","let's"
}

def _tok(s: str):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+", (s or "").lower())

def _ngram_set(toks, n):
    if len(toks) < n: 
        return set()
    return set(tuple(toks[i:i+n]) for i in range(len(toks)-n+1))

def _copy_4gram_ratio(src: str, pred: str) -> float:
    a = _ngram_set(_tok(src), 4)
    b = _ngram_set(_tok(pred), 4)
    if not b:
        return 0.0
    return len(a & b) / max(1, len(b))  # fraction of pred 4-grams that appear in src

def _keyword_coverage(src: str, pred: str, topk: int = 20) -> float:
    st = _tok(src)
    pt = set(_tok(pred))
    if not st:
        return 0.0
    cnt = Counter(st)
    keys = [w for w, _ in cnt.most_common(topk)]
    hit = sum(1 for w in keys if w in pt)
    return hit / max(1, len(keys))

def _formality_score(pred: str) -> float:
    t = _tok(pred)
    if not t:
        return 0.0
    contr = 0
    for w in t:
        if w in _CONTRACTIONS:
            contr += 1
        if w.endswith("n't"):
            contr += 1
    # fewer contractions => more formal
    score = 1.0 - contr / max(1, len(t))
    return float(max(0.0, min(1.0, score)))

def reward_p_summary(article: str, pred: str, target_len: int = 25) -> float:
    # length preference: sharp around target_len
    lt = len(_tok(pred))
    len_score = math.exp(-abs(lt - target_len) / 2.0)  # narrower => stronger preference

    # coverage of salient words
    cov = _keyword_coverage(article, pred, topk=20)

    # penalize copying
    copy = _copy_4gram_ratio(article, pred)

    # weighted sum (clip to [0,1])
    r = 0.55 * len_score + 0.55 * cov - 0.60 * copy
    clipped = float(max(0.0, min(1.0, r)))
    # return float(max(0.0, min(1.0, r)))
    return {
        "len_score": float(len_score),
        "cov": float(cov),
        "copy": float(copy),
        "raw_reward": float(r),
        "clipped_reward": float(clipped),
        "len_tok": float(lt)
    }

def reward_q_formal_paraphrase(src: str, pred: str) -> float:
    # length ratio near 1 (strongly punish "summary-like" short outputs)
    ls = len(_tok(src))
    lp = len(_tok(pred))
    if ls <= 0 or lp <= 0:
        # return 0.0
        return {'len_score': 0.0, 'form': 0.0, 'copy_mid': 0.0, 'ratio': 0.0, 'raw_reward': 0.0, 'clipped_reward': 0.0, 'len_tok_pred': 0.0, 'len_tok_src': 0.0, 'copy': 0.0}
    ratio = lp / ls
    len_score = math.exp(-abs(ratio - 1.0) / 0.25)  # narrower => stronger preference

    # similarity to src (cheap proxy): 4-gram copy ratio too high is bad,
    # but too low also suggests semantic drift; we use a "middle" target.
    copy = _copy_4gram_ratio(src, pred)
    # prefer moderate copying around ~0.20
    copy_mid = math.exp(-abs(copy - 0.20) / 0.10)

    # formality
    form = _formality_score(pred)

    r = 0.45 * len_score + 0.35 * form + 0.30 * copy_mid
    # return float(max(0.0, min(1.0, r)))
    return {
        "len_score": float(len_score),
        "form": float(form),
        "copy_mid": float(copy_mid),
        "ratio": float(ratio),
        "raw_reward": float(r),
        "clipped_reward": float(max(0.0, min(1.0, r))),
        "len_tok_pred": float(lp),
        "len_tok_src": float(ls),
        "copy": float(copy),
    }


