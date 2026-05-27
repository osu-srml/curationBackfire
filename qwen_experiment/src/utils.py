\
from __future__ import annotations
import torch
from transformers import GenerationConfig
import json
import os
from typing import Any, Dict, List
import math
import random
from src.reward import (
    _tok,
    rouge_l_f1,
    summarize_eval_reward,
    summarize_proxy_reward,
    paraphrase_eval_reward, 
    paraphrase_proxy_reward,
    reward_p_summary, 
    reward_q_formal_paraphrase,
    pick_best,
)

SUM_SYSTEM = ("You are a helpful English summarization assistant. "
              "Write a single-sentence, information-dense summary that is as short as possible. "
              "Do not add new facts or embellishments.")
PARA_SYSTEM = ("You are an English paraphrasing assistant. "
                 "Rewrite the text with different words while maintaining the core meaning."
                 "Do not add new facts.")

def mean(xs):
    return float(sum(xs) / max(1, len(xs)))

def std(xs):
    if len(xs) <= 1:
        return 0.0
    mu = mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return float(math.sqrt(var))

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def dump_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _last_line(text: str) -> str:
    parts = [p.strip() for p in (text or "").split("\n") if p.strip()]
    return parts[-1] if parts else (text or "").strip()

def truncate_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    """Truncate text to fit in max_tokens when tokenized."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    else:
        truncated = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
        return truncated

def append_jsonl(path: str, rows: List[Dict]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

@torch.inference_mode()
def batch_generate(
    tokenizer, model, system: str, users: List[str],
    max_new_tokens: int, temperature: float, top_p: float, do_sample: bool,
    gen_batch_size: int = 16,
) -> List[str]:
    model.eval()
    # decoder-only: left padding is safer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    outs: List[str] = []
    for s in range(0, len(users), gen_batch_size):
        chunk = users[s:s+gen_batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": u}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for u in chunk
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

        # For device_map="auto", model.device usually points to the embedding device; keep as-is
        enc = {k: v.to(model.device) for k, v in enc.items()}

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id, 
        )
        if do_sample:  # only set sampling params if sampling
            gen_config.temperature = temperature
            gen_config.top_p = top_p
        out = model.generate(**enc, generation_config=gen_config, use_cache=False)
        texts = tokenizer.batch_decode(out, skip_special_tokens=True)
        outs.extend([_last_line(t) for t in texts])

        # reduce fragmentation between chunks
        del enc, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return outs

def build_sum_user(article: str) -> str:
    return f"Please summarize the following text in ONE sentence:\n{article}"

def build_para_user(s: str) -> str:
    return f"Please paraphrase the following text while maintaining the core meaning:\n{s}"

def sample_rows(rows: List[Dict], n: int, rng: random.Random) -> List[Dict]:
    if n >= len(rows):
        return rows[:]
    idx = rng.sample(range(len(rows)), n)
    return [rows[i] for i in idx]

def evaluate_p(tokenizer, p_model, val_rows: List[Dict], max_new_tokens: int = 64, target_len: int=12, batch_size: int = 16, reward_tra_eval_same: bool = True, reward_type: int = 0) -> Dict[str, float]:
    len_scores = []
    covs = []
    copies = []
    raw_rewards = []
    clipped_rewards = []
    len_toks = []
    p_model.eval()
    rewards = []
    rouge = []
    for i in range(0, len(val_rows), batch_size):
        batch = val_rows[i:i+batch_size]
        users = [build_sum_user(r["article"]) for r in batch]
        preds = batch_generate(tokenizer, p_model, SUM_SYSTEM, users, max_new_tokens=max_new_tokens,
                               temperature=0.0, top_p=1.0, do_sample=False)
        for pred, r in zip(preds, batch):
            ref = r["summary"]
            rouge.append(rouge_l_f1(pred, ref))
            if reward_type == 0:
                if reward_tra_eval_same:
                    rewards.append(summarize_eval_reward(pred, ref))
                else:
                    rewards.append(summarize_proxy_reward(pred, r["article"], target_len=target_len))
            elif reward_type == 1:
                parts = reward_p_summary(r["article"], pred, target_len=target_len)
                raw_rewards.append(parts["raw_reward"])
                len_scores.append(parts["len_score"])
                covs.append(parts["cov"])
                copies.append(parts["copy"])
                clipped_rewards.append(parts["clipped_reward"])
                len_toks.append(parts["len_tok"])

        if reward_type == 0:
            return {
                "p_rougeL": float(sum(rouge) / max(1, len(rouge))),
                "p_eval_reward": float(sum(rewards) / max(1, len(rewards))),
            }
        elif reward_type == 1:
            return {
                "p_rougeL": float(sum(rouge) / max(1, len(rouge))),
                "p_eval_raw_reward": float(sum(raw_rewards) / max(1, len(raw_rewards))),
                "p_eval_len_score": float(sum(len_scores) / max(1, len(len_scores))),
                "p_eval_cov": float(sum(covs) / max(1, len(covs))),
                "p_eval_copy": float(sum(copies) / max(1, len(copies))),
                "p_eval_clipped_reward": float(sum(clipped_rewards) / max(1, len(clipped_rewards))),
                "p_eval_len_tok": float(sum(len_toks) / max(1, len(len_toks))),
            }

def evaluate_p_on_q(tokenizer, p_model, para_val_rows: List[Dict], target_len: int=12, max_new_tokens: int=80, outer_batch: int=8, gen_batch_size: int=4, reward_type: int = 1) -> Dict[str, float]:
    p_model.eval()
    len_scores = []
    covs = []
    copies = []
    raw_rewards = []
    clipped_rewards = []
    len_toks = []
    rewards = []
    lens = []
    for i in range(0, len(para_val_rows), outer_batch):
        batch = para_val_rows[i:i+outer_batch]
        articles = [r["src"] for r in batch]
        users = [build_sum_user(a) for a in articles]
        preds = batch_generate(tokenizer, p_model, SUM_SYSTEM, users,
                               max_new_tokens=max_new_tokens,
                               temperature=0.0, top_p=1.0, do_sample=False,
                               gen_batch_size=gen_batch_size)
        for a, pred in zip(articles, preds):
            if reward_type == 0:
                rewards.append(summarize_proxy_reward(pred, a, target_len=target_len))
                lens.append(len(_tok(pred)))
            elif reward_type == 1:
                parts = reward_p_summary(a, pred, target_len=target_len)
                raw_rewards.append(parts["raw_reward"])
                len_scores.append(parts["len_score"])
                covs.append(parts["cov"])
                copies.append(parts["copy"])
                clipped_rewards.append(parts["clipped_reward"])
                len_toks.append(parts["len_tok"])
        if reward_type == 0:
            return {
                'p_reward_on_qsrc': mean(rewards),
                'p_len_on_qsrc': mean(lens),
            }
        elif reward_type == 1:
            return {
                "p_eval_on_qsrc_raw_reward": float(sum(raw_rewards) / max(1, len(raw_rewards))),
                "p_eval_on_qsrc_len_score": float(sum(len_scores) / max(1, len(len_scores))),
                "p_eval_on_qsrc_cov": float(sum(covs) / max(1, len(covs))),
                "p_eval_on_qsrc_copy": float(sum(copies) / max(1, len(copies))),
                "p_eval_on_qsrc_clipped_reward": float(sum(clipped_rewards) / max(1, len(clipped_rewards))),
                "p_eval_on_qsrc_len_tok": float(sum(len_toks) / max(1, len(len_toks))),
            }

def evaluate_p_sampling(tokenizer, p_model, sum_val_rows: List[Dict], num_samples: int = 3, temperature: float=0.7, top_p: float=0.9, target_len: int=12, 
                        max_new_tokens: int = 80, outer_batch: int=8, gen_batch_size: int=4, seed: int=42, reward_type: int = 1) -> Dict[str, float]:
    p_model.eval()
    all_rewards = []
    all_raw_rewards = []
    for s in range(num_samples):
        random.seed(seed + s *1000)
        try:
            import torch
            torch.manual_seed(seed + s *1000)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + s *1000) 
        except Exception:
            pass

        rewards = []
        raw_rewards = []
        for i in range(0, len(sum_val_rows), outer_batch):
            batch = sum_val_rows[i:i+outer_batch]
            articles = [r["article"] for r in batch]
            users = [build_sum_user(a) for a in articles]
            preds = batch_generate(tokenizer, p_model, SUM_SYSTEM, users,
                                   max_new_tokens=max_new_tokens,
                                   temperature=temperature, top_p=top_p, do_sample=True,
                                   gen_batch_size=gen_batch_size)
            for a, pred in zip(articles, preds):
                if reward_type == 0:
                    rewards.append(summarize_proxy_reward(pred, a, target_len=target_len))
                elif reward_type == 1:
                    rewards.append(reward_p_summary(a, pred, target_len=target_len)["clipped_reward"])
                    raw_rewards.append(reward_p_summary(a, pred, target_len=target_len)["raw_reward"])
        all_rewards.append(mean(rewards))
        if reward_type == 1:
            all_raw_rewards.append(mean(raw_rewards))

    if reward_type == 0:
        return {
            "p_eval_reward_sample_mean": mean(all_rewards),
            "p_eval_reward_sample_std": std(all_rewards),
        }
    elif reward_type == 1:
        return {
            "p_eval_reward_sample_mean": mean(all_rewards),
            "p_eval_reward_sample_std": std(all_rewards),
            "p_eval_raw_reward_sample_mean": mean(all_raw_rewards),
            "p_eval_raw_reward_sample_std": std(all_raw_rewards),
        }

def evaluate_q_sampling(tokenizer, q_model, para_val_rows: List[Dict], num_samples: int = 3, temperature: float=0.7, top_p: float=0.9,
                        max_new_tokens: int = 80, outer_batch: int=8, gen_batch_size: int=4, seed: int=42, reward_type: int = 0) -> Dict[str, float]:
    q_model.eval()
    all_rewards = []
    all_raw_rewards = []
    for s in range(num_samples):
        random.seed(seed + s *2000)
        try:
            import torch
            torch.manual_seed(seed + s *2000)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + s *2000) 
        except Exception:
            pass

        rewards = []
        raw_rewards = []
        for i in range(0, len(para_val_rows), outer_batch):
            batch = para_val_rows[i:i+outer_batch]
            srcs = [r["src"] for r in batch]
            users = [build_para_user(s) for s in srcs]
            preds = batch_generate(tokenizer, q_model, PARA_SYSTEM, users,
                                   max_new_tokens=max_new_tokens,
                                   temperature=temperature, top_p=top_p, do_sample=True,
                                   gen_batch_size=gen_batch_size)
            for s, pred in zip(srcs, preds):
                if reward_type == 0:
                    rewards.append(paraphrase_proxy_reward(pred, s))
                elif reward_type == 1:
                    rewards.append(reward_q_formal_paraphrase(s, pred)["clipped_reward"])
                    raw_rewards.append(reward_q_formal_paraphrase(s, pred)["raw_reward"])

        all_rewards.append(mean(rewards))
        if reward_type == 1:
            all_raw_rewards.append(mean(raw_rewards))
    
    if reward_type == 0:
        return {
            "q_eval_reward_sample_mean": mean(all_rewards),
            "q_eval_reward_sample_std": std(all_rewards),
        }
    elif reward_type == 1:
        return {
            "q_eval_reward_sample_mean": mean(all_rewards),
            "q_eval_reward_sample_std": std(all_rewards),
            "q_eval_raw_reward_sample_mean": mean(all_raw_rewards),
            "q_eval_raw_reward_sample_std": std(all_raw_rewards),
        }

def evaluate_q_on_p(tokenizer, q_model, sum_val_rows: List[Dict], max_src_tokens: int=512, max_new_tokens: int=80, outer_batch: int=8, gen_batch_size: int=4, reward_type: int = 1) -> Dict[str, float]:
    q_model.eval()
    rewards = []
    ratios = []
    len_scores = []
    forms = []
    copy_mids = []
    raw_rewards = []
    len_toks_src = []
    len_toks_pred = []
    copies = []
    for i in range(0, len(sum_val_rows), outer_batch):
        batch = sum_val_rows[i:i+outer_batch]
        srcs = [truncate_to_tokens(tokenizer, r["article"], max_src_tokens) for r in batch]
        users = [build_para_user(s) for s in srcs]
        preds = batch_generate(tokenizer, q_model, PARA_SYSTEM, users,
                               max_new_tokens=max_new_tokens,
                               temperature=0.0, top_p=1.0, do_sample=False,
                               gen_batch_size=gen_batch_size)
        for s, pred in zip(srcs, preds):
            if reward_type == 0:
                rewards.append(paraphrase_proxy_reward(pred, s))
                ls, lp = len(_tok(s)), len(_tok(pred))
                ratios.append(lp / max(1, ls))
            elif reward_type == 1:
                parts = reward_q_formal_paraphrase(s, pred)
                raw_rewards.append(parts["raw_reward"])
                len_scores.append(parts["len_score"])
                forms.append(parts["form"])
                copy_mids.append(parts["copy_mid"])
                ratios.append(parts["ratio"])
                len_toks_src.append(parts["len_tok_src"])
                len_toks_pred.append(parts["len_tok_pred"])
                copies.append(parts["copy"])
                rewards.append(parts["clipped_reward"])
    if reward_type == 0:
        return {
            'q_reward_on_xsum': mean(rewards),
            'q_len_ratio_on_xsum': mean(ratios),
        }
    elif reward_type == 1:
        return {
            'q_reward_on_xsum': mean(rewards),
            'q_len_ratio_on_xsum': mean(ratios),
            "q_eval_on_xsum_raw_reward": float(sum(raw_rewards) / max(1, len(raw_rewards))),
            "q_eval_on_xsum_len_score": float(sum(len_scores) / max(1, len(len_scores))),
            "q_eval_on_xsum_form": float(sum(forms) / max(1, len(forms))),
            "q_eval_on_xsum_copy_mid": float(sum(copy_mids) / max(1, len(copy_mids))),
            "q_eval_on_xsum_len_tok_src": float(sum(len_toks_src) / max(1, len(len_toks_src))),
            "q_eval_on_xsum_len_tok_pred": float(sum(len_toks_pred) / max(1, len(len_toks_pred))),
            "q_eval_on_xsum_copy": float(sum(copies) / max(1, len(copies))),
        }

def evaluate_q(tokenizer, q_model, para_val_rows: List[Dict], max_new_tokens: int = 80, batch_size: int = 16, reward_tra_eval_same: bool = True, reward_type: int = 0) -> Dict[str, float]:
    q_model.eval()
    rewards = []
    rouge = []
    len_scores = []
    forms = []
    copy_mids = []
    ratios = []
    raw_rewards = []
    len_toks_src = []
    len_toks_pred = []
    copies = []
    for i in range(0, len(para_val_rows), batch_size):
        batch = para_val_rows[i:i+batch_size]
        users = [build_para_user(r["src"]) for r in batch]
        preds = batch_generate(tokenizer, q_model, PARA_SYSTEM, users, max_new_tokens=max_new_tokens,
                               temperature=0.0, top_p=1.0, do_sample=False)
        for pred, r in zip(preds, batch):
            ref = r["tgt"]
            rouge.append(rouge_l_f1(pred, ref))
            if reward_type == 0:
                if reward_tra_eval_same:
                    rewards.append(paraphrase_eval_reward(pred, ref))
                else:
                    rewards.append(paraphrase_proxy_reward(pred, r["src"]))
            elif reward_type == 1:
                parts = reward_q_formal_paraphrase(r["src"], pred)
                rewards.append(parts["clipped_reward"])
                raw_rewards.append(parts["raw_reward"])
                len_scores.append(parts["len_score"])
                forms.append(parts["form"])
                copy_mids.append(parts["copy_mid"])
                ratios.append(parts["ratio"])
                len_toks_src.append(parts["len_tok_src"])
                len_toks_pred.append(parts["len_tok_pred"])
                copies.append(parts["copy"])
    if reward_type == 0:
        return {
            "q_rougeL_to_para_ref": float(sum(rouge) / max(1, len(rouge))),
            "q_eval_reward": float(sum(rewards) / max(1, len(rewards))),
        }
    elif reward_type == 1:
        return {
            "q_rougeL_to_para_ref": float(sum(rouge) / max(1, len(rouge))),
            "q_eval_reward": float(sum(rewards) / max(1, len(rewards))),
            "q_eval_raw_reward": float(sum(raw_rewards) / max(1, len(raw_rewards))),
            "q_eval_len_score": float(sum(len_scores) / max(1, len(len_scores))),
            "q_eval_form": float(sum(forms) / max(1, len(forms))),
            "q_eval_copy_mid": float(sum(copy_mids) / max(1, len(copy_mids))),
            "q_eval_ratio": float(sum(ratios) / max(1, len(ratios))),
            "q_eval_len_tok_src": float(sum(len_toks_src) / max(1, len(len_toks_src))),
            "q_eval_len_tok_pred": float(sum(len_toks_pred) / max(1, len(len_toks_pred))),
            "q_eval_copy": float(sum(copies) / max(1, len(copies))),
        }

def mix_rows(real_rows: List[Dict], self_rows: List[Dict], cross_rows: List[Dict],
             lam_real: float, lam_self: float, lam_cross: float, total: int, rng: random.Random) -> List[Dict]:
    assert abs((lam_real + lam_self + lam_cross) - 1.0) < 1e-6, "lambdas must sum to 1"
    n_real = int(total * lam_real)
    n_self = int(total * lam_self)
    n_cross = total - n_real - n_self

    def pick(src, n):
        if n <= 0:
            return []
        if len(src) == 0:
            return []
        if n >= len(src):
            return rng.choices(src, k=n)
        return rng.sample(src, k=n)

    out = pick(real_rows, n_real) + pick(self_rows, n_self) + pick(cross_rows, n_cross)
    rng.shuffle(out)
    return out

def build_p_epoch_synthetics(
    tokenizer, p_model, q_model,
    p_prev_pool,  # previous round's distribution（article/summary）
    q_prev_pool,  # src/tgt
    num_self_inputs: int,
    num_cross_inputs: int,
    k: int,
    rng,
    max_new_sum: int,
    max_new_para: int,
    reward_type: int = 0,
):
    base = sample_rows_safe(p_prev_pool, num_self_inputs, rng)
    articles = [r["article"] for r in base]

    # p self raw: 1 candidate
    users = [build_sum_user(a) for a in articles]
    self_raw_summ = batch_generate(
        tokenizer, p_model, SUM_SYSTEM, users,
        max_new_tokens=max_new_sum,
        temperature=0.9, top_p=0.95, do_sample=True
    )
    p_self_raw = [{"article": a, "summary": s} for a, s in zip(articles, self_raw_summ)]

    # p self curated: K candidates + pick by summarize_proxy_reward
    usersK, owner = [], []
    for i, a in enumerate(articles):
        for _ in range(k):
            usersK.append(build_sum_user(a))
            owner.append(i)
    cands = batch_generate(
        tokenizer, p_model, SUM_SYSTEM, usersK,
        max_new_tokens=max_new_sum,
        temperature=0.9, top_p=0.95, do_sample=True
    )
    p_self_cur = []
    for i, a in enumerate(articles):
        cs = [cands[j] for j in range(len(owner)) if owner[j] == i]
        if reward_type == 0:
            scores = [summarize_proxy_reward(c, a, target_len=12) for c in cs]
        elif reward_type == 1:
            scores = [reward_p_summary(a, c, target_len=12)["clipped_reward"] for c in cs]
        chosen, _, _ = pick_best(cs, scores)
        p_self_cur.append({"article": a, "summary": chosen})

    # cross: from q_prev_pool 
    base_q = sample_rows_safe(q_prev_pool, num_cross_inputs, rng)
    xs = [r["src"] for r in base_q]          # x comes from q's input field

    # cross raw: one q generation per x
    users = [build_para_user(x) for x in xs]
    cross_raw = batch_generate(tokenizer, q_model, PARA_SYSTEM, users,
                               max_new_tokens=max_new_para,
                               temperature=0.9, top_p=0.95, do_sample=True)
    p_cross_raw = [{"article": x, "summary": y} for x, y in zip(xs, cross_raw)]

    # cross curated: K q generations per x, pick by q-side proxy reward w.r.t x
    usersK, owner = [], []
    for i, x in enumerate(xs):
        for _ in range(k):
            usersK.append(build_para_user(x))
            owner.append(i)
    cands = batch_generate(tokenizer, q_model, PARA_SYSTEM, usersK,
                           max_new_tokens=max_new_para,
                           temperature=0.9, top_p=0.95, do_sample=True)
    p_cross_cur = []
    for i, x in enumerate(xs):
        cs = [cands[j] for j in range(len(owner)) if owner[j] == i]
        if reward_type == 0:
            scores = [paraphrase_proxy_reward(c, x) for c in cs]
        elif reward_type == 1:
            scores = [reward_q_formal_paraphrase(x, c)["clipped_reward"] for c in cs]
        chosen, _, _ = pick_best(cs, scores)
        p_cross_cur.append({"article": x, "summary": chosen})

    return p_self_raw, p_self_cur, p_cross_raw, p_cross_cur

def build_q_epoch_synthetics(
    tokenizer, p_model, q_model,
    q_prev_pool,     # last round's distribution（src/tgt）
    p_prev_pool,     # article/summary
    num_self_inputs: int, 
    num_cross_inputs: int,
    k: int,
    rng,
    max_new_para: int,
    max_new_sum: int,
    reward_type: int = 0,
):
    base = sample_rows_safe(q_prev_pool, num_self_inputs, rng)
    srcs = [r["src"] for r in base]

    # q self raw: 1 paraphrase 
    users = [build_para_user(s) for s in srcs]
    self_raw = batch_generate(
        tokenizer, q_model, PARA_SYSTEM, users,
        max_new_tokens=max_new_para,
        temperature=0.9, top_p=0.95, do_sample=True
    )
    q_self_raw = [{"src": s, "tgt": y} for s, y in zip(srcs, self_raw)]

    # q self curated: K paraphrases + pick by paraphrase_proxy_reward(pred, src)
    usersK, owner = [], []
    for i, s in enumerate(srcs):
        for _ in range(k):
            usersK.append(build_para_user(s))
            owner.append(i)
    cands = batch_generate(
        tokenizer, q_model, PARA_SYSTEM, usersK,
        max_new_tokens=max_new_para,
        temperature=0.9, top_p=0.95, do_sample=True
    )
    q_self_cur = []
    for i, s in enumerate(srcs):
        cs = [cands[j] for j in range(len(owner)) if owner[j] == i]
        if reward_type == 0:
            scores = [paraphrase_proxy_reward(c, s) for c in cs]
        elif reward_type == 1:
            scores = [reward_q_formal_paraphrase(s, c)["clipped_reward"] for c in cs]
        chosen, _, _ = pick_best(cs, scores)
        q_self_cur.append({"src": s, "tgt": chosen})

    # cross: from p_prev_pool
    base_p = sample_rows_safe(p_prev_pool, num_cross_inputs, rng)
    xs = [r["article"] for r in base_p]      # x comes from p's input field

    # cross raw: p generates y for each x
    users = [build_sum_user(x) for x in xs]
    cross_raw = batch_generate(tokenizer, p_model, SUM_SYSTEM, users,
                               max_new_tokens=max_new_sum,
                               temperature=0.9, top_p=0.95, do_sample=True)
    q_cross_raw = [{"src": x, "tgt": y} for x, y in zip(xs, cross_raw)]

    # cross curated: K p generations per x, pick by p-side proxy reward w.r.t x
    usersK, owner = [], []
    for i, x in enumerate(xs):
        for _ in range(k):
            usersK.append(build_sum_user(x))
            owner.append(i)
    cands = batch_generate(tokenizer, p_model, SUM_SYSTEM, usersK,
                           max_new_tokens=max_new_sum,
                           temperature=0.9, top_p=0.95, do_sample=True)
    q_cross_cur = []
    for i, x in enumerate(xs):
        cs = [cands[j] for j in range(len(owner)) if owner[j] == i]
        if reward_type == 0:
            scores = [summarize_proxy_reward(c, x, target_len=12) for c in cs]
        elif reward_type == 1:
            scores = [reward_p_summary(x, c, target_len=12)['clipped_reward'] for c in cs]
        chosen, _, _ = pick_best(cs, scores)
        q_cross_cur.append({"src": x, "tgt": chosen})

    return q_self_raw, q_self_cur, q_cross_raw, q_cross_cur

def split_counts(N, lam_real, lam_self, lam_cross, rho_self_cur, rho_cross_cur):
    n_real  = int(N * lam_real)
    n_self  = int(N * lam_self)
    n_cross = N - n_real - n_self

    n_self_cur  = int(n_self * rho_self_cur)
    n_self_raw  = n_self - n_self_cur
    n_cross_cur = int(n_cross * rho_cross_cur)
    n_cross_raw = n_cross - n_cross_cur

    return n_real, n_self_raw, n_self_cur, n_cross_raw, n_cross_cur

def sample_rows_safe(rows, n, rng):
    if n <= 0:
        return []
    if len(rows) <= n:
        out = rows[:]
        rng.shuffle(out)
        return out
    idx = rng.sample(range(len(rows)), n)
    return [rows[i] for i in idx]

def tag_rows(rows, tag: str):
    out = []
    for r in rows:
        rr = dict(r)
        rr["_src"] = tag
        out.append(rr)
    return out
