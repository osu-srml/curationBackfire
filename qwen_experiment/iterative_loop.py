from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple
import warnings, traceback
from transformers import GenerationConfig
import torch
from tqdm import tqdm
import math
from typing import Dict, List

from src.data import safe_jsonl_read, safe_jsonl_write, load_idiom_list
from src.modeling import load_base_model, load_lora_into_base, attach_lora
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
from src.sft import train_lora_sft
from src.utils import sample_rows_safe, tag_rows, split_counts, evaluate_p, evaluate_q, evaluate_p_on_q, evaluate_q_on_p, evaluate_p_sampling, evaluate_q_sampling, build_p_epoch_synthetics, build_q_epoch_synthetics, build_sum_user, build_para_user

SUM_SYSTEM = ("You are a helpful English summarization assistant. "
              "Write a single-sentence, information-dense summary that is as short as possible. "
              "Do not add new facts or embellishments.")
PARA_SYSTEM = ("You are an English paraphrasing assistant. "
                 "Rewrite the text with different words while maintaining the core meaning."
                 "Do not add new facts.")


def main():
    start_time = time.time()
    loop_start_times = []
    loop_end_times = []
    loop_buildData_times = []
    loop_trian_times = []
    loop_eval_times = []
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--p_lora", type=str, required=True)
    ap.add_argument("--q_lora", type=str, required=True)

    ap.add_argument("--sum_train_jsonl", type=str, required=True)
    ap.add_argument("--sum_val_jsonl", type=str, required=True)
    ap.add_argument("--para_train_jsonl", type=str, required=True)
    ap.add_argument("--para_val_jsonl", type=str, required=True)

    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--samples_per_iter", type=int, default=1024)
    ap.add_argument("--k_candidates", type=int, default=4)

    ap.add_argument("--lambda_real_p", type=float, default=0.3)
    ap.add_argument("--lambda_self_p", type=float, default=0.2)
    ap.add_argument("--lambda_cross_p", type=float, default=0.5)

    ap.add_argument("--lambda_real_q", type=float, default=0.3)
    ap.add_argument("--lambda_self_q", type=float, default=0.2)
    ap.add_argument("--lambda_cross_q", type=float, default=0.5)

    # curation rho (for example, rho_p * lambda_self_p = final wight for curated self data in p-model)
    ap.add_argument("--rho_self_cur_p", type=float, default=0.5)
    ap.add_argument("--rho_cross_cur_p", type=float, default=0.5)
    ap.add_argument("--rho_self_cur_q", type=float, default=0.5)
    ap.add_argument("--rho_cross_cur_q", type=float, default=0.5)

    ap.add_argument("--train_steps_per_iter", type=int, default=120)
    ap.add_argument("--train_bs", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--dtype", type=str, choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--eval_num_samples", type=int, default=500)

    ap.add_argument("--max_seq_len_p", type=int, default=1024, help="SFT max_seq_len for p-model (summarization).")
    ap.add_argument("--max_seq_len_q", type=int, default=512, help="SFT max_seq_len for q-model (paraphrase).")

    # reward type: 0 for original rewards (rouge and proxy), 1 for the new reward_p_summary and reward_q_formal_paraphrase
    ap.add_argument("--reward_tra_eval_same", type=int, default=0, help="whether to use the same reward function for training and evaluation")
 
    ap.add_argument("--reward_type", type=int, default=0, 
                    help="type of reward to use, 0 for original rewards (rouge and proxy), 1 for the new reward_p_summary and reward_q_formal_paraphrase")
    args = ap.parse_args()

    reward_tra_eval_same = bool(args.reward_tra_eval_same)
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    # Load datasets
    sum_train = safe_jsonl_read(args.sum_train_jsonl)
    sum_val = safe_jsonl_read(args.sum_val_jsonl)

    para_train_path = os.path.join(os.path.dirname(args.para_train_jsonl), "para_train.jsonl")
    para_val_path = os.path.join(os.path.dirname(args.para_val_jsonl), "para_val.jsonl")
    para_val = safe_jsonl_read(para_val_path) if os.path.exists(para_val_path) else []
    para_train = safe_jsonl_read(para_train_path) if os.path.exists(para_train_path) else []

    eval_n = min(args.eval_num_samples, len(sum_val), len(para_val))
    sum_val_eval = sample_rows_safe(sum_val, eval_n, rng)
    para_val_eval = sample_rows_safe(para_val, eval_n, rng)

    diag_n = min(200, len(sum_val_eval), len(para_val_eval))
    sum_val_diag = sum_val_eval[:diag_n]
    para_val_diag = para_val_eval[:diag_n]

    # first round uses real data as the mixture data distribution
    p_prev_pool = sum_train[:]  
    q_prev_pool = para_train[:]

    # Load base once
    bundle_p = load_base_model(args.base_model, dtype=args.dtype, device_map="auto", load_in_4bit=args.load_in_4bit)
    bundle_q = load_base_model(args.base_model, dtype=args.dtype, device_map="auto", load_in_4bit=args.load_in_4bit)
    tok = bundle_p.tokenizer

    # Load LoRA adapters
    p_model = load_lora_into_base(bundle_p.model, args.p_lora)
    q_model = load_lora_into_base(bundle_q.model, args.q_lora)
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch device_count =", torch.cuda.device_count())

    print("p param device =", next(p_model.parameters()).device)
    print("q param device =", next(q_model.parameters()).device)

    print("p hf_device_map =", getattr(p_model, "hf_device_map", None))
    print("q hf_device_map =", getattr(q_model, "hf_device_map", None))

    # Evaluation before loop
    metrics = []
    # epoch 0: baseline eval
    m0 = {"epoch": 0}
    m0.update(evaluate_p(tok, p_model, sum_val_eval, reward_tra_eval_same=reward_tra_eval_same, reward_type=args.reward_type))
    m0.update(evaluate_q(tok, q_model, para_val_eval, reward_tra_eval_same=reward_tra_eval_same, reward_type=args.reward_type))
    m0.update(evaluate_p_on_q(tok, p_model, para_val_diag, reward_type=args.reward_type))
    m0.update(evaluate_q_on_p(tok, q_model, sum_val_diag, reward_type=args.reward_type))
    m0.update(evaluate_p_sampling(tok, p_model, sum_val_diag, reward_type=args.reward_type))
    m0.update(evaluate_q_sampling(tok, q_model, para_val_diag, reward_type=args.reward_type))
    metrics.append(m0)
    safe_jsonl_write(os.path.join(args.out_dir, "metrics.jsonl"), metrics)

    # Main loop
    for ep in range(1, args.iters + 1):
        ep_start_time = time.time()
        loop_start_times.append(ep_start_time)
        print(f"=== Iteration {ep} / {args.iters} ===")
        epoch_dir = os.path.join(args.out_dir, f"epoch_{ep:03d}")
        os.makedirs(epoch_dir, exist_ok=True)

        Np = args.samples_per_iter  # samples_per_iter = samples_per_epoch
        Nq = args.samples_per_iter

        npr, nps_raw, nps_cur, npc_raw, npc_cur = split_counts(
            Np, args.lambda_real_p, args.lambda_self_p, args.lambda_cross_p,
            args.rho_self_cur_p, args.rho_cross_cur_p
        )
        nqr, nqs_raw, nqs_cur, nqc_raw, nqc_cur = split_counts(
            Nq, args.lambda_real_q, args.lambda_self_q, args.lambda_cross_q,
            args.rho_self_cur_q, args.rho_cross_cur_q
        )

        # real samples
        p_real = sample_rows_safe(sum_train, npr, rng)
        q_real = sample_rows_safe(para_train, nqr, rng)

        # synthetics sampled from previous mixed pool
        p_self_inputs  = nps_raw + nps_cur
        p_cross_inputs = npc_raw + npc_cur

        p_self_raw, p_self_cur, p_cross_raw, p_cross_cur = build_p_epoch_synthetics(
            tok, p_model, q_model,
            p_prev_pool=p_prev_pool,
            q_prev_pool=q_prev_pool,
            num_self_inputs=p_self_inputs,
            num_cross_inputs=p_cross_inputs,
            k=args.k_candidates, rng=rng,
            max_new_sum=64, max_new_para=80, reward_type=args.reward_type
        )

        q_self_inputs  = nqs_raw + nqs_cur
        q_cross_inputs = nqc_raw + nqc_cur

        q_self_raw, q_self_cur, q_cross_raw, q_cross_cur = build_q_epoch_synthetics(
            tok, p_model, q_model,
            q_prev_pool=q_prev_pool,
            p_prev_pool=p_prev_pool,
            num_self_inputs=q_self_inputs,
            num_cross_inputs=q_cross_inputs,
            k=args.k_candidates, rng=rng,
            max_new_para=80, max_new_sum=64, reward_type=args.reward_type
        )

        # build final epoch train mixes
        p_train = []
        p_train += tag_rows(sample_rows_safe(p_real, npr, rng), "p_real")
        p_train += tag_rows(sample_rows_safe(p_self_raw, nps_raw, rng), "p_self_raw")
        p_train += tag_rows(sample_rows_safe(p_self_cur, nps_cur, rng), "p_self_cur")
        p_train += tag_rows(sample_rows_safe(p_cross_raw, npc_raw, rng), "p_cross_raw")
        p_train += tag_rows(sample_rows_safe(p_cross_cur, npc_cur, rng), "p_cross_cur")
        rng.shuffle(p_train)

        q_train = []
        q_train += tag_rows(sample_rows_safe(q_real, nqr, rng), "q_real")
        q_train += tag_rows(sample_rows_safe(q_self_raw, nqs_raw, rng), "q_self_raw")
        q_train += tag_rows(sample_rows_safe(q_self_cur, nqs_cur, rng), "q_self_cur")
        q_train += tag_rows(sample_rows_safe(q_cross_raw, nqc_raw, rng), "q_cross_raw")
        q_train += tag_rows(sample_rows_safe(q_cross_cur, nqc_cur, rng), "q_cross_cur")
        rng.shuffle(q_train)

        # save epoch train data for analysis
        safe_jsonl_write(os.path.join(epoch_dir, "p_train.jsonl"), p_train)
        safe_jsonl_write(os.path.join(epoch_dir, "q_train.jsonl"), q_train)

        t1 = time.time()
        loop_buildData_times.append(t1 - ep_start_time)
        print(f"Data for iteration {ep} built in {t1 - ep_start_time:.2f} seconds.")

        # ===== train one epoch worth of steps on top of current LoRA weights =====
        p_train_stats, train_loss_p = train_lora_sft(
            model=p_model,
            tokenizer=tok,
            train_rows=p_train,
            val_rows=None,
            system_prompt=SUM_SYSTEM,
            user_builder=lambda r: build_sum_user(r["article"]),
            target_key="summary",
            out_dir=os.path.join(epoch_dir, "p_lora_updated"),
            max_seq_len=args.max_seq_len_p,
            lr=args.lr,
            num_epochs=1.0,
            train_batch_size=args.train_bs,
            grad_accum=args.grad_accum,
            max_steps=args.train_steps_per_iter,
            bf16=(args.dtype == "bf16"),
            fp16=(args.dtype == "fp16"),
            gradient_checkpointing=True,
        )

        q_train_stats, train_loss_q = train_lora_sft(
            model=q_model,
            tokenizer=tok,
            train_rows=q_train,
            val_rows=None,
            system_prompt=PARA_SYSTEM,
            user_builder=lambda r: build_para_user(r["src"]),
            target_key="tgt",
            out_dir=os.path.join(epoch_dir, "q_lora_updated"),
            max_seq_len=args.max_seq_len_q,
            lr=args.lr,
            num_epochs=1.0,
            train_batch_size=args.train_bs,
            grad_accum=args.grad_accum,
            max_steps=args.train_steps_per_iter,
            bf16=(args.dtype == "bf16"),
            fp16=(args.dtype == "fp16"),
            gradient_checkpointing=True,
        )
        t2 = time.time()
        loop_trian_times.append(t2 - t1)
        print(f"Training for iteration {ep} done in {t2 - t1:.2f} seconds.")

        # evaluate 
        me = {"epoch": ep}
        if (ep in [1, 2, 3, 4, 5] or ep % 6 == 0) and ep != args.iters:
            me.update(evaluate_p(tok, p_model, sum_val_eval, reward_tra_eval_same=reward_tra_eval_same, reward_type=args.reward_type))
            me.update(evaluate_q(tok, q_model, para_val_eval, reward_tra_eval_same=reward_tra_eval_same, reward_type=args.reward_type))
            me.update(evaluate_p_on_q(tok, p_model, para_val_diag, reward_type=args.reward_type))
            me.update(evaluate_q_on_p(tok, q_model, sum_val_diag, reward_type=args.reward_type))
            me.update(evaluate_p_sampling(tok, p_model, sum_val_diag, reward_type=args.reward_type))
            me.update(evaluate_q_sampling(tok, q_model, para_val_diag, reward_type=args.reward_type))
            t3 = time.time()
            me['eval_time'] = t3 - t2
            loop_eval_times.append({f'epoch {ep} eval': t3 - t2})
            print(f"Evaluation for iteration {ep} done in {t3 - t2:.2f} seconds.")

        # training loss
        me["p_train_loss"] = train_loss_p
        me["q_train_loss"] = train_loss_q
        me['data_build_time'] = t1 - ep_start_time
        me['train_time'] = t2 - t1
        # update prev pools for next epoch
        p_prev_pool = p_train
        q_prev_pool = q_train

        ep_end_time = time.time()
        loop_end_times.append(ep_end_time)
        print(f"Iteration {ep} done in {ep_end_time - ep_start_time:.2f} seconds.")

        metrics.append(me)
        safe_jsonl_write(os.path.join(args.out_dir, "metrics.jsonl"), metrics)

    final = {"epoch": "final"}
    final.update(evaluate_p(tok, p_model, sum_val_eval, reward_tra_eval_same=reward_tra_eval_same, reward_type=args.reward_type))
    final.update(evaluate_q(tok, q_model, para_val_eval, reward_tra_eval_same=reward_tra_eval_same, reward_type=args.reward_type))
    metrics.append(final)
    safe_jsonl_write(os.path.join(args.out_dir, "metrics.jsonl"), metrics)
    print(f"Done. Metrics written to {os.path.join(args.out_dir, 'metrics.jsonl')}")
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
