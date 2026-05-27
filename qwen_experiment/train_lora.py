from __future__ import annotations
import os
is_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
from typing import Dict, List

from src.data import safe_jsonl_read
from src.modeling import load_base_model, attach_lora
from src.sft import train_lora_sft

SUM_SYSTEM = ("You are a helpful English summarization assistant. "
              "Write a single-sentence, information-dense summary that is as short as possible. "
              "Do not add new facts or embellishments.")
PARA_SYSTEM = ("You are an English paraphrasing assistant. "
                 "Rewrite the text with different words while maintaining the core meaning."
                 "Do not add new facts.")


def build_sum_user(row: Dict) -> str:
    return f"Please summarize the following text in ONE sentence:\n{row['article']}"

def build_para_user(row: Dict) -> str:
    return f"Please paraphrase the following text while maintaining the core meaning:\n{row['src']}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, choices=["summarize", "paraphrase"], required=True)
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--val_jsonl", type=str, default=None)
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out_dir", type=str, required=True)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # Training
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num_epochs", type=float, default=1.0)
    ap.add_argument("--train_bs", type=int, default=2)
    ap.add_argument("--eval_bs", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--dtype", type=str, choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--load_in_4bit", action="store_true")
    args = ap.parse_args()

    train_rows = safe_jsonl_read(args.train_jsonl)
    val_rows = safe_jsonl_read(args.val_jsonl) if args.val_jsonl else None

    bundle = load_base_model(
        args.base_model,
        dtype=args.dtype,
        device_map="auto",
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=True,
    )
    model = attach_lora(
        bundle.model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    if args.task == "summarize":
        system = SUM_SYSTEM
        user_builder = build_sum_user
        # target is reference summary
        target_key = "summary"
        # Some JSONL may use "tgt"; support both
        if train_rows and "summary" not in train_rows[0] and "tgt" in train_rows[0]:
            target_key = "tgt"
    else:
        system = PARA_SYSTEM
        user_builder = build_para_user
        target_key = "tgt"

    bf16 = args.dtype == "bf16"
    fp16 = args.dtype == "fp16"

    train_lora_sft(
        model=model,
        tokenizer=bundle.tokenizer,
        train_rows=train_rows,
        val_rows=val_rows,
        system_prompt=system,
        user_builder=user_builder,
        target_key=target_key,
        out_dir=args.out_dir,
        max_seq_len=args.max_seq_len,
        lr=args.lr,
        num_epochs=args.num_epochs,
        train_batch_size=args.train_bs,
        eval_batch_size=args.eval_bs,
        grad_accum=args.grad_accum,
        max_steps=args.max_steps,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=(not is_ddp),
    )

    print(f"Saved LoRA adapter to: {args.out_dir}")

if __name__ == "__main__":
    main()
