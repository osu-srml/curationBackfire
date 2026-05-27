\
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

import torch
from torch.utils.data import Dataset
import inspect
# from transformers import Trainer, TrainingArguments
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from .reward import rouge_l_f1

def build_chat_prompt(tokenizer, system: str, user: str, add_generation_prompt: bool) -> str:
    messages = [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": user.strip()},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

@dataclass
class ChatExample:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]

class ChatSFTDataset(Dataset):
    """SFT dataset where loss is only on assistant tokens (prompt tokens masked to -100)."""

    def __init__(
        self,
        rows: List[Dict],
        tokenizer,
        system_prompt: str,
        user_builder,
        target_key: str,
        max_seq_len: int = 512,
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.user_builder = user_builder
        self.target_key = target_key
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.rows)

    def _make(self, row: Dict) -> ChatExample:
        user = self.user_builder(row)
        assistant = (row.get(self.target_key) or "").strip()

        # Build texts
        prompt_text = build_chat_prompt(
            self.tokenizer, self.system_prompt, user,
            add_generation_prompt=True
        )

        # Make tokenizer behavior explicit
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            # for decoder-only models
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Tokenize prompt and assistant separately (no truncation here!)
        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        # For assistant, append EOS so the model learns to stop
        assistant_ids = self.tokenizer(
            assistant,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        if len(assistant_ids) == 0:
            return None
        assistant_ids = assistant_ids + [self.tokenizer.eos_token_id]

        max_len = int(self.max_seq_len)

        # Ensure assistant fits (keep at least 1 token + eos)
        if len(assistant_ids) >= max_len:
            # Keep the *last* tokens of assistant (left-truncate), but always end with EOS
            assistant_ids = assistant_ids[-(max_len - 1):] + [self.tokenizer.eos_token_id]

        # Truncate prompt to leave room for assistant (this is the key fix)
        avail = max_len - len(assistant_ids)
        if avail <= 0:
            # shouldn't happen due to step (2), but be safe
            return None
        prompt_ids = prompt_ids[-avail:]  # keep tail of prompt

        # Build final sequence
        input_ids = prompt_ids + assistant_ids
        attention_mask = [1] * len(input_ids)

        # Labels: mask prompt, supervise assistant
        labels = [-100] * len(prompt_ids) + assistant_ids[:]

        # Safety: if something went wrong, drop
        if all(x == -100 for x in labels):
            return None

        return ChatExample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )


    def __getitem__(self, idx):
        n = len(self.rows)
        for _ in range(8):  # try a few times; dataset should be mostly valid
            ex = self._make(self.rows[idx])
            if ex is not None:
                return ex
            idx = (idx + 1) % n
        # If still invalid, raise to surface a real data issue
        raise RuntimeError("Too many invalid SFT examples (all labels=-100). Check max_seq_len / truncation.")

@dataclass
class DataCollatorForChatSFT:
    tokenizer: Any

    def __call__(self, features: List[ChatExample]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f.input_ids) for f in features)
        input_ids = []
        attention_mask = []
        labels = []
        pad_id = self.tokenizer.pad_token_id
        for f in features:
            pad = max_len - len(f.input_ids)
            input_ids.append(f.input_ids + [pad_id] * pad)
            attention_mask.append(f.attention_mask + [0] * pad)
            labels.append(f.labels + [-100] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def simple_eval_metrics(tokenizer, system_prompt: str, user_builder, target_key: str):
    """A lightweight eval metric: ROUGE-L between decoded model outputs and target.
    Only used when `predict_with_generate=True` (we enable it in training args).
    """
    def _compute(eval_pred):
        preds, labels = eval_pred
        # preds can be tuples
        if isinstance(preds, tuple):
            preds = preds[0]
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        else:
            preds = np.asarray(preds)

        # if logits: [B,T,V] -> [B,T]
        if preds.ndim == 3:
            preds = preds.argmax(axis=-1)

        preds = preds.astype(np.int64)

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        else:
            labels = np.asarray(labels)
        labels = labels.astype(np.int64)
        labels[labels == -100] = tokenizer.pad_token_id

        preds = np.clip(preds, 0, tokenizer.vocab_size - 1)
        labels = np.clip(labels, 0, tokenizer.vocab_size - 1)

        pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # A rough extraction: take the last assistant chunk
        def tail_assistant(s: str) -> str:
            s = s.strip()
            # Often chat templates include "assistant" tokens; keep tail after last newline.
            parts = [p.strip() for p in s.split("\n") if p.strip()]
            return parts[-1] if parts else s

        scores = []
        for p, t in zip(pred_texts, label_texts):
            scores.append(rouge_l_f1(tail_assistant(p), tail_assistant(t)))
        return {"rougeL": float(sum(scores) / max(1, len(scores)))}
    return _compute

def train_lora_sft(
    model,
    tokenizer,
    train_rows: List[Dict],
    val_rows: Optional[List[Dict]],
    system_prompt: str,
    user_builder,
    target_key: str,
    out_dir: str,
    max_seq_len: int = 512,
    lr: float = 2e-4,
    num_epochs: float = 1.0,
    train_batch_size: int = 2,
    grad_accum: int = 8,
    eval_batch_size: int = 2,
    warmup_ratio: float = 0.03,
    logging_steps: int = 20,
    save_steps: int = 200,
    eval_steps: int = 200,
    max_steps: Optional[int] = None,
    bf16: bool = False,
    fp16: bool = True,
    gradient_checkpointing: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)
    if hasattr(model, "config"):
        model.config.use_cache = False

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    train_ds = ChatSFTDataset(
        rows=train_rows,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        user_builder=user_builder,
        target_key=target_key,
        max_seq_len=max_seq_len,
    )
    val_ds = None
    if val_rows is not None and len(val_rows) > 0:
        val_ds = ChatSFTDataset(
            rows=val_rows,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            user_builder=user_builder,
            target_key=target_key,
            max_seq_len=max_seq_len,
        )

    collator = DataCollatorForChatSFT(tokenizer=tokenizer)
    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=lr,
        num_train_epochs=num_epochs if max_steps is None else 1.0,
        max_steps=max_steps if max_steps is not None else -1,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=2,
        bf16=bf16,
        fp16=fp16 and (not bf16),
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        eval_strategy="steps" if val_ds is not None else "no",
        predict_with_generate=True if val_ds is not None else False,
        generation_max_length=max_seq_len+64,
        generation_num_beams=1,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=simple_eval_metrics(tokenizer, system_prompt, user_builder, target_key) if val_ds is not None else None,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable_params =", trainable)
    assert trainable > 0, "No trainable params — LoRA is likely frozen."

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    trainer.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    hist = trainer.state.log_history
    losses = [h["loss"] for h in hist if "loss" in h]
    train_loss = float(sum(losses[-20:]) / max(1, len(losses[-20:]))) if losses else None

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir, train_loss
