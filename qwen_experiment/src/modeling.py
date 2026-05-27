from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel

@dataclass
class ModelBundle:
    tokenizer: Any
    model: Any

def load_base_model(
    base_model: str,
    dtype: str = "fp16",
    device_map: str | Dict[str, int] | None = "auto",
    load_in_4bit: bool = False,
    trust_remote_code: bool = True,
) -> ModelBundle:
    torch_dtype = None
    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "fp32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unknown dtype: {dtype}")
    
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1
    if is_distributed:
        device_map = None

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    # Qwen usually has pad_token unset for causal LM; set to eos to avoid warnings.
    # if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    kwargs = dict(
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code, 
    )
    if load_in_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit = True, 
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_type = torch_dtype, 
            bnb_4bit_use_double_quant = True,
        )
        kwargs["quantization_config"] = bnb_cfg

    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    model.config.use_cache = False
    return ModelBundle(tokenizer=tokenizer, model=model)

def attach_lora(
    base_model,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[list[str]] = None,
) -> Any:
    """Wrap a base model with a trainable LoRA adapter (PEFT)."""
    if target_modules is None:
        # Reasonable default for Qwen-like architectures.
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    return get_peft_model(base_model, cfg)

def load_lora_into_base(base_model, lora_dir: str, is_trainable: bool = True) -> Any:
    """Load an existing adapter weights directory onto the base model."""
    return PeftModel.from_pretrained(base_model, lora_dir, is_trainable=is_trainable)

def merge_and_unload(peft_model) -> Any:
    """Optional: merges LoRA into base weights (not used by default)."""
    return peft_model.merge_and_unload()
