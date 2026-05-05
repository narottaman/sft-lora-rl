"""
src/model_utils.py

Unified model loader for all three fine-tuning methods.
Returns (model, tokenizer) ready for SFTTrainer or GRPOTrainer.

Methods:
  "full"  — all weights trainable, bf16
  "lora"  — frozen base + LoRA adapters on attn + MLP projections
  "qlora" — 4-bit NF4 quantized base + LoRA adapters (bitsandbytes)
"""

import torch
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def get_tokenizer(model_name: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"   # required by SFTTrainer / GRPOTrainer
    return tok


# ── Parameter counting ────────────────────────────────────────────────────────

def count_parameters(model) -> tuple[int, int]:
    """Returns (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total


def log_param_counts(model, run_name: str):
    trainable, total = count_parameters(model)
    pct = 100.0 * trainable / total if total else 0.0
    print(f"[model] {run_name} — trainable: {trainable:,} / {total:,} ({pct:.2f}%)")
    if wandb.run:
        wandb.log({
            "trainable_params": trainable,
            "total_params":     total,
            "trainable_pct":    pct,
        })
    return trainable, total, pct


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_full(model_name: str, config: dict):
    """Full fine-tuning — all weights in bf16, all trainable."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.enable_input_require_grads()
    return model, get_tokenizer(model_name)


def load_lora(model_name: str, config: dict):
    """LoRA — bf16 base model with low-rank adapter on all projection layers."""
    lcfg = config["lora"]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["lora_alpha"],
        target_modules=lcfg["target_modules"],
        lora_dropout=lcfg["lora_dropout"],
        bias=lcfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    return model, get_tokenizer(model_name)


def load_qlora(model_name: str, config: dict):
    """QLoRA — 4-bit NF4 quantized base + LoRA adapters in bf16 compute dtype."""
    qcfg = config["qlora"]
    lcfg = config["lora"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qcfg["load_in_4bit"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    # Required before applying LoRA to a quantized model
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["lora_alpha"],
        target_modules=lcfg["target_modules"],
        lora_dropout=lcfg["lora_dropout"],
        bias=lcfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    return model, get_tokenizer(model_name)


# ── Factory ───────────────────────────────────────────────────────────────────

_LOADERS = {
    "full":  load_full,
    "lora":  load_lora,
    "qlora": load_qlora,
}


def get_model_and_tokenizer(
    model_name: str,
    method: str,
    config: dict,
    run_name: str = "",
):
    """
    Factory — returns (model, tokenizer) for the given method.
    Logs trainable parameter counts to W&B if a run is active.

    Args:
        model_name : HuggingFace model ID, e.g. "Qwen/Qwen2.5-0.5B-Instruct"
        method     : one of "full", "lora", "qlora"
        config     : parsed config.yaml dict
        run_name   : label for console logging
    """
    if method not in _LOADERS:
        raise ValueError(f"Unknown method '{method}'. Choose: {list(_LOADERS)}")

    print(f"[model] Loading {model_name} — method={method}")
    model, tokenizer = _LOADERS[method](model_name, config)
    log_param_counts(model, run_name or method)
    return model, tokenizer