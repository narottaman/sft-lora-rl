"""
scripts/train_grpo.py

Phase 3 — GRPO reinforcement learning on GSM8K.
Starts from any Phase 1/2 SFT checkpoint (full, lora, or qlora)
and applies policy gradient training using execution accuracy as reward.

Weight update behavior per method:
  full  — all weights trainable, highest task performance, risk of forgetting
  lora  — only adapter matrices updated, base frozen, best generalization
  qlora — same as lora but base is 4-bit quantized, most memory efficient

GRPO algorithm (same as DeepSeek-R1):
  1. Sample G completions per prompt (num_generations)
  2. Reward each: 1.0 if #### answer correct, 0.0 otherwise
  3. Normalize rewards within the group: advantage = (r - mean) / std
  4. KL penalty (beta) keeps policy close to SFT initialization
  5. Policy gradient update on adapter weights (lora/qlora) or all weights (full)

Usage:
    # Single experiment
    python scripts/train_grpo.py --method lora   --model small  --run-name grpo_lora_small
    python scripts/train_grpo.py --method qlora  --model small  --run-name grpo_qlora_small
    python scripts/train_grpo.py --method full   --model small  --run-name grpo_full_small
    python scripts/train_grpo.py --method lora   --model medium --run-name grpo_lora_medium
    python scripts/train_grpo.py --method qlora  --model medium --run-name grpo_qlora_medium
    python scripts/train_grpo.py --method full   --model medium --run-name grpo_full_medium

    # Smoke test
    python scripts/train_grpo.py --method lora --model small --smoke-test
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import wandb
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

from src.data_utils import load_grpo_datasets, extract_answer, is_correct
from src.model_utils import log_param_counts
from src.evaluate import evaluate_gsm8k


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def peak_gpu_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


# ── Reward function ───────────────────────────────────────────────────────────

def make_reward_fn():
    """
    Returns a GRPO-compatible reward function.
    Signature: reward_fn(prompts, completions, **kwargs) -> list[float]
    ground_truth is passed via dataset columns in **kwargs.

    Reward:
        1.0 if extracted #### number matches ground_truth exactly
        0.0 otherwise
    """
    def reward_fn(prompts, completions, ground_truth=None, **kwargs):
        gt_list = ground_truth if ground_truth is not None else [""] * len(completions)
        return [
            1.0 if is_correct(completion, gt) else 0.0
            for completion, gt in zip(completions, gt_list)
        ]
    return reward_fn


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_checkpoint_for_grpo(
    sft_checkpoint: str,
    method: str,
    base_model_name: str,
    config: dict,
):
    """
    Load the SFT checkpoint as the starting policy for GRPO.

    Full:  load checkpoint directly, keep all weights trainable.
    LoRA:  load base model + adapter, keep adapter trainable, base frozen.
    QLoRA: load base in bf16 (NOT 4-bit) + reload adapter weights.
           We use bf16 for GRPO because:
           - 4-bit NF4 cannot accumulate gradients for policy updates
           - The LoRA adapter was trained in bf16 compute dtype anyway
           - bf16 base + LoRA adapter = same trainable params as QLoRA SFT
           - Memory is higher than QLoRA SFT but adapter still trains correctly

    Returns (model, tokenizer) with only the correct parameters requiring grad.
    """
    print(f"[grpo] Loading checkpoint: {sft_checkpoint}")
    print(f"[grpo] Method: {method} | Base: {base_model_name}")

    if method == "full":
        # All weights are trainable — same as SFT loading
        tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            sft_checkpoint,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.enable_input_require_grads()

    elif method == "lora":
        # Base frozen, only adapter matrices get GRPO gradient updates
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        # is_trainable=True keeps adapter weights requiring grad
        model = PeftModel.from_pretrained(base, sft_checkpoint, is_trainable=True)
        model.enable_input_require_grads()

    elif method == "qlora":
        # Load base in bf16 for gradient computation, reload adapter
        # The base model weights stay frozen — only adapter is updated by GRPO
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16,   # bf16, not 4-bit — needed for gradients
            device_map="auto",
            trust_remote_code=True,
        )
        # Load the adapter that was trained during QLoRA SFT
        model = PeftModel.from_pretrained(base, sft_checkpoint, is_trainable=True)
        model.enable_input_require_grads()
        print("[grpo] QLoRA checkpoint: base loaded in bf16 for GRPO gradient computation")

    else:
        raise ValueError(f"Unknown method: {method}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # GRPOTrainer prefers left-padding for generation

    return model, tokenizer


def load_base_model_for_grpo(method: str, base_model_name: str, config: dict):
    """
    Cold start — no SFT checkpoint available.
    Builds model from scratch with same architecture as SFT.
    Not recommended — SFT initialization makes GRPO much more stable.
    """
    print(f"[grpo] Cold start from base model: {base_model_name}")
    lcfg = config["lora"]

    if method == "full":
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.enable_input_require_grads()

    else:  # lora or qlora — attach fresh adapters
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        lora_config = LoraConfig(
            r=lcfg["r"],
            lora_alpha=lcfg["lora_alpha"],
            target_modules=lcfg["target_modules"],
            lora_dropout=lcfg["lora_dropout"],
            bias=lcfg["bias"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GRPO RL training on GSM8K")
    parser.add_argument("--method",   required=True, choices=["full", "lora", "qlora"])
    parser.add_argument("--model",    required=True, choices=["small", "medium"])
    parser.add_argument("--run-name", type=str, default=None,
                        help="W&B run name. Defaults to grpo_{method}_{model}.")
    parser.add_argument("--sft-checkpoint", type=str, default=None,
                        help="Path to SFT checkpoint. Defaults to outputs/{method}_{model}.")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--no-eval",    action="store_true")
    parser.add_argument("--smoke-test", action="store_true",
                        help="50 samples, 1 epoch — fast pipeline test")
    args = parser.parse_args()

    config     = load_config(args.config)
    model_name = config["models"][args.model]
    gcfg       = config["grpo"]
    run_name   = args.run_name or f"grpo_{args.method}_{args.model}"
    output_dir = os.path.join(config["sft"]["output_base_dir"], run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Resolve SFT checkpoint path
    sft_ckpt = args.sft_checkpoint or os.path.join(
        config["sft"]["output_base_dir"],
        f"{args.method}_{args.model}"
    )

    # Smoke test overrides
    max_train = 50  if args.smoke_test else gcfg["max_train_samples"]
    max_eval  = 30  if args.smoke_test else gcfg["max_eval_samples"]
    epochs    = 1   if args.smoke_test else gcfg["num_train_epochs"]

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        name=run_name,
        job_type="grpo",
        tags=[args.method, args.model, "phase3", "rl", "grpo"],
        config={
            "model_name":       model_name,
            "method":           args.method,
            "model_size":       args.model,
            "phase":            3,
            "sft_checkpoint":   sft_ckpt,
            "dataset":          "openai/gsm8k",
            "max_train_samples": max_train,
            "max_eval_samples":  max_eval,
            **gcfg,
        },
    )

    # ── Model — load FIRST so tokenizer is available for data formatting ──────
    if os.path.isdir(sft_ckpt):
        model, tokenizer = load_checkpoint_for_grpo(
            sft_ckpt, args.method, model_name, config
        )
    else:
        print(f"[grpo] WARNING: checkpoint not found at {sft_ckpt}, using cold start")
        model, tokenizer = load_base_model_for_grpo(args.method, model_name, config)

    log_param_counts(model, run_name)

    # ── Data — pass tokenizer so apply_chat_template uses correct format ──────
    train_ds, eval_ds = load_grpo_datasets(
        tokenizer=tokenizer,
        max_train_samples=max_train,
        max_eval_samples=max_eval,
    )

    # ── Reward function ───────────────────────────────────────────────────────
    reward_fn = make_reward_fn()

    # ── GRPO config ───────────────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=gcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=gcfg["gradient_accumulation_steps"],
        learning_rate=gcfg["learning_rate"],
        lr_scheduler_type=gcfg["lr_scheduler_type"],
        warmup_ratio=gcfg["warmup_ratio"],
        bf16=gcfg["bf16"],
        fp16=gcfg["fp16"],
        logging_steps=gcfg["logging_steps"],
        save_steps=gcfg["save_steps"],
        save_total_limit=gcfg["save_total_limit"],
        report_to="wandb",
        run_name=run_name,
        max_completion_length=gcfg["max_completion_length"],
        num_generations=gcfg["num_generations"],
        beta=gcfg["beta"],
        #max_prompt_length=gcfg["max_seq_length"],
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        reward_funcs=reward_fn,
    )

    print(f"\n{'='*60}")
    print(f"  GRPO Run:   {run_name}")
    print(f"  Model:      {model_name}")
    print(f"  Method:     {args.method}")
    print(f"  SFT init:   {sft_ckpt}")
    print(f"  Train:      {len(train_ds):,} samples x {epochs} epochs")
    print(f"  G (gens):   {gcfg['num_generations']} per prompt")
    print(f"  KL beta:    {gcfg['beta']}")
    print(f"{'='*60}\n")

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    mem_gb  = peak_gpu_gb()

    wandb.log({
        "train_time_sec":     elapsed,
        "train_time_min":     elapsed / 60,
        "peak_gpu_memory_gb": mem_gb,
    })
    print(f"\n[grpo] Done — {elapsed/60:.1f} min | Peak GPU: {mem_gb:.1f} GB")

    # ── Save ──────────────────────────────────────────────────────────────────
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[grpo] Checkpoint saved -> {output_dir}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    if not args.no_eval:
        print("\n[eval] Running GSM8K accuracy evaluation...")
        evaluate_gsm8k(
            model=trainer.model,
            tokenizer=tokenizer,
            run_name=run_name,
            max_samples=config["dataset"]["max_eval_samples"],
        )

    wandb.finish()
    print(f"\n[done] {run_name} complete.\n")


if __name__ == "__main__":
    main()