"""
scripts/train_grpo.py

Phase 3 — GRPO reinforcement learning on GSM8K.
Starts from a Phase 1/2 SFT checkpoint and applies policy gradient training
using execution accuracy as the binary reward signal.

GRPO (Group Relative Policy Optimization) — same algorithm as DeepSeek-R1:
  - Sample G completions per prompt (num_generations)
  - Reward each: 1.0 if #### answer correct, 0.0 otherwise
  - Normalize rewards within the group: r_i = (r_i - mean) / std
  - KL penalty (beta) prevents the policy from drifting too far from SFT init

Usage:
    # Phase 3 Experiment A — best small model + GRPO
    python scripts/train_grpo.py \
        --sft-checkpoint /scratch/ngangada/portfolio/sft-lora-rl/outputs/lora_small \
        --method lora \
        --model small \
        --run-name grpo_lora_small

    # Phase 3 Experiment B — best large model + GRPO
    python scripts/train_grpo.py \
        --sft-checkpoint /scratch/ngangada/portfolio/sft-lora-rl/outputs/lora_medium \
        --method lora \
        --model medium \
        --run-name grpo_lora_medium

    # Phase 3 Experiment C — full SFT + GRPO (no adapters)
    python scripts/train_grpo.py \
        --sft-checkpoint /scratch/ngangada/portfolio/sft-lora-rl/outputs/full_small \
        --method full \
        --model small \
        --run-name grpo_full_small

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
from peft import PeftModel

from src.data_utils import load_grpo_datasets, extract_answer, is_correct
from src.model_utils import get_tokenizer, log_param_counts
from src.evaluate import evaluate_gsm8k


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def peak_gpu_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


# ── Reward function ───────────────────────────────────────────────────────────

def make_reward_fn(tokenizer):
    """
    Returns a reward function compatible with GRPOTrainer's signature:
        reward_fn(prompts, completions, **kwargs) -> list[float]

    GRPOTrainer passes:
        prompts     : list[str]  — the input prompts (same as dataset 'prompt' field)
        completions : list[str]  — model-generated completions
        **kwargs    : extra dataset columns, including 'ground_truth'

    Reward:
        1.0  if the extracted #### number matches ground_truth exactly
        0.0  otherwise
    """
    def reward_fn(prompts, completions, ground_truth=None, **kwargs):
        rewards = []
        gt_list = ground_truth if ground_truth is not None else [""] * len(completions)
        for completion, gt in zip(completions, gt_list):
            reward = 1.0 if is_correct(completion, gt) else 0.0
            rewards.append(reward)
        return rewards

    return reward_fn


# ── Model loading for Phase 3 ─────────────────────────────────────────────────

def load_sft_checkpoint(sft_checkpoint: str, method: str, base_model_name: str, config: dict):
    """
    Load the SFT checkpoint as the starting policy for GRPO.
    For LoRA: loads base + adapter (does NOT merge — we keep adapter for continued training).
    For full: loads the saved model directly.
    """
    if method == "full":
        tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            sft_checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.enable_input_require_grads()

    elif method in ("lora", "qlora"):
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

        if method == "qlora":
            # For GRPO we need bf16 (not 4-bit) to compute gradients properly
            # QLoRA checkpoint: load base in bf16, reload adapter weights
            print("[grpo] QLoRA checkpoint: loading in bf16 for GRPO training")
            base = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

        model = PeftModel.from_pretrained(base, sft_checkpoint, is_trainable=True)
        model.enable_input_require_grads()

    else:
        raise ValueError(f"Unknown method: {method}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # GRPOTrainer prefers left-padding for generation

    return model, tokenizer


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GRPO RL training on GSM8K")
    parser.add_argument("--sft-checkpoint", type=str, default=None,
                        help="Path to Phase 1/2 SFT checkpoint. "
                             "If omitted, starts from base model (cold start, not recommended).")
    parser.add_argument("--model",  default="small",  choices=["small", "medium"])
    parser.add_argument("--method", default="lora",   choices=["full", "lora", "qlora"])
    parser.add_argument("--run-name",  type=str, default=None,
                        help="W&B run name. Defaults to grpo_{method}_{model}.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--no-eval",    action="store_true")
    parser.add_argument("--smoke-test", action="store_true",
                        help="50 samples, 1 epoch — test the reward fn and trainer loop")
    args = parser.parse_args()

    config     = load_config(args.config)
    model_name = config["models"][args.model]
    gcfg       = config["grpo"]
    run_name   = args.run_name or f"grpo_{args.method}_{args.model}"
    output_dir = os.path.join(config["sft"]["output_base_dir"], run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Determine SFT checkpoint
    sft_ckpt = args.sft_checkpoint
    if sft_ckpt is None:
        sft_ckpt = os.path.join(config["sft"]["output_base_dir"], f"{args.method}_{args.model}")
        print(f"[grpo] --sft-checkpoint not set, using default: {sft_ckpt}")

    # Smoke test overrides
    max_train = 50   if args.smoke_test else gcfg["max_train_samples"]
    max_eval  = 30   if args.smoke_test else gcfg["max_eval_samples"]
    epochs    = 1    if args.smoke_test else gcfg["num_train_epochs"]

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        name=run_name,
        job_type="grpo",
        tags=[args.method, args.model, "phase3", "rl", "grpo"],
        config={
            "model_name":      model_name,
            "method":          args.method,
            "model_size":      args.model,
            "phase":           3,
            "sft_checkpoint":  sft_ckpt,
            "dataset":         "openai/gsm8k",
            "max_train_samples": max_train,
            "max_eval_samples":  max_eval,
            **gcfg,
        },
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, eval_ds = load_grpo_datasets(
        max_train_samples=max_train,
        max_eval_samples=max_eval,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    if os.path.isdir(sft_ckpt):
        print(f"[grpo] Loading SFT checkpoint: {sft_ckpt}")
        model, tokenizer = load_sft_checkpoint(sft_ckpt, args.method, model_name, config)
    else:
        print(f"[grpo] WARNING: SFT checkpoint not found at {sft_ckpt}")
        print(f"[grpo] Starting from base model {model_name} (cold start)")
        from src.model_utils import load_lora, load_full, load_qlora
        loaders = {"full": load_full, "lora": load_lora, "qlora": load_qlora}
        model, tokenizer = loaders[args.method](model_name, config)

    log_param_counts(model, run_name)

    # ── Reward fn ─────────────────────────────────────────────────────────────
    reward_fn = make_reward_fn(tokenizer)

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
        # GRPO-specific
        max_new_tokens=gcfg["max_new_tokens"],
        num_generations=gcfg["num_generations"],    # G — samples per prompt
        beta=gcfg["beta"],                          # KL penalty coefficient
        max_prompt_length=gcfg["max_seq_length"],
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=grpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        reward_funcs=reward_fn,
    )

    print(f"\n{'='*60}")
    print(f"  Starting GRPO: {run_name}")
    print(f"  Model:         {model_name}")
    print(f"  SFT init:      {sft_ckpt}")
    print(f"  Train samples: {len(train_ds):,} × {epochs} epochs")
    print(f"  Generations G: {gcfg['num_generations']} per prompt")
    print(f"  KL beta:       {gcfg['beta']}")
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
    print(f"[grpo] Checkpoint saved → {output_dir}")

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
    print(f"\n✅  {run_name} complete.\n")


if __name__ == "__main__":
    main()