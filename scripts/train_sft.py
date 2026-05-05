"""
scripts/train_sft.py

Phase 1 + Phase 2 training — SFT, LoRA, QLoRA on Qwen2.5-0.5B and 3B.

Every run is a single (model_size, method) combination.
W&B tracks: loss curves, trainable %, GPU memory, GSM8K accuracy.

Usage:
    python scripts/train_sft.py --model small --method full     # Phase 1
    python scripts/train_sft.py --model small --method lora     # Phase 1
    python scripts/train_sft.py --model small --method qlora    # Phase 1
    python scripts/train_sft.py --model medium --method lora    # Phase 2
    python scripts/train_sft.py --model medium --method qlora   # Phase 2
    python scripts/train_sft.py --model medium --method full    # optional 3B full

Flags:
    --no-eval     skip post-training GSM8K evaluation (faster for debugging)
    --smoke-test  100 samples, 1 epoch — verify pipeline works before Sol

NOTE on TRL versions:
    TRL >= 0.13 moved max_seq_length / dataset_text_field / packing
    out of SFTConfig and into SFTTrainer directly.
    This file is written for TRL >= 0.13.
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import wandb
from trl import SFTConfig, SFTTrainer

from src.data_utils import load_sft_datasets
from src.model_utils import get_model_and_tokenizer
from src.evaluate import evaluate_gsm8k


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def peak_gpu_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SFT / LoRA / QLoRA training on GSM8K")
    parser.add_argument("--model",  default="small",  choices=["small", "medium"])
    parser.add_argument("--method", default="lora",   choices=["full", "lora", "qlora"])
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--no-eval",    action="store_true", help="Skip accuracy eval")
    parser.add_argument("--smoke-test", action="store_true", help="Tiny run to test pipeline")
    args = parser.parse_args()

    config     = load_config(args.config)
    model_name = config["models"][args.model]
    run_name   = f"{args.method}_{args.model}"
    tcfg       = config["sft"]
    output_dir = os.path.join(tcfg["output_base_dir"], run_name)
    os.makedirs(output_dir, exist_ok=True)

    # smoke-test overrides
    max_train = 100 if args.smoke_test else config["dataset"]["max_train_samples"]
    max_eval  = 50  if args.smoke_test else config["dataset"]["max_eval_samples"]
    epochs    = 1   if args.smoke_test else tcfg["num_train_epochs"]

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        name=run_name,
        job_type="sft",
        tags=[args.method, args.model, "phase1" if args.model == "small" else "phase2"],
        config={
            "model_name":        model_name,
            "method":            args.method,
            "model_size":        args.model,
            "phase":             1 if args.model == "small" else 2,
            "dataset":           "openai/gsm8k",
            "max_train_samples": max_train,
            "max_eval_samples":  max_eval,
            **tcfg,
            **(config["lora"]  if args.method in ("lora", "qlora") else {}),
            **(config["qlora"] if args.method == "qlora"           else {}),
        },
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, eval_ds = load_sft_datasets(
        max_train_samples=max_train,
        max_eval_samples=max_eval,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model, tokenizer = get_model_and_tokenizer(
        model_name=model_name,
        method=args.method,
        config=config,
        run_name=run_name,
    )

    # ── Training args ─────────────────────────────────────────────────────────
    # NOTE: max_seq_length / dataset_text_field / packing belong on SFTTrainer
    # in TRL >= 0.13, NOT on SFTConfig. Putting them on SFTConfig raises:
    # TypeError: SFTConfig.__init__() got an unexpected keyword argument
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        warmup_ratio=tcfg["warmup_ratio"],
        bf16=tcfg["bf16"],
        fp16=tcfg["fp16"],
        logging_steps=tcfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=tcfg["eval_steps"],
        save_strategy="steps",
        save_steps=tcfg["save_steps"],
        save_total_limit=tcfg["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb",
        run_name=run_name,
        max_length=tcfg["max_seq_length"],
        dataset_text_field="text",
        packing=tcfg["packing"],
    )
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )
    print(f"\n{'='*60}")
    print(f"  Starting: {run_name}")
    print(f"  Model:    {model_name}")
    print(f"  Method:   {args.method}")
    print(f"  Train:    {len(train_ds):,} samples x {epochs} epochs")
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
    print(f"\n[train] Done — {elapsed/60:.1f} min | Peak GPU: {mem_gb:.1f} GB")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[train] Checkpoint saved -> {output_dir}")

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