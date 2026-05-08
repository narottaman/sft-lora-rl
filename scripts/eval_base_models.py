"""
scripts/eval_base_models.py

Evaluates raw pretrained base models (no fine-tuning) across all benchmarks.
Establishes the true zero-shot baseline.

FIXES:
  1. max_new_tokens=512 for GSM8K — base models write verbose reasoning
     that gets truncated at 256 tokens before reaching the final answer.
  2. regex MC extractor — base models write "The answer is A" not just "A",
     so strip()[0] grabs 'T'. regex \\b[ABCD]\\b finds the letter correctly.
  3. flexible extract_answer — catches "makes $18", "= 18 dollars", "#### 18"
     so base and fine-tuned models are evaluated consistently.

Expected results (from paper Zhuang et al. 2025):
  Qwen2.5-0.5B-Instruct: GSM8K ~45.5% zero-shot
  Qwen2.5-3B-Instruct:   GSM8K ~70%+ zero-shot

Usage:
    python scripts/eval_base_models.py
    python scripts/eval_base_models.py --model small
    python scripts/eval_base_models.py --model medium
"""

import os
import sys
import json
import re
import argparse
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import wandb
import yaml
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.data_utils import extract_answer   # same flexible extractor as evaluate.py


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


BASE_MODELS = {
    "small":  "Qwen/Qwen2.5-0.5B-Instruct",
    "medium": "Qwen/Qwen2.5-3B-Instruct",
}

MMLU_SUBJECTS = [
    "high_school_mathematics", "elementary_mathematics",
    "high_school_computer_science", "college_computer_science",
    "high_school_physics", "high_school_chemistry",
    "logical_fallacies", "formal_logic",
    "world_history", "us_history",
]


def load_base_model(model_name: str):
    print(f"[base] Loading: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"[base] {params/1e9:.2f}B parameters")
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def build_prompt(tokenizer, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_mc(output: str) -> str:
    """
    Extract A/B/C/D from multiple choice output.
    FIX: regex instead of strip()[0].
    Base models write 'The answer is A' — strip()[0] grabs 'T' not 'A'.
    """
    m = re.search(r'\b([ABCD])\b', output[:150])
    return m.group(1).upper() if m else ""


# ── Benchmarks ────────────────────────────────────────────────────────────────

def eval_gsm8k(model, tokenizer, n_samples=300) -> dict:
    """
    Uses 512 max_new_tokens and flexible extractor.
    Same 300-sample count as our fine-tuned model evals for fair comparison.
    """
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.select(range(min(n_samples, len(ds))))

    system = (
        "You are a helpful math assistant. Solve step by step. "
        "Write the final answer as: #### <number>"
    )
    correct = 0
    debug_shown = 0
    for ex in tqdm(ds, desc="GSM8K", leave=False):
        prompt = build_prompt(tokenizer, system, ex["question"])
        output = generate(model, tokenizer, prompt, max_new_tokens=512)
        pred   = extract_answer(output)
        gt     = extract_answer(ex["answer"])
        if pred and gt and pred == gt:
            correct += 1
        if debug_shown < 3:
            print(f"\n  [gsm8k] Q: {ex['question'][:60]}...")
            print(f"  [gsm8k] Output: {output[:150]}...")
            print(f"  [gsm8k] Pred={pred} GT={gt} Correct={pred==gt if pred and gt else False}")
            debug_shown += 1

    return {"gsm8k_accuracy": correct / len(ds),
            "gsm8k_correct": correct, "gsm8k_total": len(ds)}


def eval_mmlu(model, tokenizer, n_per_subject=15) -> dict:
    system = "Answer the multiple choice question. Reply with only A, B, C, or D."
    correct = total = 0
    for subject in tqdm(MMLU_SUBJECTS, desc="MMLU", leave=False):
        try:
            ds = load_dataset("cais/mmlu", subject, split="test")
            ds = ds.select(range(min(n_per_subject, len(ds))))
        except Exception:
            continue
        for ex in ds:
            choices    = ex["choices"]
            choice_str = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", choices))
            user       = f"Question: {ex['question']}\n\n{choice_str}"
            prompt     = build_prompt(tokenizer, system, user)
            output     = generate(model, tokenizer, prompt, max_new_tokens=10)
            pred       = extract_mc(output)
            gt         = "ABCD"[ex["answer"]]
            if pred == gt:
                correct += 1
            total += 1
    return {"mmlu_accuracy": correct / total if total else 0,
            "mmlu_correct": correct, "mmlu_total": total}


def eval_arc_easy(model, tokenizer, n_samples=100) -> dict:
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    ds = ds.select(range(min(n_samples, len(ds))))
    system = "Answer the science question. Reply with only A, B, C, or D."
    correct = 0
    for ex in tqdm(ds, desc="ARC-Easy", leave=False):
        labels     = ex["choices"]["label"]
        texts      = ex["choices"]["text"]
        choice_str = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
        user       = f"Question: {ex['question']}\n\n{choice_str}"
        prompt     = build_prompt(tokenizer, system, user)
        output     = generate(model, tokenizer, prompt, max_new_tokens=10)
        pred       = extract_mc(output)
        if pred == ex["answerKey"].upper():
            correct += 1
    return {"arc_easy_accuracy": correct / len(ds),
            "arc_easy_correct": correct, "arc_easy_total": len(ds)}


def eval_hellaswag(model, tokenizer, n_samples=100) -> dict:
    ds = load_dataset("Rowan/hellaswag", split="validation")
    ds = ds.select(range(min(n_samples, len(ds))))
    system = "Choose the most logical continuation. Reply with only A, B, C, or D."
    correct = 0
    for ex in tqdm(ds, desc="HellaSwag", leave=False):
        endings    = ex["endings"]
        choice_str = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", endings))
        user       = f"Context: {ex['ctx']}\n\nContinuation:\n{choice_str}"
        prompt     = build_prompt(tokenizer, system, user)
        output     = generate(model, tokenizer, prompt, max_new_tokens=10)
        pred       = extract_mc(output)
        gt         = "ABCD"[int(ex["label"])]
        if pred == gt:
            correct += 1
    return {"hellaswag_accuracy": correct / len(ds),
            "hellaswag_correct": correct, "hellaswag_total": len(ds)}


BENCHMARKS = {
    "gsm8k":     eval_gsm8k,
    "mmlu":      eval_mmlu,
    "arc_easy":  eval_arc_easy,
    "hellaswag": eval_hellaswag,
}

DEFAULT_BENCHMARKS = ["gsm8k", "mmlu", "arc_easy", "hellaswag"]


def evaluate_base_model(model_size, model_name, benchmarks, config) -> dict:
    print(f"\n{'='*60}")
    print(f"  Base model: {model_name} ({model_size})")
    print(f"  Benchmarks: {benchmarks}")
    print(f"{'='*60}")

    run_name = f"base_{model_size}"
    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        name=run_name,
        job_type="baseline",
        tags=[model_size, "base", "no-finetuning"],
        config={"model_name": model_name, "model_size": model_size,
                "checkpoint": "base", "phase": 0, "type": "base"},
        reinit=True,
    )

    model, tokenizer = load_base_model(model_name)
    results = {"checkpoint": run_name, "method": "none", "model": model_size,
               "phase": 0, "type": "base", "model_name": model_name}

    for bench in benchmarks:
        if bench not in BENCHMARKS:
            continue
        print(f"\n  Running {bench}...")
        t0 = time.time()
        r  = BENCHMARKS[bench](model, tokenizer)
        r[f"{bench}_time_sec"] = time.time() - t0
        results.update(r)
        wandb.log({k: v for k, v in r.items() if isinstance(v, float)})
        for k, v in r.items():
            if "accuracy" in k:
                print(f"    {k}: {v*100:.1f}%")

    peak_gb = torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else 0
    wandb.log({"peak_gpu_memory_gb": peak_gb})
    wandb.finish()
    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, choices=["small", "medium"])
    parser.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS,
                        choices=list(BENCHMARKS.keys()))
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--output-dir", default="data/benchmark_results")
    args = parser.parse_args()

    config = load_config(args.config)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    models = {args.model: BASE_MODELS[args.model]} if args.model else BASE_MODELS

    all_results = []
    for size, name in models.items():
        result = evaluate_base_model(size, name, args.benchmarks, config)
        all_results.append(result)

    out_path = os.path.join(args.output_dir, "base_models.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[done] Saved → {out_path}")

    print(f"\n{'='*60}")
    print("Base Model Results")
    print("=" * 60)
    for r in all_results:
        print(f"\n{r['model_name']}")
        for k, v in r.items():
            if "accuracy" in k:
                print(f"  {k:<30}: {v*100:.1f}%")


if __name__ == "__main__":
    main()