"""
scripts/eval_benchmarks.py

Evaluates all saved checkpoints across multiple benchmarks to test
generalization — specifically whether fine-tuning method (full vs LoRA vs QLoRA)
affects how much general capability is preserved after SFT and GRPO.

Benchmarks:
  gsm8k      — math reasoning (our training task, baseline)
  mmlu       — general knowledge across 57 subjects
  hellaswag  — commonsense reasoning / language understanding
  arc_easy   — grade school science, multiple choice
  truthfulqa — factual accuracy, tests hallucination
  humaneval  — code generation (tests cross-task generalization)

Usage:
    # Evaluate all checkpoints on all benchmarks
    python scripts/eval_benchmarks.py

    # Single checkpoint
    python scripts/eval_benchmarks.py --checkpoint outputs/lora_small --method lora --model small

    # Specific benchmarks only
    python scripts/eval_benchmarks.py --benchmarks gsm8k mmlu arc_easy

    # Skip slow benchmarks
    python scripts/eval_benchmarks.py --benchmarks gsm8k mmlu arc_easy hellaswag

All results logged to W&B and saved to data/benchmark_results.json
"""

import os
import re
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import wandb
import yaml
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def peak_gpu_gb():
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


# ── All checkpoints to evaluate ───────────────────────────────────────────────

ALL_CHECKPOINTS = [
    # Phase 1 SFT — 0.5B
    {"name": "full_small",        "method": "full",   "model": "small",  "phase": 1, "type": "sft"},
    {"name": "lora_small",        "method": "lora",   "model": "small",  "phase": 1, "type": "sft"},
    {"name": "qlora_small",       "method": "qlora",  "model": "small",  "phase": 1, "type": "sft"},
    # Phase 2 SFT — 3B
    {"name": "full_medium",       "method": "full",   "model": "medium", "phase": 2, "type": "sft"},
    {"name": "lora_medium",       "method": "lora",   "model": "medium", "phase": 2, "type": "sft"},
    {"name": "qlora_medium",      "method": "qlora",  "model": "medium", "phase": 2, "type": "sft"},
    # Phase 3 GRPO — 0.5B
    {"name": "grpo_full_small",   "method": "full",   "model": "small",  "phase": 3, "type": "grpo"},
    {"name": "grpo_lora_small",   "method": "lora",   "model": "small",  "phase": 3, "type": "grpo"},
    {"name": "grpo_qlora_small",  "method": "qlora",  "model": "small",  "phase": 3, "type": "grpo"},
    # Phase 3 GRPO — 3B
    {"name": "grpo_full_medium",  "method": "full",   "model": "medium", "phase": 3, "type": "grpo"},
    {"name": "grpo_lora_medium",  "method": "lora",   "model": "medium", "phase": 3, "type": "grpo"},
    {"name": "grpo_qlora_medium", "method": "qlora",  "model": "medium", "phase": 3, "type": "grpo"},
]

MODEL_NAMES = {
    "small":  "Qwen/Qwen2.5-0.5B-Instruct",
    "medium": "Qwen/Qwen2.5-3B-Instruct",
}


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_for_eval(checkpoint_path: str, method: str, base_model_name: str):
    """Load checkpoint for inference — merges adapters for LoRA/QLoRA."""
    if method == "full":
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, checkpoint_path)
        model = model.merge_and_unload()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Greedy generation."""
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


# ── GSM8K ─────────────────────────────────────────────────────────────────────

def eval_gsm8k(model, tokenizer, n_samples=150) -> dict:
    """Math reasoning — our training task."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.select(range(min(n_samples, len(ds))))

    system = ("You are a helpful math assistant. Solve step by step. "
              "Write the final answer as: #### <number>")
    correct = 0
    for ex in tqdm(ds, desc="GSM8K", leave=False):
        prompt = build_prompt(tokenizer, system, ex["question"])
        output = generate(model, tokenizer, prompt, max_new_tokens=300)
        match = re.search(r"####\s*([\-\d,\.]+)", output)
        pred = match.group(1).replace(",", "") if match else None
        gt   = re.search(r"####\s*([\-\d,\.]+)", ex["answer"])
        gt   = gt.group(1).replace(",", "") if gt else None
        if pred and gt and pred == gt:
            correct += 1

    return {"gsm8k_accuracy": correct / len(ds), "gsm8k_correct": correct, "gsm8k_total": len(ds)}


# ── MMLU ──────────────────────────────────────────────────────────────────────

MMLU_SUBJECTS = [
    "high_school_mathematics", "elementary_mathematics",
    "high_school_computer_science", "college_computer_science",
    "high_school_physics", "high_school_chemistry",
    "logical_fallacies", "formal_logic",
    "world_history", "us_history",
]

def eval_mmlu(model, tokenizer, n_per_subject=15) -> dict:
    """General knowledge — 10 subjects, tests if fine-tuning hurt broad knowledge."""
    system = ("Answer the following multiple choice question. "
              "Reply with only the letter A, B, C, or D.")
    correct = total = 0

    for subject in tqdm(MMLU_SUBJECTS, desc="MMLU subjects", leave=False):
        try:
            ds = load_dataset("cais/mmlu", subject, split="test")
            ds = ds.select(range(min(n_per_subject, len(ds))))
        except Exception:
            continue

        for ex in ds:
            choices = ex["choices"]
            choice_str = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", choices))
            user = f"Question: {ex['question']}\n\n{choice_str}"
            prompt = build_prompt(tokenizer, system, user)
            output = generate(model, tokenizer, prompt, max_new_tokens=10)
            m = re.search(r'\b([ABCD])\b', output[:150]); pred = m.group(1).upper() if m else ""
            gt   = "ABCD"[ex["answer"]]
            if pred == gt:
                correct += 1
            total += 1

    return {"mmlu_accuracy": correct / total if total else 0,
            "mmlu_correct": correct, "mmlu_total": total}


# ── HellaSwag ─────────────────────────────────────────────────────────────────

def eval_hellaswag(model, tokenizer, n_samples=100) -> dict:
    """Commonsense reasoning — multiple choice sentence completion."""
    ds = load_dataset("Rowan/hellaswag", split="validation")
    ds = ds.select(range(min(n_samples, len(ds))))

    system = ("Choose the most logical continuation. "
              "Reply with only the letter A, B, C, or D.")
    correct = 0
    for ex in tqdm(ds, desc="HellaSwag", leave=False):
        endings = ex["endings"]
        choice_str = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", endings))
        user = f"Context: {ex['ctx']}\n\nContinuation:\n{choice_str}"
        prompt = build_prompt(tokenizer, system, user)
        output = generate(model, tokenizer, prompt, max_new_tokens=10)
        m = re.search(r'\b([ABCD])\b', output[:150]); pred = m.group(1).upper() if m else ""
        gt   = "ABCD"[int(ex["label"])]
        if pred == gt:
            correct += 1

    return {"hellaswag_accuracy": correct / len(ds),
            "hellaswag_correct": correct, "hellaswag_total": len(ds)}


# ── ARC-Easy ──────────────────────────────────────────────────────────────────

def eval_arc_easy(model, tokenizer, n_samples=100) -> dict:
    """Grade school science — multiple choice."""
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    ds = ds.select(range(min(n_samples, len(ds))))

    system = ("Answer the science question. "
              "Reply with only the letter A, B, C, or D.")
    correct = 0
    for ex in tqdm(ds, desc="ARC-Easy", leave=False):
        labels  = ex["choices"]["label"]
        texts   = ex["choices"]["text"]
        choice_str = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
        user = f"Question: {ex['question']}\n\n{choice_str}"
        prompt = build_prompt(tokenizer, system, user)
        output = generate(model, tokenizer, prompt, max_new_tokens=10)
        m = re.search(r'\b([ABCD])\b', output[:150]); pred = m.group(1).upper() if m else ""
        if pred == ex["answerKey"].upper():
            correct += 1

    return {"arc_easy_accuracy": correct / len(ds),
            "arc_easy_correct": correct, "arc_easy_total": len(ds)}


# ── TruthfulQA ────────────────────────────────────────────────────────────────

def eval_truthfulqa(model, tokenizer, n_samples=100) -> dict:
    """
    Factual accuracy / hallucination test.
    Uses the multiple-choice variant — model picks most truthful answer.
    """
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    ds = ds.select(range(min(n_samples, len(ds))))

    system = ("Answer the question truthfully. "
              "Reply with only the letter of the most accurate answer.")
    correct = 0
    for ex in tqdm(ds, desc="TruthfulQA", leave=False):
        choices = ex["mc1_targets"]["choices"]
        labels  = [chr(65 + i) for i in range(len(choices))]
        choice_str = "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
        user = f"Question: {ex['question']}\n\n{choice_str}"
        prompt = build_prompt(tokenizer, system, user)
        output = generate(model, tokenizer, prompt, max_new_tokens=10)
        m = re.search(r'\b([ABCD])\b', output[:150]); pred = m.group(1).upper() if m else ""
        # Correct answer is always index 0 in mc1_targets
        gt = "A"
        if pred == gt:
            correct += 1

    return {"truthfulqa_accuracy": correct / len(ds),
            "truthfulqa_correct": correct, "truthfulqa_total": len(ds)}


# ── HumanEval (simplified) ────────────────────────────────────────────────────

def eval_humaneval(model, tokenizer, n_samples=50) -> dict:
    """
    Code generation — tests if math RL hurt coding ability.
    Simplified: checks if generated code contains correct function structure
    and key solution elements rather than executing code.
    Uses pass@1 estimation via keyword matching.
    """
    try:
        ds = load_dataset("openai/openai_humaneval", split="test")
    except Exception:
        ds = load_dataset("evalplus/humanevalplus", split="test")

    ds = ds.select(range(min(n_samples, len(ds))))

    system = ("You are an expert Python programmer. "
              "Complete the function. Write only the function body, no explanation.")
    passed = 0
    for ex in tqdm(ds, desc="HumanEval", leave=False):
        prompt_text = ex["prompt"]
        user = f"Complete this Python function:\n\n{prompt_text}"
        full_prompt = build_prompt(tokenizer, system, user)
        output = generate(model, tokenizer, full_prompt, max_new_tokens=256)

        # Heuristic check: does output contain return statement and key solution
        # This is a proxy metric — not full execution-based eval
        canonical = ex.get("canonical_solution", "")
        has_return = "return" in output
        # Check overlap with key tokens from canonical solution
        canonical_tokens = set(re.findall(r'\b\w+\b', canonical.lower()))
        output_tokens    = set(re.findall(r'\b\w+\b', output.lower()))
        overlap = len(canonical_tokens & output_tokens) / max(len(canonical_tokens), 1)
        if has_return and overlap > 0.4:
            passed += 1

    return {"humaneval_pass_at_1_proxy": passed / len(ds),
            "humaneval_passed": passed, "humaneval_total": len(ds)}


# ── Benchmark registry ────────────────────────────────────────────────────────

BENCHMARKS = {
    "gsm8k":      eval_gsm8k,
    "mmlu":       eval_mmlu,
    "hellaswag":  eval_hellaswag,
    "arc_easy":   eval_arc_easy,
    "truthfulqa": eval_truthfulqa,
    "humaneval":  eval_humaneval,
}

DEFAULT_BENCHMARKS = ["gsm8k", "mmlu", "arc_easy", "hellaswag"]


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate_checkpoint(ckpt_info: dict, benchmarks: list, config: dict,
                        output_base: str) -> dict:
    """Run all benchmarks on a single checkpoint."""
    name       = ckpt_info["name"]
    method     = ckpt_info["method"]
    model_size = ckpt_info["model"]
    ckpt_path  = os.path.join(output_base, name)
    base_model = MODEL_NAMES[model_size]

    if not os.path.isdir(ckpt_path):
        print(f"[skip] Checkpoint not found: {ckpt_path}")
        return {}

    print(f"\n{'='*60}")
    print(f"  Checkpoint: {name}")
    print(f"  Method:     {method} | Model: {model_size}")
    print(f"  Benchmarks: {benchmarks}")
    print(f"{'='*60}")

    # W&B run per checkpoint
    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        name=f"bench_{name}",
        job_type="benchmark",
        tags=[method, model_size, ckpt_info["type"], "generalization"],
        config={
            "checkpoint": name,
            "method":     method,
            "model_size": model_size,
            "phase":      ckpt_info["phase"],
            "type":       ckpt_info["type"],
            "benchmarks": benchmarks,
        },
        reinit=True,
    )

    model, tokenizer = load_model_for_eval(ckpt_path, method, base_model)
    all_results = {"checkpoint": name, "method": method, "model": model_size,
                   "phase": ckpt_info["phase"], "type": ckpt_info["type"]}

    for bench_name in benchmarks:
        if bench_name not in BENCHMARKS:
            print(f"[warn] Unknown benchmark: {bench_name}")
            continue
        print(f"\n  Running {bench_name}...")
        t0 = time.time()
        results = BENCHMARKS[bench_name](model, tokenizer)
        elapsed = time.time() - t0
        results[f"{bench_name}_time_sec"] = elapsed
        all_results.update(results)
        # Log numeric results to W&B
        wandb.log({k: v for k, v in results.items() if isinstance(v, float)})
        # Print summary
        for k, v in results.items():
            if "accuracy" in k or "pass" in k:
                print(f"    {k}: {v:.4f}")

    wandb.log({"peak_gpu_memory_gb": peak_gpu_gb()})
    wandb.finish()

    # Free GPU memory before next checkpoint
    del model
    torch.cuda.empty_cache()

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Multi-benchmark evaluation")
    parser.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS,
                        choices=list(BENCHMARKS.keys()),
                        help="Which benchmarks to run")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Single checkpoint name (e.g. lora_small). "
                             "If not set, evaluates all checkpoints.")
    parser.add_argument("--method", type=str, default=None,
                        choices=["full", "lora", "qlora"])
    parser.add_argument("--model", type=str, default=None,
                        choices=["small", "medium"])
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output-dir", default="data/benchmark_results",
                        help="Directory to save JSON results")
    args = parser.parse_args()

    config      = load_config(args.config)
    output_base = config["sft"]["output_base_dir"]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Determine which checkpoints to evaluate
    if args.checkpoint:
        # Single checkpoint mode
        ckpt_list = [c for c in ALL_CHECKPOINTS if c["name"] == args.checkpoint]
        if not ckpt_list:
            # Build ad-hoc entry
            ckpt_list = [{"name": args.checkpoint, "method": args.method or "lora",
                          "model": args.model or "small", "phase": 0, "type": "custom"}]
    else:
        ckpt_list = ALL_CHECKPOINTS

    print(f"\nEvaluating {len(ckpt_list)} checkpoint(s) on: {args.benchmarks}")
    print(f"Output dir: {args.output_dir}\n")

    all_results = []
    for ckpt_info in ckpt_list:
        results = evaluate_checkpoint(ckpt_info, args.benchmarks, config, output_base)
        if results:
            all_results.append(results)

    # Save all results to JSON
    out_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[done] Results saved → {out_path}")

    # Print comparison table
    if all_results:
        print(f"\n{'='*80}")
        print(f"{'Checkpoint':<25} " +
              " ".join(f"{b[:10]:>12}" for b in args.benchmarks))
        print("=" * 80)
        for r in all_results:
            row = f"{r['checkpoint']:<25} "
            for b in args.benchmarks:
                key = f"{b}_accuracy" if b != "humaneval" else "humaneval_pass_at_1_proxy"
                val = r.get(key, 0.0)
                row += f"{val:>12.3f} "
            print(row)
        print("=" * 80)


if __name__ == "__main__":
    main()