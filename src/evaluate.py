"""
src/evaluate.py

Post-training GSM8K evaluation.
  - Greedy decoding (deterministic, fair comparison across all methods)
  - Exact-match accuracy on the #### number
  - Logs per-question W&B Table + aggregate metrics

Can be called from train scripts or run standalone:
    python src/evaluate.py \
        --checkpoint /scratch/ngangada/portfolio/sft-lora-rl/outputs/lora_small \
        --method lora \
        --base-model Qwen/Qwen2.5-0.5B-Instruct
"""

import os
import sys
import argparse
import torch
import wandb
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_utils import load_raw_test, build_chat_prompt, extract_answer, is_correct


# ── Inference ─────────────────────────────────────────────────────────────────

def load_for_inference(checkpoint: str, method: str, base_model_name: str):
    """
    Load a trained checkpoint for inference only.
    LoRA/QLoRA: loads base model, applies adapter, merges weights.
    Full: loads checkpoint directly.
    """
    if method == "full":
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # Load base model, then overlay the saved LoRA adapter
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, checkpoint)
        model = model.merge_and_unload()   # fuse adapter into weights

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 256) -> str:
    """Greedy generation for a single question — no sampling."""
    prompt = build_chat_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Strip the prompt tokens, decode only generated part
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate_gsm8k(
    model,
    tokenizer,
    run_name: str,
    max_samples: int = 300,
    max_new_tokens: int = 256,
) -> dict:
    """
    Run evaluation on GSM8K test set.
    Returns metrics dict; logs to W&B if run is active.
    """
    test_data = load_raw_test(max_samples=max_samples)
    correct = 0
    rows = []

    for item in tqdm(test_data, desc=f"Eval [{run_name}]"):
        raw_output = generate_answer(model, tokenizer, item["question"], max_new_tokens)
        predicted  = extract_answer(raw_output)
        correct_flag = is_correct(raw_output, item["answer"])
        if correct_flag:
            correct += 1

        rows.append({
            "question":     item["question"][:150],
            "ground_truth": item["answer"],
            "predicted":    predicted or "N/A",
            "raw_output":   raw_output[:400],
            "correct":      correct_flag,
        })

    accuracy = correct / len(test_data) if test_data else 0.0
    metrics = {
        "gsm8k_accuracy":  accuracy,
        "correct_count":   correct,
        "total_evaluated": len(test_data),
    }

    print(f"\n[eval] {run_name} → GSM8K accuracy: {accuracy:.4f} ({correct}/{len(test_data)})")

    if wandb.run:
        wandb.log(metrics)
        table = wandb.Table(
            columns=["question", "ground_truth", "predicted", "raw_output", "correct"]
        )
        for r in rows:
            table.add_data(
                r["question"], r["ground_truth"],
                r["predicted"], r["raw_output"], r["correct"]
            )
        wandb.log({"eval_table": table})

    return metrics


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--method",      required=True, choices=["full", "lora", "qlora"])
    parser.add_argument("--base-model",  default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--run-name",    default="standalone_eval")
    args = parser.parse_args()

    model, tokenizer = load_for_inference(args.checkpoint, args.method, args.base_model)
    evaluate_gsm8k(model, tokenizer, args.run_name, args.max_samples)