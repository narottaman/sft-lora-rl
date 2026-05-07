"""
src/evaluate.py

Post-training GSM8K evaluation.
  - Greedy decoding (deterministic, fair comparison across all methods)
  - Exact-match accuracy on the #### number
  - Logs per-question W&B Table + aggregate metrics

KEY FIXES:
1. Resets tokenizer.padding_side to "right" before eval — GRPO sets it to
   "left" for rollout generation which breaks single-sample greedy decoding.
2. Prints first 3 raw outputs for debugging so you can see what model generates.
3. Uses tokenizer-aware build_chat_prompt for any model size.
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


def load_for_inference(checkpoint: str, method: str, base_model_name: str):
    """Load a trained checkpoint for inference only."""
    if method == "full":
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
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
        model = PeftModel.from_pretrained(base, checkpoint)
        model = model.merge_and_unload()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 256) -> str:
    """
    Greedy generation for a single question.

    IMPORTANT: padding_side must be "right" for single-sample generation.
    GRPO training sets padding_side="left" for batch rollouts — we reset it
    here so eval works correctly regardless of what training did.
    """
    # Always use right padding for single-sample greedy decoding
    tokenizer.padding_side = "right"

    prompt = build_chat_prompt(question, tokenizer)
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


def evaluate_gsm8k(
    model,
    tokenizer,
    run_name: str,
    max_samples: int = 100,
    max_new_tokens: int = 256,
) -> dict:
    """
    Run evaluation on GSM8K test set.
    Returns metrics dict; logs to W&B if run is active.
    """
    # Reset padding side — critical after GRPO which sets it to "left"
    tokenizer.padding_side = "right"

    test_data = load_raw_test(max_samples=max_samples)
    correct = 0
    rows = []

    for i, item in enumerate(tqdm(test_data, desc=f"Eval [{run_name}]")):
        raw_output = generate_answer(model, tokenizer, item["question"], max_new_tokens)
        predicted   = extract_answer(raw_output)
        correct_flag = is_correct(raw_output, item["answer"])
        if correct_flag:
            correct += 1

        # Print first 3 outputs so you can debug what the model is generating
        if i < 3:
            print(f"\n[eval debug] Question {i+1}: {item['question'][:80]}...")
            print(f"[eval debug] Ground truth: {item['answer']}")
            print(f"[eval debug] Raw output:   {raw_output[:200]}")
            print(f"[eval debug] Extracted:    {predicted}")
            print(f"[eval debug] Correct:      {correct_flag}")

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

    print(f"\n[eval] {run_name} -> GSM8K accuracy: {accuracy:.4f} ({correct}/{len(test_data)})")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--method",      required=True, choices=["full", "lora", "qlora"])
    parser.add_argument("--base-model",  default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--run-name",    default="standalone_eval")
    args = parser.parse_args()

    model, tokenizer = load_for_inference(args.checkpoint, args.method, args.base_model)
    evaluate_gsm8k(model, tokenizer, args.run_name, args.max_samples)