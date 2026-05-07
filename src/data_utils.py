"""
src/data_utils.py

GSM8K dataset utilities used across all three phases.

GSM8K record format (HuggingFace openai/gsm8k):
  question : "Natalia sold clips to 48 of her friends in April..."
  answer   : "Natalia sold 48/2 = <<48/2=24>>24 clips...\\n#### 72"

Ground truth is the integer after "####".

KEY FIX: Uses tokenizer.apply_chat_template() instead of hardcoded
<|im_start|> strings. The hardcoded format worked for 0.5B but caused
loss=10.86 on 3B because the tokenizer wasn't properly aligning tokens.
apply_chat_template works correctly for any Qwen model size.
"""

import re
from typing import Optional
from datasets import load_dataset


SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Solve the problem step by step. "
    "At the end of your solution, write the final answer as a number on its own line "
    "in the format: #### <number>"
)


# ── Answer extraction ─────────────────────────────────────────────────────────

def extract_answer(text: str) -> Optional[str]:
    """
    Pull the integer/decimal after '####' from model output or ground truth.
    Strips commas from numbers like 1,234 → 1234.
    Returns None if no match found.
    """
    match = re.search(r"####\s*([\-\d,\.]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    return None


def is_correct(prediction: str, ground_truth: str) -> bool:
    """Exact-match comparison after normalization."""
    pred = extract_answer(prediction)
    return pred is not None and pred == ground_truth


# ── Prompt formatting using tokenizer.apply_chat_template ────────────────────

def build_chat_prompt(question: str, tokenizer) -> str:
    """
    Build inference prompt using tokenizer.apply_chat_template.
    Works correctly for any Qwen model size (0.5B, 3B, 7B, etc).
    add_generation_prompt=True appends the assistant turn opener.
    """
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def format_for_sft(example: dict, tokenizer) -> dict:
    """
    Maps a raw GSM8K example to the 'text' field SFTTrainer expects.
    Uses apply_chat_template so the format matches the model's expected
    input regardless of model size.
    Full conversation: system + user + assistant (with answer).
    """
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def format_for_grpo(example: dict, tokenizer) -> dict:
    """
    Maps a raw GSM8K example to fields GRPOTrainer expects.
    Uses apply_chat_template for the prompt (no assistant turn).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": example["question"]},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return {
        "prompt":       prompt,
        "ground_truth": extract_answer(example["answer"]) or "",
    }


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_sft_datasets(
    tokenizer,
    max_train_samples: Optional[int] = 2000,
    max_eval_samples:  Optional[int] = 100,
):
    """
    Returns (train_dataset, eval_dataset) formatted for SFTTrainer.
    Tokenizer is passed in so apply_chat_template uses the correct format.
    """
    ds = load_dataset("openai/gsm8k", "main")
    train_ds = ds["train"]
    eval_ds  = ds["test"]

    if max_train_samples:
        train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))
    if max_eval_samples:
        eval_ds  = eval_ds.select(range(min(max_eval_samples,  len(eval_ds))))

    train_ds = train_ds.map(
        lambda ex: format_for_sft(ex, tokenizer),
        remove_columns=train_ds.column_names,
    )
    eval_ds = eval_ds.map(
        lambda ex: format_for_sft(ex, tokenizer),
        remove_columns=eval_ds.column_names,
    )

    print(f"[data] SFT — train: {len(train_ds):,} | eval: {len(eval_ds):,}")
    return train_ds, eval_ds


def load_grpo_datasets(
    tokenizer,
    max_train_samples: Optional[int] = 2000,
    max_eval_samples:  Optional[int] = 100,
):
    """
    Returns (train_dataset, eval_dataset) formatted for GRPOTrainer.
    """
    ds = load_dataset("openai/gsm8k", "main")
    train_ds = ds["train"]
    eval_ds  = ds["test"]

    if max_train_samples:
        train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))
    if max_eval_samples:
        eval_ds  = eval_ds.select(range(min(max_eval_samples,  len(eval_ds))))

    train_ds = train_ds.map(
        lambda ex: format_for_grpo(ex, tokenizer),
        remove_columns=train_ds.column_names,
    )
    eval_ds = eval_ds.map(
        lambda ex: format_for_grpo(ex, tokenizer),
        remove_columns=eval_ds.column_names,
    )

    print(f"[data] GRPO — train: {len(train_ds):,} | eval: {len(eval_ds):,}")
    return train_ds, eval_ds


def load_raw_test(max_samples: Optional[int] = 100) -> list[dict]:
    """
    Returns raw (question, answer) dicts for standalone evaluation.
    'answer' is the extracted #### number, not the full chain-of-thought.
    """
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return [
        {"question": ex["question"], "answer": extract_answer(ex["answer"])}
        for ex in ds
    ]