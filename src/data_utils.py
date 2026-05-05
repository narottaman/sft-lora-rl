"""
src/data_utils.py

GSM8K dataset utilities used across all three phases.

GSM8K record format (HuggingFace openai/gsm8k):
  question : "Natalia sold clips to 48 of her friends in April..."
  answer   : "Natalia sold 48/2 = <<48/2=24>>24 clips...\\n#### 72"

Ground truth is the integer after "####".
We use Qwen2.5-Instruct chat format for all prompts.
"""

import re
from typing import Optional
from datasets import load_dataset


# ── Prompt config ─────────────────────────────────────────────────────────────

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


# ── Prompt formatting ─────────────────────────────────────────────────────────

def build_chat_prompt(question: str) -> str:
    """
    Qwen2.5-Instruct chat format.
    Used for inference; SFTTrainer uses apply_chat_template internally.
    """
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def format_for_sft(example: dict) -> dict:
    """
    Maps a raw GSM8K example to the 'text' field SFTTrainer expects.
    Full prompt + answer so the model learns both the reasoning and #### format.
    """
    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{example['question']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['answer']}<|im_end|>"
    )
    return {"text": text}


def format_for_grpo(example: dict) -> dict:
    """
    Maps a raw GSM8K example to fields GRPOTrainer expects:
      - 'prompt'       : the input (system + user turn)
      - 'ground_truth' : extracted #### number for reward function
    GRPOTrainer generates completions itself; we only supply the prompt.
    """
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{example['question']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return {
        "prompt": prompt,
        "ground_truth": extract_answer(example["answer"]) or "",
    }


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_sft_datasets(
    max_train_samples: Optional[int] = 2000,
    max_eval_samples:  Optional[int] = 300,
):
    """
    Returns (train_dataset, eval_dataset) formatted for SFTTrainer.
    Each example has a single 'text' field.
    """
    ds = load_dataset("openai/gsm8k", "main")
    train_ds = ds["train"]
    eval_ds  = ds["test"]

    if max_train_samples:
        train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))
    if max_eval_samples:
        eval_ds  = eval_ds.select(range(min(max_eval_samples,  len(eval_ds))))

    train_ds = train_ds.map(format_for_sft,  remove_columns=train_ds.column_names)
    eval_ds  = eval_ds.map(format_for_sft,   remove_columns=eval_ds.column_names)

    print(f"[data] SFT — train: {len(train_ds):,} | eval: {len(eval_ds):,}")
    return train_ds, eval_ds


def load_grpo_datasets(
    max_train_samples: Optional[int] = 2000,
    max_eval_samples:  Optional[int] = 300,
):
    """
    Returns (train_dataset, eval_dataset) formatted for GRPOTrainer.
    Each example has 'prompt' and 'ground_truth' fields.
    """
    ds = load_dataset("openai/gsm8k", "main")
    train_ds = ds["train"]
    eval_ds  = ds["test"]

    if max_train_samples:
        train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))
    if max_eval_samples:
        eval_ds  = eval_ds.select(range(min(max_eval_samples,  len(eval_ds))))

    train_ds = train_ds.map(format_for_grpo, remove_columns=train_ds.column_names)
    eval_ds  = eval_ds.map(format_for_grpo,  remove_columns=eval_ds.column_names)

    print(f"[data] GRPO — train: {len(train_ds):,} | eval: {len(eval_ds):,}")
    return train_ds, eval_ds


def load_raw_test(max_samples: Optional[int] = 300) -> list[dict]:
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