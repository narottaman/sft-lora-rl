"""
src/data_utils.py

GSM8K dataset utilities used across all three phases.

KEY FIXES:
1. apply_chat_template() for all model sizes — not hardcoded <|im_start|>.
2. clean_gsm8k_answer() strips <<expr=result>> calculator annotations from
   training data. These annotations are GSM8K dataset artifacts that waste
   model capacity — especially harmful for 0.5B models which degrade from
   45.5% base accuracy to 29-31% when trained on the raw annotated format.
   Training on clean reasoning text recovers this loss.
3. Flexible extract_answer() works for both fine-tuned (####) and base models
   ("The answer is X") — same extractor everywhere for fair comparison.
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


# ── Answer cleaning ───────────────────────────────────────────────────────────

def clean_gsm8k_answer(answer: str) -> str:
    """
    Strip <<expr=result>> calculator annotations from GSM8K answers.
    These are dataset artifacts, not natural reasoning.

    Example:
      Raw:     '48/2 = <<48/2=24>>24 clips ... #### 72'
      Cleaned: '48/2 = 24 clips ... #### 72'

    Why this matters for 0.5B models:
      The base Qwen2.5-0.5B-Instruct scores 45.5% on GSM8K zero-shot.
      After SFT on raw annotated data it drops to 29-31% because the model
      wastes its limited capacity learning the annotation syntax instead of
      mathematical reasoning. Stripping annotations lets the model focus on
      learning correct step-by-step reasoning.
    """
    cleaned = re.sub(r'<<[^>]+>>', '', answer)
    cleaned = re.sub(r'  +', ' ', cleaned)
    return cleaned.strip()


# ── Answer extraction ─────────────────────────────────────────────────────────

def extract_answer(text: str) -> Optional[str]:
    """
    Flexible answer extraction — works for both fine-tuned and base models.

    Priority:
      1. #### N       — fine-tuned model format
      2. makes/earns N — natural language
      3. = N dollars  — end of calculation
      4. last number  — fallback
    """
    if not text:
        return None

    m = re.search(r'####\s*([\-\d,\.]+)', text)
    if m:
        return m.group(1).replace(",", "").rstrip(".").strip()

    m = re.search(
        r'(?:makes|earns|answer is|answer:|total is)\s+\$?\s*([\-\d,\.]+)',
        text, re.IGNORECASE
    )
    if m:
        return m.group(1).replace(",", "").rstrip(".").strip()

    m = re.search(
        r'=\s*\$?\s*([\-\d,\.]+)\s*(?:dollars?|cents?|each)?\s*[.\n]',
        text, re.IGNORECASE
    )
    if m:
        return m.group(1).replace(",", "").rstrip(".").strip()

    numbers = re.findall(r'(?<!\w)([\-]?\d[\d,]*\.?\d*)(?!\w)', text)
    if numbers:
        return numbers[-1].replace(",", "").rstrip(".").strip()

    return None


def is_correct(prediction: str, ground_truth: str) -> bool:
    pred = extract_answer(prediction)
    return pred is not None and pred == ground_truth


# ── Prompt formatting ─────────────────────────────────────────────────────────

def build_chat_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def format_for_sft(example: dict, tokenizer) -> dict:
    """
    Format for SFT training. Cleans <<expr=result>> annotations from answer
    before training so the model learns clean reasoning, not annotation syntax.
    """
    clean_answer = clean_gsm8k_answer(example["answer"])
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": example["question"]},
        {"role": "assistant", "content": clean_answer},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    return {"text": text}


def format_for_grpo(example: dict, tokenizer) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": example["question"]},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    return {
        "prompt":       prompt,
        "ground_truth": extract_answer(example["answer"]) or "",
    }


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_sft_datasets(tokenizer, max_train_samples=2000, max_eval_samples=100):
    ds = load_dataset("openai/gsm8k", "main")
    train_ds = ds["train"]
    eval_ds  = ds["test"]

    if max_train_samples:
        train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))
    if max_eval_samples:
        eval_ds  = eval_ds.select(range(min(max_eval_samples, len(eval_ds))))

    train_ds = train_ds.map(lambda ex: format_for_sft(ex, tokenizer),
                            remove_columns=train_ds.column_names)
    eval_ds  = eval_ds.map(lambda ex: format_for_sft(ex, tokenizer),
                           remove_columns=eval_ds.column_names)

    print(f"[data] SFT — train: {len(train_ds):,} | eval: {len(eval_ds):,}")
    print(f"[data] Note: <<expr=result>> annotations stripped from training answers")
    return train_ds, eval_ds


def load_grpo_datasets(tokenizer, max_train_samples=2000, max_eval_samples=100):
    ds = load_dataset("openai/gsm8k", "main")
    train_ds = ds["train"]
    eval_ds  = ds["test"]

    if max_train_samples:
        train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))
    if max_eval_samples:
        eval_ds  = eval_ds.select(range(min(max_eval_samples, len(eval_ds))))

    train_ds = train_ds.map(lambda ex: format_for_grpo(ex, tokenizer),
                            remove_columns=train_ds.column_names)
    eval_ds  = eval_ds.map(lambda ex: format_for_grpo(ex, tokenizer),
                           remove_columns=eval_ds.column_names)

    print(f"[data] GRPO — train: {len(train_ds):,} | eval: {len(eval_ds):,}")
    return train_ds, eval_ds


def load_raw_test(max_samples=300):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return [
        {"question": ex["question"], "answer": extract_answer(ex["answer"])}
        for ex in ds
    ]