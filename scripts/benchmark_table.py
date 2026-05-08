"""
scripts/benchmark_table.py

Reads data/benchmark_results/*.json and prints a formatted comparison table.
Run after eval_benchmarks.slurm completes.

Usage:
    python scripts/benchmark_table.py
    python scripts/benchmark_table.py --results-dir data/benchmark_results
"""

import os
import json
import glob
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="data/benchmark_results")
    args = parser.parse_args()

    # Load all result files
    all_results = []
    pattern = os.path.join(args.results_dir, "*.json")
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            all_results.extend(data)
        elif isinstance(data, dict):
            all_results.append(data)

    if not all_results:
        print(f"No results found in {args.results_dir}")
        return

    # Define display order
    ORDER = [
        "full_small", "lora_small", "qlora_small",
        "full_medium", "lora_medium", "qlora_medium",
        "grpo_full_small", "grpo_lora_small", "grpo_qlora_small",
        "grpo_full_medium", "grpo_lora_medium", "grpo_qlora_medium",
    ]

    # Sort results by order
    result_map = {r["checkpoint"]: r for r in all_results}
    sorted_results = [result_map[n] for n in ORDER if n in result_map]

    benchmarks = ["gsm8k", "mmlu", "arc_easy", "hellaswag", "truthfulqa", "humaneval"]
    bench_keys = {
        "gsm8k":      "gsm8k_accuracy",
        "mmlu":       "mmlu_accuracy",
        "arc_easy":   "arc_easy_accuracy",
        "hellaswag":  "hellaswag_accuracy",
        "truthfulqa": "truthfulqa_accuracy",
        "humaneval":  "humaneval_pass_at_1_proxy",
    }

    # Only show benchmarks that have data
    available = [b for b in benchmarks
                 if any(bench_keys[b] in r for r in sorted_results)]

    # Print table
    header = f"{'Checkpoint':<25} {'Type':<5} {'Model':<7} "
    header += " ".join(f"{b[:9]:>10}" for b in available)
    sep = "─" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)

    prev_phase = None
    for r in sorted_results:
        phase = r.get("phase", "?")
        if phase != prev_phase:
            if prev_phase is not None:
                print()
            prev_phase = phase

        row = f"{r['checkpoint']:<25} {r.get('type','?'):<5} {r.get('model','?'):<7} "
        for b in available:
            val = r.get(bench_keys[b], None)
            row += f"{val*100:>9.1f}% " if val is not None else f"{'—':>10} "
        print(row)

    print(sep)
    print(f"\nTotal checkpoints: {len(sorted_results)}")
    print(f"Benchmarks: {available}")

    # Print key insight
    if "gsm8k" in available and "mmlu" in available:
        print("\n── Key Generalization Insight ──")
        sft_full_gsm  = result_map.get("full_medium",  {}).get("gsm8k_accuracy", 0)
        grpo_full_gsm = result_map.get("grpo_full_medium", {}).get("gsm8k_accuracy", 0)
        sft_full_mmlu  = result_map.get("full_medium",  {}).get("mmlu_accuracy", 0)
        grpo_full_mmlu = result_map.get("grpo_full_medium", {}).get("mmlu_accuracy", 0)

        sft_lora_gsm  = result_map.get("lora_medium",  {}).get("gsm8k_accuracy", 0)
        grpo_lora_gsm = result_map.get("grpo_lora_medium", {}).get("gsm8k_accuracy", 0)
        sft_lora_mmlu  = result_map.get("lora_medium",  {}).get("mmlu_accuracy", 0)
        grpo_lora_mmlu = result_map.get("grpo_lora_medium", {}).get("mmlu_accuracy", 0)

        print(f"  Full SFT→GRPO:  GSM8K {sft_full_gsm*100:.1f}%→{grpo_full_gsm*100:.1f}% "
              f"(+{(grpo_full_gsm-sft_full_gsm)*100:.1f}%)  |  "
              f"MMLU {sft_full_mmlu*100:.1f}%→{grpo_full_mmlu*100:.1f}% "
              f"({(grpo_full_mmlu-sft_full_mmlu)*100:+.1f}%)")
        print(f"  LoRA SFT→GRPO:  GSM8K {sft_lora_gsm*100:.1f}%→{grpo_lora_gsm*100:.1f}% "
              f"(+{(grpo_lora_gsm-sft_lora_gsm)*100:.1f}%)  |  "
              f"MMLU {sft_lora_mmlu*100:.1f}%→{grpo_lora_mmlu*100:.1f}% "
              f"({(grpo_lora_mmlu-sft_lora_mmlu)*100:+.1f}%)")
        print()
        print("  If full GRPO shows larger MMLU drop than LoRA GRPO,")
        print("  that confirms LoRA preserves general knowledge better under RL.")


if __name__ == "__main__":
    main()


def print_with_base(results_dir="data/benchmark_results"):
    """
    Extended table that includes base model baselines at the top.
    Run after both eval_base_models.slurm and eval_benchmarks.slurm complete.
    """
    # Load base model results
    base_path = os.path.join(results_dir, "base_models.json")
    bench_path = os.path.join(results_dir, "benchmark_results.json")

    all_results = []
    for path in [base_path, bench_path]:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                all_results.extend(data)
            else:
                all_results.append(data)

    if not all_results:
        print("No results found. Run eval_base_models.slurm and eval_benchmarks.slurm first.")
        return

    ORDER = [
        "base_small", "base_medium",                          # baselines
        "full_small", "lora_small", "qlora_small",             # Phase 1 SFT
        "full_medium", "lora_medium", "qlora_medium",          # Phase 2 SFT
        "grpo_full_small", "grpo_lora_small", "grpo_qlora_small",    # Phase 3
        "grpo_full_medium", "grpo_lora_medium", "grpo_qlora_medium",
    ]

    result_map = {r["checkpoint"]: r for r in all_results}
    benchmarks = ["gsm8k", "mmlu", "arc_easy", "hellaswag", "truthfulqa"]
    bench_keys = {
        "gsm8k":      "gsm8k_accuracy",
        "mmlu":       "mmlu_accuracy",
        "arc_easy":   "arc_easy_accuracy",
        "hellaswag":  "hellaswag_accuracy",
        "truthfulqa": "truthfulqa_accuracy",
    }
    available = [b for b in benchmarks
                 if any(bench_keys[b] in r for r in all_results)]

    header = f"{'Checkpoint':<25} {'Type':<6} "
    header += " ".join(f"{b[:9]:>10}" for b in available)
    sep = "─" * len(header)

    print("\n" + sep)
    print("Full Benchmark Comparison (base → SFT → GRPO)")
    print(sep)
    print(header)
    print(sep)

    sections = [
        ("Base (zero-shot)", ["base_small", "base_medium"]),
        ("Phase 1 SFT (0.5B)", ["full_small", "lora_small", "qlora_small"]),
        ("Phase 2 SFT (3B)", ["full_medium", "lora_medium", "qlora_medium"]),
        ("Phase 3 GRPO (0.5B)", ["grpo_full_small", "grpo_lora_small", "grpo_qlora_small"]),
        ("Phase 3 GRPO (3B)", ["grpo_full_medium", "grpo_lora_medium", "grpo_qlora_medium"]),
    ]

    for section_name, names in sections:
        print(f"\n  {section_name}")
        for name in names:
            if name not in result_map:
                continue
            r = result_map[name]
            row = f"  {r['checkpoint']:<23} {r.get('type','?'):<6} "
            for b in available:
                val = r.get(bench_keys[b], None)
                row += f"{val*100:>9.1f}% " if val is not None else f"{'—':>10} "
            print(row)

    print("\n" + sep)


if __name__ == "__main__":
    import sys
    if "--with-base" in sys.argv:
        print_with_base()
    else:
        main()
