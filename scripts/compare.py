"""
scripts/compare.py

Pulls all experiment runs from W&B and prints a comparison table.
Run after all Phase 1/2/3 experiments are complete.

Usage:
    python scripts/compare.py
    python scripts/compare.py --phase 1
    python scripts/compare.py --phase 3
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def fetch_runs(entity: str, project: str) -> list:
    import wandb
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    return list(runs)


def print_table(runs: list, phase_filter: int = None):
    PHASE_MAP = {"sft": 1, "grpo": 3}

    rows = []
    for run in runs:
        jt    = run.job_type
        phase = run.config.get("phase", PHASE_MAP.get(jt, "?"))

        if phase_filter and phase != phase_filter:
            continue

        name   = run.name
        method = run.config.get("method",      "—")
        size   = run.config.get("model_size",  "—")
        acc    = run.summary.get("gsm8k_accuracy",     None)
        mem    = run.summary.get("peak_gpu_memory_gb", None)
        mins   = run.summary.get("train_time_min",     None)
        pct    = run.summary.get("trainable_pct",      None)
        state  = run.state

        rows.append({
            "phase": phase, "name": name, "method": method,
            "size": size,   "acc": acc,   "mem": mem,
            "mins": mins,   "pct": pct,   "state": state,
        })

    rows.sort(key=lambda r: (r["phase"], -(r["acc"] or 0)))

    def fmt(v, fmt_str, fallback="—"):
        return fmt_str.format(v) if v is not None else fallback

    header = (
        f"{'Phase':>6}  {'Run Name':<28} {'Method':<7} {'Size':<8} "
        f"{'Accuracy':>10} {'GPU (GB)':>10} {'Time (min)':>12} "
        f"{'Trainable%':>12}  {'State'}"
    )
    sep = "─" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)

    current_phase = None
    for r in rows:
        if r["phase"] != current_phase:
            if current_phase is not None:
                print()
            current_phase = r["phase"]
        print(
            f"{str(r['phase']):>6}  {r['name']:<28} {r['method']:<7} {r['size']:<8} "
            f"{fmt(r['acc'],  '{:.4f}'):>10} "
            f"{fmt(r['mem'],  '{:.1f}'):>10} "
            f"{fmt(r['mins'], '{:.1f}'):>12} "
            f"{fmt(r['pct'],  '{:.2f}%'):>12}  "
            f"{r['state']}"
        )

    print(sep)
    print(f"\nTotal runs shown: {len(rows)}")
    config = load_config()
    print(
        f"W&B: https://wandb.ai/"
        f"{config['wandb']['entity']}/{config['wandb']['project']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=None, choices=[1, 2, 3])
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Fetching runs from "
          f"{config['wandb']['entity']}/{config['wandb']['project']}...")

    try:
        runs = fetch_runs(config["wandb"]["entity"], config["wandb"]["project"])
        print_table(runs, phase_filter=args.phase)
    except Exception as e:
        print(f"[error] Could not fetch W&B runs: {e}")
        print("  → Make sure you are logged in: wandb login")
        sys.exit(1)


if __name__ == "__main__":
    main()