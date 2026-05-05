# from-sft-to-grpo

> Reproducible comparison of full fine-tuning, LoRA, QLoRA, and GRPO reinforcement
> learning for mathematical reasoning on GSM8K.  
> Models: Qwen2.5-0.5B and Qwen2.5-3B · Tracked via W&B · Runs on ASU Sol HPC.

---

## The Question This Project Answers

> *Does the fine-tuning method (full weights vs LoRA vs QLoRA) affect how much RL can
> improve a model? And does scale (0.5B vs 3B) matter more than the training paradigm?*

Three phases, seven experiments, one clean W&B dashboard.

---

## Experiment Design

### Phase 1 — SFT Baseline (~4 GPU-hours, Qwen2.5-0.5B)

| Run | Method | Trains | Expected GPU |
|-----|--------|--------|-------------|
| `full_small`  | Full fine-tune | 100% of weights | ~4 GB |
| `lora_small`  | LoRA (r=16) | ~1.5% of weights | ~3 GB |
| `qlora_small` | QLoRA (4-bit NF4 + LoRA) | ~1.5% of weights | ~2 GB |

### Phase 2 — Scale (~8 GPU-hours, Qwen2.5-3B)

| Run | Method | Expected GPU |
|-----|--------|-------------|
| `lora_medium`  | LoRA (r=16) | ~10 GB |
| `qlora_medium` | QLoRA (4-bit NF4 + LoRA) | ~6 GB |

> Full fine-tune on 3B excluded: requires ~36GB for weights + gradients + optimizer states.
> Can be added with `--model medium --method full` if A100-80GB nodes are available.

### Phase 3 — GRPO RL (~6 GPU-hours)

Starts from the best Phase 1/2 checkpoint in each category. Reward = 1.0 if #### answer
matches ground truth, 0.0 otherwise. Uses GRPO (same algorithm as DeepSeek-R1).

| Run | SFT Init | What it tests |
|-----|----------|--------------|
| `grpo_lora_small`  | lora_small  | Does RL improve the best small adapter? |
| `grpo_lora_medium` | lora_medium | Does RL lift the larger model further? |
| `grpo_full_small`  | full_small  | RL without adapters — full weights policy gradient |

---

## Results

> *Fill in after running all experiments*

| Phase | Run | GSM8K Acc | GPU (GB) | Time (min) | Trainable% |
|-------|-----|-----------|----------|------------|------------|
| 1 | full_small    | — | — | — | 100.0% |
| 1 | lora_small    | — | — | — | ~1.5%  |
| 1 | qlora_small   | — | — | — | ~1.5%  |
| 2 | lora_medium   | — | — | — | ~1.2%  |
| 2 | qlora_medium  | — | — | — | ~1.2%  |
| 3 | grpo_lora_small  | — | — | — | ~1.5% |
| 3 | grpo_lora_medium | — | — | — | ~1.2% |
| 3 | grpo_full_small  | — | — | — | 100.0% |

W&B Dashboard: *(link after first run)*

**Key findings** *(fill after experiments)*:
- Phase 1: QLoRA achieves X% of full fine-tune accuracy at Y% of GPU memory
- Phase 2: Scaling from 0.5B to 3B gives +Z% accuracy on GSM8K
- Phase 3: GRPO adds +W% accuracy on top of SFT baseline

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/from-sft-to-grpo
cd from-sft-to-grpo

# 2. Install
pip install -r requirements.txt

# 3. Auth
wandb login
huggingface-cli login   # needed for Qwen model weights

# 4. Smoke test — runs lora_small with 100 samples, 1 epoch
python scripts/train_sft.py --model small --method lora --smoke-test

# 5. Single experiment
python scripts/train_sft.py --model small --method lora
python scripts/train_sft.py --model small --method full
python scripts/train_sft.py --model small --method qlora

# 6. After Phase 1 + 2, run GRPO
python scripts/train_grpo.py \
    --sft-checkpoint ./outputs/lora_small \
    --method lora --model small

# 7. Compare all results
python scripts/compare.py
```

---

## Running on ASU Sol HPC

```bash
# 1. Copy project to Sol scratch
scp -r . ngangada@sol.asu.edu:/scratch/ngangada/portfolio/sft-lora-rl/

# 2. First-time setup (run once)
ssh ngangada@sol.asu.edu
bash /scratch/ngangada/portfolio/sft-lora-rl/sol/setup.sh

# 3. Phase 1 — 3 experiments in parallel (array job)
sbatch sol/phase1_sft.slurm

# 4. Phase 2 — after Phase 1 finishes
sbatch sol/phase2_scale.slurm

# 5. Phase 3 — after Phase 1 + 2 finish
sbatch sol/phase3_grpo.slurm

# Monitor
squeue -u $USER
tail -f /scratch/ngangada/portfolio/sft-lora-rl/logs/phase1_<jobid>_0.out
```

---

## Project Structure

```
from-sft-to-grpo/
├── src/
│   ├── data_utils.py      # GSM8K loading, prompt formatting for SFT + GRPO
│   ├── model_utils.py     # Model loader factory: full / LoRA / QLoRA
│   └── evaluate.py        # Greedy inference + exact-match accuracy + W&B Table
├── scripts/
│   ├── train_sft.py       # Phase 1 + 2: SFT / LoRA / QLoRA training
│   ├── train_grpo.py      # Phase 3: GRPO RL from SFT checkpoint
│   └── compare.py         # Pull W&B runs → comparison table
├── configs/
│   └── config.yaml        # All hyperparameters (models, SFT, LoRA, QLoRA, GRPO)
├── sol/
│   ├── setup.sh           # First-time Sol environment setup
│   ├── phase1_sft.slurm   # SLURM array: 3 Phase 1 experiments
│   ├── phase2_scale.slurm # SLURM array: 2 Phase 2 experiments
│   └── phase3_grpo.slurm  # SLURM array: 3 Phase 3 GRPO experiments
├── outputs/               # Saved checkpoints (gitignored)
├── logs/                  # SLURM logs (gitignored)
└── requirements.txt
```

---

## Why Each Method

**Full Fine-Tuning** — All weights updated. Highest accuracy ceiling, highest memory cost.
Viable for 0.5B on a single GPU; impractical for 7B+ without FSDP/DeepSpeed.

**LoRA** — Adds rank-16 adapters to all attention + MLP projections (~1.5% of params).
Near full fine-tune accuracy at a fraction of the memory. The practical choice for most
production fine-tuning.

**QLoRA** — Quantizes the frozen base to 4-bit NF4 (bitsandbytes), then applies LoRA.
Enables fine-tuning models that wouldn't otherwise fit in a single GPU's memory.
Small accuracy tradeoff vs LoRA due to quantization.

**GRPO** — Group Relative Policy Optimization (same as DeepSeek-R1). Generates G
completions per prompt, rewards correct #### answers with 1.0, normalizes within the group,
and uses KL penalty to keep the policy close to the SFT initialization. Binary reward from
math benchmarks is ideal — no reward hacking possible.

---

## W&B Metrics Logged

| Metric | Description |
|--------|-------------|
| `train/loss` | Training loss per step |
| `eval/loss` | Validation loss per step |
| `gsm8k_accuracy` | Exact-match on #### number, test set |
| `trainable_params` / `trainable_pct` | Parameter efficiency |
| `peak_gpu_memory_gb` | Peak VRAM usage |
| `train_time_min` | Wall-clock training time |
| `eval_table` | Per-question predictions table |
| `train/reward` *(Phase 3)* | Mean GRPO reward per step |
| `train/kl` *(Phase 3)* | KL divergence from SFT policy |

---

## Tech Stack

`transformers` · `peft` · `trl` · `bitsandbytes` · `accelerate`  
`datasets` · `wandb` · `torch` · `Qwen2.5` · `GSM8K`

---

## License

MIT © 2026 narottaman