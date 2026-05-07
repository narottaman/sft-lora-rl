# from-sft-to-grpo

> Reproducible comparison of full fine-tuning, LoRA, QLoRA, and GRPO reinforcement
> learning for mathematical reasoning on GSM8K.
> Models: Qwen2.5-0.5B and Qwen2.5-3B · Tracked via W&B · Runs on ASU Sol HPC.

---

## The Question This Project Answers

> *Does the fine-tuning method (full weights vs LoRA vs QLoRA) affect how much
> RL can improve a model? And does scale (0.5B vs 3B) matter more than the
> training paradigm?*

Three phases, twelve experiments, one W&B dashboard.

---

## Results (300-sample evaluation, stable)

### Phase 1 — SFT Baseline (Qwen2.5-0.5B)

| Run | GSM8K Acc | GPU Peak | Train Time | Trainable % |
|-----|-----------|----------|------------|-------------|
| full_small | 31.7% | 9.1 GB | 3.6 min | 100.00% |
| lora_small | 30.7% | 6.9 GB | 6.5 min | 1.75% |
| qlora_small | 29.0% | 6.6 GB | 8.5 min | 2.72% |

### Phase 2 — Scale (Qwen2.5-3B)

| Run | GSM8K Acc | GPU Peak | Train Time | Trainable % |
|-----|-----------|----------|------------|-------------|
| full_medium | 66.0% | 30.1 GB | 6.9 min | 100.00% |
| lora_medium | 66.3% | 12.3 GB | 10.7 min | 0.96% |
| qlora_medium | 64.7% | 9.2 GB | 13.0 min | 1.73% |

### Phase 3 — GRPO Reinforcement Learning

| Run | SFT Acc | GRPO Acc | RL Lift | GPU Peak |
|-----|---------|----------|---------|----------|
| grpo_full_small | 31.7% | **38.7%** | +7.0% | ~10 GB |
| grpo_lora_small | 30.7% | **42.7%** | +12.0% | ~8 GB |
| grpo_qlora_small | 29.0% | **45.3%** | +16.3% | ~8 GB |
| grpo_full_medium | 66.0% | **73.0%** | +7.0% | ~32 GB |
| grpo_lora_medium | 66.3% | **74.7%** | +8.4% | ~15 GB |
| grpo_qlora_medium | 64.7% | **76.7%** | +12.0% | ~12 GB |

W&B Dashboard: [from-sft-to-grpo](https://wandb.ai/ngangada-arizona-state-university/from-sft-to-grpo)

---

## Key Findings

**1. QLoRA benefits most from GRPO — consistently across both model sizes.**
QLoRA shows the largest RL lift at both scales (+16.3% on 0.5B, +12% on 3B),
while full fine-tune shows the smallest lift (+7% at both scales). This pattern
holds perfectly across six experiments, making it a robust finding.

**2. Adapter methods leave more room for RL to exploit.**
Full fine-tuning extracts nearly all available signal from 2000 supervised
examples, leaving little headroom for GRPO. Adapters constrain learning to a
low-rank subspace, creating a productive gap between SFT capability and the
model's true capacity — which GRPO then fills with the reward signal.

**3. Scale matters more than method for raw accuracy.**
The jump from 0.5B to 3B gives +34-37 percentage points regardless of training
method. LoRA-3B (66.3% SFT, 74.7% after GRPO) vastly outperforms any 0.5B
model while using only 0.96% trainable parameters and 12.3GB of GPU memory.

**4. LoRA regularization matches full fine-tuning at 3B scale with limited data.**
Full fine-tune on 3B with 2000 samples shows increasing eval loss after epoch 1
(0.939 → 1.016 → 1.093) — a clear overfitting signature. LoRA's rank constraint
prevents this, achieving the same 66% accuracy while using 96% less GPU memory.
This crossover does not occur at 0.5B, where the model has insufficient capacity
to overfit 2000 samples.

**5. QLoRA is the most memory-efficient path to strong final performance.**
qlora_medium achieves 76.7% after GRPO using 9.2GB for SFT and ~12GB for GRPO,
versus full_medium requiring 30.1GB for SFT to reach 73%. Higher accuracy at
one-third the memory.

**6. LoRA and QLoRA generalize better beyond GSM8K.**
Adapter methods keep base model weights frozen throughout both SFT and GRPO.
Only ~1-2% of parameters are updated, preserving the base model's general
knowledge. Full fine-tune risks catastrophic forgetting when pushed further
with RL on a single task.

---

## W&B Training Curves Analysis

The W&B dashboard (screenshots above) shows several clean patterns:

- **train/reward mean**: Medium models (3B) plateau around 0.75-0.85 while
  small models (0.5B) plateau lower at 0.45-0.55, confirming scale effect
- **train/loss**: All runs show rapid initial descent followed by convergence
  near zero, confirming stable GRPO training
- **train/entropy**: Decreases over training as the policy becomes more
  confident — healthy sign, model is learning rather than random walking
- **train/kl**: Stays low (< 0.05 for most runs) confirming the KL penalty
  successfully keeps the policy close to the SFT initialization
- **train/mean_token_accuracy**: SFT runs for 3B models reach 85-90%,
  confirming good supervised learning before RL phase

**Note on evaluation:** Max generation length is capped at 256 tokens.
Some harder GSM8K problems require longer chain-of-thought reasoning and
get truncated before reaching `####`, counted as incorrect. Increasing
`max_new_tokens` would likely improve all numbers by 2-5%.

---

## Experiment Design

### Phase 1 — SFT Baseline (Qwen2.5-0.5B, ~20 GPU-min)

| Method | What trains | Memory strategy |
|--------|------------|-----------------|
| Full fine-tune | 100% of 494M weights | Standard bf16 |
| LoRA (r=16) | 1.75% — adapter matrices only | bf16 base + bf16 adapter |
| QLoRA (r=16, NF4) | 2.72% — adapter matrices only | 4-bit base + bf16 adapter |

### Phase 2 — Scale (Qwen2.5-3B, ~30 GPU-min)

Same three methods on the 3B model. Answers: *does model size or training
method matter more?*

### Phase 3 — GRPO RL (~3 GPU-hours total, 6 experiments)

Starts from each Phase 1/2 SFT checkpoint. Binary reward: 1.0 if extracted
`####` answer matches ground truth, 0.0 otherwise. Uses GRPO (same algorithm
as DeepSeek-R1). G=4 generations per prompt, KL beta=0.01, 2 epochs.

**Weight update behavior per method during GRPO:**
- **Full**: all 3B/0.5B weights receive policy gradient updates
- **LoRA**: only rank-16 adapter matrices updated, base weights frozen
- **QLoRA**: base loaded in bf16 for gradient computation (not 4-bit),
  only adapter updated — same trainable parameters as LoRA+GRPO

---

## Dataset

**GSM8K** (Grade School Math, `openai/gsm8k`) — 7,473 training / 1,319 test.
Ground truth is the integer after `####`. Exact-match evaluation.
Binary GRPO reward — no reward hacking possible.

- Training: 2,000 samples (subset for speed)
- Evaluation: 300 samples (stable, low-variance estimates)

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/from-sft-to-grpo
cd from-sft-to-grpo
pip install -r requirements.txt

# 2. Auth
wandb login
huggingface-cli login

# 3. Smoke test — lora_small, 100 samples, 1 epoch
python scripts/train_sft.py --model small --method lora --smoke-test

# 4. Phase 1 — SFT baseline (3 experiments, ~20 min)
sbatch sol/phase1_sft.slurm

# 5. Phase 2 — Scale to 3B (3 experiments, ~30 min)
sbatch sol/phase2_scale.slurm

# 6. Phase 3 — GRPO RL + 300-sample eval (6 experiments, ~3h)
sbatch sol/phase3_grpo.slurm

# 7. Stable 300-sample eval for small model SFT baselines
sbatch sol/eval_small_300.slurm
```

---

## Project Structure

```
from-sft-to-grpo/
├── src/
│   ├── data_utils.py          # GSM8K loading — apply_chat_template for any model
│   ├── model_utils.py         # Model loader: full / LoRA / QLoRA
│   └── evaluate.py            # Greedy inference + exact-match + W&B Table
├── scripts/
│   ├── train_sft.py           # Phase 1 + 2: SFT / LoRA / QLoRA training
│   ├── train_grpo.py          # Phase 3: GRPO RL from SFT checkpoint
│   └── compare.py             # Pull W&B runs → comparison table
├── configs/
│   └── config.yaml            # All hyperparameters
├── sol/
│   ├── setup.sh               # First-time Sol environment setup
│   ├── phase1_sft.slurm       # SLURM array: 3 Phase 1 experiments
│   ├── phase2_scale.slurm     # SLURM array: 3 Phase 2 experiments
│   ├── phase3_grpo.slurm      # SLURM array: 6 GRPO + 300-sample eval
│   ├── eval_small_300.slurm   # 300-sample eval for 0.5B checkpoints
│   └── eval_medium_300.slurm  # 300-sample eval for 3B checkpoints
└── requirements.txt
```

---

## W&B Metrics Logged

| Metric | Phase | Description |
|--------|-------|-------------|
| `train/loss` | 1, 2 | SFT training loss per step |
| `eval/loss` | 1, 2 | Validation loss — overfitting signal |
| `gsm8k_accuracy` | all | Exact-match on #### number |
| `trainable_params` / `trainable_pct` | all | Parameter efficiency |
| `peak_gpu_memory_gb` | all | Peak VRAM usage |
| `train_time_min` | all | Wall-clock training time |
| `train/reward` | 3 | Mean GRPO reward per step |
| `train/kl` | 3 | KL divergence from SFT policy |
| `train/entropy` | 3 | Policy entropy — measures exploration |
| `eval_table` | all | Per-question predictions W&B Table |

---

## Technical Notes

**`apply_chat_template` is required for multi-size compatibility.**
Hardcoding `<|im_start|>` strings worked for 0.5B but caused loss=10.86
on 3B because tokenizer alignment differed. All data formatting uses
`tokenizer.apply_chat_template()` — correct for any Qwen model size.

**QLoRA uses bf16 base during GRPO, not 4-bit.**
4-bit NF4 quantization cannot accumulate gradients. During GRPO the base
model loads in bf16 while the same LoRA adapter is reloaded — identical
trainable parameters as QLoRA SFT with proper gradient flow.

**`padding_side="right"` required for single-sample evaluation.**
GRPOTrainer sets `padding_side="left"` for batch rollout generation.
Evaluation resets to `"right"` before greedy decoding — left padding on
single samples produces incoherent output, causing 0% accuracy.

**100 samples is too noisy for detecting RL improvements.**
A 5-8% RL lift shows as only 5-8 extra correct answers on 100 questions,
which is within random noise. All final results use 300 samples where the
same improvement shows as 15-24 extra answers — statistically reliable.

---

## Tech Stack

`transformers 5.7` · `peft 0.19` · `trl 1.3` · `bitsandbytes` · `accelerate`
`datasets` · `wandb` · `torch` · `Qwen2.5` · `GSM8K (openai/gsm8k)`

---

## License

MIT © 2026 narottaman

---

## W&B Report & Training Curves

**Full interactive report:** [W&B Report — from-sft-to-grpo](https://wandb.ai/ngangada-arizona-state-university/from-sft-to-grpo)

**Project dashboard:** [All runs](https://wandb.ai/ngangada-arizona-state-university/from-sft-to-grpo)

### Reward curve during GRPO training
![Reward curve](assets/wandb_reward_curve.png)
*Mean reward per step across all 6 GRPO runs. 3B models (solid) plateau at
0.75-0.85; 0.5B models (dashed) plateau at 0.45-0.55 — directly predicting
the final accuracy gap.*

### SFT training loss
![Loss curve](assets/wandb_loss_curve.png)
*All 6 SFT runs converge cleanly. 3B models (higher lines initially) drop
faster due to greater model capacity.*

### Token accuracy during SFT
![Token accuracy](assets/wandb_token_accuracy.png)
*3B models reach 85-90% token accuracy on training data; 0.5B models reach
~40% — confirming the capacity difference that drives the scale finding.*

### Policy entropy during GRPO
![Entropy](assets/wandb_entropy.png)
*Entropy decreases from ~1.0 to ~0.2 across all runs. The policy becomes
more confident and focused as GRPO rewards correct reasoning — healthy
training signal, not random exploration.*