#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH --job-name=grpo-phase3
#SBATCH --output=/scratch/ngangada/portfolio/sft-lora-rl/logs/phase3_%j_%a.out
#SBATCH --error=/scratch/ngangada/portfolio/sft-lora-rl/logs/phase3_%j_%a.err
#SBATCH -t 0-08:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngangada@asu.edu
#SBATCH --export=NONE

# =============================================================================
# Phase 3 — GRPO Reinforcement Learning
#   Array 0: grpo_lora_small   — best small model + GRPO
#   Array 1: grpo_lora_medium  — best large model + GRPO
#   Array 2: grpo_full_small   — full SFT + GRPO (no adapters baseline)
#
# IMPORTANT: Run this AFTER Phase 1 + 2 are complete.
#   Check: ls /scratch/ngangada/portfolio/sft-lora-rl/outputs/
#   Should see: full_small/, lora_small/, qlora_small/, lora_medium/, qlora_medium/
#
# Submit:  sbatch sol/phase3_grpo.slurm
# =============================================================================

echo "=================================================="
echo "Phase 3 — GRPO Reinforcement Learning"
echo "Job ID:     $SLURM_JOB_ID"
echo "Array idx:  $SLURM_ARRAY_TASK_ID"
echo "Node:       $SLURMD_NODENAME"
echo "GPU(s):     $SLURM_JOB_GPUS"
echo "Start:      $(date)"
echo "=================================================="

source ~/envs/sft_lora_rl/bin/activate

export WANDB_API_KEY="${WANDB_API_KEY}"
export HF_HOME="/scratch/ngangada/hf_cache"
export TRANSFORMERS_CACHE="/scratch/ngangada/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/scratch/ngangada/portfolio/sft-lora-rl:$PYTHONPATH"

BASE="/scratch/ngangada/portfolio/sft-lora-rl/outputs"

nvidia-smi
echo ""

mkdir -p /scratch/ngangada/portfolio/sft-lora-rl/logs
mkdir -p /scratch/ngangada/portfolio/sft-lora-rl/outputs
cd /scratch/ngangada/portfolio/sft-lora-rl

# ── Experiment map ────────────────────────────────────────────────────────────
# Each entry: "run_name|method|model|sft_checkpoint"
EXPERIMENTS=(
    "grpo_lora_small|lora|small|${BASE}/lora_small"
    "grpo_lora_medium|lora|medium|${BASE}/lora_medium"
    "grpo_full_small|full|small|${BASE}/full_small"
)

IFS='|' read -r RUN_NAME METHOD MODEL SFT_CKPT \
    <<< "${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

echo "--- GRPO Experiment ---"
echo "  Run name:       ${RUN_NAME}"
echo "  Method:         ${METHOD}"
echo "  Model:          ${MODEL}"
echo "  SFT checkpoint: ${SFT_CKPT}"
echo ""

# Verify checkpoint exists
if [ ! -d "${SFT_CKPT}" ]; then
    echo "ERROR: SFT checkpoint not found at ${SFT_CKPT}"
    echo "Run Phase 1/2 first: sbatch sol/phase1_sft.slurm && sbatch sol/phase2_scale.slurm"
    exit 1
fi

python scripts/train_grpo.py \
    --sft-checkpoint "${SFT_CKPT}" \
    --method         "${METHOD}" \
    --model          "${MODEL}" \
    --run-name       "${RUN_NAME}" \
    --config         configs/config.yaml

echo ""
echo "=================================================="
echo "Phase 3 [${RUN_NAME}] complete."
echo "End: $(date)"
echo "=================================================="
