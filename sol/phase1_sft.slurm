#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH --job-name=sft-phase1
#SBATCH --output=/scratch/ngangada/portfolio/sft-lora-rl/logs/phase1_%j_%a.out
#SBATCH --error=/scratch/ngangada/portfolio/sft-lora-rl/logs/phase1_%j_%a.err
#SBATCH -t 0-02:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngangada@asu.edu
#SBATCH --export=NONE

# =============================================================================
# Phase 1 — SFT Baseline (Qwen2.5-0.5B)
#   Array 0: full fine-tune   (~2 GPU-hours, ~4GB peak)
#   Array 1: LoRA             (~1.5 GPU-hours, ~3GB peak)
#   Array 2: QLoRA            (~1.5 GPU-hours, ~2GB peak)
#
# Submit:  sbatch sol/phase1_sft.slurm
# Status:  squeue -u $USER
# Logs:    tail -f /scratch/ngangada/portfolio/sft-lora-rl/logs/phase1_<id>_0.out
# =============================================================================

echo "=================================================="
echo "Phase 1 — SFT Baseline"
echo "Job ID:     $SLURM_JOB_ID"
echo "Array idx:  $SLURM_ARRAY_TASK_ID"
echo "Node:       $SLURMD_NODENAME"
echo "GPU(s):     $SLURM_JOB_GPUS"
echo "Start:      $(date)"
echo "=================================================="

# ── Environment ───────────────────────────────────────────────────────────────
source ~/envs/sft_lora_rl/bin/activate

export WANDB_API_KEY="${WANDB_API_KEY}"
export HF_HOME="/scratch/ngangada/hf_cache"
export TRANSFORMERS_CACHE="/scratch/ngangada/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/scratch/ngangada/portfolio/sft-lora-rl:$PYTHONPATH"

nvidia-smi
echo ""

mkdir -p /scratch/ngangada/portfolio/sft-lora-rl/logs
mkdir -p /scratch/ngangada/portfolio/sft-lora-rl/outputs
cd /scratch/ngangada/portfolio/sft-lora-rl

# ── Experiment map ────────────────────────────────────────────────────────────
METHODS=("full" "lora" "qlora")
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}

echo "--- Experiment: model=small, method=${METHOD} ---"

python scripts/train_sft.py \
    --model  small \
    --method "${METHOD}" \
    --config configs/config.yaml

echo ""
echo "=================================================="
echo "Phase 1 [${METHOD}_small] complete."
echo "End: $(date)"
echo "=================================================="
