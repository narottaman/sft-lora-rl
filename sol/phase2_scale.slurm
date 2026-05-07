#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH --job-name=sft-phase2
#SBATCH --output=/scratch/ngangada/portfolio/sft-lora-rl/logs/phase2_%j_%a.out
#SBATCH --error=/scratch/ngangada/portfolio/sft-lora-rl/logs/phase2_%j_%a.err
#SBATCH -t 0-06:00:00
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
# Phase 2 — Scale (Qwen2.5-3B)
#   Array 0: LoRA        (~10 min, ~12GB peak)   — rerun with bug fix
#   Array 1: QLoRA       (~13 min, ~9GB peak)    — already succeeded (65%)
#   Array 2: Full SFT    (~30 min, ~24GB peak)   — new, best checkpoint for RL
#
# Submit all three:   sbatch sol/phase2_scale.slurm
# Submit lora only:   sbatch --array=0 sol/phase2_scale.slurm
# Submit full only:   sbatch --array=2 sol/phase2_scale.slurm
# =============================================================================

echo "=================================================="
echo "Phase 2 — Scale (3B)"
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

nvidia-smi
echo ""

mkdir -p /scratch/ngangada/portfolio/sft-lora-rl/logs
mkdir -p /scratch/ngangada/portfolio/sft-lora-rl/outputs
cd /scratch/ngangada/portfolio/sft-lora-rl

# ── Experiment map ────────────────────────────────────────────────────────────
#   0: lora  — fixed version (apply_chat_template), rerun after bug fix
#   1: qlora — already ran successfully at 65%, reruns won't hurt
#   2: full  — all 3B weights trainable, best possible checkpoint for Phase 3 RL
METHODS=("lora" "qlora" "full")
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}

echo "--- Experiment: model=medium, method=${METHOD} ---"

python scripts/train_sft.py \
    --model  medium \
    --method "${METHOD}" \
    --config configs/config.yaml

echo ""
echo "=================================================="
echo "Phase 2 [${METHOD}_medium] complete."
echo "End: $(date)"
echo "=================================================="
