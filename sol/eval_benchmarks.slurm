#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH --job-name=bench-eval
#SBATCH --output=/scratch/ngangada/portfolio/sft-lora-rl/logs/bench_%j_%a.out
#SBATCH --error=/scratch/ngangada/portfolio/sft-lora-rl/logs/bench_%j_%a.err
#SBATCH -t 0-04:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-11
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngangada@asu.edu
#SBATCH --export=NONE

# =============================================================================
# Multi-benchmark generalization evaluation
# One checkpoint per array job — all run in parallel
#
# Array map (12 checkpoints):
#   0:  full_small          6:  grpo_full_small
#   1:  lora_small          7:  grpo_lora_small
#   2:  qlora_small         8:  grpo_qlora_small
#   3:  full_medium         9:  grpo_full_medium
#   4:  lora_medium         10: grpo_lora_medium
#   5:  qlora_medium        11: grpo_qlora_medium
#
# Submit all:     sbatch sol/eval_benchmarks.slurm
# SFT only:       sbatch --array=0-5 sol/eval_benchmarks.slurm
# GRPO only:      sbatch --array=6-11 sol/eval_benchmarks.slurm
# Small only:     sbatch --array=0-2,6-8 sol/eval_benchmarks.slurm
# Medium only:    sbatch --array=3-5,9-11 sol/eval_benchmarks.slurm
# Single ckpt:    sbatch --array=1 sol/eval_benchmarks.slurm
#
# Benchmarks run: gsm8k, mmlu, arc_easy, hellaswag
# To add truthfulqa and humaneval: add them to --benchmarks below
# =============================================================================

echo "=================================================="
echo "Multi-Benchmark Generalization Eval"
echo "Job ID:     $SLURM_JOB_ID"
echo "Array idx:  $SLURM_ARRAY_TASK_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start:      $(date)"
echo "=================================================="

source ~/envs/sft_lora_rl/bin/activate

export WANDB_API_KEY="${WANDB_API_KEY}"
export HF_HOME="/scratch/ngangada/hf_cache"
export TRANSFORMERS_CACHE="/scratch/ngangada/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/scratch/ngangada/portfolio/sft-lora-rl:$PYTHONPATH"

cd /scratch/ngangada/portfolio/sft-lora-rl
mkdir -p logs data/benchmark_results

# Checkpoint name for this array index
CHECKPOINTS=(
    "full_small"
    "lora_small"
    "qlora_small"
    "full_medium"
    "lora_medium"
    "qlora_medium"
    "grpo_full_small"
    "grpo_lora_small"
    "grpo_qlora_small"
    "grpo_full_medium"
    "grpo_lora_medium"
    "grpo_qlora_medium"
)

CKPT=${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}

echo "Checkpoint: ${CKPT}"
echo ""

python scripts/eval_benchmarks.py \
    --checkpoint  "${CKPT}" \
    --benchmarks  gsm8k mmlu arc_easy hellaswag \
    --output-dir  data/benchmark_results \
    --config      configs/config.yaml

echo ""
echo "=================================================="
echo "Benchmark eval [${CKPT}] complete."
echo "End: $(date)"
echo "=================================================="
