#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH --job-name=base-eval
#SBATCH --output=/scratch/ngangada/portfolio/sft-lora-rl/logs/base_eval_%j_%a.out
#SBATCH --error=/scratch/ngangada/portfolio/sft-lora-rl/logs/base_eval_%j_%a.err
#SBATCH -t 0-02:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngangada@asu.edu
#SBATCH --export=NONE

# =============================================================================
# Base model evaluation — no fine-tuning, raw pretrained weights
# Establishes true zero-shot baseline for comparison with all fine-tuned models
#
#   Array 0: Qwen2.5-0.5B-Instruct (small base)
#   Array 1: Qwen2.5-3B-Instruct   (medium base)
#
# Submit both:    sbatch sol/eval_base_models.slurm
# Small only:     sbatch --array=0 sol/eval_base_models.slurm
# Medium only:    sbatch --array=1 sol/eval_base_models.slurm
# =============================================================================

echo "=================================================="
echo "Base Model Evaluation (zero-shot)"
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

MODELS=("small" "medium")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Model size: ${MODEL}"
echo ""

python scripts/eval_base_models.py \
    --model       "${MODEL}" \
    --benchmarks  gsm8k mmlu arc_easy hellaswag \
    --output-dir  data/benchmark_results \
    --config      configs/config.yaml

echo ""
echo "=================================================="
echo "Base eval [${MODEL}] complete."
echo "End: $(date)"
echo "=================================================="
