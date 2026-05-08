#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH --job-name=grpo-eval
#SBATCH --output=/scratch/ngangada/portfolio/sft-lora-rl/logs/phase3_eval_%j_%a.out
#SBATCH --error=/scratch/ngangada/portfolio/sft-lora-rl/logs/phase3_eval_%j_%a.err
#SBATCH -t 0-01:30:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-5
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngangada@asu.edu
#SBATCH --export=NONE

echo "=================================================="
echo "Phase 3 — GRPO Eval Only"
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
BASE="/scratch/ngangada/portfolio/sft-lora-rl/outputs"

# Format: "checkpoint_dir|method|base_model|run_name"
EXPERIMENTS=(
    "grpo_full_small|full|Qwen/Qwen2.5-0.5B-Instruct|grpo_full_small_eval"
    "grpo_lora_small|lora|Qwen/Qwen2.5-0.5B-Instruct|grpo_lora_small_eval"
    "grpo_qlora_small|qlora|Qwen/Qwen2.5-0.5B-Instruct|grpo_qlora_small_eval"
    "grpo_full_medium|full|Qwen/Qwen2.5-3B-Instruct|grpo_full_medium_eval"
    "grpo_lora_medium|lora|Qwen/Qwen2.5-3B-Instruct|grpo_lora_medium_eval"
    "grpo_qlora_medium|qlora|Qwen/Qwen2.5-3B-Instruct|grpo_qlora_medium_eval"
)

IFS='|' read -r CKPT METHOD BASE_MODEL RUN_NAME \
    <<< "${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

echo "Checkpoint: ${BASE}/${CKPT}"
echo "Method:     ${METHOD}"
echo "Base model: ${BASE_MODEL}"
echo "Run name:   ${RUN_NAME}"
echo ""

# Verify checkpoint exists
if [ ! -d "${BASE}/${CKPT}" ]; then
    echo "ERROR: checkpoint not found at ${BASE}/${CKPT}"
    echo "Available checkpoints:"
    ls "${BASE}/" | grep grpo
    exit 1
fi

python src/evaluate.py \
    --checkpoint  "${BASE}/${CKPT}" \
    --method      "${METHOD}" \
    --base-model  "${BASE_MODEL}" \
    --run-name    "${RUN_NAME}" \
    --max-samples 100

echo ""
echo "=================================================="
echo "Eval [${RUN_NAME}] complete."
echo "End: $(date)"
echo "=================================================="