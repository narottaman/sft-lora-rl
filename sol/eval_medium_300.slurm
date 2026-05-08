#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH --job-name=eval-medium-300
#SBATCH --output=/scratch/ngangada/portfolio/sft-lora-rl/logs/eval_medium_300_%j_%a.out
#SBATCH --error=/scratch/ngangada/portfolio/sft-lora-rl/logs/eval_medium_300_%j_%a.err
#SBATCH -t 0-02:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-5
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngangada@asu.edu
#SBATCH --export=NONE

# =============================================================================
# Re-evaluation of all 3B model checkpoints with 300 samples
# Covers Phase 2 (SFT) and Phase 3 (GRPO) for medium models only
#
# Why 300 samples:
#   100 samples has too much variance — a real 5-8% improvement shows as
#   only ~5-8 extra correct answers which is within noise. 300 samples
#   gives ~15-24 extra correct answers for the same improvement, which is
#   statistically stable.
#
# Array map:
#   0: lora_medium   SFT checkpoint  (Phase 2 baseline)
#   1: qlora_medium  SFT checkpoint  (Phase 2 baseline)
#   2: full_medium   SFT checkpoint  (Phase 2 baseline)
#   3: grpo_lora_medium   GRPO checkpoint  (Phase 3)
#   4: grpo_qlora_medium  GRPO checkpoint  (Phase 3)
#   5: grpo_full_medium   GRPO checkpoint  (Phase 3)
#
# Submit all:       sbatch sol/eval_medium_300.slurm
# Submit SFT only:  sbatch --array=0-2 sol/eval_medium_300.slurm
# Submit GRPO only: sbatch --array=3-5 sol/eval_medium_300.slurm
# =============================================================================

echo "=================================================="
echo "Medium Model Re-Evaluation (300 samples)"
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
BASE_MODEL_3B="Qwen/Qwen2.5-3B-Instruct"

# Format: "checkpoint_dir|method|base_model|run_name"
EXPERIMENTS=(
    "lora_medium|lora|${BASE_MODEL_3B}|lora_medium_eval300"
    "qlora_medium|qlora|${BASE_MODEL_3B}|qlora_medium_eval300"
    "full_medium|full|${BASE_MODEL_3B}|full_medium_eval300"
    "grpo_lora_medium|lora|${BASE_MODEL_3B}|grpo_lora_medium_eval300"
    "grpo_qlora_medium|qlora|${BASE_MODEL_3B}|grpo_qlora_medium_eval300"
    "grpo_full_medium|full|${BASE_MODEL_3B}|grpo_full_medium_eval300"
)

IFS='|' read -r CKPT METHOD BASE_MODEL RUN_NAME \
    <<< "${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

echo "Checkpoint: ${BASE}/${CKPT}"
echo "Method:     ${METHOD}"
echo "Run name:   ${RUN_NAME}"
echo ""

# Verify checkpoint exists
if [ ! -d "${BASE}/${CKPT}" ]; then
    echo "ERROR: checkpoint not found at ${BASE}/${CKPT}"
    echo "Available checkpoints:"
    ls "${BASE}/"
    exit 1
fi

python src/evaluate.py \
    --checkpoint  "${BASE}/${CKPT}" \
    --method      "${METHOD}" \
    --base-model  "${BASE_MODEL}" \
    --run-name    "${RUN_NAME}" \
    --max-samples 300

echo ""
echo "=================================================="
echo "Eval [${RUN_NAME}] complete."
echo "End: $(date)"
echo "=================================================="
