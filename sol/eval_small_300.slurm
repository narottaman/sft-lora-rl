#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH --job-name=eval-small-300
#SBATCH --output=/scratch/ngangada/portfolio/sft-lora-rl/logs/eval_small_300_%j_%a.out
#SBATCH --error=/scratch/ngangada/portfolio/sft-lora-rl/logs/eval_small_300_%j_%a.err
#SBATCH -t 0-01:00:00
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

# =============================================================================
# Evaluation of all 0.5B model checkpoints with 300 samples
# Covers Phase 1 (SFT) and Phase 3 (GRPO) for small models
#
# Array map:
#   0: full_small        SFT checkpoint  (Phase 1 baseline)
#   1: lora_small        SFT checkpoint  (Phase 1 baseline)
#   2: qlora_small       SFT checkpoint  (Phase 1 baseline)
#   3: grpo_full_small   GRPO checkpoint (Phase 3)
#   4: grpo_lora_small   GRPO checkpoint (Phase 3)
#   5: grpo_qlora_small  GRPO checkpoint (Phase 3)
#
# Submit all:        sbatch sol/eval_small_300.slurm
# Submit SFT only:   sbatch --array=0-2 sol/eval_small_300.slurm
# Submit GRPO only:  sbatch --array=3-5 sol/eval_small_300.slurm
# =============================================================================

echo "=================================================="
echo "Small Model Evaluation (300 samples)"
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
BASE_MODEL_05B="Qwen/Qwen2.5-0.5B-Instruct"

# Format: "checkpoint_dir|method|base_model|run_name"
EXPERIMENTS=(
    "full_small|full|${BASE_MODEL_05B}|full_small_eval300"
    "lora_small|lora|${BASE_MODEL_05B}|lora_small_eval300"
    "qlora_small|qlora|${BASE_MODEL_05B}|qlora_small_eval300"
    "grpo_full_small|full|${BASE_MODEL_05B}|grpo_full_small_eval300"
    "grpo_lora_small|lora|${BASE_MODEL_05B}|grpo_lora_small_eval300"
    "grpo_qlora_small|qlora|${BASE_MODEL_05B}|grpo_qlora_small_eval300"
)

IFS='|' read -r CKPT METHOD BASE_MODEL RUN_NAME \
    <<< "${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

echo "Checkpoint: ${BASE}/${CKPT}"
echo "Method:     ${METHOD}"
echo "Run name:   ${RUN_NAME}"
echo ""

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
