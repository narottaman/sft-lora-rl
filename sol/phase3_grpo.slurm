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
#SBATCH --array=0-5
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngangada@asu.edu
#SBATCH --export=NONE

# =============================================================================
# Phase 3 — GRPO on all 6 SFT checkpoints
# Each array job: trains GRPO then evaluates with 300 samples immediately
#
#   Array 0: grpo_full_small    full_small   + GRPO  (~1h,  ~12GB)
#   Array 1: grpo_lora_small    lora_small   + GRPO  (~1h,  ~8GB)
#   Array 2: grpo_qlora_small   qlora_small  + GRPO  (~1h,  ~8GB)
#   Array 3: grpo_full_medium   full_medium  + GRPO  (~2h,  ~35GB)
#   Array 4: grpo_lora_medium   lora_medium  + GRPO  (~2h,  ~15GB)
#   Array 5: grpo_qlora_medium  qlora_medium + GRPO  (~2h,  ~15GB)
#
# Submit all 6:          sbatch sol/phase3_grpo.slurm
# Submit small only:     sbatch --array=0-2 sol/phase3_grpo.slurm
# Submit medium only:    sbatch --array=3-5 sol/phase3_grpo.slurm
# Submit one specific:   sbatch --array=0 sol/phase3_grpo.slurm
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
# Format: "run_name|method|model|sft_checkpoint|base_model_hf"
# base_model_hf needed for 300-sample eval (LoRA/QLoRA load base + adapter)
EXPERIMENTS=(
    "grpo_full_small|full|small|${BASE}/full_small|Qwen/Qwen2.5-0.5B-Instruct"
    "grpo_lora_small|lora|small|${BASE}/lora_small|Qwen/Qwen2.5-0.5B-Instruct"
    "grpo_qlora_small|qlora|small|${BASE}/qlora_small|Qwen/Qwen2.5-0.5B-Instruct"
    "grpo_full_medium|full|medium|${BASE}/full_medium|Qwen/Qwen2.5-3B-Instruct"
    "grpo_lora_medium|lora|medium|${BASE}/lora_medium|Qwen/Qwen2.5-3B-Instruct"
    "grpo_qlora_medium|qlora|medium|${BASE}/qlora_medium|Qwen/Qwen2.5-3B-Instruct"
)

IFS='|' read -r RUN_NAME METHOD MODEL SFT_CKPT BASE_MODEL \
    <<< "${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

echo "--- GRPO Experiment ---"
echo "  Run name:   ${RUN_NAME}"
echo "  Method:     ${METHOD}"
echo "  Model:      ${MODEL}"
echo "  SFT init:   ${SFT_CKPT}"
echo "  Base model: ${BASE_MODEL}"
echo ""

# Verify SFT checkpoint exists before starting
if [ ! -d "${SFT_CKPT}" ]; then
    echo "ERROR: SFT checkpoint not found: ${SFT_CKPT}"
    echo "Run Phase 1/2 first and verify:"
    echo "  ls ${BASE}/"
    exit 1
fi

# ── Step 1: GRPO Training ─────────────────────────────────────────────────────
echo "--- Step 1: GRPO Training ---"
python scripts/train_grpo.py \
    --method         "${METHOD}" \
    --model          "${MODEL}" \
    --run-name       "${RUN_NAME}" \
    --sft-checkpoint "${SFT_CKPT}" \
    --config         configs/config.yaml \
    --no-eval

# Verify GRPO checkpoint was saved before running eval
GRPO_CKPT="${BASE}/${RUN_NAME}"
if [ ! -d "${GRPO_CKPT}" ]; then
    echo "ERROR: GRPO checkpoint not saved at ${GRPO_CKPT}"
    exit 1
fi

# ── Step 2: Stable Evaluation with 300 samples ───────────────────────────────
echo ""
echo "--- Step 2: Evaluation with 300 samples ---"
python src/evaluate.py \
    --checkpoint  "${GRPO_CKPT}" \
    --method      "${METHOD}" \
    --base-model  "${BASE_MODEL}" \
    --run-name    "${RUN_NAME}_eval300" \
    --max-samples 300

echo ""
echo "=================================================="
echo "Phase 3 [${RUN_NAME}] complete — training + eval done."
echo "End: $(date)"
echo "=================================================="
