#!/bin/bash
# =============================================================================
# sol/setup.sh — First-time environment setup on ASU Sol HPC
#
# Run this ONCE manually before submitting any SLURM jobs:
#   ssh ngangada@sol.asu.edu
#   cd /scratch/ngangada/portfolio/sft-lora-rl
#   bash sol/setup.sh
# =============================================================================

set -e  # exit on first error

echo "=================================================="
echo "Setting up sft-lora-rl environment on Sol"
echo "=================================================="

# ── Load modules ──────────────────────────────────────────────────────────────
module load python/3.11
module load cuda/12.1

# ── Create directories ────────────────────────────────────────────────────────
mkdir -p /scratch/ngangada/hf_cache
mkdir -p /scratch/ngangada/portfolio/sft-lora-rl/logs
mkdir -p /scratch/ngangada/portfolio/sft-lora-rl/outputs

echo "[1/5] Directories created"

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -d ~/envs/sft_lora_rl ]; then
    python -m venv ~/envs/sft_lora_rl
    echo "[2/5] Virtual environment created at ~/envs/sft_lora_rl"
else
    echo "[2/5] Virtual environment already exists — skipping"
fi

source ~/envs/sft_lora_rl/bin/activate

# ── Install dependencies ──────────────────────────────────────────────────────
pip install --upgrade pip --quiet

echo "[3/5] Installing requirements..."
pip install -r /scratch/ngangada/portfolio/sft-lora-rl/requirements.txt --quiet

# Optional: flash-attn speeds up attention ~2x on A100 but takes 10min to compile
# Uncomment if you have time:
# pip install flash-attn --no-build-isolation

echo "[4/5] Dependencies installed"

# ── Auth ──────────────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Authentication — paste your tokens when prompted"
echo ""

echo "--- Weights & Biases ---"
wandb login

echo ""
echo "--- HuggingFace (needed for Qwen model weights) ---"
huggingface-cli login

# ── Add env vars to .bashrc ───────────────────────────────────────────────────
echo ""
echo "Add these to your ~/.bashrc on Sol (copy-paste):"
echo ""
echo "  export WANDB_API_KEY='<your_key>'"
echo "  export HF_HOME='/scratch/ngangada/hf_cache'"
echo "  export TRANSFORMERS_CACHE='/scratch/ngangada/hf_cache'"
echo "  export TOKENIZERS_PARALLELISM=false"
echo ""

# ── Smoke test ────────────────────────────────────────────────────────────────
echo "Running smoke test (100 samples, 1 epoch — should finish in ~2 min)..."
cd /scratch/ngangada/portfolio/sft-lora-rl

python scripts/train_sft.py \
    --model small \
    --method lora \
    --smoke-test \
    --no-eval

echo ""
echo "=================================================="
echo "Setup complete! Submit jobs with:"
echo "  sbatch sol/phase1_sft.slurm"
echo "=================================================="