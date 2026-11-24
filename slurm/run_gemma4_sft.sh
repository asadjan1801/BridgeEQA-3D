#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM job: Fine-tune Gemma 4 31B-it on bridge inspection data
# Single node: 4× L40S 48GB. Method: QLoRA 4-bit + DeepSpeed ZeRO-2.
#
# Submit (from the repository root):
#   SFT_VARIANT=mc sbatch slurm/run_gemma4_sft.sh   # Multi-Context  (data_v3_all,   168,540 samples)
#   SFT_VARIANT=sc sbatch slurm/run_gemma4_sft.sh   # Single-Context (data_v3_mixed,  56,180 samples)
#
# Optional environment variables:
#   BRIDGE_FT_ENV  prefix of the "bridge_ft" conda env; not needed if it is already active
#   BRIDGE_ROOT    workspace root (default /home/projects); FINETUNE_DIR defaults to
#                  $BRIDGE_ROOT/bridge_finetune and must contain data_v3_all/ or data_v3_mixed/
#   CUDA_HOME      DeepSpeed only checks that this exists; a stub directory is sufficient
# See finetune/train/gemma4_sft.py for the remaining options.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=gemma4-sft
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=15-00:00:00
#SBATCH --output=logs/gemma4_sft_%j.log
#SBATCH --error=logs/gemma4_sft_%j.err
# Add --partition / --account / --nodelist as required by your cluster.

set -euo pipefail

echo "=================================================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURM_NODELIST"
echo "GPUs        : 4× L40S 48GB"
echo "Method      : QLoRA 4-bit + DeepSpeed ZeRO-2"
echo "Start time  : $(date)"
echo "=================================================="

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
BRIDGE_ROOT="${BRIDGE_ROOT:-/home/projects}"
export SFT_VARIANT="${SFT_VARIANT:-mc}"
export FINETUNE_DIR="${FINETUNE_DIR:-$BRIDGE_ROOT/bridge_finetune}"
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"

# ── Environment ───────────────────────────────────────────────────────────────
if [[ -n "${BRIDGE_FT_ENV:-}" ]]; then export PATH="$BRIDGE_FT_ENV/bin:$PATH"; fi

# CUDA_HOME shim for DeepSpeed's compatibility check (no actual compilation needed)
export CUDA_HOME="${CUDA_HOME:-$HOME/.local/cuda_shim}"

# Hugging Face cache on shared storage
export HF_HOME="${HF_HOME:-$BRIDGE_ROOT/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE"

# Move Triton autotune cache off NFS (NFS hangs caused prior crashes during save)
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}_${SLURM_JOB_ID}"
# DeepSpeed has a separate Triton autotune cache that uses TRITON_HOME (not TRITON_CACHE_DIR)
export TRITON_HOME="/tmp/triton_home_${USER}_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR" "$TRITON_HOME"

# Verify /tmp is local (not NFS) — if it is NFS, all our local-disk fallbacks are useless
echo "─── /tmp filesystem ───"
df -T /tmp
echo "─── home filesystem (for comparison) ───"
df -T $HOME

# ── Sanity checks ─────────────────────────────────────────────────────────────
echo ""
echo "Python      : $(which python)"
echo "Transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "PEFT        : $(python -c 'import peft; print(peft.__version__)')"
echo "DeepSpeed   : $(python -c 'import deepspeed; print(deepspeed.__version__)')"
echo "PyTorch     : $(python -c 'import torch; print(torch.__version__)')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Verify dataset
case "$SFT_VARIANT" in
    mc) DATA_DIR="${SFT_DATA_DIR:-$FINETUNE_DIR/data_v3_all}" ;;
    sc) DATA_DIR="${SFT_DATA_DIR:-$FINETUNE_DIR/data_v3_mixed}" ;;
    *)  echo "ERROR: SFT_VARIANT must be mc or sc"; exit 1 ;;
esac
TRAIN_FILE="$DATA_DIR/train.jsonl"
VAL_FILE="$DATA_DIR/val.jsonl"
if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "ERROR: train.jsonl not found at $TRAIN_FILE"
    exit 1
fi
echo "Variant       : $SFT_VARIANT"
echo "Train samples : $(wc -l < $TRAIN_FILE)"
echo "Val   samples : $(wc -l < $VAL_FILE)"
echo ""

# ── Launch training ───────────────────────────────────────────────────────────
cd "$REPO_DIR"

torchrun --nproc_per_node=4 --master_port=$((29500 + SLURM_JOB_ID % 1000)) finetune/train/gemma4_sft.py \
    2>&1 | tee "$LOG_DIR/gemma4_sft_${SLURM_JOB_ID}_train.log"

echo ""
echo "=================================================="
echo "Training finished at: $(date)"
echo "Checkpoints saved under: $FINETUNE_DIR/checkpoints/"
echo "=================================================="
