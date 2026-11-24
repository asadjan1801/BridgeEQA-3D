#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM job: Merge a LoRA adapter into the Gemma 4 31B base model (for vLLM serving)
# Runs on CPU with high memory — no GPU needed.
#
# Submit (from the repository root):
#   SFT_VARIANT=mc sbatch slurm/run_merge_lora.sh   # merges v3all/manual-checkpoint-10400
#   SFT_VARIANT=sc sbatch slurm/run_merge_lora.sh   # merges v3mixed/checkpoint-2700
#
# Optional environment variables: BRIDGE_FT_ENV, BRIDGE_ROOT (default /home/projects),
# FINETUNE_DIR, ADAPTER_PATH, OUTPUT_PATH (see finetune/train/merge_lora.py).
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=merge-lora
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --output=logs/merge_lora_%j.log
#SBATCH --error=logs/merge_lora_%j.err
# Add --partition / --account / --nodelist as required by your cluster.

set -euo pipefail

echo "=================================================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURM_NODELIST"
echo "Task        : Merge LoRA adapter into Gemma 4 31B"
echo "Start time  : $(date)"
echo "=================================================="

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
BRIDGE_ROOT="${BRIDGE_ROOT:-/home/projects}"
export SFT_VARIANT="${SFT_VARIANT:-mc}"
export FINETUNE_DIR="${FINETUNE_DIR:-$BRIDGE_ROOT/bridge_finetune}"

if [[ -n "${BRIDGE_FT_ENV:-}" ]]; then export PATH="$BRIDGE_FT_ENV/bin:$PATH"; fi
export HF_HOME="${HF_HOME:-$BRIDGE_ROOT/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

echo "Python: $(which python)"
echo "PEFT:   $(python -c 'import peft; print(peft.__version__)')"

cd "$REPO_DIR"
python finetune/train/merge_lora.py

echo ""
echo "=================================================="
echo "Merge finished at: $(date)"
echo "=================================================="
