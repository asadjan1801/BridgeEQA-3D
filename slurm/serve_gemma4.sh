#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM job: Serve a merged fine-tuned Gemma 4 31B-it via vLLM
# Tensor-parallel across 2× L40S 48GB.
#
# Prerequisite: the merged model exists at $MODEL_PATH (created by slurm/run_merge_lora.sh),
# or point MODEL_PATH at a downloaded copy of the released merged weights.
#
# Submit (from the repository root):
#   SFT_VARIANT=mc sbatch slurm/serve_gemma4.sh
#   SFT_VARIANT=sc sbatch slurm/serve_gemma4.sh
# Test   : curl http://<gpu-node>:8002/v1/models
#
# Both variants are served under the name "gemma4-bridge" (as in the paper runs).
#
# Optional environment variables: BRIDGE_FT_ENV, BRIDGE_ROOT (default /home/projects),
# FINETUNE_DIR, MODEL_PATH.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=gemma4-serve
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/serve_gemma4_%j.log
#SBATCH --error=logs/serve_gemma4_%j.err
# Add --partition / --account / --nodelist as required by your cluster.

set -euo pipefail

echo "=================================================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURM_NODELIST"
echo "GPUs        : 2× L40S 48GB"
echo "Task        : Serve Gemma 4 31B-it (merged fine-tune) via vLLM"
echo "Start time  : $(date)"
echo "=================================================="

# ── Paths ─────────────────────────────────────────────────────────────────────
BRIDGE_ROOT="${BRIDGE_ROOT:-/home/projects}"
FINETUNE_DIR="${FINETUNE_DIR:-$BRIDGE_ROOT/bridge_finetune}"
case "${SFT_VARIANT:-mc}" in
    mc) DEFAULT_MODEL_PATH="$FINETUNE_DIR/checkpoints/gemma4-31b-bridge-v3all-merged" ;;
    sc) DEFAULT_MODEL_PATH="$FINETUNE_DIR/checkpoints/gemma4-31b-bridge-v3mixed-merged" ;;
    *)  echo "ERROR: SFT_VARIANT must be mc or sc"; exit 1 ;;
esac
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
PORT=8002

# ── Environment ───────────────────────────────────────────────────────────────
if [[ -n "${BRIDGE_FT_ENV:-}" ]]; then export PATH="$BRIDGE_FT_ENV/bin:$PATH"; fi
export HF_HOME="${HF_HOME:-$BRIDGE_ROOT/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

echo ""
echo "Python      : $(which python)"
echo "vLLM        : $(python -c 'import vllm; print(vllm.__version__)')"
echo "Model       : $MODEL_PATH"
echo "Port        : $PORT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Verify merged model exists
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "ERROR: Merged model not found at $MODEL_PATH"
    echo "Run slurm/run_merge_lora.sh first."
    exit 1
fi

# ── Launch vLLM server ────────────────────────────────────────────────────────
# tensor-parallel=2 to spread the 31B model across 2 GPUs
# No LoRA flags needed — this is the fully merged model.
# --reasoning-parser / --tool-call-parser gemma4 are required: the fine-tune emits
# Gemma 4 channel tokens before its answer, and the embodied agent relies on tool calls.
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "gemma4-bridge" \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --quantization fp8 \
    --max-model-len 65536 \
    --port $PORT \
    --host 0.0.0.0 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4
