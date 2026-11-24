#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM job: Serve the original (untuned) Gemma 4 31B-it via vLLM, FP8 on 2× L40S.
# Used by the PDF-processing pipeline (slurm/run_process_pdfs.sh).
#
# Submit (from the repository root): sbatch slurm/serve_gemma4_base.sh
# Test   : curl http://<gpu-node>:8001/v1/models
#
# Optional environment variables:
#   BRIDGE_FT_ENV  prefix of the "bridge_ft" conda env (environment/finetune_conda_environment.yml);
#                  not needed if the env is already active when you call sbatch
#   BRIDGE_ROOT    workspace root holding .hf_cache (default /home/projects)
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=gemma4-base
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/serve_gemma4_base_%j.log
#SBATCH --error=logs/serve_gemma4_base_%j.err
# Add --partition / --account / --nodelist as required by your cluster.

set -euo pipefail

echo "=================================================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURM_NODELIST"
echo "Task        : Serve Gemma 4 31B-it (original) via vLLM"
echo "Start time  : $(date)"
echo "=================================================="

MODEL_NAME="google/gemma-4-31B-it"
PORT=8001
BRIDGE_ROOT="${BRIDGE_ROOT:-/home/projects}"

if [[ -n "${BRIDGE_FT_ENV:-}" ]]; then export PATH="$BRIDGE_FT_ENV/bin:$PATH"; fi
export HF_HOME="${HF_HOME:-$BRIDGE_ROOT/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

echo "Python      : $(which python)"
echo "vLLM        : $(python -c 'import vllm; print(vllm.__version__)')"
echo "Model       : $MODEL_NAME"
echo "Port        : $PORT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --served-model-name "gemma4-base" \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --quantization fp8 \
    --max-model-len 49152 \
    --port $PORT \
    --host 0.0.0.0 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --enforce-eager
