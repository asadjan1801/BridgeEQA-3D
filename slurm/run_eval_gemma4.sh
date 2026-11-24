#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM job: Run the BridgeEQA 5-configuration evaluation against a served fine-tune.
# Runs on a CPU node — the model is served separately via slurm/serve_gemma4.sh.
#
# Submit (from the repository root):
#   VLLM_SERVER_URL=http://<gpu-node>:8002/v1 sbatch slurm/run_eval_gemma4.sh
#
# Output: agent_eval/pdf_processed_vermont_pages_greater_than_20_eval_gemma4-bridge_cached_output/
# Both fine-tunes are served as "gemma4-bridge", so they write to the same directory.
# For the paper, the SC run's directory was renamed to
#   ..._eval_gemma4-bridge_cached_output_v3mixed_BACKUP/
# which is the path agent_eval/run_analysis_judge31fl.py expects.
#
# Optional environment variables:
#   BRIDGE_ROOT      workspace root (default /home/projects); the 200 test scenes are read from
#                    $BRIDGE_ROOT/pdf_processed_vermont_pages_greater_than_20
#   VLLM_SERVER_URL  default http://localhost:8002/v1
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=gemma4-eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/eval_gemma4_%j.log
#SBATCH --error=logs/eval_gemma4_%j.err
# Add --partition / --account / --nodelist as required by your cluster.

set -euo pipefail

echo "=================================================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURM_NODELIST"
echo "Task        : BridgeEQA evaluation with fine-tuned Gemma 4 31B"
echo "Start time  : $(date)"
echo "=================================================="

# ── Environment ───────────────────────────────────────────────────────────────
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
BRIDGE_ROOT="${BRIDGE_ROOT:-/home/projects}"
cd "$REPO_DIR/agent_eval"
# The evaluation harness uses uv, not conda
export HF_HOME="${HF_HOME:-$BRIDGE_ROOT/.hf_cache}"
export LOGFIRE_SEND_TO_LOGFIRE=false

# ── Wait for model server to be ready ─────────────────────────────────────────
SERVER_URL="${VLLM_SERVER_URL:-http://localhost:8002/v1}"
echo "Waiting for vLLM server at $SERVER_URL ..."
for i in $(seq 1 60); do
    if curl -s "$SERVER_URL/models" > /dev/null 2>&1; then
        echo "Server is ready!"
        curl -s "$SERVER_URL/models" | python -m json.tool
        break
    fi
    echo "  Attempt $i/60 — server not ready, waiting 30s..."
    sleep 30
done

# Final check
if ! curl -s "$SERVER_URL/models" > /dev/null 2>&1; then
    echo "ERROR: vLLM server not reachable after 30 minutes"
    exit 1
fi

# ── Run evaluation ────────────────────────────────────────────────────────────
echo ""
echo "Starting evaluation..."
echo "  Model: openai/gemma4-bridge"
echo "  Server: $SERVER_URL"
echo "  Max scenes: 200"
echo "  Concurrency: 1"
echo "  Agent configs: 5 (3 no-tools + 2 sg_oracle)"
echo ""

uv run python run_eval_finetuned.py \
    --model_name "openai/gemma4-bridge" \
    --server_urls "$SERVER_URL" \
    --max_scenes 200 \
    --n_concurrent 1 \
    --scene_dir "$BRIDGE_ROOT/pdf_processed_vermont_pages_greater_than_20"

echo ""
echo "=================================================="
echo "Evaluation finished at: $(date)"
echo "=================================================="
