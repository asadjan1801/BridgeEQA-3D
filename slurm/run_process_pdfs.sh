#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM job: Process all Vermont inspection PDFs with the locally served Gemma 4 31B
# (4-stage pipeline: PDF parsing -> image-reference mapping -> scene graph -> ratings + QA).
#
# Prerequisite: slurm/serve_gemma4_base.sh is running and reachable at $VLLM_SERVER_URL.
#
# Submit (from the repository root):
#   VLLM_SERVER_URL=http://<gpu-node>:8001/v1 sbatch slurm/run_process_pdfs.sh
#
# Optional environment variables:
#   BRIDGE_ROOT     workspace root (default /home/projects); input PDFs are read from
#                   $BRIDGE_ROOT/vermont_reports_less_than_30_MB and scenes written to
#                   $BRIDGE_ROOT/pdf_processed_ALL_vermont_reports (override with
#                   PDF_INPUT_DIR / PDF_OUTPUT_DIR)
#   VLLM_SERVER_URL default http://localhost:8001/v1
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=process-pdfs
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=logs/process_pdfs_%j.log
#SBATCH --error=logs/process_pdfs_%j.err
# Add --partition / --account / --nodelist as required by your cluster.

set -euo pipefail

echo "=================================================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURM_NODELIST"
echo "Task        : Process all Vermont PDFs with local Gemma 4 31B"
echo "Start time  : $(date)"
echo "=================================================="

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
BRIDGE_ROOT="${BRIDGE_ROOT:-/home/projects}"
export BRIDGE_ROOT
cd "$REPO_DIR/agent_eval"
export HF_HOME="${HF_HOME:-$BRIDGE_ROOT/.hf_cache}"
export LOGFIRE_SEND_TO_LOGFIRE=false
export VLLM_SERVER_URL="${VLLM_SERVER_URL:-http://localhost:8001/v1}"
export VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-gemma4-base}"

# Wait for server
echo "Waiting for vLLM server at $VLLM_SERVER_URL ..."
for i in $(seq 1 60); do
    if curl -s "$VLLM_SERVER_URL/models" > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    echo "  Attempt $i/60 — waiting 30s..."
    sleep 30
done

if ! curl -s "$VLLM_SERVER_URL/models" > /dev/null 2>&1; then
    echo "ERROR: Server not reachable after 30 minutes"
    exit 1
fi

echo ""
echo "Processing PDFs..."
uv run python process_pdfs_local.py

echo ""
echo "=================================================="
echo "Finished at: $(date)"
echo "=================================================="
