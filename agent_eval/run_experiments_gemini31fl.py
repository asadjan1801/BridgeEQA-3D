"""Generate BridgeEQA agent eval for gemini-3.1-flash-lite on the 200 test scenes.

Runs the standard 5 agent configurations (no_tools_{images,sg_context,both} +
sg_oracle_{sg_context,both}) end-to-end, populating an experiments.db at
  pdf_processed_vermont_pages_greater_than_20_eval_gemini-3.1-flash-lite_cached_output/

Uses Gemini direct API (litellm/gemini/...) — requires GEMINI_API_KEY in .env.
Outputs land in the current working directory (agent_eval/).

Run: `cd agent_eval && uv run python run_experiments_gemini31fl.py`
"""

import asyncio
import os
from pathlib import Path

# Must set before importing skit (which configures logfire)
os.environ.setdefault("LOGFIRE_SEND_TO_LOGFIRE", "false")

# Install Gemini 3.x thought_signature patches BEFORE importing run_experiments
# (which transitively imports openai-agents-sdk + litellm).
import gemini3x_thought_signature_patches  # noqa: F401

from run_experiments import run_concurrent_evaluations


# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME = "litellm/gemini/gemini-3.1-flash-lite"
N_CONCURRENT = 2        # conservative — free-tier Gemini throttles aggressively
MAX_SCENES = 200
SCENE_DIRS_PATH = Path(os.environ.get("BRIDGE_ROOT", "/home/projects")) / "pdf_processed_vermont_pages_greater_than_20"
OUTPUT_DIR = f"pdf_processed_vermont_pages_greater_than_20_eval_{Path(MODEL_NAME).name}_cached_output"


if __name__ == "__main__":
    print(f"Model:        {MODEL_NAME}")
    print(f"Scene dir:    {SCENE_DIRS_PATH}")
    print(f"Output dir:   {OUTPUT_DIR}")
    print(f"Max scenes:   {MAX_SCENES}")
    print(f"Concurrency:  {N_CONCURRENT}")
    asyncio.run(
        run_concurrent_evaluations(
            scene_dirs_path=SCENE_DIRS_PATH,
            output_dir=OUTPUT_DIR,
            n_concurrent=N_CONCURRENT,
            max_scenes=MAX_SCENES,
            model_name=MODEL_NAME,
        )
    )
