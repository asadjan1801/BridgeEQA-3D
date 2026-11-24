"""
Run BridgeEQA experiments with a fine-tuned model served via OpenAI-compatible API.

Bypasses CloudFileStorage (S3) by encoding local images as base64 data URIs.
Runs the five BridgeEQA agent configurations reported in Table 2 of the paper:
three tool-free VLM configurations (images / scene graph / both) and two embodied-agent
configurations with the BridgeEQA image-graph navigation tools (scene graph / both).

Usage:
    # With the fine-tuned model already served by vLLM (slurm/serve_gemma4.sh):
    LOGFIRE_SEND_TO_LOGFIRE=false uv run python run_eval_finetuned.py \
        --model_name "openai/gemma4-bridge" \
        --server_urls "http://localhost:8002/v1" \
        --scene_dir /path/to/pdf_processed_vermont_pages_greater_than_20
"""

import asyncio
import argparse
import base64
import io
import itertools
import json
import os
import re
from pathlib import Path
from typing import Dict, List

from PIL import Image

# Must set before importing skit (which configures logfire)
os.environ.setdefault("LOGFIRE_SEND_TO_LOGFIRE", "false")

import logfire
import skit.agent_configurations as agent_cfg
from skit.agent_configurations import AgentConfig
from skit.skit_agents.tools.map import scene_graph_interface_tool
from skit.skit_agents.tools.oracle import oracle_tool
from skit.skit_agents.scene_graph_schema import SceneGraph
from skit.skit_agents.qa_response import InspectionQuestionAnswerResponse
from skit.skit_agents.embodied.agent import EmbodiedInspectionAgent
from skit.bridge_inspection_schemas import NbiScore
from agents import Agent, ModelSettings, Runner, RunResult
from agents.extensions.models.litellm_model import LitellmModel
import dataclasses

# Import the experiment runner but we'll monkey-patch functions that use S3
import litellm
import run_experiments

# Increase timeout for local model inference (default 600s is too short)
litellm.request_timeout = 3600

# Global output directory for results (set in main, used by patched process_scene)
RESULTS_OUTPUT_DIR: Path | None = None


# The five agent configurations of Table 2 (3 tool-free + 2 embodied agent with navigation tools).
FINETUNE_EVAL_CONFIGS = {
    "embodied_no_tools_images": AgentConfig("embodied", [], "images"),
    "embodied_no_tools_sg_context": AgentConfig("embodied", [], "sg_context"),
    "embodied_no_tools_both": AgentConfig("embodied", [], "both"),
    "embodied_sg_oracle_sg_context": AgentConfig("embodied", [scene_graph_interface_tool, oracle_tool], "sg_context"),
    "embodied_sg_oracle_both": AgentConfig("embodied", [scene_graph_interface_tool, oracle_tool], "both"),
}


MAX_IMAGE_PIXELS = 1003520  # matches server --max_pixels
MAX_IMAGES_PER_PROMPT = 40  # cap images to stay within vLLM context window


def image_to_base64_url(file_path: Path) -> str:
    """Convert a local image file to a resized base64 data URI (JPEG).

    Resizes to fit within MAX_IMAGE_PIXELS to avoid sending huge payloads
    that the server would resize anyway.
    """
    img = Image.open(file_path)
    img = img.convert("RGB")

    w, h = img.size
    total_pixels = w * h
    if total_pixels > MAX_IMAGE_PIXELS:
        scale = (MAX_IMAGE_PIXELS / total_pixels) ** 0.5
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def prepare_images_for_prompt_local(scene_dir: str) -> list[dict]:
    """Replacement for run_experiments.prepare_images_for_prompt that uses local base64."""
    image_dir = Path(scene_dir) / "images"
    images = []
    count = 0
    for image_file_path in sorted(image_dir.iterdir()):
        if image_file_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if count >= MAX_IMAGES_PER_PROMPT:
            break
        b64_url = image_to_base64_url(image_file_path)
        images.append({"type": "input_image", "image_url": b64_url})
        images.append({"type": "input_text", "text": f"Image name: {image_file_path.name}"})
        count += 1
    return images


def create_node_and_image_url_mapping_local(scene_dir: str, scene_graph: SceneGraph) -> Dict[int, str]:
    """Replacement for run_experiments.create_node_and_image_url_mapping that uses local base64."""
    image_dir = Path(scene_dir) / "images"
    node_to_image_mapping = {}
    for idx, node in enumerate(scene_graph.nodes):
        image_file_path = image_dir / Path(node.image_name)
        if image_file_path.exists():
            node_to_image_mapping[idx] = image_to_base64_url(image_file_path)
        else:
            node_to_image_mapping[idx] = ""
    return node_to_image_mapping


def patch_agent_model(server_urls: List[str], litellm_model_name: str):
    """Patch Agent to use LitellmModel, round-robin across server_urls.

    Each newly constructed Agent gets the next URL in the rotation, so scenes
    processed concurrently end up spread across the server pool.

    Also caps max_tokens at 2048 — the fine-tuned model sometimes fails to emit
    EOS and will happily generate tens of thousands of tokens, causing
    litellm-side timeouts.
    """
    _original_init = Agent.__init__
    _url_cycle = itertools.cycle(server_urls)

    def _patched_init(self, *args, **kwargs):
        kwargs["model"] = LitellmModel(
            model=litellm_model_name,
            base_url=next(_url_cycle),
            api_key="dummy",
        )
        # The fine-tuned model loops on `reference_images` lists without
        # repetition penalty, burning the whole 4096-token budget. A modest
        # frequency_penalty kills the loop without hurting non-looping outputs.
        ms_defaults = dict(max_tokens=2048, frequency_penalty=0.3)
        existing_ms = kwargs.get("model_settings")
        if existing_ms is None:
            kwargs["model_settings"] = ModelSettings(**ms_defaults)
        else:
            updates = {k: v for k, v in ms_defaults.items() if getattr(existing_ms, k) is None}
            if updates:
                kwargs["model_settings"] = dataclasses.replace(existing_ms, **updates)
        _original_init(self, *args, **kwargs)

    Agent.__init__ = _patched_init


# ─── Fine-tuned model output extraction ───────────────────────────────────────
# The fine-tuned Qwen2.5-VL was trained to produce prose followed by a fenced
# JSON block (e.g. ```json\n{"answer": "...", "score": 6}\n```). The default
# `EmbodiedInspectionAgent.async_inspect` sends this raw text through a
# `reformatting_agent` with `output_type=InspectionQuestionAnswerResponse`,
# which expects the *whole* response to be a JSON object and therefore fails
# with `ModelBehaviorError: Invalid JSON when parsing ...`.
#
# We bypass the reformatting step entirely and parse the fine-tuned model's
# output ourselves.

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"\{[^{}]*\"(?:answer|score|condition_rating)\"[^{}]*\}", re.DOTALL)
_INLINE_JSON_SCORE_RE = re.compile(r'"score"\s*:\s*(\d+)')
_IMG_CITATION_RE = re.compile(r"\[img:([^\]]+)\]")

# Fallback prose patterns observed in fine-tuned model outputs:
#   "Condition Rating: 6"         "Condition Rating is 6"       "Condition Rating of 6"
#   "rated a 6"                    "rated 6"                     "rating of 6"
#   "rated as 5 - FAIR"           "rated as 8"                   "are rated as a 7"
#   "score: 6"                     "scored 6"                    "Rating: 6"
# The model sometimes writes the rating both inline and in a Markdown header.
# All patterns constrain the score to a single digit 0-9 (NBI scale).
_PROSE_SCORE_PATTERNS = [
    re.compile(r"(?i)condition\s+rating(?:\s+(?:is|of|:))?\s*[:\-]?\s*([0-9])\b"),
    re.compile(r"(?i)\brating(?:\s+(?:is|of|:))?\s*[:\-]?\s*([0-9])\b"),
    # "rated a 6", "rated 6", "rated as 6", "rated as a 6", "rated as 5 - FAIR"
    re.compile(r"(?i)\brated\s+(?:as\s+)?(?:a\s+)?([0-9])\b"),
    re.compile(r"(?i)\bscore(?:\s+(?:is|of|:))?\s*[:\-]?\s*([0-9])\b"),
    re.compile(r"(?i)\bscored\s+([0-9])\b"),
    # "NBI rating for the wingwalls is 7" — rating ... <stuff> ... is/of/= N
    re.compile(r"(?i)\brating\b[^.\n]{0,80}?\b(?:is|of|=|at)\s+([0-9])\b"),
    # Markdown header variant: "### Condition Rating\n\n6" or "## Score\n6"
    re.compile(r"(?im)^#{1,6}\s*(?:condition\s+rating|rating|score)[^\n]*\n+\s*([0-9])\b"),
]


def _score_from_prose(text: str) -> int | None:
    """Best-effort score extraction from prose. Returns the FIRST match across
    the ordered patterns — the ordering is deliberate, more specific first."""
    for pat in _PROSE_SCORE_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                v = int(m.group(1))
                if 0 <= v <= 9:
                    return v
            except ValueError:
                continue
    return None


def _extract_inspection_response(text: str) -> InspectionQuestionAnswerResponse:
    """Parse fine-tuned model's output into the expected schema.

    Handles these shapes seen in the smoke eval:
      - Prose + trailing ```json { "score": N } ``` block
      - Prose + ```json { "answer": "...", "score": N, ... } ```
      - Prose only with "Condition Rating: N" / "rated a N" / etc.
      - Markdown header "### Condition Rating" followed by a digit
    """
    answer = text.strip()
    score = None

    match = _JSON_FENCE_RE.search(text) or _BARE_JSON_RE.search(text)
    if match:
        json_str = match.group(1) if match.re is _JSON_FENCE_RE else match.group(0)
        prose_before = text[: match.start()].strip().rstrip("#").strip()
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            data = None

        if isinstance(data, dict):
            if isinstance(data.get("answer"), str) and data["answer"].strip():
                answer = data["answer"].strip()
            elif prose_before:
                answer = prose_before
            raw_score = data.get("score")
            if raw_score is None and isinstance(data.get("condition_rating"), dict):
                raw_score = data["condition_rating"].get("score")
            if isinstance(raw_score, (int, float)):
                score = int(raw_score)
            elif isinstance(raw_score, str) and raw_score.strip().isdigit():
                score = int(raw_score.strip())

    # Fallback 1: inline JSON-ish "score": N anywhere in the text.
    if score is None:
        m = _INLINE_JSON_SCORE_RE.search(text)
        if m:
            try:
                score = int(m.group(1))
            except ValueError:
                pass

    # Fallback 2: natural-language patterns ("Condition Rating: 6", etc.)
    if score is None:
        score = _score_from_prose(text)

    # Extract [img:filename] citations from model output
    cited_images = list(dict.fromkeys(_IMG_CITATION_RE.findall(text)))  # dedupe, preserve order

    # Also check JSON reference_images array if present
    if not cited_images and match:
        json_str = match.group(1) if match.re is _JSON_FENCE_RE else match.group(0)
        try:
            data = json.loads(json_str)
            if isinstance(data, dict) and isinstance(data.get("reference_images"), list):
                cited_images = [img for img in data["reference_images"] if isinstance(img, str)]
        except (json.JSONDecodeError, ValueError):
            pass

    return InspectionQuestionAnswerResponse(
        answer=answer,
        reference_images=cited_images if cited_images else None,
        condition_rating=NbiScore(score=score) if score is not None else None,
    )


_CITATION_SUPPLEMENT = (
    "\n\n## Image Citation Format\n"
    "When referencing specific images in your answer, use inline [img:filename] "
    "citations wherever a specific image supports a claim. For example:\n"
    '"The deck shows moderate spalling [img:abc123.png] with efflorescence '
    'visible on the underside [img:def456.png]."\n'
    "List all cited filenames (deduplicated) in the reference_images array."
)


def patch_embodied_agent():
    """Replace EmbodiedInspectionAgent.async_inspect to bypass the reformatting_agent.

    The reformatting_agent expects pure-JSON responses and uses output_type=
    InspectionQuestionAnswerResponse, which fails with the fine-tuned model's
    prose+JSON output. Instead, we let the SDK's Runner.run() handle the full
    multi-turn tool calling loop natively (scene_graph_interface_tool, oracle_tool)
    and parse the final output ourselves via _extract_inspection_response().

    This requires vLLM to be served with --tool-call-parser gemma4 so that
    Gemma 4's native <|tool_call> tokens are translated into OpenAI-format
    tool_calls that the Agents SDK can execute automatically.
    """

    def _patched_async_inspect(self, messages):
        message_ref = messages

        async def inner():
            assert self.context is not None
            with logfire.span("Embodied Inspection Agent (finetuned)"):
                # Append [img:filename] citation guidance so the eval prompt
                # matches what the fine-tuned model learned during training.
                if _CITATION_SUPPLEMENT not in (self.think_agent.instructions or ""):
                    self.think_agent.instructions = (
                        (self.think_agent.instructions or "") + _CITATION_SUPPLEMENT
                    )

                msgs = message_ref
                if isinstance(msgs, RunResult):
                    msgs = msgs.to_input_list()
                result = await Runner.run(
                    self.think_agent, msgs, context=self.context, max_turns=20
                )
                final = result.final_output
                if not isinstance(final, InspectionQuestionAnswerResponse):
                    result.final_output = _extract_inspection_response(str(final))
                return result

        return inner()

    EmbodiedInspectionAgent.async_inspect = _patched_async_inspect


def patch_process_scene():
    """Patch process_scene so it reads from a per-scene sandbox of symlinks and
    writes results there — never into the source scene directory.

    Why:
      - Most scene dirs in the BridgeEQA dataset are owned by another user and
        not group-writable, so ``run_experiments.process_scene``'s direct writes
        to ``scene_dir / '<agent>_agent_results.json'`` raise PermissionError.
      - Even for writable scenes we don't want to overwrite baseline results.

    How it works: for each scene, we create ``RESULTS_OUTPUT_DIR/_sandbox/<name>``
    containing symlinks to the read-only inputs (``qa_pairs.json``,
    ``scene_graph.json``, ``images/``), then call the original ``process_scene``
    with that sandbox path. All writes land in the sandbox. After it returns,
    we copy the real (non-symlink) JSON files into
    ``RESULTS_OUTPUT_DIR/<scene_name>/`` and tear down the sandbox.
    """
    import shutil

    _original_process_scene = run_experiments.process_scene

    READ_ENTRIES = ("qa_pairs.json", "scene_graph.json", "images")

    async def _patched_process_scene(scene_dir, model_name: str, experiment_tracker):
        real_scene = Path(scene_dir)
        scene_name = real_scene.name

        if RESULTS_OUTPUT_DIR is None:
            # Shouldn't happen in normal use, but fall back to in-place.
            await _original_process_scene(real_scene, model_name, experiment_tracker)
            return

        sandbox_root = RESULTS_OUTPUT_DIR / "_sandbox"
        sandbox_root.mkdir(parents=True, exist_ok=True)
        sandbox = sandbox_root / scene_name
        if sandbox.exists():
            shutil.rmtree(sandbox)
        sandbox.mkdir(parents=True)

        # Symlink the read-only inputs. `open()` and `Path.iterdir()` both
        # follow symlinks transparently, so process_scene reads the real files.
        for entry in READ_ENTRIES:
            src = real_scene / entry
            if src.exists():
                (sandbox / entry).symlink_to(src)

        try:
            await _original_process_scene(sandbox, model_name, experiment_tracker)

            out_scene_dir = RESULTS_OUTPUT_DIR / scene_name
            out_scene_dir.mkdir(parents=True, exist_ok=True)

            # Copy every real JSON file that process_scene wrote (the symlinks
            # we created are skipped). This captures all *_agent_results.json
            # plus the aggregated file.
            for f in sandbox.iterdir():
                if f.is_symlink() or not f.is_file() or f.suffix != ".json":
                    continue
                shutil.copy2(f, out_scene_dir / f.name)
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)

    run_experiments.process_scene = _patched_process_scene


def main():
    parser = argparse.ArgumentParser(
        description="Run BridgeEQA experiments with fine-tuned model"
    )
    parser.add_argument(
        "--model_name",
        default="openai/gemma4-bridge",
        help="Model name for litellm routing (default: openai/gemma4-bridge)",
    )
    parser.add_argument(
        "--server_urls",
        default="http://localhost:8001/v1",
        help="Comma-separated list of OpenAI-compatible model server URLs. "
             "Requests are round-robined across them.",
    )
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=200,
        help="Maximum number of scenes to evaluate (default: 200)",
    )
    parser.add_argument(
        "--n_concurrent",
        type=int,
        default=1,
        help="Number of concurrent evaluations (default: 1, local model is the bottleneck)",
    )
    parser.add_argument(
        "--scene_dir",
        default=os.path.join(os.environ.get("BRIDGE_ROOT", "/home/projects"), "pdf_processed_vermont_pages_greater_than_20"),
        help="Path to scene directories",
    )
    args = parser.parse_args()

    model_name = args.model_name
    scene_dirs_path = Path(args.scene_dir)
    output_dir = f"{scene_dirs_path.name}_eval_{Path(model_name).name}_cached_output"
    server_urls = [u.strip() for u in args.server_urls.split(",") if u.strip()]

    print("=" * 60)
    print("BridgeEQA Evaluation — Fine-tuned Model")
    print("=" * 60)
    print(f"Model:        {model_name}")
    print(f"Servers:      {server_urls}")
    print(f"Scene dir:    {scene_dirs_path}")
    print(f"Output dir:   {output_dir}")
    print(f"Max scenes:   {args.max_scenes}")
    print(f"Concurrency:  {args.n_concurrent}")
    print(f"Agent configs: {list(FINETUNE_EVAL_CONFIGS.keys())}")
    print("=" * 60)

    # 1. Monkey-patch AGENT_CONFIGS to only include no-tools configs
    agent_cfg.AGENT_CONFIGS = FINETUNE_EVAL_CONFIGS

    # 2. Monkey-patch S3 functions with local base64 versions
    run_experiments.prepare_images_for_prompt = prepare_images_for_prompt_local
    run_experiments.create_node_and_image_url_mapping = create_node_and_image_url_mapping_local

    # 3. Patch Agent to use LitellmModel, round-robin across the server pool
    patch_agent_model(server_urls, model_name)

    # 3b. Patch embodied agent output parsing (bypass reformatting_agent).
    patch_embodied_agent()

    # 4. Patch process_scene to write results to output_dir, not scene dirs
    global RESULTS_OUTPUT_DIR
    RESULTS_OUTPUT_DIR = Path(output_dir)
    RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    patch_process_scene()

    # The model_name passed here is used for ExperimentTracker logging only
    asyncio.run(
        run_experiments.run_concurrent_evaluations(
            scene_dirs_path=scene_dirs_path,
            output_dir=output_dir,
            n_concurrent=args.n_concurrent,
            max_scenes=args.max_scenes,
            model_name=model_name,
        )
    )


if __name__ == "__main__":
    main()
