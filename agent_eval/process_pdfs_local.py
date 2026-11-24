"""
Process all Vermont bridge inspection PDFs using local Gemma 4 31B model.

Prerequisites:
  - vLLM server running (bridge_finetune/serve_gemma4_base.sh)
  - Server URL reachable at $VLLM_SERVER_URL

Usage:
  LOGFIRE_SEND_TO_LOGFIRE=false uv run python process_pdfs_local.py
"""
import os
import sys
import json
import time
import asyncio
import signal
from pathlib import Path
from typing import Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("LOGFIRE_SEND_TO_LOGFIRE", "false")

import logfire

BRIDGE_ROOT = Path(os.environ.get("BRIDGE_ROOT", "/home/projects"))
INPUT_DIR = Path(os.environ.get("PDF_INPUT_DIR", BRIDGE_ROOT / "vermont_reports_less_than_30_MB"))
OUTPUT_DIR = Path(os.environ.get("PDF_OUTPUT_DIR", BRIDGE_ROOT / "pdf_processed_ALL_vermont_reports"))
SERVER_URL = os.environ.get("VLLM_SERVER_URL", "http://localhost:8001/v1")
MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "gemma4-base")
MIN_IMAGE_WIDTH = 200
MIN_IMAGE_HEIGHT = 200
USE_CACHE = True


async def process_single_pdf(pdf_path: Path) -> Dict[str, Any]:
    """Process one PDF with the local model pipeline."""
    from skit.pipelines.local_pdf_qa_gen import configure
    from skit.pipelines.pdf_pipeline import local_pdf_pipeline

    configure(base_url=SERVER_URL, model_name=MODEL_NAME)

    scene_id = pdf_path.stem
    result = {
        "pdf_path": str(pdf_path),
        "scene_id": scene_id,
        "status": "pending",
        "error": None,
        "processing_time": 0,
        "nodes_count": 0,
    }

    # Check cache
    scene_dir = OUTPUT_DIR / scene_id
    sg_exists = (scene_dir / "scene_graph.json").exists()
    qa_exists = (scene_dir / "qa_pairs.json").exists()
    mappings_ok = False
    mappings_file = scene_dir / "image_reference_mappings.json"
    if mappings_file.exists():
        try:
            with open(mappings_file) as f:
                mappings_ok = bool(json.load(f))
        except Exception:
            pass

    if sg_exists and qa_exists and mappings_ok and USE_CACHE:
        result["status"] = "cached"
        return result

    start = time.time()
    try:
        scene_graph = await local_pdf_pipeline(
            pdf_path=str(pdf_path),
            scene_id=scene_id,
            output_folder=str(OUTPUT_DIR),
            min_image_width=MIN_IMAGE_WIDTH,
            min_image_height=MIN_IMAGE_HEIGHT,
            use_cache=USE_CACHE,
        )
        if scene_graph:
            result["status"] = "success"
            result["nodes_count"] = len(scene_graph.nodes)
        else:
            result["status"] = "failed"
            result["error"] = "No scene graph generated"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logfire.error(f"Error processing {pdf_path.name}: {e}", _exc_info=True)

    result["processing_time"] = time.time() - start
    return result


def process_worker(pdf_path: Path) -> Dict[str, Any]:
    """Subprocess wrapper — runs the async function in its own event loop."""
    return asyncio.run(process_single_pdf(pdf_path))


def log(msg: str):
    """Print with flush so SLURM logs update in real time."""
    print(msg, flush=True)


def main():
    MAX_WORKERS = 2  # Limited by GPU memory / concurrent vLLM requests

    # Gather all PDFs
    pdfs = sorted(INPUT_DIR.glob("*.pdf"))
    log(f"Found {len(pdfs)} PDFs in {INPUT_DIR}")

    # Filter already-processed
    def needs_processing(pdf: Path) -> bool:
        sid = pdf.stem
        d = OUTPUT_DIR / sid
        if not (d / "scene_graph.json").exists():
            return True
        if not (d / "qa_pairs.json").exists():
            return True
        mf = d / "image_reference_mappings.json"
        if not mf.exists():
            return True
        try:
            with open(mf) as f:
                return not bool(json.load(f))
        except Exception:
            return True

    to_process = [p for p in pdfs if needs_processing(p)]
    log(f"Already complete: {len(pdfs) - len(to_process)}")
    log(f"Will process: {len(to_process)}")

    if not to_process:
        log("Nothing to do.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    shutdown = False
    def handle_signal(sig, frame):
        nonlocal shutdown
        if not shutdown:
            shutdown = True
            log("\nShutdown requested — finishing running jobs...")
        else:
            sys.exit(1)
    signal.signal(signal.SIGINT, handle_signal)

    completed = successful = failed = cached = 0
    total = len(to_process)

    # Process in batches
    batch_size = 10
    for batch_start in range(0, total, batch_size):
        if shutdown:
            break
        batch = to_process[batch_start:batch_start + batch_size]
        log(f"\nBatch {batch_start//batch_size + 1}: processing {len(batch)} PDFs "
            f"({total - batch_start - len(batch)} remaining)")

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_worker, p): p for p in batch}
            for future in as_completed(futures):
                pdf = futures[future]
                completed += 1
                try:
                    r = future.result(timeout=600)
                    if r["status"] == "success":
                        successful += 1
                        log(f"[{completed}/{total}] OK: {pdf.name} "
                            f"({r['nodes_count']} nodes, {r['processing_time']:.0f}s)")
                    elif r["status"] == "cached":
                        cached += 1
                        log(f"[{completed}/{total}] CACHED: {pdf.name}")
                    else:
                        failed += 1
                        log(f"[{completed}/{total}] FAIL: {pdf.name} — {r.get('error', '?')}")
                except Exception as e:
                    failed += 1
                    log(f"[{completed}/{total}] ERROR: {pdf.name} — {e}")

    log(f"\n{'='*60}")
    log(f"Done. Processed: {completed}/{total}")
    log(f"  Success: {successful}  Cached: {cached}  Failed: {failed}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
