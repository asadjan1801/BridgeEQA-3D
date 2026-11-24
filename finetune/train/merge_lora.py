"""
Merge LoRA adapter into the base Gemma 4 31B-it model.

vLLM 0.19 does not support LoRA for Gemma4ForConditionalGeneration,
so we merge the adapter weights into the base model and save the
result as a full model that vLLM can serve directly.

Launch via SLURM (from the repository root):
    SFT_VARIANT=mc sbatch slurm/run_merge_lora.sh
    SFT_VARIANT=sc sbatch slurm/run_merge_lora.sh

Environment variables (optional): SFT_VARIANT ("mc" default / "sc"), FINETUNE_DIR,
ADAPTER_PATH and OUTPUT_PATH (override the variant defaults directly).
"""

import os
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

# Adapters that were merged and evaluated in the paper.
_VARIANTS = {
    "mc": ("gemma4-31b-bridge-v3all/manual-checkpoint-10400", "gemma4-31b-bridge-v3all-merged"),
    "sc": ("gemma4-31b-bridge-v3mixed/checkpoint-2700", "gemma4-31b-bridge-v3mixed-merged"),
}
SFT_VARIANT = os.environ.get("SFT_VARIANT", "mc").lower()
if SFT_VARIANT not in _VARIANTS:
    raise ValueError(f"SFT_VARIANT must be one of {sorted(_VARIANTS)}, got {SFT_VARIANT!r}")
_CKPT_DIR = Path(os.environ.get("FINETUNE_DIR", "/home/projects/bridge_finetune")) / "checkpoints"

MODEL_NAME = "google/gemma-4-31B-it"
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", str(_CKPT_DIR / _VARIANTS[SFT_VARIANT][0]))
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", str(_CKPT_DIR / _VARIANTS[SFT_VARIANT][1]))


def main():
    print("=" * 60)
    print("Merging LoRA adapter into base model")
    print(f"  Base model:  {MODEL_NAME}")
    print(f"  Adapter:     {ADAPTER_PATH}")
    print(f"  Output:      {OUTPUT_PATH}")
    print("=" * 60)

    # Load base model in bf16 (no quantization — we need full precision for merge)
    print("\nLoading base model in bf16...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",  # merge on CPU to avoid GPU OOM
    )

    # Load and merge LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    # Save merged model
    print(f"\nSaving merged model to {OUTPUT_PATH}...")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    model.save_pretrained(OUTPUT_PATH, safe_serialization=True)

    # Save processor/tokenizer too
    print("Saving processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    processor.save_pretrained(OUTPUT_PATH)

    print(f"\nMerge complete! Model saved to: {OUTPUT_PATH}")
    print(f"Size: {sum(f.stat().st_size for f in Path(OUTPUT_PATH).rglob('*') if f.is_file()) / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
