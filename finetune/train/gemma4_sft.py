"""
Fine-tune Gemma 4 31B-it on bridge inspection QA data.

Hardware : single node, 4× NVIDIA L40S 48GB
Method   : QLoRA 4-bit NF4 + LoRA (r=32, alpha=64) + DeepSpeed ZeRO-2
Dataset  : SFT JSONL produced by finetune/convert_dataset_v3.py
             SFT_VARIANT=mc -> data_v3_all   (Multi-Context,  168,540 train samples)
             SFT_VARIANT=sc -> data_v3_mixed (Single-Context,  56,180 train samples)

Launch (single-node multi-GPU via SLURM, from the repository root):
    SFT_VARIANT=mc sbatch slurm/run_gemma4_sft.sh
    SFT_VARIANT=sc sbatch slurm/run_gemma4_sft.sh

Environment variables (all optional; defaults reproduce the original layout):
    SFT_VARIANT    "mc" (default) or "sc"
    FINETUNE_DIR   workspace holding data_v3_*/ and checkpoints/ (default /home/projects/bridge_finetune)
    SFT_DATA_DIR   override the dataset directory directly
    LOCAL_SCRATCH  node-local disk for in-training checkpoints (default /tmp)
    NUM_EPOCHS     override the epoch count for the selected variant
    MANUAL_SAVE_EVERY  adapter-save cadence in optimizer steps (default 200)
"""

import json
import re
import os
from pathlib import Path
from PIL import Image

import torch
from datasets import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_kbit_training
from safetensors.torch import save_file
from trl import SFTTrainer, SFTConfig

# ── Configuration ────────────────────────────────────────────────────────────

# Variant -> (dataset dir, run name, epoch schedule as launched for the paper runs).
#   mc: 2-epoch schedule; adapter manual-checkpoint-10400 (~1.97 epochs) was merged and evaluated.
#   sc: 3-epoch schedule; adapter checkpoint-2700 (~1.54 epochs) was merged and evaluated. The SC
#       run used an earlier revision of this script that saved through the stock HF Trainer path
#       (hence "checkpoint-", not "manual-checkpoint-"); set MANUAL_SAVE_EVERY=100 to obtain an
#       adapter at step 2700 with this revision.
_VARIANTS = {
    "mc": ("data_v3_all", "gemma4-31b-bridge-v3all", 2),
    "sc": ("data_v3_mixed", "gemma4-31b-bridge-v3mixed", 3),
}
SFT_VARIANT = os.environ.get("SFT_VARIANT", "mc").lower()
if SFT_VARIANT not in _VARIANTS:
    raise ValueError(f"SFT_VARIANT must be one of {sorted(_VARIANTS)}, got {SFT_VARIANT!r}")
_DATA_NAME, _RUN_NAME, _DEFAULT_EPOCHS = _VARIANTS[SFT_VARIANT]

FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", "/home/projects/bridge_finetune"))

MODEL_NAME = "google/gemma-4-31B-it"
FINAL_OUTPUT_DIR = str(FINETUNE_DIR / "checkpoints" / _RUN_NAME)  # shared storage — final destination
OUTPUT_DIR = os.path.join(os.environ.get("LOCAL_SCRATCH", "/tmp"), _RUN_NAME)  # local disk — avoids NFS deadlock
DATA_DIR = Path(os.environ.get("SFT_DATA_DIR", FINETUNE_DIR / _DATA_NAME))
DS_CONFIG = str(Path(__file__).resolve().parent / "ds_zero2.json")

# LoRA config
LORA_RANK = 32
LORA_ALPHA = 64
MAX_SEQ_LENGTH = 4096
MAX_IMAGES_PER_SAMPLE = 6  # limit images to control memory

# Training config
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", _DEFAULT_EPOCHS))
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8  # effective batch = 4 GPUs × 1 × 8 = 32
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
SAVE_STEPS = 50      # frequent saves so we always have a checkpoint to resume from
EVAL_STEPS = 2000    # infrequent eval (eval+save right after each other was crashing)
LOGGING_STEPS = 10
SEED = 42

IMAGE_TOKEN_RE = re.compile(r"<image>")


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_jsonl_as_conversations(jsonl_path: Path) -> list[dict]:
    """Convert our JSONL format to HuggingFace conversation format for Gemma 4.

    Stores image paths (not PIL objects) to avoid OOM on large datasets.
    Images are loaded lazily in the collator.
    """
    samples = []

    with open(jsonl_path) as f:
        for line_num, line in enumerate(f, 1):
            rec = json.loads(line)
            msgs = rec["messages"]
            image_paths = rec.get("images", [])

            system_content = msgs[0]["content"]
            user_content = msgs[1]["content"]
            assistant_content = msgs[2]["content"]

            # Count and strip <image> tokens
            num_image_tokens = len(IMAGE_TOKEN_RE.findall(user_content))
            user_text = IMAGE_TOKEN_RE.sub("", user_content).strip()

            # Prepend system prompt
            user_text = f"{system_content}\n\n{user_text}"

            # Build user content: images first (Gemma 4 expects images before text)
            user_content_list = []
            valid_image_paths = []

            if image_paths and num_image_tokens > 0:
                for img_path in image_paths[:min(num_image_tokens, MAX_IMAGES_PER_SAMPLE)]:
                    p = Path(img_path)
                    if p.exists():
                        user_content_list.append({"type": "image"})
                        valid_image_paths.append(str(p))

            user_content_list.append({"type": "text", "text": user_text})

            sample = {
                "messages": [
                    {"role": "user", "content": user_content_list},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": assistant_content}
                    ]},
                ],
                "image_paths": valid_image_paths,  # lazy — loaded in collator
            }
            samples.append(sample)

            if line_num % 2000 == 0:
                print(f"  Loaded {line_num} samples...")

    print(f"  Total: {len(samples)} samples")
    return samples


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    if is_main:
        print("=" * 60)
        print("Gemma 4 31B-it Bridge Inspection Fine-tuning")
        print(f"  Model: {MODEL_NAME}")
        print(f"  Variant: {SFT_VARIANT} ({DATA_DIR})")
        print(f"  LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}")
        print(f"  DeepSpeed config: {DS_CONFIG}")
        print(f"  GPUs: {torch.cuda.device_count()}")
        print("=" * 60)

    # ── Load processor ───────────────────────────────────────────────────────
    if is_main:
        print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # ── Load model ───────────────────────────────────────────────────────────
    if is_main:
        print("Loading model (QLoRA 4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        device_map={"": local_rank},
    )
    model = prepare_model_for_kbit_training(model)

    # Freeze vision encoder
    if hasattr(model, "vision_tower"):
        for param in model.vision_tower.parameters():
            param.requires_grad = False
    elif hasattr(model, "vision_model"):
        for param in model.vision_model.parameters():
            param.requires_grad = False

    # ── Apply LoRA ───────────────────────────────────────────────────────────
    if is_main:
        print("Applying LoRA...")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        target_modules="all-linear",
        exclude_modules=".*vision_tower.*",
        task_type="CAUSAL_LM",
        modules_to_save=None,
    )
    model = get_peft_model(model, lora_config)

    if is_main:
        model.print_trainable_parameters()

    # Enable gradient checkpointing
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ── Load datasets ────────────────────────────────────────────────────────
    if is_main:
        print("\nLoading training dataset...")
    train_data = load_jsonl_as_conversations(DATA_DIR / "train.jsonl")
    if is_main:
        print(f"Training samples: {len(train_data)}")

    if is_main:
        print("Loading validation dataset...")
    val_data = load_jsonl_as_conversations(DATA_DIR / "val.jsonl")
    if is_main:
        print(f"Validation samples: {len(val_data)}")

    # ── Collator ─────────────────────────────────────────────────────────────
    def collate_fn(examples):
        """Custom collator: apply chat template, process images, tokenize."""
        texts = []
        all_images = []
        has_images = False

        for ex in examples:
            text = processor.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
            # Lazy-load images from paths
            img_paths = ex.get("image_paths", [])
            if img_paths:
                loaded = []
                for p in img_paths:
                    try:
                        loaded.append(Image.open(p).convert("RGB"))
                    except Exception:
                        pass
                if loaded:
                    all_images.append(loaded)
                    has_images = True
                else:
                    all_images.append([])
            else:
                all_images.append([])

        # Tokenize with processor (handles image tokens + text)
        proc_kwargs = dict(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )
        if has_images:
            proc_kwargs["images"] = all_images
        batch = processor(**proc_kwargs)

        # Set labels = input_ids (shifted internally by the model)
        labels = batch["input_ids"].clone()

        # Mask padding tokens in labels
        labels[labels == processor.tokenizer.pad_token_id] = -100

        # Mask user/system tokens — only train on assistant responses
        # Find the assistant turn marker and mask everything before it
        assistant_marker = processor.tokenizer.encode(
            "<|turn>model\n", add_special_tokens=False
        )
        end_turn_marker = processor.tokenizer.encode(
            "<turn|>", add_special_tokens=False
        )

        for i in range(labels.shape[0]):
            ids = batch["input_ids"][i].tolist()
            # Find last occurrence of assistant turn marker
            # Mask everything up to and including the marker
            marker_len = len(assistant_marker)
            last_assistant_pos = -1
            for j in range(len(ids) - marker_len + 1):
                if ids[j:j + marker_len] == assistant_marker:
                    last_assistant_pos = j + marker_len

            if last_assistant_pos > 0:
                labels[i, :last_assistant_pos] = -100

        batch["labels"] = labels
        return batch

    # ── Trainer ──────────────────────────────────────────────────────────────
    if is_main:
        print("\nSetting up trainer...")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=LOGGING_STEPS,
        save_strategy="no",     # disable intermediate saves — stock HF saves deadlocked the NFS-backed GPU node
        eval_strategy="no",     # also disable eval (no checkpoints to compare against anyway)
        save_total_limit=2,
        save_only_model=True,
        seed=SEED,
        report_to="none",
        deepspeed=DS_CONFIG,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    # Manual rank-0-only LoRA save callback. Bypasses HF Trainer's _save_checkpoint
    # entirely (no DeepSpeed barrier, no rng_state, no trainer_state, no NCCL gather)
    # — the multi-rank coordination in Trainer's save was what hung the GPU node.
    MANUAL_SAVE_EVERY = int(os.environ.get("MANUAL_SAVE_EVERY", 200))  # cadence
    NFS_RSYNC_AFTER_SAVE = True  # also push each checkpoint to NFS in a non-blocking subprocess

    class ManualLoraSaveCallback(TrainerCallback):
        def on_step_end(self, args, state, control, model=None, **kwargs):
            if state.global_step == 0 or state.global_step % MANUAL_SAVE_EVERY != 0:
                return control
            if not state.is_world_process_zero:
                return control
            try:
                save_dir = os.path.join(args.output_dir, f"manual-checkpoint-{state.global_step}")
                os.makedirs(save_dir, exist_ok=True)
                # Pull only LoRA params from the wrapped model and write a single safetensors file
                state_dict = get_peft_model_state_dict(model)
                save_file(state_dict, os.path.join(save_dir, "adapter_model.safetensors"))
                # Tiny adapter_config.json so it's loadable
                if hasattr(model, "peft_config"):
                    cfg = list(model.peft_config.values())[0]
                    cfg.save_pretrained(save_dir)
                print(f"[ManualSave] step={state.global_step} -> {save_dir}", flush=True)

                # Non-blocking rsync to NFS so checkpoint survives node crashes / /tmp wipe.
                # Uses Popen (no wait) — runs in background, can't hang the trainer process.
                if NFS_RSYNC_AFTER_SAVE:
                    nfs_dir = os.path.join(FINAL_OUTPUT_DIR, f"manual-checkpoint-{state.global_step}")
                    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)
                    import subprocess
                    subprocess.Popen(
                        ["rsync", "-a", f"{save_dir}/", f"{nfs_dir}/"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    print(f"[ManualSave] background rsync -> {nfs_dir}", flush=True)
            except Exception as e:
                print(f"[ManualSave] FAILED at step={state.global_step}: {e}", flush=True)
            return control

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collate_fn,
        processing_class=processor,
        callbacks=[ManualLoraSaveCallback()],
    )

    # ── Train ────────────────────────────────────────────────────────────────
    if is_main:
        print("\n" + "=" * 60)
        print("Starting training...")
        print(f"  Train samples: {len(train_data)}")
        print(f"  Val samples: {len(val_data)}")
        print(f"  Epochs: {NUM_EPOCHS}")
        n_gpus = torch.cuda.device_count()
        print(f"  Effective batch: {PER_DEVICE_BATCH_SIZE} × {GRADIENT_ACCUMULATION_STEPS} × {n_gpus} GPUs = {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * n_gpus}")
        print(f"  Learning rate: {LEARNING_RATE}")
        print("=" * 60)

    # Auto-resume disabled — incompatible with save_only_model=True + DeepSpeed.
    # On SLURM requeue, training restarts from step 0; saved adapters from prior
    # attempts can still be used for downstream eval / manual continuation.
    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────────
    if is_main:
        print("\nSaving final LoRA adapter (local)...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))

    if is_main:
        processor.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
        print(f"Model saved locally to: {OUTPUT_DIR}/final")

        # Copy to NFS at the end (single rsync, after all training I/O is done)
        import subprocess
        os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)
        print(f"Copying checkpoints from {OUTPUT_DIR} -> {FINAL_OUTPUT_DIR} ...")
        subprocess.run(
            ["rsync", "-a", "--delete", f"{OUTPUT_DIR}/", f"{FINAL_OUTPUT_DIR}/"],
            check=False,
        )
        print("Training complete!")


if __name__ == "__main__":
    main()
