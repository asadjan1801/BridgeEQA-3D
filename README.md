# BridgeEQA-3D

Code accompanying **"BridgeEQA-3D: Context-Aware Scene Graphs and an Open Vision-Language Model for Bridge Inspection"** by Asad Ur Rahman, Subin Varghese, and Vedhus Hoskere (submitted to the *Journal of Computing in Civil Engineering*).

BridgeEQA-3D extends the BridgeEQA embodied bridge-inspection question-answering framework ([Varghese et al. 2026](https://drags99.github.io/bridge-eqa/)) with a three-layer context-aware scene graph and a fine-tuned open-weight vision-language model (Gemma 4 31B-it + QLoRA). This repository contains the code used to:

1. build the training corpus from 3,220 Vermont bridge-inspection reports with a locally served Gemma 4 model;
2. audit the corpus and convert it into Single-Context (SC) and Multi-Context (MC) supervised fine-tuning (SFT) datasets;
3. fine-tune, merge, and serve the SC and MC models;
4. run the population-scale evaluation on the 200-scene BridgeEQA test set and compute the metrics in Table 2.

It also includes the per-configuration summary metrics behind Table 2. The Vermont inspection data (source reports, processed scenes, QA pairs, and SFT corpora) is not distributed in this repository.

## Repository layout

```
agent_eval/            Evaluation harness and PDF-processing pipeline (uv project, Python 3.12)
  src/skit/              Modified BridgeEQA agent/evaluation package (see "Attribution")
  process_pdfs_local.py  PDF -> image-layer scene graph + QA pairs, via a local vLLM server
  run_eval_finetuned.py  5-configuration BridgeEQA evaluation of a served fine-tune
  run_experiments.py     BridgeEQA experiment runner (used by the scripts above)
  run_experiments_gemini31fl.py      Gemini 3.1 Flash-Lite baseline evaluation
  gemini3x_thought_signature_patches.py  Required for multi-turn tool calls with Gemini 3.x
  analyze_experiments.py, run_analysis_judge31fl.py   LLM/VLM-judge analysis (Answer Correctness)
  compute_condition_rating_accuracy.py               Exact-match / within-±1 rating accuracy
finetune/              SFT data preparation and training (conda env "bridge_ft", Python 3.11)
  audit_data.py          Scene-level data-quality audit (drops incomplete scenes)
  convert_dataset_v3.py  Builds the SC (--variant_mode random) and MC (--variant_mode all) corpora
  dataset_stats.py       Corpus statistics
  audit_output/, stats_output/   Audit flags and corpus summary produced for the paper
  train/gemma4_sft.py    QLoRA fine-tuning (DeepSpeed ZeRO-2)
  train/merge_lora.py    Merges a LoRA adapter into the base model for vLLM serving
slurm/                 SLURM job scripts for every GPU/cluster step
splits/                Scene IDs for the train / validation / test splits and audit-dropped scenes
results/               Summary metrics behind Table 2
environment/           Exact package versions of the fine-tuning environment
```

## How the code maps to the paper

| Paper section | Code |
|---|---|
| Bridge Inspection Report Corpus (audit, splits, Table 1) | `finetune/audit_data.py`, `finetune/dataset_stats.py`, `splits/` |
| PDF Processing Pipeline (4 stages) | `agent_eval/process_pdfs_local.py`, `agent_eval/src/skit/pipelines/local_pdf_qa_gen.py`, `agent_eval/src/skit/pipelines/pdf_pipeline.py`, `slurm/serve_gemma4_base.sh`, `slurm/run_process_pdfs.sh` |
| SFT Data Formulation; Single- and Multi-Context | `finetune/convert_dataset_v3.py` |
| Model and Fine-Tuning Configuration | `finetune/train/gemma4_sft.py`, `finetune/train/ds_zero2.json`, `slurm/run_gemma4_sft.sh`, `finetune/train/merge_lora.py`, `slurm/run_merge_lora.sh`, `slurm/serve_gemma4.sh` |
| Population-Scale Fine-Tune Evaluation (Table 2) | `agent_eval/run_eval_finetuned.py`, `slurm/run_eval_gemma4.sh`, `agent_eval/run_experiments_gemini31fl.py`, `agent_eval/run_analysis_judge31fl.py`, `agent_eval/compute_condition_rating_accuracy.py`, `results/table2_population_scale/` |
| Context-Aware Scene-Graph Evaluation (Texas case study, Tables 3–5) | Not yet included in this repository; see "Data and model availability". |

## Table 2 results

`results/table2_population_scale/` holds the summary metrics reported in Table 2 for Gemini 3.1 Flash-Lite, the SC fine-tune, and the MC fine-tune: exact-match and within-±1 condition-rating accuracy, and LLM-judge Answer Correctness, for each of the five agent configurations. These are the files `run_analysis_judge31fl.py` and `compute_condition_rating_accuracy.py` produce at the end of the pipeline below. See [`results/README.md`](results/README.md) for details.

## Environments

Two separate environments were used.

**Evaluation harness and PDF processing** (`agent_eval/`, Python 3.12, managed by [uv](https://docs.astral.sh/uv/)). `pyproject.toml` and `uv.lock` are exactly as used for the paper. The lock resolves for Linux x86_64 with CUDA 12.8 wheels.

```bash
cd agent_eval
uv sync
cp ../.env.example .env     # then fill in GEMINI_API_KEY (judge and Gemini baseline)
```

**Fine-tuning and model serving** (`finetune/`, `slurm/`, Python 3.11.15). The conda environment is in `environment/finetune_conda_environment.yml`, and `environment/finetune_pip_freeze.txt` pins the key versions: torch 2.10.0, transformers 5.5.4, peft 0.18.1, trl 0.24.0, deepspeed 0.18.9, bitsandbytes 0.49.2, accelerate 1.13.0, and vLLM 0.19.0.

```bash
conda env create -f environment/finetune_conda_environment.yml   # creates "bridge_ft"
```

Base model: `google/gemma-4-31B-it` from Hugging Face, revision `145dc2508c480a64b47242f160d286cff94a2343` (see `environment/gemma4_base_model_revision.txt`). Access requires accepting the Gemma license on Hugging Face.

## Running the full pipeline

All scripts read data from a workspace root, `BRIDGE_ROOT` (default `/home/projects`), with this layout:

```
$BRIDGE_ROOT/
  vermont_reports_less_than_30_MB/                # 3,220 source PDFs
  pdf_processed_ALL_vermont_reports/              # processed scenes (output of step 1)
  pdf_processed_vermont_pages_greater_than_20/    # BridgeEQA released test scenes (first 200 = test set)
  bridge_finetune/data_v3_mixed/, data_v3_all/    # SFT corpora (output of step 2)
  bridge_finetune/checkpoints/                    # adapters and merged models (steps 3-4)
  .hf_cache/                                      # Hugging Face cache
```

Submit SLURM jobs from the repository root, after `mkdir -p logs`. Each script lists its environment variables in its header. Add `--partition` and `--account` for your cluster.

1. **Process PDFs** (2× L40S for serving, plus a CPU node):
   `sbatch slurm/serve_gemma4_base.sh`, then `VLLM_SERVER_URL=http://<gpu-node>:8001/v1 sbatch slurm/run_process_pdfs.sh`
2. **Audit and build the SFT corpora** (`conda activate bridge_ft`):
   ```bash
   python finetune/audit_data.py
   python finetune/convert_dataset_v3.py --variant_mode random --output_dir $BRIDGE_ROOT/bridge_finetune/data_v3_mixed   # SC
   python finetune/convert_dataset_v3.py --variant_mode all    --output_dir $BRIDGE_ROOT/bridge_finetune/data_v3_all     # MC
   ```
   The split is fixed by seed 42. `splits/` lists the resulting scene IDs: 2,795 train, 147 validation, 200 test, and 81 audit-dropped scenes (3 of which are also test scenes, so 78 are removed from the training pool).
3. **Fine-tune** (4× L40S): `SFT_VARIANT=mc sbatch slurm/run_gemma4_sft.sh` and `SFT_VARIANT=sc sbatch slurm/run_gemma4_sft.sh`
4. **Merge and serve** (CPU, then 2× L40S): `SFT_VARIANT=mc sbatch slurm/run_merge_lora.sh`, then `SFT_VARIANT=mc sbatch slurm/serve_gemma4.sh` (same for `sc`)
5. **Evaluate** on the 200 test scenes: `VLLM_SERVER_URL=http://<gpu-node>:8002/v1 sbatch slurm/run_eval_gemma4.sh`. For the Gemini 3.1 Flash-Lite baseline, run `cd agent_eval && uv run python run_experiments_gemini31fl.py`.
6. **Score**: `cd agent_eval && uv run python run_analysis_judge31fl.py && uv run python compute_condition_rating_accuracy.py`. Outputs are written to `agent_eval/graph_outputs_*/` and can be compared with `results/table2_population_scale/`.

## Reproducibility notes

- **Training runs as evaluated.** Both runs used the hyperparameters in `gemma4_sft.py` (LoRA r=32, alpha=64, learning rate 2e-5 with cosine schedule, effective batch 32, 4,096-token context, seed 42). The **MC** model was trained on a 2-epoch schedule, and adapter `manual-checkpoint-10400` (about 1.97 epochs) was merged and evaluated. The **SC** model was launched on a 3-epoch schedule, and adapter `checkpoint-2700` (about 1.54 epochs) was merged and evaluated. The SC run used an earlier revision of `gemma4_sft.py` that saved adapters through the standard Hugging Face Trainer; set `MANUAL_SAVE_EVERY=100` to get a step-2700 adapter with this revision.
- **Checkpoint saving.** On the authors' cluster, the standard Hugging Face Trainer/DeepSpeed checkpoint save deadlocked the NFS-backed GPU node. `gemma4_sft.py` therefore saves LoRA adapters from rank 0 only, writing to node-local disk and copying to shared storage in the background.
- **Serving.** Both fine-tunes are served under the name `gemma4-bridge`, so their evaluation outputs land in the same directory. The SC output directory was renamed with the suffix `_v3mixed_BACKUP`, which is the path `run_analysis_judge31fl.py` expects. The vLLM flags `--reasoning-parser gemma4 --tool-call-parser gemma4 --enable-auto-tool-choice` are required for the embodied-agent configurations.
- **Judge.** Answer Correctness uses `gemini-3.1-flash-lite` as an LLM judge (`agent_eval/src/skit/evaluation/judges.py`) with the provider's default temperature. Re-running the judge gives slightly different means (about ±0.02). The condition-rating metrics are computed programmatically and are deterministic given the agent outputs.
- **Gemini 3.x tool calls** need `gemini3x_thought_signature_patches.py`, imported before `agents` and `litellm`. `run_experiments_gemini31fl.py` does this already.
- **Absolute paths in the SFT JSONL.** `convert_dataset_v3.py` writes absolute image paths. Generate the corpora on the filesystem where training will run, or rewrite the paths.

## Data and model availability

| Resource | Location |
|---|---|
| Source inspection reports | Vermont Agency of Transportation, <https://vtrans.vermont.gov/docs/bridge-inspections> |
| 200-scene test set and processed scene graphs (BridgeEQA release) | <https://huggingface.co/datasets/hoskerelab/bridge-eqa> |
| Original BridgeEQA agent code | <https://github.com/drags99/bridge-eqa> |
| Fine-tuned SC and MC models (LoRA adapters and merged weights) | **TBA** |
| Texas case-study context-aware scene graphs, damage and component segmentations, and code | **TBA** |

The processed Vermont training and validation scenes, QA pairs, and SFT corpora are not distributed in this repository. They can be regenerated from the publicly available reports with steps 1–2 above (`splits/` lists the exact scenes in each split), or requested from the corresponding author.

## Attribution

`agent_eval/` is derived from the BridgeEQA code base by Subin Varghese and colleagues ([drags99/bridge-eqa](https://github.com/drags99/bridge-eqa), MIT License; see [`agent_eval/LICENSE-BridgeEQA`](agent_eval/LICENSE-BridgeEQA)). The version here adds the local-model PDF pipeline (`pipelines/local_pdf_qa_gen.py`), the fine-tuned-model evaluation driver, the Gemini 3.x patches, and changes to the evaluation and judge modules used for this study. The Python package keeps its original internal name, `skit`.

## Citation

If you use this code, please cite the paper (citation details will be added on publication) and BridgeEQA:

> Varghese, S., Gao, J., Rahman, A. U., and Hoskere, V. 2026. "BridgeEQA: Virtual embodied agents for real bridge inspections." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 8163–8173.

See also [`CITATION.cff`](CITATION.cff).

## Acknowledgments

This research was supported by the Texas Department of Transportation under Project No. 0-7181. The authors acknowledge the University of Houston for GPU compute resources and the Vermont Agency of Transportation for making the bridge-inspection reports publicly available.

## License

See [`LICENSE`](LICENSE).
