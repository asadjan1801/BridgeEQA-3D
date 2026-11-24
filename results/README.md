# Results

## `table2_population_scale/`

Summary metrics for Table 2 of the paper: the population-scale evaluation on the 200-scene BridgeEQA test set (1,193 questions with a numeric reference rating per agent configuration). There is one folder per model:

| Folder | Model |
|---|---|
| `gemini-3.1-flash-lite/` | Untuned Gemini 3.1 Flash-Lite baseline (G3.1FL) |
| `gemma4-bridge-sc/` | Single-Context fine-tune of Gemma 4 31B-it (SC) |
| `gemma4-bridge-mc/` | Multi-Context fine-tune of Gemma 4 31B-it (MC) |

Each folder contains:

| File | Contents |
|---|---|
| `*_aggregated_agent_results_condition_rating_accuracy.csv` | Exact-match and within-±1 accuracy per agent configuration (Table 2, rows 1–12), written by `agent_eval/compute_condition_rating_accuracy.py`. |
| `*_answer_correctness_statistics.csv` | LLM-judge Answer Correctness per agent configuration. The `Mean` column gives Table 2, rows 13–18. Written by `agent_eval/run_analysis_judge31fl.py`. |
| `*_answer_correctness_statistics_vlm.csv` | VLM-judge image-citation relevance (200 sampled questions per configuration, seed 42). Not reported in the paper. |

Agent configuration names map to the paper as follows:

| Name in files | Paper (Table 2) |
|---|---|
| `embodied_no_tools_images` / VLM w/ Images Only | VLM (images) |
| `embodied_no_tools_sg_context` / VLM w/ SG Only | VLM (scene graph) |
| `embodied_no_tools_both` / VLM w/ Images + SG | VLM (both) |
| `embodied_sg_oracle_sg_context` / Embodied Agent w/ SG Only | Embodied agent (scene graph) |
| `embodied_sg_oracle_both` / Embodied Agent w/ Images + SG | Embodied agent (both) |

The judge for both judge-based metrics was `gemini-3.1-flash-lite`.
