# Scene splits

Scene IDs are the stems of the Vermont inspection-report PDFs (for example `BridgeInspRpt-ARLINGTON-00016`).

| File | Scenes | Description |
|---|---|---|
| `test_200_scenes.txt` | 200 | Held-out test set: the first 200 scenes of the BridgeEQA release, used unchanged. Never seen during fine-tuning. |
| `train_scenes.txt` | 2,795 | Fine-tuning training split (56,180 QA pairs). |
| `val_scenes.txt` | 147 | Validation split (2,952 QA pairs). |
| `audit_dropped_scenes.txt` | 81 | Scenes dropped by `finetune/audit_data.py` because QA extraction produced no QA pairs (`no_qa`; see `finetune/audit_output/summary.md`). 3 of them are also test scenes, so 78 are removed from the training pool. |

Together these cover all 3,220 reports: 2,795 + 147 + 200 + 78 = 3,220. The train/validation split is the 95/5 split produced by `finetune/convert_dataset_v3.py` with seed 42. The Single-Context and Multi-Context corpora use identical splits.
