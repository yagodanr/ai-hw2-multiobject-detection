# AGENTS.md

UCU AI 2026 computer-vision homework: detect 8 classes (person, bicycle, car, van, truck, tricycle, bus, motor) in VisDrone2019-VID drone clips for a Kaggle competition. The final grade comes from the private leaderboard, the code checkpoints, a student-written report and a one-on-one defence.

## Ground rules

- **The report is the students' own writing.** The course checks reports for AI-generated text. Agents write code, analysis cells and markdown that explains what a cell computes; observations, conclusions and `report.md` stay with the students. Leave `**Observations:** _fill in_` placeholders where a conclusion belongs.
- **Template boundary:** the course template owns `index.html`, `assets/`, `.github/`, `report.md` and `images/`. Our code lives in `src/`, and our notes in `docs/`.
- **Deadline:** around 2026-11-09 (confirm on Kaggle). Any commit after it makes the whole submission late, so finish work early.
- **No leakage:** start from COCO or ImageNet weights only. The test clips come from the public VisDrone benchmark, so VisDrone-pretrained checkpoints are off-limits.
- **Split by clip:** frames of one video are near-duplicates, so every split, fold and comparison keeps whole clips together.

## Team split

| Person A | Person B |
|---|---|
| EDA and preprocessing, two-stage (Faster R-CNN), one-stage anchor-based (RetinaNet), temporal bonus | Anchor-free (Ultralytics YOLO), query-based (RT-DETR), final pipeline (SAHI, WBF, TTA), W&B visualisation bonus |

## Layout

Notebooks are numbered in pipeline order. Reusable code goes in `src/common/` and notebooks import it, so a protocol change is a one-place edit.

| Path | Role |
|---|---|
| `src/00_visualize_clip.ipynb` | ground truth and ignored regions on a clip, rendered to MP4 |
| `src/01_preprocess.ipynb` | EDA, frame step, label cleaning, ignore handling, COCO and YOLO export |
| `src/02_faster_rcnn.ipynb`, `src/04_retinanet.ipynb` | Person A's detectors: experiments, comparison, failure analysis, submission |
| `src/03_visualize_predictions.ipynb` | a trained run's predictions on a clip, rendered to MP4 |
| `src/common/evaluate.py` | **the evaluator: every reported number comes from it** |
| `src/common/engine.py` | shared train/validate loop: `run_experiment(name, cfg, build_fn, protocol, paths, device, family)` |
| `src/common/analysis.py` | failure analysis (per-frame errors, worst frames, miss rates, confusions) |
| `src/common/data.py`, `submission.py`, `frcnn.py` | dataset and augmentation; CSV writer; rebuild a Faster R-CNN run |
| `docs/preprocessing_improvements.md` | ranked backlog of data and inference ideas, with sources |
| `data_processed/`, `runs/<family>/<experiment>/`, `outputs/` | generated, gitignored: processed data, run artefacts (`best.pt`, `history.json`, `summary.json`, `val_dets.json`), videos |

## Protocol

Every model is compared under the same protocol. Changing any item invalidates comparisons with earlier runs.

- **Data:** train on `data_processed/train` (frame step chosen in 01, ≈ 8; ignored regions grey-filled). Validate on the **original** frames of the 7 `val/` clips listed in `data_processed/val/gt_eval_coco.json`. The grey-filled `data_processed/val/images` exist only for Ultralytics' metrics during training.
- **Metric:** `evaluate_detections` reports COCO mAP@[.5:.95], AP50, AP75, AP by size, AR and per-class AP, at up to 500 detections per image.
  - Ignored regions are crowd boxes copied once per class, because pycocotools ignores a crowd region only for its own class.
  - pycocotools' `summarize()` hard-codes 100 detections for mAP, so read the evaluator's numbers, never `summarize()`.
- **Detections:** `{"image_id", "category_id": 1..8, "bbox": [x, y, w, h] in original pixels, "score"}`. Submission classes are 0..7.
- **Inference:** score threshold ≈ 0.001 and 500 detections per image. mAP only ranks boxes, so low-score boxes never hurt.
- **Training:** long side 1280 (torchvision `min_size=720, max_size=1280`), 8 epochs, batch 4, SGD with warm-up then cosine decay, AMP, seed 42. Learning rate per model family: Faster R-CNN 0.005, RetinaNet 0.0025.
- **Experiments:** each changes one thing relative to the baseline, then a `tuned` run combines what helped.

## Working with the user's machine

- **Hardware:** a 14 GB RAM Linux machine with an AMD GPU (ROCm); one epoch takes about 5 min. Data loaders keep images as uint8 until `to_model_input` moves them to the GPU, with 2 workers. More workers, or float images on the CPU, triggered an out-of-memory kill that took VS Code down with it.
- **Editor overwrites:** VS Code saves its open copy over files changed from outside the editor. Ask the user to close a notebook's tab before editing it, then to reload it ("File: Revert File") and restart the kernel. After writing, re-read the file from disk to confirm the change stuck.
- **Testing:** notebooks are verified on CPU with a tiny fixture (a few val frames, random weights, reduced size and epochs) before they reach the user's GPU.

## Status (2026-11-06)

- **Faster R-CNN:** `baseline` done: mAP 0.2145, AP50 0.440, AP_small 0.098. Per-class AP: bus 0.027, motor 0.107 (weakest). mAP plateaus after epoch 3. Experiments `small_anchors`, `more_rois` and `giou_loss` are queued, and `tuned` is still empty.
- **RetinaNet:** notebook ready; no runs yet.
- **Pending fixes in 01:**
  - export a full-frame val ground truth for temporal work;
  - keep boxes under 2 px in `gt_eval`.
- **Kaggle:** a baseline `submission.csv` exists under `runs/faster_rcnn/baseline/`; whether it was uploaded is unknown.

## Next task: `src/05_preprocessing_ablations.ipynb`

Test the ideas in `docs/preprocessing_improvements.md` against the Faster R-CNN baseline, keeping each variant as small as possible.

**Variants** (each a training-time option of the dataset or loader, so no re-export):

| Backlog item | Variant |
|---|---|
| #2 scale matching | cap the long side of training frames at 1920 before the model's resize |
| #1 tiled training | random native-resolution crops (e.g. 640 px) using albumentations' bbox-aware crop; drop empty crops |
| #4 rare classes | `torch.utils.data.WeightedRandomSampler` with repeat-factor weights (LVIS) for frames containing rare classes |
| #10 low-res robustness | `A.Downscale` augmentation; evaluate on val downscaled to 680 px |
| #5 occlusion | drop boxes with `occlusion == 2` from training |

**Inference-only variants** (test resize, val downscaling) reuse one trained model and need no training.

**Search:** use **Optuna**.
- Queue one trial per variant plus the baseline with `study.enqueue_trial`, and add a combination trial last.
- Prune with `MedianPruner` (or `HyperbandPruner`) on per-epoch val mAP. That needs an `on_epoch_end(epoch, metrics)` callback added to `engine.run_experiment`, which calls `trial.report` and `trial.should_prune`.
- Store the study in SQLite under `runs/ablations/`, so a crash keeps finished trials.
- Use a short proxy budget (`train_step=2`, about 4 epochs; mAP plateaus by epoch 3) and say so in the notebook.
- Optuna's default TPE search and Ultralytics `model.tune()` suit continuous hyperparameters with many cheap trials. These variants are a few discrete options at about 20 min each.

**Done when:** each variant has a row (mAP, AP50, AP_small, per-class AP for bus, truck and motor) against the proxy baseline, and the notebook ran end to end on the CPU fixture.

## Skills

- `mattpocock-skills:diagnosing-bugs`: crashes, out-of-memory errors, metrics that disagree.
- `mattpocock-skills:research`: checking library behaviour (Optuna, Ultralytics, torchvision) in primary docs before relying on it.
- `dataviz`: any comparison plot.
- `mattpocock-skills:writing-for-agents`: when editing this file.
