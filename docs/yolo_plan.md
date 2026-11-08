# Plan: anchor-free detector (Ultralytics YOLO), Person B

Rubric item: one-stage anchor-free (1.0). Goal: a YOLO baseline plus ≥ 2 tuned parameters, scored with the shared evaluator so the numbers sit in the same table as Faster R-CNN (0.2171 mAP) and RetinaNet (0.1865).

## 0. Before any code (blockers found on this machine)

| Check | State in `~/Desktop/AI/hw2/ai-hw2-multiobject-detection` | Action |
|---|---|---|
| Repo version | Behind handoff 2: no `05_anchor_kmeans.ipynb`, `AGENTS.md` status dated 11-06, old `.gitignore` | `git pull` |
| `data_processed/` | **Missing** | Run `01_preprocess.ipynb` here, or copy the folder from Person A (faster; also guarantees the same frame step) |
| `ultralytics` | Not in `requirements.txt` | `pip install -U ultralytics` (needs a version with YOLO26), then add it to `requirements.txt` |
| GPU | Unknown for this machine | Decide where YOLO trains (see the question at the end) |
| `data.yaml` `path: .` | Ultralytics resolves a relative `path` against its own `datasets_dir`, not the yaml's folder (verify on first run) | The notebook writes `runs/yolo/data_abs.yaml` with an absolute path at start-up |

## 1. Model choice (verified in Ultralytics docs)

- **YOLO26s** as the baseline (9.5 M params, COCO 48.6 mAP). Faster to iterate than `m`; compare sizes only if time is left.
- YOLO26 has **no DFL** and trains two heads: one-to-many (used with NMS, the default) and one-to-one (NMS-free, `nms=False`, at most 300 boxes per image).
- It adds **STAL** (small-target-aware label assignment), which matters for our small-object bottleneck.
- **`yolo26-p2.yaml`** exists, but has **no pretrained P2 weights**: build from YAML and transfer the matching layers with `.load("yolo26s.pt")`. The new P2 head starts random, so mention this when comparing.
- COCO weights only (`yolo26s.pt`). Never VisDrone checkpoints.

## 2. Files to create

### `src/common/yolo.py` (reusable, so RT-DETR and the final pipeline import the same code)

| Function | Does |
|---|---|
| `val_items(gt_file, data_root)` | `[(image_id, path)]` of the **original** val frames from `gt_eval_coco.json` |
| `predict_coco(model, items, imgsz, conf=0.001, iou=0.7, max_det=500, nms=None, batch=4)` | `model.predict(..., stream=True)` → COCO dicts: `category_id = cls + 1`, xyxy → xywh, original pixels |
| `predict_test(model, test_csv, data_root, **kw)` | `{image_id: [(cls0, score, x, y, w, h)]}` for `common/submission.write_submission` |
| `score_checkpoints(run_dir, ...)` | Scores every `weights/epoch*.pt` with `evaluate_detections` → `history.json`; keeps the best epoch's `val_dets.json` |
| `write_summary(run_dir, name, cfg, protocol, best)` | Same `summary.json` layout as `common/engine.py`, so `load_summaries`, `error_breakdown.py` and `run_queue.py` work unchanged |

Run folder: `runs/yolo/<exp>/` holding Ultralytics' own output (`weights/`, `results.csv`) plus our `summary.json`, `history.json` and `val_dets.json`.

### `src/07_yolo.ipynb`

**Status: implemented and tested on a CPU fixture.** Number 05 is taken and 06 is reserved for the preprocessing ablations, so YOLO is 07 (RT-DETR will be 08). Sections, mirroring 04:

1. **Setup:** the `ON_KAGGLE` path branch, `RUNS_DIR = RUNS_ROOT / "yolo"`, write `data_abs.yaml`.
2. **Protocol:** one `YOLO_PROTOCOL` dict (below).
3. **Experiments:** a `BASE` config plus an `EXPERIMENTS` dict, each changing one thing.
4. **`RUN = [...]` cell:** `for name in RUN: train_and_score(name, {**BASE, **EXPERIMENTS[name]})`. Keep this exact shape so `run_queue.py 07_yolo.ipynb:baseline,imgsz_960,p2` works unattended.
5. **Comparison table:** YOLO runs next to `runs/faster_rcnn/*` and `runs/retinanet/*` via `load_summaries`. Columns: mAP, AP50, AP75, AP_small/medium/large, AR, per-class AP, FPS, minutes.
6. **Inference-only checks on the best run** (no training): NMS vs NMS-free head, and predict at a larger imgsz (1600) than trained.
7. **Error breakdown and failure frames:** `common/analysis` (`error_breakdown`, `miss_and_confusion`, `show_frame`).
8. **Test submission:** `predict_test` → `runs/yolo/<best>/submission.csv`.
9. `**Observations:** _fill in_` placeholders after each section. The text is yours to write, not an agent's.

## 3. Protocol

```python
PROTOCOL = dict(
    epochs=20,                 # same for every YOLO run; check with the timing cell first (target ~1 h per run)
    batch=0.70,                # fraction of GPU memory (AutoBatch); Ultralytics accumulates to a nominal batch of 64
    train=dict(optimizer="auto", seed=42, deterministic=True, amp=True,
               workers=6, cache=False,               # 32 GB RAM laptop, RTX 4060 8 GB
               fliplr=0.5, flipud=0.0, degrees=0.0, copy_paste=0.0, close_mosaic=3,
               save_period=1, plots=False, val=True),
    predict=dict(conf=0.001, iou=0.7, max_det=500, batch=4),
)
```

- **Budget:** match the CNNs' GPU time (8 epochs × ~5 min ≈ 40 min), not their epoch count. YOLO usually needs more epochs. Time the baseline's first epoch, set `E = floor(40 / minutes_per_epoch)`, and use the same `E` for every YOLO experiment.
- **Checkpoint choice:** Ultralytics picks `best.pt` on the grey-filled val. We re-score each epoch on the original frames with our evaluator and report the best epoch from that, the same rule `engine.py` uses.
- **ROCm:** Ultralytics' AMP self-check may fail on ROCm and silently turn AMP off. Check the first lines of the log.

## 4. Experiments (one change each)

| Name | Change | Why |
|---|---|---|
| `baseline` | yolo26s.pt, imgsz 1280 | Same resolution as the CNNs |
| `imgsz_960` | imgsz 960 (train + predict) | Required parameter 1: resolution vs small-object recall |
| `p2` | `yolo26s-p2.yaml` + `.load("yolo26s.pt")`, imgsz 1280 | Required parameter 2: stride-4 head (Faster R-CNN's edge over RetinaNet came from stride 4) |
| `imgsz_1600` | imgsz 1600 | Optional, if `imgsz_960` < baseline; pushes the "resolution is the lever" finding |
| `tuned` | Combine what helped | Same pattern as 02 and 04 |

Total ≈ 4–5 runs × 40 min. Queue them overnight with `run_queue.py` (use `../../.venv/bin/python` under `nohup`, per handoff 2).

## 5. Verification (before the GPU run)

1. **CPU fixture:** 2 epochs, imgsz 320, `fraction=0.02`, 5 val frames. The notebook must run end to end and produce `summary.json`.
2. **Evaluator sanity check:** for one checkpoint, our mAP and Ultralytics' own `val` mAP should be the same order of magnitude. A large gap usually means a class-offset bug (`cls + 1`) or a box-format bug.
3. **Submission check:** classes 0–7, ≤ 500 boxes per row, one row per test image.

## 6. What it feeds next

- `common/yolo.py` is reused by RT-DETR (`ultralytics.RTDETR`, same predict API) and by the final pipeline (SAHI tiles + full frame, WBF).
- The `p2` and imgsz results decide the final pipeline's base model and resolution.

## Sources

- [Ultralytics YOLO26 docs](https://docs.ultralytics.com/models/yolo26/): no DFL, dual heads, `nms=False`, STAL, P2 YAML without pretrained weights
- [Ultralytics end-to-end detection guide](https://docs.ultralytics.com/zh/guides/end2end-detection): NMS vs NMS-free, `max_det`
