"""Test submissions for a WBF ensemble, with per-family weights, from cached per-model test predictions.

    python ensemble_submission.py --rtdetr-weight 1 1.5 2

1. For every member in runs/pipeline/final_config.json (written by 09), predict all test frames with that member's passes
   (e.g. full + flip, merged with NMS) and cache them in runs/pipeline/test_<member>.pkl. Resumable; a member that is
   already cached is not run again, so adding a new model later only runs that model.
2. Fuse the members with WBF (IoU from final_config.json) for each RT-DETR weight given (YOLO members weigh 1), and
   write runs/pipeline/sub_ens_rtdetr_w<w>.csv.
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from time import time

import cv2
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ultralytics import RTDETR, YOLO

from common.pipeline import TileCfg, combine, fuse, predict_passes, to_submission_rows
from common.submission import write_submission

RUNS_ROOT = Path(os.environ.get("RUNS_ROOT", "../runs"))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "../multiple-object-video-detection-hw-2-ucu-ai-2026"))
CANDIDATES = {   # same names as 09: (class, run folder, full-frame imgsz)
    "yolo_low_scale_960": (YOLO, "yolo/low_scale_960", 960),
    "yolo_baseline_1280": (YOLO, "yolo/baseline", 1280),
    "yolo_final_960": (YOLO, "yolo/final_960", 960),
    "yolo_final_1280": (YOLO, "yolo/final_1280", 1280),
    "yolo_m_960": (YOLO, "yolo/m_960", 960),
    "yolo_low_scale_960_s1": (YOLO, "yolo/low_scale_960_s1", 960),
    "yolo_tuned_1280": (YOLO, "yolo/tuned", 1280),
    "yolo_m_1280": (YOLO, "yolo/m_1280", 1280),
    "yolo_step4_960": (YOLO, "yolo/step4_960", 960),
    "yolo_step2_960": (YOLO, "yolo/step2_960", 960),
    "yolo_mixup_960": (YOLO, "yolo/mixup_960", 960),
    "rtdetr_960": (RTDETR, "rtdetr/imgsz_960", 960),
    "rtdetr_final_960": (RTDETR, "rtdetr/final_960", 960),
    "rtdetr_final_960_s025": (RTDETR, "rtdetr/final_960_s025", 960),
    "rtdetr_final_1280": (RTDETR, "rtdetr/final_1280", 1280),
    "rtdetr_x_960": (RTDETR, "rtdetr/final_x_960", 960),
    "rtdetr_960_mixup": (RTDETR, "rtdetr/final_960_s025_mixup", 960),
    "rtdetr_960_seed1": (RTDETR, "rtdetr/final_960_s025_seed1", 960),
    "rtdetr_960_step4": (RTDETR, "rtdetr/final_960_s025_step4", 960),
    "rtdetr_960_step2": (RTDETR, "rtdetr/final_960_s025_step2", 960),
}

p = argparse.ArgumentParser()
p.add_argument("--rtdetr-weight", type=float, nargs="+", default=[1.0])
p.add_argument("--config", type=Path, default=RUNS_ROOT / "pipeline" / "final_config.json")
p.add_argument("--limit", type=int, help="only the first N test frames (smoke test)")
a = p.parse_args()

cfg = json.loads(a.config.read_text())
members, wbf_iou = cfg["members"], cfg["wbf_iou"]
test = pd.read_csv(DATA_ROOT / "test_images.csv")
if a.limit:
    test = test.head(a.limit)
device = 0 if torch.cuda.is_available() else "cpu"
out_dir = RUNS_ROOT / "pipeline"
out_dir.mkdir(parents=True, exist_ok=True)

preds = {}
for name, mcfg in members.items():
    cls, run, imgsz = CANDIDATES[name]
    passes = tuple(mcfg["passes"])
    f = out_dir / f"test_{name}.pkl"
    cache = pickle.loads(f.read_bytes()) if f.exists() else {}
    todo = [(k, fn) for k, fn in zip(test["image_id"], test["file_name"]) if k not in cache]
    if todo:
        model, t0 = cls(str(RUNS_ROOT / run / "best.pt")), time()
        print(f"{name}: predicting {len(todo)} test frames ({'+'.join(passes)})", flush=True)
        for n, (key, fname) in enumerate(todo, 1):
            img = cv2.imread(str(DATA_ROOT / fname))
            entry = predict_passes(model, img, imgsz, TileCfg(), device=device, passes=passes)
            cache[key] = (img.shape[:2], combine(entry, passes, tile_max_side=mcfg.get("tile_max_side")))
            if n % 500 == 0 or n == len(todo):
                f.write_bytes(pickle.dumps(cache))
                print(f"  {n}/{len(todo)}, {(time() - t0) / n:.2f} s/frame", flush=True)
        del model
        torch.cuda.empty_cache()
    preds[name] = cache

names = list(members)
for w in a.rtdetr_weight:
    weights = [w if n.startswith("rtdetr") else 1.0 for n in names]
    rows = {k: to_submission_rows(fuse([preds[n][k][1] for n in names], preds[names[0]][k][0], weights=weights, iou=wbf_iou))
            for k in test["image_id"]}
    out = out_dir / f"sub_ens_rtdetr_w{w:g}.csv"
    sub = write_submission(rows, DATA_ROOT / "test_images.csv", out)
    n_boxes = sub.PredictionString.fillna("").str.split().str.len().fillna(0) // 6
    assert n_boxes.max() <= 500
    print(f"wrote {out}: weights {dict(zip(names, weights))}", flush=True)
