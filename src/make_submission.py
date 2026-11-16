"""Test-set submission for one model (optionally with flip TTA), same inference as 09.

    python make_submission.py yolo   ../runs/yolo/baseline           1280 --flip
    python make_submission.py rtdetr ../runs/rtdetr/final_960_s025   960  --flip

Writes ../runs/pipeline/sub_<run>[_flip].csv (or --out).
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ultralytics import RTDETR, YOLO

from common.pipeline import TileCfg, predict_config, to_submission_rows
from common.submission import write_submission

p = argparse.ArgumentParser()
p.add_argument("family", choices=["yolo", "rtdetr"])
p.add_argument("run", type=Path, help="run folder with best.pt")
p.add_argument("imgsz", type=int, help="full-frame input size the model was trained at")
p.add_argument("--flip", action="store_true", help="add the horizontal-flip pass (merged with NMS)")
p.add_argument("--out", type=Path)
p.add_argument("--limit", type=int, help="only the first N test frames (smoke test)")
a = p.parse_args()

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "../multiple-object-video-detection-hw-2-ucu-ai-2026"))
TEST_CSV = DATA_ROOT / "test_images.csv"
out = a.out or Path("../runs/pipeline") / f"sub_{a.run.name}{'_flip' if a.flip else ''}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
device = 0 if torch.cuda.is_available() else "cpu"

model = (YOLO if a.family == "yolo" else RTDETR)(str(a.run / "best.pt"))
cfg = {"m": {"passes": ("full", "flip") if a.flip else ("full",)}}
test = pd.read_csv(TEST_CSV)
if a.limit:
    test = test.head(a.limit)
preds = {}
for k, (key, fname) in enumerate(zip(test["image_id"], test["file_name"]), 1):
    img = cv2.imread(str(DATA_ROOT / fname))
    preds[key] = to_submission_rows(predict_config({"m": (model, a.imgsz)}, img, TileCfg(), cfg, device=device))
    if k % 500 == 0:
        print(f"{k}/{len(test)}", flush=True)
sub = write_submission(preds, TEST_CSV, out)
n_boxes = sub.PredictionString.fillna("").str.split().str.len().fillna(0) // 6
assert n_boxes.max() <= 500
print(f"wrote {out}: {len(sub)} rows, {int((n_boxes > 0).sum())} with detections")
