"""Predictions on the video frames next to each val frame (for temporal rescoring in 09).

Val is scored on every 8th frame, but the clips have every frame, and the test set has every frame. To test
"a box that also appears in the neighbouring frames is more likely real" on val the way it would work on test,
predict the frames at offsets -2, -1, +1, +2 around every val frame with the same inference as the test CSVs
(full frame + flip, merged with NMS).

    python temporal_cache.py            # ~5-10 min on the RTX 4060
Writes ../runs/pipeline/neighbours_<model>.pkl: {val image_id: {offset: (xyxy, score, cls0)}}.
"""
import json
import os
import pickle
import sys
from pathlib import Path
from time import time

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ultralytics import RTDETR, YOLO

from common.pipeline import combine, predict_passes, TileCfg

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "../multiple-object-video-detection-hw-2-ucu-ai-2026"))
PROC_ROOT = Path(os.environ.get("PROC_ROOT", "../data_processed"))
RUNS_ROOT = Path(os.environ.get("RUNS_ROOT", "../runs"))
OFFSETS = (-2, -1, 1, 2)
MODELS = {   # name: (class, run, imgsz) -- the two models whose test predictions already exist as CSVs
    "rtdetr_final_960_s025": (RTDETR, RUNS_ROOT / "rtdetr/final_960_s025", 960),
    "yolo_baseline_1280": (YOLO, RUNS_ROOT / "yolo/baseline", 1280),
}
device = 0 if torch.cuda.is_available() else "cpu"
images = json.load(open(PROC_ROOT / "val" / "gt_eval_coco.json"))["images"]
if os.environ.get("LIMIT"):
    images = images[: int(os.environ["LIMIT"])]

for name, (cls, run, imgsz) in MODELS.items():
    out_file = RUNS_ROOT / "pipeline" / f"neighbours_{name}.pkl"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    model, cache, t0 = cls(str(run / "best.pt")), {}, time()
    for n, im in enumerate(images, 1):
        folder = Path(im["file_name"]).parent
        cache[im["id"]] = {}
        for off in OFFSETS:
            path = DATA_ROOT / folder / f"{im['frame_id'] + off:07d}.jpg"
            if not path.exists():
                continue
            img = cv2.imread(str(path))
            entry = predict_passes(model, img, imgsz, TileCfg(), device=device, passes=("full", "flip"))
            cache[im["id"]][off] = combine(entry, ("full", "flip"))
        if n % 50 == 0 or n == len(images):
            print(f"{name}: {n}/{len(images)} frames, {(time() - t0) / n:.2f} s/frame", flush=True)
    out_file.write_bytes(pickle.dumps(cache))
    print("wrote", out_file, flush=True)
