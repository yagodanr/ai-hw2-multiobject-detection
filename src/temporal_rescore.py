"""Temporal rescoring of a submission: a box that the support model also finds in the neighbouring video frames
(same class, IoU >= 0.5) gets a slightly higher score.

    new score = (1 - a) * score + a * mean over neighbours (t-1, t+1) of the best matching support score

Validated on val (frames +-1 around each scored frame, support = RT-DETR final_960_s025 + flip): a = 0.1 raised the
7-model fusion from 0.2970 to 0.3002 mAP, by +0.0015 on each half of the val clips; stronger settings (a >= 0.4) hurt.

    python temporal_rescore.py ../runs/pipeline/submission.csv ../runs/pipeline/sub_final_960_s025_flip.csv
Writes <submission>_temporal.csv next to the input.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.ops import box_iou

A, IOU_THR, OFFSETS = 0.1, 0.5, (-1, 1)


def parse(s):
    if not isinstance(s, str) or not s.strip():
        return np.zeros((0, 6), np.float32)
    return np.asarray(s.split(), np.float32).reshape(-1, 6)       # cls score x y w h


def xyxy(a):
    return np.c_[a[:, 2], a[:, 3], a[:, 2] + a[:, 4], a[:, 3] + a[:, 5]]


sub_path, sup_path = Path(sys.argv[1]), Path(sys.argv[2])
sub = pd.read_csv(sub_path)
sup = dict(zip(*pd.read_csv(sup_path)[["image_id", "PredictionString"]].values.T))
sup = {k: parse(v) for k, v in sup.items()}

out = []
for key, ps in zip(sub.image_id, sub.PredictionString):
    boxes = parse(ps)
    clip, frame = key.rsplit("_", 1)
    vals = []
    for off in OFFSETS:
        nb = sup.get(f"{clip}_{int(frame) + off:07d}")
        if nb is None:
            continue
        v = np.zeros(len(boxes), np.float32)
        if len(nb) and len(boxes):
            iou = box_iou(torch.from_numpy(xyxy(boxes)), torch.from_numpy(xyxy(nb))).numpy()
            iou[boxes[:, None, 0] != nb[None, :, 0]] = 0
            v = np.where(iou >= IOU_THR, nb[None, :, 1], 0).max(1)
        vals.append(v)
    if vals and len(boxes):
        boxes[:, 1] = (1 - A) * boxes[:, 1] + A * np.mean(vals, 0)
        boxes = boxes[np.argsort(-boxes[:, 1])]
    out.append(" ".join(f"{int(b[0])} {b[1]:.4f} {b[2]:.1f} {b[3]:.1f} {b[4]:.1f} {b[5]:.1f}" for b in boxes))

sub["PredictionString"] = out
dst = sub_path.with_name(sub_path.stem + "_temporal.csv")
sub.to_csv(dst, index=False)
print(f"wrote {dst}: {len(sub)} rows")
