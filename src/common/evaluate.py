"""COCO-style evaluation shared by every model.

Input: the ground truth `val/gt_eval_coco.json` written by 01_preprocess.ipynb (ignored regions as
iscrowd=1 for every class) and a list of detections:
    {"image_id": int, "category_id": 1..8, "bbox": [x, y, w, h], "score": float}
in pixel coordinates of the ORIGINAL image.

All metrics use up to `max_dets` (500, the submission cap) detections per image.
"""
import contextlib
import io
import time

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .data import CLASSES, to_model_input

MAX_DETS = 500


def _quiet(fn, *args, **kwargs):
    """Run a pycocotools call without its console output."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def evaluate_detections(gt_file, detections, max_dets=MAX_DETS, image_ids=None):
    """Return a dict with mAP, AP50, AP75, AP_small/medium/large, AR and per-class AP.

    `image_ids` restricts scoring to a subset of images (e.g. a quick partial validation).
    """
    gt = _quiet(COCO, str(gt_file))
    keys = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR"]
    if len(detections) == 0:
        return {k: 0.0 for k in keys} | {f"AP_{c}": 0.0 for c in CLASSES}

    dt = _quiet(gt.loadRes, list(detections))
    E = COCOeval(gt, dt, "bbox")
    E.params.maxDets = [1, 100, max_dets]
    if image_ids is not None:
        E.params.imgIds = sorted(int(i) for i in image_ids)
    _quiet(E.evaluate)
    _quiet(E.accumulate)

    # pycocotools' summarize() hard-codes 100 detections for mAP, so read the arrays directly.
    prec = E.eval["precision"]  # [IoU thresholds, recall points, classes, area ranges, maxDets]
    rec = E.eval["recall"]      # [IoU thresholds, classes, area ranges, maxDets]
    area = {lbl: i for i, lbl in enumerate(E.params.areaRngLbl)}
    iou50 = int(np.argmin(np.abs(E.params.iouThrs - 0.5)))
    iou75 = int(np.argmin(np.abs(E.params.iouThrs - 0.75)))

    def mean_valid(x):
        x = x[x > -1]
        return float(x.mean()) if x.size else float("nan")

    m = {
        "mAP": mean_valid(prec[:, :, :, area["all"], -1]),
        "AP50": mean_valid(prec[iou50, :, :, area["all"], -1]),
        "AP75": mean_valid(prec[iou75, :, :, area["all"], -1]),
        "AP_small": mean_valid(prec[:, :, :, area["small"], -1]),
        "AP_medium": mean_valid(prec[:, :, :, area["medium"], -1]),
        "AP_large": mean_valid(prec[:, :, :, area["large"], -1]),
        "AR": mean_valid(rec[:, :, area["all"], -1]),
    }
    for k, cat_id in enumerate(E.params.catIds):
        m[f"AP_{CLASSES[cat_id - 1]}"] = mean_valid(prec[:, :, k, area["all"], -1])
    return m


def to_coco_detections(outputs, image_ids):
    """Convert torchvision outputs (xyxy boxes, labels 1..8) into COCO detection dicts."""
    dets = []
    for out, img_id in zip(outputs, image_ids):
        boxes = out["boxes"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()
        labels = out["labels"].detach().cpu().numpy()
        for (x1, y1, x2, y2), s, c in zip(boxes, scores, labels):
            if c < 1:   # 0 = background slot (RetinaNet can emit it with a near-zero score)
                continue
            dets.append({"image_id": int(img_id), "category_id": int(c),
                         "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)})
    return dets


@torch.no_grad()
def measure_fps(model, images, device, warmup=5):
    """Images per second for single-image inference (batch size 1), including the model's own resize."""
    model.eval()
    for img in images[:warmup]:
        model(to_model_input([img], device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for img in images:
        model(to_model_input([img], device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return len(images) / (time.perf_counter() - start)
