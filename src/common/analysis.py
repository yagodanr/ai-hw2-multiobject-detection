"""Failure analysis shared by all detectors: per-frame error counts, worst frames, miss rates, class confusions.

Detections with score >= score_thr are matched greedily (highest score first) to ground truth of the same
class at IoU >= 0.5. Every detection is then one of:
  tp           matched an object of its class
  duplicate    overlaps (IoU >= 0.5) an already-matched object of its class
  wrong_class  overlaps (IoU >= 0.5) an object of another class
  ignored      lies mostly (>= 50%) inside an ignored region, so it is not scored
  fp           anything else
Objects without a matching detection are "missed".
"""
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.ops import box_iou

from .data import CLASSES


def size_bucket(boxes_xyxy):
    """COCO size bucket of each box (by area)."""
    b = np.asarray(boxes_xyxy, float).reshape(-1, 4)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return np.where(area < 32**2, "small", np.where(area < 96**2, "medium", "large"))


def _xyxy(boxes_xywh):
    b = np.asarray(boxes_xywh, float).reshape(-1, 4)
    return np.c_[b[:, :2], b[:, :2] + b[:, 2:]]


class GroundTruth:
    """Index of gt_eval_coco.json: objects, ignored regions and image info per image id."""

    def __init__(self, gt_file):
        coco = json.loads(Path(gt_file).read_text())
        self.images = {im["id"]: im for im in coco["images"]}
        self.objects, ignore = defaultdict(list), defaultdict(set)
        for a in coco["annotations"]:
            if a["iscrowd"]:
                ignore[a["image_id"]].add(tuple(a["bbox"]))   # one copy per class in the file; deduplicate
            else:
                self.objects[a["image_id"]].append(a)
        self.ignore = {k: _xyxy(list(v)) for k, v in ignore.items()}


def group_by_image(dets):
    by_img = defaultdict(list)
    for d in dets:
        by_img[d["image_id"]].append(d)
    return by_img


def frame_errors(gt, dets_by_img, img_id, score_thr):
    objs = gt.objects.get(img_id, [])
    gt_boxes = _xyxy([a["bbox"] for a in objs])
    gt_cls = np.array([a["category_id"] for a in objs], int)
    ds = sorted([d for d in dets_by_img.get(img_id, []) if d["score"] >= score_thr], key=lambda d: -d["score"])
    det_boxes = _xyxy([d["bbox"] for d in ds])
    det_cls = np.array([d["category_id"] for d in ds], int)
    iou = (box_iou(torch.tensor(det_boxes), torch.tensor(gt_boxes)).numpy()
           if len(ds) and len(objs) else np.zeros((len(ds), len(objs))))

    ign = gt.ignore.get(img_id, np.zeros((0, 4)))
    matched = np.zeros(len(objs), bool)
    status = []
    for i in range(len(ds)):
        same = np.where((det_cls[i] == gt_cls) & ~matched, iou[i], 0) if len(objs) else np.array([])
        if len(same) and same.max() >= 0.5:
            matched[same.argmax()] = True
            status.append("tp")
        elif len(objs) and np.where(det_cls[i] == gt_cls, iou[i], 0).max() >= 0.5:
            status.append("duplicate")        # a second box on an object of its class that is already matched
        elif len(objs) and iou[i].max() >= 0.5:
            status.append("wrong_class")
        else:
            inside = 0.0
            if len(ign):
                b = det_boxes[i]
                w = (np.minimum(b[2], ign[:, 2]) - np.maximum(b[0], ign[:, 0])).clip(0)
                h = (np.minimum(b[3], ign[:, 3]) - np.maximum(b[1], ign[:, 1])).clip(0)
                inside = (w * h).sum() / max((b[2] - b[0]) * (b[3] - b[1]), 1e-9)
            status.append("ignored" if inside >= 0.5 else "fp")
    return dict(gt_boxes=gt_boxes, gt_cls=gt_cls, matched=matched,
                det_boxes=det_boxes, det_cls=det_cls, status=status)


def error_table(gt, dets, score_thr):
    """One row per val frame: objects, missed, false positives, wrong class."""
    by_img = group_by_image(dets)
    rows = []
    for img_id, info in gt.images.items():
        e = frame_errors(gt, by_img, img_id, score_thr)
        rows.append({"image_id": img_id, "video": info["video"], "n_gt": len(e["gt_cls"]),
                     "missed": int((~e["matched"]).sum()), "fp": e["status"].count("fp"),
                     "duplicate": e["status"].count("duplicate"), "wrong_class": e["status"].count("wrong_class")})
    return pd.DataFrame(rows)


def miss_and_confusion(gt, dets, score_thr):
    """(miss rate per class x size, confusion counts true -> predicted)."""
    by_img = group_by_image(dets)
    miss_rows, conf_rows = [], []
    for img_id in gt.images:
        e = frame_errors(gt, by_img, img_id, score_thr)
        for box, c, ok in zip(e["gt_boxes"], e["gt_cls"], e["matched"]):
            miss_rows.append({"cls": CLASSES[c - 1], "size": size_bucket(box)[0], "missed": not ok})
        for box, c, s in zip(e["det_boxes"], e["det_cls"], e["status"]):
            if s == "wrong_class":
                j = box_iou(torch.tensor(box[None]), torch.tensor(e["gt_boxes"])).argmax().item()
                conf_rows.append({"true": CLASSES[e["gt_cls"][j] - 1], "predicted": CLASSES[c - 1]})
    miss = pd.DataFrame(miss_rows)
    pivot = miss.pivot_table(index="cls", columns="size", values="missed", aggfunc="mean").reindex(CLASSES)
    conf = pd.DataFrame(conf_rows)
    confusion = pd.crosstab(conf["true"], conf["predicted"]) if len(conf) else pd.DataFrame()
    return pivot, confusion


def show_frame(gt, dets, img_id, data_root, score_thr, title=""):
    """Green = correct, red = missed object, yellow = false positive, magenta = wrong class (with its label)."""
    e = frame_errors(gt, group_by_image(dets), img_id, score_thr)
    info = gt.images[img_id]
    img = cv2.imread(str(Path(data_root) / info["file_name"]))
    for box, ok in zip(e["gt_boxes"], e["matched"]):
        if not ok:
            cv2.rectangle(img, tuple(map(int, box[:2])), tuple(map(int, box[2:])), (0, 0, 255), 2)
    colors = {"tp": (0, 200, 0), "fp": (0, 255, 255), "wrong_class": (255, 0, 255)}
    for box, c, s in zip(e["det_boxes"], e["det_cls"], e["status"]):
        if s in colors:
            cv2.rectangle(img, tuple(map(int, box[:2])), tuple(map(int, box[2:])), colors[s], 1)
            if s == "wrong_class":
                cv2.putText(img, CLASSES[c - 1], tuple(map(int, box[:2])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[s], 1)
    plt.figure(figsize=(16, 9))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"{title} | {info['video']} frame {info['frame_id']}")
    plt.show()


def error_breakdown(gt, dets, score_thr=0.3):
    """TIDE-style error types (Bolya et al., 2020) for detections with score >= score_thr.

    Each detection is assigned one type, in score order per image:
      correct    IoU >= 0.5 with an unmatched object of its class
      duplicate  IoU >= 0.5 with an already-matched object of its class
      class      IoU >= 0.5 with an object of another class
      loc        0.1 <= IoU < 0.5 with an object of its class (box too far off)
      class+loc  0.1 <= IoU < 0.5 with an object of another class only
      ignored    mostly inside an ignored region (not scored)
      background no object nearby
    Objects without a correct detection are "missed". Returns counts, missed objects by size, and the mean
    IoU of correct detections (box tightness).
    """
    by_img = group_by_image(dets)
    counts = defaultdict(int)
    tp_iou, missed_size, total_size = [], defaultdict(int), defaultdict(int)
    for img_id in gt.images:
        objs = gt.objects.get(img_id, [])
        g = _xyxy([a["bbox"] for a in objs])
        gc = np.array([a["category_id"] for a in objs], int)
        ds = sorted([d for d in by_img.get(img_id, []) if d["score"] >= score_thr], key=lambda d: -d["score"])
        b = _xyxy([d["bbox"] for d in ds])
        dc = np.array([d["category_id"] for d in ds], int)
        iou = box_iou(torch.tensor(b), torch.tensor(g)).numpy() if len(ds) and len(objs) else np.zeros((len(ds), len(objs)))
        ign = gt.ignore.get(img_id, np.zeros((0, 4)))
        matched = np.zeros(len(objs), bool)
        for i in range(len(ds)):
            same = (dc[i] == gc) if len(objs) else np.zeros(0, bool)
            free = same & ~matched
            if len(objs) and free.any() and iou[i][free].max() >= 0.5:
                j = np.where(free)[0][iou[i][free].argmax()]
                matched[j] = True
                tp_iou.append(iou[i, j])
                counts["correct"] += 1
            elif len(objs) and same.any() and iou[i][same].max() >= 0.5:
                counts["duplicate"] += 1
            elif len(objs) and (~same).any() and iou[i][~same].max() >= 0.5:
                counts["class"] += 1
            elif len(objs) and same.any() and iou[i][same].max() >= 0.1:
                counts["loc"] += 1
            elif len(objs) and iou[i].max() >= 0.1:
                counts["class+loc"] += 1
            else:
                inside = 0.0
                if len(ign):
                    w = (np.minimum(b[i, 2], ign[:, 2]) - np.maximum(b[i, 0], ign[:, 0])).clip(0)
                    h = (np.minimum(b[i, 3], ign[:, 3]) - np.maximum(b[i, 1], ign[:, 1])).clip(0)
                    inside = (w * h).sum() / max((b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1]), 1e-9)
                counts["ignored" if inside >= 0.5 else "background"] += 1
        for box, ok in zip(g, matched):
            s = size_bucket(box)[0]
            total_size[s] += 1
            if not ok:
                missed_size[s] += 1
                counts["missed"] += 1
    out = dict(counts)
    out["mean IoU of correct"] = float(np.mean(tp_iou)) if tp_iou else float("nan")
    for s in ["small", "medium", "large"]:
        out[f"missed {s} %"] = missed_size[s] / max(total_size[s], 1)
    return out
