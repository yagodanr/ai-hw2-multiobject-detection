"""Rebuild a trained RetinaNet from its run folder (summary.json + best.pt), for inference outside 04_retinanet.ipynb."""
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator

from .data import CLASSES


def octave_sizes(base_sizes):
    """RetinaNet's 3 anchor scales per level: s, s*2^(1/3), s*2^(2/3) (same as 04_retinanet.ipynb)."""
    return tuple(tuple(int(round(s * 2 ** (k / 3))) for k in range(3)) for s in base_sizes)


def build_retinanet(cfg, min_size, max_size):
    """Same architecture as build_model in 04_retinanet.ipynb, without pretrained weights (they come from best.pt).
    The focal-loss settings only matter for training, so they are not needed here."""
    anchors = AnchorGenerator(octave_sizes(cfg["anchor_base"]), (tuple(cfg["aspect_ratios"]),) * len(cfg["anchor_base"]))
    model = retinanet_resnet50_fpn(
        weights=None, weights_backbone=None,
        min_size=min_size, max_size=max_size,
        anchor_generator=anchors,
        fg_iou_thresh=cfg["fg_iou_thresh"], bg_iou_thresh=cfg["bg_iou_thresh"],
        score_thresh=cfg["score_thresh"], nms_thresh=cfg["nms_thresh"],
        detections_per_img=cfg["detections_per_img"], topk_candidates=cfg["topk_candidates"],
    )
    head = model.head.classification_head
    num_classes = len(CLASSES) + 1
    head.cls_logits = nn.Conv2d(256, head.num_anchors * num_classes, kernel_size=3, padding=1)
    nn.init.constant_(head.cls_logits.bias, -math.log((1 - 0.01) / 0.01))
    head.num_classes = num_classes
    return model


def load_retinanet_run(run_dir, device):
    """Load best.pt of a finished RetinaNet experiment, using the settings stored in its summary.json."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    p = summary["protocol"]
    model = build_retinanet(summary["config"], p["min_size"], p["max_size"])
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    return model.to(device).eval()
