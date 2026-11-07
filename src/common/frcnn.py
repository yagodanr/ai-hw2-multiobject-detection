"""Rebuild a trained Faster R-CNN from its run folder (summary.json + best.pt)."""
import json
from pathlib import Path

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .data import CLASSES


def build_faster_rcnn(cfg, min_size, max_size):
    """Same architecture as in 02_faster_rcnn.ipynb, without pretrained weights (they are loaded from best.pt)."""
    anchors = AnchorGenerator(sizes=tuple(tuple(s) for s in cfg["anchor_sizes"]),
                              aspect_ratios=(tuple(cfg["aspect_ratios"]),) * len(cfg["anchor_sizes"]))
    model = fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None,
        min_size=min_size, max_size=max_size,
        rpn_anchor_generator=anchors,
        rpn_pre_nms_top_n_test=cfg["rpn_pre_nms_top_n_test"], rpn_post_nms_top_n_test=cfg["rpn_post_nms_top_n_test"],
        box_detections_per_img=cfg["box_detections_per_img"], box_score_thresh=cfg["box_score_thresh"],
        box_nms_thresh=cfg["box_nms_thresh"],
    )
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features,
                                                      len(CLASSES) + 1)
    return model


def load_faster_rcnn_run(run_dir, device):
    """Load best.pt of a finished experiment, using the settings stored in its summary.json."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    p = summary["protocol"]
    model = build_faster_rcnn(summary["config"], p["min_size"], p["max_size"])
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    return model.to(device).eval()
