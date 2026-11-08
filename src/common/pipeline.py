"""Final-pipeline inference: tiled (SAHI-style) prediction, flip TTA, and multi-model fusion.

Works with any Ultralytics model (YOLO, RT-DETR). All boxes are in original-image pixels.

    boxes = predict_image(model, img, imgsz=960, tiles=TileCfg(640, 0.2), flip=True)   # one model
    fused = fuse([boxes_yolo, boxes_detr], img.shape, weights=[2, 1])                   # several models (WBF)

`Boxes` is a tuple (xyxy [N,4] float32, score [N], cls0 [N] int).
"""
from dataclasses import dataclass

import numpy as np
import torch
from torchvision.ops import batched_nms

from .evaluate import MAX_DETS

PASSES = ("full", "flip", "tiles")


@dataclass
class TileCfg:
    size: int = 640          # tile side in original pixels
    overlap: float = 0.2     # fraction of the tile shared with the neighbour
    imgsz: int = 640         # model input size for a tile (640 = no upscaling of a 640 tile)
    conf: float = 0.01       # tiles produce many boxes; drop the very lowest before merging
    min_side: int = 1000     # frames whose long side is below this get no tiles (full frame is enough)


def tile_windows(h, w, size, overlap):
    """[(x0, y0, x1, y1)] covering the image; the last row/column is shifted inwards to stay inside."""
    stride = max(1, int(size * (1 - overlap)))

    def starts(n):
        if n <= size:
            return [0]
        s = list(range(0, n - size, stride))
        return s + [n - size]

    return [(x, y, min(x + size, w), min(y + size, h)) for y in starts(h) for x in starts(w)]


def _to_numpy(results):
    out = []
    for r in results:
        b = r.boxes
        out.append((b.xyxy.cpu().numpy().astype(np.float32), b.conf.cpu().numpy().astype(np.float32),
                    b.cls.cpu().numpy().astype(int)))
    return out


def _empty():
    return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)


def _concat(parts):
    parts = [p for p in parts if len(p[0])]
    if not parts:
        return _empty()
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


def nms_merge(boxes, iou=0.6, max_det=MAX_DETS):
    """Class-wise NMS over boxes from several passes (tiles, full frame, flip); keep the top max_det."""
    xyxy, score, cls = boxes
    if not len(xyxy):
        return boxes
    keep = batched_nms(torch.from_numpy(xyxy), torch.from_numpy(score), torch.from_numpy(cls), iou).numpy()
    keep = keep[:max_det]                     # batched_nms returns indices sorted by score
    return xyxy[keep], score[keep], cls[keep]


def predict_image(model, img, imgsz, conf=0.001, iou=0.7, max_det=MAX_DETS, tiles=None, flip=False,
                  merge_iou=0.6, device=None, batch=8):
    """All passes for one BGR image (full frame, optional tiles, optional horizontal flip), merged with NMS."""
    h, w = img.shape[:2]
    kw = dict(conf=conf, iou=iou, max_det=max_det, verbose=False)
    if device is not None:
        kw["device"] = device
    parts = []

    def full_pass(im):
        return _to_numpy(model.predict(im, imgsz=imgsz, **kw))[0]

    parts.append(full_pass(img))
    if flip:
        xyxy, s, c = full_pass(np.ascontiguousarray(img[:, ::-1]))
        xyxy = xyxy.copy()
        xyxy[:, [0, 2]] = w - xyxy[:, [2, 0]]
        parts.append((xyxy, s, c))

    if tiles is not None and max(h, w) >= tiles.min_side:
        wins = tile_windows(h, w, tiles.size, tiles.overlap)
        crops = [img[y0:y1, x0:x1] for x0, y0, x1, y1 in wins]
        tkw = {**kw, "conf": max(conf, tiles.conf)}
        for i in range(0, len(crops), batch):
            res = _to_numpy(model.predict(crops[i:i + batch], imgsz=tiles.imgsz, **tkw))
            for (x0, y0, _, _), (xyxy, s, c) in zip(wins[i:i + batch], res):
                if len(xyxy):
                    xyxy = xyxy + np.array([x0, y0, x0, y0], np.float32)
                    parts.append((xyxy, s, c))

    merged = _concat(parts)
    return merged if len(parts) == 1 else nms_merge(merged, merge_iou, max_det)


def fuse(per_model, shape, weights=None, iou=0.55, skip=0.001, max_det=MAX_DETS, top_k=1500):
    """Weighted box fusion across models (ensemble-boxes). Each model's boxes are cut to its top_k first,
    because WBF is quadratic-ish in Python and the low-score tail does not change mAP's ranking much."""
    from ensemble_boxes import weighted_boxes_fusion

    h, w = shape[:2]
    norm = np.array([w, h, w, h], np.float32)
    bl, sl, ll = [], [], []
    for xyxy, s, c in per_model:
        order = np.argsort(-s)[:top_k]
        bl.append(np.clip(xyxy[order] / norm, 0, 1).tolist())
        sl.append(s[order].tolist())
        ll.append(c[order].tolist())
    import warnings
    with warnings.catch_warnings():               # boxes clipped to zero height at the frame edge: harmless
        warnings.simplefilter("ignore")
        b, s, c = weighted_boxes_fusion(bl, sl, ll, weights=weights, iou_thr=iou, skip_box_thr=skip)
    if not len(b):
        return _empty()
    order = np.argsort(-s)[:max_det]
    return (np.asarray(b, np.float32)[order] * norm, np.asarray(s, np.float32)[order],
            np.asarray(c).astype(int)[order])


def to_coco(img_id, boxes):
    xyxy, s, c = boxes
    return [{"image_id": int(img_id), "category_id": int(k) + 1,
             "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(v)}
            for (x1, y1, x2, y2), v, k in zip(xyxy, s, c)]


def to_submission_rows(boxes):
    xyxy, s, c = boxes
    return [(int(k), float(v), float(x1), float(y1), float(x2 - x1), float(y2 - y1))
            for (x1, y1, x2, y2), v, k in zip(xyxy, s, c)]


# ---------------------------------------------------------------- cached passes (for ablations)


def predict_passes(model, img, imgsz, tiles, conf=0.001, iou=0.7, max_det=MAX_DETS, device=None, batch=8,
                   passes=PASSES):
    """The chosen passes separately, un-merged: {"full": Boxes, "flip": Boxes, "tiles": Boxes}."""
    h, w = img.shape[:2]
    kw = dict(conf=conf, iou=iou, max_det=max_det, verbose=False)
    if device is not None:
        kw["device"] = device
    out = {}
    if "full" in passes:
        out["full"] = _to_numpy(model.predict(img, imgsz=imgsz, **kw))[0]
    if "flip" in passes:
        xyxy, s, c = _to_numpy(model.predict(np.ascontiguousarray(img[:, ::-1]), imgsz=imgsz, **kw))[0]
        xyxy = xyxy.copy()
        xyxy[:, [0, 2]] = w - xyxy[:, [2, 0]]
        out["flip"] = (xyxy, s, c)
    if "tiles" not in passes:
        return out
    parts = []
    if max(h, w) >= tiles.min_side:
        wins = tile_windows(h, w, tiles.size, tiles.overlap)
        crops = [img[y0:y1, x0:x1] for x0, y0, x1, y1 in wins]
        tkw = {**kw, "conf": max(conf, tiles.conf)}
        for i in range(0, len(crops), batch):
            res = _to_numpy(model.predict(crops[i:i + batch], imgsz=tiles.imgsz, **tkw))
            for (x0, y0, _, _), (bx, bs, bc) in zip(wins[i:i + batch], res):
                if len(bx):
                    parts.append((bx + np.array([x0, y0, x0, y0], np.float32), bs, bc))
    out["tiles"] = _concat(parts)
    return out


def cache_passes(model, items, imgsz, tiles, out_file, **kw):
    """Run predict_passes over [(image_id, path)] and save {image_id: passes} to an .npz-like pickle."""
    import pickle
    import cv2
    from pathlib import Path
    from time import time

    out_file = Path(out_file)
    cache = {}
    if out_file.exists():
        cache = pickle.loads(out_file.read_bytes())
    t0, todo = time(), [(i, p) for i, p in items if i not in cache]
    for n, (img_id, path) in enumerate(todo, 1):
        img = cv2.imread(str(path))
        cache[img_id] = {"shape": img.shape[:2], **predict_passes(model, img, imgsz, tiles, **kw)}
        if n % 25 == 0 or n == len(todo):
            out_file.write_bytes(pickle.dumps(cache))           # resumable
            print(f"  {out_file.name}: {n}/{len(todo)} images, {(time() - t0) / n:.1f} s/image", flush=True)
    return cache


def small_only(boxes, max_side):
    """Keep boxes whose longer side is below max_side pixels (tiles add small objects; the full frame keeps large ones)."""
    xyxy, s, c = boxes
    keep = np.maximum(xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]) < max_side
    return xyxy[keep], s[keep], c[keep]


def combine(entry, passes=("full",), merge_iou=0.6, max_det=MAX_DETS, tile_max_side=None):
    """Merge the chosen cached passes of one image with class-wise NMS.
    tile_max_side: if set, only tile boxes smaller than this (pixels) are used."""
    parts = [small_only(entry[p], tile_max_side) if p == "tiles" and tile_max_side else entry[p] for p in passes]
    merged = _concat(parts)
    if len(passes) == 1:
        xyxy, s, c = merged
        order = np.argsort(-s)[:max_det]
        return xyxy[order], s[order], c[order]
    return nms_merge(merged, merge_iou, max_det)


def predict_config(members, img, tiles, member_cfg, wbf_iou=0.55, weights=None, device=None):
    """One image through the chosen final configuration.

    members: {name: (model, imgsz)}; member_cfg: {name: {"passes": (...), "tile_max_side": int|None}}.
    One member: NMS-merged passes. Several: each member merged, then fused with WBF."""
    per = []
    for name, (model, imgsz) in members.items():
        cfg = member_cfg[name]
        entry = predict_passes(model, img, imgsz, tiles, device=device, passes=cfg["passes"])
        per.append(combine(entry, cfg["passes"], tile_max_side=cfg.get("tile_max_side")))
    return per[0] if len(per) == 1 else fuse(per, img.shape, weights=weights, iou=wbf_iou)
