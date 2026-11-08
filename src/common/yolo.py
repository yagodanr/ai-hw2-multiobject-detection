"""Ultralytics models (YOLO, RT-DETR) on the shared protocol.

Ultralytics trains on `data_processed/` and validates on the grey-filled val copies, which is fine for its own
progress bars but not comparable with the other models. This module re-scores every saved epoch with
`common.evaluate.evaluate_detections` on the ORIGINAL val frames, and writes the same run layout as `engine.py`:

    <runs_dir>/<name>/summary.json, history.json, val_dets.json, best.pt   (ours, shared layout)
    <runs_dir>/<name>/weights/, results.csv, args.yaml                     (Ultralytics' own output)

so `load_summaries`, `error_breakdown.py` and `run_queue.py` work unchanged.
"""
import json
import re
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from .data import CLASSES
from .evaluate import MAX_DETS, evaluate_detections


# ---------------------------------------------------------------- data

def write_data_yaml(proc_root, out_file):
    """Ultralytics data config with an absolute dataset path.

    Ultralytics resolves a relative `path:` against its own `datasets_dir` setting, not against the yaml's
    folder, so we always write an absolute one (works the same locally, on Kaggle and on Colab).
    """
    cfg = {"path": str(Path(proc_root).resolve()), "train": "train/images", "val": "val/images",
           "nc": len(CLASSES), "names": list(CLASSES)}
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(out_file).write_text(yaml.safe_dump(cfg, sort_keys=False))
    return Path(out_file)


def val_items(gt_file, data_root, max_images=None):
    """[(image_id, path)] of the ORIGINAL val frames listed in gt_eval_coco.json (`file_name` is relative to data_root)."""
    images = json.loads(Path(gt_file).read_text())["images"]
    items = [(im["id"], Path(data_root) / im["file_name"]) for im in images]
    return items[:max_images] if max_images else items


# ---------------------------------------------------------------- inference

def _predict_kwargs(imgsz, conf, iou, max_det, nms, device, augment):
    kw = dict(imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, augment=augment, verbose=False)
    if nms is not None:
        # YOLO26 only: False = one-to-one (NMS-free) head. Ultralytics fixes this when the predictor is first
        # built, so a model object keeps the mode of its first predict call: use a fresh YOLO(...) to switch.
        kw["nms"] = nms
    if device is not None:
        kw["device"] = device
    return kw


def predict_boxes(model, sources, imgsz, conf=0.001, iou=0.7, max_det=MAX_DETS, nms=None,
                  device=None, augment=False, batch=4):
    """Yield (xyxy [N,4], score [N], cls0 [N]) per source, in original-image pixels. Sources: paths or BGR arrays."""
    kw = _predict_kwargs(imgsz, conf, iou, max_det, nms, device, augment)
    for i in range(0, len(sources), batch):
        chunk = [str(s) if isinstance(s, Path) else s for s in sources[i:i + batch]]
        for r in model.predict(chunk, **kw):
            b = r.boxes
            yield b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)


def predict_coco(model, items, imgsz, **kw):
    """COCO detection dicts (category_id = cls + 1, xywh) for [(image_id, path)] items."""
    dets = []
    for (img_id, _), (xyxy, score, cls) in zip(items, predict_boxes(model, [p for _, p in items], imgsz, **kw)):
        for (x1, y1, x2, y2), s, c in zip(xyxy, score, cls):
            dets.append({"image_id": int(img_id), "category_id": int(c) + 1,
                         "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)})
    return dets


def predict_test(model, test_csv, data_root, imgsz, max_images=None, **kw):
    """{image_id: [(cls0, score, x, y, w, h)]} for common.submission.write_submission."""
    test = pd.read_csv(test_csv)
    if max_images:
        test = test.head(max_images)
    paths = [Path(data_root) / f for f in test["file_name"]]
    preds = {}
    for key, (xyxy, score, cls) in zip(test["image_id"], predict_boxes(model, paths, imgsz, **kw)):
        preds[key] = [(int(c), float(s), float(x1), float(y1), float(x2 - x1), float(y2 - y1))
                      for (x1, y1, x2, y2), s, c in zip(xyxy, score, cls)]
    return preds


def measure_fps(model, paths, imgsz, device=None, warmup=5):
    """Images per second at batch size 1, FP32, images preloaded in memory (letterbox + NMS included)."""
    images = [cv2.imread(str(p)) for p in paths]
    kw = _predict_kwargs(imgsz, 0.001, 0.7, MAX_DETS, None, device, False)
    for im in images[:warmup]:
        model.predict(im, **kw)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for im in images:
        model.predict(im, **kw)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return len(images) / (time.perf_counter() - t0)


# ---------------------------------------------------------------- training + scoring

def _epoch_checkpoints(weights_dir):
    """{epoch (1-based, as in results.csv): path} for Ultralytics' epoch<k>.pt files."""
    found = {int(re.fullmatch(r"epoch(\d+)\.pt", f.name).group(1)): f
             for f in Path(weights_dir).glob("epoch*.pt") if re.fullmatch(r"epoch(\d+)\.pt", f.name)}
    if found and min(found) == 0:          # older Ultralytics numbered epochs from 0
        found = {k + 1: v for k, v in found.items()}
    return dict(sorted(found.items()))


def score_checkpoints(run_dir, model_cls, items, gt_file, pred_kw, every=1, keep_epoch_ckpts=False):
    """Score saved epochs with the shared evaluator; return (history rows, best row, best dets).

    The best epoch's checkpoint is copied to <run_dir>/best.pt (optimizer stripped).

    The last epoch is always scored. Rows merge our metrics with Ultralytics' results.csv (losses, its own mAP
    on the grey val, cumulative time) so the per-epoch curve sits next to the CNNs' history.json.
    """
    run_dir = Path(run_dir)
    ckpts = _epoch_checkpoints(run_dir / "weights")
    if not ckpts:                            # save_period disabled: fall back to last.pt
        ckpts = {None: run_dir / "weights" / "last.pt"}
    last = max(k for k in ckpts if k is not None) if any(k is not None for k in ckpts) else None
    todo = {k: p for k, p in ckpts.items() if k is None or k == last or k % every == 0}

    results = pd.read_csv(run_dir / "results.csv") if (run_dir / "results.csv").exists() else None
    if results is not None:
        results.columns = [c.strip() for c in results.columns]
    image_ids = [i for i, _ in items]

    history, best, best_dets, best_ckpt = [], None, None, None
    for epoch, ckpt in todo.items():
        model = model_cls(str(ckpt))
        dets = predict_coco(model, items, **pred_kw)
        row = {"epoch": epoch, **evaluate_detections(gt_file, dets, image_ids=image_ids)}
        if results is not None and epoch is not None and epoch in set(results["epoch"]):
            r = results[results["epoch"] == epoch].iloc[0]
            row["minutes"] = float(r["time"]) / 60 if "time" in r else float("nan")   # cumulative
            row.update({f"ul_{k}": float(v) for k, v in r.items() if k not in ("epoch", "time")})
        history.append(row)
        print(f"  epoch {epoch}: mAP {row['mAP']:.4f} AP50 {row['AP50']:.4f} AP_small {row['AP_small']:.4f}")
        if best is None or row["mAP"] > best["mAP"]:
            best, best_dets, best_ckpt = row, dets, ckpt
        del model

    shutil.copy2(best_ckpt, run_dir / "best.pt")
    try:
        from ultralytics.utils.torch_utils import strip_optimizer
        strip_optimizer(run_dir / "best.pt")       # inference-only copy, ~half the size
    except Exception as e:                          # stripping is a nicety; never lose a finished run over it
        print("strip_optimizer failed:", e)
    if not keep_epoch_ckpts:
        for k, p in ckpts.items():
            if k is not None:
                p.unlink(missing_ok=True)
    return history, best, best_dets


def train_and_score(name, cfg, protocol, data_yaml, runs_dir, gt_file, data_root, model_cls=None):
    """Train one experiment with Ultralytics, then score it with the shared evaluator.

    cfg: {"model": yaml or .pt, "weights": optional .pt to transfer into a yaml model, "imgsz": int, "batch"/"epochs": optional overrides,
          "train": {extra Ultralytics train args}, "predict": {extra predict args, e.g. nms}}.
    protocol: {"epochs", "batch", "train": {shared Ultralytics train args}, "predict": {...}, "score_every",
               "val_max_images"}.
    """
    if model_cls is None:
        from ultralytics import YOLO as model_cls
    runs_dir = Path(runs_dir).resolve()
    out = runs_dir / name
    t0 = time.time()

    model = model_cls(cfg["model"])
    if cfg.get("weights"):
        model.load(cfg["weights"])           # transfers every layer whose name and shape match
    train_args = {**protocol["train"], **cfg.get("train", {})}
    model.train(data=str(data_yaml), project=str(runs_dir), name=name, exist_ok=True,
                imgsz=cfg["imgsz"], epochs=cfg.get("epochs", protocol["epochs"]),
                batch=cfg.get("batch", protocol["batch"]),
                **train_args)
    train_minutes = (time.time() - t0) / 60
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    items = val_items(gt_file, data_root, protocol.get("val_max_images"))
    pred_kw = {"imgsz": cfg["imgsz"], **protocol["predict"], **cfg.get("predict", {})}
    print(f"[{name}] scoring checkpoints on {len(items)} original val frames")
    history, best, dets = score_checkpoints(out, model_cls, items, gt_file, pred_kw,
                                            every=protocol.get("score_every", 1))
    best["train_minutes_total"] = train_minutes

    (out / "history.json").write_text(json.dumps(history, indent=1))
    (out / "val_dets.json").write_text(json.dumps(dets))
    used = yaml.safe_load((out / "args.yaml").read_text()) if (out / "args.yaml").exists() else {}
    summary = {"name": name, "config": cfg, "protocol": protocol, "best": best,
               "ultralytics_args": {k: used.get(k) for k in ("batch", "optimizer", "lr0", "imgsz", "epochs", "amp")}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"[{name}] best epoch {best['epoch']}: mAP {best['mAP']:.4f}, AP50 {best['AP50']:.4f}, "
          f"AP_small {best['AP_small']:.4f} | training {train_minutes:.0f} min")
    return history
