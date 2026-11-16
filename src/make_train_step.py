"""Denser training set: every STEP-th training frame instead of every 8th, with the same cleaning as 01_preprocess.

    python make_train_step.py 4        # -> ../data_processed_step4/  (~6k train frames, ~2 GB, a few minutes)
    python make_train_step.py 2        # -> ../data_processed_step2/  (~12k train frames)
Env: DATA_ROOT (competition data), PROC_ROOT (our data_processed), STEP_DATA_DIR (where to write; default next to PROC_ROOT).

Only TRAIN changes. `val/` is a link to ../data_processed/val, so every model is still scored on the same 359 val frames.
The cleaning is copied from 01_preprocess.ipynb: clip boxes to the frame, drop boxes thinner than 2 px, drop training boxes
that lie more than 50% inside an ignored region, fill ignored regions with grey (114) in the image.
"""
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

STEP = int(sys.argv[1])
LABELS_ONLY = "--labels-only" in sys.argv          # for testing: write label files, no images
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "../multiple-object-video-detection-hw-2-ucu-ai-2026"))
SRC_PROC = Path(os.environ.get("PROC_ROOT", "../data_processed"))            # its val/ is reused as is
OUT_ROOT = Path(os.environ.get("STEP_DATA_DIR", SRC_PROC.parent)) / f"data_processed_step{STEP}"
MIN_BOX_PX, IGNORE_OVERLAP, FILL_VALUE, JPEG_QUALITY = 2, 0.5, (114, 114, 114), 95

coco = json.loads((DATA_ROOT / "train" / "annotations_coco.json").read_text())
images = pd.DataFrame(coco["images"]).rename(columns={"id": "image_id"})
anns = pd.DataFrame(coco["annotations"])
anns[["x", "y", "w", "h"]] = pd.DataFrame(anns["bbox"].tolist(), index=anns.index).astype(float)
anns = anns.merge(images[["image_id", "video", "frame_id", "width", "height"]], on="image_id")
objects = anns[anns["iscrowd"] == 0].reset_index(drop=True)
ignores = anns[anns["iscrowd"] == 1].reset_index(drop=True)

# clean boxes (01: clean_boxes)
x1, y1 = np.maximum(objects.x, 0), np.maximum(objects.y, 0)
x2, y2 = np.minimum(objects.x + objects.w, objects.width), np.minimum(objects.y + objects.h, objects.height)
objects = objects.assign(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
objects = objects[(objects.w >= MIN_BOX_PX) & (objects.h >= MIN_BOX_PX) & objects.category_id.between(1, 8)]

# subsample (01: subsample)
first = images.groupby("video")["frame_id"].transform("min")
sel = images[(images["frame_id"] - first) % STEP == 0].reset_index(drop=True)
objs = objects[objects.image_id.isin(sel.image_id)].reset_index(drop=True)
ign = ignores[ignores.image_id.isin(sel.image_id)].reset_index(drop=True)


def ignore_fraction(o, regions, W, H):
    mask = np.zeros((H, W), np.uint8)
    for r in regions.itertuples():
        mask[int(r.y):int(np.ceil(r.y + r.h)), int(r.x):int(np.ceil(r.x + r.w))] = 1
    integral = cv2.integral(mask)
    a1, b1 = o.x.astype(int).values, o.y.astype(int).values
    a2 = np.ceil(o.x + o.w).astype(int).clip(upper=W).values
    b2 = np.ceil(o.y + o.h).astype(int).clip(upper=H).values
    inside = integral[b2, a2] - integral[b1, a2] - integral[b2, a1] + integral[b1, a1]
    return inside / np.maximum((a2 - a1) * (b2 - b1), 1)


keep = np.ones(len(objs), bool)
ign_by_img = dict(tuple(ign.groupby("image_id")))
for img_id, idx in objs.groupby("image_id").groups.items():
    if img_id in ign_by_img:
        o = objs.loc[idx]
        keep[idx] = ignore_fraction(o, ign_by_img[img_id], int(o.width.iloc[0]), int(o.height.iloc[0])) <= IGNORE_OVERLAP
train_objs = objs[keep].reset_index(drop=True)
obj_by_img = dict(tuple(train_objs.groupby("image_id")))
print(f"step {STEP}: {len(sel)} of {len(images)} train frames, {len(train_objs)} boxes", flush=True)

out = OUT_ROOT / "train"


def work(row):
    dst = out / row.file_name
    if not LABELS_ONLY and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        regions = ign_by_img.get(row.image_id)
        src = DATA_ROOT / "train" / row.file_name
        if regions is None:
            shutil.copy2(src, dst)
        else:
            img = cv2.imread(str(src))
            for r in regions.itertuples():
                img[int(r.y):int(np.ceil(r.y + r.h)), int(r.x):int(np.ceil(r.x + r.w))] = FILL_VALUE
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    label = out / "labels" / row.video / (Path(row.file_name).stem + ".txt")
    label.parent.mkdir(parents=True, exist_ok=True)
    o = obj_by_img.get(row.image_id)
    lines = [] if o is None else [
        f"{b.category_id - 1} {(b.x + b.w / 2) / row.width:.6f} {(b.y + b.h / 2) / row.height:.6f} "
        f"{b.w / row.width:.6f} {b.h / row.height:.6f}" for b in o.itertuples()]
    label.write_text("\n".join(lines))


rows = list(sel.itertuples())
with ThreadPoolExecutor(8) as pool:
    for n, _ in enumerate(pool.map(work, rows), 1):
        if n % 2000 == 0 or n == len(rows):
            print(f"  {n}/{len(rows)} frames written", flush=True)

val_link = OUT_ROOT / "val"
if not val_link.exists():
    val_link.symlink_to((SRC_PROC / "val").resolve(), target_is_directory=True)
(OUT_ROOT / "preprocess_stats.json").write_text(json.dumps(
    {"train": {"frames_total": len(images), "frames_kept": len(sel), "boxes_kept_train": len(train_objs)},
     "config": {"step": STEP, "val": "same as data_processed (step 8)", "min_box_px": MIN_BOX_PX,
                "ignore_overlap": IGNORE_OVERLAP, "fill_value": FILL_VALUE}}, indent=2))
print("done:", OUT_ROOT.resolve())
