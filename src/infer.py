"""Inference with a trained Faster R-CNN run: plain, at another input size, or SAHI-style slicing.

No training happens here, so every option can be compared on the same trained weights.

Examples (from src/, after training has finished: never while another job uses the GPU):
    # same settings as training (720/1280): should reproduce the run's val mAP
    python infer.py --run ../runs/faster_rcnn/anchors_3scale --tag plain
    # larger test-time input
    python infer.py --run ../runs/faster_rcnn/anchors_3scale --min-size 1080 --max-size 1920 --tag full_1920
    # slices of 640 px at native resolution, 25% overlap, plus the normal full-frame pass, merged with NMS
    python infer.py --run ../runs/faster_rcnn/anchors_3scale --slice 640 --overlap 0.25 --full-pass --tag sahi_640
    # multi-scale TTA: training scale + 1.5x, merged with NMS; optionally only small boxes from the 1.5x pass
    python infer.py --run ../runs/faster_rcnn/anchors_3scale --scales 720:1280,1080:1920 --tag ms
    python infer.py --run ../runs/faster_rcnn/anchors_3scale --scales 720:1280,1080:1920 --hi-max-box 48 --tag ms_small
    # test-like validation: each val clip shrunk to a long side from the test resolution mix (use this to rank models)
    python infer.py --run ../runs/faster_rcnn/baseline --val-res test --tag plain_testres
    # test set -> submission CSV (same options)
    python infer.py --run ../runs/faster_rcnn/anchors_3scale --split test --tag plain

Val results are appended to ../runs/inference_results.csv; test submissions go to <run>/submissions/<tag>.csv.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torchvision.ops import batched_nms
from tqdm.auto import tqdm

from common.data import CocoDetection, ImageList, to_model_input
from common.evaluate import evaluate_detections
from common.frcnn import load_faster_rcnn_run
from common.submission import write_submission

ROOT = Path(__file__).resolve().parent.parent
MAX_DETS = 500


def set_input_size(model, min_size, max_size):
    """Change the model's internal resize (short side min_size, long side <= max_size)."""
    model.transform.min_size = (min_size,)
    model.transform.max_size = max_size


def slice_windows(h, w, size, overlap):
    """Top-left corners of overlapping size x size windows covering an h x w image (last window flush with the edge)."""
    step = max(1, int(size * (1 - overlap)))

    def starts(n):
        if n <= size:
            return [0]
        s = list(range(0, n - size, step))
        return s + [n - size]
    return [(y, x) for y in starts(h) for x in starts(w)]


@torch.no_grad()
def predict_image(model, img, args, device):
    """Detections for one uint8 (C, H, W) image as (boxes xyxy, scores, labels) in original pixels."""
    boxes, scores, labels = [], [], []

    if args.scales:
        # Multi-scale test-time augmentation: one full-frame pass per (min, max) size. Passes after the first are the
        # upscaled ones; with --hi-max-box they only contribute boxes up to that size (original pixels, sqrt(w*h)),
        # so large objects come from the scale the model was trained at.
        for k, (mn, mx) in enumerate(args.scales):
            set_input_size(model, mn, mx)
            out = model(to_model_input([img], device))[0]
            b, s, c = out["boxes"], out["scores"], out["labels"]
            if k > 0 and args.hi_max_box:
                keep = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).sqrt() <= args.hi_max_box
                b, s, c = b[keep], s[keep], c[keep]
            boxes.append(b); scores.append(s); labels.append(c)
        boxes, scores, labels = torch.cat(boxes), torch.cat(scores), torch.cat(labels)
        keep = batched_nms(boxes, scores, labels, args.nms)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        keep = labels >= 1
        order = scores[keep].argsort(descending=True)[:MAX_DETS]
        return boxes[keep][order].cpu(), scores[keep][order].cpu(), labels[keep][order].cpu()

    if args.slice is None or args.full_pass:
        set_input_size(model, args.min_size, args.max_size)
        out = model(to_model_input([img], device))[0]
        boxes.append(out["boxes"]); scores.append(out["scores"]); labels.append(out["labels"])

    if args.slice is not None:
        # Each slice is fed at native resolution (optionally scaled by --slice-scale): the model does not shrink it.
        side = int(round(args.slice * args.slice_scale))
        set_input_size(model, side, side)
        _, h, w = img.shape
        windows = slice_windows(h, w, args.slice, args.overlap)
        for i in range(0, len(windows), args.batch):
            chunk = windows[i:i + args.batch]
            crops = [img[:, y:y + args.slice, x:x + args.slice] for y, x in chunk]
            for (y, x), out in zip(chunk, model(to_model_input(crops, device))):
                shift = torch.tensor([x, y, x, y], device=device, dtype=out["boxes"].dtype)
                boxes.append(out["boxes"] + shift); scores.append(out["scores"]); labels.append(out["labels"])

    boxes, scores, labels = torch.cat(boxes), torch.cat(scores), torch.cat(labels)
    if args.slice is not None:   # merge duplicates from overlapping slices and the full-frame pass
        keep = batched_nms(boxes, scores, labels, args.nms)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    keep = labels >= 1
    order = scores[keep].argsort(descending=True)[:MAX_DETS]
    return boxes[keep][order].cpu(), scores[keep][order].cpu(), labels[keep][order].cpu()


def test_like_long_sides(val_images, test_csv):
    """Give each val clip a test-like long side: the test clips' long sides at evenly spaced quantiles.

    Test frames are much smaller than val frames (median long side ~1360 vs ~1904 px; some only 680), so val scored at
    its native resolution ranks models differently from the leaderboard. Each val clip is mapped (in sorted order) to
    the (k + 0.5) / n quantile of the test clips' long sides, so the 7 clips cover the test resolution mix.
    """
    test = pd.read_csv(test_csv).groupby("clip")[["width", "height"]].first()
    test_long = sorted(test.max(axis=1).tolist())
    clips = sorted({im["video"] for im in val_images})
    n = len(clips)
    return {c: int(test_long[min(len(test_long) - 1, int((k + 0.5) / n * len(test_long)))]) for k, c in enumerate(clips)}


def downscale(img, long_side):
    """Shrink a uint8 (C, H, W) image so its long side is `long_side` (never enlarges). Returns (image, scale)."""
    _, h, w = img.shape
    s = long_side / max(h, w)
    if s >= 1:
        return img, 1.0
    small = torch.nn.functional.interpolate(img[None].float(), size=(round(h * s), round(w * s)),
                                            mode="bilinear", antialias=True, align_corners=False)
    return small[0].round().clamp(0, 255).to(torch.uint8), s


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run folder with summary.json and best.pt")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--tag", required=True, help="name for this inference setting")
    p.add_argument("--min-size", type=int, default=720)
    p.add_argument("--max-size", type=int, default=1280)
    p.add_argument("--slice", type=int, default=None, help="slice size in px (enables SAHI-style slicing)")
    p.add_argument("--overlap", type=float, default=0.25)
    p.add_argument("--slice-scale", type=float, default=1.0, help="upscale factor applied to each slice")
    p.add_argument("--full-pass", action="store_true", help="with --slice: also run the whole frame and merge")
    p.add_argument("--scales", type=str, default=None,
                   help="multi-scale TTA: comma-separated min:max sizes, first = training scale, e.g. 720:1280,1080:1920")
    p.add_argument("--hi-max-box", type=float, default=None,
                   help="with --scales: keep only boxes up to this size (px, sqrt(w*h)) from the upscaled passes")
    p.add_argument("--nms", type=float, default=0.5, help="IoU for merging slice / multi-scale detections")
    p.add_argument("--batch", type=int, default=4, help="slices per forward pass")
    p.add_argument("--val-res", default="native",
                   help="val frame resolution: 'native', 'test' (per-clip test-like long sides) or a long side in px")
    p.add_argument("--data-root", default=str(ROOT / "multiple-object-video-detection-hw-2-ucu-ai-2026"))
    p.add_argument("--proc-root", default=str(ROOT / "data_processed"))
    args = p.parse_args()
    if args.scales:
        args.scales = [tuple(int(v) for v in pair.split(":")) for pair in args.scales.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Path(args.run)
    model = load_faster_rcnn_run(run, device)
    data_root, gt_file = Path(args.data_root), Path(args.proc_root) / "val" / "gt_eval_coco.json"
    t0 = time.time()

    if args.split == "val":
        ds = CocoDetection(gt_file, data_root)
        if args.val_res == "test":
            long_sides = test_like_long_sides(ds.images, data_root / "test_images.csv")
            print("val clip -> test-like long side:", long_sides)
        dets = []
        for i in tqdm(range(len(ds)), desc=f"val {args.tag}"):
            img, tgt = ds[i]
            scale = 1.0
            if args.val_res != "native":
                target = long_sides[ds.images[i]["video"]] if args.val_res == "test" else int(args.val_res)
                img, scale = downscale(img, target)
            b, s, c = predict_image(model, img, args, device)
            b = b / scale   # boxes back to original-image pixels, so the ground truth stays untouched
            dets += [{"image_id": tgt["image_id"], "category_id": int(ci), "score": float(si),
                      "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]}
                     for (x1, y1, x2, y2), si, ci in zip(b.tolist(), s.tolist(), c.tolist())]
        seconds = (time.time() - t0) / len(ds)
        metrics = evaluate_detections(gt_file, dets)
        out_dir = run / "inference"
        out_dir.mkdir(exist_ok=True)
        (out_dir / f"{args.tag}_val_dets.json").write_text(json.dumps(dets))
        row = {"run": run.name, "tag": args.tag, "val_res": args.val_res, **{k: v for k, v in vars(args).items() if k in
               ["min_size", "max_size", "slice", "overlap", "slice_scale", "full_pass", "nms", "hi_max_box"]},
               "scales": " ".join(f"{a}:{b}" for a, b in args.scales) if args.scales else "",
               "sec_per_image": seconds, **metrics}
        csv = ROOT / "runs" / "inference_results.csv"
        table = pd.concat([pd.read_csv(csv), pd.DataFrame([row])]) if csv.exists() else pd.DataFrame([row])
        table.to_csv(csv, index=False)
        print(f"{run.name}/{args.tag}: mAP {metrics['mAP']:.4f} AP50 {metrics['AP50']:.4f} "
              f"AP_small {metrics['AP_small']:.4f} AP_medium {metrics['AP_medium']:.4f} | {seconds:.2f} s/image")
    else:
        test = pd.read_csv(data_root / "test_images.csv")
        ds = ImageList([data_root / f for f in test["file_name"]], test["image_id"])
        preds = {}
        for i in tqdm(range(len(ds)), desc=f"test {args.tag}"):
            img, key = ds[i]
            b, s, c = predict_image(model, img, args, device)
            preds[key] = [(int(ci) - 1, float(si), x1, y1, x2 - x1, y2 - y1)
                          for (x1, y1, x2, y2), si, ci in zip(b.tolist(), s.tolist(), c.tolist())]
        out = run / "submissions" / f"{args.tag}.csv"
        out.parent.mkdir(exist_ok=True)
        sub = write_submission(preds, data_root / "test_images.csv", out)
        print(f"Wrote {out}: {len(sub)} rows, {sub.PredictionString.str.len().gt(0).sum()} with detections")


if __name__ == "__main__":
    main()
