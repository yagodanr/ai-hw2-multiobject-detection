"""Temporal rescoring: use neighbouring frames to re-rank a detector's per-frame detections.

Video frames are 40 ms apart, so a real object appears in many consecutive frames, while many false positives flicker.
Steps (a simplified Seq-NMS, Han et al., 2016):
  1. Link: in each clip, match detections of the same class in consecutive frames (IoU >= link_iou, greedy by IoU),
     forming tracks. Only detections with score >= link_min take part; lower ones keep their score.
  2. Rescore: a detection on a track of length >= min_len gets  s' = (1 - lam) * s + lam * track_score,
     where track_score is the mean of the track's top-k scores. Detections on shorter tracks are multiplied by
     `single_penalty` (isolated detections are more often false positives).
mAP depends only on the ranking of detections, so rescoring can lift weak but consistent detections above flickering ones.

Usage (from src/). Inputs come from infer.py (raw detections for every val frame and for the test set):
    python infer.py --run ../runs/faster_rcnn/baseline --split valall --scales 720:1280,1080:1920 --val-res test --tag ms
    python infer.py --run ../runs/faster_rcnn/baseline --split test   --scales 720:1280,1080:1920 --tag ms
    python temporal.py --run ../runs/faster_rcnn/baseline --tag ms            # grid search on val, then test submission
Outputs: ../runs/temporal_results.csv (one row per setting) and <run>/submissions/<tag>_temporal.csv.
"""
import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from common.evaluate import evaluate_detections
from common.submission import write_submission

ROOT = Path(__file__).resolve().parent.parent


def iou_matrix(a, b):
    """IoU between boxes a (N, 4) and b (M, 4), both [x, y, w, h]."""
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = (np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0])).clip(0)
    ih = (np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1])).clip(0)
    inter = iw * ih
    return inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1)[None] - inter + 1e-9)


def link_tracks(dets, link_iou, link_min):
    """Return a track id per detection (-1 = not linked). Detections are linked frame-to-frame within a clip and class."""
    track = np.full(len(dets), -1)
    by_key = defaultdict(list)   # (video, class) -> detection indices
    for i, d in enumerate(dets):
        if d["score"] >= link_min:
            by_key[(d["video"], d["category_id"])].append(i)
    next_id = 0
    for idx in by_key.values():
        by_frame = defaultdict(list)
        for i in idx:
            by_frame[dets[i]["frame"]].append(i)
        prev_frame, prev = None, []
        for f in sorted(by_frame):
            cur = by_frame[f]
            if prev and prev_frame == f - 1:
                iou = iou_matrix(np.array([dets[i]["bbox"] for i in prev]), np.array([dets[i]["bbox"] for i in cur]))
                pairs = np.argwhere(iou >= link_iou)
                pairs = pairs[np.argsort(-iou[pairs[:, 0], pairs[:, 1]])]   # greedy: best overlaps first
                used_p, used_c = set(), set()
                for p, c in pairs:
                    if p in used_p or c in used_c:
                        continue
                    used_p.add(p); used_c.add(c)
                    track[cur[c]] = track[prev[p]]
            for i in cur:
                if track[i] < 0:
                    track[i] = next_id
                    next_id += 1
            prev_frame, prev = f, cur
    return track


def rescore(dets, track, lam, top_k, min_len, single_penalty):
    """New score per detection (see module docstring)."""
    scores = np.array([d["score"] for d in dets])
    new = scores.copy()
    members = defaultdict(list)
    for i, t in enumerate(track):
        if t >= 0:
            members[t].append(i)
    for idx in members.values():
        s = scores[idx]
        if len(idx) >= min_len:
            track_score = np.sort(s)[::-1][:top_k].mean()
            new[idx] = (1 - lam) * s + lam * track_score
        else:
            new[idx] = s * single_penalty
    return new


def with_scores(dets, scores):
    return [{**d, "score": float(s)} for d, s in zip(dets, scores)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--tag", required=True, help="infer.py tag whose valall/test detections to use")
    p.add_argument("--proc-root", default=str(ROOT / "data_processed"))
    p.add_argument("--data-root", default=str(ROOT / "multiple-object-video-detection-hw-2-ucu-ai-2026"))
    args = p.parse_args()

    run = Path(args.run)
    gt_file = Path(args.proc_root) / "val" / "gt_eval_coco.json"
    labelled = {im["id"] for im in json.loads(gt_file.read_text())["images"]}
    val = json.loads((run / "inference" / f"{args.tag}_valall_dets.json").read_text())
    print(f"{len(val)} val detections on {len({d['image_id'] for d in val})} frames ({len(labelled)} labelled)")

    def score(dets):
        return evaluate_detections(gt_file, [d for d in dets if d["image_id"] in labelled])

    # Small grid, tuned on val. Each setting costs a few seconds of CPU, so a plain grid is enough here.
    grid = dict(link_iou=[0.3, 0.5], link_min=[0.05], lam=[0.3, 0.5, 0.7], top_k=[5, 15],
                min_len=[3], single_penalty=[1.0, 0.7])
    base = score(val)
    rows = [{"setting": "none", **{k: "" for k in grid}, **base}]
    print(f"no rescoring: mAP {base['mAP']:.4f} AP_small {base['AP_small']:.4f}")
    tracks = {}
    for values in itertools.product(*grid.values()):
        cfg = dict(zip(grid, values))
        key = (cfg["link_iou"], cfg["link_min"])
        if key not in tracks:
            tracks[key] = link_tracks(val, *key)
        new = rescore(val, tracks[key], cfg["lam"], cfg["top_k"], cfg["min_len"], cfg["single_penalty"])
        m = score(with_scores(val, new))
        rows.append({"setting": "rescore", **cfg, **m})
        print(f"{cfg}: mAP {m['mAP']:.4f} ({m['mAP'] - base['mAP']:+.4f}) AP_small {m['AP_small']:.4f}")

    table = pd.DataFrame(rows).assign(run=run.name, tag=args.tag)
    csv = ROOT / "runs" / "temporal_results.csv"
    (pd.concat([pd.read_csv(csv), table]) if csv.exists() else table).to_csv(csv, index=False)
    best = table.iloc[1:].sort_values("mAP", ascending=False).iloc[0]
    best_cfg = {k: best[k] for k in grid}
    print(f"\nbest on val: {best_cfg}  mAP {best['mAP']:.4f} vs {base['mAP']:.4f} without rescoring")

    test_file = run / "inference" / f"{args.tag}_test_dets.json"
    if not test_file.exists():
        print(f"no {test_file.name}: run infer.py --split test --tag {args.tag} to get a test submission")
        return
    test = json.loads(test_file.read_text())
    t = link_tracks(test, best_cfg["link_iou"], best_cfg["link_min"])
    new = rescore(test, t, best_cfg["lam"], int(best_cfg["top_k"]), int(best_cfg["min_len"]), best_cfg["single_penalty"])
    preds = defaultdict(list)
    for d, s in zip(test, new):
        x, y, w, h = d["bbox"]
        preds[d["image_id"]].append((d["category_id"] - 1, float(s), x, y, w, h))
    out = run / "submissions" / f"{args.tag}_temporal.csv"
    out.parent.mkdir(exist_ok=True)
    write_submission(preds, Path(args.data_root) / "test_images.csv", out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
