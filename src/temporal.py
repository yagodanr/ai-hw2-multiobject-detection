"""Temporal rescoring: use neighbouring frames to re-rank a detector's per-frame detections.

Video frames are 40 ms apart, so a real object appears in many consecutive frames, while many false positives flicker.
Steps (a simplified Seq-NMS, Han et al., 2016):
  1. Link: in each clip, match detections of the same class in consecutive frames (IoU >= link_iou, greedy by IoU),
     forming tracks. Only detections with score >= link_min take part; lower ones keep their score.
  2. Rescore: a detection on a track of length >= min_len gets  s' = (1 - lam) * s + lam * track_score,
     where track_score is the mean of the track's top-k scores. Detections on shorter tracks are multiplied by
     `single_penalty` (isolated detections are more often false positives).
mAP depends only on the ranking of detections, so rescoring can lift weak but consistent detections above flickering ones.

Version 2 (`temporal_v2`, compared in experiments/temporal_v2.ipynb) adds four switches on top, all post-processing:
  - motion: compensate the drone's camera motion. A homography between consecutive frames (corner points tracked with
    Lucas-Kanade optical flow, RANSAC) moves the previous boxes into the current frame before matching.
  - center_dist: also link boxes whose centres are close relative to their size, not only by IoU (a 10 px object moving
    6 px has almost no IoU with itself; idea from REPP, Sabater et al., 2020).
  - max_gap / fill: let a track skip up to max_gap frames, and insert the missing boxes (lower score) so missed objects
    come back. Plain rescoring only re-ranks boxes; filling can raise recall.
  - smooth_sigma: Gaussian smoothing of box coordinates along each track (REPP's re-coordinating).
With all switches off, temporal_v2 gives exactly the version 1 result.

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


# ---------------------------------------------------------------- version 2


def estimate_homographies(frames, long_side=960, max_corners=1500):
    """{(video, frame): H} mapping pixel coordinates of frame-1 to frame, for every frame whose predecessor exists.

    frames: dicts with "video", "frame", "path" (any order). Points are tracked on a grayscale copy shrunk to
    `long_side`, and H is rescaled to original pixels. Fewer than 20 tracked points or a failed fit -> identity.
    """
    import cv2

    by_video = defaultdict(dict)
    for f in frames:
        by_video[f["video"]][f["frame"]] = f["path"]
    out = {}
    for video, paths in by_video.items():
        prev_gray, prev_idx, s = None, None, 1.0
        for idx in sorted(paths):
            img = cv2.imread(str(paths[idx]), cv2.IMREAD_GRAYSCALE)
            if img is None:   # unreadable frame: no homography to or from it
                prev_gray = None
                continue
            s = min(1.0, long_side / max(img.shape))
            gray = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
            if prev_gray is not None and prev_idx == idx - 1:
                H = np.eye(3)
                p0 = cv2.goodFeaturesToTrack(prev_gray, max_corners, 0.01, 8)
                if p0 is not None and len(p0) >= 20:
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, winSize=(21, 21), maxLevel=3)
                    ok = st.ravel() == 1
                    if ok.sum() >= 20:
                        Hs, _ = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 3.0)
                        if Hs is not None:
                            S = np.diag([s, s, 1.0])
                            H = np.linalg.inv(S) @ Hs @ S   # back to original pixels
                out[(video, idx)] = H
            prev_gray, prev_idx = gray, idx
    return out


def warp_boxes(boxes, H):
    """Axis-aligned boxes [x, y, w, h] (N, 4) moved by homography H (the bounding box of the 4 warped corners)."""
    if H is None or len(boxes) == 0:
        return boxes
    x, y, w, h = boxes.T
    corners = np.stack([np.stack([x, y], 1), np.stack([x + w, y], 1), np.stack([x, y + h], 1),
                        np.stack([x + w, y + h], 1)], 1).reshape(-1, 2)
    pts = np.c_[corners, np.ones(len(corners))] @ H.T
    pts = (pts[:, :2] / pts[:, 2:3]).reshape(-1, 4, 2)
    x0, y0 = pts[..., 0].min(1), pts[..., 1].min(1)
    return np.stack([x0, y0, pts[..., 0].max(1) - x0, pts[..., 1].max(1) - y0], 1)


def _chain(homs, video, a, b):
    """Homography from frame a to frame b (a < b) as the product of the per-frame ones; identity where unknown."""
    H = np.eye(3)
    for g in range(a + 1, b + 1):
        H = homs.get((video, g), np.eye(3)) @ H
    return H


def link_tracks_v2(dets, link_iou, link_min, homs=None, center_dist=None, max_gap=0):
    """Track id per detection (-1 = not linked), plus the gaps to fill: [(track, frame_from, frame_to)].

    Same greedy frame-by-frame linking as link_tracks, with three options: previous boxes are moved by the camera
    motion first (homs), boxes may also link by centre distance / size (center_dist), and a track may skip up to
    max_gap frames. With homs=None, center_dist=None, max_gap=0 the result equals link_tracks.
    """
    track = np.full(len(dets), -1)
    gaps = []
    by_key = defaultdict(list)
    for i, d in enumerate(dets):
        if d["score"] >= link_min:
            by_key[(d["video"], d["category_id"])].append(i)
    next_id = 0
    for (video, _), idx in by_key.items():
        by_frame = defaultdict(list)
        for i in idx:
            by_frame[dets[i]["frame"]].append(i)
        last = {}   # track id -> (frame, detection index) of its latest box
        for f in sorted(by_frame):
            cur = by_frame[f]
            cands = [(t, lf, li) for t, (lf, li) in last.items() if f - lf <= 1 + max_gap]
            assigned = {}
            if cands:
                prev = np.array([dets[li]["bbox"] for _, _, li in cands], dtype=float)
                if homs is not None:
                    prev = np.concatenate([warp_boxes(prev[k:k + 1], _chain(homs, video, lf, f))
                                           for k, (_, lf, _) in enumerate(cands)])
                curb = np.array([dets[i]["bbox"] for i in cur], dtype=float)
                iou = iou_matrix(prev, curb)
                ok = iou >= link_iou
                key = iou.copy()
                if center_dist is not None:
                    cp = prev[:, :2] + prev[:, 2:] / 2
                    cc = curb[:, :2] + curb[:, 2:] / 2
                    size = np.sqrt((np.sqrt(prev[:, 2] * prev[:, 3])[:, None] * np.sqrt(curb[:, 2] * curb[:, 3])[None]))
                    dist = np.linalg.norm(cp[:, None] - cc[None], axis=2) / (size + 1e-9)
                    area_ratio = (prev[:, 2] * prev[:, 3])[:, None] / (curb[:, 2] * curb[:, 3] + 1e-9)[None]
                    near = (dist <= center_dist) & (area_ratio > 0.25) & (area_ratio < 4)
                    # IoU matches first (key >= link_iou), then centre-distance matches ranked by closeness (key < link_iou)
                    key = np.where(ok, iou, np.where(near, link_iou * (1 - dist / (center_dist + 1e-9)) * 0.999, 0))
                    ok = ok | near
                if max_gap:
                    key = key - 1e-3 * np.array([f - lf - 1 for _, lf, _ in cands])[:, None]   # prefer recent tracks
                pairs = np.argwhere(ok)
                pairs = pairs[np.argsort(-key[pairs[:, 0], pairs[:, 1]], kind="stable")]
                used_p, used_c = set(), set()
                for p, c in pairs:
                    if p in used_p or c in used_c:
                        continue
                    used_p.add(p); used_c.add(c)
                    assigned[c] = p
            for c, i in enumerate(cur):
                if c in assigned:
                    t, lf, _ = cands[assigned[c]]
                    if f - lf > 1:
                        gaps.append((t, lf, f))
                else:
                    t = next_id
                    next_id += 1
                track[i] = t
                last[t] = (f, i)
    return track, gaps


def smooth_boxes(dets, track, sigma):
    """Gaussian smoothing (over frames, sigma in frames) of each linked track's box coordinates. Returns new bboxes."""
    boxes = np.array([d["bbox"] for d in dets], dtype=float)
    if not sigma:
        return boxes
    members = defaultdict(list)
    for i, t in enumerate(track):
        if t >= 0:
            members[t].append(i)
    out = boxes.copy()
    for idx in members.values():
        if len(idx) < 2:
            continue
        idx = sorted(idx, key=lambda i: dets[i]["frame"])
        fr = np.array([dets[i]["frame"] for i in idx], dtype=float)
        w = np.exp(-0.5 * ((fr[:, None] - fr[None]) / sigma) ** 2)
        w /= w.sum(1, keepdims=True)
        out[idx] = w @ boxes[idx]
    return out


def temporal_v2(dets, homs=None, link_iou=0.5, link_min=0.05, lam=0.7, top_k=15, min_len=3, single_penalty=1.0,
                motion=False, center_dist=None, max_gap=0, fill=False, fill_decay=0.5, smooth_sigma=0.0,
                frame_ids=None):
    """Version 2 of the rescoring; returns a new detection list (rescored, optionally smoothed, plus filled boxes).

    homs: from estimate_homographies (needed when motion=True). frame_ids: {(video, frame): image_id}, needed for
    fill=True (a filled box needs the image id of the frame it is inserted in).
    """
    track, gaps = link_tracks_v2(dets, link_iou, link_min, homs if motion else None, center_dist, max_gap)
    scores = rescore(dets, track, lam, top_k, min_len, single_penalty)
    boxes = smooth_boxes(dets, track, smooth_sigma)
    out = [{**d, "score": float(s), "bbox": [float(v) for v in b]} for d, s, b in zip(dets, scores, boxes)]
    if fill and gaps:
        length = np.bincount(track[track >= 0])
        by_track_frame = {(t, dets[i]["frame"]): i for i, t in enumerate(track) if t >= 0}
        for t, f0, f1 in gaps:
            if length[t] < min_len:
                continue
            i0, i1 = by_track_frame[(t, f0)], by_track_frame[(t, f1)]
            d0 = dets[i0]
            video = d0["video"]
            b0, b1 = boxes[i0:i0 + 1], boxes[i1:i1 + 1]
            for g in range(f0 + 1, f1):
                if (video, g) not in frame_ids:
                    continue
                a = (g - f0) / (f1 - f0)
                # both ends moved into frame g, then interpolated
                fwd = warp_boxes(b0, _chain(homs, video, f0, g)) if motion else b0
                bwd = warp_boxes(b1, np.linalg.inv(_chain(homs, video, g, f1))) if motion else b1
                b = ((1 - a) * fwd + a * bwd)[0]
                out.append({"image_id": frame_ids[(video, g)], "video": video, "frame": g,
                            "category_id": d0["category_id"], "bbox": [float(v) for v in b],
                            "score": float(fill_decay * min(scores[i0], scores[i1])), "filled": True})
    return out


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
