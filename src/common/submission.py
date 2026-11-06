"""Write predictions in the competition's CSV format.

One row per test image: `image_id,PredictionString`, where PredictionString is a space-separated list of
`class score x y width height` groups, class 0..7, (x, y) the top-left corner in original-image pixels.
At most 500 detections per image are scored.
"""
import pandas as pd

MAX_DETS = 500


def prediction_string(dets):
    """dets: iterable of (class_0_based, score, x, y, w, h). Keeps the top MAX_DETS by score."""
    dets = sorted(dets, key=lambda d: -d[1])[:MAX_DETS]
    return " ".join(f"{int(c)} {s:.4f} {x:.1f} {y:.1f} {w:.1f} {h:.1f}" for c, s, x, y, w, h in dets)


def write_submission(preds, test_csv, out_path):
    """preds: dict image_id -> list of (class_0_based, score, x, y, w, h). Missing images get an empty row."""
    test = pd.read_csv(test_csv)
    sub = pd.DataFrame({"image_id": test["image_id"],
                        "PredictionString": [prediction_string(preds.get(k, [])) for k in test["image_id"]]})
    sub.to_csv(out_path, index=False)
    return sub
