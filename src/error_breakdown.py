"""Error-type breakdown (TIDE-style) of every finished run's val predictions.

Usage, from src/:  python error_breakdown.py [score_thr]
Writes ../runs/error_breakdown.csv and prints the table.
"""
import json
import sys
from pathlib import Path

import pandas as pd

from common.analysis import GroundTruth, error_breakdown

ROOT = Path(__file__).resolve().parent.parent
SCORE_THR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3

gt = GroundTruth(ROOT / "data_processed" / "val" / "gt_eval_coco.json")
rows = {}
for f in sorted((ROOT / "runs").glob("*/*/val_dets.json")):
    name = f"{f.parent.parent.name}/{f.parent.name}"
    rows[name] = error_breakdown(gt, json.loads(f.read_text()), SCORE_THR)
    print("done", name, flush=True)

cols = ["correct", "loc", "duplicate", "class", "class+loc", "background", "ignored", "missed",
        "mean IoU of correct", "missed small %", "missed medium %", "missed large %"]
table = pd.DataFrame(rows).T.reindex(columns=cols).fillna(0)
table.to_csv(ROOT / "runs" / "error_breakdown.csv")
pd.set_option("display.width", 250)
print(f"\nDetections with score >= {SCORE_THR}:")
print(table.round(3).to_string())
