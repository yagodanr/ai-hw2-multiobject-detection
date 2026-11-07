"""Run notebook experiments unattended, one after another.

Usage (from the src/ folder, with the project's Python environment active):
    nohup python run_queue.py > ../runs/queue.log 2>&1 &
    python run_queue.py 02_faster_rcnn.ipynb:small_anchors,giou_loss 04_retinanet.ipynb:baseline

For each notebook, the code cells *before* the `RUN = [...]` cell (imports, protocol, model, data) are executed
once in this process; then the RUN cell's loop runs for each experiment name. Finished experiments
(`<RUNS_DIR>/<name>/summary.json` exists) are skipped, and a failing experiment is logged and the queue moves on.
"""
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")   # plots in setup cells must not need a display

DEFAULT_QUEUE = [
    ("02_faster_rcnn.ipynb", ["small_anchors", "more_rois", "giou_loss"]),
    ("04_retinanet.ipynb", ["baseline", "data_anchors", "iou_thresholds", "focal_gamma"]),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_notebook(nb_path, names):
    cells = [c for c in json.loads(Path(nb_path).read_text())["cells"] if c["cell_type"] == "code"]
    sources = ["".join(c["source"]) for c in cells]
    run_idx = next(i for i, s in enumerate(sources) if s.lstrip().startswith("RUN ="))
    loop_body = "\n".join(sources[run_idx].splitlines()[1:])   # the `for name in RUN:` loop

    ns = {"__name__": "__queue__", "display": print}
    log(f"{nb_path}: running {run_idx} setup cells")
    for s in sources[:run_idx]:
        exec(compile(s, str(nb_path), "exec"), ns)

    for name in names:
        done = Path(ns["RUNS_DIR"]) / name / "summary.json"
        if done.exists():
            log(f"{nb_path}:{name} already finished, skipped")
            continue
        log(f"{nb_path}:{name} started")
        t0 = time.time()
        try:
            ns["RUN"] = [name]
            exec(compile(loop_body, str(nb_path), "exec"), ns)
            best = json.loads(done.read_text())["best"]
            log(f"{nb_path}:{name} finished in {(time.time() - t0) / 60:.0f} min: "
                f"mAP {best['mAP']:.4f}, AP50 {best['AP50']:.4f}, AP_small {best['AP_small']:.4f}")
        except Exception:
            log(f"{nb_path}:{name} FAILED\n{traceback.format_exc()}")
        finally:
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def parse_args(argv):
    if not argv:
        return DEFAULT_QUEUE
    return [(arg.split(":")[0], arg.split(":")[1].split(",")) for arg in argv]


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)   # notebooks expect to run from src/
    queue = parse_args(sys.argv[1:])
    log(f"queue: {queue}")
    for nb, names in queue:
        run_notebook(nb, names)
    log("queue done")
