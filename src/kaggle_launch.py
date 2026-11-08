"""Run the RT-DETR experiments unattended on Kaggle ("Save & Run All").

Kaggle notebook cell:
    import glob, subprocess
    subprocess.run(["python", glob.glob("/kaggle/input/**/src/kaggle_launch.py", recursive=True)[0]], check=True)

- Finds the competition data (folder with test_images.csv) and our uploaded dataset (data_processed/ + src/) under /kaggle/input.
- Copies src/ to /kaggle/working/src (writable), installs Ultralytics, writes results to /kaggle/working/runs/rtdetr/.
- With 2 GPUs (T4 x2) runs one experiment queue per GPU in parallel; with 1 GPU runs them one after another.
- Every job has a time limit so the whole session stays under Kaggle's 12 h cap (a session that hits the cap may lose its output).
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ULTRALYTICS = "ultralytics==8.4.207"
SESSION_LIMIT_H = 11.0                  # leave an hour of Kaggle's 12 h for saving the output
GPU_QUEUES = [                          # per GPU: experiments in order; the first one is the most important
    ["baseline", "lr_2e-4"],
    ["queries_500", "imgsz_960"],
]
FALLBACK = {"baseline": "baseline_fp32"}  # rerun with AMP off if this one fails (NaNs)

T0 = time.time()
INPUT, WORK = Path("/kaggle/input"), Path("/kaggle/working")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 3600:.1f}h] {msg}", flush=True)


def find_dir(marker, max_depth=5):
    """First folder under /kaggle/input (shallow walk) that contains `marker`."""
    for root, dirs, files in os.walk(INPUT, followlinks=True):   # Kaggle inputs can be symlinks
        depth = Path(root).relative_to(INPUT).parts
        if marker in files or marker in dirs:
            return Path(root)
        if len(depth) >= max_depth or "images" in dirs and "labels" in dirs:
            dirs[:] = []                 # don't descend into image folders
        else:
            dirs[:] = [d for d in dirs if d not in ("images", "labels", "ignore")]
    raise FileNotFoundError(f"{marker} not found under {INPUT}")


def main():
    data_root = find_dir("test_images.csv")
    proc_root = find_dir("gt_eval_coco.json").parent       # .../data_processed/val -> .../data_processed
    src_in = find_dir("kaggle_launch.py")
    log(f"competition data: {data_root}\nprocessed data: {proc_root}\nsrc: {src_in}")

    src = WORK / "src"
    shutil.copytree(src_in, src, dirs_exist_ok=True)
    runs = WORK / "runs"
    (runs / "rtdetr").mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", ULTRALYTICS], check=True)

    n_gpu = int(subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
                               capture_output=True, text=True).stdout.strip() or 0)
    log(f"{n_gpu} GPU(s)")
    if n_gpu == 0 and not os.environ.get("ALLOW_CPU"):          # ALLOW_CPU=1: local smoke test only
        raise SystemExit("No GPU: set Accelerator to 'GPU T4 x2' in the notebook settings.")
    queues = GPU_QUEUES if n_gpu >= 2 else [sum(GPU_QUEUES, [])]
    n_gpu = max(n_gpu, 1)

    cpus = os.cpu_count() or 4
    env_base = {**os.environ, "SRC_DIR": str(src), "DATA_ROOT": str(data_root), "PROC_ROOT": str(proc_root),
                "RUNS_ROOT": str(runs), "MPLBACKEND": "Agg", "WORKERS": str(max(1, cpus // len(queues)))}

    def done(name):
        return (runs / "rtdetr" / name / "summary.json").exists()

    def run_gpu_queue(gpu, names):
        """Shell script for one GPU: each job gets the time left in the session; failures fall back or move on."""
        lines = []
        for name in names:
            lines.append(f"run {name}")
            if name in FALLBACK:
                lines.append(f"[ -f {runs}/rtdetr/{name}/summary.json ] || run {FALLBACK[name]}")
        script = f"""
set -u
cd {src}
deadline=$(( {int(T0 + SESSION_LIMIT_H * 3600)} ))
run() {{
  left=$(( deadline - $(date +%s) ))
  if [ $left -lt 1800 ]; then echo "skip $1: only $left s left"; return; fi
  timeout $left {sys.executable} run_queue.py 08_rtdetr.ipynb:$1 || echo "$1 stopped (exit $?)"
}}
{chr(10).join(lines)}
"""
        log_file = runs / f"rtdetr_gpu{gpu}.log"
        env = {**env_base, "CUDA_VISIBLE_DEVICES": str(gpu)}
        return subprocess.Popen(["bash", "-c", script], env=env, stdout=open(log_file, "w"), stderr=subprocess.STDOUT)

    procs = [run_gpu_queue(g, names) for g, names in enumerate(queues)]
    log(f"started {len(procs)} queue(s): {queues}; logs in {runs}/rtdetr_gpu*.log")
    while any(p.poll() is None for p in procs):
        time.sleep(300)
        status = ", ".join(f"{n}={'done' if done(n) else '...'}" for q in queues for n in q)
        log(status)

    for f in sorted(runs.glob("rtdetr_gpu*.log")):
        lines = [l for l in f.read_text(errors="ignore").replace("\r", "\n").splitlines() if l.startswith("[")]
        print(f"--- {f.name}\n" + "\n".join(lines[-20:]))
    # Kaggle keeps /kaggle/working; drop the copied code so only results are in the output
    shutil.rmtree(src, ignore_errors=True)
    log("finished: download /kaggle/working/runs/rtdetr from the notebook's Output tab")


if __name__ == "__main__":
    main()
