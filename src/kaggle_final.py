"""Final pipeline on Kaggle: 09 (val ablation + choice of the fusion) -> test predictions -> temporal rescoring.

Kaggle inputs: the competition, visdrone-hw2-proc (data_processed), the src dataset with this file, and the models
dataset (models_final.zip: runs/yolo/*/best.pt, runs/rtdetr/*/best.pt, runs/pipeline/sub_final_960_s025_flip.csv).
Notebook cell:
    import glob, subprocess
    launcher = [p for p in glob.glob("/kaggle/input/**/src/kaggle_final.py", recursive=True) if "kaggle_src_v6" in p][0]
    subprocess.run(["python", launcher], check=True)

Output (/kaggle/working): 09_final_pipeline.ipynb (executed, with the tables), final_config.json,
sub_final_ensemble.csv (fusion), sub_final_ensemble_temporal.csv (fusion + temporal rescoring), final_run.log.
Env for local tests: KAGGLE_INPUT, KAGGLE_WORKING, LIMIT (test frames).
"""
import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

T0 = time.time()
INPUT = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 3600:.1f}h] {msg}"
    print(line, flush=True)
    with open(WORK / "final_run.log", "a") as f:
        f.write(line + "\n")


def find(pattern):
    hits = sorted(glob.glob(str(INPUT / "**" / pattern), recursive=True))
    if not hits:
        raise FileNotFoundError(f"{pattern} not found under {INPUT}: add that input to the notebook")
    return Path(hits[0])


def run(cmd, env, cwd):
    log("$ " + " ".join(map(str, cmd)))
    subprocess.run([str(c) for c in cmd], env=env, cwd=cwd, check=True)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    data_root = find("test_images.csv").parent
    proc_root = find("val/gt_eval_coco.json").parent.parent
    models = find("runs/yolo/tuned/best.pt").parents[2]           # .../runs
    support = models / "pipeline" / "sub_final_960_s025_flip.csv"
    src_in = Path(__file__).resolve().parent
    log(f"competition: {data_root}\nprocessed: {proc_root}\nmodels: {models}\nsrc: {src_in}")

    src, runs = WORK / "src", WORK / "runs"
    shutil.copytree(src_in, src, dirs_exist_ok=True)
    for pt in sorted(models.glob("*/*/best.pt")):                  # link the weights into a writable runs/ tree
        dst = runs / pt.relative_to(models)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.symlink_to(pt)
    log(f"{len(list(runs.glob('*/*/best.pt')))} models linked")
    if not os.environ.get("KAGGLE_INPUT"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.207", "ensemble-boxes==1.0.10"],
                       check=True)

    env = {**os.environ, "SRC_DIR": str(src), "DATA_ROOT": str(data_root), "PROC_ROOT": str(proc_root),
           "RUNS_ROOT": str(runs), "MPLBACKEND": "Agg", "PASSES": "full,flip", "SKIP_TEST": "1"}
    import nbformat
    from nbclient import NotebookClient
    log("running 09_final_pipeline.ipynb (val predictions, ablation tables, choice of the fusion)")
    nb = nbformat.read(src / "09_final_pipeline.ipynb", 4)
    os.environ.update(env)                                         # the kernel inherits these
    try:
        NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(src)}}).execute()
    finally:                                                        # keep the tables even if a later cell fails
        nbformat.write(nb, WORK / "09_final_pipeline.ipynb")
    pipe = runs / ("pipeline_fixture" if os.environ.get("FIXTURE") == "1" else "pipeline")   # where 09 wrote
    shutil.copy(pipe / "final_config.json", WORK / "final_config.json")
    log("final config: " + (WORK / "final_config.json").read_text().replace("\n", " "))

    limit = ["--limit", os.environ["LIMIT"]] if os.environ.get("LIMIT") else []
    run([sys.executable, "ensemble_submission.py", "--rtdetr-weight", "1", "--config", WORK / "final_config.json", *limit],
        env, src)
    shutil.copy(runs / "pipeline" / "sub_ens_rtdetr_w1.csv", WORK / "sub_final_ensemble.csv")
    run([sys.executable, "temporal_rescore.py", WORK / "sub_final_ensemble.csv", support], env, src)

    # keep the output small: drop the copied code and the prediction caches
    shutil.rmtree(src, ignore_errors=True)
    shutil.rmtree(runs, ignore_errors=True)                         # links to the weights + caches (GBs)
    log("done: download sub_final_ensemble_temporal.csv (and sub_final_ensemble.csv) from the Output tab")


if __name__ == "__main__":
    main()
