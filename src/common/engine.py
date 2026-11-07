"""Training and evaluation loop shared by the torchvision detectors (RetinaNet and later models).

Every model trained through `run_experiment` uses the same data, schedule shape, evaluator and output layout:
`<runs_dir>/<name>/{history.json, summary.json, best.pt, val_dets.json}`.
"""
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .data import CocoDetection, build_train_aug, collate, seed_worker_aug, to_model_input
from .evaluate import evaluate_detections, to_coco_detections


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(paths, protocol, seed):
    """Train loader (processed, augmented frames) and val loader (original val frames from gt_eval_coco.json)."""
    aug = build_train_aug(seed)
    train_ds = CocoDetection(paths["proc_root"] / "train" / "annotations_coco.json", paths["proc_root"] / "train", aug)
    if protocol["train_step"] > 1:
        train_ds = Subset(train_ds, range(0, len(train_ds), protocol["train_step"]))

    val_ds = CocoDetection(paths["gt_file"], paths["data_root"])
    if protocol.get("val_max_images"):
        val_ds = Subset(val_ds, range(min(protocol["val_max_images"], len(val_ds))))
    base = val_ds.dataset if isinstance(val_ds, Subset) else val_ds
    idx = val_ds.indices if isinstance(val_ds, Subset) else range(len(base))
    val_ids = [base.images[i]["id"] for i in idx]

    nw = protocol["num_workers"]
    train_dl = DataLoader(train_ds, batch_size=protocol["batch_size"], shuffle=True, num_workers=nw,
                          collate_fn=collate, generator=torch.Generator().manual_seed(seed),
                          worker_init_fn=seed_worker_aug(aug, seed))
    val_dl = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=nw, collate_fn=collate)
    return train_dl, val_dl, val_ids


def warmup_cosine(total_iters, warmup_iters):
    """LR multiplier: linear warm-up from 0.1% to 100%, then cosine decay to 0."""
    def f(it):
        if it < warmup_iters:
            return 0.001 + (1 - 0.001) * it / warmup_iters
        progress = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return f


def train_one_epoch(model, optimizer, scheduler, loader, scaler, device, protocol, epoch):
    model.train()
    sums, n = defaultdict(float), 0
    use_amp = protocol["amp"] and device.type == "cuda"
    for images, targets in tqdm(loader, desc=f"epoch {epoch}", leave=False):
        images = to_model_input(images, device)
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
        with torch.autocast(device.type, enabled=use_amp):
            losses = model(images, targets)
            loss = sum(losses.values())
        if not torch.isfinite(loss):
            print("non-finite loss, batch skipped:", {k: v.item() for k, v in losses.items()})
            continue
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()   # a 'scheduler before optimizer' warning on the first step is expected with AMP
        for k, v in losses.items():
            sums[k] += v.item()
        sums["loss"] += loss.item()
        n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}


@torch.no_grad()
def predict(model, loader, device):
    """COCO-format detections in original-image pixels."""
    model.eval()
    dets = []
    for images, targets in tqdm(loader, desc="predict", leave=False):
        outputs = model(to_model_input(images, device))
        dets += to_coco_detections(outputs, [t["image_id"] for t in targets])
    return dets


def run_experiment(name, cfg, build_fn, protocol, paths, device, family, use_wandb=False):
    """Train `build_fn(cfg)` under `protocol`, evaluate on val after every epoch, keep the best epoch.

    paths: dict with proc_root, data_root, gt_file, runs_dir (Path objects).
    family: model family name, used for the W&B run name (e.g. "retinanet").
    """
    out = Path(paths["runs_dir"]) / name
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(protocol["seed"])

    model = build_fn(cfg).to(device)
    train_dl, val_dl, val_ids = make_loaders(paths, protocol, protocol["seed"])
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=protocol["lr"], momentum=protocol["momentum"],
                                weight_decay=protocol["weight_decay"])
    total = protocol["epochs"] * len(train_dl)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(total, protocol["warmup_iters"]))
    scaler = torch.amp.GradScaler(device.type, enabled=protocol["amp"] and device.type == "cuda")

    run = None
    if use_wandb:
        import wandb
        run = wandb.init(project="ucu-visdrone-detection", name=f"{family}/{name}",
                         config={**cfg, **protocol}, reinit=True)

    history, best = [], None
    print(f"[{name}] {len(train_dl.dataset)} train images, {len(val_ids)} val images, {total} iterations")
    for epoch in range(1, protocol["epochs"] + 1):
        t0 = time.time()
        losses = train_one_epoch(model, optimizer, scheduler, train_dl, scaler, device, protocol, epoch)
        dets = predict(model, val_dl, device)
        metrics = evaluate_detections(paths["gt_file"], dets, image_ids=val_ids)
        row = {"epoch": epoch, "minutes": (time.time() - t0) / 60, **losses, **metrics}
        history.append(row)
        print(f"  epoch {epoch}: loss {losses.get('loss', float('nan')):.3f} | mAP {metrics['mAP']:.4f} "
              f"AP50 {metrics['AP50']:.4f} AP_small {metrics['AP_small']:.4f} | {row['minutes']:.1f} min")
        if best is None or metrics["mAP"] > best["mAP"]:
            best = row
            torch.save(model.state_dict(), out / "best.pt")
            with open(out / "val_dets.json", "w") as f:
                json.dump(dets, f)
        with open(out / "history.json", "w") as f:
            json.dump(history, f, indent=1)
        if run:
            run.log(row)

    def plain(v):
        return [plain(x) for x in v] if isinstance(v, (list, tuple)) else v
    summary = {"name": name, "config": {k: plain(v) for k, v in cfg.items()}, "protocol": protocol, "best": best}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    if run:
        run.finish()
    return history


def load_summaries(runs_dir):
    """{experiment name: summary dict} for every finished run in `runs_dir`."""
    runs = {}
    for f in sorted(Path(runs_dir).glob("*/summary.json")):
        runs[f.parent.name] = json.loads(f.read_text())
    return runs
