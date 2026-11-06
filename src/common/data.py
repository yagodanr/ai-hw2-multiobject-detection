"""Datasets and augmentation shared by the torchvision detectors."""
import json
from collections import defaultdict
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

CLASSES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]
FILL_VALUE = (114, 114, 114)  # grey used for ignored regions and padding


def build_train_aug(seed=42):
    """Train-time augmentation; same definition as in 01_preprocess.ipynb.

    Boxes are in COCO format (x, y, w, h). Boxes that end up less than 30% visible
    or under 4 px^2 are dropped.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.8, 1.2), translate_percent=(-0.05, 0.05), fill=FILL_VALUE, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["labels"], min_visibility=0.3, min_area=4),
        seed=seed,
    )


class CocoDetection(Dataset):
    """Images and boxes from a COCO json, in the format torchvision detectors expect.

    - `img_root` is the folder that `file_name` paths are relative to.
    - Crowd (ignored) regions are skipped; they matter only for evaluation.
    - Labels stay 1..8 (COCO ids), because torchvision reserves 0 for background.
    - Images are returned as uint8 tensors (C, H, W) to keep DataLoader memory low (a 4K frame is
      25 MB as uint8 vs 100 MB as float32). Convert on the GPU with `to_model_input`; the model then
      resizes and normalises them.
    """

    def __init__(self, ann_file, img_root, transforms=None):
        with open(ann_file) as f:
            coco = json.load(f)
        self.images = coco["images"]
        self.root = Path(img_root)
        self.transforms = transforms
        self.anns = defaultdict(list)
        for a in coco["annotations"]:
            if not a.get("iscrowd", 0):
                self.anns[a["image_id"]].append(a)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        img = cv2.cvtColor(cv2.imread(str(self.root / info["file_name"])), cv2.COLOR_BGR2RGB)
        anns = self.anns.get(info["id"], [])
        boxes = [a["bbox"] for a in anns]
        labels = [a["category_id"] for a in anns]

        if self.transforms is not None:
            out = self.transforms(image=img, bboxes=boxes, labels=labels)
            img, boxes, labels = out["image"], out["bboxes"], out["labels"]

        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # xywh -> xyxy
        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.as_tensor(np.asarray(labels), dtype=torch.int64).reshape(-1),
            "image_id": int(info["id"]),
        }
        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)  # uint8
        return img, target


class ImageList(Dataset):
    """Plain list of images (for the test set). Returns (uint8 image tensor, key)."""

    def __init__(self, paths, keys):
        self.paths, self.keys = list(paths), list(keys)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(str(self.paths[idx])), cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1), self.keys[idx]


def to_model_input(images, device):
    """uint8 (C, H, W) tensors -> float tensors in [0, 1] on `device` (conversion happens on the GPU)."""
    return [img.to(device, non_blocking=True).float().div_(255.0) for img in images]


def collate(batch):
    """Detection batches are lists, because images have different sizes."""
    return tuple(zip(*batch))


def seed_worker_aug(aug, base_seed):
    """worker_init_fn that gives each DataLoader worker its own, reproducible augmentation seed."""
    def init(worker_id):
        aug.set_random_seed(base_seed + worker_id)
    return init
