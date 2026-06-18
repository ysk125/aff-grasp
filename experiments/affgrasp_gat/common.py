#!/usr/bin/env python3
"""Shared GAT retraining and dataset-validation utilities."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


AFF_LIST = ["grasp", "cut", "scoop", "pound", "support", "screw", "contain", "stick"]
CLASS_NAMES = ["background", *AFF_LIST]
AFFORDANCE_BY_OBJECT = {
    "knife": "cut",
    "scissors": "cut",
    "spoon": "scoop",
    "ladle": "scoop",
    "fork": "stick",
    "hammer": "pound",
    "spatula": "support",
    "shovel": "support",
    "trowel": "support",
    "screwdriver": "screw",
    "pan": "contain",
    "cup": "contain",
}
CLASS_ID_BY_OBJECT = {
    object_name: AFF_LIST.index(affordance) + 1
    for object_name, affordance in AFFORDANCE_BY_OBJECT.items()
}
RGB_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
RGB_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
DEPTH_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
DEPTH_STD = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
PALETTE = np.array(
    [
        [0, 0, 0],
        [129, 127, 38],
        [120, 69, 125],
        [53, 125, 34],
        [0, 11, 123],
        [118, 20, 12],
        [122, 81, 25],
        [241, 134, 51],
        [128, 128, 128],
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class TrainSample:
    image_path: Path
    depth_path: Path
    label_path: Path
    object_name: str


@dataclass(frozen=True)
class AedSample:
    image_path: Path
    depth_path: Path
    label_path: Path


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def object_from_name(name: str) -> str | None:
    noun = name.lower().split("_", 1)[0].split("-", 1)[0]
    return noun if noun in CLASS_ID_BY_OBJECT else None


def resolve_depth_root(data_root: Path, depth_root: Path | None = None) -> Path:
    if depth_root is not None:
        return depth_root.resolve()
    candidates = [
        data_root / "depth",
        data_root.parent / "depth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def discover_train_samples(
    data_root: Path,
    depth_root: Path | None = None,
    depth_gray: bool = True,
    allow_missing: bool = False,
) -> tuple[list[TrainSample], list[dict]]:
    data_root = data_root.resolve()
    rgb_root = data_root / "ego_train"
    resolved_depth_root = resolve_depth_root(data_root, depth_root)
    if not rgb_root.exists():
        raise FileNotFoundError(f"Missing training RGB directory: {rgb_root}")
    samples: list[TrainSample] = []
    missing: list[dict] = []
    suffix = "-img_graydepth.png" if depth_gray else "-img_depth.png"
    for image_path in sorted(rgb_root.glob("*-img.jpg")):
        object_name = object_from_name(image_path.name)
        depth_name = image_path.name.replace("-img.jpg", suffix)
        label_name = image_path.name.replace("-img.jpg", "-label.png")
        depth_path = resolved_depth_root / depth_name
        label_path = rgb_root / label_name
        row = {
            "image": str(image_path),
            "depth": str(depth_path),
            "label": str(label_path),
            "object": object_name or "",
            "depth_exists": depth_path.exists(),
            "label_exists": label_path.exists(),
            "known_object": object_name is not None,
        }
        if depth_path.exists() and label_path.exists() and object_name is not None:
            samples.append(TrainSample(image_path, depth_path, label_path, object_name))
        else:
            missing.append(row)
    if not samples and not allow_missing:
        raise RuntimeError(
            f"No valid GAT train samples found. rgb_root={rgb_root}, depth_root={resolved_depth_root}"
        )
    return samples, missing


def discover_aed_samples(aed_root: Path) -> list[AedSample]:
    aed_root = aed_root.resolve()
    image_root = aed_root / "JPEGImages"
    depth_root = aed_root / "depth" / "depth_gray"
    label_root = aed_root / "SegmentationClassNpy"
    if not image_root.exists():
        raise FileNotFoundError(f"Missing AED image directory: {image_root}")
    samples = []
    for image_path in sorted(image_root.glob("*.jpg")):
        depth_path = depth_root / image_path.name.replace(".jpg", "_graydepth.png")
        label_path = label_root / image_path.name.replace(".jpg", ".npy")
        if depth_path.exists() and label_path.exists():
            samples.append(AedSample(image_path, depth_path, label_path))
    if not samples:
        raise RuntimeError(f"No valid AED samples found under {aed_root}")
    return samples


def label_to_class_index(raw: np.ndarray, object_name: str) -> np.ndarray:
    if raw.ndim == 3:
        raw = raw[..., 0]
    if object_name not in CLASS_ID_BY_OBJECT:
        raise ValueError(f"Unknown object name: {object_name}")
    out = np.zeros(raw.shape, dtype=np.uint8)
    out[raw == 128] = 1
    out[raw == 255] = CLASS_ID_BY_OBJECT[object_name]
    unexpected = sorted(set(np.unique(raw).astype(int)) - {0, 128, 255})
    if unexpected:
        raise ValueError(f"Unexpected training label values for {object_name}: {unexpected}")
    return out


def class_index_to_one_hot(mask: np.ndarray, num_classes: int = 9) -> torch.Tensor:
    tensor = torch.from_numpy(mask.astype(np.int64))
    return F.one_hot(tensor, num_classes=num_classes).permute(2, 0, 1).float()


def _resize_image(image: Image.Image, size: int, nearest: bool = False) -> Image.Image:
    resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.BICUBIC
    return image.resize((size, size), resampling)


def _to_tensor(image: Image.Image, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor - mean) / std


def apply_train_transform(
    image: Image.Image,
    depth: Image.Image,
    mask: Image.Image,
    resize_size: int,
    crop_size: int,
    train: bool,
    hflip_prob: float,
    vflip_prob: float,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    image = _resize_image(image, resize_size)
    depth = _resize_image(depth, resize_size)
    mask = _resize_image(mask, resize_size, nearest=True)
    if resize_size < crop_size:
        raise ValueError(f"resize_size must be >= crop_size, got {resize_size} < {crop_size}")
    if resize_size == crop_size:
        left = top = 0
    elif train:
        left = random.randint(0, resize_size - crop_size)
        top = random.randint(0, resize_size - crop_size)
    else:
        left = top = (resize_size - crop_size) // 2
    box = (left, top, left + crop_size, top + crop_size)
    image, depth, mask = image.crop(box), depth.crop(box), mask.crop(box)
    if train and random.random() < hflip_prob:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if train and random.random() < vflip_prob:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        depth = depth.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return image, depth, mask


class GatTrainDataset(Dataset):
    def __init__(
        self,
        samples: list[TrainSample],
        resize_size: int = 476,
        crop_size: int = 448,
        train: bool = True,
        augment: bool = True,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
    ):
        self.samples = samples
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.train = train
        self.hflip_prob = hflip_prob if augment else 0.0
        self.vflip_prob = vflip_prob if augment else 0.0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        depth = Image.open(sample.depth_path).convert("RGB")
        raw_label = Image.open(sample.label_path)
        image, depth, raw_label = apply_train_transform(
            image,
            depth,
            raw_label,
            self.resize_size,
            self.crop_size,
            self.train,
            self.hflip_prob,
            self.vflip_prob,
        )
        mask = label_to_class_index(np.asarray(raw_label), sample.object_name)
        return (
            _to_tensor(image, RGB_MEAN, RGB_STD),
            _to_tensor(depth, DEPTH_MEAN, DEPTH_STD),
            class_index_to_one_hot(mask),
        )


class GatAedDataset(Dataset):
    def __init__(self, samples: list[AedSample], crop_size: int = 448):
        self.samples = samples
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = _resize_image(Image.open(sample.image_path).convert("RGB"), self.crop_size)
        depth = _resize_image(Image.open(sample.depth_path).convert("RGB"), self.crop_size)
        label = np.load(sample.label_path).astype(np.uint8)
        label_image = _resize_image(Image.fromarray(label, mode="L"), self.crop_size, nearest=True)
        label_arr = np.asarray(label_image, dtype=np.uint8)
        return {
            "image": _to_tensor(image, RGB_MEAN, RGB_STD),
            "depth": _to_tensor(depth, DEPTH_MEAN, DEPTH_STD),
            "target": torch.from_numpy(label_arr.astype(np.int64)),
            "name": sample.image_path.name,
        }


def prediction_to_class_map(pred: torch.Tensor, threshold: float) -> torch.Tensor:
    if pred.ndim != 4:
        raise ValueError(f"Expected [B, 8, H, W] prediction, got {tuple(pred.shape)}")
    pred_min = pred.amin(dim=(1, 2, 3), keepdim=True)
    pred_max = pred.amax(dim=(1, 2, 3), keepdim=True)
    sim = (pred - pred_min) / (pred_max - pred_min + 1e-10)
    max_idx = sim.argmax(dim=1) + 1
    background = (sim < threshold).all(dim=1)
    max_idx[background] = 0
    return max_idx


@dataclass
class MetricState:
    confusion: np.ndarray

    @classmethod
    def create(cls) -> "MetricState":
        return cls(np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64))

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_np = pred.detach().cpu().numpy().astype(np.int64).ravel()
        target_np = target.detach().cpu().numpy().astype(np.int64).ravel()
        valid = (target_np >= 0) & (target_np < len(CLASS_NAMES))
        hist = np.bincount(
            len(CLASS_NAMES) * target_np[valid] + pred_np[valid],
            minlength=len(CLASS_NAMES) ** 2,
        )
        self.confusion += hist.reshape(len(CLASS_NAMES), len(CLASS_NAMES))

    def compute(self) -> dict:
        cm = self.confusion.astype(np.float64)
        diag = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - diag
        iou = diag / np.maximum(union, 1)
        recall = diag / np.maximum(cm.sum(axis=1), 1)
        precision = diag / np.maximum(cm.sum(axis=0), 1)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        present = cm.sum(axis=1) > 0
        foreground = present.copy()
        foreground[0] = False
        return {
            "mIoU": float(np.mean(iou[foreground])) if foreground.any() else 0.0,
            "F1": float(np.mean(np.nan_to_num(f1[1:], nan=0.0))),
            "Mean Accuracy": float(np.mean(recall[foreground])) if foreground.any() else 0.0,
            "Class IoU": iou.tolist(),
        }


def denormalize_rgb(image: torch.Tensor) -> Image.Image:
    tensor = (image.detach().cpu() * RGB_STD + RGB_MEAN).clamp(0, 1)
    arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def mask_to_rgb(mask: np.ndarray) -> Image.Image:
    mask = np.clip(mask.astype(np.int64), 0, len(PALETTE) - 1)
    return Image.fromarray(PALETTE[mask], mode="RGB")


def overlay_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    alpha = np.zeros((*mask.shape, 4), dtype=np.uint8)
    alpha[mask] = (*color, 110)
    overlay = Image.fromarray(alpha, mode="RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def save_validation_panel(path: Path, sample: TrainSample, resize: int = 224) -> None:
    image = Image.open(sample.image_path).convert("RGB").resize((resize, resize), Image.Resampling.BICUBIC)
    depth = Image.open(sample.depth_path).convert("RGB").resize((resize, resize), Image.Resampling.BICUBIC)
    raw = np.asarray(Image.open(sample.label_path))
    if raw.ndim == 3:
        raw = raw[..., 0]
    raw_img = Image.fromarray(raw.astype(np.uint8), mode="L").resize((resize, resize), Image.Resampling.NEAREST).convert("RGB")
    raw_small = np.asarray(Image.fromarray(raw.astype(np.uint8), mode="L").resize((resize, resize), Image.Resampling.NEAREST))
    tiles = [
        ("rgb", image),
        ("depth", depth),
        ("raw label", raw_img),
        ("grasp", overlay_mask(image, raw_small == 128, (0, 190, 255))),
        (AFFORDANCE_BY_OBJECT[sample.object_name], overlay_mask(image, raw_small == 255, (255, 80, 0))),
    ]
    label_h = 22
    canvas = Image.new("RGB", (resize * len(tiles), resize + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, tile) in enumerate(tiles):
        canvas.paste(tile, (idx * resize, label_h))
        draw.text((idx * resize + 4, 4), label, fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def dataset_statistics(samples: list[TrainSample]) -> dict:
    label_value_counts: Counter[int] = Counter()
    noun_counts: Counter[str] = Counter()
    class_image_counts: Counter[str] = Counter()
    class_pixel_counts: Counter[str] = Counter()
    missing_grasp = []
    missing_functional = []
    for sample in samples:
        noun_counts[sample.object_name] += 1
        raw = np.asarray(Image.open(sample.label_path))
        if raw.ndim == 3:
            raw = raw[..., 0]
        values, counts = np.unique(raw, return_counts=True)
        for value, count in zip(values, counts):
            label_value_counts[int(value)] += int(count)
        mask = label_to_class_index(raw, sample.object_name)
        present_classes = set(np.unique(mask).astype(int))
        for class_id in sorted(present_classes - {0}):
            class_image_counts[CLASS_NAMES[class_id]] += 1
        if 1 not in present_classes:
            missing_grasp.append(str(sample.label_path))
        functional_id = CLASS_ID_BY_OBJECT[sample.object_name]
        if functional_id not in present_classes:
            missing_functional.append(str(sample.label_path))
        pixel_counts = np.bincount(mask.ravel(), minlength=len(CLASS_NAMES))
        for class_id, count in enumerate(pixel_counts):
            class_pixel_counts[CLASS_NAMES[class_id]] += int(count)
    return {
        "train_samples": len(samples),
        "noun_counts": dict(sorted(noun_counts.items())),
        "label_value_pixel_counts": {str(k): v for k, v in sorted(label_value_counts.items())},
        "class_image_counts": {name: class_image_counts[name] for name in CLASS_NAMES[1:]},
        "class_pixel_counts": {name: class_pixel_counts[name] for name in CLASS_NAMES},
        "missing_grasp_labels": missing_grasp,
        "missing_functional_labels": missing_functional,
        "expected_sample_count_note": "Public run is expected around 329-336 samples; current public data may differ.",
    }


def trainable_parameter_rows(model: torch.nn.Module) -> list[dict]:
    rows = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            rows.append({"name": name, "shape": list(param.shape), "parameters": int(param.numel())})
    return rows


def is_expected_lora_coverage(rows: list[dict]) -> bool:
    qkv_lora = [row["name"] for row in rows if "qkv" in row["name"] and "lora" in row["name"].lower()]
    blocks = set()
    for name in qkv_lora:
        parts = name.split(".")
        for idx, part in enumerate(parts[:-1]):
            if part == "blocks" and parts[idx + 1].isdigit():
                blocks.add(int(parts[idx + 1]))
    return blocks == set(range(12))


def cosine_lr(epoch: int, epochs: int, base_lr: float, eta_min: float = 0.0) -> float:
    return eta_min + 0.5 * (base_lr - eta_min) * (1.0 + math.cos(math.pi * epoch / epochs))

