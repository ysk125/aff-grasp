#!/usr/bin/env python3
"""Shared utilities for Aff-Grasp semantic segmentation experiments.

The code in this package intentionally lives outside the original Aff-Grasp
implementation. It reads the same assets but writes to separate experiment
outputs.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


CLASS_NAMES = ["background", "grasp", "cut", "scoop", "pound", "support", "screw", "contain", "stick"]
AFFORDANCE_BY_OBJECT = {
    "knife": 2,
    "scissors": 2,
    "spoon": 3,
    "ladle": 3,
    "fork": 8,
    "hammer": 4,
    "spatula": 5,
    "shovel": 5,
    "trowel": 5,
    "screwdriver": 6,
    "pan": 7,
    "cup": 7,
}
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
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.EXPERIMENT)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def object_from_label(path: Path) -> str | None:
    noun = path.name.lower().split("_", 1)[0].split("-", 1)[0]
    return noun if noun in AFFORDANCE_BY_OBJECT else None


def training_label_to_class_index(raw: np.ndarray, label_path: Path) -> np.ndarray:
    if raw.ndim == 3:
        raw = raw[..., 0]
    obj = object_from_label(label_path)
    if obj is None:
        raise ValueError(f"Could not infer object affordance from label path: {label_path}")
    out = np.zeros(raw.shape, dtype=np.uint8)
    out[raw == 128] = 1
    out[raw == 255] = AFFORDANCE_BY_OBJECT[obj]
    return out


def discover_train_samples(train_root: Path) -> list[tuple[Path, Path]]:
    samples = []
    for label_path in sorted(train_root.rglob("*-label.png")):
        if label_path.name.endswith(".metadata"):
            continue
        image_path = label_path.with_name(label_path.name.replace("-label.png", "-img.jpg"))
        if image_path.exists() and object_from_label(label_path) is not None:
            samples.append((image_path, label_path))
    if not samples:
        raise RuntimeError(f"No train samples found under {train_root}")
    return samples


def discover_aed_samples(aed_root: Path) -> list[tuple[Path, Path]]:
    samples = []
    for image_path in sorted((aed_root / "JPEGImages").glob("*.jpg")):
        gt_path = aed_root / "SegmentationClassNpy" / f"{image_path.stem}.npy"
        if gt_path.exists():
            samples.append((image_path, gt_path))
    if not samples:
        raise RuntimeError(f"No AED samples found under {aed_root}")
    return samples


def ensure_splits(
    split_dir: Path,
    train_root: Path,
    aed_root: Path,
    val_ratio: float = 0.15,
    seed: int = 1311,
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    train_file, val_file, test_file = split_dir / "train.txt", split_dir / "val.txt", split_dir / "test.txt"
    if train_file.exists() and val_file.exists() and test_file.exists():
        return
    train_samples = discover_train_samples(train_root)
    rng = random.Random(seed)
    indices = list(range(len(train_samples)))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(indices) * val_ratio)))
    val_indices = set(indices[:val_count])
    train_rows, val_rows = [], []
    for idx, (image_path, label_path) in enumerate(train_samples):
        row = f"{image_path}\t{label_path}"
        (val_rows if idx in val_indices else train_rows).append(row)
    test_rows = [f"{image_path}\t{gt_path}" for image_path, gt_path in discover_aed_samples(aed_root)]
    train_file.write_text("\n".join(train_rows) + "\n")
    val_file.write_text("\n".join(val_rows) + "\n")
    test_file.write_text("\n".join(test_rows) + "\n")


def read_split(path: Path) -> list[tuple[Path, Path]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        image_path, target_path = line.split("\t")
        rows.append((Path(image_path), Path(target_path)))
    return rows


def resize_and_crop(
    image: Image.Image,
    target: Image.Image,
    resize_size: int,
    crop_size: int,
    train: bool,
) -> tuple[Image.Image, Image.Image]:
    image = image.convert("RGB").resize((resize_size, resize_size), Image.Resampling.BICUBIC)
    target = target.resize((resize_size, resize_size), Image.Resampling.NEAREST)
    if resize_size == crop_size:
        return image, target
    max_offset = resize_size - crop_size
    if train:
        left = random.randint(0, max_offset)
        top = random.randint(0, max_offset)
    else:
        left = max_offset // 2
        top = max_offset // 2
    box = (left, top, left + crop_size, top + crop_size)
    return image.crop(box), target.crop(box)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor - MEAN) / STD


class AffGraspSegDataset(Dataset):
    def __init__(self, rows: list[tuple[Path, Path]], resize_size: int, crop_size: int, train: bool, source: str):
        self.rows = rows
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.train = train
        self.source = source

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        image_path, target_path = self.rows[idx]
        image = Image.open(image_path).convert("RGB")
        if self.source == "train":
            target_arr = training_label_to_class_index(np.asarray(Image.open(target_path)), target_path)
        else:
            target_arr = np.load(target_path).astype(np.uint8)
        target = Image.fromarray(target_arr, mode="L")
        image, target = resize_and_crop(image, target, self.resize_size, self.crop_size, self.train)
        return {
            "image": image_to_tensor(image),
            "target": torch.from_numpy(np.asarray(target, dtype=np.int64)),
            "name": image_path.name,
        }


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.r = r
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, r, bias=False)
        self.lora_b = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scale


class FeatureAdapter(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SegFormerSegmentationModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ImportError as exc:
            raise RuntimeError(
                "SegFormer experiments require the transformers package. "
                "Rebuild the Docker image after pulling this update."
            ) from exc

        backbone = str(cfg.get("backbone", "mit_b5"))
        model_id = str(cfg.get("hf_model_id", "nvidia/segformer-b5-finetuned-ade-640-640"))
        local_model_path = Path(str(cfg.get("hf_model_path", "")))
        model_source = str(local_model_path) if local_model_path.is_dir() else model_id
        num_labels = len(CLASS_NAMES)
        if bool(cfg.get("pretrained", False)):
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_source,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
            )
        else:
            if backbone not in {"mit_b0", "nvidia/mit-b0", "mit_b5", "nvidia/mit-b5"}:
                raise ValueError(f"Unsupported local SegFormer backbone: {backbone}")
            config_kwargs = {}
            if backbone in {"mit_b5", "nvidia/mit-b5"}:
                config_kwargs = {
                    "depths": [3, 6, 40, 3],
                    "hidden_sizes": [64, 128, 320, 512],
                    "decoder_hidden_size": 768,
                    "num_attention_heads": [1, 2, 5, 8],
                    "sr_ratios": [8, 4, 2, 1],
                    "patch_sizes": [7, 3, 3, 3],
                    "strides": [4, 2, 2, 2],
                }
            config = SegformerConfig(
                num_labels=num_labels,
                id2label={idx: name for idx, name in enumerate(CLASS_NAMES)},
                label2id={name: idx for idx, name in enumerate(CLASS_NAMES)},
                **config_kwargs,
            )
            self.model = SegformerForSemanticSegmentation(config)
        apply_transformer_freeze_policy(self.model, cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=x).logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


class TimmSegmentationModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            cfg["backbone"],
            pretrained=bool(cfg.get("pretrained", False)),
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        channels = self.backbone.feature_info.channels()
        decoder_channels = int(cfg.get("decoder_channels", 128))
        self.adapters = nn.ModuleList(
            [
                FeatureAdapter(ch, int(cfg.get("adapter_reduction", 4))) if cfg.get("use_adapters") else nn.Identity()
                for ch in channels
            ]
        )
        self.lateral = nn.ModuleList([nn.Conv2d(ch, decoder_channels, kernel_size=1) for ch in channels])
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_channels * len(channels), decoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, len(CLASS_NAMES), kernel_size=1),
        )
        apply_timm_freeze_policy(self, cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        size = features[0].shape[-2:]
        outputs = []
        for feat, adapter, lateral in zip(features, self.adapters, self.lateral):
            feat = adapter(feat)
            feat = lateral(feat)
            if feat.shape[-2:] != size:
                feat = F.interpolate(feat, size=size, mode="bilinear", align_corners=False)
            outputs.append(feat)
        logits = self.fuse(torch.cat(outputs, dim=1))
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UPerNetHead(nn.Module):
    def __init__(self, in_channels: list[int], channels: int, num_classes: int, pool_scales=(1, 2, 3, 6), dropout: float = 0.1):
        super().__init__()
        self.ppm = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    ConvNormAct(in_channels[-1], channels, kernel_size=1, padding=0),
                )
                for scale in pool_scales
            ]
        )
        self.ppm_bottleneck = ConvNormAct(in_channels[-1] + len(pool_scales) * channels, channels)
        self.lateral = nn.ModuleList([ConvNormAct(ch, channels, kernel_size=1, padding=0) for ch in in_channels[:-1]])
        self.fpn = nn.ModuleList([ConvNormAct(channels, channels) for _ in in_channels[:-1]])
        self.fpn_bottleneck = ConvNormAct(len(in_channels) * channels, channels)
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        ppm_outputs = [features[-1]]
        for branch in self.ppm:
            pooled = branch(features[-1])
            ppm_outputs.append(F.interpolate(pooled, size=features[-1].shape[-2:], mode="bilinear", align_corners=False))
        laterals = [lateral(feat) for lateral, feat in zip(self.lateral, features[:-1])]
        laterals.append(self.ppm_bottleneck(torch.cat(ppm_outputs, dim=1)))
        for idx in range(len(laterals) - 1, 0, -1):
            laterals[idx - 1] = laterals[idx - 1] + F.interpolate(
                laterals[idx], size=laterals[idx - 1].shape[-2:], mode="bilinear", align_corners=False
            )
        fpn_outputs = [conv(laterals[idx]) for idx, conv in enumerate(self.fpn)]
        fpn_outputs.append(laterals[-1])
        target_size = fpn_outputs[0].shape[-2:]
        fpn_outputs = [
            output if output.shape[-2:] == target_size else F.interpolate(output, size=target_size, mode="bilinear", align_corners=False)
            for output in fpn_outputs
        ]
        return self.classifier(self.dropout(self.fpn_bottleneck(torch.cat(fpn_outputs, dim=1))))


class OfficialInternImageSegmentationModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        internimage_root = Path(os.environ.get("INTERNIMAGE_ROOT", "/opt/InternImage/classification"))
        module_path = internimage_root / "models" / "intern_image.py"
        if not module_path.exists():
            raise RuntimeError(
                f"Official InternImage source was not found at {module_path}. "
                "Build the Docker image with AFFGRASP_WITH_INTERNIMAGE=1."
            )
        if str(internimage_root) not in sys.path:
            sys.path.insert(0, str(internimage_root))
        spec = importlib.util.spec_from_file_location("affgrasp_official_intern_image", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load official InternImage module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ImportError as exc:
            raise RuntimeError(
                "Official InternImage or its DCNv3 CUDA extension could not be imported. "
                "Rebuild with AFFGRASP_WITH_INTERNIMAGE=1 and inspect the DCNv3 build log."
            ) from exc

        backbone_cfg = dict(cfg.get("internimage_backbone", {}))
        self.backbone = module.InternImage(num_classes=0, **backbone_cfg)
        channels = [int(ch) for ch in cfg.get("feature_channels", [80, 160, 320, 640])]
        self.adapters = nn.ModuleList(
            [FeatureAdapter(ch, int(cfg.get("adapter_reduction", 4))) if cfg.get("use_adapters") else nn.Identity() for ch in channels]
        )
        self.decode_head = UPerNetHead(
            channels,
            int(cfg.get("decoder_channels", 512)),
            len(CLASS_NAMES),
            tuple(cfg.get("pool_scales", [1, 2, 3, 6])),
        )
        if bool(cfg.get("pretrained", False)):
            self.load_ade20k_checkpoint(Path(str(cfg["checkpoint_path"])))
        apply_timm_freeze_policy(self, cfg)

    def load_ade20k_checkpoint(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"InternImage ADE20K checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint)
        backbone_state = {key.removeprefix("backbone."): value for key, value in state.items() if key.startswith("backbone.")}
        backbone_result = self.backbone.load_state_dict(backbone_state, strict=False)

        mapped = {}
        for key, value in state.items():
            target = None
            if key.startswith("decode_head.psp_modules."):
                target = key.replace("decode_head.psp_modules.", "ppm.").replace(".conv.", ".0.").replace(".bn.", ".1.")
            elif key.startswith("decode_head.bottleneck."):
                target = key.replace("decode_head.bottleneck.", "ppm_bottleneck.").replace("conv.", "0.").replace("bn.", "1.")
            elif key.startswith("decode_head.lateral_convs."):
                target = key.replace("decode_head.lateral_convs.", "lateral.").replace(".conv.", ".0.").replace(".bn.", ".1.")
            elif key.startswith("decode_head.fpn_convs."):
                target = key.replace("decode_head.fpn_convs.", "fpn.").replace(".conv.", ".0.").replace(".bn.", ".1.")
            elif key.startswith("decode_head.fpn_bottleneck."):
                target = key.replace("decode_head.fpn_bottleneck.", "fpn_bottleneck.").replace("conv.", "0.").replace("bn.", "1.")
            if target is not None:
                mapped[target] = value
        head_result = self.decode_head.load_state_dict(mapped, strict=False)
        loaded = len(backbone_state) + len(mapped)
        if loaded == 0:
            raise RuntimeError(f"No ADE20K weights were loaded from {checkpoint_path}")
        print(
            f"Loaded InternImage ADE20K weights: {loaded} tensors; "
            f"backbone missing={len(backbone_result.missing_keys)}, head missing={len(head_result.missing_keys)}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features_seq_out(x)
        features = [feat.permute(0, 3, 1, 2).contiguous() for feat in features]
        outputs = [adapter(feat) for feat, adapter in zip(features, self.adapters)]
        logits = self.decode_head(outputs)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def replace_lora_modules(
    module: nn.Module,
    target: str | list[str] | tuple[str, ...],
    r: int,
    alpha: float,
    dropout: float,
    allowed_path_tokens: list[str] | tuple[str, ...] | None = None,
    prefix: str = "",
) -> int:
    count = 0
    targets = [target] if isinstance(target, str) else list(target)
    for name, child in list(module.named_children()):
        path = f"{prefix}.{name}" if prefix else name
        allowed = allowed_path_tokens is None or any(token in path for token in allowed_path_tokens)
        if isinstance(child, nn.Linear) and allowed and any(item in name for item in targets):
            setattr(module, name, LoRALinear(child, r, alpha, dropout))
            count += 1
        else:
            count += replace_lora_modules(child, target, r, alpha, dropout, allowed_path_tokens, path)
    return count


def parse_lora_targets(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def apply_timm_freeze_policy(model: TimmSegmentationModel, cfg: dict) -> None:
    mode = cfg.get("freeze_mode", "full")
    if mode in {"frozen", "lora", "adapter"}:
        for param in model.backbone.parameters():
            param.requires_grad = False
    elif mode == "partial":
        for name, param in model.backbone.named_parameters():
            param.requires_grad = any(token in name for token in ["stages.2", "stages.3", "layers.2", "layers.3", "blocks.2", "blocks.3"])
    elif mode == "full":
        for param in model.backbone.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown freeze_mode: {mode}")
    if cfg.get("use_lora"):
        targets = parse_lora_targets(cfg.get("lora_target", "qkv"))
        count = replace_lora_modules(
            model.backbone,
            targets,
            int(cfg.get("lora_r", 8)),
            float(cfg.get("lora_alpha", 4)),
            float(cfg.get("lora_dropout", 0.1)),
        )
        if count == 0:
            raise RuntimeError(f"No LoRA target modules matched {targets!r}")
    for name in ["adapters", "lateral", "fuse", "decode_head"]:
        module = getattr(model, name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True


def apply_transformer_freeze_policy(model: nn.Module, cfg: dict) -> None:
    mode = cfg.get("freeze_mode", "full")
    encoder = model.segformer.encoder
    if mode in {"frozen", "lora", "adapter"}:
        for param in encoder.parameters():
            param.requires_grad = False
    elif mode == "partial":
        for name, param in encoder.named_parameters():
            param.requires_grad = any(token in name for token in ["block.2", "block.3", "patch_embeddings.3"])
    elif mode == "full":
        for param in encoder.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown freeze_mode: {mode}")
    if cfg.get("use_lora"):
        targets = parse_lora_targets(cfg.get("lora_target", "query,value"))
        stage_indices = cfg.get("lora_stage_indices", [2, 3])
        allowed_paths = [f"block.{int(idx)}" for idx in stage_indices]
        count = replace_lora_modules(
            encoder,
            targets,
            int(cfg.get("lora_r", 8)),
            float(cfg.get("lora_alpha", 4)),
            float(cfg.get("lora_dropout", 0.1)),
            allowed_paths,
        )
        if count == 0:
            raise RuntimeError(f"No SegFormer LoRA target modules matched {targets!r} under {allowed_paths!r}")
    if mode == "adapter":
        raise ValueError("Adapter mode is not implemented for the Transformers SegFormer backend.")
    for param in model.decode_head.parameters():
        param.requires_grad = True


def build_model(cfg: dict) -> nn.Module:
    if cfg.get("model_name") == "segformer":
        return SegFormerSegmentationModel(cfg)
    if cfg.get("model_name") == "internimage":
        backend = cfg.get("backend", "official")
        if backend == "official":
            return OfficialInternImageSegmentationModel(cfg)
        if backend == "timm":
            return TimmSegmentationModel(cfg)
        raise ValueError(f"Unknown InternImage backend: {backend}")
    return TimmSegmentationModel(cfg)


def parameter_summary(model: nn.Module) -> dict:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_ratio": trainable / total if total else 0.0,
    }


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma * ce).mean()
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(target, num_classes=len(CLASS_NAMES)).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        denominator = probs.sum(dims) + one_hot.sum(dims)
        dice = 1.0 - ((2 * intersection + 1.0) / (denominator + 1.0)).mean()
        return self.alpha * focal + dice


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
        hist = np.bincount(len(CLASS_NAMES) * target_np[valid] + pred_np[valid], minlength=len(CLASS_NAMES) ** 2)
        self.confusion += hist.reshape(len(CLASS_NAMES), len(CLASS_NAMES))

    def compute(self, ignore_background: bool = True) -> dict:
        cm = self.confusion.astype(np.float64)
        diag = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - diag
        precision = diag / np.maximum(cm.sum(axis=0), 1)
        recall = diag / np.maximum(cm.sum(axis=1), 1)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        iou = diag / np.maximum(union, 1)
        start = 1 if ignore_background else 0
        present = cm.sum(axis=1) > 0
        mask = present.copy()
        mask[:start] = False
        return {
            "mIoU": float(np.mean(iou[mask])) if mask.any() else 0.0,
            "F1": float(np.mean(f1[mask])) if mask.any() else 0.0,
            "Accuracy": float(diag.sum() / max(cm.sum(), 1)),
        }


def mask_to_rgb(mask: np.ndarray) -> Image.Image:
    mask = np.clip(mask.astype(np.int64), 0, len(PALETTE) - 1)
    return Image.fromarray(PALETTE[mask], mode="RGB")


def denormalize(image: torch.Tensor) -> Image.Image:
    tensor = (image.detach().cpu() * STD + MEAN).clamp(0, 1)
    arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def save_panel(path: Path, image: torch.Tensor, target: torch.Tensor, pred: torch.Tensor, error: torch.Tensor | None = None) -> None:
    tiles = [denormalize(image), mask_to_rgb(target.cpu().numpy()), mask_to_rgb(pred.cpu().numpy())]
    if error is not None:
        err = (error.cpu().numpy().astype(np.uint8) * 255)
        tiles.append(Image.fromarray(err, mode="L").convert("RGB"))
    w, h = tiles[0].size
    canvas = Image.new("RGB", (w * len(tiles), h), "white")
    for idx, tile in enumerate(tiles):
        canvas.paste(tile.resize((w, h), Image.Resampling.NEAREST), (idx * w, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def copy_config(config_path: Path, output_dir: Path, cfg: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        shutil.copy2(config_path, output_dir / "config.py")
    write_json(output_dir / "config.yaml", cfg)
