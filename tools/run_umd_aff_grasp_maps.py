#!/usr/bin/env python3
"""Run Aff-Grasp on RGB-D Part Affordance Dataset images without GT evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from tqdm import tqdm


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

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
RGB_HINTS = ("rgb", "image", "color")
DEPTH_HINTS = ("depth", "dep")


def _normalize_key(path: Path) -> str:
    stem = path.stem.lower()
    for token in (*RGB_HINTS, *DEPTH_HINTS, "registered", "crop", "mask", "label"):
        stem = re.sub(rf"(^|[_\-.]){token}($|[_\-.])", "_", stem)
    stem = re.sub(r"[^a-z0-9]+", "_", stem)
    return stem.strip("_")


def _looks_like_depth(path: Path) -> bool:
    text = "/".join(part.lower() for part in path.parts)
    return any(hint in text for hint in DEPTH_HINTS)


def _looks_like_rgb(path: Path) -> bool:
    text = "/".join(part.lower() for part in path.parts)
    return any(hint in text for hint in RGB_HINTS) and not _looks_like_depth(path)


def _discover_pairs(dataset_root: Path) -> list[tuple[Path, Path | None]]:
    images = [p for p in dataset_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    depth_by_key = {}
    rgb_candidates = []
    for path in images:
        key = _normalize_key(path)
        if _looks_like_depth(path):
            depth_by_key.setdefault(key, path)
        elif _looks_like_rgb(path) or path.suffix.lower() in {".jpg", ".jpeg"}:
            rgb_candidates.append(path)

    pairs = []
    seen = set()
    for rgb in sorted(rgb_candidates):
        key = _normalize_key(rgb)
        if rgb in seen:
            continue
        seen.add(rgb)
        pairs.append((rgb, depth_by_key.get(key)))
    return pairs


def _prepare_rgb(image: Image.Image, crop_size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((crop_size, crop_size), resample=Image.Resampling.BICUBIC)
    tensor = TF.to_tensor(image)
    return TF.normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def _prepare_depth(depth_path: Path | None, crop_size: int) -> torch.Tensor:
    if depth_path is None:
        return torch.zeros(3, crop_size, crop_size)
    depth = Image.open(depth_path).convert("RGB").resize((crop_size, crop_size), resample=Image.Resampling.BICUBIC)
    tensor = TF.to_tensor(depth)
    return TF.normalize(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))


def _mask_to_rgb(mask: np.ndarray) -> Image.Image:
    mask = np.asarray(mask, dtype=np.int64)
    mask = np.clip(mask, 0, len(PALETTE) - 1)
    return Image.fromarray(PALETTE[mask], mode="RGB")


def _overlay(image: Image.Image, mask_rgb: Image.Image, alpha: float) -> Image.Image:
    return Image.blend(image.convert("RGB"), mask_rgb.convert("RGB").resize(image.size, Image.Resampling.NEAREST), alpha)


def _panel(original: Image.Image, overlay: Image.Image, pred: Image.Image, title: str) -> Image.Image:
    tiles = [("image", original.convert("RGB")), ("pred overlay", overlay), ("pred mask", pred.convert("RGB"))]
    w, h = tiles[0][1].size
    label_h = 24
    out = Image.new("RGB", (w * len(tiles), h + label_h), "white")
    draw = ImageDraw.Draw(out)
    for idx, (label, tile) in enumerate(tiles):
        x = idx * w
        out.paste(tile, (x, label_h))
        draw.text((x + 6, 5), label if idx else title[:80], fill=(0, 0, 0))
    return out


def _pred_to_mask(pred_norm: torch.Tensor, threshold: float) -> np.ndarray:
    sim = pred_norm.squeeze(0).flatten(1).detach().cpu().numpy()
    max_idx = np.argmax(sim, axis=0)
    bg_idx = np.all(sim < threshold, axis=0)
    out = np.zeros(sim.shape[-1], dtype=np.int16)
    for idx in range(sim.shape[0]):
        out[max_idx == idx] = idx + 1
    out[bg_idx] = 0
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aff-root", default="affordance-learning")
    parser.add_argument("--dataset-root", default="datasets/umd_part_affordance")
    parser.add_argument("--model-file", default="pretrained_aff_grasp.pth")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--crop-size", type=int, default=448)
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--save-raw", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    aff_root = (repo_root / args.aff_root).resolve()
    dataset_root = (repo_root / args.dataset_root).resolve()
    if not aff_root.exists():
        raise FileNotFoundError(f"Missing upstream code directory: {aff_root}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Missing UMD dataset directory: {dataset_root}")

    os.chdir(aff_root)
    sys.path.insert(0, str(aff_root))
    torch.cuda.set_device("cuda:" + args.gpu)

    from data.ego_video_data import AFF_LIST
    from models.GAT import Net

    if args.output_dir is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (aff_root / "results" / run_name / "umd").resolve()
    else:
        output_dir = (aff_root / args.output_dir).resolve()
    pred_dir = output_dir / "pred_masks"
    overlay_dir = output_dir / "pred_overlays"
    panel_dir = output_dir / "panels"
    raw_dir = output_dir / "raw_pred_npy"
    for path in (pred_dir, overlay_dir, panel_dir):
        path.mkdir(parents=True, exist_ok=True)
    if args.save_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    pairs = _discover_pairs(dataset_root)
    if args.max_samples is not None:
        pairs = pairs[: args.max_samples]
    if not pairs:
        raise RuntimeError(f"No RGB images found under {dataset_root}")

    model = Net().cuda()
    checkpoint = torch.load(args.model_file, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    rows = []
    for idx, (rgb_path, depth_path) in enumerate(tqdm(pairs)):
        original = Image.open(rgb_path).convert("RGB")
        rgb = _prepare_rgb(original, args.crop_size).unsqueeze(0).cuda()
        depth = _prepare_depth(depth_path, args.crop_size).unsqueeze(0).cuda()
        with torch.no_grad():
            pred = model(rgb, depth)
        pred_min, pred_max = pred.min(), pred.max()
        pred_norm = (pred - pred_min) / (pred_max - pred_min + 1e-10)
        pred_mask = _pred_to_mask(pred_norm, args.threshold).reshape(args.crop_size, args.crop_size)
        pred_rgb = _mask_to_rgb(pred_mask).resize(original.size, Image.Resampling.NEAREST)
        pred_overlay = _overlay(original, pred_rgb, args.overlay_alpha)

        rel_name = rgb_path.relative_to(dataset_root).with_suffix("")
        save_stem = f"{idx:04d}_{'_'.join(rel_name.parts)}"
        pred_path = pred_dir / f"{save_stem}_pred.png"
        overlay_path = overlay_dir / f"{save_stem}_overlay.png"
        panel_path = panel_dir / f"{save_stem}_panel.png"
        raw_path = ""

        pred_rgb.save(pred_path)
        pred_overlay.save(overlay_path)
        _panel(original, pred_overlay, pred_rgb, rgb_path.name).save(panel_path)
        if args.save_raw:
            raw_path = str(raw_dir / f"{save_stem}_pred_norm.npy")
            np.save(raw_path, pred_norm.squeeze(0).detach().cpu().numpy())

        rows.append(
            {
                "index": idx,
                "rgb_path": str(rgb_path),
                "depth_path": str(depth_path) if depth_path else "",
                "has_depth": depth_path is not None,
                "pred_mask": str(pred_path),
                "pred_overlay": str(overlay_path),
                "panel": str(panel_path),
                "raw_pred": raw_path,
            }
        )

    with (output_dir / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "num_samples": len(rows),
        "threshold": args.threshold,
        "class_names": AFF_LIST,
        "dataset_root": str(dataset_root),
        "note": "UMD GT conversion is intentionally not performed in this first pass.",
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved UMD qualitative outputs to: {output_dir}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
