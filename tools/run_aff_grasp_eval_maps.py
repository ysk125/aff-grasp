#!/usr/bin/env python3
"""Run Aff-Grasp evaluation and save qualitative segmentation maps.

Run this from the repository root after preparing the upstream
`affordance-learning` directory:

    python tools/run_aff_grasp_eval_maps.py --gpu 0 --save-raw
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
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


def _as_int_pair(ori_size) -> tuple[int, int]:
    if isinstance(ori_size, (list, tuple)):
        return int(ori_size[0]), int(ori_size[1])
    values = ori_size.detach().cpu().numpy().tolist()
    return int(values[0]), int(values[1])


def _as_name(img_name) -> str:
    if isinstance(img_name, (list, tuple)):
        return str(img_name[0])
    return str(img_name)


def _mask_to_rgb(mask: np.ndarray) -> Image.Image:
    mask = np.asarray(mask, dtype=np.int64)
    mask = np.clip(mask, 0, len(PALETTE) - 1)
    return Image.fromarray(PALETTE[mask], mode="RGB")


def _overlay(image: Image.Image, mask_rgb: Image.Image, alpha: float) -> Image.Image:
    image = image.convert("RGB")
    mask_rgb = mask_rgb.convert("RGB").resize(image.size, resample=Image.Resampling.NEAREST)
    return Image.blend(image, mask_rgb, alpha)


def _panel(original: Image.Image, overlay: Image.Image, pred: Image.Image, gt: Image.Image) -> Image.Image:
    tiles = [
        ("image", original.convert("RGB")),
        ("pred overlay", overlay.convert("RGB")),
        ("pred mask", pred.convert("RGB")),
        ("gt mask", gt.convert("RGB")),
    ]
    w, h = tiles[0][1].size
    label_h = 24
    out = Image.new("RGB", (w * len(tiles), h + label_h), "white")
    draw = ImageDraw.Draw(out)
    for idx, (label, tile) in enumerate(tiles):
        x = idx * w
        out.paste(tile, (x, label_h))
        draw.text((x + 6, 5), label, fill=(0, 0, 0))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aff-root", default="affordance-learning")
    parser.add_argument("--data-root", default="ag_dataset")
    parser.add_argument("--test-dir", default="Affordance_Evaluation_Dataset")
    parser.add_argument("--model-file", default="pretrained_aff_grasp.pth")
    parser.add_argument("--output-dir", default="qualitative_outputs/affordance_eval")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--crop-size", type=int, default=448)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--save-raw", action="store_true", help="Save raw normalized prediction maps as .npy.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    aff_root = (repo_root / args.aff_root).resolve()
    if not aff_root.exists():
        raise FileNotFoundError(f"Missing upstream code directory: {aff_root}")

    os.chdir(aff_root)
    sys.path.insert(0, str(aff_root))

    torch.cuda.set_device("cuda:" + args.gpu)

    from data.ego_video_data import AFF_LIST, TestData
    from models.GAT import Net
    from utils.evaluation import process_seg, scores

    output_dir = (aff_root / args.output_dir).resolve()
    pred_dir = output_dir / "pred_masks"
    overlay_dir = output_dir / "pred_overlays"
    gt_dir = output_dir / "gt_masks"
    panel_dir = output_dir / "panels"
    raw_dir = output_dir / "raw_pred_npy"
    for path in (pred_dir, overlay_dir, gt_dir, panel_dir):
        path.mkdir(parents=True, exist_ok=True)
    if args.save_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    dataset = TestData(data_root=args.data_root, crop_size=args.crop_size, test_dir=args.test_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = Net().cuda()
    checkpoint = torch.load(args.model_file, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    rows = []
    preds, gts = [], []

    for step, (image, dep, ann_test, obj_name, ori_size, img_name) in enumerate(tqdm(loader)):
        if args.max_samples is not None and step >= args.max_samples:
            break

        ann_fg = ann_test[:, 1:].cuda().float()
        with torch.no_grad():
            pred = model(image.cuda(), dep.cuda())

        pred_min, pred_max = pred.min(), pred.max()
        pred_norm = (pred - pred_min) / (pred_max - pred_min + 1e-10)
        pred_flat, gt_flat = process_seg(pred_norm, ann_fg, td=args.threshold)
        preds.append(pred_flat)
        gts.append(gt_flat)

        width, height = _as_int_pair(ori_size)
        name = _as_name(img_name)
        stem = Path(name).stem
        save_stem = f"{step:04d}_{stem}"

        pred_mask = pred_flat.reshape(args.crop_size, args.crop_size)
        gt_mask = gt_flat.reshape(args.crop_size, args.crop_size)
        pred_rgb = _mask_to_rgb(pred_mask).resize((width, height), resample=Image.Resampling.NEAREST)
        gt_rgb = _mask_to_rgb(gt_mask).resize((width, height), resample=Image.Resampling.NEAREST)

        image_path = Path(dataset.data_root) / name
        original = Image.open(image_path).convert("RGB")
        pred_overlay = _overlay(original, pred_rgb, args.overlay_alpha)
        comparison = _panel(original, pred_overlay, pred_rgb, gt_rgb)

        pred_rgb.save(pred_dir / f"{save_stem}_pred.png")
        pred_overlay.save(overlay_dir / f"{save_stem}_overlay.png")
        gt_rgb.save(gt_dir / f"{save_stem}_gt.png")
        comparison.save(panel_dir / f"{save_stem}_panel.png")
        if args.save_raw:
            np.save(raw_dir / f"{save_stem}_pred_norm.npy", pred_norm.squeeze(0).detach().cpu().numpy())

        rows.append(
            {
                "index": step,
                "image": name,
                "object": obj_name[0] if isinstance(obj_name, (list, tuple)) else str(obj_name),
                "pred_mask": str(pred_dir / f"{save_stem}_pred.png"),
                "pred_overlay": str(overlay_dir / f"{save_stem}_overlay.png"),
                "gt_mask": str(gt_dir / f"{save_stem}_gt.png"),
                "panel": str(panel_dir / f"{save_stem}_panel.png"),
            }
        )

    metrics = scores(gts, preds, n_class=len(AFF_LIST) + 1, ignore_zero=True) if rows else {}
    summary = {
        "num_samples": len(rows),
        "threshold": args.threshold,
        "class_names": AFF_LIST,
        "mean_iou": float(metrics.get("Mean IoU", 0.0)) if metrics else None,
        "mean_accuracy": float(metrics.get("Mean Accuracy", 0.0)) if metrics else None,
        "f1": float(metrics.get("F1", 0.0)) if metrics else None,
    }

    with (output_dir / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["index"])
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved qualitative outputs to: {output_dir}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
