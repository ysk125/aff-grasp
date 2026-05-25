#!/usr/bin/env python3
"""Run Aff-Grasp evaluation and save low-IoU qualitative segmentation maps.

Run this from the repository root after preparing the upstream
`affordance-learning` directory:

    python tools/run_aff_grasp_eval_maps.py --save-filter low-iou --max-samples 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

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


def _class_iou(pred_flat: np.ndarray, gt_flat: np.ndarray, n_class: int) -> dict[int, float | None]:
    out = {}
    for cls_idx in range(n_class):
        pred_cls = pred_flat == cls_idx
        gt_cls = gt_flat == cls_idx
        union = np.logical_or(pred_cls, gt_cls).sum()
        if union == 0:
            out[cls_idx] = None
            continue
        out[cls_idx] = float(np.logical_and(pred_cls, gt_cls).sum() / union)
    return out


def _mean_present_iou(class_iou: dict[int, float | None], include_background: bool) -> float:
    start = 0 if include_background else 1
    values = [v for k, v in class_iou.items() if k >= start and v is not None]
    return float(np.mean(values)) if values else 0.0


def _should_save(
    save_filter: str,
    image_miou: float,
    foreground_iou: float,
    iou_threshold: float,
    foreground_iou_threshold: float,
) -> tuple[bool, str]:
    if save_filter == "all":
        return True, "all"
    if save_filter == "none":
        return False, "none"
    reasons = []
    if image_miou < iou_threshold:
        reasons.append(f"image_miou<{iou_threshold:.3f}")
    if foreground_iou < foreground_iou_threshold:
        reasons.append(f"foreground_iou<{foreground_iou_threshold:.3f}")
    return bool(reasons), ";".join(reasons) if reasons else "good"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aff-root", default="affordance-learning")
    parser.add_argument("--data-root", default="ag_dataset")
    parser.add_argument("--test-dir", default="Affordance_Evaluation_Dataset")
    parser.add_argument("--model-file", default="pretrained_aff_grasp.pth")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--crop-size", type=int, default=448)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--save-filter", choices=["all", "low-iou", "none"], default="low-iou")
    parser.add_argument("--iou-threshold", type=float, default=0.60)
    parser.add_argument("--foreground-iou-threshold", type=float, default=0.50)
    parser.add_argument("--always-save-top-k", type=int, default=0)
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

    if args.output_dir is None:
        from datetime import datetime

        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (aff_root / "results" / run_name / "aed").resolve()
    else:
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
    saved_count = 0
    pending_top_k = []

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
        per_class_iou = _class_iou(pred_flat, gt_flat, n_class=len(AFF_LIST) + 1)
        image_miou = _mean_present_iou(per_class_iou, include_background=False)
        foreground_iou = float(
            np.logical_and(pred_flat > 0, gt_flat > 0).sum()
            / (np.logical_or(pred_flat > 0, gt_flat > 0).sum() + 1e-12)
        )
        save_sample, save_reason = _should_save(
            args.save_filter,
            image_miou,
            foreground_iou,
            args.iou_threshold,
            args.foreground_iou_threshold,
        )

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

        paths = {
            "pred_mask": "",
            "pred_overlay": "",
            "gt_mask": "",
            "panel": "",
            "raw_pred": "",
        }
        if save_sample:
            paths["pred_mask"] = str(pred_dir / f"{save_stem}_pred.png")
            paths["pred_overlay"] = str(overlay_dir / f"{save_stem}_overlay.png")
            paths["gt_mask"] = str(gt_dir / f"{save_stem}_gt.png")
            paths["panel"] = str(panel_dir / f"{save_stem}_panel.png")
            pred_rgb.save(paths["pred_mask"])
            pred_overlay.save(paths["pred_overlay"])
            gt_rgb.save(paths["gt_mask"])
            comparison.save(paths["panel"])
            if args.save_raw:
                paths["raw_pred"] = str(raw_dir / f"{save_stem}_pred_norm.npy")
                np.save(paths["raw_pred"], pred_norm.squeeze(0).detach().cpu().numpy())
            saved_count += 1
        elif args.always_save_top_k > 0:
            pending_top_k.append(
                {
                    "rank_value": image_miou,
                    "step": step,
                    "save_stem": save_stem,
                    "pred_rgb": pred_rgb.copy(),
                    "pred_overlay": pred_overlay.copy(),
                    "gt_rgb": gt_rgb.copy(),
                    "comparison": comparison.copy(),
                    "raw_pred": pred_norm.squeeze(0).detach().cpu().numpy() if args.save_raw else None,
                }
            )

        rows.append(
            {
                "index": step,
                "image": name,
                "object": obj_name[0] if isinstance(obj_name, (list, tuple)) else str(obj_name),
                "saved": save_sample,
                "save_reason": save_reason,
                "image_miou": image_miou,
                "foreground_iou": foreground_iou,
                "class_iou_json": json.dumps(per_class_iou, sort_keys=True),
                **paths,
            }
        )

    if args.always_save_top_k > 0 and pending_top_k:
        top_k = sorted(pending_top_k, key=lambda item: item["rank_value"])[: args.always_save_top_k]
        for item in top_k:
            save_stem = item["save_stem"]
            pred_path = pred_dir / f"{save_stem}_pred.png"
            overlay_path = overlay_dir / f"{save_stem}_overlay.png"
            gt_path = gt_dir / f"{save_stem}_gt.png"
            panel_path = panel_dir / f"{save_stem}_panel.png"
            item["pred_rgb"].save(pred_path)
            item["pred_overlay"].save(overlay_path)
            item["gt_rgb"].save(gt_path)
            item["comparison"].save(panel_path)
            raw_path = ""
            if args.save_raw and item["raw_pred"] is not None:
                raw_path = str(raw_dir / f"{save_stem}_pred_norm.npy")
                np.save(raw_path, item["raw_pred"])
            for row in rows:
                if row["index"] == item["step"]:
                    row.update(
                        {
                            "saved": True,
                            "save_reason": "top-k-lowest-miou",
                            "pred_mask": str(pred_path),
                            "pred_overlay": str(overlay_path),
                            "gt_mask": str(gt_path),
                            "panel": str(panel_path),
                            "raw_pred": raw_path,
                        }
                    )
                    saved_count += 1
                    break

    metrics = scores(gts, preds, n_class=len(AFF_LIST) + 1, ignore_zero=True) if rows else {}
    summary = {
        "num_samples": len(rows),
        "threshold": args.threshold,
        "save_filter": args.save_filter,
        "iou_threshold": args.iou_threshold,
        "foreground_iou_threshold": args.foreground_iou_threshold,
        "saved_samples": saved_count,
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
