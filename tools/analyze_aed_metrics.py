#!/usr/bin/env python3
"""Compute AED region, boundary, and background false-positive metrics."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis_common import (
    CLASS_NAMES,
    aggregate_class_metrics,
    assign_rank_tertiles,
    assign_tertiles,
    binary_metrics,
    bootstrap_spearman,
    boundary_metrics,
    class_confusion,
    group_means,
    matched_gt_regions,
    read_csv,
    resolve_from,
    spearman,
    write_csv,
    write_json,
)


def _load_mask(root: Path, value: str) -> np.ndarray:
    return np.asarray(Image.open(resolve_from(root, value)), dtype=np.uint8)


def _background_complexity(rgb: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    background = gt == 0
    if not background.any():
        return 0.0, 0.0
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.hypot(grad_x, grad_y)
    return float(np.var(laplacian[background])), float(np.mean(gradient[background] >= 64.0))


def _mean_present(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _save_region_bin_plot(region_rows: list[dict], figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    dimensions = [
        ("area_bin", "GT area tertile"),
        ("bbox_aspect_ratio_bin", "BBox aspect ratio tertile"),
        ("elongation_bin", "Elongation tertile"),
    ]
    order = ["lower", "middle", "upper"]
    for axis, (key, title) in zip(axes, dimensions):
        means = []
        for value in order:
            metrics = [float(row["f1"]) for row in region_rows if row[key] == value]
            means.append(float(np.mean(metrics)) if metrics else 0.0)
        axis.bar(order, means, color=["#4878A8", "#F2B134", "#C65D4B"])
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.set_ylabel("Mean region F1")
    fig.tight_layout()
    fig.savefig(figures_dir / "region_f1_by_shape_tertile.png", dpi=160)
    plt.close(fig)


def _save_boundary_plot(image_rows: list[dict], figures_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(
        [float(row["image_miou"]) for row in image_rows],
        [float(row["boundary_f1_diag2pct"]) for row in image_rows],
        s=14,
        alpha=0.55,
        color="#2D6A6A",
    )
    axis.set_xlabel("Image mIoU")
    axis.set_ylabel("Boundary F1 (2% diagonal)")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(figures_dir / "boundary_f1_vs_image_miou.png", dpi=160)
    plt.close(fig)


def _save_background_plot(image_rows: list[dict], figures_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(
        [float(row["background_laplacian_variance"]) for row in image_rows],
        [float(row["background_fp_rate"]) for row in image_rows],
        s=14,
        alpha=0.55,
        color="#A05244",
    )
    axis.set_xlabel("Background Laplacian variance")
    axis.set_ylabel("Background false-positive rate")
    axis.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(figures_dir / "background_fp_vs_laplacian_variance.png", dpi=160)
    plt.close(fig)


def _write_report(root: Path, summary: dict) -> None:
    report = f"""# Aff-Grasp AED Weakness Analysis

Run ID: `{summary["run_id"]}`

## Dataset

- AED images: {summary["num_images"]}
- GT components: {summary["num_regions"]}
- Inference threshold: {summary["threshold"]}
- DepthFeature Injector: enabled

## Experiment 2: Region Shape

Region-level IoU, recall, and F1 are stored in `metrics/region_metrics.csv`.
GT components use 8-connectivity and same-class predictions are matched one-to-one
with the Hungarian algorithm. Unmatched GT components receive zero scores.

## Experiment 3: Boundary Quality

Boundary metrics are stored in `metrics/image_metrics.csv` and
`metrics/class_metrics.csv`. Results include a 2% image-diagonal tolerance and a
strict 3-pixel tolerance.

## Experiment 4: Background False Positives

Per-image background false-positive rates and automatic complexity measures are
stored in `metrics/image_metrics.csv`. Sobel edge density is the fraction of GT
background pixels whose gradient magnitude is at least 64.

## Next Steps

1. Review every blinded panel in `review/`.
2. Export `annotations.csv`.
3. Run `tools/merge_aed_review_annotations.py`.
4. Download the full training data and run `tools/analyze_train_distribution.py`.
"""
    (root / "report.md").write_text(report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", required=True)
    args = parser.parse_args()

    root = Path(args.analysis_root).resolve()
    manifest = read_csv(root / "manifest.csv")
    config = json.loads((root / "run_config.json").read_text())
    dataset_root = Path(config["dataset_root"])
    figures_dir = root / "figures"
    metrics_dir = root / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    confusion = np.zeros((len(CLASS_NAMES) + 1, len(CLASS_NAMES) + 1), dtype=np.int64)
    image_rows = []
    region_rows = []
    class_boundary = defaultdict(lambda: defaultdict(list))

    for row in manifest:
        pred = _load_mask(root, row["pred_mask"])
        gt = _load_mask(root, row["gt_mask"])
        confusion += class_confusion(pred, gt)
        rgb = np.asarray(Image.open(dataset_root / "JPEGImages" / row["image"]).convert("RGB").resize(pred.shape[::-1]))
        laplacian_variance, sobel_edge_density = _background_complexity(rgb, gt)

        gt_background = gt == 0
        pred_foreground = pred > 0
        false_positive = np.logical_and(gt_background, pred_foreground).sum()
        widths = {"diag2pct": int(math.ceil(math.hypot(*pred.shape) * 0.02)), "3px": 3}
        boundary_by_width = defaultdict(list)
        for class_id, class_name in enumerate(CLASS_NAMES, start=1):
            gt_class = gt == class_id
            pred_class = pred == class_id
            if not np.logical_or(gt_class, pred_class).any():
                continue
            if gt_class.any():
                for suffix, width in widths.items():
                    values = boundary_metrics(pred_class, gt_class, width)
                    for key, value in values.items():
                        boundary_by_width[f"{key}_{suffix}"].append(value)
                        class_boundary[class_name][f"{key}_{suffix}"].append(value)
                for region_index, region in enumerate(matched_gt_regions(pred_class, gt_class)):
                    region_rows.append(
                        {
                            "image": row["image"],
                            "object": row["object"],
                            "class_id": class_id,
                            "class_name": class_name,
                            "region_index": region_index,
                            "image_pixels": int(gt.size),
                            "area_fraction": float(region["area"] / gt.size),
                            **region,
                        }
                    )
        image_output = {
            "index": int(row["index"]),
            "image": row["image"],
            "object": row["object"],
            "image_miou": float(row["image_miou"]),
            "foreground_iou": float(row["foreground_iou"]),
            "background_fp_rate": float(false_positive / max(gt_background.sum(), 1)),
            "fp_share": float(false_positive / max(pred_foreground.sum(), 1)),
            "background_laplacian_variance": laplacian_variance,
            "background_sobel_edge_density": sobel_edge_density,
            "panel": row["panel"],
        }
        for key, values in boundary_by_width.items():
            image_output[key] = _mean_present(values)
        image_rows.append(image_output)

    assign_rank_tertiles(image_rows, "image_miou", "miou_group")
    for source, target in [
        ("area", "area_bin"),
        ("bbox_aspect_ratio", "bbox_aspect_ratio_bin"),
        ("elongation", "elongation_bin"),
    ]:
        assign_tertiles(region_rows, source, target)

    class_rows = aggregate_class_metrics(confusion)
    boundary_keys = [
        f"{metric}_{suffix}"
        for suffix in ["diag2pct", "3px"]
        for metric in ["boundary_iou", "boundary_precision", "boundary_recall", "boundary_f1", "boundary_neighborhood_iou"]
    ]
    for row in class_rows:
        for key in boundary_keys:
            row[key] = _mean_present(class_boundary[row["class_name"]][key])
    for row in image_rows:
        for key in boundary_keys:
            row.setdefault(key, 0.0)

    correlations = {
        "region_shape": {},
        "background_false_positive": {
            "laplacian_variance": spearman(
                [row["background_laplacian_variance"] for row in image_rows],
                [row["background_fp_rate"] for row in image_rows],
            ),
            "sobel_edge_density": spearman(
                [row["background_sobel_edge_density"] for row in image_rows],
                [row["background_fp_rate"] for row in image_rows],
            ),
        },
    }
    for feature in ["area", "area_fraction", "bbox_aspect_ratio", "elongation"]:
        values = [row[feature] for row in region_rows]
        f1_values = [row["f1"] for row in region_rows]
        correlations["region_shape"][feature] = {
            **spearman(values, f1_values),
            "bootstrap_95ci": bootstrap_spearman(values, f1_values),
        }

    write_csv(metrics_dir / "image_metrics.csv", image_rows)
    write_csv(metrics_dir / "region_metrics.csv", region_rows)
    write_csv(metrics_dir / "class_metrics.csv", class_rows)
    region_bin_summary = []
    for key in ["area_bin", "bbox_aspect_ratio_bin", "elongation_bin"]:
        for row in group_means(region_rows, key, ["iou", "recall", "f1"]):
            region_bin_summary.append({"feature": key.removesuffix("_bin"), "bin": row.pop(key), **row})
    write_csv(metrics_dir / "region_bin_summary.csv", region_bin_summary)
    write_json(metrics_dir / "correlations.json", correlations)
    summary = {
        "run_id": root.name,
        "num_images": len(image_rows),
        "num_regions": len(region_rows),
        "threshold": config["threshold"],
        "class_names": CLASS_NAMES,
    }
    write_json(metrics_dir / "analysis_summary.json", summary)
    _save_region_bin_plot(region_rows, figures_dir)
    _save_boundary_plot(image_rows, figures_dir)
    _save_background_plot(image_rows, figures_dir)
    _write_report(root, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
