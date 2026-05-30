#!/usr/bin/env python3
"""Analyze Aff-Grasp training-label distribution and compare with AED metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis_common import AFFORDANCE_BY_OBJECT, CLASS_NAMES, connected_components, read_csv, spearman, write_csv, write_json


def _object_name(path: Path) -> str | None:
    noun = path.name.lower().split("_", 1)[0].split("-", 1)[0]
    return noun if noun in AFFORDANCE_BY_OBJECT else None


def _task_mask(raw: np.ndarray, class_name: str) -> np.ndarray:
    class_id = CLASS_NAMES.index(class_name) + 1
    output = np.zeros(raw.shape, dtype=np.uint8)
    output[raw == 128] = 1
    output[raw == 255] = class_id
    return output


def _plot_comparison(rows: list[dict], figures_dir: Path, x_key: str, x_label: str, filename: str, log1p: bool = False) -> None:
    x_values = np.array([float(row[x_key]) for row in rows])
    if log1p:
        x_values = np.log1p(x_values)
    y_values = np.array([float(row["aed_f1"]) for row in rows])
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(x_values, y_values, s=44, color="#2D6A6A")
    for x, y, row in zip(x_values, y_values, rows):
        axis.annotate(row["class_name"], (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.set_xlabel(x_label)
    axis.set_ylabel("AED class F1")
    axis.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(figures_dir / filename, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--train-root", default="affordance-learning/ag_dataset/ego_train")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    root = Path(args.analysis_root).resolve()
    train_root = (repo_root / args.train_root).resolve()
    if not train_root.exists():
        raise FileNotFoundError(f"Missing training data: {train_root}. Run tools/download_aff_grasp_assets.py without --eval-only.")
    label_paths = sorted(path for path in train_root.rglob("*-label.png") if path.is_file() and not path.name.endswith(".metadata"))
    if not label_paths:
        raise RuntimeError(f"No *-label.png files found under {train_root}")

    stats_by_class = {
        name: {"image_count": 0, "pixel_count": 0, "instance_count": 0, "areas": []} for name in CLASS_NAMES
    }
    total_pixels = 0
    unrecognized = []
    for label_path in label_paths:
        object_name = _object_name(label_path)
        if object_name is None:
            unrecognized.append(label_path.as_posix())
            continue
        raw = np.asarray(Image.open(label_path))
        if raw.ndim == 3:
            raw = raw[..., 0]
        labels = _task_mask(raw, AFFORDANCE_BY_OBJECT[object_name])
        total_pixels += int(labels.size)
        for class_id, class_name in enumerate(CLASS_NAMES, start=1):
            class_mask = labels == class_id
            if not class_mask.any():
                continue
            current = stats_by_class[class_name]
            current["image_count"] += 1
            current["pixel_count"] += int(class_mask.sum())
            regions = connected_components(class_mask)
            current["instance_count"] += len(regions)
            current["areas"].extend(region["area"] for region in regions)

    aed_by_class = {row["class_name"]: row for row in read_csv(root / "metrics" / "class_metrics.csv")}
    output = []
    for class_name in CLASS_NAMES:
        values = stats_by_class[class_name]
        areas = np.asarray(values.pop("areas"), dtype=np.float64)
        aed = aed_by_class[class_name]
        output.append(
            {
                "class_name": class_name,
                **values,
                "pixel_fraction": float(values["pixel_count"] / max(total_pixels, 1)),
                "mean_region_area": float(np.mean(areas)) if len(areas) else 0.0,
                "median_region_area": float(np.median(areas)) if len(areas) else 0.0,
                "q1_region_area": float(np.quantile(areas, 0.25)) if len(areas) else 0.0,
                "q3_region_area": float(np.quantile(areas, 0.75)) if len(areas) else 0.0,
                "aed_iou": float(aed["iou"]),
                "aed_recall": float(aed["recall"]),
                "aed_f1": float(aed["f1"]),
            }
        )

    comparisons = {}
    for key in ["pixel_count", "instance_count", "median_region_area"]:
        x_values = [math.log1p(float(row[key])) if key != "median_region_area" else float(row[key]) for row in output]
        comparisons[key] = spearman(x_values, [float(row["aed_f1"]) for row in output])
    summary = {
        "training_label_files": len(label_paths),
        "analyzed_label_files": len(label_paths) - len(unrecognized),
        "unrecognized_label_files": len(unrecognized),
        "note": "Class-level correlations are exploratory because Aff-Grasp has only eight affordance classes.",
        "spearman_vs_aed_f1": comparisons,
    }

    metrics_dir = root / "metrics"
    figures_dir = root / "figures"
    figures_dir.mkdir(exist_ok=True)
    write_csv(metrics_dir / "train_distribution.csv", output)
    write_json(metrics_dir / "train_distribution_summary.json", summary)
    if unrecognized:
        (metrics_dir / "unrecognized_training_labels.txt").write_text("\n".join(unrecognized) + "\n")
    _plot_comparison(output, figures_dir, "pixel_count", "log1p(train pixel count)", "train_pixels_vs_aed_f1.png", log1p=True)
    _plot_comparison(output, figures_dir, "instance_count", "log1p(train instance count)", "train_instances_vs_aed_f1.png", log1p=True)
    _plot_comparison(output, figures_dir, "median_region_area", "Median train region area", "train_region_area_vs_aed_f1.png")
    with (root / "report.md").open("a") as stream:
        stream.write(
            f"""

## Experiment 5: Training Distribution

- Training label files: {summary["training_label_files"]}
- Analyzed label files: {summary["analyzed_label_files"]}
- Unrecognized label files: {summary["unrecognized_label_files"]}
- Summary: `metrics/train_distribution_summary.json`
- Per-class values: `metrics/train_distribution.csv`

The eight-class correlations are exploratory and do not establish causality.
"""
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
