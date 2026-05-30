#!/usr/bin/env python3
"""Shared helpers for Aff-Grasp AED weakness analysis."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage, stats
from scipy.optimize import linear_sum_assignment


CLASS_NAMES = ["grasp", "cut", "scoop", "pound", "support", "screw", "contain", "stick"]
FAILURE_TAGS = [
    "miss_small_region",
    "miss_thin_region",
    "boundary_error",
    "background_false_positive",
    "under_segmentation",
    "over_segmentation",
    "class_confusion",
    "no_obvious_failure",
    "other",
]
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)


def relative_to(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def resolve_from(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    return {
        "iou": safe_div(tp, tp + fp + fn),
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
    }


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.uint8)
    if not mask.any():
        return mask.astype(bool)
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    return (mask - eroded).astype(bool)


def dilate(mask: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        return mask.astype(bool)
    kernel = np.ones((width * 2 + 1, width * 2 + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def boundary_metrics(pred: np.ndarray, gt: np.ndarray, width: int) -> dict[str, float]:
    pred_boundary = mask_boundary(pred)
    gt_boundary = mask_boundary(gt)
    pred_band = dilate(pred_boundary, width)
    gt_band = dilate(gt_boundary, width)
    matched_pred = np.logical_and(pred_boundary, gt_band).sum()
    matched_gt = np.logical_and(gt_boundary, pred_band).sum()
    precision = safe_div(matched_pred, pred_boundary.sum())
    recall = safe_div(matched_gt, gt_boundary.sum())
    neighborhood = dilate(gt_boundary, width)
    neighborhood_metrics = binary_metrics(np.logical_and(pred, neighborhood), np.logical_and(gt, neighborhood))
    return {
        "boundary_iou": binary_metrics(pred_band, gt_band)["iou"],
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": safe_div(2 * precision * recall, precision + recall),
        "boundary_neighborhood_iou": neighborhood_metrics["iou"],
    }


def connected_components(mask: np.ndarray) -> list[dict]:
    labels, count = ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))
    regions = []
    for label_id in range(1, count + 1):
        ys, xs = np.nonzero(labels == label_id)
        if len(xs) == 0:
            continue
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        covariance = np.cov(np.stack([xs, ys])) if len(xs) > 1 else np.zeros((2, 2))
        eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))[::-1]
        major = float(4.0 * math.sqrt(eigenvalues[0])) if eigenvalues.size else 0.0
        minor = float(4.0 * math.sqrt(eigenvalues[1])) if eigenvalues.size > 1 else 0.0
        regions.append(
            {
                "label_id": label_id,
                "mask": labels == label_id,
                "area": int(len(xs)),
                "width": width,
                "height": height,
                "bbox_aspect_ratio": float(max(width / height, height / width)),
                "major_axis_length": major,
                "minor_axis_length": minor,
                "elongation": float(major / max(minor, 1e-6)),
            }
        )
    return regions


def matched_gt_regions(pred: np.ndarray, gt: np.ndarray) -> list[dict]:
    gt_regions = connected_components(gt)
    pred_regions = connected_components(pred)
    if not gt_regions:
        return []
    ious = np.zeros((len(gt_regions), len(pred_regions)), dtype=np.float64)
    for gt_idx, gt_region in enumerate(gt_regions):
        for pred_idx, pred_region in enumerate(pred_regions):
            ious[gt_idx, pred_idx] = binary_metrics(pred_region["mask"], gt_region["mask"])["iou"]
    matches = {}
    if pred_regions:
        row_indices, col_indices = linear_sum_assignment(-ious)
        matches = {int(row): int(col) for row, col in zip(row_indices, col_indices)}
    rows = []
    for gt_idx, gt_region in enumerate(gt_regions):
        pred_region = pred_regions[matches[gt_idx]] if gt_idx in matches else None
        metrics = binary_metrics(pred_region["mask"], gt_region["mask"]) if pred_region else {
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
        rows.append({**{k: v for k, v in gt_region.items() if k != "mask"}, **metrics})
    return rows


def assign_tertiles(rows: list[dict], source: str, target: str) -> None:
    if not rows:
        return
    values = np.array([float(row[source]) for row in rows], dtype=np.float64)
    lower, upper = np.quantile(values, [1 / 3, 2 / 3])
    for row, value in zip(rows, values):
        row[target] = "lower" if value <= lower else "middle" if value <= upper else "upper"


def assign_rank_tertiles(rows: list[dict], source: str, target: str) -> None:
    """Assign near-equal tertiles, using row order as a deterministic tie breaker."""
    ordered = sorted(enumerate(rows), key=lambda item: (float(item[1][source]), item[0]))
    for rank, (_, row) in enumerate(ordered):
        row[target] = ["lower", "middle", "upper"][min(2, rank * 3 // len(rows))]


def spearman(values_x, values_y) -> dict[str, float | None]:
    x = np.asarray(values_x, dtype=np.float64)
    y = np.asarray(values_y, dtype=np.float64)
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return {"rho": None, "pvalue": None}
    result = stats.spearmanr(x, y)
    return {"rho": float(result.statistic), "pvalue": float(result.pvalue)}


def bootstrap_spearman(values_x, values_y, iterations: int = 1000, seed: int = 1311) -> dict[str, float | None]:
    x = np.asarray(values_x, dtype=np.float64)
    y = np.asarray(values_y, dtype=np.float64)
    if len(x) < 2:
        return {"low": None, "high": None}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        indices = rng.integers(0, len(x), len(x))
        value = spearman(x[indices], y[indices])["rho"]
        if value is not None and np.isfinite(value):
            samples.append(value)
    if not samples:
        return {"low": None, "high": None}
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"low": float(low), "high": float(high)}


def class_confusion(pred: np.ndarray, gt: np.ndarray, class_count: int = 9) -> np.ndarray:
    return np.bincount(class_count * gt.astype(int).ravel() + pred.astype(int).ravel(), minlength=class_count**2).reshape(
        class_count, class_count
    )


def aggregate_class_metrics(confusion: np.ndarray) -> list[dict]:
    rows = []
    for class_id in range(1, len(CLASS_NAMES) + 1):
        tp = float(confusion[class_id, class_id])
        fp = float(confusion[:, class_id].sum() - tp)
        fn = float(confusion[class_id, :].sum() - tp)
        rows.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id - 1],
                "iou": safe_div(tp, tp + fp + fn),
                "precision": safe_div(tp, tp + fp),
                "recall": safe_div(tp, tp + fn),
                "f1": safe_div(2 * tp, 2 * tp + fp + fn),
            }
        )
    return rows


def group_means(rows: list[dict], group_key: str, value_keys: list[str]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row[group_key]].append(row)
    out = []
    for group, items in sorted(groups.items()):
        output = {group_key: group, "count": len(items)}
        for key in value_keys:
            output[f"mean_{key}"] = float(np.mean([float(item[key]) for item in items]))
        out.append(output)
    return out
