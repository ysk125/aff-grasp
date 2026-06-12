"""Background-aware diagnostics for Aff-Grasp semantic segmentation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DIAGNOSTIC_FIELDS = [
    "model_name",
    "experiment_type",
    "miou_with_background",
    "f1_with_background",
    "accuracy_with_background",
    "miou_without_background",
    "f1_without_background",
    "foreground_only_accuracy",
    "gt_foreground_ratio",
    "predicted_foreground_ratio",
    "foreground_ratio_gap",
    "num_images",
    "num_valid_pixels",
]

KEY_CONFUSIONS = [
    ("grasp", "background"),
    ("cut", "background"),
    ("scoop", "background"),
    ("contain", "scoop"),
    ("screw", "stick"),
    ("stick", "background"),
]


def confusion_matrix(
    prediction: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_labels: Iterable[int] = (255,),
) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.int64).reshape(-1)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target shapes differ: {prediction.shape} != {target.shape}")

    valid = (target >= 0) & (target < num_classes)
    for label in ignore_labels:
        valid &= target != label
    valid &= (prediction >= 0) & (prediction < num_classes)
    encoded = num_classes * target[valid] + prediction[valid]
    return np.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)


def normalized_by_gt(confusion: np.ndarray) -> np.ndarray:
    confusion = np.asarray(confusion, dtype=np.float64)
    row_totals = confusion.sum(axis=1, keepdims=True)
    return np.divide(confusion, row_totals, out=np.zeros_like(confusion), where=row_totals > 0)


def _macro_metrics(confusion: np.ndarray, include_background: bool) -> tuple[float, float]:
    confusion = np.asarray(confusion, dtype=np.float64)
    true_positive = np.diag(confusion)
    gt_count = confusion.sum(axis=1)
    pred_count = confusion.sum(axis=0)
    union = gt_count + pred_count - true_positive
    denominator_f1 = gt_count + pred_count
    iou = np.divide(true_positive, union, out=np.zeros_like(true_positive), where=union > 0)
    f1 = np.divide(2 * true_positive, denominator_f1, out=np.zeros_like(true_positive), where=denominator_f1 > 0)

    # Match the existing evaluator: average only classes present in ground truth.
    present = gt_count > 0
    if not include_background and len(present):
        present[0] = False
    if not np.any(present):
        return 0.0, 0.0
    return float(iou[present].mean()), float(f1[present].mean())


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, float | int]:
    confusion = np.asarray(confusion, dtype=np.int64)
    total = int(confusion.sum())
    correct = int(np.diag(confusion).sum())
    foreground_gt = int(confusion[1:, :].sum())
    foreground_pred = int(confusion[:, 1:].sum())
    foreground_correct = int(np.diag(confusion)[1:].sum())
    miou_bg, f1_bg = _macro_metrics(confusion, include_background=True)
    miou_fg, f1_fg = _macro_metrics(confusion, include_background=False)
    gt_ratio = foreground_gt / total if total else 0.0
    pred_ratio = foreground_pred / total if total else 0.0
    return {
        "miou_with_background": miou_bg,
        "f1_with_background": f1_bg,
        "accuracy_with_background": correct / total if total else 0.0,
        "miou_without_background": miou_fg,
        "f1_without_background": f1_fg,
        "foreground_only_accuracy": foreground_correct / foreground_gt if foreground_gt else 0.0,
        "gt_foreground_ratio": gt_ratio,
        "predicted_foreground_ratio": pred_ratio,
        "foreground_ratio_gap": pred_ratio - gt_ratio,
        "num_valid_pixels": total,
    }


def write_rows(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_csv(path: Path, confusion: np.ndarray, class_names: Sequence[str]) -> None:
    matrix = np.asarray(confusion)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gt\\pred", *class_names])
        for name, row in zip(class_names, matrix):
            writer.writerow([name, *row.tolist()])


def write_confusion_png(
    path: Path,
    confusion: np.ndarray,
    class_names: Sequence[str],
    normalized: bool,
) -> None:
    matrix = normalized_by_gt(confusion) if normalized else np.asarray(confusion, dtype=np.float64)
    scale = matrix.max() if matrix.size and matrix.max() > 0 else 1.0
    cell, margin_left, margin_top = 74, 120, 92
    width = margin_left + cell * len(class_names) + 12
    height = margin_top + cell * len(class_names) + 12
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for column, name in enumerate(class_names):
        draw.text((margin_left + column * cell + 4, 52), name[:10], fill="black", font=font)
    for row_index, (name, row) in enumerate(zip(class_names, matrix)):
        y = margin_top + row_index * cell
        draw.text((5, y + cell // 2 - 5), name[:14], fill="black", font=font)
        for column, value in enumerate(row):
            x = margin_left + column * cell
            intensity = int(255 * float(value) / scale)
            fill = (255 - intensity, 255 - intensity // 2, 255)
            draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=fill, outline=(220, 220, 220))
            label = f"{value:.3f}" if normalized else str(int(value))
            draw.text((x + 5, y + cell // 2 - 5), label, fill="black", font=font)
    draw.text((margin_left, 8), "columns: prediction / rows: ground truth", fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_ranked_case_panels(
    test_dir: Path,
    per_image_rows: Sequence[Mapping[str, object]],
    panel_paths: Mapping[str, Path],
    limit: int = 20,
) -> None:
    rankings = {
        "foreground_underprediction_cases": sorted(per_image_rows, key=lambda row: float(row["foreground_ratio_gap"])),
        "foreground_overprediction_cases": sorted(per_image_rows, key=lambda row: float(row["foreground_ratio_gap"]), reverse=True),
    }
    for directory_name, rows in rankings.items():
        destination = test_dir / directory_name
        destination.mkdir(parents=True, exist_ok=True)
        for old_panel in destination.glob("*.png"):
            old_panel.unlink()
        for rank, row in enumerate(rows[:limit], start=1):
            image_id = str(row["image_id"])
            source = panel_paths.get(image_id)
            if source is None or not source.exists():
                continue
            target = destination / f"{rank:02d}_{Path(image_id).stem}.png"
            with Image.open(source).convert("RGB") as panel:
                strip = Image.new("RGB", (panel.width, 34), "white")
                canvas = Image.new("RGB", (panel.width, panel.height + strip.height), "white")
                canvas.paste(panel, (0, 0))
                canvas.paste(strip, (0, panel.height))
                text = (
                    f"GT fg={float(row['gt_foreground_ratio']):.4f}  "
                    f"Pred fg={float(row['predicted_foreground_ratio']):.4f}  "
                    f"Gap={float(row['foreground_ratio_gap']):+.4f}"
                )
                ImageDraw.Draw(canvas).text((8, panel.height + 10), text, fill="black", font=ImageFont.load_default())
                canvas.save(target)


def write_diagnostic_outputs(
    test_dir: Path,
    model_name: str,
    experiment_type: str,
    aggregate_confusion: np.ndarray,
    per_image_rows: Sequence[Mapping[str, object]],
    class_names: Sequence[str],
    panel_paths: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    summary = {
        "model_name": model_name,
        "experiment_type": experiment_type,
        **metrics_from_confusion(aggregate_confusion),
        "num_images": len(per_image_rows),
    }
    write_rows(test_dir / "diagnostics_summary.csv", [summary], DIAGNOSTIC_FIELDS)
    per_image_fields = ["image_id", *[field for field in DIAGNOSTIC_FIELDS if field not in {"model_name", "experiment_type", "num_images"}]]
    write_rows(test_dir / "diagnostics_per_image.csv", per_image_rows, per_image_fields)
    write_confusion_csv(test_dir / "confusion_matrix_raw.csv", aggregate_confusion, class_names)
    normalized = normalized_by_gt(aggregate_confusion)
    write_confusion_csv(test_dir / "confusion_matrix_normalized_by_gt.csv", normalized, class_names)
    write_confusion_png(test_dir / "confusion_matrix_raw.png", aggregate_confusion, class_names, normalized=False)
    write_confusion_png(test_dir / "confusion_matrix_normalized_by_gt.png", aggregate_confusion, class_names, normalized=True)
    class_ids = {name: index for index, name in enumerate(class_names)}
    key_rows = []
    for gt_name, pred_name in KEY_CONFUSIONS:
        gt_index = class_ids[gt_name]
        pred_index = class_ids[pred_name]
        key_rows.append(
            {
                "gt_class": gt_name,
                "pred_class": pred_name,
                "raw_count": int(aggregate_confusion[gt_index, pred_index]),
                "gt_normalized_rate": float(normalized[gt_index, pred_index]),
            }
        )
    write_rows(
        test_dir / "key_confusions.csv",
        key_rows,
        ["gt_class", "pred_class", "raw_count", "gt_normalized_rate"],
    )
    if panel_paths:
        save_ranked_case_panels(test_dir, per_image_rows, panel_paths)
    return summary
