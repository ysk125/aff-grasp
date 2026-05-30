#!/usr/bin/env python3
"""Merge blinded AED review tags with metrics and summarize Experiment 1."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis_common import FAILURE_TAGS, read_csv, write_csv, write_json


GROUPS = ["lower", "middle", "upper"]


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    rate = successes / total
    denominator = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    radius = z * math.sqrt((rate * (1 - rate) + z**2 / (4 * total)) / total) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _risk_ratio(lower_success: int, lower_total: int, upper_success: int, upper_total: int) -> float | None:
    upper_rate = upper_success / upper_total if upper_total else 0.0
    return (lower_success / lower_total) / upper_rate if lower_total and upper_rate else None


def _cliffs_delta(values_a: list[float], values_b: list[float]) -> float | None:
    if not values_a or not values_b:
        return None
    greater = sum(a > b for a in values_a for b in values_b)
    lower = sum(a < b for a in values_a for b in values_b)
    return float((greater - lower) / (len(values_a) * len(values_b)))


def _tag_rows(merged: list[dict]) -> list[dict]:
    counts = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    for row in merged:
        group = row["miou_group"]
        totals[group] += 1
        for tag in FAILURE_TAGS:
            counts[tag][group] += int(row[tag])
    output = []
    for tag in FAILURE_TAGS:
        lower_count = counts[tag]["lower"]
        upper_count = counts[tag]["upper"]
        table = np.array(
            [
                [lower_count, totals["lower"] - lower_count],
                [upper_count, totals["upper"] - upper_count],
            ]
        )
        _, pvalue = stats.fisher_exact(table)
        risk_ratio = _risk_ratio(lower_count, totals["lower"], upper_count, totals["upper"])
        for group in GROUPS:
            count = counts[tag][group]
            total = totals[group]
            low, high = _wilson(count, total)
            output.append(
                {
                    "tag": tag,
                    "miou_group": group,
                    "count": count,
                    "total": total,
                    "rate": count / total if total else 0.0,
                    "wilson_95ci_low": low,
                    "wilson_95ci_high": high,
                    "lower_vs_upper_risk_ratio": risk_ratio,
                    "lower_vs_upper_fisher_pvalue": float(pvalue),
                }
            )
    return output


def _plot_tags(rows: list[dict], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(FAILURE_TAGS))
    width = 0.25
    colors = {"lower": "#C65D4B", "middle": "#F2B134", "upper": "#4878A8"}
    for offset, group in enumerate(GROUPS):
        values = [next(float(row["rate"]) for row in rows if row["tag"] == tag and row["miou_group"] == group) for tag in FAILURE_TAGS]
        axis.bar(positions + (offset - 1) * width, values, width, label=group, color=colors[group])
    axis.set_xticks(positions, FAILURE_TAGS, rotation=35, ha="right")
    axis.set_ylabel("Tagged image rate")
    axis.set_ylim(0, 1)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_background(complex_values: list[float], simple_values: list[float], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.boxplot([simple_values, complex_values], labels=["simple", "complex"])
    axis.set_ylabel("Background false-positive rate")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    root = Path(args.analysis_root).resolve()
    annotations_path = Path(args.annotations).resolve() if args.annotations else root / "review" / "annotations.csv"
    image_rows = read_csv(root / "metrics" / "image_metrics.csv")
    annotations = {row["review_id"]: row for row in read_csv(annotations_path)}
    missing = []
    merged = []
    for image in image_rows:
        review_id = f"{int(image['index']):04d}"
        annotation = annotations.get(review_id)
        if annotation is None or not _as_bool(annotation.get("reviewed", "")):
            missing.append(review_id)
            continue
        output = {**image, "review_id": review_id}
        for key in ["reviewed", "complex_background", "uncertain", *FAILURE_TAGS]:
            output[key] = _as_bool(annotation.get(key, ""))
        output["note"] = annotation.get("note", "")
        merged.append(output)
    if missing and not args.allow_incomplete:
        raise RuntimeError(f"{len(missing)} images are not reviewed. Finish the blinded review or pass --allow-incomplete.")
    if not merged:
        raise RuntimeError("No reviewed annotations found.")

    tag_summary = _tag_rows(merged)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    metrics = root / "metrics"
    write_csv(metrics / "reviewed_image_metrics.csv", merged)
    write_csv(metrics / "failure_tag_summary.csv", tag_summary)
    _plot_tags(tag_summary, figures / "failure_tag_rate_by_miou_group.png")

    complex_values = [float(row["background_fp_rate"]) for row in merged if row["complex_background"]]
    simple_values = [float(row["background_fp_rate"]) for row in merged if not row["complex_background"]]
    mann_whitney = stats.mannwhitneyu(complex_values, simple_values, alternative="two-sided") if complex_values and simple_values else None
    background_summary = {
        "reviewed_images": len(merged),
        "missing_images": len(missing),
        "complex_background_images": len(complex_values),
        "simple_background_images": len(simple_values),
        "complex_background_median_fp_rate": float(np.median(complex_values)) if complex_values else None,
        "simple_background_median_fp_rate": float(np.median(simple_values)) if simple_values else None,
        "mann_whitney_u": float(mann_whitney.statistic) if mann_whitney else None,
        "mann_whitney_pvalue": float(mann_whitney.pvalue) if mann_whitney else None,
        "cliffs_delta_complex_vs_simple": _cliffs_delta(complex_values, simple_values),
    }
    write_json(metrics / "manual_background_summary.json", background_summary)
    if complex_values and simple_values:
        _plot_background(complex_values, simple_values, figures / "background_fp_by_manual_complexity.png")

    with (root / "report.md").open("a") as stream:
        stream.write(
            f"""

## Experiment 1: Blinded Failure Tags

- Reviewed images: {len(merged)}
- Unreviewed images: {len(missing)}
- Summary: `metrics/failure_tag_summary.csv`
- Figure: `figures/failure_tag_rate_by_miou_group.png`

## Experiment 4: Manual Background Complexity

- Complex-background images: {len(complex_values)}
- Simple-background images: {len(simple_values)}
- Summary: `metrics/manual_background_summary.json`
"""
        )
    print(json.dumps(background_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
