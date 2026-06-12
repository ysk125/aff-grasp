#!/usr/bin/env python3
"""Generate background-aware AED diagnostics from trained experiment checkpoints."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch

from experiments.affgrasp_mmseg.diagnostics import DIAGNOSTIC_FIELDS, write_rows
from experiments.affgrasp_mmseg.eval_affgrasp_mmseg import evaluate


CONFIGS = {
    "segformer_a": "experiments/affgrasp_mmseg/configs/segformer_affgrasp/segformer_a.py",
    "segformer_b": "experiments/affgrasp_mmseg/configs/segformer_affgrasp/segformer_b.py",
    "segformer_c": "experiments/affgrasp_mmseg/configs/segformer_affgrasp/segformer_c.py",
    "segformer_d": "experiments/affgrasp_mmseg/configs/segformer_affgrasp/segformer_d.py",
    "internimage_a": "experiments/affgrasp_mmseg/configs/internimage_affgrasp/internimage_a.py",
    "internimage_c": "experiments/affgrasp_mmseg/configs/internimage_affgrasp/internimage_c.py",
    "internimage_d": "experiments/affgrasp_mmseg/configs/internimage_affgrasp/internimage_d.py",
}


def read_summary(path: Path) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one diagnostics row in {path}, found {len(rows)}")
    return rows[0]


def collect_summaries(output_root: Path, experiments: list[str]) -> list[dict[str, object]]:
    rows = []
    missing = []
    for experiment in experiments:
        path = output_root / experiment / "test" / "diagnostics_summary.csv"
        if path.exists():
            rows.append(read_summary(path))
        else:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing diagnostics summaries:\n" + "\n".join(missing))
    write_rows(output_root / "all_experiments_diagnostics_summary.csv", rows, DIAGNOSTIC_FIELDS)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--aed-root", default="affordance-learning/ag_dataset/Affordance_Evaluation_Dataset")
    parser.add_argument("--split-dir", default="experiments/splits")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--experiments", nargs="+", choices=sorted(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    if args.summarize_only:
        collect_summaries(output_root, args.experiments)
        print(f"Saved: {output_root / 'all_experiments_diagnostics_summary.csv'}")
        return 0

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for full AED diagnostic reruns")

    for experiment in args.experiments:
        checkpoint = output_root / experiment / "checkpoints" / "best.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        print(f"\n== diagnostics: {experiment} ==")
        evaluate(
            Path(CONFIGS[experiment]),
            checkpoint,
            output_root / experiment / "test",
            Path(args.aed_root).resolve(),
            Path(args.split_dir).resolve(),
            device,
            max_samples=args.max_samples,
            diagnostics_only=True,
        )

    collect_summaries(output_root, args.experiments)
    print(f"Saved: {output_root / 'all_experiments_diagnostics_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
