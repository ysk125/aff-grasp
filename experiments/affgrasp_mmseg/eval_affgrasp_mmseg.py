#!/usr/bin/env python3
"""Evaluate trained follow-up segmentation experiments on AED."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.affgrasp_mmseg.common import (
    AffGraspSegDataset,
    MetricState,
    build_model,
    discover_aed_samples,
    load_config,
    read_split,
    save_panel,
    write_csv,
)


def evaluate(
    config_path: Path,
    checkpoint: Path,
    output_dir: Path,
    aed_root: Path,
    split_dir: Path,
    device,
    cfg: dict | None = None,
    max_samples: int | None = None,
) -> dict:
    cfg = cfg or load_config(config_path)
    test_file = split_dir / "test.txt"
    rows = read_split(test_file) if test_file.exists() else discover_aed_samples(aed_root)
    if max_samples is not None:
        rows = rows[:max_samples]
    dataset = AffGraspSegDataset(rows, cfg["resize_size"], cfg["crop_size"], train=False, source="aed")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = build_model(cfg).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    metrics = MetricState.create()
    image_rows = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader)):
            image = batch["image"].to(device)
            target = batch["target"].to(device)
            logits = model(image)
            pred = logits.argmax(dim=1)
            metrics.update(pred, target)
            image_metrics = MetricState.create()
            image_metrics.update(pred, target)
            image_result = image_metrics.compute(ignore_background=True)
            error = pred.squeeze(0).cpu() != batch["target"].squeeze(0)
            panel_path = output_dir / "visualizations" / f"{idx:04d}_{batch['name'][0]}.png"
            save_panel(panel_path, batch["image"].squeeze(0), batch["target"].squeeze(0), pred.squeeze(0).cpu(), error)
            image_rows.append(
                {
                    "index": idx,
                    "name": batch["name"][0],
                    "mIoU": image_result["mIoU"],
                    "F1": image_result["F1"],
                    "Accuracy": image_result["Accuracy"],
                    "panel": str(panel_path),
                }
            )
    result = metrics.compute(ignore_background=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "image_manifest.csv", image_rows)
    write_csv(output_dir / "metrics.csv", [{"model_name": cfg["model_name"], "experiment_type": cfg["experiment_type"], **result}])
    worst_dir = output_dir / "worst_cases"
    worst_rows = sorted(image_rows, key=lambda row: row["mIoU"])[: min(24, len(image_rows))]
    for rank, row in enumerate(worst_rows, start=1):
        source = Path(row["panel"])
        destination = worst_dir / f"{rank:02d}_{source.name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        row["worst_case_panel"] = str(destination)
    write_csv(output_dir / "worst_cases.csv", worst_rows)
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--aed-root", default="affordance-learning/ag_dataset/Affordance_Evaluation_Dataset")
    parser.add_argument("--split-dir", default="experiments/splits")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate(
        Path(args.config),
        Path(args.checkpoint),
        Path(args.output_dir),
        Path(args.aed_root).resolve(),
        Path(args.split_dir).resolve(),
        device,
        max_samples=args.max_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
