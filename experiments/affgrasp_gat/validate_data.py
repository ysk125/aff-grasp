#!/usr/bin/env python3
"""Validate Aff-Grasp GAT training data before retraining."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from experiments.affgrasp_gat.common import (
    CLASS_NAMES,
    dataset_statistics,
    discover_train_samples,
    save_validation_panel,
    write_csv,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="affordance-learning/ag_dataset")
    parser.add_argument("--depth-root", default=None)
    parser.add_argument("--output-dir", default="outputs/gat_data_validation")
    parser.add_argument("--visualization-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    data_root = Path(args.data_root).resolve()
    depth_root = Path(args.depth_root).resolve() if args.depth_root else None
    samples, missing = discover_train_samples(data_root, depth_root=depth_root, allow_missing=True)
    stats = dataset_statistics(samples) if samples else {"train_samples": 0}
    stats["data_root"] = str(data_root)
    stats["depth_root"] = str(depth_root) if depth_root else ""
    stats["missing_or_invalid_pairs"] = len(missing)
    stats["known_classes"] = CLASS_NAMES

    write_json(output_dir / "dataset_statistics.json", stats)
    write_csv(
        output_dir / "missing_files.csv",
        missing,
        fieldnames=["image", "depth", "label", "object", "depth_exists", "label_exists", "known_object"],
    )

    rng = random.Random(args.seed)
    selected = samples[:]
    rng.shuffle(selected)
    for idx, sample in enumerate(selected[: args.visualization_count]):
        save_validation_panel(output_dir / "sample_visualizations" / f"{idx:04d}_{sample.image_path.stem}.png", sample)

    print(f"train samples: {len(samples)}")
    print(f"missing/invalid pairs: {len(missing)}")
    print(f"saved: {output_dir}")
    if len(samples) == 0:
        raise RuntimeError("No valid train samples were found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

