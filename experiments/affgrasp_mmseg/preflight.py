#!/usr/bin/env python3
"""Preflight checks before running Aff-Grasp follow-up experiments on a GPU server."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from experiments.affgrasp_mmseg.common import discover_aed_samples, discover_train_samples, ensure_splits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", default="affordance-learning/ag_dataset/ego_train")
    parser.add_argument("--aed-root", default="affordance-learning/ag_dataset/Affordance_Evaluation_Dataset")
    parser.add_argument("--split-dir", default="experiments/splits")
    parser.add_argument("--check-timm-models", action="store_true")
    args = parser.parse_args()

    train_root = Path(args.train_root).resolve()
    aed_root = Path(args.aed_root).resolve()
    split_dir = Path(args.split_dir).resolve()
    train_samples = discover_train_samples(train_root)
    aed_samples = discover_aed_samples(aed_root)
    ensure_splits(split_dir, train_root, aed_root)
    report = {
        "train_root": str(train_root),
        "aed_root": str(aed_root),
        "split_dir": str(split_dir),
        "train_samples": len(train_samples),
        "aed_samples": len(aed_samples),
    }
    try:
        report["hostname"] = subprocess.check_output(["hostname"], text=True).strip()
        report["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        report["gpu_warning"] = str(exc)
    if args.check_timm_models:
        import timm

        names = set(timm.list_models())
        report["timm_has_mit_b0"] = "mit_b0" in names
        report["timm_has_internimage_t_1k_224"] = "internimage_t_1k_224" in names
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

