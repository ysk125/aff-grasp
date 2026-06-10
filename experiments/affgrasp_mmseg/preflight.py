#!/usr/bin/env python3
"""Preflight checks before running Aff-Grasp follow-up experiments on a GPU server."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from experiments.affgrasp_mmseg.common import build_model, discover_aed_samples, discover_train_samples, ensure_splits, load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", default="affordance-learning/ag_dataset/ego_train")
    parser.add_argument("--aed-root", default="affordance-learning/ag_dataset/Affordance_Evaluation_Dataset")
    parser.add_argument("--split-dir", default="experiments/splits")
    parser.add_argument("--check-timm-models", action="store_true")
    parser.add_argument("--check-config", default="experiments/affgrasp_mmseg/configs/segformer_affgrasp/segformer_a.py")
    parser.add_argument("--check-internimage", action="store_true")
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
        try:
            import timm

            names = set(timm.list_models())
            report["timm_has_mit_b0"] = "mit_b0" in names
            report["timm_has_internimage_t_1k_224"] = "internimage_t_1k_224" in names
        except ImportError as exc:
            report["timm_warning"] = str(exc)
        try:
            import transformers

            report["transformers_available"] = True
            report["transformers_version"] = transformers.__version__
        except ImportError as exc:
            report["transformers_available"] = False
            report["transformers_warning"] = str(exc)
        try:
            cfg = load_config(args.check_config)
            model = build_model(cfg)
            report["check_config"] = str(Path(args.check_config).resolve())
            report["check_model_name"] = cfg.get("model_name")
            report["check_backbone"] = cfg.get("backbone")
            report["check_model_class"] = type(model).__name__
        except Exception as exc:
            report["check_model_error"] = str(exc)
    if args.check_internimage:
        intern_config = Path("experiments/affgrasp_mmseg/configs/internimage_affgrasp/internimage_a.py")
        try:
            import mmpretrain

            report["mmpretrain_available"] = True
            report["mmpretrain_version"] = getattr(mmpretrain, "__version__", "unknown")
        except ImportError as exc:
            report["mmpretrain_available"] = False
            report["mmpretrain_warning"] = str(exc)
        try:
            cfg = load_config(intern_config)
            model = build_model(cfg)
            report["internimage_check_config"] = str(intern_config.resolve())
            report["internimage_backend"] = cfg.get("backend")
            report["internimage_model_class"] = type(model).__name__
        except Exception as exc:
            report["internimage_check_model_error"] = str(exc)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
