#!/usr/bin/env python3
"""Download and arrange assets needed to run Aff-Grasp evaluation.

The upstream code expects:
- affordance-learning/ag_dataset/Affordance_Evaluation_Dataset
- affordance-learning/ag_dataset/ego_train
- affordance-learning/depth
- affordance-learning/dinov2_vitb14_pretrain.pth

This script downloads the public Hugging Face dataset/model repos, then creates
links or copies so the original train.py/test.py can be used with minimal flags.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download


DATA_REPO = "Gen1113/Data_for_Aff-Grasp"
MODEL_REPO = "Gen1113/Model_for_Aff-Grasp"
DINO_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/"
    "dinov2_vitb14_pretrain.pth"
)


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return
    try:
        rel_src = os.path.relpath(src, dst.parent)
        dst.symlink_to(rel_src, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _find_dir(root: Path, name: str) -> Path | None:
    for path in root.rglob(name):
        if path.is_dir():
            return path
    return None


def _find_checkpoint(root: Path) -> Path | None:
    candidates = []
    for suffix in ("*.pth", "*.pt", "*.ckpt"):
        candidates.extend(root.rglob(suffix))
    non_dino = [p for p in candidates if "dinov2" not in p.name.lower()]
    return sorted(non_dino or candidates, key=lambda p: p.stat().st_size, reverse=True)[0] if candidates else None


def _download_dino(dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading DINOv2 weight to {dst}")
    urllib.request.urlretrieve(DINO_URL, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aff-root", default="affordance-learning")
    parser.add_argument("--cache-dir", default="third_party/aff-grasp-assets")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating links.")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Download only Affordance_Evaluation_Dataset instead of the full dataset repo.",
    )
    parser.add_argument("--skip-dino", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    aff_root = (repo_root / args.aff_root).resolve()
    cache_dir = (repo_root / args.cache_dir).resolve()
    data_root = aff_root / "ag_dataset"

    aff_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset repo: {DATA_REPO}")
    allow_patterns = None
    if args.eval_only:
        allow_patterns = [
            "Data_for-Aff-Grasp/Affordance_Evaluation_Dataset/**",
            "Data_for_Aff-Grasp/Affordance_Evaluation_Dataset/**",
            "Data_for_Aff_Grasp/Affordance_Evaluation_Dataset/**",
            "Affordance_Evaluation_Dataset/**",
        ]
    data_snapshot = Path(
        snapshot_download(
            DATA_REPO,
            repo_type="dataset",
            local_dir=cache_dir / "Data_for_Aff-Grasp",
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
        )
    )

    print(f"Downloading model repo: {MODEL_REPO}")
    model_snapshot = Path(
        snapshot_download(
            MODEL_REPO,
            repo_type="model",
            local_dir=cache_dir / "Model_for_Aff-Grasp",
            local_dir_use_symlinks=False,
        )
    )

    required_dirs = [("Affordance_Evaluation_Dataset", data_root / "Affordance_Evaluation_Dataset")]
    if not args.eval_only:
        required_dirs.extend(
            [
                ("ego_train", data_root / "ego_train"),
                ("depth", aff_root / "depth"),
            ]
        )

    for dirname, dst in required_dirs:
        src = _find_dir(data_snapshot, dirname)
        if src is None:
            print(f"Warning: could not find dataset directory named {dirname}", file=sys.stderr)
            continue
        _link_or_copy(src, dst, copy=args.copy)
        print(f"Ready: {dst}")

    if not args.skip_dino:
        _download_dino(aff_root / "dinov2_vitb14_pretrain.pth")

    ckpt = _find_checkpoint(model_snapshot)
    if ckpt is not None:
        dst = aff_root / "pretrained_aff_grasp.pth"
        _link_or_copy(ckpt, dst, copy=args.copy)
        print(f"Ready: {dst}")
    else:
        print("Warning: no .pth/.pt/.ckpt model checkpoint found in model repo.", file=sys.stderr)

    print("\nNext:")
    print(f"  cd {aff_root}")
    print("  python test.py --data_root ag_dataset --model_file pretrained_aff_grasp.pth --gpu 0 --viz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
