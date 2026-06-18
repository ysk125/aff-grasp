#!/usr/bin/env python3
"""Retrain the official Aff-Grasp GAT model with corrected data loading."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.affgrasp_gat.common import (
    AFF_LIST,
    CLASS_NAMES,
    GatAedDataset,
    GatTrainDataset,
    MetricState,
    cosine_lr,
    dataset_statistics,
    discover_aed_samples,
    discover_train_samples,
    is_expected_lora_coverage,
    prediction_to_class_map,
    set_seed,
    trainable_parameter_rows,
    write_csv,
    write_json,
)


def load_official_gat(source_root: Path, runtime_root: Path):
    source_root = source_root.resolve()
    runtime_root = runtime_root.resolve()
    if not (source_root / "models" / "GAT.py").exists():
        raise FileNotFoundError(
            f"Missing official GAT source at {source_root / 'models' / 'GAT.py'}. "
            "Run tools/setup_upstream_aff_grasp.sh or pass --source-root."
        )
    if not (runtime_root / "dinov2_vitb14_pretrain.pth").exists():
        raise FileNotFoundError(f"Missing DINOv2 weight: {runtime_root / 'dinov2_vitb14_pretrain.pth'}")
    sys.path.insert(0, str(source_root))
    os.chdir(runtime_root)
    from models.GAT import Net

    return Net


def write_environment(output_dir: Path) -> None:
    lines = [sys.version.replace("\n", " ")]
    for command in (["git", "rev-parse", "HEAD"], ["nvidia-smi"]):
        try:
            result = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            lines.append(f"$ {' '.join(command)}")
            lines.append(result.stdout.strip())
        except OSError as exc:
            lines.append(f"$ {' '.join(command)} failed: {exc}")
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=False, text=True, stdout=subprocess.PIPE)
        lines.append("$ python -m pip freeze")
        lines.append(freeze.stdout.strip())
    except OSError as exc:
        lines.append(f"$ python -m pip freeze failed: {exc}")
    (output_dir / "environment.txt").write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def evaluate(model, loader, device: torch.device, threshold: float) -> dict:
    model.eval()
    metrics = MetricState.create()
    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="eval"):
            image = batch["image"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            pred = model(image, depth)
            pred_class = prediction_to_class_map(pred, threshold=threshold)
            metrics.update(pred_class, target)
    return metrics.compute()


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def default_run_name(args: argparse.Namespace) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"{timestamp}_gat_{args.scheduler}"
        f"_seed{args.seed}_bs{args.batch_size}_ep{args.epochs}"
    )


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        run_name = args.run_name or default_run_name(args)
        output_dir = Path(args.output_root) / run_name
    output_dir = output_dir.resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Output path exists and is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}. "
            "Pass --overwrite, --output-dir, or a different --run-name."
        )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="upstream-aff-grasp/affordance-learning")
    parser.add_argument("--runtime-root", default="affordance-learning")
    parser.add_argument("--data-root", default="affordance-learning/ag_dataset")
    parser.add_argument("--depth-root", default=None)
    parser.add_argument("--aed-root", default=None)
    parser.add_argument("--output-root", default="outputs/gat_retraining")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resize-size", type=int, default=476)
    parser.add_argument("--crop-size", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--scheduler", choices=["cosine", "multistep"], default="cosine")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--final-threshold", type=float, default=0.8)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-aed-samples", type=int, default=None)
    parser.add_argument("--disable-augmentation", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not torch.cuda.is_available():
        raise RuntimeError("The official GAT implementation constructs DINOv2 on CUDA; CUDA is required.")
    device = torch.device("cuda")
    set_seed(args.seed)

    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["resolved_output_dir"] = str(output_dir)
    config["class_names"] = CLASS_NAMES
    write_json(output_dir / "config.yaml", config)

    data_root = Path(args.data_root).resolve()
    depth_root = Path(args.depth_root).resolve() if args.depth_root else None
    samples, missing = discover_train_samples(data_root, depth_root=depth_root)
    if args.max_train_samples:
        samples = samples[: args.max_train_samples]
    write_json(output_dir / "dataset_statistics.json", dataset_statistics(samples))
    write_csv(
        output_dir / "missing_files.csv",
        missing,
        fieldnames=["image", "depth", "label", "object", "depth_exists", "label_exists", "known_object"],
    )
    print(f"num train samples: {len(samples)}")
    assert len(samples) > 0

    train_dataset = GatTrainDataset(
        samples,
        resize_size=args.resize_size,
        crop_size=args.crop_size,
        train=True,
        augment=not args.disable_augmentation,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    aed_loader = None
    if not args.skip_eval:
        aed_root = Path(args.aed_root).resolve() if args.aed_root else data_root / "Affordance_Evaluation_Dataset"
        aed_samples = discover_aed_samples(aed_root)
        if args.max_aed_samples:
            aed_samples = aed_samples[: args.max_aed_samples]
        aed_loader = DataLoader(
            GatAedDataset(aed_samples, crop_size=args.crop_size),
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    Net = load_official_gat(Path(args.source_root), Path(args.runtime_root))
    model = Net().to(device)
    trainable_rows = trainable_parameter_rows(model)
    write_csv(output_dir / "trainable_parameters.txt", trainable_rows, fieldnames=["name", "shape", "parameters"])
    write_json(
        output_dir / "model_checks.json",
        {
            "trainable_parameters": sum(row["parameters"] for row in trainable_rows),
            "trainable_parameter_tensors": len(trainable_rows),
            "all_12_qkv_lora_blocks_detected": is_expected_lora_coverage(trainable_rows),
        },
    )

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 12], gamma=0.1)

    history: list[dict] = []
    best_miou = -1.0
    best_epoch = 0
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_environment(output_dir)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_focal = 0.0
        total_dice = 0.0
        seen = 0
        for image, depth, label in tqdm(train_loader, leave=False, desc=f"epoch {epoch}"):
            image = image.to(device, non_blocking=True)
            depth = depth.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            pred, loss_dict = model(image, depth, label=label)
            loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}: {float(loss.detach().cpu())}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch = image.size(0)
            seen += batch
            total_loss += float(loss.detach().cpu()) * batch
            total_focal += float(loss_dict["focal"].detach().cpu()) * batch
            total_dice += float(loss_dict["dice"].detach().cpu()) * batch
        row = {
            "epoch": epoch,
            "train_total_loss": total_loss / max(seen, 1),
            "train_focal_loss": total_focal / max(seen, 1),
            "train_dice_loss": total_dice / max(seen, 1),
            "learning_rate": current_lr(optimizer),
        }
        if aed_loader is not None:
            eval_metrics = evaluate(model, aed_loader, device, threshold=args.threshold)
            row.update(
                {
                    "test_miou": eval_metrics["mIoU"] * 100,
                    "test_f1": eval_metrics["F1"] * 100,
                    "test_mean_accuracy": eval_metrics["Mean Accuracy"] * 100,
                    **{
                        f"iou_{name}": value * 100
                        for name, value in zip(CLASS_NAMES, eval_metrics["Class IoU"], strict=True)
                    },
                }
            )
            if row["test_miou"] > best_miou:
                best_miou = row["test_miou"]
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": best_epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "config": config,
                        "metrics": row,
                    },
                    checkpoint_dir / "best.pth",
                )
        history.append(row)
        write_csv(output_dir / "history.csv", history)
        print(row)
        scheduler.step()
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
                "metrics": row,
            },
            checkpoint_dir / "last.pth",
        )

    if aed_loader is not None and (checkpoint_dir / "best.pth").exists():
        checkpoint = torch.load(checkpoint_dir / "best.pth", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        final_metrics = evaluate(model, aed_loader, device, threshold=args.final_threshold)
        write_json(
            output_dir / "final_threshold_metrics.json",
            {
                "threshold": args.final_threshold,
                "mIoU": final_metrics["mIoU"] * 100,
                "F1": final_metrics["F1"] * 100,
                "Mean Accuracy": final_metrics["Mean Accuracy"] * 100,
                "Class IoU": {
                    name: value * 100
                    for name, value in zip(CLASS_NAMES, final_metrics["Class IoU"], strict=True)
                },
            },
        )

    print(f"best epoch: {best_epoch}, best mIoU@{args.threshold}: {best_miou:.2f}")
    print(f"saved: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
