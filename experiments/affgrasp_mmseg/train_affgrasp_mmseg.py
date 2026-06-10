#!/usr/bin/env python3
"""Train SegFormer/InternImage-style Aff-Grasp semantic segmentation experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.affgrasp_mmseg.common import (
    AffGraspSegDataset,
    FocalDiceLoss,
    MetricState,
    build_model,
    copy_config,
    ensure_splits,
    load_config,
    parameter_summary,
    read_split,
    save_panel,
    set_seed,
    write_csv,
)


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> dict:
    model.train(train)
    metrics = MetricState.create()
    total_loss = 0.0
    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, leave=False):
            image = batch["image"].to(device)
            target = batch["target"].to(device)
            output = model(image)
            if isinstance(output, tuple):
                logits, auxiliary_logits = output
                loss = criterion(logits, target) + model.auxiliary_loss_weight * criterion(auxiliary_logits, target)
            else:
                logits = output
                loss = criterion(logits, target)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            pred = logits.argmax(dim=1)
            metrics.update(pred, target)
            total_loss += float(loss.detach().cpu()) * image.size(0)
    out = metrics.compute(ignore_background=True)
    out["loss"] = total_loss / max(len(loader.dataset), 1)
    return out


def save_visualizations(model, dataset, output_dir: Path, device, limit: int = 24) -> None:
    model.eval()
    rows = []
    for idx in range(min(len(dataset), limit)):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        target = sample["target"]
        with torch.no_grad():
            pred = model(image).argmax(dim=1).squeeze(0).cpu()
        error = pred != target
        out = output_dir / "visualizations" / f"{idx:04d}_{sample['name']}.png"
        save_panel(out, sample["image"], target, pred, error)
        rows.append({"index": idx, "name": sample["name"], "panel": str(out)})
    write_csv(output_dir / "visualizations.csv", rows)


def build_optimizer(model, cfg: dict) -> torch.optim.Optimizer:
    backbone_params = []
    task_params = []
    backbone_lr = float(cfg.get("backbone_lr", cfg["lr"]))
    task_lr = float(cfg["lr"])
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = name.startswith("model.segformer.encoder.") or name.startswith("backbone.")
        is_peft = any(token in name for token in ["lora_a", "lora_b", "adapters."])
        if is_backbone and not is_peft:
            backbone_params.append(param)
        else:
            task_params.append(param)
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr, "group_name": "backbone"})
    if task_params:
        groups.append({"params": task_params, "lr": task_lr, "group_name": "task"})
    if not groups:
        raise RuntimeError("No trainable parameters found for optimizer")
    print(
        json.dumps(
            {
                "optimizer_groups": [
                    {
                        "name": group["group_name"],
                        "lr": group["lr"],
                        "parameters": sum(param.numel() for param in group["params"]),
                    }
                    for group in groups
                ]
            },
            indent=2,
        )
    )
    return torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--train-root", default="affordance-learning/ag_dataset/ego_train")
    parser.add_argument("--aed-root", default="affordance-learning/ag_dataset/Affordance_Evaluation_Dataset")
    parser.add_argument("--split-dir", default="experiments/splits")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--test-after", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    set_seed(int(cfg.get("seed", 0)))
    train_root = Path(args.train_root).resolve()
    aed_root = Path(args.aed_root).resolve()
    split_dir = Path(args.split_dir).resolve()
    ensure_splits(split_dir, train_root, aed_root)

    output_dir = Path(args.output_root).resolve() / cfg["experiment_type"]
    copy_config(config_path, output_dir, cfg)
    train_rows = read_split(split_dir / "train.txt")
    val_rows = read_split(split_dir / "val.txt")
    if args.max_train_samples:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_val_samples:
        val_rows = val_rows[: args.max_val_samples]
    train_dataset = AffGraspSegDataset(train_rows, cfg["resize_size"], cfg["crop_size"], train=True, source="train")
    val_dataset = AffGraspSegDataset(val_rows, cfg["resize_size"], cfg["crop_size"], train=False, source="train")
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)

    model = build_model(cfg).to(device)
    params = parameter_summary(model)
    print(json.dumps(params, indent=2))
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    with (output_dir / "logs" / "parameter_summary.json").open("w") as stream:
        json.dump(params, stream, indent=2)

    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(value) for value in cfg.get("lr_milestones", [10, 12])],
        gamma=float(cfg.get("lr_gamma", 0.1)),
    )
    criterion = FocalDiceLoss(alpha=float(cfg.get("focal_alpha", 1.0)))
    best_miou = -1.0
    history = []
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        write_csv(output_dir / "logs" / "history.csv", history)
        print(json.dumps(row, indent=2))
        if val_metrics["mIoU"] > best_miou:
            best_miou = val_metrics["mIoU"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "metrics": val_metrics,
                },
                ckpt_dir / "best.pth",
            )
        scheduler.step()
    best_checkpoint = torch.load(ckpt_dir / "best.pth", map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["model"])
    save_visualizations(model, val_dataset, output_dir, device)
    best_row = max(history, key=lambda item: item["val_mIoU"])
    metrics_row = {
        "model_name": cfg["model_name"],
        "experiment_type": cfg["experiment_type"],
        "input_resolution": cfg["crop_size"],
        "mIoU": best_row["val_mIoU"],
        "F1": best_row["val_F1"],
        "Accuracy": best_row["val_Accuracy"],
        **params,
        "best_epoch": best_row["epoch"],
        "checkpoint_path": str(ckpt_dir / "best.pth"),
    }
    write_csv(output_dir / "metrics.csv", [metrics_row])
    if args.test_after:
        from experiments.affgrasp_mmseg.eval_affgrasp_mmseg import evaluate

        evaluate(
            config_path,
            ckpt_dir / "best.pth",
            output_dir / "test",
            aed_root,
            split_dir,
            device,
            cfg,
            max_samples=args.max_test_samples,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
