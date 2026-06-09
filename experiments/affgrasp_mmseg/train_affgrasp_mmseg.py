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
            logits = model(image)
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
    parser.add_argument("--test-after", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

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

    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
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
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "metrics": val_metrics}, ckpt_dir / "best.pth")
    save_visualizations(model, val_dataset, output_dir, device)
    metrics_row = {
        "model_name": cfg["model_name"],
        "experiment_type": cfg["experiment_type"],
        "input_resolution": cfg["crop_size"],
        "mIoU": history[-1]["val_mIoU"],
        "F1": history[-1]["val_F1"],
        "Accuracy": history[-1]["val_Accuracy"],
        **params,
        "best_epoch": max(history, key=lambda item: item["val_mIoU"])["epoch"],
        "checkpoint_path": str(ckpt_dir / "best.pth"),
    }
    write_csv(output_dir / "metrics.csv", [metrics_row])
    if args.test_after:
        from experiments.affgrasp_mmseg.eval_affgrasp_mmseg import evaluate

        evaluate(config_path, ckpt_dir / "best.pth", output_dir / "test", aed_root, split_dir, device, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

