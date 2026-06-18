#!/usr/bin/env python3
"""Regression tests for the GAT retraining utilities."""

from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.affgrasp_gat.common import (
    CLASS_ID_BY_OBJECT,
    MetricState,
    dataset_statistics,
    discover_train_samples,
    label_to_class_index,
    prediction_to_class_map,
    resolve_depth_root,
)
from experiments.affgrasp_gat.train_gat import resolve_output_dir


class AffGraspGatTests(unittest.TestCase):
    def test_discovers_spec_layout_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "Data_for-Aff-Grasp"
            train = root / "ego_train"
            depth = root / "depth"
            train.mkdir(parents=True)
            depth.mkdir()
            Image.new("RGB", (4, 4)).save(train / "knife_000-img.jpg")
            Image.fromarray(np.array([[0, 128], [255, 0]], dtype=np.uint8)).save(train / "knife_000-label.png")
            Image.new("RGB", (4, 4)).save(depth / "knife_000-img_graydepth.png")

            samples, missing = discover_train_samples(root)

            self.assertEqual(len(samples), 1)
            self.assertEqual(missing, [])
            self.assertEqual(samples[0].object_name, "knife")
            self.assertEqual(resolve_depth_root(root), depth.resolve())

    def test_discovers_existing_asset_layout_depth_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            aff_root = Path(temp) / "affordance-learning"
            data_root = aff_root / "ag_dataset"
            train = data_root / "ego_train"
            depth = aff_root / "depth"
            train.mkdir(parents=True)
            depth.mkdir(parents=True)
            Image.new("RGB", (4, 4)).save(train / "cup_000-img.jpg")
            Image.fromarray(np.array([[0, 128], [255, 0]], dtype=np.uint8)).save(train / "cup_000-label.png")
            Image.new("RGB", (4, 4)).save(depth / "cup_000-img_graydepth.png")

            samples, missing = discover_train_samples(data_root)

            self.assertEqual(len(samples), 1)
            self.assertEqual(missing, [])
            self.assertEqual(samples[0].depth_path, (depth / "cup_000-img_graydepth.png").resolve())

    def test_label_conversion_maps_functional_region_from_object_name(self) -> None:
        raw = np.array([[0, 128, 255]], dtype=np.uint8)
        converted = label_to_class_index(raw, "hammer")

        self.assertEqual(converted.tolist(), [[0, 1, CLASS_ID_BY_OBJECT["hammer"]]])

    def test_label_conversion_rejects_interpolated_values(self) -> None:
        raw = np.array([[0, 127, 255]], dtype=np.uint8)
        with self.assertRaises(ValueError):
            label_to_class_index(raw, "knife")

    def test_prediction_threshold_sets_background(self) -> None:
        pred = torch.zeros(1, 8, 2, 2)
        pred[:, 2] = 0.9
        pred[:, :, 0, 0] = 0.1

        out = prediction_to_class_map(pred, threshold=0.8)

        self.assertEqual(out[0, 0, 0].item(), 0)
        self.assertEqual(out[0, 1, 1].item(), 3)

    def test_metrics_ignore_background_in_macro_scores(self) -> None:
        metrics = MetricState.create()
        pred = torch.tensor([[0, 1], [0, 2]])
        target = torch.tensor([[0, 1], [2, 2]])

        result = metrics.update(pred, target)
        result = metrics.compute()

        self.assertLess(result["mIoU"], 1.0)
        self.assertGreater(result["mIoU"], 0.0)

    def test_dataset_statistics_reports_class_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dataset"
            train = root / "ego_train"
            depth = root / "depth"
            train.mkdir(parents=True)
            depth.mkdir()
            Image.new("RGB", (4, 4)).save(train / "fork_000-img.jpg")
            Image.fromarray(np.array([[0, 128], [255, 0]], dtype=np.uint8)).save(train / "fork_000-label.png")
            Image.new("RGB", (4, 4)).save(depth / "fork_000-img_graydepth.png")
            samples, _ = discover_train_samples(root)

            stats = dataset_statistics(samples)

            self.assertEqual(stats["train_samples"], 1)
            self.assertEqual(stats["class_image_counts"]["grasp"], 1)
            self.assertEqual(stats["class_image_counts"]["stick"], 1)

    def test_resolve_output_dir_uses_named_run_under_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = Namespace(
                output_dir=None,
                output_root=str(Path(temp) / "gat_retraining"),
                run_name="baseline_cosine_seed0",
                overwrite=False,
            )

            output_dir = resolve_output_dir(args)

            self.assertEqual(output_dir, (Path(temp) / "gat_retraining" / "baseline_cosine_seed0").resolve())

    def test_resolve_output_dir_keeps_explicit_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = Namespace(
                output_dir=str(Path(temp) / "custom"),
                output_root=str(Path(temp) / "gat_retraining"),
                run_name="ignored",
                overwrite=False,
            )

            output_dir = resolve_output_dir(args)

            self.assertEqual(output_dir, (Path(temp) / "custom").resolve())

    def test_resolve_output_dir_rejects_nonempty_directory_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "gat_retraining" / "baseline"
            output.mkdir(parents=True)
            (output / "history.csv").write_text("epoch,loss\n", encoding="utf-8")
            args = Namespace(
                output_dir=None,
                output_root=str(Path(temp) / "gat_retraining"),
                run_name="baseline",
                overwrite=False,
            )

            with self.assertRaises(FileExistsError):
                resolve_output_dir(args)


if __name__ == "__main__":
    unittest.main()
