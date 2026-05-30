#!/usr/bin/env python3
"""Unit tests for AED analysis metric helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from analysis_common import assign_rank_tertiles, binary_metrics, boundary_metrics, matched_gt_regions
from analyze_train_distribution import _task_mask
from download_aff_grasp_assets import _link_or_copy


class AnalysisMetricTests(unittest.TestCase):
    def test_binary_metrics(self) -> None:
        gt = np.array([[1, 1], [0, 0]], dtype=bool)
        pred = np.array([[1, 0], [1, 0]], dtype=bool)
        metrics = binary_metrics(pred, gt)
        self.assertAlmostEqual(metrics["iou"], 1 / 3)
        self.assertAlmostEqual(metrics["recall"], 1 / 2)
        self.assertAlmostEqual(metrics["f1"], 1 / 2)

    def test_unmatched_gt_region_receives_zero(self) -> None:
        gt = np.zeros((8, 8), dtype=bool)
        gt[1:3, 1:3] = True
        gt[5:7, 5:7] = True
        pred = np.zeros((8, 8), dtype=bool)
        pred[1:3, 1:3] = True
        rows = matched_gt_regions(pred, gt)
        self.assertEqual(len(rows), 2)
        self.assertEqual(sorted(row["iou"] for row in rows), [0.0, 1.0])

    def test_thin_region_has_large_aspect_ratio(self) -> None:
        gt = np.zeros((8, 12), dtype=bool)
        gt[3:4, 1:11] = True
        rows = matched_gt_regions(gt, gt)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["bbox_aspect_ratio"], 10.0)
        self.assertEqual(rows[0]["f1"], 1.0)

    def test_boundary_metrics_are_perfect_for_equal_masks(self) -> None:
        mask = np.zeros((12, 12), dtype=bool)
        mask[3:9, 4:10] = True
        metrics = boundary_metrics(mask, mask, width=3)
        self.assertEqual(metrics["boundary_iou"], 1.0)
        self.assertEqual(metrics["boundary_f1"], 1.0)
        self.assertEqual(metrics["boundary_neighborhood_iou"], 1.0)

    def test_training_mask_mapping(self) -> None:
        raw = np.array([[0, 128, 255]], dtype=np.uint8)
        mapped = _task_mask(raw, "cut")
        np.testing.assert_array_equal(mapped, np.array([[0, 1, 2]], dtype=np.uint8))

    def test_rank_tertiles_are_nearly_equal(self) -> None:
        rows = [{"score": 0.5} for _ in range(7)]
        assign_rank_tertiles(rows, "score", "group")
        self.assertEqual([row["group"] for row in rows].count("lower"), 3)
        self.assertEqual([row["group"] for row in rows].count("middle"), 2)
        self.assertEqual([row["group"] for row in rows].count("upper"), 2)

    def test_broken_asset_symlink_is_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            destination = root / "destination"
            destination.symlink_to(root / "missing", target_is_directory=True)
            _link_or_copy(source, destination, copy=False)
            self.assertEqual(destination.resolve(), source.resolve())


if __name__ == "__main__":
    unittest.main()
