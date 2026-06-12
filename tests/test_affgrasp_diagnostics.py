#!/usr/bin/env python3
"""Unit tests for background-aware segmentation diagnostics."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from experiments.affgrasp_mmseg.diagnostics import (
    confusion_matrix,
    metrics_from_confusion,
    normalized_by_gt,
    write_diagnostic_outputs,
)


class AffGraspDiagnosticsTests(unittest.TestCase):
    def test_perfect_prediction_scores_one(self) -> None:
        matrix = np.diag([10, 4, 3, 2, 1, 2, 3, 4, 5])
        result = metrics_from_confusion(matrix)
        self.assertEqual(result["miou_with_background"], 1.0)
        self.assertEqual(result["miou_without_background"], 1.0)
        self.assertEqual(result["foreground_only_accuracy"], 1.0)
        self.assertEqual(result["foreground_ratio_gap"], 0.0)

    def test_background_only_prediction_exposes_foreground_failure(self) -> None:
        matrix = np.zeros((9, 9), dtype=np.int64)
        matrix[0, 0] = 900
        matrix[1, 0] = 100
        result = metrics_from_confusion(matrix)
        self.assertEqual(result["accuracy_with_background"], 0.9)
        self.assertEqual(result["foreground_only_accuracy"], 0.0)
        self.assertEqual(result["gt_foreground_ratio"], 0.1)
        self.assertEqual(result["predicted_foreground_ratio"], 0.0)
        self.assertEqual(result["foreground_ratio_gap"], -0.1)

    def test_macro_average_excludes_gt_absent_classes(self) -> None:
        matrix = np.zeros((9, 9), dtype=np.int64)
        matrix[0, 0] = 10
        matrix[1, 1] = 5
        result = metrics_from_confusion(matrix)
        self.assertEqual(result["miou_with_background"], 1.0)
        self.assertEqual(result["miou_without_background"], 1.0)

    def test_ignore_label_is_not_counted(self) -> None:
        target = np.array([[0, 1, 255]])
        prediction = np.array([[0, 0, 8]])
        matrix = confusion_matrix(prediction, target, num_classes=9)
        self.assertEqual(int(matrix.sum()), 2)
        self.assertEqual(int(matrix[0, 0]), 1)
        self.assertEqual(int(matrix[1, 0]), 1)

    def test_normalization_is_by_gt_row(self) -> None:
        matrix = np.zeros((9, 9), dtype=np.int64)
        matrix[1, 0] = 1
        matrix[1, 1] = 3
        normalized = normalized_by_gt(matrix)
        self.assertAlmostEqual(normalized[1, 0], 0.25)
        self.assertAlmostEqual(normalized[1, 1], 0.75)
        self.assertEqual(float(normalized[2].sum()), 0.0)

    def test_expected_output_files_are_written(self) -> None:
        matrix = np.diag([10, 1, 1, 1, 1, 1, 1, 1, 1])
        per_image = [{"image_id": "sample.jpg", **metrics_from_confusion(matrix)}]
        class_names = ["background", "grasp", "cut", "scoop", "pound", "support", "screw", "contain", "stick"]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_diagnostic_outputs(root, "segformer", "segformer_a", matrix, per_image, class_names)
            for name in [
                "diagnostics_summary.csv",
                "diagnostics_per_image.csv",
                "confusion_matrix_raw.csv",
                "confusion_matrix_normalized_by_gt.csv",
                "confusion_matrix_raw.png",
                "confusion_matrix_normalized_by_gt.png",
                "key_confusions.csv",
            ]:
                self.assertTrue((root / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
