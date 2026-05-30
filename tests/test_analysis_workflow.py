#!/usr/bin/env python3
"""Small end-to-end test for the post-inference AED analysis workflow."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class AnalysisWorkflowTests(unittest.TestCase):
    def test_post_inference_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "analysis" / "fixture"
            dataset = Path(temp) / "dataset"
            panels = root / "panels"
            pred_dir = root / "label_masks" / "pred"
            gt_dir = root / "label_masks" / "gt"
            jpeg_dir = dataset / "JPEGImages"
            for path in [panels, pred_dir, gt_dir, jpeg_dir]:
                path.mkdir(parents=True)

            manifest = []
            for index in range(2):
                image_name = f"sample_{index}.jpg"
                stem = f"{index:04d}_sample_{index}"
                rgb = np.full((16, 16, 3), 80 + index * 60, dtype=np.uint8)
                gt = np.zeros((16, 16), dtype=np.uint8)
                gt[3:10, 4:11] = 1 + index
                pred = gt.copy()
                if index:
                    pred[3:5, 4:11] = 0
                    pred[12:14, 12:14] = 1
                Image.fromarray(rgb).save(jpeg_dir / image_name)
                Image.fromarray(pred).save(pred_dir / f"{stem}_pred.png")
                Image.fromarray(gt).save(gt_dir / f"{stem}_gt.png")
                Image.fromarray(rgb).save(panels / f"{stem}_panel.png")
                manifest.append(
                    {
                        "index": index,
                        "image": image_name,
                        "object": "fixture",
                        "saved": True,
                        "save_reason": "analysis",
                        "image_miou": 1.0 if index == 0 else 0.7,
                        "foreground_iou": 1.0 if index == 0 else 0.7,
                        "class_iou_json": "{}",
                        "pred_mask": f"label_masks/pred/{stem}_pred.png",
                        "pred_overlay": "",
                        "gt_mask": f"label_masks/gt/{stem}_gt.png",
                        "panel": f"panels/{stem}_panel.png",
                        "raw_pred": "",
                    }
                )
            _write_csv(root / "manifest.csv", manifest)
            (root / "run_config.json").write_text(
                json.dumps({"dataset_root": str(dataset), "threshold": 0.8, "class_names": ["grasp", "cut"]})
            )

            self._run("tools/analyze_aed_metrics.py", "--analysis-root", str(root))
            self._run("tools/build_aed_review_bundle.py", "--analysis-root", str(root))
            self.assertTrue((root / "metrics" / "region_metrics.csv").exists())
            self.assertTrue((root / "review" / "index.html").exists())

            annotations = []
            for index in range(2):
                annotations.append(
                    {
                        "review_id": f"{index:04d}",
                        "reviewed": "true",
                        "complex_background": "true" if index else "false",
                        "uncertain": "false",
                        "miss_small_region": "false",
                        "miss_thin_region": "false",
                        "boundary_error": "true" if index else "false",
                        "background_false_positive": "true" if index else "false",
                        "under_segmentation": "false",
                        "over_segmentation": "false",
                        "class_confusion": "false",
                        "no_obvious_failure": "false",
                        "other": "false",
                        "note": "",
                    }
                )
            _write_csv(root / "review" / "annotations.csv", annotations)
            self._run("tools/merge_aed_review_annotations.py", "--analysis-root", str(root))
            self.assertTrue((root / "metrics" / "failure_tag_summary.csv").exists())

            train_root = Path(temp) / "ego_train"
            train_root.mkdir()
            Image.fromarray(np.array([[0, 128, 255]], dtype=np.uint8)).save(train_root / "knife-sample-label.png")
            Image.fromarray(np.array([[0, 128, 255]], dtype=np.uint8)).save(train_root / "cup-sample-label.png")
            self._run("tools/analyze_train_distribution.py", "--analysis-root", str(root), "--train-root", str(train_root))
            self.assertTrue((root / "metrics" / "train_distribution.csv").exists())

    def _run(self, *args: str) -> None:
        subprocess.run([sys.executable, *args], cwd=REPO_ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


if __name__ == "__main__":
    unittest.main()
