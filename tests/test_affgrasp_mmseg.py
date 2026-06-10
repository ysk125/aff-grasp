#!/usr/bin/env python3
"""Regression tests for the follow-up segmentation experiment contract."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

try:
    import torch.nn as nn
except ModuleNotFoundError:
    nn = None
else:
    from experiments.affgrasp_mmseg.common import (
        apply_transformer_freeze_policy,
        load_config,
        validate_split_isolation,
        validate_internimage_checkpoint_keys,
    )
    from experiments.affgrasp_mmseg.train_affgrasp_mmseg import optimizer_group_name, should_drop_last


REPO_ROOT = Path(__file__).resolve().parents[1]


if nn is not None:
    class _FakeSegFormer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.segformer = nn.Module()
            self.segformer.encoder = nn.Module()
            self.segformer.encoder.patch_embeddings = nn.ModuleList([nn.Linear(2, 2) for _ in range(4)])
            self.segformer.encoder.block = nn.ModuleList([nn.Linear(2, 2) for _ in range(4)])
            self.decode_head = nn.Linear(2, 2)


@unittest.skipUnless(nn is not None, "PyTorch is required for the mmseg experiment contract tests")
class AffGraspMmsegContractTests(unittest.TestCase):
    def test_internimage_configs_match_official_ade20k_architecture(self) -> None:
        expected = {
            "core_op": "DCNv3",
            "channels": 80,
            "depths": [4, 4, 21, 4],
            "groups": [5, 10, 20, 40],
            "mlp_ratio": 4.0,
            "drop_path_rate": 0.3,
            "norm_layer": "LN",
            "layer_scale": 1.0,
            "offset_scale": 1.0,
            "post_norm": True,
            "with_cp": False,
        }
        config_dir = REPO_ROOT / "experiments" / "affgrasp_mmseg" / "configs" / "internimage_affgrasp"
        for name in ["internimage_a.py", "internimage_c.py", "internimage_d.py"]:
            with self.subTest(config=name):
                cfg = load_config(config_dir / name)
                self.assertEqual(cfg["internimage_backbone"], expected)

    def test_internimage_checkpoint_validation_allows_only_replaced_heads(self) -> None:
        backbone = SimpleNamespace(missing_keys=[], unexpected_keys=[])
        head = SimpleNamespace(missing_keys=["classifier.weight", "classifier.bias"], unexpected_keys=[])
        auxiliary = SimpleNamespace(missing_keys=["classifier.weight", "classifier.bias"], unexpected_keys=[])
        validate_internimage_checkpoint_keys(backbone, head, auxiliary)

    def test_internimage_checkpoint_validation_rejects_architecture_mismatch(self) -> None:
        backbone = SimpleNamespace(missing_keys=["levels.0.norm.0.weight"], unexpected_keys=["levels.0.blocks.0.gamma1"])
        head = SimpleNamespace(missing_keys=["classifier.weight", "classifier.bias"], unexpected_keys=[])
        auxiliary = SimpleNamespace(missing_keys=["classifier.weight", "classifier.bias"], unexpected_keys=[])
        with self.assertRaises(RuntimeError):
            validate_internimage_checkpoint_keys(backbone, head, auxiliary)

    def test_segformer_partial_tuning_includes_stage_three_patch_embedding(self) -> None:
        model = _FakeSegFormer()
        apply_transformer_freeze_policy(model, {"freeze_mode": "partial", "use_lora": False})
        encoder = model.segformer.encoder
        self.assertFalse(encoder.patch_embeddings[0].weight.requires_grad)
        self.assertFalse(encoder.patch_embeddings[1].weight.requires_grad)
        self.assertTrue(encoder.patch_embeddings[2].weight.requires_grad)
        self.assertTrue(encoder.patch_embeddings[3].weight.requires_grad)

    def test_split_validation_rejects_aed_in_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train_root = root / "ego_train"
            aed_root = root / "aed"
            split_dir = root / "splits"
            train_root.mkdir()
            aed_root.mkdir()
            split_dir.mkdir()
            train_image = train_root / "train.jpg"
            train_label = train_root / "train-label.png"
            aed_image = aed_root / "test.jpg"
            aed_label = aed_root / "test.npy"
            for path in [train_image, train_label, aed_image, aed_label]:
                path.touch()
            (split_dir / "train.txt").write_text(f"{train_image}\t{train_label}\n")
            (split_dir / "val.txt").write_text(f"{aed_image}\t{aed_label}\n")
            (split_dir / "test.txt").write_text(f"{aed_image}\t{aed_label}\n")
            with self.assertRaises(RuntimeError):
                validate_split_isolation(split_dir, train_root, aed_root)

    def test_optimizer_parameter_groups(self) -> None:
        self.assertEqual(optimizer_group_name("model.segformer.encoder.block.3.weight"), "backbone")
        self.assertEqual(optimizer_group_name("model.decode_head.linear_c.0.weight"), "decoder")
        self.assertEqual(optimizer_group_name("model.decode_head.classifier.weight"), "classifier")
        self.assertEqual(optimizer_group_name("model.segformer.encoder.block.3.query.lora_a.weight"), "peft")
        self.assertEqual(optimizer_group_name("adapters.2.net.0.weight"), "peft")

    def test_singleton_final_batch_is_dropped(self) -> None:
        self.assertTrue(should_drop_last(281, 4))
        self.assertTrue(should_drop_last(281, 8))
        self.assertFalse(should_drop_last(280, 4))
        self.assertFalse(should_drop_last(282, 4))


if __name__ == "__main__":
    unittest.main()
