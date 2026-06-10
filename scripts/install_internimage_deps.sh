#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U openmim mmengine
python -m mim install "mmcv>=2.0.0,<2.2.0"
python -m pip install "mmpretrain>=1.2.0,<1.3.0"

python - <<'PY'
import mmpretrain
from mmpretrain.registry import MODELS

model = MODELS.build(
    dict(
        type="InternImage",
        channels=80,
        depths=[4, 4, 21, 4],
        groups=[5, 10, 20, 40],
        mlp_ratio=4.0,
        drop_path_rate=0.2,
        out_indices=(0, 1, 2, 3),
    )
)
print("mmpretrain", getattr(mmpretrain, "__version__", "unknown"))
print("InternImage backend OK:", type(model).__name__)
PY
