# q Docker Qualitative Inference Checklist

Goal: on lab server `q`, run the released Aff-Grasp affordance model inside
Docker, save low-IoU AED segmentation maps, and run inference-only maps for RGB-D
Part Affordance Dataset.

## Short Path

```bash
ssh q
cd ~/workspace/Lab/project
git clone https://github.com/ysk125/aff-grasp.git
cd aff-grasp
bash scripts/q_check_env.sh
bash scripts/build_docker.sh
bash scripts/run_docker.sh 0
```

Inside Docker:

```bash
bash scripts/bootstrap_affgrasp.sh
python tools/run_aff_grasp_eval_maps.py \
  --save-filter low-iou \
  --iou-threshold 0.60 \
  --foreground-iou-threshold 0.50 \
  --max-samples 10
```

## AED Full Run

```bash
python tools/run_aff_grasp_eval_maps.py \
  --save-filter low-iou \
  --iou-threshold 0.60 \
  --foreground-iou-threshold 0.50 \
  --save-raw
```

Outputs:

```text
affordance-learning/results/<timestamp>/aed/
```

AED saves only weak examples by default: `image_miou < 0.60` or
`foreground_iou < 0.50`. All processed samples are still recorded in
`manifest.csv`.

## UMD Inference

Put RGB-D Part Affordance Dataset here:

```text
datasets/umd_part_affordance/
```

```bash
python tools/run_umd_aff_grasp_maps.py \
  --dataset-root datasets/umd_part_affordance \
  --max-samples 10
```

Full first pass:

```bash
python tools/run_umd_aff_grasp_maps.py \
  --dataset-root datasets/umd_part_affordance \
  --max-samples 300
```

Outputs:

```text
affordance-learning/results/<timestamp>/umd/
```

Important files:

- `panels/`: side-by-side image, predicted overlay, predicted mask, ground truth
- `pred_overlays/`: prediction overlaid on the original RGB image
- `pred_masks/`: predicted segmentation map only
- `gt_masks/`: ground-truth segmentation map, AED only
- `manifest.csv`: image-to-output file index
- `summary.json`: sample count and metric summary
- `raw_pred_npy/`: normalized per-affordance maps, saved only with `--save-raw`

## Affordance colors

The output mask indices use the upstream affordance order:

```text
0 background
1 grasp
2 cut
3 scoop
4 pound
5 support
6 screw
7 contain
8 stick
```

Keep `panels/` for fast visual review and `raw_pred_npy/` if later analysis or
threshold tuning is needed.
