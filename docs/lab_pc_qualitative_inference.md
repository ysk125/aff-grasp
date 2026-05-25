# Lab PC Qualitative Inference Checklist

Goal: run the released Aff-Grasp affordance model on
`Affordance_Evaluation_Dataset` and keep segmentation maps for qualitative
inspection.

## 1. Clone this preparation repo

```bash
git clone https://github.com/ysk125/aff-grasp.git
cd aff-grasp
```

## 2. Get the upstream Aff-Grasp code

```bash
git clone https://github.com/Reagan1311/Aff-Grasp.git upstream-aff-grasp
ln -s upstream-aff-grasp/affordance-learning affordance-learning
```

Or use the helper script:

```bash
bash tools/setup_upstream_aff_grasp.sh
```

If symlinks are inconvenient on the lab machine, copy the directory instead:

```bash
cp -r upstream-aff-grasp/affordance-learning ./affordance-learning
```

## 3. Create the Python environment

```bash
python -m venv .venv-affgrasp
source .venv-affgrasp/bin/activate
pip install --upgrade pip
pip install -r requirements-affgrasp.txt
```

Install the CUDA-specific PyTorch wheel first if the default one does not match
the lab PC CUDA driver.

## 4. Build the CUDA extension

```bash
cd affordance-learning/models/dino/ops
python setup.py build install
cd ../../../..
```

## 5. Download only evaluation data and weights

```bash
python tools/download_aff_grasp_assets.py --eval-only
```

This prepares:

- `affordance-learning/ag_dataset/Affordance_Evaluation_Dataset`
- `affordance-learning/pretrained_aff_grasp.pth`
- `affordance-learning/dinov2_vitb14_pretrain.pth`

## 6. Run inference and save qualitative maps

```bash
python tools/run_aff_grasp_eval_maps.py --gpu 0 --save-raw
```

For a quick smoke test:

```bash
python tools/run_aff_grasp_eval_maps.py --gpu 0 --max-samples 10
```

Outputs are saved under:

```text
affordance-learning/qualitative_outputs/affordance_eval/
```

Important files:

- `panels/`: side-by-side image, predicted overlay, predicted mask, ground truth
- `pred_overlays/`: prediction overlaid on the original RGB image
- `pred_masks/`: predicted segmentation map only
- `gt_masks/`: ground-truth segmentation map
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
