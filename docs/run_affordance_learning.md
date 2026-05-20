# Running the Aff-Grasp Affordance Model

This repo follows the upstream `affordance-learning` entry points. The quickest
first target is evaluation with the released checkpoint.

## 1. Create an environment

```bash
python -m venv .venv-affgrasp
source .venv-affgrasp/bin/activate
pip install --upgrade pip
pip install -r requirements-affgrasp.txt
```

Install a CUDA-specific PyTorch build first if the default wheel does not match
your GPU driver.

## 2. Build the CUDA attention op

The ViT-Adapter code uses the Deformable-DETR multi-scale deformable attention
CUDA extension.

```bash
cd affordance-learning/models/dino/ops
python setup.py build install
cd ../../../..
```

## 3. Download data and weights

```bash
python tools/download_aff_grasp_assets.py
```

The script downloads:

- `Gen1113/Data_for_Aff-Grasp`
- `Gen1113/Model_for_Aff-Grasp`
- `dinov2_vitb14_pretrain.pth` from the official DINOv2 public weight URL

It then prepares the paths expected by the original code under
`affordance-learning/`.

## 4. Run evaluation

```bash
cd affordance-learning
python test.py --data_root ag_dataset --model_file pretrained_aff_grasp.pth --gpu 0 --viz
```

Outputs are written next to the checkpoint path. Use `--gpu 1` or another GPU id
when needed.

## Notes

- `models/GAT.py` loads `dinov2_vitb14_pretrain.pth` from the current working
  directory, so run `test.py` from inside `affordance-learning`.
- `test.py` currently assumes CUDA and calls `torch.cuda.set_device`, so a CPU
  only run needs a small code change.
- Training uses `train.py --data_root ag_dataset` after the same asset setup.
