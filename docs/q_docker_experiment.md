# q Docker Experiment

Use this after logging in to the lab server. Do not share SSH passwords or
tokens.

## 1. Login and clone

```bash
ssh q
cd ~/workspace/Lab/project
git clone https://github.com/ysk125/aff-grasp.git
cd aff-grasp
```

If the repository already exists:

```bash
cd ~/workspace/Lab/project/aff-grasp
git pull
```

## 2. Check q before running

```bash
bash scripts/q_check_env.sh
```

Pick one free GPU from `nvidia-smi`. The examples below use GPU `0`.

## 3. Build and enter Docker

```bash
bash scripts/build_docker.sh
bash scripts/run_docker.sh 0
```

Inside Docker:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 4. Bootstrap Aff-Grasp

Inside Docker:

```bash
bash scripts/bootstrap_affgrasp.sh
```

This clones upstream Aff-Grasp, builds the CUDA extension, and downloads AED,
the released model checkpoint, and the DINOv2-base checkpoint.

## 5. AED smoke test

```bash
python tools/run_aff_grasp_eval_maps.py \
  --save-filter low-iou \
  --iou-threshold 0.60 \
  --foreground-iou-threshold 0.50 \
  --max-samples 10
```

## 6. AED full qualitative run

```bash
python tools/run_aff_grasp_eval_maps.py \
  --save-filter low-iou \
  --iou-threshold 0.60 \
  --foreground-iou-threshold 0.50 \
  --save-raw
```

AED outputs are written under:

```text
affordance-learning/results/<timestamp>/aed/
```

Only low-IoU examples are saved as PNGs. Every processed sample still appears in
`manifest.csv`.

## 7. UMD inference

Place RGB-D Part Affordance Dataset under:

```text
datasets/umd_part_affordance/
```

Then run:

```bash
python tools/run_umd_aff_grasp_maps.py \
  --dataset-root datasets/umd_part_affordance \
  --max-samples 10

python tools/run_umd_aff_grasp_maps.py \
  --dataset-root datasets/umd_part_affordance \
  --max-samples 300
```

UMD outputs are written under:

```text
affordance-learning/results/<timestamp>/umd/
```

UMD is inference-only in this first pass. Ground-truth label conversion and
quantitative evaluation are intentionally deferred.
