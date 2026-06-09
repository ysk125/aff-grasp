# SegFormer / InternImage q Server Runbook

## Current Implementation Status

The previous `experiments/` tree only contained `__pycache__` files, so it was
not runnable or tracked by Git. The experiment sources have been restored under
`experiments/affgrasp_mmseg/`.

Implemented:

- Seven experiment configs: SegFormer A/B/C/D and InternImage A/C/D
- Shared Aff-Grasp dataset conversion for `ego_train` and AED
- Fixed train/val/test split generation under `experiments/splits`
- 9-class class-index target masks
- Focal + Dice loss
- mIoU, F1, Accuracy
- Parameter count logging
- Prediction panels
- Checkpoint and metrics outputs
- Detached Docker launcher for q

Important note: the implementation is lightweight PyTorch/timm based rather
than full MMSegmentation. It preserves the planned behavior and output layout,
but avoids adding fragile MMCV/MMEngine CUDA dependencies before the first q
smoke test.

## Before Running on q

Use one free GPU only. Confirm server state first:

```bash
hostname
pwd
whoami
nvidia-smi
free -h
docker ps
```

Make sure assets are ready:

```bash
cd ~/workspace/aff-grasp
bash scripts/run_docker.sh 0
python experiments/affgrasp_mmseg/preflight.py --check-timm-models
exit
```

Expected:

- `train_samples` is greater than zero
- `aed_samples` is `721`
- `timm_has_mit_b0` is `true`

If `timm_has_internimage_t_1k_224` is `false`, run SegFormer first and defer
InternImage until the q Docker image has a timm version with InternImage.

## Recommended First Smoke Test

Inside Docker:

```bash
python experiments/affgrasp_mmseg/train_affgrasp_mmseg.py \
  --config experiments/affgrasp_mmseg/configs/segformer_affgrasp/segformer_a.py \
  --epochs 1 \
  --max-train-samples 8 \
  --max-val-samples 4 \
  --gpu 0
```

This checks data loading, model creation, loss, metrics, checkpointing, and
visualization without occupying the GPU for long.

## Detached Full Run

From the q host, not inside Docker:

```bash
bash scripts/run_mmseg_experiment_detached.sh \
  0 \
  experiments/affgrasp_mmseg/configs/segformer_affgrasp/segformer_a.py \
  affgrasp-segformer-a
```

Monitor:

```bash
docker logs -f affgrasp-segformer-a
docker ps -a --filter name=affgrasp-segformer-a
nvidia-smi
```

Outputs:

```text
outputs/segformer_a/
  checkpoints/best.pth
  logs/history.csv
  logs/parameter_summary.json
  metrics.csv
  visualizations/
  test/metrics.csv
  test/visualizations/
  config.py
  config.yaml
```

## Experiment Order

Run one experiment at a time:

```text
segformer_a
segformer_d
segformer_b
segformer_c
internimage_a
internimage_d
internimage_c
```

Do not start multiple training jobs on the same GPU. Remove stopped containers
after checking logs:

```bash
docker rm affgrasp-segformer-a
```
