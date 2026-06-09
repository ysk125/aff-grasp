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

## Smoke Tests For All Experiments

Inside Docker:

```bash
bash experiments/affgrasp_mmseg/run_all_smoke_tests.sh 0
```

This runs all seven configs sequentially with one epoch, eight train samples,
and four validation samples. It checks data loading, model creation, freezing
policy, loss, metrics, checkpointing, and visualization without occupying the
GPU for long.

Smoke-test outputs:

```text
outputs_smoke/
  segformer_a/
  segformer_b/
  segformer_c/
  segformer_d/
  internimage_a/
  internimage_c/
  internimage_d/
```

## Detached Full Run For All Experiments

From the q host, not inside Docker:

```bash
bash scripts/run_all_mmseg_experiments_detached.sh 0 affgrasp-mmseg-all
```

Monitor:

```bash
docker logs -f affgrasp-mmseg-all
docker ps -a --filter name=affgrasp-mmseg-all
nvidia-smi
```

Outputs:

```text
outputs/
  all_experiments_status.tsv
  _logs/
  segformer_a/
  segformer_d/
  segformer_b/
  segformer_c/
  internimage_a/
  internimage_d/
  internimage_c/
```

Each experiment directory contains:

```text
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

The all-run script uses the planned order and runs one experiment at a time:

```text
segformer_a
segformer_d
segformer_b
segformer_c
internimage_a
internimage_d
internimage_c
```

Do not start multiple training jobs on the same GPU. The all-run script is
sequential by design. Remove the stopped container after checking logs:

```bash
docker rm affgrasp-mmseg-all
```
