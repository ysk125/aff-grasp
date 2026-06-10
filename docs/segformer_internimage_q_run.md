# SegFormer / InternImage q Server Runbook

## Current Implementation Status

The previous `experiments/` tree only contained `__pycache__` files, so it was
not runnable or tracked by Git. The experiment sources have been restored under
`experiments/affgrasp_mmseg/`.

Implemented:

- SegFormer A/B/C/D experiment configs using MiT-B5 with the SegFormer MLP decode head
- InternImage A/C/D experiment configs using InternImage-S with a UPerNet decode head
- InternImage A/C/D configs are retained as experimental placeholders, but are not run by default
- Shared Aff-Grasp dataset conversion for `ego_train` and AED
- Fixed train/val/test split generation under `experiments/splits`
- 9-class class-index target masks
- Focal + Dice loss
- mIoU, F1, Accuracy
- Parameter count logging
- Prediction panels
- Checkpoint and metrics outputs
- Detached Docker launcher for q

Important note: SegFormer no longer depends on `timm` model names. The q image
must include `transformers`, and `preflight.py --check-timm-models` now also
checks whether the Transformers SegFormer model can be instantiated. InternImage
uses the MMPretrain InternImage backend and requires optional `mmcv`/`mmpretrain`
dependencies before it can be enabled.

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
- `transformers_available` is `true`
- `check_model_class` is `SegFormerSegmentationModel`

It is fine if `timm_has_mit_b0` is `false`; SegFormer now uses Transformers,
not timm. If `timm_has_internimage_t_1k_224` is `false`, keep InternImage
disabled and run SegFormer first.

## Optional InternImage Setup

InternImage is not installed by the base Docker image because its `mmcv`/DCNv3
dependency stack is heavier and may need server-specific wheel resolution. After
SegFormer smoke tests pass, install and validate the optional backend inside the
Docker container:

```bash
bash scripts/install_internimage_deps.sh
python experiments/affgrasp_mmseg/preflight.py --check-timm-models --check-internimage
```

Expected InternImage fields:

```text
mmpretrain_available: true
internimage_model_class: MMPretrainInternImageSegmentationModel
```

The model families are selected to keep total capacity in the same broad range
as the DINOv2-base + GAT baseline:

```text
DINOv2-base + GAT head: about 86M plus the GAT-specific layers
SegFormer MiT-B5 + MLP head: about 85M
InternImage-S + UPerNet head: about 80M
```

Use the `check_parameters` and `internimage_parameters` fields reported by
`preflight.py` as the authoritative counts for the exact local implementation.

## Smoke Tests For All Experiments

Inside Docker:

```bash
bash experiments/affgrasp_mmseg/run_all_smoke_tests.sh 0
```

By default this runs the four production SegFormer configs sequentially with
one epoch, eight train samples, and four validation samples. It checks data
loading, model creation, freezing policy, LoRA, loss, metrics, checkpointing,
and visualization without occupying the GPU for long.

Smoke-test outputs:

```text
outputs_smoke/
  segformer_a/
  segformer_b/
  segformer_c/
  segformer_d/
```

InternImage placeholders can be included only for dependency debugging:

```bash
AFFGRASP_INCLUDE_EXPERIMENTAL_INTERNIMAGE=1 bash experiments/affgrasp_mmseg/run_all_smoke_tests.sh 0
```

Run this only after `--check-internimage` succeeds.

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

The all-run script runs one production SegFormer experiment at a time:

```text
segformer_a
segformer_d
segformer_b
segformer_c
```

`segformer_c` is the LoRA condition. LoRA is inserted only into the later
SegFormer encoder stages, stage 3 and stage 4 (`block.2` and `block.3` in the
Transformers implementation), targeting the attention `query` and `value`
linear layers.

Do not start multiple training jobs on the same GPU. The all-run script is
sequential by design. Remove the stopped container after checking logs:

```bash
docker rm affgrasp-mmseg-all
```
