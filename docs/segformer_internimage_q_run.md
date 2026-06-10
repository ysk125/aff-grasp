# SegFormer / InternImage q Server Runbook

## Current Implementation Status

The previous `experiments/` tree only contained `__pycache__` files, so it was
not runnable or tracked by Git. The experiment sources have been restored under
`experiments/affgrasp_mmseg/`.

Implemented:

- SegFormer A/B/C/D experiment configs using MiT-B5 with the SegFormer MLP decode head
- InternImage A/C/D experiment configs using the official InternImage-S backbone with a UPerNet decode head
- InternImage is optional and is not run by default until its DCNv3 build passes
- SegFormer-B5 is initialized from the NVIDIA ADE20K checkpoint
- InternImage-S and its UPerNet head are initialized from the official ADE20K checkpoint
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
uses the official OpenGVLab implementation and its DCNv3 CUDA extension.

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

InternImage is not installed by the default Docker build because its DCNv3 CUDA
extension must be compiled for the server GPU architecture. Build
an InternImage-enabled image on the q host:

```bash
AFFGRASP_WITH_INTERNIMAGE=1 bash scripts/build_docker.sh
bash scripts/run_docker.sh 1
```

Then validate the backend inside the Docker container:

```bash
python experiments/affgrasp_mmseg/preflight.py --check-timm-models --check-internimage
```

`scripts/install_internimage_deps.sh` remains available for interactive DCNv3
debugging. Use the Docker build option for smoke tests and detached full runs.

Expected InternImage fields:

```text
internimage_backend: official
internimage_model_class: OfficialInternImageSegmentationModel
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
The ADE20K classifiers are not reused because ADE20K has 150 classes; only the
final classifier layers are newly initialized for the nine Aff-Grasp labels.

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
