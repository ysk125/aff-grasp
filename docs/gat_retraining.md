# GAT Retraining and Training-Data Validation

This workflow reproduces the public Aff-Grasp GAT setup while avoiding the
known training-data path mismatch in the upstream `TrainData` implementation.
The model itself is still loaded from the official Aff-Grasp source, so this is
not a substitute architecture.

## 1. Prepare official source and assets

```bash
bash scripts/setup_gat_runtime.sh
python tools/download_aff_grasp_assets.py
```

If `affordance-learning/` already contains downloaded assets, keep it as the
runtime asset directory and clone the official source into
`upstream-aff-grasp/affordance-learning`. `scripts/setup_gat_runtime.sh` handles
that layout by linking the official `data/`, `models/`, `utils/`, `train.py`,
and `test.py` into the asset directory without replacing downloaded weights or
datasets.

The retraining command expects:

```text
upstream-aff-grasp/affordance-learning/models/GAT.py
affordance-learning/dinov2_vitb14_pretrain.pth
affordance-learning/ag_dataset/ego_train/
affordance-learning/ag_dataset/Affordance_Evaluation_Dataset/
affordance-learning/ag_dataset/depth/ or affordance-learning/depth/
```

## 2. Validate training data

```bash
python -m experiments.affgrasp_gat.validate_data \
  --data-root affordance-learning/ag_dataset \
  --output-dir outputs/gat_data_validation
```

This writes:

```text
outputs/gat_data_validation/
├── dataset_statistics.json
├── missing_files.csv
└── sample_visualizations/
```

Check that:

- `train_samples` is non-zero and roughly comparable to the public run
  expectation of about 329-336 samples.
- `missing_or_invalid_pairs` is 0.
- label values are basically `0`, `128`, and `255`.
- all eight affordance classes have non-zero image and pixel counts.
- sampled panels show RGB, depth, and masks aligned.

## 3. Small overfit test

```bash
python -m experiments.affgrasp_gat.train_gat \
  --source-root upstream-aff-grasp/affordance-learning \
  --runtime-root affordance-learning \
  --data-root affordance-learning/ag_dataset \
  --output-root outputs/gat_retraining \
  --run-name overfit_16_noaug_seed0 \
  --max-train-samples 16 \
  --disable-augmentation \
  --skip-eval \
  --epochs 5 \
  --batch-size 8 \
  --gpu 0
```

Use this before full retraining. Loss should clearly decrease on the repeated
small subset.

## 4. Full retraining

```bash
python -m experiments.affgrasp_gat.train_gat \
  --source-root upstream-aff-grasp/affordance-learning \
  --runtime-root affordance-learning \
  --data-root affordance-learning/ag_dataset \
  --output-root outputs/gat_retraining \
  --run-name baseline_cosine_seed0_ep15 \
  --scheduler cosine \
  --epochs 15 \
  --batch-size 8 \
  --gpu 0
```

By default, each epoch evaluates AED at threshold `0.7` and selects
`checkpoints/best.pth` by foreground mIoU. After training, it also evaluates the
best checkpoint at threshold `0.8` for comparison with the reported paper value.

Outputs include:

```text
outputs/gat_retraining/baseline_cosine_seed0_ep15/
├── config.yaml
├── history.csv
├── trainable_parameters.txt
├── dataset_statistics.json
├── missing_files.csv
├── model_checks.json
├── final_threshold_metrics.json
├── environment.txt
└── checkpoints/
    ├── best.pth
    └── last.pth
```

Use `--scheduler multistep` to compare with the current upstream GitHub
training script instead of the public-run-log-like cosine schedule.

## 5. Lab GPU server Docker run

Check the server before starting long jobs:

```bash
bash scripts/q_check_env.sh
```

Build the Docker image on the selected server:

```bash
bash scripts/build_docker.sh
```

This image is intentionally scoped to the current GAT experiment only:

- official GAT retraining
- AED inference with public or retrained GAT checkpoints
- `history.csv`, AED mIoU/F1/Mean Accuracy, class IoU, and review artifacts

It does not pre-download SegFormer weights and does not build InternImage
dependencies.

Start full GAT retraining in a detached container:

```bash
bash scripts/run_gat_retraining_detached.sh 0 affgrasp-gat-train
docker logs -f affgrasp-gat-train
```

The script sets:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0 inside the selected one-GPU container
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
OPENBLAS_NUM_THREADS=4
NUMEXPR_NUM_THREADS=4
DataLoader num_workers=4 by default
```

Override defaults when needed:

```bash
GAT_OUTPUT_DIR=outputs/gat_retraining_multistep \
GAT_SCHEDULER=multistep \
GAT_NUM_WORKERS=2 \
bash scripts/run_gat_retraining_detached.sh 1 affgrasp-gat-train-ms
```

If the official source has already been cloned and the server should not touch
the network, set:

```bash
AFFGRASP_OFFLINE=1 bash scripts/run_gat_retraining_detached.sh 0 affgrasp-gat-train
```

Run AED inference with the existing public checkpoint:

```bash
bash scripts/run_gat_aed_inference_detached.sh 0 pretrained_aff_grasp.pth affgrasp-gat-aed-public
docker logs -f affgrasp-gat-aed-public
```

Run AED inference with a retrained checkpoint:

```bash
bash scripts/run_gat_aed_inference_detached.sh \
  0 \
  outputs/gat_retraining_cosine/checkpoints/best.pth \
  affgrasp-gat-aed-retrained
```

The AED inference script reuses the existing public-weight inference path in
`tools/run_aff_grasp_eval_maps.py`, then runs the AED metric and review-bundle
analysis.

If `--output-dir` is passed, that exact directory is used for backward
compatibility. Otherwise the command writes under `--output-root` using
`--run-name`; if neither is supplied, a timestamped run directory is created.
Existing non-empty directories are rejected unless `--overwrite` is passed, so
new GAT retraining runs do not silently overwrite earlier checkpoints.
