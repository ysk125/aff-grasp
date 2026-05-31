# AED Weakness Analysis

This workflow exports compact class-ID masks for all 721 AED images, computes
region, boundary, and background false-positive metrics, and builds a blinded
manual-review bundle. Keep the baseline model unchanged: the DepthFeature
Injector remains enabled.

## 1. Prepare the m server

```bash
ssh m
cd ~/workspace/aff-grasp
git pull
bash scripts/build_docker.sh
```

If assets are not ready, enter Docker and bootstrap once:

```bash
bash scripts/run_docker.sh 0
bash scripts/bootstrap_affgrasp.sh
exit
```

The bootstrap validates that AED contains at least 721 RGB images, GT arrays,
and depth images. It also validates the released checkpoint and checks that the
MSDA CUDA extension can be imported.

## 2. Run AED export and automatic analysis detached

Check that GPU 0 is available. Before the first full run, use the interactive
container for a three-image smoke test:

```bash
bash scripts/run_docker.sh 0
python tools/run_aff_grasp_eval_maps.py \
  --artifact-profile analysis \
  --threshold 0.8 \
  --max-samples 3
exit
```

Then start the named container:

```bash
nvidia-smi
bash scripts/run_aed_analysis_detached.sh 0
docker logs -f affgrasp-aed-analysis
```

It is safe to disconnect SSH after the container starts. Check later with:

```bash
docker ps -a --filter name=affgrasp-aed-analysis
docker logs --tail 80 affgrasp-aed-analysis
```

The output is written to:

```text
analysis/<run_id>/
```

After checking the logs, remove the stopped container:

```bash
docker rm affgrasp-aed-analysis
```

## 3. Review all AED panels locally

Copy `analysis/<run_id>` to the local machine, then:

```bash
python -m http.server 8000 \
  --directory analysis/<run_id>
```

Open `http://localhost:8000/review/`. Scores, mIoU groups, and object names stay hidden
during review. Export `annotations.csv` after all 721 panels are reviewed.

Merge the blinded annotations:

```bash
python tools/merge_aed_review_annotations.py \
  --analysis-root analysis/<run_id> \
  --annotations /path/to/annotations.csv
```

## 4. Analyze the full training distribution

Inside Docker on m, download the full training data and run:

```bash
python tools/download_aff_grasp_assets.py
python tools/analyze_train_distribution.py \
  --analysis-root analysis/<run_id>
```

The eight-class correlations are exploratory. Treat them as evidence for
follow-up experiments, not as causal proof.

## Important outputs

```text
metrics/image_metrics.csv
metrics/region_metrics.csv
metrics/class_metrics.csv
metrics/train_distribution.csv
metrics/failure_tag_summary.csv
figures/
report.md
run_config.json
```
