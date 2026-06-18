# Experiments

Experiment families are split by model lineage:

| Directory | Purpose |
|---|---|
| `affgrasp_gat/` | Official Aff-Grasp GAT validation and retraining wrappers |
| `affgrasp_mmseg/` | SegFormer and InternImage replacement-model experiments |

## GAT retraining

Start with data validation:

```bash
python -m experiments.affgrasp_gat.validate_data \
  --data-root affordance-learning/ag_dataset \
  --output-dir outputs/gat_data_validation
```

Then run a small overfit check before full training:

```bash
python -m experiments.affgrasp_gat.train_gat \
  --output-root outputs/gat_retraining \
  --run-name overfit_16_noaug_seed0 \
  --max-train-samples 16 \
  --disable-augmentation \
  --skip-eval \
  --epochs 5 \
  --batch-size 8 \
  --gpu 0
```

Full runs should use unique `--run-name` values under
`outputs/gat_retraining/`. See `docs/gat_retraining.md`.

## SegFormer / InternImage

The seven replacement-model experiments are configured under:

```text
experiments/affgrasp_mmseg/configs/
```

Use `experiments/affgrasp_mmseg/run_all_experiments.sh` directly inside the
runtime environment, or `scripts/run_all_mmseg_experiments_detached.sh` for a
detached Docker run. See `docs/segformer_internimage_q_run.md`.
