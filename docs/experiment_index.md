# Aff-Grasp experiment index

This repository now keeps experiment code, run artifacts, and notes separated.
Use this file as the first entry point before starting the next run.

## Directory roles

| Path | Role | Git status |
|---|---|---|
| `experiments/affgrasp_gat/` | GAT data validation and retraining code | source code |
| `experiments/affgrasp_mmseg/` | SegFormer / InternImage training, evaluation, diagnostics | source code |
| `scripts/` | Detached Docker / server helper scripts | source code |
| `tools/` | Analysis and asset setup utilities | source code |
| `docs/` | Experiment notes, procedures, and interpretation | source code |
| `outputs/` | New run outputs and checkpoints | ignored |
| `analysis/` | Generated analysis bundles and review panels | ignored |
| `result/` | Recovered or archived result artifacts | ignored |
| `third_party/` | Downloaded external assets | ignored |
| `affordance-learning/` | Runtime copy of official assets and weights | ignored |
| `upstream-aff-grasp/` | Official Aff-Grasp source checkout | ignored |

## Completed or documented experiments

| Theme | Code entry point | Main document | Artifact location |
|---|---|---|---|
| GAT weight inference / AED analysis | `tools/run_aff_grasp_eval_maps.py`, `tools/run_umd_aff_grasp_maps.py`, `tools/run_affgrasp_diagnostics.py` | `docs/lab_pc_qualitative_inference.md`, `docs/aed_weakness_analysis.md` | `analysis/`, `result/` |
| SegFormer / InternImage 7 experiments | `experiments/affgrasp_mmseg/run_all_experiments.sh` | `docs/segformer_internimage_q_run.md`, `docs/affgrasp_experiments_summary.md` | `outputs/`, recovered under `result/affgrasp_full_results/` |
| GAT retraining preparation | `experiments/affgrasp_gat/validate_data.py`, `experiments/affgrasp_gat/train_gat.py` | `docs/gat_retraining.md` | `outputs/gat_retraining/<run_name>/` |

## Next GAT retraining layout

Run all new GAT training attempts under:

```text
outputs/gat_retraining/<run_name>/
```

Recommended names:

```text
overfit_16_noaug_seed0
baseline_cosine_seed0_ep15
baseline_multistep_seed0_ep15
```

Each run directory should contain `config.yaml`, `history.csv`,
`trainable_parameters.txt`, `dataset_statistics.json`, `model_checks.json`,
`environment.txt`, and `checkpoints/`. The training command rejects an existing
non-empty run directory unless `--overwrite` is passed.

## Notes

- Keep code and reproducible procedures in `experiments/`, `scripts/`, `tools/`,
  and `docs/`.
- Keep generated files in ignored directories: `outputs/`, `analysis/`, or
  `result/`.
- Put quick research notes under `docs/notes/` instead of the repository root.
- Do not move large generated artifacts into source directories.
