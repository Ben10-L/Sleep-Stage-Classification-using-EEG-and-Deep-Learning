# Start Here — Local VS Code Workflow

This is the unified source repository for all phases of the EEG sleep-stage project.

## First: keep your previous experiment files

**Do not delete** `EEG_Sleep_Stage_Complete_Research_Course` yet. It contains the local Sleep-EDF files and your earlier measured ML results.

From this new repository, copy them non-destructively:

```bash
python3 scripts/migrate_existing_project.py \
  --from-dir ../EEG_Sleep_Stage_Complete_Research_Course

python3 scripts/migrate_existing_project.py \
  --from-dir ../EEG_Sleep_Stage_Complete_Research_Course --yes
```

See [`docs/MIGRATION.md`](docs/MIGRATION.md) for details and path variations.

## Install once

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
python -m pytest -q
```

Use the same `.venv/bin/python` interpreter in VS Code and Jupyter notebooks.

## Verify your existing real data

```bash
python scripts/run_pipeline.py inspect \
  --data data/processed/sleep_edf_trimmed.npz
```

This audit must confirm your 9 subjects, five stages, finite samples, and subject-disjoint splits.

## Continue where the Mac MPS error interrupted training

```bash
python scripts/run_pipeline.py deep \
  --data data/processed/sleep_edf_trimmed.npz \
  --model cnn --epochs 20 --batch-size 16 \
  --device auto --output results/real/cnn
```

This revision includes the fixed-bin pooling replacement. **If MPS is still unsupported on your installed PyTorch build**, run the identical command with `--device cpu`. The CPU smoke test is verified; the Apple GPU must be checked on your own machine.

## Next experiments

1. Train `lstm` and `transformer` using the CLI instructions in README.
2. Rerun classical ML for a consistent comparison because the SVM calibration was updated.
3. Generate ensemble prediction files, saliency, confusion matrices, and the report from **measured** outputs.
4. Only then decide what results are ready to include in a public GitHub README.

**Important:** Existing trained checkpoints and previous metrics are not part of this ZIP. The migration script copies local files without deleting or overwriting them. The `.ipynb` files are for learning; the `.py` files drive repeatable training.
