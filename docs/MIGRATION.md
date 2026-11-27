# Migrate Existing Local EEG Data and Results

This project replaces the old source tree while **preserving the local data** you have already downloaded and preprocessed. Keep the original folder as a backup until you have validated the new project.

## Old local work that should be preserved

Your previous run was reported as using nine Sleep-EDF Sleep Cassette recordings and produced:

- `data/raw/mne_cache/physionet-sleep-data/` — original EDF pairs;
- `data/raw/manifest.json` — their locations;
- `data/processed/sleep_edf.npz` — original epoch archive;
- `data/processed/sleep_edf_trimmed.npz` — pilot wake-trimmed archive;
- `results/real/ml/` — locally measured baseline, if it still exists.

The source release does **not** bundle these files; they reside on your Mac.

## Safely copy the local files

Put the old and new project folders in the same parent directory, then **from the new project root**:

```bash
python scripts/migrate_existing_project.py --from-dir ../EEG_Sleep_Stage_Complete_Research_Course
```

Review the dry-run file count. Then:

```bash
python scripts/migrate_existing_project.py --from-dir ../EEG_Sleep_Stage_Complete_Research_Course --yes
```

Copies only `data/raw/`, `data/processed/`, `results/`; leaves the old folder unchanged. If a destination file exists with a different size, it refuses to overwrite it. It updates absolute paths inside the copied manifest to point to the new folder where possible. **Do not copy `.venv`, `src`, `.git`, or old Python scripts.**

## Recreate virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

## Confirm the local real dataset remains usable

```bash
python scripts/run_pipeline.py inspect --data data/processed/sleep_edf_trimmed.npz
```

Expected for the previous nine-subject pilot: 8,781 epochs; shape `(8781, 2, 3000)`; nine subjects; finite samples; subject-disjoint partitions. These values come from the previous terminal outputs, **not from data contained in this source ZIP**.

The new source revision changes SVM calibration and MPS pooling. Previous metric files remain historical records, but **do not claim they were reproduced by the new revision**. For a consistent comparison, rerun classical ML in a new output directory, then train neural networks with the new source. Use different output folders for different cohorts and trim policies to prevent accidental mixing.

To fetch an additional subject in the future, the downloader now merges rather than overwrites existing manifest entries:

```bash
python scripts/run_pipeline.py fetch --subjects 9 --recordings 1
```

If you later rebuild the dataset with ten subjects, preserve the nine-subject NPZ and results separately before replacing them. Retraining becomes necessary because the subject split can change.
