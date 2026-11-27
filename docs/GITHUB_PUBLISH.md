# GitHub Publishing Checklist

The package is a **complete software implementation**, not a claim that every real-data experiment has already been executed.

## Verify locally first

```bash
python -m pytest -q
python scripts/run_pipeline.py --help
python scripts/run_pipeline.py inspect --data data/processed/sleep_edf_trimmed.npz
```

If the processed EEG is not yet in this folder, migrate it using [MIGRATION.md](MIGRATION.md). To make a research-results README, perform the full real-data experiments, review the observed measurements, and publish only appropriately documented results and non-sensitive figures.

## Before committing

- Verify `.gitignore` excludes `.venv/`, `data/raw/`, `data/processed/`, `results/`, checkpoint files, and any credentials.
- Never commit downloaded EDF recordings or large locally processed NPZ arrays.
- Do not claim synthetic demo metrics are clinical or subject-independent benchmark evidence.
- If reporting real results, specify cohort, subject split, trim policy, seed, model revision, provenance and uncertainty.
- Read `README.md` and `docs/RESEARCH_INTEGRITY.md`.

## Publish

Create an empty repository named `eeg-sleep-stage-classification` under your GitHub account (do not initialize a separate README in GitHub).

```bash
git init
git branch -M main
git add .
git status --short
git commit -m "feat: release modular EEG sleep-stage research pipeline"
git remote add origin https://github.com/Ben10-L/eeg-sleep-stage-classification.git
git push -u origin main
```

If a remote, `main` history, or README already exists, **inspect it** using `git remote -v`, `git branch -vv` and `git log --oneline -5` before attempting to merge. Never force-push to resolve an unexpected history conflict.

## After publication

Check that README links resolve, GitHub Actions CI passes, and neither the repository nor its history contains raw EEG data or secrets. Include only reproducible, verified claims in your portfolio or internship applications.
