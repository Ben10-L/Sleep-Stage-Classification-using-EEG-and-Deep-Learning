# Add Phase 1.2 to my GitHub repository

These instructions assume my GitHub repository already exists. The ZIP file
contains project files but intentionally does **not** include a `.git` folder.

## 1. Open my existing repository

```bash
# If the repository is already cloned, just enter its directory:
cd /path/to/my/existing-eeg-repository

git status
```

If it has not been cloned yet, I should clone the exact URL from the green
**Code** button on my own GitHub repository, rather than guessing its name.

## 2. Copy the Phase 1.2 files

After extracting the ZIP file, copy its **contents** into the repository root.
On macOS, for example:

```bash
# Replace this path if the ZIP was extracted somewhere else.
rsync -av ~/Downloads/EEG_Sleep_Stage_Phase_1_2/ ./
```

If README.md, LICENSE or .gitignore already contain useful existing content,
review the differences before overwriting or merging them. Never copy a
`.git` folder into another repository.

## 3. Set up the environment and verify reproducibility

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/generate_lesson_figures.py
jupyter lab notebooks/01_02_eeg_signal_foundations.ipynb
```

## 4. Commit only after reviewing the changes

```bash
git status
git diff -- README.md

git add .github .gitignore LICENSE README.md pyproject.toml \
  src tests notebooks scripts docs data/README.md figures/.gitkeep

git commit -m "feat(eeg): add reproducible Phase 1.2 foundations"
git push
```

If the repository has no upstream branch yet, `git push -u origin main` may be
appropriate when the current branch is actually named `main`. Confirm the
branch with `git branch --show-current` before running it.

## 5. Commit policy

- Commit learning material only after understanding it and successfully
  reproducing its execution on my own computer.
- Keep code and scientific claims accurate.
- Never commit patient recordings, secrets, passwords, or temporary data.
- Update `docs/RESEARCH_LOG.md` with experiments that I actually performed.
- Do not mark Phase 1.2 complete until the quiz and exercises are done.
