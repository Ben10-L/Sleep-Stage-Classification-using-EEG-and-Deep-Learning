# Import into the existing GitHub repository

The **previous Phase 1.2 lesson is preserved** inside this expanded project. If your GitHub repository already exists, back up local changes before replacing any files.

```bash
cd ~/Desktop  # choose your own local workspace
unzip ~/Downloads/EEG_Sleep_Stage_Complete_Research_Course.zip
cd EEG_Sleep_Stage_Complete_Research_Course
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

To create a fresh repository after checking your account and remote URL:

```bash
git init
# Replace with the actual repository URL you created on GitHub:
git remote add origin https://github.com/Ben10-L/eeg-sleep-stage-classification.git
git branch -M main
git add .
git commit -m "feat: add full EEG sleep staging research learning pipeline"
git push -u origin main
```

**If the remote already has commits**, do not blindly `git push --force` or initialize this directory as an unrelated history. Clone the existing repository first, copy the new project files into that clone (without replacing your own notes blindly), review `git diff`, then commit and push normally.

Recommended milestone commits after each phase: `docs: complete literature matrix`, `feat: implement EEG preprocessing`, `feat: add subject-independent baselines`, `docs: analyze real test metrics`.
