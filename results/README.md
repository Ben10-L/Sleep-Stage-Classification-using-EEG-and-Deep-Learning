# Locally generated experiment artifacts

Runs write model checkpoints, prediction CSV, measured metrics JSON, confusion plots, learning curves and Markdown reports under `results/`.

These measured outputs must come from actual executions; they are **not** bundled in this source release. `results/*` is ignored by Git to prevent accidentally publishing personal recordings, model weights, or misleading simulated measurements.

To publish verified non-sensitive metrics intentionally, copy reviewed tables and selected figures to `docs/` or another tracked directory, record cohort and experiment settings, and ensure that all results use the same dataset and subject split.
