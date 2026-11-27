# Release Notes — Unified Local Pipeline 1.1.0

## Software changes

- Integrated the earlier MPS-compatible CNN / BiLSTM / Transformer pooling fix.
- Removed deprecated `SVC(probability=True)`; class probabilities now use `CalibratedClassifierCV(SVC(...), ensemble=False)` inside the training partition.
- Introduced incremental, resumable PhysioNet fetch manifest and bounded retry.
- Integrated wake trimming as `run_pipeline.py trim`, retaining the standalone compatibility command.
- Added dataset audit CLI, explicit dataset hashes, per-epoch deep learning progress logging, and subject partition metadata.
- Made checkpoint metadata compatible with `torch.load(..., weights_only=True)`.
- Added figures from existing measured metrics, safe non-destructive migration, VS Code settings/tasks and regression tests.
- Updated README and code-reference documentation to this exact source revision.

## Validation scope

- Unit and integration tests run on CPU; GPU/MPS validation must occur on Apple hardware.
- Offline synthetic end-to-end CLI run checked model training, safe loading, ensembles, saliency and report generation.
- The release has **not** downloaded the user's private local files or retrained their real EEG models in this environment.

## Model comparison warning

The source changes SVM calibration and temporal pooling, so old measurements should not be presented as results produced by this release. Rerun models with a fixed dataset and split, then document the exact software revision and dataset checksum.
