# Sleep Stage Classification — Experimental Results

**Status:** Results from a locally processed dataset; independently verify experimental design.

## Reproducibility

- Dataset file: `data/processed/sleep_edf_trimmed.npz`
- Declared source: `Sleep-EDF Expanded / PhysioNet`
- Dataset SHA-256: `6d2a4e2979761a6e05156a09a97770ad71f4183df397c574019be9b178e6d4c5`
- Splits: subjects must not overlap between training, validation and testing.
- Selection: validation data only; test held out until final evaluation.

## Recorded outcomes

| Model/run | Accuracy | Macro F1 | Balanced accuracy | Cohen kappa |
|---|---:|---:|---:|---:|
| cnn_metrics | 0.826 | 0.782 | 0.784 | 0.757 |
| ensemble_metrics | 0.808 | 0.767 | 0.770 | 0.734 |
| lstm_metrics | 0.689 | 0.679 | 0.694 | 0.585 |
| ml_metrics | 0.798 | 0.750 | 0.755 | 0.719 |
| transformer_metrics | 0.759 | 0.728 | 0.728 | 0.668 |

## Scientific limitations

- Demonstration waveforms must never be presented as clinical EEG data.
- The data subset and split seed must be declared in any report.
- Evaluation must include per-class errors, subject counts and uncertainty.
- Comparing models on the test set repeatedly can bias claims of superiority.
- Gradients describe model sensitivity, not validated neurophysiological mechanisms.

## Next steps

Repeat on all eligible real subjects with a pre-registered protocol;
analyze sensitivity to channel choice, scoring rules and subgroup shift.
