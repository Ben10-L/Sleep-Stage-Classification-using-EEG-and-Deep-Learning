# Project Architecture and Design Decisions

## Separation of concerns

1. **Acquisition:** `downloading.py` retrieves and manifests verified PhysioNet recordings, persisting progress after each pair.
2. **Transformation:** `dataset.py` aligns EDF samples to hypnogram annotations, performs continuous EEG filtering and creates 30 s epochs; `trimming.py` applies explicit pilot wake trim; `audit.py` examines class balance and identity integrity.
3. **Feature engineering:** `features.py` implements per-channel time-domain, spectral and Hjorth descriptors for interpretable baselines.
4. **Model definitions:** `models.py` contains tensor-only PyTorch architectures, without dataset or filesystem logic. `MPSCompatibleAvgPool1d` pads to equal temporal bins before ordinary average pooling.
5. **Training:** `ml.py` and `deep_learning.py` handle subject-level splitting, training-set fitting/normalization, validation-only selection, held-out evaluation, and model artifacts.
6. **Evaluation:** `evaluation.py` centralizes metrics and aligned prediction artifacts. `ensemble.py` checks sample identity before combining probabilities.
7. **Interpretation & reporting:** `explain.py` computes local input gradients; `report.py` only summarizes measured JSON metrics; the plot script visualizes existing measured outcomes.
8. **Interfaces:** `scripts/run_pipeline.py` is the public CLI, while `notebooks/` are educational exploration surfaces and `tests/` guard core invariants.

## Data flow

```text
PhysioNet EDF + annotated hypnograms
               |
        verified download
               |
      incremental manifest
               |
        MNE continuous EEG
               |
        30-second epochs  ----> original NPZ
               |
        pilot wake trimming -> trimmed NPZ -> audit
               |
        subject-level split (seed 42)
               |
      +--------+---------+
      |                  |
   ML features         raw EEG
      |                  |
 LR / calibrated SVM / RF     CNN / BiLSTM / Transformer
      |                  |
      +--------+---------+
               |
    held-out predictions
               |
    ensemble / report / plots
```

## Scientific safeguards

- Split subjects, not epochs; no subject may appear in both training and evaluation.
- Normalize input with training subjects only.
- Model candidates and training epochs are selected on validation, not test, performance.
- Use macro F1 as the primary metric; also inspect rare classes and confusion matrices.
- Save dataset SHA-256 with new trained runs, and check it in reports.
- Keep entire local data and checkpoints out of Git.
- Do not equate synthetic testing with biological validation.
- Do not infer causal neuronal mechanisms from gradient saliency.

## Known methodological boundaries

- Pilot wake-trimming uses retained labeled-epoch counts, not exact original timestamps. Different records may have gaps from excluded unknown/movement segments.
- Single held-out partitions and small subject counts have high uncertainty.
- SVM calibration is fitted inside the training partition; it is not optimized using test subjects.
- Native Apple MPS behavior differs across PyTorch versions; CPU fallback remains supported.
- The `LSTM` architecture class is a CNN front-end followed by bidirectional LSTM; it is **not** an epoch-to-epoch multi-epoch sequence model.
