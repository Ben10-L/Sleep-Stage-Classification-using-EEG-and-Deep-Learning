# Phase 1.7 — Proposed research protocol (draft)

## Aim

Compare classical ML features, lightweight CNN, CNN-LSTM, Transformer and fixed-weight probability fusion on five-class EEG sleep staging.

## Dataset definition

Sleep-EDF Expanded sleep-cassette recordings, Fpz–Cz and Pz–Oz at 100 Hz; 30-second windows. Merge original stages 3 and 4 into N3. Exclude movement and unknown. Explicitly log which subjects, nights and annotations are used.

## Leakage policy

Use participant-level disjoint train / validation / test partitions, with both nights of each individual in the same split. Fit scalers on train only. Choose architecture/epochs using validation only. Never use test to tune ensemble weights.

## Baselines and ablations

- Logistic Regression, RBF SVM, Random Forest on spectral and Hjorth features.
- CNN, CNN-LSTM, lightweight Transformer on two channels.
- Uniform-average probabilities as a fixed ensemble comparison.
- Potential ablations: each channel separately, removing classes 3/4 mapping, wake-trimming policy, data quantity, preprocessing settings. These are proposals—not all implemented.

## Endpoints

Primary: macro F1 on held-out subjects. Secondary: accuracy, balanced accuracy, Cohen's kappa, per-stage precision/recall/F1, confusion matrix, subject-cluster bootstrap CI. Report number of participants per split, class distribution and seed.

## Hypotheses (to be revised after literature review)

H1: Learned representations may improve some per-class metrics relative to handcrafted features when there is enough labeled data.

H2: N1 may remain challenging because EEG signatures overlap adjacent stages.

H3: A uniform ensemble may help under some conditions, but may also degrade accuracy; no benefit is assumed.

**Interpretation:** these are scientifically testable expectations, not confirmed results or claims of novelty.
