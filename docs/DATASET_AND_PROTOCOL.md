# Dataset, label mapping, and evaluation protocol

## Data origin

I use PhysioNet's Sleep-EDF Expanded database, specifically MNE's sleep-cassette download route. EDF recordings and hypnogram annotations are obtained in paired files. EEG Fpz–Cz and Pz–Oz are sampled at 100 Hz. These channels are voltage *differences*; they are not individual unreferenced electrodes.

## Historical sleep scoring

Sleep-EDF source annotations follow the historical Rechtschaffen & Kales (R&K) stages. To construct a five-class target, I map the labels as follows:

| Original hypnogram label | Pipeline target |
|---|---|
| Sleep stage W | W |
| Sleep stage 1 | N1 |
| Sleep stage 2 | N2 |
| Sleep stage 3 | N3 |
| Sleep stage 4 | N3 |
| Sleep stage R | REM |
| Sleep stage ? | Exclude |
| Movement time | Exclude |

Combining stage 3 and stage 4 is common for computational comparisons but **does not imply an AASM expert rescoring** of the recording. The `dataset.py` code extracts only complete, annotated epochs.

## Preprocessing

The implementation applies a zero-phase 0.5–40 Hz filter on the selected continuous EEG channels before extracting 30-second epochs. Data are returned in microvolts. Filtering influences band-edge behavior and can create challenges near large discontinuities. Artifact handling and outlier removal have not been validated in a clinical study.

## Splits and scientific integrity

All epochs from the same subject (including multiple nights) remain in one of train, validation, or test. Scaling statistics are learned only from training subjects. Model selection/early stopping occurs using validation subjects; the test subjects are held out for reporting.

The default 10-subject demo and small real-subset training are **not publication-scale protocols**. In particular, truncation with `--max-epochs-per-record` preferentially selects earlier periods and biases the stage distribution. Before any scientific conclusion, I will document cohort selection, wake trimming, class imbalance, multiple seeds, group cross-validation, sensitivity to filtering and channel selection, and confidence intervals.

## Dataset safety and citation

Raw EDF files, derived subject-level signals, and checkpoints are excluded from version control. The software MIT license does not license these third-party recordings. Please see [PhysioNet's dataset page](https://physionet.org/content/sleep-edfx/1.0.0/) for license, access and preferred citation.
