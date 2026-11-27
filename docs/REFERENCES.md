# Scientific reading list and source credits

1. PhysioNet. [Sleep-EDF Expanded, Version 1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/). Dataset, channel definitions, annotations and access instructions. Cite the original study and PhysioNet according to the official page.
2. MNE-Python. [Sleep stage classification tutorial](https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html). EDF annotation loading and basic sleep-stage workflows.
3. MNE-Python. [sleep_physionet.age.fetch_data](https://mne.tools/stable/generated/mne.datasets.sleep_physionet.age.fetch_data.html). Sleep-cassette subset retrieval API.
4. Supratak, A., Dong, H., Wu, C. & Guo, Y. (2017). [DeepSleepNet: A Model for Automatic Sleep Stage Scoring Based on Raw Single-Channel EEG](https://arxiv.org/abs/1703.04046). CNN/RNN motivation.
5. Scikit-learn. [GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html). Participant-independent evaluation splits.
6. MNE-Python. [MNE documentation](https://mne.tools/stable/index.html). Filter/sampling conventions and EEG practice.
7. PyTorch. [Conv1d](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html). Input `(batch, channels, time)` conventions.

## How to read a paper

Record bibliographic metadata, research question, original dataset, participant split strategy, annotation mapping, baselines, primary metric, uncertainty analysis, failure cases and limits. Compare implementation details with original supplementary material. Do not claim a research gap merely because a single paper did not include a model.
