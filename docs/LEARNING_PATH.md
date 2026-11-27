# Learning Roadmap — Full Source Code Available from Day One

The repository includes all lessons and executable source code up front. Follow this sequence when studying it later:

| Step | Notebook | Practical focus | Evidence of understanding |
|---|---|---|---|
| 1.1 | `01_01_sleep_fundamentals.ipynb` | Sleep stages and physiological features | Explain W/N1/N2/N3/REM |
| 1.2 | `01_02_eeg_signal_foundations.ipynb` | Neurons, EEG, frequency and sampling | Explain `x[n]`, sampling rate, and epoch tensor |
| 1.3 | `01_03_signal_processing.ipynb` | Filtering, PSD, FFT and spectrograms | Compare time/frequency domains |
| 1.4 | `01_04_sleep_edf.ipynb` | EDF and hypnogram content | Inspect annotation-to-label mapping |
| 1.5 | `01_05_literature.ipynb` | Scientific literature | Create a literature matrix |
| 1.6 | `01_06_research_gap.ipynb` | Research problem formulation | Separate established evidence from hypotheses |
| 1.7 | `01_07_research_protocol.ipynb` | Metrics, cohort and leakage design | Define validation/test protocol in advance |
| 2 | `02_data_pipeline.ipynb` | Real-data preprocessing | Inspect NPZ, group splits, wake trim caveats |
| 3 | `03_classical_ml.ipynb` | Feature engineering and ML baselines | Explain features and model selection |
| 4 | `04_deep_learning.ipynb` | CNN, BiLSTM, Transformer | Interpret train loss and validation F1 |
| 5 | `05_ensemble_learning.ipynb` | Probability fusion | Verify sample ID alignment |
| 6 | `06_explainability.ipynb` | Input gradients | Understand limitations of saliency |
| 7 | `07_scientific_report.ipynb` | Scientific reporting | Produce a factual research report |

## How to use the code

1. Open the `.ipynb` for concepts and plots in VS Code.
2. Read the Python module listed in `README.md` that implements the same concept.
3. Execute an existing command from `scripts/run_pipeline.py`.
4. Inspect the saved metrics, predictions and figures.
5. Write your observations in `docs/RESEARCH_LOG.md` or a private lab notebook.
6. Only then consider changing architectures or designing new experiments.

The synthetic demo is for **software understanding**. It cannot be used as validation evidence for sleep-stage classification.
