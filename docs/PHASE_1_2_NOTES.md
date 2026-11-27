# Phase 1.2 — EEG Signals: Research Notebook

These are my study notes for understanding the biological, physical, and
mathematical foundations of EEG before building sleep-stage classifiers.

## Note 06 — Biological origin of EEG

EEG signals originate mainly from the summed postsynaptic activity of large
populations of cortical neurons, especially appropriately oriented pyramidal
neurons. Synchronized neuronal activity generates electrical fields that can
be detected at the scalp. EEG is **not** a direct recording of single-neuron
action potentials.

## Note 07 — Electroencephalography

Electroencephalography is a non-invasive method for recording electrical
potential differences related to brain activity using scalp electrodes.
Scalp EEG amplitude is commonly expressed in microvolts (µV), with
`1 µV = 10⁻⁶ V`. EEG offers millisecond-scale temporal resolution.

## Note 08 — EEG electrodes and channels

The International 10–20 System standardizes electrode positions. A physical
electrode is distinct from a channel: a channel is a recorded potential
difference. In the Sleep-EDF Expanded recordings, two EEG derivations are
`Fpz–Cz` and `Pz–Oz`, both sampled at 100 Hz.

`V_Fpz-Cz(t) = V_Fpz(t) − V_Cz(t)` and
`V_Pz-Oz(t) = V_Pz(t) − V_Oz(t)`.

## Note 09 — Oscillations and frequency bands

Frequency describes cycles per second in hertz; amplitude describes the
magnitude of an oscillation. For a sinusoidal component, the period is
`T = 1/f`.

| Band | Common approximate range |
|---|---|
| Delta | 0.5–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–13 Hz |
| Beta | 13–30 Hz |
| Gamma | >30 Hz |

These ranges are conventions, not strict physiological boundaries. A band alone
cannot define a sleep stage. EEG stage scoring also involves morphology and,
in clinical polysomnography, eye and muscle activity.

## Note 10 — Continuous-time EEG model

An idealized oscillation can be modeled as `x(t) = A sin(2πft + φ)`, where `A`
is amplitude, `f` frequency, and `φ` phase. A schematic EEG signal may be
approximated by the superposition of oscillations plus variable noise:

`x(t) ≈ Σₖ Aₖ sin(2πfₖt + φₖ) + ε(t)`.

Real EEG is non-stationary and may contain non-oscillatory transients, artifacts,
and complex mixtures of sources.

## Note 11 — Discrete-time EEG and sampling

For sampling frequency `f_s`, the interval between samples is `T_s = 1/f_s`,
and `x[n] = x(n/f_s)`. At 100 Hz, the sampling interval is 10 ms. The Nyquist
frequency is `f_s / 2 = 50 Hz`. Frequencies above this limit cannot be uniquely
reconstructed from such samples; anti-aliasing filtering is necessary during
acquisition.

## Note 12 — EEG epoch tensors

For 30-second epochs at 100 Hz: `N = f_s × 30 = 3000 samples / channel`.
With two EEG derivations, one epoch has shape `(2, 3000)`. A batch of 32
channels-first epochs has shape `(32, 2, 3000)`; a five-class classifier would
output five logits for each epoch.

**Important:** Epochs must eventually align exactly with the dataset's expert
annotations. Subject-wise separation must happen before any evaluation
procedure that can learn subject-related patterns.

## Sources

- PhysioNet: https://physionet.org/content/sleep-edfx/1.0.0/
- MNE reference tutorial: https://mne.tools/stable/auto_tutorials/preprocessing/55_setting_eeg_reference.html
- EEG biological basis: https://www.ncbi.nlm.nih.gov/books/NBK390351/
