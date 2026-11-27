# Phase 1.2 — Exercises (do not skip)

Complete the notebook and answer these questions *in your own words* before
marking Phase 1.2 as completed.

## Scientific reasoning

1. What is the main source of scalp EEG, and why is it different from an action
   potential recorded from a single neuron?
2. What distinguishes an electrode from an EEG channel?
3. In a 10 Hz oscillation, how many cycles occur in one second? What is its
   period in milliseconds?
4. What is the Nyquist frequency when EEG is sampled at 100 Hz? Why is a 70 Hz
   oscillation problematic at that sampling rate?
5. At 100 Hz, how many numeric values appear in a 30-second, two-channel EEG
   epoch? What is the shape of a batch containing 32 epochs?

## Coding tasks

- [ ] Run the notebook from top to bottom without errors.
- [ ] Plot a 2 Hz and a 10 Hz sinusoid on the same time axis. Describe the
      difference in period and frequency.
- [ ] Replace the synthetic epoch's sampling rate (100 Hz) by 128 Hz and
      observe the resulting samples-per-epoch count.
- [ ] Generate a 90-second, two-channel synthetic recording and segment it
      into three epochs of shape `(2, 3000)`.
- [ ] Explain why a synthetic wave containing a sleep spindle is not enough to
      claim it is clinically classified as stage N2.
- [ ] Run `python -m pytest -q` and confirm all tests pass locally.

## Reflection

What evidence would you need before writing the statement:
"My neural network accurately classifies real sleep stages"?

Do **not** write this claim until a real dataset, validated labels, rigorous
subject-independent evaluation, and actual measured results exist.
