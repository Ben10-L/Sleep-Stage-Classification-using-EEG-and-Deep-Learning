"""Educational EEG signal tools for the sleep-staging research project.

All generated signals are synthetic and must not be mistaken for clinical EEG.
"""

from .signals import (
    EEG_CHANNELS,
    FREQUENCY_BANDS,
    nyquist_frequency,
    sample_count,
    simulate_action_potential,
    simulate_sleep_stage,
    simulate_two_channel_eeg,
    sinusoid,
    time_vector,
)
from .epoching import segment_epochs

__all__ = [
    "EEG_CHANNELS",
    "FREQUENCY_BANDS",
    "nyquist_frequency",
    "sample_count",
    "simulate_action_potential",
    "simulate_sleep_stage",
    "simulate_two_channel_eeg",
    "sinusoid",
    "time_vector",
    "segment_epochs",
]
