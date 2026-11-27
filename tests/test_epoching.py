"""Unit tests for channels-first non-overlapping epoching."""

import numpy as np
import pytest

from eeg_sleep.epoching import segment_epochs


def test_one_channel_90_seconds_produces_three_epochs():
    recording = np.arange(9000, dtype=float)
    epochs = segment_epochs(recording, sampling_rate_hz=100)
    assert epochs.shape == (3, 1, 3000)
    assert epochs[0, 0, 0] == 0
    assert epochs[2, 0, -1] == 8999


def test_two_channels_keep_correct_alignment():
    recording = np.vstack([np.arange(6000), np.arange(6000) + 100_000])
    epochs = segment_epochs(recording, sampling_rate_hz=100)
    assert epochs.shape == (2, 2, 3000)
    assert epochs[1, 0, 0] == 3000
    assert epochs[1, 1, 0] == 103000
    assert epochs[1, 1, -1] == 105999


def test_drop_incomplete_last_epoch():
    recording = np.arange(3100)
    epochs = segment_epochs(recording, sampling_rate_hz=100)
    assert epochs.shape == (1, 1, 3000)


def test_reject_incomplete_last_epoch_when_requested():
    with pytest.raises(ValueError, match="incomplete samples"):
        segment_epochs(np.arange(3100), sampling_rate_hz=100, drop_incomplete=False)


def test_reject_short_recording():
    with pytest.raises(ValueError, match="shorter than one epoch"):
        segment_epochs(np.ones(10), sampling_rate_hz=100)


def test_reject_nonfinite_recording():
    with pytest.raises(ValueError, match="finite"):
        segment_epochs(np.array([float("nan")] * 3000), sampling_rate_hz=100)


def test_epochs_do_not_mutate_original_recording():
    original = np.ones((2, 3000))
    epochs = segment_epochs(original, sampling_rate_hz=100)
    epochs[0, 0, 0] = 999
    assert original[0, 0] == 1
