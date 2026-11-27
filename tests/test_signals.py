"""Unit tests for EEG sampling and synthetic signal behavior."""

import numpy as np
import pytest

from eeg_sleep.signals import (
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


def test_30_second_epoch_at_100_hz_contains_3000_samples():
    assert sample_count(100, 30) == 3000


def test_sampling_axis_excludes_right_endpoint():
    times = time_vector(100, 30)
    assert times.shape == (3000,)
    assert times[0] == 0
    assert times[-1] == pytest.approx(29.99)


def test_nyquist_frequency():
    assert nyquist_frequency(100) == 50


def test_sinusoid_at_quarter_period_reaches_amplitude():
    t = np.array([0.0, 0.025])  # Quarter-cycle at 10 Hz = 0.025 s.
    y = sinusoid(t, frequency_hz=10, amplitude_uv=40)
    np.testing.assert_allclose(y, [0.0, 40.0], atol=1e-12)


def test_deterministic_two_channel_epoch():
    t, first = simulate_two_channel_eeg(seed=7)
    _, second = simulate_two_channel_eeg(seed=7)
    assert t.shape == (3000,)
    assert first.shape == (len(EEG_CHANNELS), 3000)
    np.testing.assert_array_equal(first, second)


def test_sleep_stage_signals_are_finite_and_same_shape():
    for label in ("W", "N1", "N2", "N3", "REM"):
        t, signal = simulate_sleep_stage(label)
        assert t.shape == (300,)
        assert signal.shape == (300,)
        assert np.all(np.isfinite(signal))


def test_frequency_bands_demo_below_nyquist():
    assert all(0 < band.example_hz < nyquist_frequency(100) for band in FREQUENCY_BANDS)


def test_action_potential_is_a_stylized_mv_curve():
    t, membrane_mv = simulate_action_potential()
    assert t.shape == membrane_mv.shape
    assert membrane_mv.min() < -70
    assert membrane_mv.max() >= 30


@pytest.mark.parametrize("rate,duration", [(0, 30), (-100, 30), (100, 0), (100, -1), (100, 0.333)])
def test_invalid_sampling_configuration_rejected(rate, duration):
    with pytest.raises(ValueError):
        sample_count(rate, duration)


def test_invalid_stage_label_rejected():
    with pytest.raises(ValueError):
        simulate_sleep_stage("N4")
