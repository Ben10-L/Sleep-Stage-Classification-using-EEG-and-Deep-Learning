"""Synthetic EEG signals and basic sampling mathematics.

The functions in this module demonstrate concepts from Phase 1.2. They do not
implement a physiological model, validated EEG generator, or sleep-stage scorer.

Waveform amplitudes are expressed in microvolts (µV); time is in seconds unless
otherwise indicated. Simulations are deterministic when given the same seed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

EEG_CHANNELS: tuple[str, str] = ("Fpz-Cz", "Pz-Oz")


@dataclass(frozen=True)
class FrequencyBand:
    """An illustrative EEG frequency band; boundaries vary across references."""

    name: str
    lower_hz: float
    upper_hz: float
    example_hz: float


FREQUENCY_BANDS: tuple[FrequencyBand, ...] = (
    FrequencyBand("Delta", 0.5, 4.0, 2.0),
    FrequencyBand("Theta", 4.0, 8.0, 6.0),
    FrequencyBand("Alpha", 8.0, 13.0, 10.0),
    FrequencyBand("Beta", 13.0, 30.0, 20.0),
    FrequencyBand("Gamma", 30.0, 45.0, 38.0),
)


def _positive_finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive; got {value!r}")
    return number


def sample_count(sampling_rate_hz: float, duration_seconds: float) -> int:
    """Return the exact integer number of equally spaced samples in an epoch."""
    fs = _positive_finite(sampling_rate_hz, "sampling_rate_hz")
    duration = _positive_finite(duration_seconds, "duration_seconds")
    count = fs * duration
    nearest = round(count)
    if nearest == 0 or not math.isclose(count, nearest, abs_tol=1e-9, rel_tol=1e-12):
        raise ValueError("sampling_rate_hz * duration_seconds must be an integer >= 1")
    return nearest


def time_vector(sampling_rate_hz: float, duration_seconds: float) -> NDArray[np.float64]:
    """Return sample times 0, 1/fs, ..., (N-1)/fs (the endpoint is excluded)."""
    n_samples = sample_count(sampling_rate_hz, duration_seconds)
    return np.arange(n_samples, dtype=np.float64) / sampling_rate_hz


def nyquist_frequency(sampling_rate_hz: float) -> float:
    """Return half the sampling rate, in hertz."""
    return _positive_finite(sampling_rate_hz, "sampling_rate_hz") / 2.0


def sinusoid(
    times_seconds: NDArray[np.float64],
    *,
    frequency_hz: float,
    amplitude_uv: float = 1.0,
    phase_radians: float = 0.0,
) -> NDArray[np.float64]:
    """Evaluate A * sin(2*pi*f*t + phase) in microvolts.

    This function does not prevent frequencies above Nyquist: such frequencies
    are useful when explicitly demonstrating aliasing.
    """
    t = np.asarray(times_seconds, dtype=np.float64)
    if t.ndim != 1 or not np.all(np.isfinite(t)):
        raise ValueError("times_seconds must be a finite 1D array")
    if not math.isfinite(frequency_hz) or frequency_hz < 0:
        raise ValueError("frequency_hz must be finite and non-negative")
    if not math.isfinite(amplitude_uv) or not math.isfinite(phase_radians):
        raise ValueError("amplitude and phase must be finite")
    return amplitude_uv * np.sin(2.0 * np.pi * frequency_hz * t + phase_radians)


def simulate_action_potential(n_points: int = 700) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a *stylized* single-neuron action potential (not an EEG trace).

    Piecewise interpolation is used purely to visualize depolarization,
    repolarization, and after-hyperpolarization. This is not a conductance model.
    """
    if n_points < 30:
        raise ValueError("n_points must be at least 30")
    time_ms = np.linspace(0.0, 7.0, n_points)
    knots_ms = np.array([0.0, 1.0, 1.5, 1.75, 2.05, 3.0, 4.2, 5.0, 7.0])
    voltage_mv = np.array([-70.0, -70.0, -55.0, 30.0, 35.0, -70.0, -82.0, -70.0, -70.0])
    return time_ms, np.interp(time_ms, knots_ms, voltage_mv)


def _gaussian(times: NDArray[np.float64], center: float, width: float) -> NDArray[np.float64]:
    return np.exp(-0.5 * ((times - center) / width) ** 2)


def simulate_sleep_stage(
    stage: str,
    *,
    sampling_rate_hz: float = 100.0,
    duration_seconds: float = 3.0,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create a pedagogical 1-channel waveform *resembling* one sleep stage.

    These handcrafted examples are for visual comparison only. Their EEG
    morphology is not clinically validated, and they must not be used as a
    synthetic training set or as evidence of classification performance.
    """
    normalized_stage = stage.upper()
    if normalized_stage not in {"W", "N1", "N2", "N3", "REM"}:
        raise ValueError(f"Unknown sleep-stage label: {stage!r}")
    t = time_vector(sampling_rate_hz, duration_seconds)
    if sampling_rate_hz <= 90.0:
        # This check is conservative: the demonstration uses components <= 32 Hz.
        if sampling_rate_hz <= 64.0:
            raise ValueError("This illustrative simulation requires fs > 64 Hz")
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=2.5, size=t.size)
    if normalized_stage == "W":
        signal = sinusoid(t, frequency_hz=10.0, amplitude_uv=12.0) + sinusoid(
            t, frequency_hz=20.0, amplitude_uv=3.0
        )
    elif normalized_stage == "N1":
        signal = sinusoid(t, frequency_hz=6.0, amplitude_uv=11.0) + sinusoid(
            t, frequency_hz=3.0, amplitude_uv=3.0
        )
    elif normalized_stage == "N2":
        background = sinusoid(t, frequency_hz=6.0, amplitude_uv=5.0)
        spindle = 17.0 * _gaussian(t, center=0.85, width=0.17) * np.sin(2 * np.pi * 13.0 * t)
        k_complex = -28.0 * _gaussian(t, center=2.0, width=0.095) + 15.0 * _gaussian(
            t, center=2.22, width=0.14
        )
        signal = background + spindle + k_complex
    elif normalized_stage == "N3":
        signal = sinusoid(t, frequency_hz=1.5, amplitude_uv=35.0) + sinusoid(
            t, frequency_hz=3.0, amplitude_uv=10.0
        )
    else:
        signal = (
            sinusoid(t, frequency_hz=6.0, amplitude_uv=5.0)
            + sinusoid(t, frequency_hz=11.0, amplitude_uv=3.5)
            + sinusoid(t, frequency_hz=18.0, amplitude_uv=2.0)
        )
    return t, np.asarray(signal + noise, dtype=np.float64)


def simulate_two_channel_eeg(
    *,
    sampling_rate_hz: float = 100.0,
    duration_seconds: float = 30.0,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return time and a synthetic two-channel EEG array (2, n_samples).

    The names Fpz-Cz and Pz-Oz are used to illustrate the expected structure of
    Sleep-EDF input. The waveforms are not actual measurements from those sites.
    """
    t = time_vector(sampling_rate_hz, duration_seconds)
    if sampling_rate_hz <= 64:
        raise ValueError("The demonstration requires a sample rate above 64 Hz")
    rng = np.random.default_rng(seed)
    first = (
        sinusoid(t, frequency_hz=2.0, amplitude_uv=13.0)
        + sinusoid(t, frequency_hz=6.0, amplitude_uv=7.0)
        + sinusoid(t, frequency_hz=10.0, amplitude_uv=4.0)
        + rng.normal(0.0, 3.5, size=t.size)
    )
    second = (
        sinusoid(t, frequency_hz=1.5, amplitude_uv=17.0, phase_radians=0.35)
        + sinusoid(t, frequency_hz=9.0, amplitude_uv=8.0)
        + sinusoid(t, frequency_hz=20.0, amplitude_uv=3.0)
        + rng.normal(0.0, 3.5, size=t.size)
    )
    spindle = 12.0 * _gaussian(t, center=8.0, width=0.3) * np.sin(2 * np.pi * 13.0 * t)
    first += spindle
    second += 0.6 * spindle
    return t, np.vstack((first, second)).astype(np.float64)
