"""Publication-style teaching plots for EEG foundations.

These figures visualize simulated signals, not actual Sleep-EDF recordings.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .signals import (
    EEG_CHANNELS,
    FREQUENCY_BANDS,
    simulate_action_potential,
    simulate_sleep_stage,
    simulate_two_channel_eeg,
    sinusoid,
    time_vector,
)


def _setup_axes(ax: plt.Axes, x_label: str, y_label: str) -> None:
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.22, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)


def plot_action_potential() -> Figure:
    """Stylized action potential, explicitly distinguished from scalp EEG."""
    time_ms, voltage_mv = simulate_action_potential()
    fig, ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
    ax.plot(time_ms, voltage_mv, linewidth=2.2)
    ax.axhline(-70, linestyle="--", linewidth=1, label="Resting potential (illustrative)")
    ax.axhline(-55, linestyle=":", linewidth=1, label="Threshold (illustrative)")
    ax.set_title("Single-neuron action potential — conceptual example")
    _setup_axes(ax, "Time (ms)", "Membrane potential (mV)")
    ax.legend(loc="upper right", fontsize=8)
    return fig


def plot_frequency_bands() -> Figure:
    """Contrast the periods of common EEG oscillations over one second."""
    t = time_vector(500, 1.0)
    fig, axes = plt.subplots(5, 1, figsize=(10, 8), sharex=True, layout="constrained")
    for ax, band in zip(axes, FREQUENCY_BANDS):
        y = sinusoid(t, frequency_hz=band.example_hz, amplitude_uv=20.0)
        ax.plot(t, y, linewidth=1.4)
        ax.set_ylabel(f"{band.name}\n(µV)")
        ax.set_ylim(-27, 27)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("EEG frequency bands — synthetic sinusoidal components")
    return fig


def plot_stage_signatures() -> Figure:
    """Show handcrafted, non-clinical waveform patterns for five labels."""
    stages = ("W", "N1", "N2", "N3", "REM")
    fig, axes = plt.subplots(5, 1, figsize=(11, 8), sharex=True, layout="constrained")
    for ax, stage in zip(axes, stages):
        t, y = simulate_sleep_stage(stage, seed=42)
        ax.plot(t, y, linewidth=1)
        ax.set_ylabel(f"{stage}\n(µV)")
        ax.set_ylim(-65, 65)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Illustrative stage-associated EEG patterns — NOT real patient data")
    return fig


def plot_sampling() -> Figure:
    """Illustrate continuous evaluation versus uniform discrete sampling."""
    t_continuous = np.linspace(0.0, 1.0, 2001, endpoint=False)
    t_sampled = time_vector(20, 1.0)
    y_continuous = sinusoid(t_continuous, frequency_hz=3.0, amplitude_uv=30)
    y_sampled = sinusoid(t_sampled, frequency_hz=3.0, amplitude_uv=30)
    fig, ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
    ax.plot(t_continuous, y_continuous, label="Continuous-time idealized waveform", alpha=0.65)
    ax.stem(t_sampled, y_sampled, basefmt=" ", label="20 discrete samples")
    _setup_axes(ax, "Time (s)", "Amplitude (µV)")
    ax.set_title("Sampling a 3 Hz sinusoid at 20 samples per second")
    ax.legend(fontsize=8)
    return fig


def plot_two_channel_epoch() -> Figure:
    """Plot the first five seconds from a synthetic 2-channel 30-second epoch."""
    t, data = simulate_two_channel_eeg()
    visible = t < 5.0
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, layout="constrained")
    for index, ax in enumerate(axes):
        ax.plot(t[visible], data[index, visible], linewidth=0.9)
        ax.set_title(f"Synthetic {EEG_CHANNELS[index]} derivation", fontsize=10)
        _setup_axes(ax, "Time (s)" if index == 1 else "", "Amplitude (µV)")
    fig.suptitle("Illustrative two-channel EEG epoch (30 s total, 5 s shown)")
    return fig
