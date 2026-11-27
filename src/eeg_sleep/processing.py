"""Phase 1.3: reproducible signal processing. Values are in microvolts.

Filtering each epoch independently is convenient for demonstrations but creates
boundary transients. Real preparation should prefer filtering continuous records
before epoching when memory and experimental design allow it.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, spectrogram


def bandpass(data: np.ndarray, fs: float = 100.0, low: float = 0.5,
             high: float = 40.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth filtering on the final (time) dimension."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 0 or arr.shape[-1] < 40:
        raise ValueError("Input must contain at least 40 time samples")
    if not (0 < low < high < fs / 2):
        raise ValueError("Require 0 < low < high < fs/2")
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, arr, axis=-1).astype(np.float32)


def power_spectrum(data: np.ndarray, fs: float = 100.0,
                   nperseg: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD; output units are input-unit²/Hz."""
    signal = np.asarray(data, dtype=np.float64)
    if signal.ndim == 0 or signal.shape[-1] < 2:
        raise ValueError("At least two samples required")
    return welch(signal, fs=fs, nperseg=min(nperseg, signal.shape[-1]), axis=-1)


def time_frequency(data: np.ndarray, fs: float = 100.0,
                   nperseg: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spectrogram of a single-channel EEG epoch (power spectral density)."""
    signal = np.asarray(data, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("Expected one EEG channel")
    seg = min(nperseg, len(signal))
    return spectrogram(signal, fs=fs, nperseg=seg, noverlap=seg // 2, mode="psd")
