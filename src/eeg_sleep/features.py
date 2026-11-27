"""Phase 3: interpretable time-domain and spectral baseline features."""
from __future__ import annotations
import numpy as np
from .processing import power_spectrum

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}


def feature_names(n_channels: int = 2) -> list[str]:
    names = []
    for channel in range(n_channels):
        prefix = f"ch{channel+1}_"
        names += [prefix + item for item in ("std", "rms", "line_length",
                                             "hjorth_activity", "hjorth_mobility",
                                             "hjorth_complexity")]
        names += [prefix + f"relative_{band}" for band in BANDS]
    return names


def extract_features(epochs: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """Return (n_epochs, n_features) without using test-set statistics.

    Input: epochs (N,C,T) in microvolts. Relative band power is normalized
    within the [0.5, min(45, fs/2)) Hz region, not per subject.
    """
    x = np.asarray(epochs, dtype=np.float64)
    if x.ndim != 3 or x.shape[-1] < 64:
        raise ValueError("Expected epochs shaped (n_epochs, n_channels, >=64 samples)")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite EEG input")
    rows = []
    for epoch in x:
        vals=[]
        for channel in epoch:
            diff = np.diff(channel)
            second = np.diff(diff)
            activity = float(np.var(channel))
            diff_var = float(np.var(diff))
            dd_var = float(np.var(second))
            mobility = np.sqrt(diff_var / max(activity, 1e-12))
            diff_mobility = np.sqrt(dd_var / max(diff_var, 1e-12))
            complexity = diff_mobility / max(mobility, 1e-12)
            vals.extend([float(np.std(channel)), float(np.sqrt(np.mean(channel**2))),
                         float(np.mean(np.abs(diff))), activity,
                         float(mobility), float(complexity)])
            freq, power = power_spectrum(channel, fs=fs, nperseg=min(512, len(channel)))
            valid = (freq >= 0.5) & (freq < min(45., fs/2))
            total = float(np.sum(power[valid])) + 1e-12
            for lo, hi in BANDS.values():
                m = (freq >= lo) & (freq < min(hi, fs/2))
                vals.append(float(np.sum(power[m]) / total))
        rows.append(vals)
    return np.asarray(rows, dtype=np.float32)
