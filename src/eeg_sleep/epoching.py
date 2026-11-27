"""Simple epoch segmentation to introduce the shape of sleep-staging inputs."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .signals import sample_count


def segment_epochs(
    recording: ArrayLike,
    *,
    sampling_rate_hz: float,
    epoch_duration_seconds: float = 30.0,
    drop_incomplete: bool = True,
) -> NDArray[np.float64]:
    """Segment a channels-first recording into non-overlapping epochs.

    Parameters
    ----------
    recording:
        Array shaped (n_samples,) or (n_channels, n_samples).
    sampling_rate_hz:
        Samples per second, per channel.
    epoch_duration_seconds:
        Length of an epoch in seconds.
    drop_incomplete:
        If True, drop trailing samples that cannot form a complete epoch.
        If False, reject recordings with an incomplete last epoch.

    Returns
    -------
    np.ndarray
        Shape (n_epochs, n_channels, samples_per_epoch). Input values are
        copied to avoid unexpected coupling between epoch tensors and raw data.

    Notes
    -----
    This function is a numerical demonstration; it does not align PSG data to
    hypnogram annotations or apply preprocessing. That comes in later phases.
    """
    values = np.asarray(recording, dtype=np.float64)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError("recording must be a nonempty 1D or 2D channels-first array")
    if not np.all(np.isfinite(values)):
        raise ValueError("recording must contain finite numeric values")
    points_per_epoch = sample_count(sampling_rate_hz, epoch_duration_seconds)
    n_channels, n_samples = values.shape
    n_epochs, leftover = divmod(n_samples, points_per_epoch)
    if n_epochs == 0:
        raise ValueError("recording is shorter than one epoch")
    if leftover and not drop_incomplete:
        raise ValueError(f"recording ends with {leftover} incomplete samples")
    usable = values[:, : n_epochs * points_per_epoch]
    epochs = usable.reshape(n_channels, n_epochs, points_per_epoch)
    return np.transpose(epochs, (1, 0, 2)).copy()
