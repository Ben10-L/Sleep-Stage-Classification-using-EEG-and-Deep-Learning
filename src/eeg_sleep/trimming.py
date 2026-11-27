"""Wake-trimming of already-epoched Sleep-EDF recordings.

Pilot approximation: preserve 30 minutes of *retained labeled epochs* around
first and last non-W epochs, plus all intervening epochs, per recording.
This is NOT exact raw annotation-clock-time trimming if intervals were removed.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .constants import STAGES

EPOCH_KEYS = ("X", "y", "groups", "recordings", "sample_ids")


def trim_dataset(source: str | Path, destination: str | Path, wake_margin_minutes: int = 30) -> dict:
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Do not overwrite the original processed dataset")
    if wake_margin_minutes < 0:
        raise ValueError("Wake margin must be >= 0")
    with np.load(source, allow_pickle=False) as data:
        if not set(EPOCH_KEYS + ("fs",)).issubset(data.files):
            raise ValueError("Archive missing one or more required arrays")
        y = data["y"]
        recs = data["recordings"].astype(str)
        if not len(y) or not np.isin(y, range(len(STAGES))).all():
            raise ValueError("Invalid stage labels")
        if any(len(data[key]) != len(y) for key in EPOCH_KEYS):
            raise ValueError("Inconsistent epoch array lengths")
        epoch_sec = data["X"].shape[2] / float(data["fs"])
        seconds = wake_margin_minutes * 60
        margin = round(seconds / epoch_sec)
        if abs(margin * epoch_sec - seconds) > 1e-5:
            raise ValueError("Wake margin must be divisible by epoch duration")
        keep = np.zeros(len(y), dtype=bool)
        for rec in np.unique(recs):
            indices = np.flatnonzero(recs == rec)
            sleep = np.flatnonzero(y[indices] != 0)
            if len(sleep) == 0:
                raise ValueError(f"No sleep-labelled epochs: {rec}")
            first = max(0, int(sleep[0]) - margin)
            last = min(len(indices), int(sleep[-1]) + margin + 1)
            keep[indices[first:last]] = True
            print(f"[trim] {rec}: {last-first}/{len(indices)} epochs", flush=True)
        result = {key: data[key][keep] if key in EPOCH_KEYS else data[key] for key in data.files}
        result["wake_margin_minutes"] = np.array(wake_margin_minutes)
        result["preprocessing_notes"] = np.array(
            "Pilot wake trim by retained epoch counts, not original EDF timestamps; "
            "preserves intervening wake epochs."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **result)
    counts = np.bincount(result["y"], minlength=len(STAGES))
    summary = {"source": str(source), "output": str(destination),
               "original_epochs": int(len(y)), "trimmed_epochs": int(keep.sum()),
               "subjects": len(np.unique(result["groups"])),
               "class_counts": dict(zip(STAGES, map(int, counts)))}
    return summary
