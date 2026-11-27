"""Small reproducible datasets for software smoke tests and tutorials ONLY.

Do not use synthetic waveform classification performance as scientific evidence.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .signals import simulate_sleep_stage
from .constants import STAGES


def make_demo_dataset(output: str | Path, n_subjects: int = 10,
                      repeats_per_stage: int = 3, seed: int = 42) -> Path:
    """Generate all five classes for each fictitious subject (2, 3000)."""
    if n_subjects < 5 or repeats_per_stage < 1:
        raise ValueError("Need >= 5 fictitious subjects and at least one repeat")
    rng = np.random.default_rng(seed)
    X, y, groups, ids = [], [], [], []
    for subj in range(n_subjects):
        for cls, stage in enumerate(STAGES):
            for repeat in range(repeats_per_stage):
                # Only illustrative signals: class differences are intentionally
                # easy to detect; results have no clinical or research meaning.
                _, a = simulate_sleep_stage(stage, duration_seconds=30, seed=int(rng.integers(1_000_000)))
                _, b = simulate_sleep_stage(stage, duration_seconds=30, seed=int(rng.integers(1_000_000)))
                X.append(np.stack([a,b]).astype(np.float32))
                y.append(cls); groups.append(f"demo_subject_{subj:02d}")
                ids.append(f"demo_{subj:02d}_{stage}_{repeat:02d}")
    p = Path(output); p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, X=np.stack(X), y=np.asarray(y),
                        groups=np.asarray(groups), sample_ids=np.asarray(ids), fs=100.0,
                        provenance="SYNTHETIC — NOT REAL EEG")
    return p
