"""Read-only audit of NPZ data with simple subject split diagnostics."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import numpy as np
from .dataset import load_arrays, read_provenance
from .constants import STAGES
from .splits import subject_split


def inspect_dataset(path: str | Path, seed: int = 42) -> dict:
    X, y, groups, ids, fs = load_arrays(path)
    counts = np.bincount(y, minlength=len(STAGES))
    result = {
        "file": str(path), "provenance": read_provenance(path),
        "n_epochs": len(y), "shape": list(X.shape), "sampling_rate_hz": fs,
        "class_counts": dict(zip(STAGES, map(int, counts))),
        "class_percent": dict(zip(STAGES, [round(100*int(c)/len(y), 2) for c in counts])),
        "n_subjects": len(np.unique(groups)), "subject_ids": sorted(set(groups)),
        "all_finite": bool(np.isfinite(X).all()),
        "unique_sample_ids": len(set(ids)) == len(ids),
    }
    if len(np.unique(groups)) >= 5:
        tr, va, te = subject_split(groups, seed=seed)
        result["splits"] = {
            name: {"n_epochs": len(indices), "subjects": sorted(set(groups[indices])),
                   "class_counts": dict(zip(STAGES, map(int, np.bincount(y[indices], minlength=5))))}
            for name, indices in (("train",tr),("validation",va),("test",te))
        }
    return result
