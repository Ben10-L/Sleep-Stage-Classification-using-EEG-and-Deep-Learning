"""Phase 3: leakage-resistant train/validation/test subject partitions."""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def subject_split(groups: np.ndarray, *, test_size: float = .2,
                  val_size: float = .2, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row indices; each subject occurs in exactly one partition.

    val_size is the proportion of remaining training subjects after test split.
    Use a predeclared seed; do not resplit to improve test results.
    """
    groups = np.asarray(groups).astype(str)
    if groups.ndim != 1 or np.unique(groups).size < 5:
        raise ValueError("At least five distinct subjects required for a 3-way split")
    ids = np.arange(len(groups))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval, test = next(splitter.split(ids, groups=groups))
    val_splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed + 1)
    train_rel, val_rel = next(val_splitter.split(trainval, groups=groups[trainval]))
    train, val = trainval[train_rel], trainval[val_rel]
    assert set(groups[train]).isdisjoint(groups[val])
    assert set(groups[train]).isdisjoint(groups[test])
    assert set(groups[val]).isdisjoint(groups[test])
    return train, val, test
