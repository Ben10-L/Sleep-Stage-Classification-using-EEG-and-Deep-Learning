"""Phase 5: fixed-weight late fusion of independent model probabilities.

Weights are uniform by design; do not tune ensemble weights on the test set.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .constants import STAGES
from .evaluation import classification_metrics, write_predictions, write_metrics


def average_predictions(paths:list[str | Path], output_dir:str | Path) -> dict:
    if len(paths)<2: raise ValueError("At least two model predictions required")
    frames=[pd.read_csv(path).sort_values("sample_id").reset_index(drop=True) for path in paths]
    reference=frames[0]
    keys=["sample_id","subject_id","truth"]
    cols=[f"p_{s}" for s in STAGES]
    if reference.sample_id.duplicated().any(): raise ValueError("Duplicate sample ID")
    for f in frames:
        if f.sample_id.duplicated().any() or not f[keys].equals(reference[keys]):
            raise ValueError("Predictions refer to different samples/subjects/ground truth")
        if (f[cols].to_numpy() < 0).any() or not np.allclose(f[cols].sum(axis=1),1, atol=1e-3):
            raise ValueError("Invalid class probabilities")
    probs=np.mean([f[cols].to_numpy() for f in frames],axis=0)
    metrics=classification_metrics(reference.truth.to_numpy(),np.argmax(probs,axis=1))
    metrics["models"]=list(map(str,paths))
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    write_predictions(out/"ensemble_test_predictions.csv",reference.truth.to_numpy(),probs,
                      reference.subject_id.to_numpy(),reference.sample_id.to_numpy())
    write_metrics(metrics,out/"ensemble_metrics.json",
                  note="Fixed uniform ensemble. Do not select ensemble based on test performance.")
    return metrics
