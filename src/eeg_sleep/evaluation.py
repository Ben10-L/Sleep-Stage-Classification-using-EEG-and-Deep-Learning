"""Class-level evaluation, transparent result saving, and subject bootstrap."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from .constants import STAGES


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=range(5), average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=list(range(5)))),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(5))).tolist(),
        "classification_report": classification_report(y_true, y_pred,
            labels=list(range(5)), target_names=STAGES, output_dict=True, zero_division=0),
    }


def subject_bootstrap_f1(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray,
                         *, repeats: int = 300, seed: int = 42) -> tuple[float,float]:
    """95% percentile CI using subject-cluster bootstrap, not epoch bootstrap."""
    true, pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups)
    subjects = np.unique(groups)
    if len(subjects) < 2:
        raise ValueError("At least two independent test subjects needed")
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repeats):
        draws = rng.choice(subjects, size=len(subjects), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == subj) for subj in draws])
        scores.append(f1_score(true[indices], pred[indices], labels=list(range(5)),
                               average="macro", zero_division=0))
    return tuple(float(x) for x in np.quantile(scores, [.025, .975]))


def write_metrics(metrics: dict, output: str | Path, *, note: str) -> Path:
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"note": note, **metrics}, indent=2), encoding="utf8")
    return path


def write_predictions(output: str | Path, y: np.ndarray, probs: np.ndarray,
                      groups: np.ndarray, ids: np.ndarray) -> Path:
    data = pd.DataFrame({"sample_id": ids, "subject_id": groups, "truth": y,
                         "pred": np.argmax(probs, axis=1)})
    for k, stage in enumerate(STAGES):
        data[f"p_{stage}"] = probs[:,k]
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    return path
