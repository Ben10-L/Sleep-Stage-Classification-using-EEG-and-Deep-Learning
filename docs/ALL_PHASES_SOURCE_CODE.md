# Complete Python Source Code — v1.1.0

This reference is generated from the exact Python files in this release. For executable notebooks, open `notebooks/` in VS Code.


---

## `src/eeg_sleep/__init__.py`

```python
"""Educational EEG signal tools for the sleep-staging research project.

All generated signals are synthetic and must not be mistaken for clinical EEG.
"""

from .signals import (
    EEG_CHANNELS,
    FREQUENCY_BANDS,
    nyquist_frequency,
    sample_count,
    simulate_action_potential,
    simulate_sleep_stage,
    simulate_two_channel_eeg,
    sinusoid,
    time_vector,
)
from .epoching import segment_epochs

__all__ = [
    "EEG_CHANNELS",
    "FREQUENCY_BANDS",
    "nyquist_frequency",
    "sample_count",
    "simulate_action_potential",
    "simulate_sleep_stage",
    "simulate_two_channel_eeg",
    "sinusoid",
    "time_vector",
    "segment_epochs",
]
```

---

## `src/eeg_sleep/audit.py`

```python
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
```

---

## `src/eeg_sleep/constants.py`

```python
"""Stage conventions used consistently across the project."""
STAGES = ("W", "N1", "N2", "N3", "REM")
STAGE_TO_INT = {label: index for index, label in enumerate(STAGES)}
INT_TO_STAGE = {v: k for k, v in STAGE_TO_INT.items()}
# The Sleep-EDF annotations include R and both 3 and 4; merge 3/4 to N3.
ANNOTATION_TO_STAGE = {
    "Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2",
    "Sleep stage 3": "N3", "Sleep stage 4": "N3", "Sleep stage R": "REM",
}
EXCLUDED_ANNOTATIONS = {"Sleep stage ?", "Movement time"}
EEG_PICKS = ("EEG Fpz-Cz", "EEG Pz-Oz")
DEFAULT_FS = 100.0
DEFAULT_EPOCH_SECONDS = 30.0
```

---

## `src/eeg_sleep/dataset.py`

```python
"""Phase 1.4 / 2: Sleep-EDF EDF + hypnogram ingestion.

EEG from MNE is converted from V to microvolts. This pipeline does not mix
subjects across partitions; annotation classes 3/4 are combined as N3.
Data are NEVER distributed with the source code.
"""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
from .constants import ANNOTATION_TO_STAGE, EEG_PICKS, STAGE_TO_INT


def subject_id_from_filename(path: str | Path) -> str:
    """SC4001E0-PSG -> SC400; both nights of one person have same group id."""
    match = re.match(r"^(SC\d{3}|ST\d{3})", Path(path).stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unrecognized PhysioNet Sleep-EDF filename: {path}")
    return match.group(1).upper()


def load_sleep_edf_record(psg_path: str | Path, hyp_path: str | Path,
                          *, epoch_seconds: int = 30,
                          max_epochs: int | None = None,
                          filter_continuous: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Extract fully annotated 30-s epochs from a single EDF recording.

    Uses MNE annotation alignment. Annotation intervals with unknown stage or
    movement are excluded. Long annotations are expanded into 30-s segments.
    This lightweight baseline filters each 30-s epoch later, if required.
    """
    import mne
    raw = mne.io.read_raw_edf(str(psg_path), preload=False, verbose="ERROR")
    raw.set_annotations(mne.read_annotations(str(hyp_path)), verbose="ERROR")
    for ch in EEG_PICKS:
        if ch not in raw.ch_names:
            raise ValueError(f"Missing required EEG channel {ch} in {psg_path}")
    # Filter continuous EEG channels before segmentation to reduce epoch-edge artifacts.
    # This loads only the chosen EEG channels into memory (one record at a time).
    raw.pick(list(EEG_PICKS))
    if filter_continuous:
        raw.load_data(verbose="ERROR")
        raw.filter(l_freq=0.5, h_freq=40.0, picks="all", phase="zero", verbose="ERROR")
    fs = float(raw.info["sfreq"])
    points = round(fs * epoch_seconds)
    if abs(fs * epoch_seconds - points) > 1e-8:
        raise ValueError("epoch_seconds × sampling frequency must be integral")
    x_out, y_out = [], []
    for ann in raw.annotations:
        label = ANNOTATION_TO_STAGE.get(ann["description"])
        if label is None:
            continue
        n_intervals = int((float(ann["duration"]) + 1e-5) // epoch_seconds)
        for interval in range(n_intervals):
            start_sec = float(ann["onset"]) + interval * epoch_seconds
            # Annotations are already aligned after raw.set_annotations.
            sample = int(raw.time_as_index([start_sec], use_rounding=True)[0])
            if sample < 0 or sample + points > raw.n_times:
                continue
            value = raw.get_data(picks=list(EEG_PICKS), start=sample,
                                 stop=sample + points, verbose="ERROR") * 1e6
            if value.shape != (len(EEG_PICKS), points) or not np.isfinite(value).all():
                continue
            x_out.append(value.astype(np.float32))
            y_out.append(STAGE_TO_INT[label])
            if max_epochs is not None and len(y_out) >= max_epochs:
                break
        if max_epochs is not None and len(y_out) >= max_epochs:
            break
    if not x_out:
        raise ValueError(f"No valid annotated epochs found in {psg_path}")
    return np.stack(x_out), np.asarray(y_out, dtype=np.int64)


def build_dataset(pairs: list[tuple[str | Path, str | Path]],
                  *, max_epochs_per_record: int | None = None,
                  output: str | Path = "data/processed/sleep_edf.npz") -> Path:
    """Process file pairs; retain source recording and subject identifiers."""
    xs, ys, groups, recs, sample_ids = [], [], [], [], []
    for psg, hyp in pairs:
        x, y = load_sleep_edf_record(psg, hyp, max_epochs=max_epochs_per_record)
        subject = subject_id_from_filename(psg)
        xs.append(x); ys.append(y)
        groups.extend([subject] * len(y))
        recs.extend([Path(psg).stem] * len(y))
        sample_ids.extend([f"{Path(psg).stem}:{i:06d}" for i in range(len(y))])
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=np.concatenate(xs), y=np.concatenate(ys),
                        groups=np.asarray(groups), recordings=np.asarray(recs),
                        sample_ids=np.asarray(sample_ids), fs=100.0,
                        provenance="Sleep-EDF Expanded / PhysioNet")
    return path


def load_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    with np.load(path, allow_pickle=False) as data:
        x = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        groups = data["groups"].astype(str)
        ids = data["sample_ids"].astype(str)
        fs = float(data["fs"])
    if x.ndim != 3 or len(x) != len(y) or len(y) != len(groups) or len(ids) != len(y):
        raise ValueError("Invalid dataset shape")
    if not np.isfinite(x).all() or not np.isfinite(fs) or fs <= 0:
        raise ValueError("Non-finite EEG samples or invalid sampling rate")
    if not np.all((y >= 0) & (y < 5)):
        raise ValueError("Invalid stage labels")
    if len(set(ids)) != len(ids):
        raise ValueError("sample_ids must be unique")
    return x, y, groups, ids, fs


def read_provenance(path: str | Path) -> str:
    """Return the declared data source without loading EEG arrays into memory."""
    with np.load(path, allow_pickle=False) as archive:
        if "provenance" not in archive.files:
            return "UNSPECIFIED — verify data source"
        return str(archive["provenance"].item())
```

---

## `src/eeg_sleep/deep_learning.py`

```python
"""Phase 4: train only on train groups, early stop on validation groups.

The untouched test groups are evaluated once after selecting the best epoch.
This is a learning-grade protocol, not a fully powered publication benchmark.
"""
from __future__ import annotations
from pathlib import Path
from .provenance import sha256_file
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .dataset import load_arrays, read_provenance
from .splits import subject_split
from .models import get_model
from .evaluation import classification_metrics, subject_bootstrap_f1, write_metrics, write_predictions


def resolve_device(preference:str="auto") -> torch.device:
    if preference!="auto": return torch.device(preference)
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends,"mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def normalize_by_train(X:np.ndarray, train_indices:np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Channel mean and std estimated from training subjects ONLY."""
    means=X[train_indices].mean(axis=(0,2),keepdims=True)
    stds=X[train_indices].std(axis=(0,2),keepdims=True).clip(min=1e-6)
    transformed=(X-means)/stds
    return transformed.astype(np.float32),means.astype(np.float32),stds.astype(np.float32)


def _probs(model, loader, device):
    model.eval(); probabilities=[]
    with torch.no_grad():
        for X,_ in loader:
            logits=model(X.to(device))
            probabilities.append(torch.softmax(logits,dim=-1).cpu().numpy())
    return np.concatenate(probabilities)


def train_deep(dataset_path:str | Path, output_dir:str | Path, *, architecture:str="cnn",
               epochs:int=8,batch_size:int=16,lr:float=1e-3,seed:int=42,
               device_name:str="auto", patience:int=4) -> dict:
    if epochs < 1 or batch_size < 1: raise ValueError("epochs and batch_size must be positive")
    torch.manual_seed(seed);np.random.seed(seed)
    X,y,groups,ids,fs=load_arrays(dataset_path)
    tr,va,te=subject_split(groups, seed=seed)
    X,mean,std=normalize_by_train(X,tr)
    def loader(indices,shuffle=False):
        tensor=TensorDataset(torch.from_numpy(X[indices]), torch.from_numpy(y[indices]))
        return DataLoader(tensor,batch_size=batch_size,shuffle=shuffle,num_workers=0)
    train_loader=loader(tr,True);val_loader=loader(va);test_loader=loader(te)
    device=resolve_device(device_name)
    print(f"[deep] model={architecture} device={device} samples train/val/test={len(tr)}/{len(va)}/{len(te)}", flush=True)
    model=get_model(architecture, channels=X.shape[1]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-3)
    counts=np.bincount(y[tr],minlength=5)
    weights=np.where(counts>0,1/np.maximum(counts,1),0).astype(np.float32)
    weights=weights / max(weights.sum(),1e-9)*5
    criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,device=device))
    best=-1.0;best_state=None;best_epoch=0;no_improve=0;history=[]
    from sklearn.metrics import f1_score
    for epoch in range(1,epochs+1):
        model.train();epoch_loss=0.;seen=0
        for xb,yb in train_loader:
            xb=xb.to(device);yb=yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits=model(xb)
            loss=criterion(logits,yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            epoch_loss+=float(loss.item())*len(yb);seen+=len(yb)
        probs=_probs(model,val_loader,device)
        f1=float(f1_score(y[va],np.argmax(probs,axis=1),labels=list(range(5)),
                          average="macro",zero_division=0))
        history.append({"epoch":epoch,"training_loss":epoch_loss/seen,"validation_macro_f1":f1})
        print(f"[deep] epoch {epoch:02d}/{epochs} train_loss={epoch_loss/seen:.4f} val_macro_f1={f1:.4f}", flush=True)
        if f1 > best+1e-6:
            best=f1;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch;no_improve=0
        else:
            no_improve+=1
        if no_improve>=patience: break
    assert best_state is not None
    model.load_state_dict(best_state)
    test_probs=_probs(model,test_loader,device)
    pred=np.argmax(test_probs,axis=1)
    metrics=classification_metrics(y[te],pred)
    metrics["macro_f1_subject_ci95"]=list(subject_bootstrap_f1(y[te],pred,groups[te],seed=seed)) if len(set(groups[te]))>1 else None
    metrics["data_source"] = read_provenance(dataset_path)
    metrics["dataset_file"] = str(dataset_path)
    metrics["dataset_sha256"] = sha256_file(dataset_path)
    metrics["random_seed"] = seed
    metrics["subject_split"] = {"train": [str(v) for v in sorted(set(groups[tr]))],
                                "validation": [str(v) for v in sorted(set(groups[va]))],
                                "test": [str(v) for v in sorted(set(groups[te]))]}
    metrics.update({"architecture":architecture,"best_epoch":best_epoch,"validation_macro_f1":best,
                    "history":history,"n_train":len(tr),"n_val":len(va),"n_test":len(te),
                    "n_train_subjects":len(set(groups[tr])),"n_val_subjects":len(set(groups[va])),
                    "n_test_subjects":len(set(groups[te])),"device":str(device)})
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    torch.save({"model_state":best_state,"architecture":architecture,"channels":X.shape[1],
                "mean":mean.tolist(),"std":std.tolist(),"fs":fs,"seed":seed,
                "architecture_version":"mps-safe-fixed-bin-v1",
                "dataset_sha256":metrics["dataset_sha256"],
                "train_subjects":[str(v) for v in sorted(set(groups[tr]))],
                "val_subjects":[str(v) for v in sorted(set(groups[va]))],
                "test_subjects":[str(v) for v in sorted(set(groups[te]))]},out/f"{architecture}_checkpoint.pt")
    note="Held-out subject evaluation; synthetic demos are not scientific evidence."
    write_metrics(metrics,out/f"{architecture}_metrics.json",note=note)
    write_predictions(out/f"{architecture}_test_predictions.csv",y[te],test_probs,groups[te],ids[te])
    return metrics
```

---

## `src/eeg_sleep/downloading.py`

```python
"""Resumable acquisition of public Sleep-EDF cassette PSG/annotation pairs.

Each successful pair is persisted in a manifest before the next download.
MNE/Pooch handles file hashes and cache reuse. We never distribute EDF files.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable


def read_manifest(path: str | Path) -> list[list[str]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(rows, list) or any(not isinstance(item, list) or len(item) != 2 for item in rows):
        raise ValueError(f"Invalid pair manifest: {path}")
    return rows


def save_manifest(path: str | Path, rows: list[list[str]]) -> None:
    """Atomic update so a reset cannot destroy the last successful manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf8")
    os.replace(tmp, path)


def fetch_incrementally(
    subjects: list[int],
    recordings: list[int],
    *,
    cache: str | Path = "data/raw/mne_cache",
    manifest: str | Path = "data/raw/manifest.json",
    retries: int = 3,
    retry_delay: float = 5.0,
    fetcher: Callable | None = None,
) -> list[list[str]]:
    """Fetch one night at a time, safely merging new entries with existing ones.

    Cache files are validated by MNE/Pooch, not merely by filename existence.
    A failure preserves every previously fetched manifest entry.
    """
    if retries < 1 or retry_delay < 0:
        raise ValueError("retries must be >=1 and retry_delay must be >=0")
    if not subjects or not recordings:
        raise ValueError("At least one subject and recording required")
    if fetcher is None:
        from mne.datasets.sleep_physionet.age import fetch_data
        fetcher = fetch_data
    old = read_manifest(manifest)
    # Keep existing entries; deduplicate on PSG filename, not the order of requests.
    records = {Path(a).name: [str(a), str(b)] for a, b in old}
    for subject in dict.fromkeys(subjects):
        for recording in dict.fromkeys(recordings):
            for attempt in range(1, retries + 1):
                try:
                    returned = fetcher(subjects=[subject], recording=[recording], path=str(cache))
                    for psg, hyp in returned:
                        if not Path(psg).is_file() or not Path(hyp).is_file():
                            raise FileNotFoundError(f"Missing downloaded EDF pair: {psg}, {hyp}")
                        records[Path(psg).name] = [str(Path(psg).resolve()), str(Path(hyp).resolve())]
                        save_manifest(manifest, [records[key] for key in sorted(records)])
                    print(f"[fetch] subject={subject} night={recording} OK; cached pairs={len(records)}", flush=True)
                    break
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as exc:
                    print(f"[fetch] subject={subject} night={recording} attempt={attempt}/{retries}: {type(exc).__name__}: {exc}", flush=True)
                    if attempt == retries:
                        raise RuntimeError(
                            f"Download failed for subject={subject}, recording={recording}. "
                            f"Saved manifest with {len(records)} completed pairs at {manifest}. "
                            "Rerun the same command to resume."
                        ) from exc
                    time.sleep(retry_delay * attempt)
    return [records[key] for key in sorted(records)]
```

---

## `src/eeg_sleep/ensemble.py`

```python
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
```

---

## `src/eeg_sleep/epoching.py`

```python
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
```

---

## `src/eeg_sleep/evaluation.py`

```python
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
```

---

## `src/eeg_sleep/explain.py`

```python
"""Phase 6: input-gradient saliency (educational explanation, not causal proof)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .dataset import load_arrays
from .models import get_model
from .deep_learning import resolve_device


def generate_saliency(dataset_path:str | Path, checkpoint:str | Path, output_path:str | Path,
                      sample_index:int=0, device_name:str="auto") -> Path:
    """Gradient magnitude for model's own predicted class at a selected epoch."""
    X,y,groups,ids,fs=load_arrays(dataset_path)
    if not 0<=sample_index<len(X): raise IndexError(sample_index)
    device=resolve_device(device_name)
    data=torch.load(checkpoint,map_location="cpu",weights_only=True)
    model=get_model(data["architecture"],channels=int(data["channels"]))
    model.load_state_dict(data["model_state"])
    model.to(device).eval()
    normalized=((X[sample_index:sample_index+1]-np.asarray(data["mean"],dtype=np.float32))/np.asarray(data["std"],dtype=np.float32)).astype(np.float32)
    x=torch.tensor(normalized,device=device,requires_grad=True)
    logits=model(x)
    class_id=int(logits.argmax(dim=1).item())
    model.zero_grad(set_to_none=True)
    logits[0,class_id].backward()
    saliency=x.grad.detach().abs().cpu().numpy()[0]
    t=np.arange(X.shape[-1])/fs
    fig,ax=plt.subplots(2,1,figsize=(10,5),sharex=True,layout="constrained")
    ax[0].plot(t,X[sample_index,0],linewidth=.5)
    ax[0].set(ylabel="Fpz-Cz (µV)",title=f"Illustrative gradient saliency | predicted class {class_id}")
    ax[1].plot(t,saliency[0],color="tab:orange",linewidth=.55)
    ax[1].set(xlabel="Time (seconds)",ylabel="|∂logit/∂input|")
    path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=140);plt.close(fig)
    return path
```

---

## `src/eeg_sleep/features.py`

```python
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
```

---

## `src/eeg_sleep/ml.py`

```python
"""Phase 3: subject-independent classical machine-learning baselines."""
from __future__ import annotations
from pathlib import Path
from .provenance import sha256_file
import json
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from .dataset import load_arrays, read_provenance
from .features import extract_features, feature_names
from .splits import subject_split
from .evaluation import classification_metrics, subject_bootstrap_f1, write_metrics, write_predictions


def train_classical(dataset_path: str | Path, output_dir: str | Path,
                    seed: int = 42) -> dict:
    X,y,subjects,ids,fs = load_arrays(dataset_path)
    train_idx,val_idx,test_idx = subject_split(subjects, seed=seed)
    features = extract_features(X, fs=fs)
    candidates = {
        "logistic_regression": Pipeline([("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=seed))]),
        "rbf_svm": Pipeline([("scaler",StandardScaler()),
            ("model", CalibratedClassifierCV(
                SVC(kernel="rbf", class_weight="balanced", random_state=seed),
                method="sigmoid", cv=3, ensemble=False))]),
        "random_forest": RandomForestClassifier(n_estimators=120, min_samples_leaf=2,
                                                 class_weight="balanced", random_state=seed, n_jobs=-1),
    }
    scores={}; fitted={}
    for name, estimator in candidates.items():
        print(f"[ml] training {name}", flush=True)
        estimator.fit(features[train_idx], y[train_idx])
        pred=estimator.predict(features[val_idx])
        scores[name]=classification_metrics(y[val_idx], pred)["macro_f1"]
        print(f"[ml] {name} validation_macro_f1={scores[name]:.4f}", flush=True)
        fitted[name]=estimator
    winner=max(scores, key=lambda k: (scores[k], k))
    model=fitted[winner]
    p=model.predict_proba(features[test_idx])
    # sklearn model.classes_ can omit an absent class in small experiments.
    aligned=np.zeros((len(p),5), dtype=np.float64)
    for col,c in enumerate(model.classes_): aligned[:,int(c)]=p[:,col]
    metrics=classification_metrics(y[test_idx], np.argmax(aligned,axis=1))
    metrics["macro_f1_subject_ci95"] = list(subject_bootstrap_f1(
        y[test_idx], np.argmax(aligned, axis=1), subjects[test_idx], seed=seed)) if len(set(subjects[test_idx]))>1 else None
    metrics["data_source"] = read_provenance(dataset_path)
    metrics["dataset_file"] = str(dataset_path)
    metrics["dataset_sha256"] = sha256_file(dataset_path)
    metrics["random_seed"] = seed
    metrics["subject_split"] = {"train": [str(v) for v in sorted(set(subjects[train_idx]))],
                                "validation": [str(v) for v in sorted(set(subjects[val_idx]))],
                                "test": [str(v) for v in sorted(set(subjects[test_idx]))]}
    metrics.update({"selected_by_validation":winner,"validation_macro_f1":scores,
                    "n_train":len(train_idx),"n_val":len(val_idx),"n_test":len(test_idx),
                    "n_train_subjects":len(set(subjects[train_idx])),
                    "n_val_subjects":len(set(subjects[val_idx])),
                    "n_test_subjects":len(set(subjects[test_idx]))})
    out=Path(output_dir);out.mkdir(parents=True, exist_ok=True)
    note="Evaluation on held-out subjects. Synthetic datasets are demonstration only, not research evidence."
    write_metrics(metrics,out/"ml_metrics.json",note=note)
    write_predictions(out/"ml_test_predictions.csv",y[test_idx],aligned,subjects[test_idx],ids[test_idx])
    joblib.dump({"model":model,"feature_names":feature_names(X.shape[1]),"fs":fs},out/"ml_model.joblib")
    (out/"split_subjects.json").write_text(json.dumps({"train":sorted(set(subjects[train_idx])),
        "val":sorted(set(subjects[val_idx])),"test":sorted(set(subjects[test_idx]))},indent=2))
    return metrics
```

---

## `src/eeg_sleep/models.py`

```python
"""Phase 4: three lightweight research baselines for 30-second raw EEG.

PyTorch expects (batch, channels, time). Model definitions are deliberately
small enough to study and run on Apple Silicon MPS, CPU or CUDA.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class MPSCompatibleAvgPool1d(nn.Module):
    """Produce equal-width averaged temporal bins without adaptive pooling.

    PyTorch's MPS adaptive average pooling can reject lengths that are not
    divisible by the requested output size. Repeat the last feature sample to
    reach a multiple, then use ordinary average pooling on every device.

    This is an explicit pooling design choice, not numerically identical to
    AdaptiveAvgPool1d when non-divisible lengths are involved.
    """

    def __init__(self, output_size: int):
        super().__init__()
        if output_size < 1:
            raise ValueError("output_size must be positive")
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected an EEG feature tensor (batch, channels, time)")
        length = x.size(-1)
        if length < self.output_size:
            raise ValueError("Input length must be >= target number of bins")
        pad = (-length) % self.output_size
        if pad:
            last = x[..., -1:].expand(*x.shape[:-1], pad)
            x = torch.cat((x, last), dim=-1)
        width = x.size(-1) // self.output_size
        return F.avg_pool1d(x, kernel_size=width, stride=width)


class EEGCNN(nn.Module):
    def __init__(self, channels:int=2, classes:int=5):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv1d(channels,16,kernel_size=9,stride=4,padding=4),nn.BatchNorm1d(16),nn.ReLU(),
            nn.Conv1d(16,32,kernel_size=9,stride=4,padding=4),nn.BatchNorm1d(32),nn.ReLU(),
            nn.Conv1d(32,64,kernel_size=7,stride=4,padding=3),nn.BatchNorm1d(64),nn.ReLU(),
            MPSCompatibleAvgPool1d(16),nn.Flatten())
        self.head=nn.Linear(64*16,classes)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.head(self.features(x))


class EEGLSTM(nn.Module):
    def __init__(self, channels:int=2, classes:int=5):
        super().__init__()
        self.front=nn.Sequential(nn.Conv1d(channels,32,9,stride=5,padding=4),nn.ReLU(),
                                 nn.Conv1d(32,48,7,stride=5,padding=3),nn.ReLU(),
                                 MPSCompatibleAvgPool1d(40))
        self.rnn=nn.LSTM(48,48,batch_first=True,bidirectional=True)
        self.head=nn.Linear(96,classes)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x=self.front(x).transpose(1,2)
        encoded,_=self.rnn(x)
        return self.head(encoded.mean(dim=1))


class EEGTransformer(nn.Module):
    def __init__(self, channels:int=2, classes:int=5, tokens:int=48):
        super().__init__()
        self.front=nn.Sequential(nn.Conv1d(channels,48,15,stride=8,padding=7),nn.GELU(),
                                 MPSCompatibleAvgPool1d(tokens))
        self.position=nn.Parameter(torch.zeros(1,tokens,48))
        block=nn.TransformerEncoderLayer(d_model=48,nhead=4,dim_feedforward=96,
                                         dropout=0.1,batch_first=True,activation="gelu")
        self.encoder=nn.TransformerEncoder(block,num_layers=1)
        self.head=nn.Linear(48,classes)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x=self.front(x).transpose(1,2)+self.position
        return self.head(self.encoder(x).mean(dim=1))


def get_model(name:str,channels:int=2,classes:int=5)->nn.Module:
    models={"cnn":EEGCNN,"lstm":EEGLSTM,"transformer":EEGTransformer}
    if name not in models:
        raise ValueError(f"Unknown model {name}. Choose from {list(models)}")
    return models[name](channels=channels,classes=classes)
```

---

## `src/eeg_sleep/plotting.py`

```python
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
```

---

## `src/eeg_sleep/processing.py`

```python
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
```

---

## `src/eeg_sleep/provenance.py`

```python
"""File-based provenance helpers used by scientific experiment artifacts."""
from __future__ import annotations
from pathlib import Path
import hashlib


def sha256_file(path: str | Path) -> str:
    """Compute a deterministic dataset digest with bounded memory."""
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()
```

---

## `src/eeg_sleep/report.py`

```python
"""Phase 7: generate a transparent summary strictly from observed result JSONs."""
from __future__ import annotations
from pathlib import Path
from .provenance import sha256_file
import json


def create_report(results_dir:str | Path, output:str | Path, dataset_path:str | Path) -> Path:
    directory=Path(results_dir)
    observations=[]
    for p in sorted(directory.rglob("*_metrics.json")):
        record=json.loads(p.read_text())
        observations.append((p.stem,record))
    if not observations:
        raise ValueError("No measured metrics exist. Run training first; do not invent results.")
    digest=sha256_file(dataset_path)
    for label, record in observations:
        if record.get("dataset_sha256") and record["dataset_sha256"] != digest:
            raise ValueError(f"Metrics from a different dataset file: {label}")
    synthetic=False
    import numpy as np
    with np.load(dataset_path, allow_pickle=False) as data:
        provenance = str(data["provenance"].item()) if "provenance" in data.files else "UNSPECIFIED"
    synthetic = "SYNTHETIC" in provenance.upper()
    mismatched = [label for label, record in observations
                  if record.get("data_source") is not None
                  and record["data_source"] != provenance]
    if mismatched:
        raise ValueError(
            "Results from incompatible datasets are mixed in one report: "
            + ", ".join(mismatched)
            + ". Choose a results folder containing a single data source."
        )
    lines=["# Sleep Stage Classification — Experimental Results", "",
           "**Status:** " + ("Software demonstration using SIMULATED EEG (not a scientific study)." if synthetic
                           else "Results from a locally processed dataset; independently verify experimental design."),
           "", "## Reproducibility", "",f"- Dataset file: `{dataset_path}`",
           f"- Declared source: `{provenance}`",
           f"- Dataset SHA-256: `{digest}`",
           "- Splits: subjects must not overlap between training, validation and testing.",
           "- Selection: validation data only; test held out until final evaluation.",
           "", "## Recorded outcomes", "",
           "| Model/run | Accuracy | Macro F1 | Balanced accuracy | Cohen kappa |",
           "|---|---:|---:|---:|---:|"]
    for label,r in observations:
        lines.append(f"| {label} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {r['balanced_accuracy']:.3f} | {r['cohen_kappa']:.3f} |")
    lines.extend(["", "## Scientific limitations", "",
                  "- Demonstration waveforms must never be presented as clinical EEG data.",
                  "- The data subset and split seed must be declared in any report.",
                  "- Evaluation must include per-class errors, subject counts and uncertainty.",
                  "- Comparing models on the test set repeatedly can bias claims of superiority.",
                  "- Gradients describe model sensitivity, not validated neurophysiological mechanisms.",
                  "", "## Next steps", "", "Repeat on all eligible real subjects with a pre-registered protocol;",
                  "analyze sensitivity to channel choice, scoring rules and subgroup shift."])
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text("\n".join(lines)+"\n",encoding="utf8")
    return p
```

---

## `src/eeg_sleep/signals.py`

```python
"""Synthetic EEG signals and basic sampling mathematics.

The functions in this module demonstrate concepts from Phase 1.2. They do not
implement a physiological model, validated EEG generator, or sleep-stage scorer.

Waveform amplitudes are expressed in microvolts (µV); time is in seconds unless
otherwise indicated. Simulations are deterministic when given the same seed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

EEG_CHANNELS: tuple[str, str] = ("Fpz-Cz", "Pz-Oz")


@dataclass(frozen=True)
class FrequencyBand:
    """An illustrative EEG frequency band; boundaries vary across references."""

    name: str
    lower_hz: float
    upper_hz: float
    example_hz: float


FREQUENCY_BANDS: tuple[FrequencyBand, ...] = (
    FrequencyBand("Delta", 0.5, 4.0, 2.0),
    FrequencyBand("Theta", 4.0, 8.0, 6.0),
    FrequencyBand("Alpha", 8.0, 13.0, 10.0),
    FrequencyBand("Beta", 13.0, 30.0, 20.0),
    FrequencyBand("Gamma", 30.0, 45.0, 38.0),
)


def _positive_finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive; got {value!r}")
    return number


def sample_count(sampling_rate_hz: float, duration_seconds: float) -> int:
    """Return the exact integer number of equally spaced samples in an epoch."""
    fs = _positive_finite(sampling_rate_hz, "sampling_rate_hz")
    duration = _positive_finite(duration_seconds, "duration_seconds")
    count = fs * duration
    nearest = round(count)
    if nearest == 0 or not math.isclose(count, nearest, abs_tol=1e-9, rel_tol=1e-12):
        raise ValueError("sampling_rate_hz * duration_seconds must be an integer >= 1")
    return nearest


def time_vector(sampling_rate_hz: float, duration_seconds: float) -> NDArray[np.float64]:
    """Return sample times 0, 1/fs, ..., (N-1)/fs (the endpoint is excluded)."""
    n_samples = sample_count(sampling_rate_hz, duration_seconds)
    return np.arange(n_samples, dtype=np.float64) / sampling_rate_hz


def nyquist_frequency(sampling_rate_hz: float) -> float:
    """Return half the sampling rate, in hertz."""
    return _positive_finite(sampling_rate_hz, "sampling_rate_hz") / 2.0


def sinusoid(
    times_seconds: NDArray[np.float64],
    *,
    frequency_hz: float,
    amplitude_uv: float = 1.0,
    phase_radians: float = 0.0,
) -> NDArray[np.float64]:
    """Evaluate A * sin(2*pi*f*t + phase) in microvolts.

    This function does not prevent frequencies above Nyquist: such frequencies
    are useful when explicitly demonstrating aliasing.
    """
    t = np.asarray(times_seconds, dtype=np.float64)
    if t.ndim != 1 or not np.all(np.isfinite(t)):
        raise ValueError("times_seconds must be a finite 1D array")
    if not math.isfinite(frequency_hz) or frequency_hz < 0:
        raise ValueError("frequency_hz must be finite and non-negative")
    if not math.isfinite(amplitude_uv) or not math.isfinite(phase_radians):
        raise ValueError("amplitude and phase must be finite")
    return amplitude_uv * np.sin(2.0 * np.pi * frequency_hz * t + phase_radians)


def simulate_action_potential(n_points: int = 700) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a *stylized* single-neuron action potential (not an EEG trace).

    Piecewise interpolation is used purely to visualize depolarization,
    repolarization, and after-hyperpolarization. This is not a conductance model.
    """
    if n_points < 30:
        raise ValueError("n_points must be at least 30")
    time_ms = np.linspace(0.0, 7.0, n_points)
    knots_ms = np.array([0.0, 1.0, 1.5, 1.75, 2.05, 3.0, 4.2, 5.0, 7.0])
    voltage_mv = np.array([-70.0, -70.0, -55.0, 30.0, 35.0, -70.0, -82.0, -70.0, -70.0])
    return time_ms, np.interp(time_ms, knots_ms, voltage_mv)


def _gaussian(times: NDArray[np.float64], center: float, width: float) -> NDArray[np.float64]:
    return np.exp(-0.5 * ((times - center) / width) ** 2)


def simulate_sleep_stage(
    stage: str,
    *,
    sampling_rate_hz: float = 100.0,
    duration_seconds: float = 3.0,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create a pedagogical 1-channel waveform *resembling* one sleep stage.

    These handcrafted examples are for visual comparison only. Their EEG
    morphology is not clinically validated, and they must not be used as a
    synthetic training set or as evidence of classification performance.
    """
    normalized_stage = stage.upper()
    if normalized_stage not in {"W", "N1", "N2", "N3", "REM"}:
        raise ValueError(f"Unknown sleep-stage label: {stage!r}")
    t = time_vector(sampling_rate_hz, duration_seconds)
    if sampling_rate_hz <= 90.0:
        # This check is conservative: the demonstration uses components <= 32 Hz.
        if sampling_rate_hz <= 64.0:
            raise ValueError("This illustrative simulation requires fs > 64 Hz")
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=2.5, size=t.size)
    if normalized_stage == "W":
        signal = sinusoid(t, frequency_hz=10.0, amplitude_uv=12.0) + sinusoid(
            t, frequency_hz=20.0, amplitude_uv=3.0
        )
    elif normalized_stage == "N1":
        signal = sinusoid(t, frequency_hz=6.0, amplitude_uv=11.0) + sinusoid(
            t, frequency_hz=3.0, amplitude_uv=3.0
        )
    elif normalized_stage == "N2":
        background = sinusoid(t, frequency_hz=6.0, amplitude_uv=5.0)
        spindle = 17.0 * _gaussian(t, center=0.85, width=0.17) * np.sin(2 * np.pi * 13.0 * t)
        k_complex = -28.0 * _gaussian(t, center=2.0, width=0.095) + 15.0 * _gaussian(
            t, center=2.22, width=0.14
        )
        signal = background + spindle + k_complex
    elif normalized_stage == "N3":
        signal = sinusoid(t, frequency_hz=1.5, amplitude_uv=35.0) + sinusoid(
            t, frequency_hz=3.0, amplitude_uv=10.0
        )
    else:
        signal = (
            sinusoid(t, frequency_hz=6.0, amplitude_uv=5.0)
            + sinusoid(t, frequency_hz=11.0, amplitude_uv=3.5)
            + sinusoid(t, frequency_hz=18.0, amplitude_uv=2.0)
        )
    return t, np.asarray(signal + noise, dtype=np.float64)


def simulate_two_channel_eeg(
    *,
    sampling_rate_hz: float = 100.0,
    duration_seconds: float = 30.0,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return time and a synthetic two-channel EEG array (2, n_samples).

    The names Fpz-Cz and Pz-Oz are used to illustrate the expected structure of
    Sleep-EDF input. The waveforms are not actual measurements from those sites.
    """
    t = time_vector(sampling_rate_hz, duration_seconds)
    if sampling_rate_hz <= 64:
        raise ValueError("The demonstration requires a sample rate above 64 Hz")
    rng = np.random.default_rng(seed)
    first = (
        sinusoid(t, frequency_hz=2.0, amplitude_uv=13.0)
        + sinusoid(t, frequency_hz=6.0, amplitude_uv=7.0)
        + sinusoid(t, frequency_hz=10.0, amplitude_uv=4.0)
        + rng.normal(0.0, 3.5, size=t.size)
    )
    second = (
        sinusoid(t, frequency_hz=1.5, amplitude_uv=17.0, phase_radians=0.35)
        + sinusoid(t, frequency_hz=9.0, amplitude_uv=8.0)
        + sinusoid(t, frequency_hz=20.0, amplitude_uv=3.0)
        + rng.normal(0.0, 3.5, size=t.size)
    )
    spindle = 12.0 * _gaussian(t, center=8.0, width=0.3) * np.sin(2 * np.pi * 13.0 * t)
    first += spindle
    second += 0.6 * spindle
    return t, np.vstack((first, second)).astype(np.float64)
```

---

## `src/eeg_sleep/splits.py`

```python
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
```

---

## `src/eeg_sleep/synthetic.py`

```python
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
```

---

## `src/eeg_sleep/trimming.py`

```python
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
```

---

## `scripts/generate_lesson_figures.py`

```python
"""Save lesson figures to ./figures using synthetic example signals."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eeg_sleep.plotting import (
    plot_action_potential,
    plot_frequency_bands,
    plot_sampling,
    plot_stage_signatures,
    plot_two_channel_epoch,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures"


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    figures = {
        "01_action_potential.png": plot_action_potential,
        "02_eeg_frequency_bands.png": plot_frequency_bands,
        "03_sleep_stage_patterns.png": plot_stage_signatures,
        "04_sampling.png": plot_sampling,
        "05_synthetic_two_channel_eeg.png": plot_two_channel_epoch,
    }
    for filename, create_figure in figures.items():
        fig = create_figure()
        destination = OUTPUT / filename
        fig.savefig(destination, dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
```

---

## `scripts/migrate_existing_project.py`

```python
"""Non-destructive migration of local data and measured outputs from an old project.

Never migrate the old virtual environment, source modules, or Git metadata.
Use --yes to make copies; without it, the script only previews the plan.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def migrate(old: Path, new: Path, *, apply: bool = False) -> dict:
    old, new = old.expanduser().resolve(), new.expanduser().resolve()
    if not old.is_dir() or old == new:
        raise ValueError("Provide a different existing --from-dir folder")
    counts = {"copied": 0, "already_present": 0, "conflicts": [], "planned": 0}
    for relative in ("data/raw", "data/processed", "results"):
        src_folder = old / relative
        if not src_folder.exists():
            continue
        for src in sorted(p for p in src_folder.rglob("*") if p.is_file()):
            if src.is_symlink():
                raise ValueError(f"Symbolic links are not migrated: {src}")
            rel = src.relative_to(old)
            dst = new / rel
            counts["planned"] += 1
            if dst.exists():
                if dst.stat().st_size == src.stat().st_size:
                    counts["already_present"] += 1
                else:
                    counts["conflicts"].append(str(rel))
                continue
            if apply:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                counts["copied"] += 1
    if counts["conflicts"]:
        raise ValueError("Existing target files differ; migration did not overwrite: " + ", ".join(counts["conflicts"]))
    # Rewrite manifest paths only after copying, and only if we created a fresh copy.
    source_manifest = old / "data/raw/manifest.json"
    target_manifest = new / "data/raw/manifest.json"
    if apply and source_manifest.exists() and target_manifest.exists():
        pairs = json.loads(target_manifest.read_text())
        fixed=[]
        for item in pairs:
            record=[]
            for path in item:
                candidate=Path(path).expanduser()
                if candidate.is_absolute() and candidate.is_relative_to(old):
                    relocated = new / candidate.relative_to(old)
                    record.append(str(relocated) if relocated.exists() else path)
                else:
                    record.append(path)
            fixed.append(record)
        target_manifest.write_text(json.dumps(fixed, indent=2) + "\n", encoding="utf8")
    return counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-dir", type=Path, required=True)
    p.add_argument("--yes", action="store_true", help="Copy files rather than only preview")
    args=p.parse_args()
    print(json.dumps(migrate(args.from_dir, ROOT, apply=args.yes), indent=2))
    if not args.yes:
        print("Dry run only. Add --yes to copy data and results.")

if __name__ == '__main__': main()
```

---

## `scripts/plot_experiment_results.py`

```python
"""Visualize measured predictions and training history, never invent scores."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eeg_sleep.constants import STAGES


def plot_metrics(json_file: Path, output_dir: Path) -> list[Path]:
    metrics = json.loads(json_file.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    created = []
    if 'confusion_matrix' in metrics:
        matrix = np.asarray(metrics['confusion_matrix'])
        fig, ax = plt.subplots(figsize=(6.2, 5.4), layout='constrained')
        image = ax.imshow(matrix, cmap='Blues')
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                ax.text(c, r, str(matrix[r,c]), ha='center', va='center',
                        color='white' if matrix[r,c] > matrix.max()/2 else 'black', fontsize=9)
        ax.set(xticks=range(5), xticklabels=STAGES, yticks=range(5), yticklabels=STAGES,
               xlabel='Predicted stage', ylabel='True stage', title=json_file.parent.name + ' — held-out test')
        fig.colorbar(image, ax=ax, shrink=.8, label='Epochs')
        path=output_dir/(json_file.stem+'_confusion.png')
        fig.savefig(path, dpi=160)
        plt.close(fig)
        created.append(path)
    if 'history' in metrics and metrics['history']:
        history=metrics['history']
        epochs=[int(entry['epoch']) for entry in history]
        fig, axes=plt.subplots(2, 1, figsize=(7,5.5), sharex=True, layout='constrained')
        axes[0].plot(epochs, [entry['training_loss'] for entry in history], marker='o', ms=3)
        axes[0].set(ylabel='Training loss', title='Deep-learning training history')
        axes[1].plot(epochs, [entry['validation_macro_f1'] for entry in history], marker='o', ms=3)
        axes[1].set(xlabel='Epoch', ylabel='Val macro F1', ylim=(0,1))
        path=output_dir/(json_file.stem+'_learning_curve.png')
        fig.savefig(path,dpi=160)
        plt.close(fig)
        created.append(path)
    if not created:
        raise ValueError('No available confusion matrix or training history in the supplied metrics file')
    return created


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('results/figures'))
    a=p.parse_args()
    for created in plot_metrics(a.metrics, a.output):
        print(created)

if __name__=='__main__': main()
```

---

## `scripts/run_pipeline.py`

```python
"""Command line entrypoint for the end-to-end EEG sleep-stage pipeline."""
from __future__ import annotations
import argparse
import json


def main() -> None:
    p = argparse.ArgumentParser(description="EEG sleep-stage research pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Create SYNTHETIC software-test data (not research)")
    demo.add_argument("--output", default="data/processed/synthetic_demo.npz")
    demo.add_argument("--subjects", type=int, default=10)
    demo.add_argument("--repeats", type=int, default=3)

    fetch = sub.add_parser("fetch", help="Fetch Sleep-EDF with incremental manifest + retry")
    fetch.add_argument("--subjects", type=int, nargs="+", default=list(range(10)))
    fetch.add_argument("--recordings", type=int, nargs="+", default=[1])
    fetch.add_argument("--cache", default="data/raw/mne_cache")
    fetch.add_argument("--manifest", default="data/raw/manifest.json")
    fetch.add_argument("--retries", type=int, default=3)
    fetch.add_argument("--retry-delay", type=float, default=5.0)

    prepare = sub.add_parser("prepare", help="Create 30-second labeled epochs from EDF")
    prepare.add_argument("--manifest", default="data/raw/manifest.json")
    prepare.add_argument("--output", default="data/processed/sleep_edf.npz")
    prepare.add_argument("--max-epochs-per-record", type=int, default=None)

    trim = sub.add_parser("trim", help="Pilot wake trimming; leaves original archive intact")
    trim.add_argument("--input", default="data/processed/sleep_edf.npz")
    trim.add_argument("--output", default="data/processed/sleep_edf_trimmed.npz")
    trim.add_argument("--wake-margin-minutes", type=int, default=30)

    audit = sub.add_parser("inspect", help="Audit class balance, shapes, subjects, and split")
    audit.add_argument("--data", default="data/processed/sleep_edf_trimmed.npz")
    audit.add_argument("--seed", type=int, default=42)

    ml = sub.add_parser("ml", help="Fit classical ML baselines with val-only selection")
    ml.add_argument("--data", default="data/processed/synthetic_demo.npz")
    ml.add_argument("--output", default="results/ml")
    ml.add_argument("--seed", type=int, default=42)

    deep = sub.add_parser("deep", help="Train CNN, BiLSTM, or Transformer")
    deep.add_argument("--data", default="data/processed/synthetic_demo.npz")
    deep.add_argument("--output", default="results/deep")
    deep.add_argument("--model", choices=["cnn", "lstm", "transformer"], default="cnn")
    deep.add_argument("--epochs", type=int, default=8)
    deep.add_argument("--batch-size", type=int, default=16)
    deep.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    deep.add_argument("--seed", type=int, default=42)
    deep.add_argument("--lr", type=float, default=1e-3)
    deep.add_argument("--patience", type=int, default=4)

    ensemble = sub.add_parser("ensemble", help="Average class probabilities of aligned runs")
    ensemble.add_argument("--inputs", nargs="+", required=True)
    ensemble.add_argument("--output", default="results/ensemble")

    explain = sub.add_parser("explain", help="Plot local gradient saliency")
    explain.add_argument("--data", default="data/processed/synthetic_demo.npz")
    explain.add_argument("--checkpoint", default="results/deep/cnn_checkpoint.pt")
    explain.add_argument("--output", default="results/deep/saliency.png")
    explain.add_argument("--index", type=int, default=0)
    explain.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")

    report = sub.add_parser("report", help="Write Markdown summary from measured metrics")
    report.add_argument("--results", default="results")
    report.add_argument("--data", default="data/processed/synthetic_demo.npz")
    report.add_argument("--output", default="results/EXPERIMENT_REPORT.md")

    args = p.parse_args()
    if args.command == "demo":
        from eeg_sleep.synthetic import make_demo_dataset
        print(f"SYNTHETIC ONLY: {make_demo_dataset(args.output, args.subjects, args.repeats)}")
    elif args.command == "fetch":
        from eeg_sleep.downloading import fetch_incrementally
        pairs = fetch_incrementally(args.subjects, args.recordings, cache=args.cache,
                                    manifest=args.manifest, retries=args.retries,
                                    retry_delay=args.retry_delay)
        print(f"Manifest: {args.manifest} | Saved {len(pairs)} verified EDF pairs")
    elif args.command == "prepare":
        from eeg_sleep.dataset import build_dataset
        from eeg_sleep.downloading import read_manifest
        print(build_dataset(read_manifest(args.manifest),
                            max_epochs_per_record=args.max_epochs_per_record, output=args.output))
    elif args.command == "trim":
        from eeg_sleep.trimming import trim_dataset
        print(json.dumps(trim_dataset(args.input, args.output, args.wake_margin_minutes), indent=2))
    elif args.command == "inspect":
        from eeg_sleep.audit import inspect_dataset
        print(json.dumps(inspect_dataset(args.data, args.seed), indent=2))
    elif args.command == "ml":
        from eeg_sleep.ml import train_classical
        print(json.dumps(train_classical(args.data, args.output, seed=args.seed), indent=2))
    elif args.command == "deep":
        from eeg_sleep.deep_learning import train_deep
        result = train_deep(args.data, args.output, architecture=args.model,
                            epochs=args.epochs, batch_size=args.batch_size,
                            device_name=args.device, seed=args.seed, lr=args.lr,
                            patience=args.patience)
        print(json.dumps(result, indent=2))
    elif args.command == "ensemble":
        from eeg_sleep.ensemble import average_predictions
        print(json.dumps(average_predictions(args.inputs, args.output), indent=2))
    elif args.command == "explain":
        from eeg_sleep.explain import generate_saliency
        print(generate_saliency(args.data, args.checkpoint, args.output,
                                sample_index=args.index, device_name=args.device))
    elif args.command == "report":
        from eeg_sleep.report import create_report
        print(create_report(args.results, args.output, args.data))


if __name__ == "__main__":
    main()
```

---

## `scripts/trim_wake_sleep_edf.py`

```python
"""Compatibility entrypoint for wake trimming; use run_pipeline.py trim too."""
from __future__ import annotations
import argparse
import json
from eeg_sleep.trimming import trim_dataset

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/processed/sleep_edf.npz")
    p.add_argument("--output", default="data/processed/sleep_edf_trimmed.npz")
    p.add_argument("--wake-margin-minutes", type=int, default=30)
    a = p.parse_args()
    print(json.dumps(trim_dataset(a.input, a.output, a.wake_margin_minutes), indent=2))

if __name__ == "__main__":
    main()
```
