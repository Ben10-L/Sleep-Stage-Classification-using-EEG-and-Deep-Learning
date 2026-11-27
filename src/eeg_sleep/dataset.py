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
