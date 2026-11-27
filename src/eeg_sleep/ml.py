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
