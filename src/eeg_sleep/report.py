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
