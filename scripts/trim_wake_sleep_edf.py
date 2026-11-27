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
