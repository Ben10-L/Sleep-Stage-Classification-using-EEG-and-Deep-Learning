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
