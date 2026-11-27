# Local-only biomedical data

The repository does **not** include EDF recordings, extracted epochs, or neural-network weights. Download from [Sleep-EDF Expanded on PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/) and comply with its license.

Expected local layout after the workflow:

```text
data/
├── raw/
│   ├── manifest.json
│   └── mne_cache/physionet-sleep-data/*.edf
└── processed/
    ├── sleep_edf.npz
    └── sleep_edf_trimmed.npz
```

All files under `data/raw` and `data/processed` are ignored by Git. To reuse data from an earlier project without downloading again, see [`docs/MIGRATION.md`](../docs/MIGRATION.md).
