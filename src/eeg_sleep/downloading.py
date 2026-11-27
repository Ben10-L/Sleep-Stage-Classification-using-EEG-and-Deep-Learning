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
