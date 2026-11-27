"""Regression tests for the unified real-data workflow (no internet required)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
from eeg_sleep.downloading import fetch_incrementally, read_manifest
from eeg_sleep.trimming import trim_dataset
from eeg_sleep.audit import inspect_dataset
from eeg_sleep.synthetic import make_demo_dataset

def attach_recording_labels(path):
    """Synthetic fixture with explicit recording IDs for wake-trim integration."""
    with np.load(path) as archive:
        arrays = {k: archive[k] for k in archive.files}
    arrays["recordings"] = np.asarray(arrays["groups"]).astype(str)
    np.savez_compressed(path, **arrays)
    return path


def test_fetch_manifest_survives_interruption(tmp_path):
    cache = tmp_path / 'cache'
    cache.mkdir()
    manifest = tmp_path / 'manifest.json'
    called = []

    def fake_fetcher(*, subjects, recording, path):
        subject = subjects[0]
        called.append(subject)
        if subject == 2:
            raise ConnectionResetError('simulated connection drop')
        psg = cache / f'SC40{subject}1E0-PSG.edf'
        hyp = cache / f'SC40{subject}1EC-Hypnogram.edf'
        psg.write_bytes(b'demo')
        hyp.write_bytes(b'demo')
        return [[psg, hyp]]

    with pytest.raises(RuntimeError, match='Rerun the same command'):
        fetch_incrementally([0,1,2], [1], cache=cache, manifest=manifest,
                            retries=1, retry_delay=0, fetcher=fake_fetcher)
    assert len(read_manifest(manifest)) == 2
    assert called == [0,1,2]
    again = fetch_incrementally([0,1], [1], cache=cache, manifest=manifest,
                                retries=1, retry_delay=0, fetcher=fake_fetcher)
    assert len(again) == 2
    assert len({row[0] for row in again}) == 2


def test_trim_preserves_original_and_metadata(tmp_path):
    source = attach_recording_labels(make_demo_dataset(tmp_path / 'original.npz', n_subjects=10, repeats_per_stage=2))
    with np.load(source) as original:
        keys = list(original.files)
        before_len = len(original['y'])
    output = tmp_path / 'trimmed.npz'
    summary = trim_dataset(source, output, wake_margin_minutes=0)
    assert output.exists()
    with np.load(output) as d:
        assert set(keys).issubset(d.files)
        assert len(d['X']) == len(d['y']) == len(d['groups'])
        assert len(d['y']) <= before_len
        assert np.isfinite(d['X']).all()
    with np.load(source) as original:
        assert len(original['y']) == before_len
    assert summary['trimmed_epochs'] <= before_len


def test_dataset_audit_is_read_only(tmp_path):
    source = make_demo_dataset(tmp_path / 'demo.npz', n_subjects=10, repeats_per_stage=2)
    audit = inspect_dataset(source)
    assert audit['shape'] == [100,2,3000]
    assert audit['n_subjects'] == 10
    assert audit['all_finite']
    assert sum(audit['class_counts'].values()) == 100
    splits = audit['splits']
    group_sets = [set(splits[part]['subjects']) for part in ('train','validation','test')]
    assert not group_sets[0].intersection(group_sets[1])
    assert not group_sets[0].intersection(group_sets[2])
    assert not group_sets[1].intersection(group_sets[2])


def test_invalid_trim_parameters(tmp_path):
    source = attach_recording_labels(make_demo_dataset(tmp_path/'demo.npz', n_subjects=10, repeats_per_stage=1))
    with pytest.raises(ValueError, match='overwrite'):
        trim_dataset(source, source)
    with pytest.raises(ValueError, match='margin'):
        trim_dataset(source, tmp_path/'out.npz', wake_margin_minutes=-1)
