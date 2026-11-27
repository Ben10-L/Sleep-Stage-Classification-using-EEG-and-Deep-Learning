import json
from pathlib import Path
import importlib.util
import numpy as np

def load_script(script):
    spec=importlib.util.spec_from_file_location(script.stem,script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

ROOT=Path(__file__).resolve().parents[1]

def test_migration_preserves_and_rewrites_manifest(tmp_path):
    module=load_script(ROOT/'scripts/migrate_existing_project.py')
    source=tmp_path/'old'; target=tmp_path/'new'
    raw=source/'data/raw/mne_cache/physionet-sleep-data'
    raw.mkdir(parents=True)
    psg=raw/'SC4001E0-PSG.edf'; psg.write_bytes(b'test recording')
    hyp=raw/'SC4001EC-Hypnogram.edf'; hyp.write_bytes(b'annotation')
    manifest=source/'data/raw/manifest.json'
    manifest.write_text(json.dumps([[str(psg),str(hyp)]]))
    preview=module.migrate(source,target,apply=False)
    assert preview['copied']==0 and not target.exists()
    summary=module.migrate(source,target,apply=True)
    assert summary['copied']==3
    new_manifest=target/'data/raw/manifest.json'
    items=json.loads(new_manifest.read_text())
    assert all(Path(p).exists() and Path(p).is_relative_to(target) for p in items[0])
    assert psg.exists(), 'Original must never be deleted'

def test_plot_metrics(tmp_path):
    module=load_script(ROOT/'scripts/plot_experiment_results.py')
    file=tmp_path/'metrics.json'
    file.write_text(json.dumps({'confusion_matrix':np.eye(5,dtype=int).tolist(),
                                'history':[{'epoch':1,'training_loss':1.0,'validation_macro_f1':0.3}]}))
    outputs=module.plot_metrics(file,tmp_path/'figs')
    assert len(outputs)==2 and all(p.exists() and p.stat().st_size>1000 for p in outputs)
