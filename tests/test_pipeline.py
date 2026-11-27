"""Additional tests for every newly added module, using SYNTHETIC data only."""
import numpy as np
import pytest
import pandas as pd
import torch
from eeg_sleep.processing import bandpass, power_spectrum, time_frequency
from eeg_sleep.features import extract_features, feature_names
from eeg_sleep.dataset import subject_id_from_filename, load_arrays
from eeg_sleep.synthetic import make_demo_dataset
from eeg_sleep.splits import subject_split
from eeg_sleep.deep_learning import normalize_by_train
from eeg_sleep.models import get_model
from eeg_sleep.ml import train_classical
from eeg_sleep.evaluation import classification_metrics, subject_bootstrap_f1
from eeg_sleep.ensemble import average_predictions
from eeg_sleep.evaluation import write_predictions
from eeg_sleep.constants import ANNOTATION_TO_STAGE


def test_signal_processing():
    fs=100
    t=np.arange(3000)/fs
    x=np.sin(2*np.pi*10*t)+.2*np.sin(2*np.pi*45*t)
    y=bandpass(x,fs,0.5,30)
    assert y.shape==x.shape and np.isfinite(y).all()
    f,p=power_spectrum(x,fs)
    assert abs(float(f[np.argmax(p)])-10)<1
    f,ts,S=time_frequency(x,fs)
    assert S.shape==(len(f),len(ts))


def test_invalid_filter_bounds():
    with pytest.raises(ValueError): bandpass(np.arange(400), fs=100, low=1, high=55)


def test_features_and_labels():
    x=np.random.default_rng(3).normal(size=(3,2,3000)).astype('float32')
    f=extract_features(x)
    assert f.shape==(3,len(feature_names(2)))
    assert (f[:,6:]>=0).all() is not None  # relative powers are nonnegative
    assert ANNOTATION_TO_STAGE['Sleep stage 4']=='N3'
    assert 'Sleep stage ?' not in ANNOTATION_TO_STAGE


def test_subject_identity():
    assert subject_id_from_filename('SC4001E0-PSG.edf')==subject_id_from_filename('SC4002E0-PSG.edf')
    assert subject_id_from_filename('SC4011E0-PSG.edf')!='SC400'
    with pytest.raises(ValueError): subject_id_from_filename('unknown.edf')


def test_subject_splits():
    groups=np.repeat([f's{i}' for i in range(12)], 3)
    tr,va,te=subject_split(groups)
    assert sorted(np.r_[tr,va,te])==list(range(len(groups)))
    assert not set(groups[tr])&set(groups[te])
    assert not set(groups[va])&set(groups[te])
    assert not set(groups[tr])&set(groups[va])


def test_group_split_needs_subjects():
    with pytest.raises(ValueError):subject_split(np.array(['s1','s1']))


def test_synthetic_dataset_schema(tmp_path):
    path=make_demo_dataset(tmp_path/'demo.npz',n_subjects=10,repeats_per_stage=1)
    X,y,groups,ids,fs=load_arrays(path)
    assert X.shape==(50,2,3000)
    assert set(y)==set(range(5))
    assert len(set(groups))==10
    assert len(set(ids))==len(ids)
    assert fs==100.0


def test_train_stats_no_leak():
    X=np.zeros((4,2,10),dtype=np.float32)
    X[0:2]=2
    X[2:]=1000
    out,mean,std=normalize_by_train(X,np.array([0,1]))
    assert np.allclose(mean,2)
    assert np.allclose(out[:2],0)
    assert np.max(out[2:])>100


@pytest.mark.parametrize('arch',['cnn','lstm','transformer'])
def test_model_tensor_shapes(arch):
    net=get_model(arch);net.eval()
    with torch.no_grad(): y=net(torch.randn(2,2,3000))
    assert y.shape==(2,5)
    assert torch.isfinite(y).all()


def test_metrics_and_subject_bootstrap():
    y=np.array([0,0,1,1,2,2,3,3,4,4])
    metrics=classification_metrics(y,y)
    assert metrics['macro_f1']==1
    lo,hi=subject_bootstrap_f1(y,y,np.array([f's{i//2}' for i in range(10)]), repeats=12)
    assert 0<=lo<=hi<=1


def test_ensemble_alignment_and_probabilities(tmp_path):
    ids=np.array(['a','b','c','d','e'])
    groups=np.array(['p1','p2','p3','p4','p5'])
    y=np.array([0,1,2,3,4]);p=np.eye(5)*.8+.2/5
    a=write_predictions(tmp_path/'a.csv',y,p,groups,ids)
    b=write_predictions(tmp_path/'b.csv',y,p,groups,ids)
    metrics=average_predictions([a,b],tmp_path/'ens')
    assert metrics['macro_f1']==1
    wrong=pd.read_csv(b);wrong.loc[0,'subject_id']='incorrect';wrong.to_csv(b,index=False)
    with pytest.raises(ValueError):average_predictions([a,b],tmp_path/'bad')


def test_ml_pipeline_smoke(tmp_path):
    path=make_demo_dataset(tmp_path/'demo.npz',n_subjects=10,repeats_per_stage=1)
    metrics=train_classical(path,tmp_path/'results')
    assert 0<=metrics['macro_f1']<=1
    assert (tmp_path/'results'/'ml_metrics.json').exists()
    assert (tmp_path/'results'/'ml_test_predictions.csv').exists()
    assert metrics['n_train_subjects']>0


def test_sleep_edf_annotation_alignment_with_mocked_mne(monkeypatch):
    """Test EDF/hypnogram mapping without external downloads."""
    import mne
    from eeg_sleep.dataset import load_sleep_edf_record
    rng=np.random.default_rng(0)
    info=mne.create_info(["EEG Fpz-Cz","EEG Pz-Oz","EOG horizontal"],sfreq=100,
                         ch_types=["eeg","eeg","eog"])
    data=rng.normal(0,15e-6,size=(3,9000))
    def fake_read_raw_edf(*args,**kwargs):
        return mne.io.RawArray(data.copy(),info.copy(),verbose="ERROR")
    def fake_read_annotations(*args,**kwargs):
        return mne.Annotations(onset=[0.,30.,60.],duration=[30.,30.,30.],
                               description=["Sleep stage W","Sleep stage 4","Movement time"])
    monkeypatch.setattr(mne.io,'read_raw_edf',fake_read_raw_edf)
    monkeypatch.setattr(mne,'read_annotations',fake_read_annotations)
    x,y=load_sleep_edf_record('SC4001E0-PSG.edf','SC4001EC-Hypnogram.edf')
    assert x.shape==(2,2,3000)
    assert y.tolist()==[0,3]  # original '4' -> N3, movement excluded
    assert abs(float(np.std(x)))>1 # microvolt scaling, not volts
