"""Phase 4: train only on train groups, early stop on validation groups.

The untouched test groups are evaluated once after selecting the best epoch.
This is a learning-grade protocol, not a fully powered publication benchmark.
"""
from __future__ import annotations
from pathlib import Path
from .provenance import sha256_file
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .dataset import load_arrays, read_provenance
from .splits import subject_split
from .models import get_model
from .evaluation import classification_metrics, subject_bootstrap_f1, write_metrics, write_predictions


def resolve_device(preference:str="auto") -> torch.device:
    if preference!="auto": return torch.device(preference)
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends,"mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def normalize_by_train(X:np.ndarray, train_indices:np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Channel mean and std estimated from training subjects ONLY."""
    means=X[train_indices].mean(axis=(0,2),keepdims=True)
    stds=X[train_indices].std(axis=(0,2),keepdims=True).clip(min=1e-6)
    transformed=(X-means)/stds
    return transformed.astype(np.float32),means.astype(np.float32),stds.astype(np.float32)


def _probs(model, loader, device):
    model.eval(); probabilities=[]
    with torch.no_grad():
        for X,_ in loader:
            logits=model(X.to(device))
            probabilities.append(torch.softmax(logits,dim=-1).cpu().numpy())
    return np.concatenate(probabilities)


def train_deep(dataset_path:str | Path, output_dir:str | Path, *, architecture:str="cnn",
               epochs:int=8,batch_size:int=16,lr:float=1e-3,seed:int=42,
               device_name:str="auto", patience:int=4) -> dict:
    if epochs < 1 or batch_size < 1: raise ValueError("epochs and batch_size must be positive")
    torch.manual_seed(seed);np.random.seed(seed)
    X,y,groups,ids,fs=load_arrays(dataset_path)
    tr,va,te=subject_split(groups, seed=seed)
    X,mean,std=normalize_by_train(X,tr)
    def loader(indices,shuffle=False):
        tensor=TensorDataset(torch.from_numpy(X[indices]), torch.from_numpy(y[indices]))
        return DataLoader(tensor,batch_size=batch_size,shuffle=shuffle,num_workers=0)
    train_loader=loader(tr,True);val_loader=loader(va);test_loader=loader(te)
    device=resolve_device(device_name)
    print(f"[deep] model={architecture} device={device} samples train/val/test={len(tr)}/{len(va)}/{len(te)}", flush=True)
    model=get_model(architecture, channels=X.shape[1]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-3)
    counts=np.bincount(y[tr],minlength=5)
    weights=np.where(counts>0,1/np.maximum(counts,1),0).astype(np.float32)
    weights=weights / max(weights.sum(),1e-9)*5
    criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,device=device))
    best=-1.0;best_state=None;best_epoch=0;no_improve=0;history=[]
    from sklearn.metrics import f1_score
    for epoch in range(1,epochs+1):
        model.train();epoch_loss=0.;seen=0
        for xb,yb in train_loader:
            xb=xb.to(device);yb=yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits=model(xb)
            loss=criterion(logits,yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            epoch_loss+=float(loss.item())*len(yb);seen+=len(yb)
        probs=_probs(model,val_loader,device)
        f1=float(f1_score(y[va],np.argmax(probs,axis=1),labels=list(range(5)),
                          average="macro",zero_division=0))
        history.append({"epoch":epoch,"training_loss":epoch_loss/seen,"validation_macro_f1":f1})
        print(f"[deep] epoch {epoch:02d}/{epochs} train_loss={epoch_loss/seen:.4f} val_macro_f1={f1:.4f}", flush=True)
        if f1 > best+1e-6:
            best=f1;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch;no_improve=0
        else:
            no_improve+=1
        if no_improve>=patience: break
    assert best_state is not None
    model.load_state_dict(best_state)
    test_probs=_probs(model,test_loader,device)
    pred=np.argmax(test_probs,axis=1)
    metrics=classification_metrics(y[te],pred)
    metrics["macro_f1_subject_ci95"]=list(subject_bootstrap_f1(y[te],pred,groups[te],seed=seed)) if len(set(groups[te]))>1 else None
    metrics["data_source"] = read_provenance(dataset_path)
    metrics["dataset_file"] = str(dataset_path)
    metrics["dataset_sha256"] = sha256_file(dataset_path)
    metrics["random_seed"] = seed
    metrics["subject_split"] = {"train": [str(v) for v in sorted(set(groups[tr]))],
                                "validation": [str(v) for v in sorted(set(groups[va]))],
                                "test": [str(v) for v in sorted(set(groups[te]))]}
    metrics.update({"architecture":architecture,"best_epoch":best_epoch,"validation_macro_f1":best,
                    "history":history,"n_train":len(tr),"n_val":len(va),"n_test":len(te),
                    "n_train_subjects":len(set(groups[tr])),"n_val_subjects":len(set(groups[va])),
                    "n_test_subjects":len(set(groups[te])),"device":str(device)})
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    torch.save({"model_state":best_state,"architecture":architecture,"channels":X.shape[1],
                "mean":mean.tolist(),"std":std.tolist(),"fs":fs,"seed":seed,
                "architecture_version":"mps-safe-fixed-bin-v1",
                "dataset_sha256":metrics["dataset_sha256"],
                "train_subjects":[str(v) for v in sorted(set(groups[tr]))],
                "val_subjects":[str(v) for v in sorted(set(groups[va]))],
                "test_subjects":[str(v) for v in sorted(set(groups[te]))]},out/f"{architecture}_checkpoint.pt")
    note="Held-out subject evaluation; synthetic demos are not scientific evidence."
    write_metrics(metrics,out/f"{architecture}_metrics.json",note=note)
    write_predictions(out/f"{architecture}_test_predictions.csv",y[te],test_probs,groups[te],ids[te])
    return metrics
