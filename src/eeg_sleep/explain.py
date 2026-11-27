"""Phase 6: input-gradient saliency (educational explanation, not causal proof)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .dataset import load_arrays
from .models import get_model
from .deep_learning import resolve_device


def generate_saliency(dataset_path:str | Path, checkpoint:str | Path, output_path:str | Path,
                      sample_index:int=0, device_name:str="auto") -> Path:
    """Gradient magnitude for model's own predicted class at a selected epoch."""
    X,y,groups,ids,fs=load_arrays(dataset_path)
    if not 0<=sample_index<len(X): raise IndexError(sample_index)
    device=resolve_device(device_name)
    data=torch.load(checkpoint,map_location="cpu",weights_only=True)
    model=get_model(data["architecture"],channels=int(data["channels"]))
    model.load_state_dict(data["model_state"])
    model.to(device).eval()
    normalized=((X[sample_index:sample_index+1]-np.asarray(data["mean"],dtype=np.float32))/np.asarray(data["std"],dtype=np.float32)).astype(np.float32)
    x=torch.tensor(normalized,device=device,requires_grad=True)
    logits=model(x)
    class_id=int(logits.argmax(dim=1).item())
    model.zero_grad(set_to_none=True)
    logits[0,class_id].backward()
    saliency=x.grad.detach().abs().cpu().numpy()[0]
    t=np.arange(X.shape[-1])/fs
    fig,ax=plt.subplots(2,1,figsize=(10,5),sharex=True,layout="constrained")
    ax[0].plot(t,X[sample_index,0],linewidth=.5)
    ax[0].set(ylabel="Fpz-Cz (µV)",title=f"Illustrative gradient saliency | predicted class {class_id}")
    ax[1].plot(t,saliency[0],color="tab:orange",linewidth=.55)
    ax[1].set(xlabel="Time (seconds)",ylabel="|∂logit/∂input|")
    path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=140);plt.close(fig)
    return path
