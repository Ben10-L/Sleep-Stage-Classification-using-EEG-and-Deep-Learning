"""Phase 4: three lightweight research baselines for 30-second raw EEG.

PyTorch expects (batch, channels, time). Model definitions are deliberately
small enough to study and run on Apple Silicon MPS, CPU or CUDA.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class MPSCompatibleAvgPool1d(nn.Module):
    """Produce equal-width averaged temporal bins without adaptive pooling.

    PyTorch's MPS adaptive average pooling can reject lengths that are not
    divisible by the requested output size. Repeat the last feature sample to
    reach a multiple, then use ordinary average pooling on every device.

    This is an explicit pooling design choice, not numerically identical to
    AdaptiveAvgPool1d when non-divisible lengths are involved.
    """

    def __init__(self, output_size: int):
        super().__init__()
        if output_size < 1:
            raise ValueError("output_size must be positive")
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected an EEG feature tensor (batch, channels, time)")
        length = x.size(-1)
        if length < self.output_size:
            raise ValueError("Input length must be >= target number of bins")
        pad = (-length) % self.output_size
        if pad:
            last = x[..., -1:].expand(*x.shape[:-1], pad)
            x = torch.cat((x, last), dim=-1)
        width = x.size(-1) // self.output_size
        return F.avg_pool1d(x, kernel_size=width, stride=width)


class EEGCNN(nn.Module):
    def __init__(self, channels:int=2, classes:int=5):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv1d(channels,16,kernel_size=9,stride=4,padding=4),nn.BatchNorm1d(16),nn.ReLU(),
            nn.Conv1d(16,32,kernel_size=9,stride=4,padding=4),nn.BatchNorm1d(32),nn.ReLU(),
            nn.Conv1d(32,64,kernel_size=7,stride=4,padding=3),nn.BatchNorm1d(64),nn.ReLU(),
            MPSCompatibleAvgPool1d(16),nn.Flatten())
        self.head=nn.Linear(64*16,classes)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.head(self.features(x))


class EEGLSTM(nn.Module):
    def __init__(self, channels:int=2, classes:int=5):
        super().__init__()
        self.front=nn.Sequential(nn.Conv1d(channels,32,9,stride=5,padding=4),nn.ReLU(),
                                 nn.Conv1d(32,48,7,stride=5,padding=3),nn.ReLU(),
                                 MPSCompatibleAvgPool1d(40))
        self.rnn=nn.LSTM(48,48,batch_first=True,bidirectional=True)
        self.head=nn.Linear(96,classes)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x=self.front(x).transpose(1,2)
        encoded,_=self.rnn(x)
        return self.head(encoded.mean(dim=1))


class EEGTransformer(nn.Module):
    def __init__(self, channels:int=2, classes:int=5, tokens:int=48):
        super().__init__()
        self.front=nn.Sequential(nn.Conv1d(channels,48,15,stride=8,padding=7),nn.GELU(),
                                 MPSCompatibleAvgPool1d(tokens))
        self.position=nn.Parameter(torch.zeros(1,tokens,48))
        block=nn.TransformerEncoderLayer(d_model=48,nhead=4,dim_feedforward=96,
                                         dropout=0.1,batch_first=True,activation="gelu")
        self.encoder=nn.TransformerEncoder(block,num_layers=1)
        self.head=nn.Linear(48,classes)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x=self.front(x).transpose(1,2)+self.position
        return self.head(self.encoder(x).mean(dim=1))


def get_model(name:str,channels:int=2,classes:int=5)->nn.Module:
    models={"cnn":EEGCNN,"lstm":EEGLSTM,"transformer":EEGTransformer}
    if name not in models:
        raise ValueError(f"Unknown model {name}. Choose from {list(models)}")
    return models[name](channels=channels,classes=classes)
