"""Visualize measured predictions and training history, never invent scores."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eeg_sleep.constants import STAGES


def plot_metrics(json_file: Path, output_dir: Path) -> list[Path]:
    metrics = json.loads(json_file.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    created = []
    if 'confusion_matrix' in metrics:
        matrix = np.asarray(metrics['confusion_matrix'])
        fig, ax = plt.subplots(figsize=(6.2, 5.4), layout='constrained')
        image = ax.imshow(matrix, cmap='Blues')
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                ax.text(c, r, str(matrix[r,c]), ha='center', va='center',
                        color='white' if matrix[r,c] > matrix.max()/2 else 'black', fontsize=9)
        ax.set(xticks=range(5), xticklabels=STAGES, yticks=range(5), yticklabels=STAGES,
               xlabel='Predicted stage', ylabel='True stage', title=json_file.parent.name + ' — held-out test')
        fig.colorbar(image, ax=ax, shrink=.8, label='Epochs')
        path=output_dir/(json_file.stem+'_confusion.png')
        fig.savefig(path, dpi=160)
        plt.close(fig)
        created.append(path)
    if 'history' in metrics and metrics['history']:
        history=metrics['history']
        epochs=[int(entry['epoch']) for entry in history]
        fig, axes=plt.subplots(2, 1, figsize=(7,5.5), sharex=True, layout='constrained')
        axes[0].plot(epochs, [entry['training_loss'] for entry in history], marker='o', ms=3)
        axes[0].set(ylabel='Training loss', title='Deep-learning training history')
        axes[1].plot(epochs, [entry['validation_macro_f1'] for entry in history], marker='o', ms=3)
        axes[1].set(xlabel='Epoch', ylabel='Val macro F1', ylim=(0,1))
        path=output_dir/(json_file.stem+'_learning_curve.png')
        fig.savefig(path,dpi=160)
        plt.close(fig)
        created.append(path)
    if not created:
        raise ValueError('No available confusion matrix or training history in the supplied metrics file')
    return created


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('results/figures'))
    a=p.parse_args()
    for created in plot_metrics(a.metrics, a.output):
        print(created)

if __name__=='__main__': main()
