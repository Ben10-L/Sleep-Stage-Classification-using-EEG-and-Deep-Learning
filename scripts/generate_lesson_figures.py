"""Save lesson figures to ./figures using synthetic example signals."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eeg_sleep.plotting import (
    plot_action_potential,
    plot_frequency_bands,
    plot_sampling,
    plot_stage_signatures,
    plot_two_channel_epoch,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures"


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    figures = {
        "01_action_potential.png": plot_action_potential,
        "02_eeg_frequency_bands.png": plot_frequency_bands,
        "03_sleep_stage_patterns.png": plot_stage_signatures,
        "04_sampling.png": plot_sampling,
        "05_synthetic_two_channel_eeg.png": plot_two_channel_epoch,
    }
    for filename, create_figure in figures.items():
        fig = create_figure()
        destination = OUTPUT / filename
        fig.savefig(destination, dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
