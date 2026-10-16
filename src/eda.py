"""Generate compact exploratory plots for the processed heart dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "heart.csv"


def create_eda_report(
    data_path: str | Path = DATA_PATH,
    output_dir: str | Path = PROJECT_ROOT / "reports",
) -> list[Path]:
    """Save target-distribution and feature-correlation plots."""
    frame = pd.read_csv(data_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []

    figure, axis = plt.subplots(figsize=(6, 4))
    sns.countplot(data=frame, x="target", hue="target", legend=False, ax=axis)
    axis.set(
        title="Heart disease target distribution",
        xlabel="Disease",
        ylabel="Patients",
    )
    figure.tight_layout()
    target_plot = destination / "target_distribution.png"
    figure.savefig(target_plot, dpi=150)
    plt.close(figure)
    outputs.append(target_plot)

    figure, axis = plt.subplots(figsize=(11, 9))
    sns.heatmap(frame.corr(numeric_only=True), cmap="vlag", center=0, ax=axis)
    axis.set_title("Feature correlation")
    figure.tight_layout()
    correlation_plot = destination / "feature_correlations.png"
    figure.savefig(correlation_plot, dpi=150)
    plt.close(figure)
    outputs.append(correlation_plot)
    return outputs


if __name__ == "__main__":
    print("Saved plots:", create_eda_report())
