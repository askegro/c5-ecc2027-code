from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .signals import ExcitationDesign


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_excitation_spectra(designs: dict[str, ExcitationDesign], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for name, design in designs.items():
        x = design.q - np.mean(design.q)
        f = np.fft.rfftfreq(len(x), design.dt_s)
        p = np.abs(np.fft.rfft(x)) ** 2 / len(x)
        valid = f > 0
        ax.loglog(f[valid], p[valid], label=name.replace("_", " "))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Sequence energy (a.u.)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _save(fig, output)


def plot_metric_boxplot(results: pd.DataFrame, metric: str, ylabel: str, output: Path) -> None:
    data = []
    labels = []
    for method, frame in results.groupby("method", sort=False):
        values = frame[metric].dropna().to_numpy(float)
        if len(values):
            data.append(values)
            labels.append(method.replace("_", "\n"))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if data:
        ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, output)


def plot_chemistry_comparison(results: pd.DataFrame, metric: str, ylabel: str, output: Path) -> None:
    methods = list(dict.fromkeys(results["method"].tolist()))
    chemistries = list(dict.fromkeys(results["chemistry"].tolist()))
    x = np.arange(len(methods), dtype=float)
    width = 0.8 / max(len(chemistries), 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for j, chemistry in enumerate(chemistries):
        medians = [
            results[(results["chemistry"] == chemistry) & (results["method"] == method)][metric].median()
            for method in methods
        ]
        ax.bar(x + (j - (len(chemistries) - 1) / 2) * width, medians, width=width, label=chemistry)
    ax.set_xticks(x, [m.replace("_", "\n") for m in methods])
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    _save(fig, output)


def plot_duration_accuracy(results: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    grouped = results.groupby("method")
    for method, frame in grouped:
        ax.scatter(frame["diagnostic_refresh_time_s"], frame["validation_voltage_rmse_mv"], alpha=0.35, label=method.replace("_", " "))
    ax.set_xlabel("Pack refresh time (s)")
    ax.set_ylabel("Independent voltage RMSE (mV)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, output)
