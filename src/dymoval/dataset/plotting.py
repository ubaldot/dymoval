"""
Plotting utilities for Dataset visualization.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from .core import Dataset


# ====================================================
# Single dataset plotting
# ====================================================


def plot_dataset(
    ds: Dataset,
    *names: str,
    overlay: bool = False,
    figsize: tuple[float, float] | None = None,
):
    """
    Plot signals from a Dataset.

    Parameters
    ----------
    ds : Dataset
    names : str
        Signal names to plot (all if empty)
    overlay : bool
        If True, plot all signals on the same axis
    figsize : tuple
        Optional figure size

    Returns
    -------
    matplotlib.figure.Figure
    """

    if not names:
        names = tuple(ds.data.keys())

    if overlay:
        fig, ax = plt.subplots(figsize=figsize)

        for name in names:
            if name not in ds.data:
                raise KeyError(f"Signal '{name}' not found in dataset")

            ax.plot(ds.time, ds.data[name], label=name)

        ax.set_title("Overlay signals")
        ax.legend()
        ax.grid(True)

        return fig

    # Stacked plots
    fig, axes = plt.subplots(len(names), 1, sharex=True, figsize=figsize)

    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        if name not in ds.data:
            raise KeyError(f"Signal '{name}' not found in dataset")

        ax.plot(ds.time, ds.data[name])
        ax.set_title(name)
        ax.grid(True)

    fig.tight_layout()
    return fig


# ====================================================
# Dataset comparison plotting
# ====================================================


def plot_compare(
    ds1: Dataset,
    ds2: Dataset,
    *names: str,
    align: bool = True,
    labels: tuple[str, str] = ("ref", "cmp"),
    figsize: tuple[float, float] | None = None,
):
    """
    Compare two datasets by plotting shared signals.

    Parameters
    ----------
    ds1, ds2 : Dataset
    names : str
        Signals to compare (auto-detected if empty)
    align : bool
        Align datasets on common time base before plotting
    labels : tuple[str, str]
        Labels for legend
    figsize : tuple
        Optional figure size

    Returns
    -------
    matplotlib.figure.Figure
    """

    # Align datasets if requested
    if align:
        ds1, ds2 = ds1.align(ds2)

    # Auto-detect common signals
    if not names:
        names = tuple(set(ds1.data) & set(ds2.data))

    if not names:
        raise ValueError("No common signals to compare")

    fig, axes = plt.subplots(len(names), 1, sharex=True, figsize=figsize)

    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        if name not in ds1.data or name not in ds2.data:
            raise KeyError(f"Signal '{name}' missing in one of the datasets")

        ax.plot(ds1.time, ds1.data[name], label=labels[0])
        ax.plot(ds2.time, ds2.data[name], label=labels[1])

        ax.set_title(name)
        ax.legend()
        ax.grid(True)

    fig.tight_layout()
    return fig


# ====================================================
# Multi-dataset overlay (advanced)
# ====================================================


def plot_multi(
    reference: Dataset,
    *others: Dataset,
    names: tuple[str, ...] | None = None,
    figsize: tuple[float, float] | None = None,
):
    """
    Overlay multiple datasets for comparison.

    Parameters
    ----------
    reference : Dataset
        Base dataset
    others : Dataset
        Additional datasets
    names : tuple[str]
        Signals to plot (all if None)
    figsize : tuple
        Optional figure size

    Returns
    -------
    matplotlib.figure.Figure
    """

    if names is None:
        names = tuple(reference.data.keys())

    fig, ax = plt.subplots(figsize=figsize)
