"""Plotting helpers that operate on more than one object.

Single-object plotting lives on the objects themselves
(``Signal.plot``, ``Dataset.plot``, ``Dataset.plot_spectrum``, ...).
This module only adds what cannot belong to a single object, namely
plotting a bunch of loose signals and comparing datasets.
"""

from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .dataset import Dataset
from .scope import DatasetScope, SpectrumScope
from .signal import SPECTRUM_MODES, Signal, SpectrumMode

__all__ = [
    "plot_signals",
    "plot_dataset",
    "plot_compare",
    "plot_spectrum_compare",
]


def plot_signals(
    *signals: Signal | tuple[Signal, ...],
    with_scope: bool = True,
) -> Figure:
    """Plot loose :class:`dymoval.signal.Signal`, one subplot per group.

    Unlike :meth:`dymoval.dataset.Dataset.plot`, the signals need not be
    aligned: this is the function to use to eyeball raw logs *before*
    building a ``Dataset``. Signals passed as a tuple are overlaid on the
    same axes.

    Parameters
    ----------
    signals :
        The signals to plot. A tuple of signals is drawn on one subplot.
    with_scope :
        Attach an interactive scope to the figure.
    """
    groups: list[tuple[Signal, ...]] = []

    for item in signals:
        group = item if isinstance(item, tuple) else (item,)

        for sig in group:
            if not isinstance(sig, Signal):
                raise TypeError("All the arguments must be Signal instances")

        groups.append(group)

    if not groups:
        raise ValueError("At least one signal is required")

    fig = plt.figure(
        constrained_layout=True, figsize=(10, 2.0 * len(groups) + 1)
    )

    if with_scope:
        subfigs = fig.subfigures(1, 2, width_ratios=[3.8, 1.2])
        host: Any = subfigs[0]
    else:
        host = fig

    axes = list(np.atleast_1d(host.subplots(len(groups), 1)))

    for ax, group in zip(axes, groups):
        for sig in group:
            sig._plot_standard(ax=ax)

        ax.legend()
        ax.grid(True)

    if with_scope:
        panel_ax = subfigs[1].add_subplot()
        panel_ax.set_anchor("N")
        DatasetScope(fig, axes, panel_ax)

    return fig


def plot_dataset(
    ds: Dataset,
    *names: str | tuple[str, ...],
    with_scope: bool = True,
) -> Figure:
    """Functional alias of :meth:`dymoval.dataset.Dataset.plot`."""
    return ds.plot(*names, with_scope=with_scope)


# ====================================================
# Internals
# ====================================================
def _resolve_labels(
    datasets: Sequence[Dataset], labels: Sequence[str] | None
) -> list[str]:
    if labels is None:
        return [f"ds{i}" for i in range(len(datasets))]

    if len(labels) != len(datasets):
        raise ValueError("'labels' must have one entry per dataset")

    return list(labels)


def _common_names(
    datasets: Sequence[Dataset], names: Sequence[str]
) -> list[str]:
    if names:
        for i, ds in enumerate(datasets):
            missing = [name for name in names if name not in ds]

            if missing:
                raise KeyError(f"Signals {missing} missing in dataset #{i}")

        return list(names)

    common = set(datasets[0].names())

    for ds in datasets[1:]:
        common &= set(ds.names())

    # keep the reference dataset ordering
    ordered = [name for name in datasets[0].names() if name in common]

    if not ordered:
        raise ValueError("No common signals to compare")

    return ordered


def _aligned(datasets: Sequence[Dataset], align: bool) -> list[Dataset]:
    if not align:
        return list(datasets)

    reference = datasets[0]
    new_time = reference.time()

    for ds in datasets[1:]:
        t = ds.time()
        new_time = new_time[(new_time >= t[0]) & (new_time <= t[-1])]

    if new_time.size == 0:
        raise ValueError("Datasets do not overlap in time")

    return [ds.resample(new_time) for ds in datasets]


def _check_datasets(
    reference: Dataset, others: Sequence[Dataset]
) -> list[Dataset]:
    datasets = [reference, *others]

    if len(datasets) < 2:
        raise ValueError("At least two datasets are required")

    for ds in datasets:
        if not isinstance(ds, Dataset):
            raise TypeError("All the arguments must be Dataset instances")

    return datasets


# ====================================================
# Comparison
# ====================================================
def plot_compare(
    reference: Dataset,
    *others: Dataset,
    names: Sequence[str] = (),
    labels: Sequence[str] | None = None,
    align: bool = True,
    with_scope: bool = True,
) -> Figure:
    """Overlay the same signals coming from several datasets.

    One subplot per signal; one line per dataset.

    Parameters
    ----------
    reference :
        The dataset defining the time base and the signal ordering.
    others :
        The datasets to compare against ``reference``.
    names :
        Signals to compare. Defaults to the signals common to all the
        datasets.
    labels :
        One legend label per dataset. Defaults to ``ds0``, ``ds1``, ...
    align :
        Resample every dataset on the common time base first.
    with_scope :
        Attach an interactive scope to the figure.
    """
    datasets = _check_datasets(reference, others)
    labels_ = _resolve_labels(datasets, labels)
    names_ = _common_names(datasets, names)
    datasets = _aligned(datasets, align)

    fig = plt.figure(
        constrained_layout=True, figsize=(10, 2.0 * len(names_) + 1)
    )

    if with_scope:
        subfigs = fig.subfigures(1, 2, width_ratios=[3.8, 1.2])
        host: Any = subfigs[0]
    else:
        host = fig

    axes = list(np.atleast_1d(host.subplots(len(names_), 1, sharex=True)))

    for ax, name in zip(axes, names_):
        for ds, label in zip(datasets, labels_):
            ds[name]._plot_standard(ax=ax, label=f"{name} ({label})")

        ax.set_ylabel(reference[name]._ylabel())
        ax.legend()
        ax.grid(True)

    if with_scope:
        panel_ax = subfigs[1].add_subplot()
        panel_ax.set_anchor("N")
        DatasetScope(fig, axes, panel_ax)

    return fig


def plot_spectrum_compare(
    reference: Dataset,
    *others: Dataset,
    names: Sequence[str] = (),
    labels: Sequence[str] | None = None,
    align: bool = True,
    with_scope: bool = True,
    xscale: str = "linear",
    yscale: str = "linear",
    mode: SpectrumMode = "psd_welch",
) -> Figure:
    """Overlay the spectra of the same signals from several datasets.

    ``mode="amplitude"`` is not supported here: use
    :meth:`dymoval.dataset.Dataset.plot_spectrum` on each dataset instead,
    since the magnitude/phase layout does not overlay meaningfully.
    """
    if mode not in SPECTRUM_MODES:
        raise ValueError(
            f"Invalid mode: {mode!r}. Allowed: {list(SPECTRUM_MODES)}"
        )

    if mode == "amplitude":
        raise ValueError(
            "mode='amplitude' is not supported by plot_spectrum_compare"
        )

    datasets = _check_datasets(reference, others)
    labels_ = _resolve_labels(datasets, labels)
    names_ = _common_names(datasets, names)
    datasets = _aligned(datasets, align)

    fig = plt.figure(
        constrained_layout=True, figsize=(10, 2.0 * len(names_) + 1)
    )

    if with_scope:
        subfigs = fig.subfigures(1, 2, width_ratios=[3.8, 1.2])
        host: Any = subfigs[0]
    else:
        host = fig

    axes = list(np.atleast_1d(host.subplots(len(names_), 1, sharex=True)))

    for ax, name in zip(axes, names_):
        for ds, label in zip(datasets, labels_):
            ds[name]._plot_spectrum_standard(
                ax=ax,
                xscale=xscale,
                yscale=yscale,
                mode=mode,
                label=f"{name} ({label})",
            )

        ax.legend()
        ax.grid(True)

    if with_scope:
        panel_ax = subfigs[1].add_subplot()
        panel_ax.set_anchor("N")
        SpectrumScope(fig, axes, panel_ax)

    return fig
