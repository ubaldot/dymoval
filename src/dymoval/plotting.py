"""Plotting helpers that operate on more than one object.

Single-object plotting lives on the objects themselves
(``Signal.plot``, ``Dataset.plot``, ``Dataset.plot_spectrum``, ...).
This module only adds what cannot belong to a single object, namely
plotting a bunch of loose signals and comparing datasets.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Sequence

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .dataset import _AX_HEIGHT, _AX_WIDTH, Dataset
from .scope import DatasetScope, Layout, SpectrumScope, scope_subplots
from .signal import Scale, Signal, SpectrumMode, SpectrumScale, _check_mode

__all__ = [
    "plot_signals",
    "plot_dataset",
    "plot_compare",
    "plot_coverage_compare",
    "plot_spectrum_compare",
]


def plot_signals(
    *signals: Signal | tuple[Signal, ...],
    with_scope: bool = True,
    layout: Layout = "constrained",
    ax_height: float = _AX_HEIGHT,
    ax_width: float = _AX_WIDTH,
    **kwargs: Any,
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
    layout :
        *matplotlib* layout engine.
    ax_height :
        Height, in inches, of each subplot.
    ax_width :
        Width, in inches, of the figure.
    **kwargs :
        Forwarded to ``matplotlib.axes.Axes.plot``.
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

    fig, axes, panel_ax = scope_subplots(
        len(groups),
        with_scope=with_scope,
        figsize=(ax_width, ax_height * len(groups) + 1),
        layout=layout,
    )

    for ax, group in zip(axes, groups):
        for sig in group:
            sig._plot_standard(ax=ax, **kwargs)

        ax.legend()
        ax.grid(True)

    if panel_ax is not None:
        DatasetScope(fig, axes, panel_ax)

    return fig


def plot_dataset(
    ds: Dataset,
    *names: str | tuple[str, ...],
    with_scope: bool = True,
    **kwargs: Any,
) -> Figure:
    """Functional alias of :meth:`dymoval.dataset.Dataset.plot`."""
    return ds.plot(*names, with_scope=with_scope, **kwargs)


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


def _compare(
    reference: Dataset,
    others: Sequence[Dataset],
    *,
    names: Sequence[str],
    labels: Sequence[str] | None,
    align: bool,
    with_scope: bool,
    draw: Callable[[Axes, Signal, str], Any],
    scope_cls: type[DatasetScope],
    finish: Callable[[Axes, str], Any] | None = None,
    layout: Layout = "constrained",
    ax_height: float = _AX_HEIGHT,
    ax_width: float = _AX_WIDTH,
) -> Figure:
    """Shared skeleton of the ``*_compare`` functions.

    ``draw(ax, signal, label)`` is the only per-flavour part, together
    with the scope class to attach and the optional per-axes
    ``finish(ax, name)`` touch-up.
    """
    datasets = _check_datasets(reference, others)
    labels_ = _resolve_labels(datasets, labels)
    names_ = _common_names(datasets, names)
    datasets = _aligned(datasets, align)

    fig, axes, panel_ax = scope_subplots(
        len(names_),
        with_scope=with_scope,
        figsize=(ax_width, ax_height * len(names_) + 1),
        layout=layout,
        sharex=True,
    )

    for ax, name in zip(axes, names_):
        for ds, label in zip(datasets, labels_):
            draw(ax, ds[name], f"{name} ({label})")

        if finish is not None:
            finish(ax, name)

        ax.legend()
        ax.grid(True)

    if panel_ax is not None:
        scope_cls(fig, axes, panel_ax)

    return fig


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
    layout: Layout = "constrained",
    ax_height: float = _AX_HEIGHT,
    ax_width: float = _AX_WIDTH,
    **kwargs: Any,
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
    layout :
        *matplotlib* layout engine.
    ax_height, ax_width :
        Height of each subplot and width of the figure, in inches.
    **kwargs :
        Forwarded to ``matplotlib.axes.Axes.plot``.
    """
    return _compare(
        reference,
        others,
        names=names,
        labels=labels,
        align=align,
        with_scope=with_scope,
        draw=lambda ax, sig, label: sig._plot_standard(
            ax=ax, label=label, **kwargs
        ),
        scope_cls=DatasetScope,
        # the reference dataset dictates the unit shown on the y axis
        finish=lambda ax, name: ax.set_ylabel(reference[name]._ylabel()),
        layout=layout,
        ax_height=ax_height,
        ax_width=ax_width,
    )


def plot_coverage_compare(
    reference: Dataset,
    *others: Dataset,
    names: Sequence[str] = (),
    labels: Sequence[str] | None = None,
    nbins: int = 100,
    alpha: float = 1.0,
    histtype: Literal["bar", "barstacked", "step", "stepfilled"] = "step",
    align: bool = False,
    layout: Layout = "constrained",
    ax_height: float = 1.8,
    ax_width: float = 7.0,
    **kwargs: Any,
) -> Figure:
    """Overlay the coverage histograms of the same signals from several
    datasets.

    One subplot per signal; one histogram per dataset. There is no scope
    here, exactly as for :meth:`dymoval.dataset.Dataset.plot_coverage`:
    the abscissa is a signal value, not time or frequency.

    Parameters
    ----------
    reference :
        The dataset defining the signal ordering.
    others :
        The datasets to compare against ``reference``.
    names :
        Signals to compare. Defaults to the signals common to all the
        datasets.
    labels :
        One legend label per dataset. Defaults to ``ds0``, ``ds1``, ...
    nbins :
        Number of histogram bins.
    alpha :
        Opacity of the histograms.
    histtype :
        Passed to ``matplotlib``. The default ``"step"`` keeps overlaid
        distributions readable.
    align :
        Resample every dataset on the common time base first. Off by
        default: a distribution does not need a shared time base.
    layout :
        *matplotlib* layout engine.
    ax_height, ax_width :
        Height of each subplot and width of the figure, in inches.
    **kwargs :
        Forwarded to ``matplotlib.axes.Axes.hist``.
    """
    datasets = _check_datasets(reference, others)
    labels_ = _resolve_labels(datasets, labels)
    names_ = _common_names(datasets, names)
    datasets = _aligned(datasets, align)

    fig, axes, _ = scope_subplots(
        len(names_),
        with_scope=False,
        figsize=(ax_width, ax_height * len(names_) + 1),
        layout=layout,
    )

    for ax, name in zip(axes, names_):
        for ds, label in zip(datasets, labels_):
            ds[name]._plot_coverage_standard(
                ax=ax,
                nbins=nbins,
                alpha=alpha,
                histtype=histtype,
                label=f"{name} ({label})",
                **kwargs,
            )

        ax.legend()

    return fig


def plot_spectrum_compare(
    reference: Dataset,
    *others: Dataset,
    names: Sequence[str] = (),
    labels: Sequence[str] | None = None,
    align: bool = True,
    with_scope: bool = True,
    xscale: Scale = "linear",
    yscale: SpectrumScale = "linear",
    mode: SpectrumMode = "psd_welch",
    layout: Layout = "constrained",
    ax_height: float = _AX_HEIGHT,
    ax_width: float = _AX_WIDTH,
    **kwargs: Any,
) -> Figure:
    """Overlay the spectra of the same signals from several datasets.

    ``mode="amplitude"`` is not supported here: use
    :meth:`dymoval.dataset.Dataset.plot_spectrum` on each dataset instead,
    since the magnitude/phase layout does not overlay meaningfully.

    ``layout``, ``ax_height`` and ``ax_width`` control the figure geometry
    as in :func:`plot_compare`, and ``**kwargs`` are forwarded to
    ``matplotlib.axes.Axes.plot``.
    """
    _check_mode(mode)

    if mode == "amplitude":
        raise ValueError(
            "mode='amplitude' is not supported by plot_spectrum_compare"
        )

    def draw(ax: Axes, sig: Signal, label: str) -> None:
        sig._plot_spectrum_standard(
            ax=ax,
            xscale=xscale,
            yscale=yscale,
            mode=mode,
            label=label,
            **kwargs,
        )

    return _compare(
        reference,
        others,
        names=names,
        labels=labels,
        align=align,
        with_scope=with_scope,
        draw=draw,
        scope_cls=SpectrumScope,
        layout=layout,
        ax_height=ax_height,
        ax_width=ax_width,
    )
