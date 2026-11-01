"""Shared matplotlib figure infrastructure.

This internal module owns layout types, default geometry, generic figure
construction, and interactive time-interval selection. Click-to-inspect scope
behavior belongs in :mod:`dymoval.scope`.
"""

from __future__ import annotations

from typing import Any, Literal, cast, get_args

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

Layout = Literal["constrained", "compressed", "tight", "none"]
LAYOUTS: tuple[str, ...] = get_args(Layout)

# The following constants are used by other modules
_AX_WIDTH = 10.0
_AX_HEIGHT = 2.0

_COVERAGE_AX_WIDTH = 7.0
_COVERAGE_AX_HEIGHT = 1.8

_GRID_AX_WIDTH = 4.445
_GRID_AX_HEIGHT = 1.8

_SIGNAL_FIGSIZE = (10.0, 5.0)
_SIGNAL_SPECTRUM_FIGSIZE = (10.0, 4.0)

_PANEL_WIDTH_RATIOS = (3.8, 1.2)


def _factorize(n: int) -> tuple[int, int]:
    """Choose a near-square subplot grid with room for ``n`` axes."""
    r = int(np.ceil(np.sqrt(n)))
    b = int(np.ceil(n / r))
    return r, b


def _subplot_size(
    nrows: int,
    ncols: int = 1,
    *,
    autosize: bool = False,
    ax_width: float | None = None,
    ax_height: float | None = None,
    defaults: tuple[float, float] = (_AX_WIDTH, _AX_HEIGHT),
    padding: float = 1.0,
) -> tuple[float, float] | None:
    """Resolve panel sizing, leaving unspecified dimensions to rc defaults."""
    if not autosize and ax_width is None and ax_height is None:
        return None
    rc_width, rc_height = plt.rcParams["figure.figsize"]
    width = (
        ncols * (ax_width if ax_width is not None else defaults[0])
        if autosize or ax_width is not None
        else rc_width
    )
    height = (
        nrows * (ax_height if ax_height is not None else defaults[1]) + padding
        if autosize or ax_height is not None
        else rc_height
    )
    return float(width), float(height)


def scope_subplots(
    nrows: int = 1,
    ncols: int = 1,
    *,
    with_scope: bool = True,
    figsize: tuple[float, float] | None = None,
    layout: Layout | None = None,
    autosize: bool = False,
    **kwargs: Any,
) -> tuple[Figure, list[Axes], Axes | None]:
    """Create axes with rc defaults, or panel-aware sizing when opted in.

    Explicit ``figsize`` overrides ``autosize``. ``layout=None`` inherits
    Matplotlib settings; ``layout="none"`` disables automatic layout.
    """

    if layout is not None and layout not in LAYOUTS:
        raise ValueError(f"'layout' must be one of {LAYOUTS}, got {layout!r}")

    fig = cast(
        Figure,
        plt.figure(
            layout=layout,
            figsize=(
                figsize
                if figsize is not None
                else _subplot_size(nrows, ncols, autosize=autosize)
            ),
        ),
    )

    if with_scope:
        subfigs = fig.subfigures(1, 2, width_ratios=list(_PANEL_WIDTH_RATIOS))
        assert isinstance(subfigs, np.ndarray)
        host: Any = subfigs[0]
        panel_ax: Axes | None = subfigs[1].add_subplot()
        assert panel_ax is not None
        panel_ax.set_anchor("N")
    else:
        host = fig
        panel_ax = None

    axes = cast(
        list[Axes],
        list(np.atleast_1d(host.subplots(nrows, ncols, **kwargs)).ravel()),
    )
    return fig, axes, panel_ax


def _pick_time_interval(
    fig: Figure,
    tin: float,
    tout: float,
    title: str = "Trim the data.",
    verbosity: int = 0,
) -> tuple[float, float]:  # pragma: no cover
    """Let the user select a time interval with matplotlib pan/zoom tools."""

    axes = fig.get_axes()
    if not axes:
        raise ValueError("Cannot pick a time interval on an empty figure.")

    selection = {"tin": float(tin), "tout": float(tout)}

    def update_time_interval(ax: Axes) -> None:
        left, right = ax.get_xlim()
        selection["tin"] = max(float(left), 0.0)
        selection["tout"] = max(float(right), 0.0)

        if verbosity != 0:
            print(
                f"Updated time interval: {selection['tin']} to "
                f"{selection['tout']}"
            )

    cid = axes[0].callbacks.connect("xlim_changed", update_time_interval)
    fig.suptitle(title)

    try:
        while fig in [plt.figure(num) for num in plt.get_fignums()]:
            plt.pause(0.1)
    finally:
        axes[0].remove_callback(cid)
        fig.clear()
        manager = fig.canvas.manager
        if manager is not None:
            manager.destroy()

    return selection["tin"], selection["tout"]
