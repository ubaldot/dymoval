"""Interactive scopes attached to matplotlib figures.

Scopes implement the *interaction* layer of dymoval:

- :class:`BaseScope` holds everything that is common (panel, cursors,
  highlighting, reset, statistics).
- :class:`SignalScope` is used by ``Signal.plot(with_scope=True)``.
- :class:`DatasetScope` is used by ``Dataset.plot(with_scope=True)``.
- :class:`SpectrumScope` and :class:`AmplitudeSpectrumScope` are used by
  the ``plot_spectrum(with_scope=True)`` counterparts.

Design rules enforced here:

- every scope registers itself in ``fig._scopes`` so that it is not
  garbage collected;
- a panel axes owns **one** text artist, stored in
  ``panel_ax._shared_info_text``, shared by every scope writing to it;
- pressing ``r`` resets **all** scopes attached to the figure.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence, get_args

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = [
    "BaseScope",
    "SignalScope",
    "DatasetScope",
    "SpectrumScope",
    "AmplitudeSpectrumScope",
    "scope_subplots",
]

#: matplotlib layout engines accepted by the plotting API
Layout = Literal["constrained", "compressed", "tight", "none"]
LAYOUTS: tuple[str, ...] = get_args(Layout)

_HINT = "Click on a signal\n(press 'r' to reset)"

# ================================================
# Default figure geometry, in inches.
#
# Every plotting entry point in the package sizes its figure from one of
# these pairs, so that the look stays consistent and there is a single
# place to tune it. `ax_height`/`ax_width` arguments override them.
# ================================================

#: one axes per signal, stacked vertically (time plots, spectra, ...)
_AX_WIDTH = 10.0
_AX_HEIGHT = 2.0

#: coverage plots: a grid of narrow histograms
_COVERAGE_AX_WIDTH = 7.0
_COVERAGE_AX_HEIGHT = 1.8

#: square-ish grids of small axes (residuals correlations, x/y plots)
_GRID_AX_WIDTH = 4.445
_GRID_AX_HEIGHT = 1.8

#: standalone single-`Signal` figures
_SIGNAL_FIGSIZE = (10.0, 5.0)
_SIGNAL_SPECTRUM_FIGSIZE = (10.0, 4.0)

#: width ratio between the plotting area and the scope info panel
_PANEL_WIDTH_RATIOS = (3.8, 1.2)

# Half-width (in samples) of the window used to resolve which line was
# clicked when several lines overlap.
_PICK_WINDOW = 5


# Backwards-compatible module-level helpers retained for tests and
# external code that imported the old private helpers. These delegate
# the simple logic previously available at module scope.


def _fmt(value: float | None, unit: str = "") -> str:
    if value is None or np.isnan(value):
        text = "n/a"
    elif value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
        text = f"{value:.3e}"
    else:
        text = f"{value:.3f}"

    return f"{text} {unit}".rstrip()


def _line_name(line: Any) -> str:
    signal = getattr(line, "_signal", None)

    if signal is not None:
        return str(signal.name)

    return str(line.get_label() or "signal")


def _real_lines(ax: Axes) -> list[Any]:
    return [
        line
        for line in ax.get_lines()
        if not getattr(line, "_scope_artifact", False)
    ]


def _tag_artifact(artist: Any) -> Any:
    artist._scope_artifact = True
    return artist


def scope_subplots(
    nrows: int = 1,
    ncols: int = 1,
    *,
    with_scope: bool = True,
    figsize: tuple[float, float] | None = None,
    layout: Layout = "constrained",
    **kwargs: Any,
) -> tuple[Figure, list[Axes], Axes | None]:
    """Create a figure laid out for an interactive scope.

    When ``with_scope`` is set the figure is split into two subfigures: the
    plotting area on the left and the scope info panel on the right. The
    caller is left with attaching the scope itself, since only the caller
    knows which scope class and which axes grouping it needs.

    ``layout`` is the *matplotlib* figure layout engine; ``"none"`` disables
    automatic layout altogether.

    Returns ``(fig, axes, panel_ax)``, where ``axes`` is *flat* and
    ``panel_ax`` is ``None`` when ``with_scope`` is ``False``.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"'layout' must be one of {LAYOUTS}, got {layout!r}")

    fig = plt.figure(
        layout=None if layout == "none" else layout, figsize=figsize
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

    axes = list(np.atleast_1d(host.subplots(nrows, ncols, **kwargs)).ravel())

    return fig, axes, panel_ax


def _pick_time_interval(
    fig: Figure,
    tin: float,
    tout: float,
    title: str = "Trim the data.",
    verbosity: int = 0,
) -> tuple[float, float]:  # pragma: no cover
    """Let the user pick a time interval by zooming on ``fig``.

    The interval is read from the x-limits of the first axes, which the
    user adjusts with the matplotlib pan/zoom tools. The call blocks
    until the figure is closed and then returns the last limits seen,
    clipped to non-negative values. ``tin``/``tout`` are the values
    returned if the user never zooms.

    Note
    ----
    This cannot be covered by automated tests since it requires manual
    interaction.
    """
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
    except Exception as e:
        print(f"An error occurred {e}")
    finally:
        axes[0].remove_callback(cid)
        fig.clear()
        manager = fig.canvas.manager
        if manager is not None:
            manager.destroy()

    return selection["tin"], selection["tout"]


# ============================================================
# Base class (shared logic)
# ============================================================
class BaseScope:
    """Common scope machinery: panel, cursors, highlighting and reset."""

    #: labels used by the info panel for the abscissa/ordinate
    x_symbol = "t"
    y_symbol = "y"
    #: whether ``_update_display`` shows min/max/RMS statistics
    show_statistics = True

    def __init__(
        self,
        fig: Figure,
        axes: Axes | Sequence[Axes] | None,
        panel_ax: Axes,
    ) -> None:
        self.fig = fig
        self.axes: list[Axes] = self._as_axes_list(axes)
        self.panel_ax = panel_ax

        # selection state
        self.current_line: Any = None
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None

        self.clicks: list[tuple[float, float]] = []
        self.cursor_lines: list[Any] = []
        self.cursor_points: list[Any] = []

        self._register()
        self._init_panel()
        self._connect()

    # ====================================================
    # Lifetime
    # ====================================================
    def _register(self) -> None:
        """Keep the scope alive as long as the figure lives."""
        scopes = getattr(self.fig, "_scopes", None)

        if scopes is None:
            scopes = []
            self.fig._scopes = scopes  # type: ignore[attr-defined]

        scopes.append(self)

    @property
    def scopes(self) -> list["BaseScope"]:
        """All the scopes attached to the same figure."""
        return getattr(self.fig, "_scopes", [self])

    # ====================================================
    # Panel (one shared text artist per panel axes)
    # ====================================================
    def _init_panel(self) -> None:
        self.panel_ax.axis("off")

        if getattr(self.panel_ax, "_shared_info_text", None) is None:
            text = self.panel_ax.text(
                0.05,
                0.98,
                _HINT,
                va="top",
                transform=self.panel_ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="wheat"),
                wrap=True,
            )
            self.panel_ax._shared_info_text = text  # type: ignore[attr-defined]

    @property
    def info_text(self) -> Any:
        """The text artist shared by every scope of this panel."""
        return self.panel_ax._shared_info_text  # type: ignore[attr-defined]

    # ====================================================
    # Events
    # ====================================================
    def _connect(self) -> None:
        # The reset handler is global: connect it only once per figure.
        if not getattr(self.fig, "_scope_key_connected", False):
            self.fig.canvas.mpl_connect("key_press_event", self._on_key)
            self.fig._scope_key_connected = True  # type: ignore[attr-defined]

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    # -------------------------------
    # Small helpers moved here to keep the module tidy
    # -------------------------------
    def _as_axes_list(self, axes: Any) -> list[Axes]:
        """Normalize whatever matplotlib returned into ``list[Axes]``."""
        if axes is None:
            return []
        if isinstance(axes, np.ndarray):
            return list(axes.ravel())
        if isinstance(axes, (list, tuple)):
            return list(axes)
        return [axes]

    def _fmt(self, value: float | None, unit: str = "") -> str:
        """Format a number for the info panel."""
        if value is None or np.isnan(value):
            text = "n/a"
        elif value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
            text = f"{value:.3e}"
        else:
            text = f"{value:.3f}"

        return f"{text} {unit}".rstrip()

    def _line_name(self, line: Any) -> str:
        signal = getattr(line, "_signal", None)

        if signal is not None:
            return str(signal.name)

        return str(line.get_label() or "signal")

    def _real_lines(self, ax: Axes) -> list[Any]:
        """Lines of ``ax`` excluding cursors/markers drawn by a scope."""
        return [
            line
            for line in ax.get_lines()
            if not getattr(line, "_scope_artifact", False)
        ]

    def _tag_artifact(self, artist: Any) -> Any:
        """Mark an artist as scope-generated so that it is never selectable."""
        artist._scope_artifact = True
        return artist

    def _on_click(self, event: Any) -> None:
        if event.inaxes not in self.axes or event.xdata is None:
            return

        line = self._pick_line(event)

        if line is not None:
            self._process_click(line, event.xdata)

    def _pick_line(self, event: Any) -> Any:
        """Return the line closest to the click, or ``None``."""
        ax = event.inaxes
        x_click = event.xdata
        y_click = event.ydata

        best_line = None
        best_dist = np.inf

        for line in self._real_lines(ax):
            xdata = np.asarray(line.get_xdata(), dtype=float)
            ydata = np.asarray(line.get_ydata(), dtype=float)

            if xdata.size == 0:
                continue

            idx0 = int(np.argmin(np.abs(xdata - x_click)))

            i_min = max(0, idx0 - _PICK_WINDOW)
            i_max = min(len(xdata), idx0 + _PICK_WINDOW + 1)

            # Distances are computed in display coordinates so that very
            # different x/y scales do not bias the selection.
            pts = ax.transData.transform(
                np.column_stack([xdata[i_min:i_max], ydata[i_min:i_max]])
            )
            click_pt = ax.transData.transform([[x_click, y_click]])

            dists = np.hypot(
                pts[:, 0] - click_pt[0, 0], pts[:, 1] - click_pt[0, 1]
            )

            if np.all(np.isnan(dists)):
                continue

            dist = float(np.nanmin(dists))

            if dist < best_dist:
                best_dist = dist
                best_line = line

        return best_line

    # ====================================================
    # Shared click processing (CORE REUSE)
    # ====================================================
    def _process_click(self, line: Any, x_click: float) -> None:
        xdata = np.asarray(line.get_xdata(), dtype=float)
        ydata = np.asarray(line.get_ydata(), dtype=float)

        if xdata.size == 0:
            return

        idx = int(np.argmin(np.abs(xdata - x_click)))

        x_sel = float(xdata[idx])
        y_sel = float(ydata[idx])

        self.current_line = line
        self.x = xdata
        self.y = ydata

        self._highlight(line)
        self._add_cursor(line, x_sel, y_sel, idx)

        self.clicks.append((x_sel, y_sel))

        # keep the last two selections only
        while len(self.clicks) > 2:
            self.clicks.pop(0)

        self._update_display()
        self.fig.canvas.draw_idle()

    def _add_cursor(
        self, line: Any, x_sel: float, y_sel: float, idx: int
    ) -> None:
        """Draw the cursor for the selected point (overridable)."""
        color = line.get_color()
        ax = line.axes

        vline = self._tag_artifact(
            ax.axvline(x_sel, linestyle="--", color=color)
        )
        (point,) = ax.plot(x_sel, y_sel, marker="o", color=color)
        self._tag_artifact(point)

        self.cursor_lines.append(vline)
        self.cursor_points.append(point)

        self._trim_cursors(keep=2)

    def _trim_cursors(self, keep: int) -> None:
        while len(self.cursor_lines) > keep:
            self.cursor_lines.pop(0).remove()

        while len(self.cursor_points) > keep:
            self.cursor_points.pop(0).remove()

    # ----------------------------------------------------
    # Highlighting
    # ----------------------------------------------------
    def _baseline_linewidths(self) -> dict[Any, float]:
        """Original line widths, captured once per figure."""
        widths = getattr(self.fig, "_scope_linewidths", None)

        if widths is None:
            widths = {}
            self.fig._scope_linewidths = widths  # type: ignore[attr-defined]

        for ax in self.fig.axes:
            for line in self._real_lines(ax):
                widths.setdefault(line, line.get_linewidth())

        return widths

    def _highlight(self, line: Any) -> None:
        for other, width in self._baseline_linewidths().items():
            other.set_linewidth(3 if other is line else width)

    def _clear_highlight(self) -> None:
        for line, width in self._baseline_linewidths().items():
            line.set_linewidth(width)

    # ====================================================
    # Reset
    # ====================================================
    def _on_key(self, event: Any) -> None:
        if event.key != "r":
            return

        for scope in list(self.scopes):
            scope.reset(redraw=False)

        self.info_text.set_text(f"Reset\n{_HINT}")
        self.fig.canvas.draw_idle()

    def reset(self, redraw: bool = True) -> None:
        """Clear cursors, selections and highlighting of this scope."""
        for artist in self.cursor_lines + self.cursor_points:
            artist.remove()

        self.cursor_lines.clear()
        self.cursor_points.clear()
        self.clicks.clear()

        self.current_line = None
        self.x = None
        self.y = None

        self._clear_highlight()

        if redraw:
            self.info_text.set_text(f"Reset\n{_HINT}")
            self.fig.canvas.draw_idle()

    # ====================================================
    # Computation
    # ====================================================
    def _compute_stats(
        self, x1: float, x2: float
    ) -> tuple[float, float, float]:
        if self.x is None or self.y is None:
            return np.nan, np.nan, np.nan

        xmin, xmax = sorted([x1, x2])
        mask = (self.x >= xmin) & (self.x <= xmax)

        y_win = self.y[mask]

        if y_win.size == 0 or np.all(np.isnan(y_win)):
            return np.nan, np.nan, np.nan

        return (
            float(np.nanmin(y_win)),
            float(np.nanmax(y_win)),
            float(np.sqrt(np.nanmean(y_win**2))),
        )

    # ====================================================
    # Display
    # ====================================================
    def _label(self) -> str:
        if self.current_line is None:
            return "signal"

        return self._line_name(self.current_line)

    def _units(self) -> tuple[str, str]:
        """Return ``(x_unit, y_unit)`` of the current selection."""
        signal = getattr(self.current_line, "_signal", None)

        if signal is None:
            return "s", ""

        return (signal.time_unit or ""), (signal.unit or "")

    def _update_display(self) -> None:
        if self.current_line is None:
            return

        x_unit, y_unit = self._units()
        xs, ys = self.x_symbol, self.y_symbol

        header = f"{self._label()}\n\n"

        if len(self.clicks) == 1:
            x1, y1 = self.clicks[0]

            self.info_text.set_text(
                f"{header}"
                f"{xs}1 = {self._fmt(x1, x_unit)}\n"
                f"{ys}1 = {self._fmt(y1, y_unit)}\n\n"
                f"Select second point\n"
                f"(press 'r' to reset)"
            )
            return

        if len(self.clicks) < 2:
            return

        (x1, y1), (x2, y2) = self.clicks

        body = (
            f"{header}"
            f"{xs}1 = {self._fmt(x1, x_unit)}\n"
            f"{xs}2 = {self._fmt(x2, x_unit)}\n\n"
            f"{ys}1 = {self._fmt(y1, y_unit)}\n"
            f"{ys}2 = {self._fmt(y2, y_unit)}\n\n"
            f"\u0394{xs} = {self._fmt(x2 - x1, x_unit)}\n"
            f"\u0394{ys} = {self._fmt(y2 - y1, y_unit)}\n\n"
        )

        if self.show_statistics:
            ymin, ymax, rms = self._compute_stats(x1, x2)
            body += (
                f"min = {self._fmt(ymin, y_unit)}\n"
                f"max = {self._fmt(ymax, y_unit)}\n"
                f"RMS = {self._fmt(rms, y_unit)}\n\n"
            )

        self.info_text.set_text(body + "(press 'r' to reset)")


# ============================================================
# Single-signal scope
# ============================================================
class SignalScope(BaseScope):
    """Scope bound to a single axes holding a single signal."""

    def __init__(
        self,
        fig: Figure,
        ax: Axes,
        panel_ax: Axes,
        signal: Any = None,
    ) -> None:
        self.signal = signal
        super().__init__(fig, ax, panel_ax)

    def _label(self) -> str:
        if self.signal is not None:
            return str(self.signal.name)
        return super()._label()

    def _units(self) -> tuple[str, str]:
        if self.signal is not None:
            return (self.signal.time_unit or ""), (self.signal.unit or "")
        return super()._units()


# ============================================================
# Multi-signal Dataset scope
# ============================================================
class DatasetScope(BaseScope):
    """Scope spanning several axes, one per plot group."""


# ============================================================
# Spectrum scopes
# ============================================================
class SpectrumScope(DatasetScope):
    """Frequency-domain scope (magnitude-like spectra)."""

    x_symbol = "f"
    y_symbol = "A"
    show_statistics = False

    def _units(self) -> tuple[str, str]:
        signal = getattr(self.current_line, "_signal", None)
        y_unit = "" if signal is None else (signal.unit or "")
        return "Hz", y_unit


class AmplitudeSpectrumScope(SpectrumScope):
    """Scope linking a magnitude axes with its phase axes.

    A click selects a frequency: the cursor is mirrored on both axes and
    the panel reports magnitude *and* phase at that frequency.
    """

    def __init__(
        self,
        fig: Figure,
        mag_ax: Axes,
        phase_ax: Axes,
        panel_ax: Axes,
    ) -> None:
        self.mag_ax = mag_ax
        self.phase_ax = phase_ax
        self.phases: list[float] = []

        super().__init__(fig, [mag_ax, phase_ax], panel_ax)

    # ----------------------------------------------------
    # Selection
    # ----------------------------------------------------
    def _phase_line_for(self, line: Any) -> Any:
        """Phase line matching the clicked magnitude line (by name)."""
        name = self._line_name(line)
        candidates = self._real_lines(self.phase_ax)

        for phase_line in candidates:
            if self._line_name(phase_line) == name:
                return phase_line

        return candidates[0] if candidates else None

    def _process_click(self, line: Any, x_click: float) -> None:
        # Always drive the selection from the magnitude axes so that both
        # cursors refer to the very same frequency.
        if line.axes is self.phase_ax:
            mag_lines = self._real_lines(self.mag_ax)

            if not mag_lines:
                return

            name = self._line_name(line)
            line = next(
                (m for m in mag_lines if self._line_name(m) == name),
                mag_lines[0],
            )

        super()._process_click(line, x_click)

    def _add_cursor(
        self, line: Any, x_sel: float, y_sel: float, idx: int
    ) -> None:
        color = line.get_color()

        vline_mag = self.mag_ax.axvline(x_sel, linestyle="--", color=color)
        (point_mag,) = self.mag_ax.plot(x_sel, y_sel, marker="o", color=color)
        vline_ph = self.phase_ax.axvline(x_sel, linestyle="--", color=color)

        points = [point_mag]
        phase_value = np.nan
        phase_line = self._phase_line_for(line)

        if phase_line is not None:
            phase_data = np.asarray(phase_line.get_ydata(), dtype=float)

            if idx < phase_data.size:
                phase_value = float(phase_data[idx])

            if not np.isnan(phase_value):
                (point_ph,) = self.phase_ax.plot(
                    x_sel, phase_value, marker="o", color=color
                )
                points.append(point_ph)

        for artist in [vline_mag, vline_ph, *points]:
            self._tag_artifact(artist)

        self.cursor_lines.extend([vline_mag, vline_ph])
        self.cursor_points.extend(points)

        self.phases.append(phase_value)

        while len(self.phases) > 2:
            self.phases.pop(0)

        # two cursors per selection, at most two selections
        self._trim_cursors(keep=4)

    def reset(self, redraw: bool = True) -> None:
        """Clear spectrum selections and reset the shared scope state."""
        self.phases.clear()
        super().reset(redraw=redraw)

    # ----------------------------------------------------
    # Display
    # ----------------------------------------------------
    def _update_display(self) -> None:
        if self.current_line is None:
            return

        _, y_unit = self._units()
        header = f"{self._label()}\n\n"

        if len(self.clicks) == 1:
            f1, m1 = self.clicks[0]
            p1 = self.phases[0] if self.phases else np.nan

            self.info_text.set_text(
                f"{header}"
                f"f1 = {self._fmt(f1, 'Hz')}\n"
                f"|A1| = {self._fmt(m1, y_unit)}\n"
                f"\u2220A1 = {self._fmt(p1, 'deg')}\n\n"
                f"Select second point\n"
                f"(press 'r' to reset)"
            )
            return

        if len(self.clicks) < 2:
            return

        (f1, m1), (f2, m2) = self.clicks
        p1, p2 = (self.phases + [np.nan, np.nan])[:2]

        self.info_text.set_text(
            f"{header}"
            f"f1 = {self._fmt(f1, 'Hz')}\n"
            f"f2 = {self._fmt(f2, 'Hz')}\n\n"
            f"|A1| = {self._fmt(m1, y_unit)}\n"
            f"|A2| = {self._fmt(m2, y_unit)}\n\n"
            f"\u2220A1 = {self._fmt(p1, 'deg')}\n"
            f"\u2220A2 = {self._fmt(p2, 'deg')}\n\n"
            f"\u0394f = {self._fmt(f2 - f1, 'Hz')}\n"
            f"\u0394|A| = {self._fmt(m2 - m1, y_unit)}\n"
            f"\u0394\u2220A = {self._fmt(p2 - p1, 'deg')}\n\n"
            f"(press 'r' to reset)"
        )
