from __future__ import annotations

import numpy as np


# ============================================================
# Base class (shared logic)
# ============================================================
class BaseScope:
    def __init__(self, fig, panel_ax):
        self.fig = fig
        self.panel_ax = panel_ax

        # prevent garbage collection
        setattr(fig, "_scope", self)

        # state
        self.current_line = None
        self.x = None
        self.y = None

        self.clicks: list[tuple[float, float]] = []
        self.cursor_lines = []
        self.cursor_points = []
        self.original_linewidths = {}

        # UI + events
        self._init_panel()
        self._connect()

    # ====================================================
    # Panel
    # ====================================================
    def _init_panel(self):
        self.panel_ax.axis("off")

        self.info_text = self.panel_ax.text(
            0.05,
            0.98,
            "Click on a signal\n(press 'r' to reset)",
            va="top",
            transform=self.panel_ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat"),
            wrap=True,
        )

    # ====================================================
    # Events
    # ====================================================
    def _connect(self):
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ====================================================
    # Shared click processing (CORE REUSE)
    # ====================================================
    def _process_click(self, line, x_click):
        xdata = np.asarray(line.get_xdata())
        ydata = np.asarray(line.get_ydata())

        idx = np.argmin(np.abs(xdata - x_click))

        x_sel = xdata[idx]
        y_sel = ydata[idx]

        color = line.get_color()
        ax = line.axes

        # store current selection
        self.current_line = line
        self.x = xdata
        self.y = ydata

        # ---------------------------------------------
        # Highlight line
        # ---------------------------------------------
        for ax_ in self.fig.axes:
            for l in ax_.get_lines():
                if l not in self.original_linewidths:
                    self.original_linewidths[l] = l.get_linewidth()

                if l is line:
                    l.set_linewidth(3)
                else:
                    l.set_linewidth(1)

        # ---------------------------------------------
        # Draw cursor + marker
        # ---------------------------------------------
        vline = ax.axvline(
            x_sel,
            linestyle="--",
            color=color,
        )

        (point,) = ax.plot(
            x_sel,
            y_sel,
            marker="o",
            color=color,
        )

        self.cursor_lines.append(vline)
        self.cursor_points.append(point)

        self.clicks.append((x_sel, y_sel))

        # keep max 2
        if len(self.cursor_lines) > 2:
            self.cursor_lines.pop(0).remove()
            self.cursor_points.pop(0).remove()
            self.clicks.pop(0)

        self._update_display()
        self.fig.canvas.draw_idle()

    # ====================================================
    # Reset
    # ====================================================
    def _on_key(self, event):
        if event.key != "r":
            return

        for c in self.cursor_lines:
            c.remove()
        for p in self.cursor_points:
            p.remove()

        self.cursor_lines.clear()
        self.cursor_points.clear()
        self.clicks.clear()

        for line, lw in self.original_linewidths.items():
            line.set_linewidth(lw)

        self.info_text.set_text(
            "Reset\nClick on a signal\n(press 'r' to reset)"
        )

        self.fig.canvas.draw_idle()

    # ====================================================
    # Computation
    # ====================================================
    def _compute_stats(self, x1, x2):
        if self.x is None or self.y is None:
            return np.nan, np.nan, np.nan

        xmin, xmax = sorted([x1, x2])
        mask = (self.x >= xmin) & (self.x <= xmax)

        y_win = self.y[mask]

        if len(y_win) == 0:
            return np.nan, np.nan, np.nan

        return (
            np.min(y_win),
            np.max(y_win),
            np.sqrt(np.mean(y_win**2)),
        )

    # ====================================================
    # Display
    # ====================================================
    def _update_display(self):
        if self.current_line is None:
            return

        label = self.current_line.get_label() or "signal"

        # Get units
        time_unit = "s"
        value_unit = ""

        if hasattr(self.current_line, "_signal"):
            sig = self.current_line._signal

            time_unit = sig.time_unit or ""
            value_unit = sig.unit or ""

        if len(self.clicks) == 1:
            t1, y1 = self.clicks[0]

            self.info_text.set_text(
                f"{label}\n\n"
                f"t1 = {t1:.3f} {time_unit}\n"
                f"y1 = {y1:.3f} {value_unit}\n\n"
                f"Select second point\n"
                f"(press 'r' to reset)"
            )
            return

        if len(self.clicks) < 2:
            return

        (t1, y1), (t2, y2) = self.clicks

        ymin, ymax, rms = self._compute_stats(t1, t2)

        dt = t2 - t1
        dy = y2 - y1

        self.info_text.set_text(
            f"{label}\n\n"
            f"t1 = {t1:.3f} {time_unit}\n"
            f"t2 = {t2:.3f} {time_unit}\n\n"
            f"y1 = {y1:.3f} {value_unit}\n"
            f"y2 = {y2:.3f} {value_unit}\n\n"
            f"Δt = {dt:.3f} {time_unit}\n"
            f"Δy = {dy:.3f} {value_unit}\n\n"
            f"min = {ymin:.3f} {value_unit}\n"
            f"max = {ymax:.3f} {value_unit}\n"
            f"RMS = {rms:.3f} {value_unit}\n\n"
            f"(press 'r' to reset)"
        )


# ============================================================
# Single-signal scope
# ============================================================
class SignalScope(BaseScope):
    def __init__(self, fig, ax, panel_ax, signal):
        self.ax = ax
        self.signal = signal

        super().__init__(fig, panel_ax)

        # override units
        self.time_unit = signal.time_unit or ""
        self.value_unit = signal.unit or ""

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return

        lines = self.ax.get_lines()
        if not lines:
            return

        # only one signal → take first
        line = lines[0]

        self._process_click(line, event.xdata)

    # override display to include units
    def _update_display(self):
        if self.current_line is None:
            return

        label = self.signal.name

        time_unit = self.time_unit
        value_unit = self.value_unit

        if len(self.clicks) == 1:
            t1, y1 = self.clicks[0]

            self.info_text.set_text(
                f"{label}\n\n"
                f"t1 = {t1:.3f} {time_unit}\n"
                f"y1 = {y1:.3f} {value_unit}\n\n"
                f"Select second point\n"
                f"(press 'r' to reset)"
            )
            return

        if len(self.clicks) < 2:
            return

        (t1, y1), (t2, y2) = self.clicks

        ymin, ymax, rms = self._compute_stats(t1, t2)

        dt = t2 - t1
        dy = y2 - y1

        self.info_text.set_text(
            f"{label}\n\n"
            f"t1 = {t1:.3f} {time_unit}\n"
            f"t2 = {t2:.3f} {time_unit}\n\n"
            f"y1 = {y1:.3f} {value_unit}\n"
            f"y2 = {y2:.3f} {value_unit}\n\n"
            f"Δt = {dt:.3f} {time_unit}\n"
            f"Δy = {dy:.3f} {value_unit}\n\n"
            f"min = {ymin:.3f} {value_unit}\n"
            f"max = {ymax:.3f} {value_unit}\n"
            f"RMS = {rms:.3f} {value_unit}\n\n"
            f"(press 'r' to reset)"
        )


# ============================================================
# Multi-signal Dataset scope
# ============================================================
class DatasetScope(BaseScope):
    def __init__(self, fig, axes, panel_ax):
        # normalize axes → always list
        if isinstance(axes, np.ndarray):
            self.axes = list(axes.ravel())
        elif isinstance(axes, (list, tuple)):
            self.axes = list(axes)
        else:
            self.axes = [axes]

        super().__init__(fig, panel_ax)

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    def _on_click(self, event):
        if event.inaxes not in self.axes or event.xdata is None:
            return

        ax = event.inaxes
        x_click = event.xdata

        best_line = None
        best_dist = np.inf

        x_click = event.xdata
        y_click = event.ydata

        best_line = None
        best_idx = None
        best_dist = np.inf

        for line in ax.get_lines():
            xdata = np.asarray(line.get_xdata())
            ydata = np.asarray(line.get_ydata())
            # --- find closest index in X
            idx0 = np.argmin(np.abs(xdata - x_click))

            # --- define small window around idx
            window = 5  # you can tune this

            i_min = max(0, idx0 - window)
            i_max = min(len(xdata), idx0 + window)

            x_win = xdata[i_min:i_max]
            y_win = ydata[i_min:i_max]

            # --- compute full 2D distance
            dx = x_win - x_click
            dy = y_win - y_click
            dists = np.hypot(dx, dy)

            local_idx = np.argmin(dists)

            # map back to global index
            idx = i_min + local_idx

            x_sel = xdata[idx]
            y_sel = ydata[idx]

            dist = dists[local_idx]

            if dist < best_dist:
                best_dist = dist
                best_line = line
                best_idx = idx

        if best_line is None:
            return

        self._process_click(best_line, x_click)
