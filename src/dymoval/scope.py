from __future__ import annotations

import numpy as np


# ============================================================
# Lightweight interactive scope for Signal
# ============================================================
class SignalScope:
    def __init__(self, fig, ax, panel_ax, signal: "Signal"):
        self.fig = fig
        self.ax = ax
        self.panel_ax = panel_ax
        self.signal = signal

        self.current_line = None
        self.x = None
        self.y = None

        self.original_linewidths = {}
        self.cursor_points = []
        # prevent garbage collection
        setattr(fig, "_signal_scope", self)

        # data
        self.x = (
            signal.time
            if signal.time is not None
            else np.arange(len(signal.values))
        )
        self.y = signal.values

        # state
        self.clicks: list[tuple[float, float]] = []
        self.cursor_lines = []

        # UI
        self._init_panel()
        self._connect()

    # ====================================================
    # Panel
    # ====================================================
    def _init_panel(self):
        self.panel_ax.axis("off")

        self.info_text = self.panel_ax.text(
            0.05,
            0.95,
            f"{self.signal.name}\n Interactive Scope ",
            va="top",
            transform=self.panel_ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat"),
        )

    # ====================================================
    # Events
    # ====================================================
    def _connect(self):
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ====================================================
    # Click handling
    # ====================================================
    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return

        x_click = event.xdata

        best_line = None
        best_idx = None
        best_dist = np.inf

        # ---------------------------------------------
        # Find closest line + sample
        # ---------------------------------------------
        for line in self.ax.get_lines():
            xdata = np.asarray(line.get_xdata())

            idx = np.argmin(np.abs(xdata - x_click))
            dist = abs(xdata[idx] - x_click)

            if dist < best_dist:
                best_dist = dist
                best_line = line
                best_idx = idx

        if best_line is None or best_idx is None:
            return

        # ---------------------------------------------
        # Extract selected point
        # ---------------------------------------------
        xdata = np.asarray(best_line.get_xdata())
        ydata = np.asarray(best_line.get_ydata())

        x_sel = xdata[best_idx]
        y_sel = ydata[best_idx]

        line_color = best_line.get_color()

        # store selected line + data
        self.current_line = best_line
        self.x = xdata
        self.y = ydata

        # ---------------------------------------------
        # Highlight selected line
        # ---------------------------------------------
        for line in self.ax.get_lines():
            if line not in self.original_linewidths:
                self.original_linewidths[line] = line.get_linewidth()

            if line is best_line:
                line.set_linewidth(3)  # highlight
            else:
                line.set_linewidth(1)  # dim others

        # ---------------------------------------------
        # Draw cursor (vertical line)
        # ---------------------------------------------
        vline = self.ax.axvline(
            x_sel,
            linestyle="--",
            color=line_color,
        )

        self.cursor_lines.append(vline)

        # ---------------------------------------------
        # Draw marker (round point)
        # ---------------------------------------------
        (point,) = self.ax.plot(
            x_sel,
            y_sel,
            marker="o",
            color=line_color,
        )

        self.cursor_points.append(point)

        self.clicks.append((x_sel, y_sel))

        # keep max 2
        if len(self.cursor_lines) > 2:
            self.cursor_lines.pop(0).remove()
            self.cursor_points.pop(0).remove()
            self.clicks.pop(0)

        # update panel text
        self._update_display()

        self.fig.canvas.draw_idle()

    # ====================================================
    # Key handling
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

        # restore original linewidths
        for line, lw in self.original_linewidths.items():
            line.set_linewidth(lw)

        self.info_text.set_text(
            f"{self.signal.name}\n\n Interactive Scope \n"
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
    # Display update
    # ====================================================
    def _update_display(self):
        time_unit = self.signal.time_unit or ""
        value_unit = self.signal.unit or ""

        # ---- first click only
        if len(self.clicks) == 1:
            t1, y1 = self.clicks[0]

            self.info_text.set_text(
                f"{self.signal.name}\n\n"
                f"t1 = {t1:.3f} {time_unit},\n "
                f"y1 = {y1:.3f} {value_unit}\n\n"
            )

            self.fig.canvas.draw_idle()
            return

        if len(self.clicks) < 2:
            return

        # ---- two points
        (t1, y1), (t2, y2) = self.clicks

        ymin, ymax, rms = self._compute_stats(t1, t2)

        dt = t2 - t1
        dy = y2 - y1

        self.info_text.set_text(
            f"{self.signal.name}\n\n"
            f"t1 = {t1:.3f} {time_unit},\n"
            f"t2 = {t2:.3f} {time_unit},\n\n"
            f"y1 = {y1:.3f} {value_unit}\n"
            f"y2 = {y2:.3f} {value_unit}\n\n"
            f"Δt = {dt:.3f} {time_unit}\n"
            f"Δy = {dy:.3f} {value_unit}\n\n"
            f"min = {ymin:.3f}\n"
            f"max = {ymax:.3f}\n"
            f"rms = {rms:.3f}\n\n"
            f"press 'r' to reset"
        )

        self.fig.canvas.draw_idle()
