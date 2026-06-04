from __future__ import annotations

from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft
from scipy.signal import detrend
from .scope import SignalScope


@dataclass
class Signal:
    name: str
    values: np.ndarray
    time: np.ndarray | None = None
    unit: str | None = None
    time_unit: str | None = "s"

    def __post_init__(self) -> None:
        if not isinstance(self.values, np.ndarray):
            raise TypeError(f"{self.name}: values must be numpy array")

        if self.values.ndim != 1:
            raise ValueError(f"{self.name}: values must be 1D")

        if self.time is not None:
            if not isinstance(self.time, np.ndarray):
                raise TypeError(f"{self.name}: time must be numpy array")

            if len(self.time) != len(self.values):
                raise ValueError(
                    f"{self.name}: time and values length mismatch"
                )

    # ----------------------------
    # Convenience
    # ----------------------------
    def copy(self) -> "Signal":
        return Signal(
            name=self.name,
            values=self.values.copy(),
            time=None if self.time is None else self.time.copy(),
            unit=self.unit,
            time_unit=self.time_unit,
        )

    def detrend(self) -> "Signal":
        return Signal(
            name=self.name,
            values=detrend(self.values),
            time=self.time,
            unit=self.unit,
            time_unit=self.time_unit,
        )

    def fft(self) -> tuple[np.ndarray, np.ndarray]:
        if self.time is None:
            raise ValueError(f"{self.name}: time required for FFT")

        dt = np.mean(np.diff(self.time))
        freq = fft.fftfreq(len(self.values), d=dt)
        spec = np.abs(fft.fft(self.values))

        return freq, spec

    def resample(self, new_time: np.ndarray) -> "Signal":
        from scipy.interpolate import interp1d

        f = interp1d(
            self.time,
            self.values,
            bounds_error=False,
            fill_value=(self.values[0], self.values[-1]),
            assume_sorted=True,
        )

        return Signal(
            name=self.name,
            values=f(new_time),
            time=new_time,
            unit=self.unit,
            time_unit=self.time_unit,
        )

    def get_sampling_period(self) -> float:
        if self.time is None:
            raise ValueError(f"{self.name}: time not defined")

        dt = np.diff(self.time)

        if not np.all(dt > 0):
            raise ValueError(f"{self.name}: time must be increasing")

        return float(np.mean(dt))

    def _plot_standard(self, ax=None, **kwargs):

        if ax is None:
            fig, ax = plt.subplots()

        if self.time is None:
            ax.plot(self.values, **kwargs)
            ax.set_xlabel("samples")
        else:
            ax.plot(self.time, self.values, **kwargs)
            ax.set_xlabel(f"time [{self.time_unit}]")

        if self.unit:
            ax.set_ylabel(f"{self.name} [{self.unit}]")
        else:
            ax.set_ylabel(self.name)

        ax.grid(True)

        return ax

    def _plot_scope(self, **kwargs):

        fig = plt.figure(constrained_layout=True, figsize=(10, 5))
        subfigs = fig.subfigures(1, 2, width_ratios=[3.8, 1.2])

        # ---- main axis ----
        ax = subfigs[0].subplots()

        if self.time is None:
            ax.plot(self.values, label=self.name, **kwargs)
            ax.set_xlabel("samples")
        else:
            ax.plot(self.time, self.values, label=self.name, **kwargs)
            ax.set_xlabel(f"time [{self.time_unit}]")

        if self.unit:
            ax.set_ylabel(f"{self.name} [{self.unit}]")
        else:
            ax.set_ylabel(self.name)

        ax.grid(True)

        # ---- panel axis ----
        panel_ax = subfigs[1].add_subplot()
        panel_ax.axis("off")

        SignalScope(fig, ax, panel_ax, self)

        return fig

    def plot(self, ax=None, with_scope: bool = True, **kwargs):
        if with_scope:
            return self._plot_scope(**kwargs)
        return self._plot_standard(ax=ax, **kwargs)
