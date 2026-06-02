from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import detrend

from mpl_measurements import InteractiveScope

from ..config import SIGNAL_KIND, Signal_type
from .signal import Signal


@dataclass
class Dataset:
    time: np.ndarray
    data: dict[str, np.ndarray]
    meta: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self._validate()

    # ----------------------------
    # Validation
    # ----------------------------
    def _validate(self) -> None:
        if self.time.ndim != 1:
            raise ValueError("time must be 1D")

        if not np.all(np.diff(self.time) > 0):
            raise ValueError("time must be strictly increasing")

        n = len(self.time)

        for name, values in self.data.items():
            if not isinstance(values, np.ndarray):
                raise TypeError(f"{name} is not numpy array")

            if values.ndim != 1:
                raise ValueError(f"{name} must be 1D")

            if len(values) != n:
                raise ValueError(f"{name} mismatch with time")

    # ----------------------------
    # Accessors
    # ----------------------------
    def signals(self) -> list[Signal]:
        return [
            Signal(name=name, values=values, time=self.time)
            for name, values in self.data.items()
        ]

    def get(self, name: str) -> Signal:
        return Signal(name=name, values=self.data[name], time=self.time)

    def copy(self) -> Self:
        return Dataset(
            time=self.time.copy(),
            data={k: v.copy() for k, v in self.data.items()},
            meta=deepcopy(self.meta),
        )

    # ----------------------------
    # Processing
    # ----------------------------
    def detrend(self) -> Self:
        return Dataset(
            self.time.copy(),
            {k: detrend(v) for k, v in self.data.items()},
            deepcopy(self.meta),
        )

    def fft(self):
        from scipy import fft

        fs = 1.0 / np.mean(np.diff(self.time))
        result = {}

        for name, values in self.data.items():
            freq = fft.fftfreq(len(values), d=1 / fs)
            spec = np.abs(fft.fft(values))
            result[name] = (freq, spec)

        return result

    def resample(self, new_time: np.ndarray, kind: str = "linear") -> Self:
        new_data = {}

        for name, values in self.data.items():
            f = interp1d(
                self.time,
                values,
                kind=kind,
                bounds_error=False,
                fill_value=(values[0], values[-1]),
                assume_sorted=True,
            )
            new_data[name] = f(new_time)

        return Dataset(new_time, new_data, deepcopy(self.meta))

    def align(self, other: "Dataset", how: str = "intersection"):
        if how == "intersection":
            t_min = max(self.time[0], other.time[0])
            t_max = min(self.time[-1], other.time[-1])
            new_time = self.time[(self.time >= t_min) & (self.time <= t_max)]

        elif how == "union":
            new_time = np.unique(np.concatenate([self.time, other.time]))

        else:
            raise ValueError("Invalid align mode")

        return self.resample(new_time), other.resample(new_time)

    # ----------------------------
    # Grouping
    # ----------------------------
    def select(self, kind: Signal_type) -> "Dataset":
        selected = {
            k: v for k, v in self.data.items() if SIGNAL_KIND.get(k) == kind
        }
        return Dataset(self.time, selected, deepcopy(self.meta))

    def inputs(self) -> "Dataset":
        return self.select("input")

    def outputs(self) -> "Dataset":
        return self.select("output")

    # ----------------------------
    # Pipeline
    # ----------------------------
    def pipe(self, func, *args, **kwargs) -> "Dataset":
        result = func(self, *args, **kwargs)
        if not isinstance(result, Dataset):
            raise TypeError("Pipeline must return Dataset")
        return result

    # ----------------------------
    # Interactive
    # ----------------------------
    def scope(self, *names: str) -> InteractiveScope:
        if not names:
            names = tuple(self.data.keys())

        signals = [Signal(name, self.data[name], self.time) for name in names]

        return InteractiveScope(signals)

    def scope_compare(self, other: "Dataset", *names: str):
        ds1, ds2 = self.align(other)

        if not names:
            names = tuple(set(ds1.data) & set(ds2.data))

        signals = []

        for name in names:
            signals.append(Signal(f"{name}_ref", ds1.data[name], ds1.time))
            signals.append(Signal(f"{name}_cmp", ds2.data[name], ds2.time))

        return InteractiveScope(signals)
