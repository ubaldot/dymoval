from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np
from scipy import fft
from scipy.signal import detrend

from ..config import Signal_type


@dataclass
class Signal:
    name: str
    values: np.ndarray
    time: np.ndarray | None = None
    unit: str | None = None
    kind: Signal_type = "time"

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

    def copy(self) -> "Signal":
        return Signal(
            name=self.name,
            values=self.values.copy(),
            time=None if self.time is None else self.time.copy(),
            unit=self.unit,
            kind=self.kind,
        )

    def detrend(self) -> "Signal":
        return Signal(
            name=self.name,
            values=detrend(self.values),
            time=self.time,
            unit=self.unit,
            kind=self.kind,
        )

    def fft(self) -> tuple[np.ndarray, np.ndarray]:
        if self.time is None:
            raise ValueError(f"{self.name}: time required for FFT")

        dt = np.mean(np.diff(self.time))
        freq = fft.fftfreq(len(self.values), d=dt)
        spec = np.abs(fft.fft(self.values))
        return freq, spec
