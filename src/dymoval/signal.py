"""The :class:`Signal` class.

``Signal`` is responsible for

- computation (detrend, resample, spectra, ...),
- *primitive* plotting, i.e. drawing **one** signal on a given axes.

Orchestration, grouping and layout belong to :class:`dymoval.dataset.Dataset`,
which reuses the private primitives (``_plot_standard``,
``_plot_spectrum_standard``, ...) and never calls the public plotting
methods internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, get_args

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.interpolate import interp1d
from scipy.signal import detrend as _detrend
from scipy.signal import welch

from .scope import (
    AmplitudeSpectrumScope,
    SignalScope,
    SpectrumScope,
    scope_subplots,
)

__all__ = ["Signal", "SPECTRUM_MODES"]

SpectrumMode = Literal["amplitude", "power", "psd", "psd_welch"]
SPECTRUM_MODES: tuple[str, ...] = get_args(SpectrumMode)

#: magnitudes below ``_PHASE_MASK_RATIO * max(magnitude)`` carry no
#: meaningful phase information and are masked out.
_PHASE_MASK_RATIO = 0.05

_EPS = 1e-12


@dataclass
class Signal:
    """A single, uniformly sampled signal."""

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

            if self.time.ndim != 1:
                raise ValueError(f"{self.name}: time must be 1D")

            if len(self.time) != len(self.values):
                raise ValueError(
                    f"{self.name}: time and values length mismatch"
                )

    def __len__(self) -> int:
        return len(self.values)

    # ================================================
    # Convenience
    # ================================================
    def _replace(self, **changes: Any) -> "Signal":
        """Return a new ``Signal``, overriding the given fields."""
        fields: dict[str, Any] = dict(
            name=self.name,
            values=self.values,
            time=self.time,
            unit=self.unit,
            time_unit=self.time_unit,
        )
        fields.update(changes)

        return Signal(**fields)

    def copy(self) -> "Signal":
        return self._replace(
            values=self.values.copy(),
            time=None if self.time is None else self.time.copy(),
        )

    # ================================================
    # Processing
    # ================================================
    def detrend(self) -> "Signal":
        """Remove the best straight-line fit from the signal."""
        return self._replace(values=_detrend(self.values))

    def remove_mean(self) -> "Signal":
        """Subtract the signal mean."""
        return self._replace(values=self.values - np.nanmean(self.values))

    def remove_constant(self, value: float) -> "Signal":
        """Subtract a user-defined constant."""
        return self._replace(values=self.values - value)

    def apply(
        self, func: Callable[[Any], Any], unit: str | None = None
    ) -> "Signal":
        """Apply ``func`` to the signal values.

        ``func`` is tried on the whole array first (vectorized) and falls
        back to an element-wise evaluation. ``unit`` overrides the signal
        unit, since a transformation usually changes it.
        """
        try:
            values = np.asarray(func(self.values), dtype=float)

            if values.shape != self.values.shape:
                raise ValueError
        except (ValueError, TypeError):
            values = np.array(
                [float(func(v)) for v in self.values], dtype=float
            )

        return self._replace(
            values=values, unit=self.unit if unit is None else unit
        )

    def low_pass_filter(self, cutoff: float) -> "Signal":
        """Filter the signal with a first-order low-pass filter.

        ``cutoff`` is expressed in Hz and must satisfy
        ``0 < cutoff < 1 / sampling_period``.
        """
        fs = 1.0 / self.get_sampling_period()

        if cutoff <= 0:
            raise ValueError(
                f"{self.name}: cut-off frequency must be positive."
            )

        if cutoff >= fs:
            raise ValueError(
                f"{self.name}: cut-off frequency must be smaller than the "
                f"sampling frequency ({fs})."
            )

        u = self.values
        y = np.empty_like(u, dtype=float)

        alpha = cutoff / fs
        y[0] = u[0]

        for k in range(len(u) - 1):
            y[k + 1] = (1.0 - alpha) * y[k] + alpha * u[k]

        return self._replace(values=y)

    def trim(
        self,
        tin: float | None = None,
        tout: float | None = None,
        shift_to_zero: bool = True,
    ) -> "Signal":
        """Keep the samples with ``tin <= time <= tout``.

        ``None`` means "from the beginning" / "until the end". When
        ``shift_to_zero`` is set the resulting time vector starts at 0.
        """
        if self.time is None:
            raise ValueError(f"{self.name}: time required for trimming")

        t_start = self.time[0] if tin is None else tin
        t_end = self.time[-1] if tout is None else tout

        if t_end < t_start:
            raise ValueError(f"{self.name}: tin must be smaller than tout")

        mask = (self.time >= t_start) & (self.time <= t_end)

        if not np.any(mask):
            raise ValueError(
                f"{self.name}: no samples in [{t_start}, {t_end}]"
            )

        time = self.time[mask]

        if shift_to_zero:
            time = time - time[0]

        return self._replace(values=self.values[mask], time=time)

    def resample(self, new_time: np.ndarray) -> "Signal":
        """Linearly resample the signal on ``new_time``."""
        if self.time is None:
            raise ValueError(f"{self.name}: time required for resampling")

        f = interp1d(
            self.time,
            self.values,
            bounds_error=False,
            fill_value=(self.values[0], self.values[-1]),
            assume_sorted=True,
        )

        return self._replace(values=f(new_time), time=new_time)

    def get_sampling_period(self) -> float:
        """Return the (uniform) sampling period."""
        if self.time is None:
            raise ValueError(f"{self.name}: time not defined")

        if len(self.time) < 2:
            raise ValueError(f"{self.name}: at least two samples required")

        dt = np.diff(self.time)

        if not np.all(dt > 0):
            raise ValueError(f"{self.name}: time must be increasing")

        if not np.allclose(dt, dt[0]):
            raise ValueError(f"{self.name}: non-uniform sampling")

        return float(np.mean(dt))

    # ================================================
    # Frequency domain
    # ================================================
    def _compute_fft(self) -> tuple[np.ndarray, np.ndarray, int, float]:
        dt = self.get_sampling_period()
        n = len(self.values)

        y = np.fft.rfft(self.values)
        freq = np.fft.rfftfreq(n, dt)

        return freq, y, n, dt

    def fft(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(frequency, complex one-sided spectrum)``."""
        freq, y, _, _ = self._compute_fft()
        return freq, y

    def spectrum(
        self, mode: SpectrumMode = "psd_welch"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(frequency, spectrum)`` for the requested ``mode``."""
        freq, spectrum, _ = self._compute_spectrum(mode)
        return freq, spectrum

    def _compute_spectrum(
        self, mode: SpectrumMode
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Return ``(freq, spectrum, phase_deg_or_None)``.

        The phase is returned for ``mode="amplitude"`` only. It is
        unwrapped, expressed in degrees, and masked (``nan``) wherever the
        magnitude is negligible, since the phase is meaningless there.
        """
        if mode not in SPECTRUM_MODES:
            raise ValueError(
                f"Invalid mode: {mode!r}. Allowed: {list(SPECTRUM_MODES)}"
            )

        freq, y, n, dt = self._compute_fft()
        fs = 1.0 / dt

        if mode == "amplitude":
            magnitude = np.abs(y)

            phase = np.unwrap(np.angle(y)) * 180.0 / np.pi
            threshold = _PHASE_MASK_RATIO * np.max(magnitude, initial=0.0)
            phase = np.where(magnitude > threshold, phase, np.nan)

            return freq, magnitude, phase

        if mode == "power":
            return freq, np.abs(y) ** 2, None

        if mode == "psd":
            return freq, (np.abs(y) ** 2) / (n * fs), None

        # psd_welch
        freq, spectrum = welch(
            self.values,
            fs=fs,
            window="hann",
            nperseg=min(256, len(self.values)),
        )

        return freq, spectrum, None

    # ------------------------------------------------
    # Labels
    # ------------------------------------------------
    def _spectrum_ylabel(self, mode: SpectrumMode, yscale: str) -> str:
        unit = self.unit or ""

        if yscale == "db":
            return "Amplitude [dB]" if mode == "amplitude" else "Power [dB]"

        if mode == "amplitude":
            return f"Amplitude [{unit}]" if unit else "Amplitude"

        if mode == "power":
            return f"Power [{unit}\u00b2]" if unit else "Power"

        return f"PSD [{unit}\u00b2/Hz]" if unit else "PSD"

    def _ylabel(self) -> str:
        return f"{self.name} [{self.unit}]" if self.unit else self.name

    @staticmethod
    def _to_db(spectrum: np.ndarray, mode: SpectrumMode) -> np.ndarray:
        factor = 20.0 if mode == "amplitude" else 10.0
        return np.asarray(factor * np.log10(np.maximum(spectrum, _EPS)))

    # ================================================
    # Time-domain plotting
    # ================================================
    def _plot_standard(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Draw the signal on ``ax`` (created if ``None``)."""
        if ax is None:
            _, ax = plt.subplots()

        kwargs.setdefault("label", self.name)

        if self.time is None:
            (line,) = ax.plot(self.values, **kwargs)
            ax.set_xlabel("samples")
        else:
            (line,) = ax.plot(self.time, self.values, **kwargs)
            ax.set_xlabel(f"time [{self.time_unit}]")

        # attach metadata so that scopes can recover units and names
        line._signal = self  # type: ignore[attr-defined]

        ax.set_ylabel(self._ylabel())
        ax.grid(True)

        return ax

    def _plot_scope(self, **kwargs: Any) -> Figure:
        fig, axes, panel_ax = scope_subplots(figsize=(10, 5))
        ax = axes[0]

        self._plot_standard(ax=ax, **kwargs)

        assert panel_ax is not None
        SignalScope(fig, ax, panel_ax, self)

        return fig

    def plot(
        self,
        ax: Axes | None = None,
        with_scope: bool = True,
        **kwargs: Any,
    ) -> Figure | Axes:
        """Plot the signal, optionally with an interactive scope."""
        if with_scope:
            return self._plot_scope(**kwargs)

        return self._plot_standard(ax=ax, **kwargs)

    # ================================================
    # Frequency-domain plotting
    # ================================================
    def _plot_spectrum_standard(
        self,
        ax: Axes | None = None,
        xscale: str = "linear",
        yscale: str = "linear",
        mode: SpectrumMode = "psd_welch",
        **kwargs: Any,
    ) -> Axes:
        """Draw the magnitude-like spectrum of the signal on ``ax``."""
        if ax is None:
            _, ax = plt.subplots()

        freq, spectrum, _ = self._compute_spectrum(mode)

        if yscale == "db":
            spectrum = self._to_db(spectrum, mode)

        kwargs.setdefault("label", self.name)

        (line,) = ax.plot(freq, spectrum, **kwargs)
        line._signal = self  # type: ignore[attr-defined]

        ax.set_xscale(xscale)

        if yscale == "log":
            ax.set_yscale("log")

        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(self._spectrum_ylabel(mode, yscale))
        ax.grid(True)

        return ax

    def _plot_phase_standard(
        self,
        ax: Axes,
        xscale: str = "linear",
        **kwargs: Any,
    ) -> Axes:
        """Draw the (masked, unwrapped) phase spectrum on ``ax``."""
        freq, _, phase = self._compute_spectrum("amplitude")

        assert phase is not None  # "amplitude" always provides the phase

        kwargs.setdefault("label", self.name)

        (line,) = ax.plot(freq, phase, **kwargs)
        line._signal = self  # type: ignore[attr-defined]

        ax.set_xscale(xscale)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Phase [deg]")
        ax.grid(True)

        return ax

    def _plot_spectrum_amplitude(
        self,
        mag_ax: Axes,
        phase_ax: Axes,
        xscale: str = "linear",
        yscale: str = "linear",
        **kwargs: Any,
    ) -> tuple[Axes, Axes]:
        """Draw magnitude *and* phase on the two given axes."""
        self._plot_spectrum_standard(
            ax=mag_ax,
            xscale=xscale,
            yscale=yscale,
            mode="amplitude",
            **kwargs,
        )
        self._plot_phase_standard(ax=phase_ax, xscale=xscale, **kwargs)

        mag_ax.set_xlabel("")

        return mag_ax, phase_ax

    def _plot_spectrum_amplitude_scope(
        self,
        xscale: str = "linear",
        yscale: str = "linear",
        **kwargs: Any,
    ) -> Figure:
        fig, axes, panel_ax = scope_subplots(
            2, 1, figsize=(10, 5), sharex=True
        )
        mag_ax, phase_ax = axes

        self._plot_spectrum_amplitude(
            mag_ax=mag_ax,
            phase_ax=phase_ax,
            xscale=xscale,
            yscale=yscale,
            **kwargs,
        )

        mag_ax.legend()

        assert panel_ax is not None
        AmplitudeSpectrumScope(fig, mag_ax, phase_ax, panel_ax)

        return fig

    def _plot_spectrum_scope(
        self,
        xscale: str = "linear",
        yscale: str = "linear",
        mode: SpectrumMode = "psd_welch",
        **kwargs: Any,
    ) -> Figure:
        if mode == "amplitude":
            return self._plot_spectrum_amplitude_scope(
                xscale=xscale, yscale=yscale, **kwargs
            )

        fig, axes, panel_ax = scope_subplots(figsize=(10, 4))
        ax = axes[0]

        self._plot_spectrum_standard(
            ax=ax, xscale=xscale, yscale=yscale, mode=mode, **kwargs
        )
        ax.legend()

        assert panel_ax is not None
        SpectrumScope(fig, ax, panel_ax)

        return fig

    def plot_spectrum(
        self,
        ax: Axes | None = None,
        with_scope: bool = True,
        xscale: str = "linear",
        yscale: str = "linear",
        mode: SpectrumMode = "psd_welch",
        **kwargs: Any,
    ) -> Figure | Axes:
        """Plot the spectrum of the signal.

        With ``mode="amplitude"`` both the magnitude and the phase are
        shown, stacked vertically.
        """
        if with_scope:
            return self._plot_spectrum_scope(
                xscale=xscale, yscale=yscale, mode=mode, **kwargs
            )

        if mode == "amplitude":
            if ax is not None:
                raise ValueError(
                    "mode='amplitude' draws magnitude and phase and "
                    "therefore creates its own figure: 'ax' is not allowed."
                )

            fig, (mag_ax, phase_ax) = plt.subplots(2, 1, sharex=True)

            self._plot_spectrum_amplitude(
                mag_ax=mag_ax,
                phase_ax=phase_ax,
                xscale=xscale,
                yscale=yscale,
                **kwargs,
            )
            mag_ax.legend()
            fig.tight_layout()

            return fig

        return self._plot_spectrum_standard(
            ax=ax, xscale=xscale, yscale=yscale, mode=mode, **kwargs
        )
