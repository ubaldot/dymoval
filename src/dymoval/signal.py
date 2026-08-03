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
    _SIGNAL_FIGSIZE,
    _SIGNAL_SPECTRUM_FIGSIZE,
    AmplitudeSpectrumScope,
    SignalScope,
    SpectrumScope,
    scope_subplots,
)

__all__ = ["Signal", "SPECTRUM_MODES", "SCALES", "SPECTRUM_SCALES"]

SpectrumMode = Literal["amplitude", "power", "psd", "psd_welch"]
SPECTRUM_MODES: tuple[str, ...] = get_args(SpectrumMode)

#: how a frequency axis may be scaled
Scale = Literal["linear", "log"]
SCALES: tuple[str, ...] = get_args(Scale)

#: how a spectrum magnitude axis may be scaled. ``"db"`` converts the
#: values to decibels, the others only change the axis scale.
SpectrumScale = Literal["linear", "log", "db"]
SPECTRUM_SCALES: tuple[str, ...] = get_args(SpectrumScale)


def _check_scale(xscale: Scale, yscale: SpectrumScale) -> None:
    """Raise if either scale is not a supported value."""
    if xscale not in SCALES:
        raise ValueError(
            f"Invalid xscale: {xscale!r}. Allowed: {list(SCALES)}"
        )

    if yscale not in SPECTRUM_SCALES:
        raise ValueError(
            f"Invalid yscale: {yscale!r}. Allowed: {list(SPECTRUM_SCALES)}"
        )


#: magnitudes below ``_PHASE_MASK_RATIO * max(magnitude)`` carry no
#: meaningful phase information and are masked out.
_PHASE_MASK_RATIO = 0.05

_EPS = 1e-12


def _check_mode(mode: SpectrumMode) -> None:
    """Raise if ``mode`` is not a supported spectrum mode."""
    if mode not in SPECTRUM_MODES:
        raise ValueError(
            f"Invalid mode: {mode!r}. Allowed: {list(SPECTRUM_MODES)}"
        )


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
    # Missing data
    # ================================================
    def has_nans(self) -> bool:
        """Whether the signal holds at least one ``NaN``."""
        return bool(np.isnan(self.values).any())

    def nan_intervals(self) -> list[tuple[float, float]]:
        """Return the ``[start, end]`` intervals where the signal is ``NaN``.

        Intervals are closed and expressed on the signal time vector, or in
        samples when ``time`` is ``None``. A ``NaN`` sitting alone gives a
        degenerate interval whose bounds are equal.

        Example
        -------
        >>> Signal("s", np.array([1.0, np.nan, np.nan, 4.0])).nan_intervals()
        [(1.0, 2.0)]
        """
        mask = np.isnan(self.values)

        if not mask.any():
            return []

        x = (
            np.arange(len(self.values), dtype=float)
            if self.time is None
            else self.time
        )

        # boundaries of the runs of True
        padded = np.concatenate(([False], mask, [False]))
        edges = np.flatnonzero(np.diff(padded.astype(np.int8)))

        return [
            (float(x[start]), float(x[stop - 1]))
            for start, stop in zip(edges[::2], edges[1::2])
        ]

    def remove_nans(
        self,
        fill: Literal["interpolate", "drop"] = "interpolate",
        **kwargs: Any,
    ) -> "Signal":
        """Get rid of the ``NaN`` samples.

        Parameters
        ----------
        fill:
            - ``"interpolate"`` replaces each ``NaN`` with a linear
              interpolation of its neighbours. Leading and trailing
              ``NaN``\\ s, having no neighbour on one side, are held
              constant at the closest valid sample.
            - ``"drop"`` removes the ``NaN`` samples altogether. The time
              vector then stops being uniformly sampled, so the result
              cannot be used to build a :class:`dymoval.dataset.Dataset`
              before being resampled.
        **kwargs:
            Forwarded to ``numpy.interp`` when interpolating.

        Note
        ----
        Interpolating is the safe choice for a signal that must keep its
        uniform sampling, which is what a ``Dataset`` requires.
        """
        if fill not in ("interpolate", "drop"):
            raise ValueError(
                f"{self.name}: 'fill' must be 'interpolate' or 'drop', "
                f"got {fill!r}"
            )

        mask = np.isnan(self.values)

        if not mask.any():
            return self.copy()

        if mask.all():
            raise ValueError(f"{self.name}: all the samples are NaN")

        if fill == "drop":
            return self._replace(
                values=self.values[~mask],
                time=None if self.time is None else self.time[~mask],
            )

        x = (
            np.arange(len(self.values), dtype=float)
            if self.time is None
            else self.time
        )

        values = self.values.copy()
        # np.interp holds the end values constant outside the valid range,
        # which is exactly the wanted behaviour for leading/trailing NaNs.
        values[mask] = np.interp(
            x[mask], x[~mask], self.values[~mask], **kwargs
        )

        return self._replace(values=values)

    # ================================================
    # Frequency domain
    # ================================================
    def _compute_fft(self) -> tuple[np.ndarray, np.ndarray, int, float]:
        dt = self.get_sampling_period()
        n = len(self.values)

        # Normalised by n so that Parseval's theorem holds, i.e. so that the
        # energy computed in the time domain equals the energy computed in
        # the frequency domain.
        y = np.fft.rfft(self.values) / n
        freq = np.fft.rfftfreq(n, dt)

        return freq, y, n, dt

    def _one_sided_scale(self, n: int, nbins: int) -> np.ndarray:
        """Weights folding the negative frequencies onto the positive ones.

        ``rfft`` only returns the non-negative half of a spectrum that is
        symmetric for a real-valued signal. Every bin therefore stands for
        itself *and* for its negative twin, and must be counted twice. The
        exceptions are DC, which has no twin, and, when ``n`` is even, the
        Nyquist bin, which is its own twin.
        """
        scale = np.full(nbins, 2.0)
        scale[0] = 1.0

        if n % 2 == 0 and nbins > 1:
            scale[-1] = 1.0

        return scale

    def fft(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(frequency, complex one-sided spectrum)``.

        The spectrum is normalised by the number of samples. It is *not*
        folded: use :py:meth:`spectrum` for a one-sided magnitude.
        """
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

        All the modes are one-sided: the negative frequencies are folded
        onto the positive ones, so that a sine of amplitude ``A`` peaks at
        ``A`` in ``amplitude`` mode, and so that ``power`` sums, and
        ``psd`` integrates, to the mean square of the signal.
        """
        _check_mode(mode)

        freq, y, n, dt = self._compute_fft()
        fs = 1.0 / dt
        fold = self._one_sided_scale(n, len(y))

        if mode == "amplitude":
            magnitude = np.abs(y) * fold

            phase = np.unwrap(np.angle(y)) * 180.0 / np.pi
            threshold = _PHASE_MASK_RATIO * np.max(magnitude, initial=0.0)
            phase = np.where(magnitude > threshold, phase, np.nan)

            return freq, magnitude, phase

        if mode == "power":
            return freq, (np.abs(y) ** 2) * fold, None

        if mode == "psd":
            # Divide by the width of a frequency bin, so that integrating
            # the result over the frequency axis gives back the power.
            delta_f = fs / n

            return freq, (np.abs(y) ** 2) * fold / delta_f, None

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
    def _spectrum_ylabel(
        self, mode: SpectrumMode, yscale: SpectrumScale
    ) -> str:
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
    def _draw(
        self,
        ax: Axes | None,
        x: np.ndarray | None,
        y: np.ndarray,
        xlabel: str,
        ylabel: str,
        **kwargs: Any,
    ) -> Axes:
        """Draw one curve of this signal on ``ax`` (created if ``None``).

        The single place where a ``Signal`` touches matplotlib: it labels
        the line with the signal name, tags it with a back-reference so
        that the scopes can recover names and units, and decorates the
        axes.
        """
        if ax is None:
            _, ax = plt.subplots()

        kwargs.setdefault("label", self.name)

        (line,) = (
            ax.plot(y, **kwargs) if x is None else ax.plot(x, y, **kwargs)
        )

        # attach metadata so that scopes can recover units and names
        line._signal = self  # type: ignore[attr-defined]

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)

        return ax

    def _shade_nans(
        self, ax: Axes, color: Any = None, alpha: float = 0.2, **kwargs: Any
    ) -> Axes:
        """Shade on ``ax`` the time intervals where the signal is ``NaN``.

        A *primitive*: it draws this signal only, on axes owned by the
        caller. ``color`` defaults to the color of the last line drawn.
        """
        for start, stop in self.nan_intervals():
            ax.axvspan(start, stop, color=color, alpha=alpha, **kwargs)

        return ax

    def _plot_standard(
        self, ax: Axes | None = None, shade_nans: bool = True, **kwargs: Any
    ) -> Axes:
        """Draw the signal on ``ax`` (created if ``None``).

        Gaps in the data are shaded with the color of the curve, unless
        ``shade_nans`` is unset.
        """
        ax = self._draw(
            ax,
            self.time,
            self.values,
            "samples" if self.time is None else f"time [{self.time_unit}]",
            self._ylabel(),
            **kwargs,
        )

        if shade_nans:
            self._shade_nans(ax, color=ax.get_lines()[-1].get_color())

        return ax

    def _plot_scope(self, **kwargs: Any) -> Figure:
        fig, axes, panel_ax = scope_subplots(figsize=_SIGNAL_FIGSIZE)
        ax = axes[0]

        self._plot_standard(ax=ax, **kwargs)

        assert panel_ax is not None
        SignalScope(fig, ax, panel_ax, self)

        return fig

    def plot(
        self,
        ax: Axes | None = None,
        with_scope: bool | None = None,
        **kwargs: Any,
    ) -> Figure | Axes:
        """Plot the signal, optionally with an interactive scope.

        Returns the created :class:`~matplotlib.figure.Figure`, or ``ax``
        itself when the caller provides one. A scope owns its whole
        figure, so passing ``ax`` turns it off unless ``with_scope`` is
        explicitly set, in which case it is an error.
        """
        with_scope = ax is None if with_scope is None else with_scope

        if with_scope:
            if ax is not None:
                raise ValueError(
                    f"{self.name}: a scope needs its own figure and "
                    "cannot draw on the passed 'ax'. Pass either 'ax' "
                    "or with_scope=True, not both."
                )

            return self._plot_scope(**kwargs)

        return self._plot_standard(ax=ax, **kwargs)

    def _plot_coverage_standard(
        self,
        ax: Axes | None = None,
        nbins: int = 100,
        **kwargs: Any,
    ) -> Axes:
        """Draw the histogram of the signal values on ``ax``."""
        if ax is None:
            _, ax = plt.subplots()

        kwargs.setdefault("label", self.name)

        ax.hist(self.values, bins=nbins, **kwargs)

        ax.set_xlabel(self._ylabel())
        ax.set_ylabel("count")
        ax.grid(True)

        return ax

    # ================================================
    # Frequency-domain plotting
    # ================================================
    def _plot_spectrum_standard(
        self,
        ax: Axes | None = None,
        xscale: Scale = "linear",
        yscale: SpectrumScale = "linear",
        mode: SpectrumMode = "psd_welch",
        **kwargs: Any,
    ) -> Axes:
        """Draw the magnitude-like spectrum of the signal on ``ax``."""
        _check_scale(xscale, yscale)

        freq, spectrum, _ = self._compute_spectrum(mode)

        if yscale == "db":
            spectrum = self._to_db(spectrum, mode)

        ax = self._draw(
            ax,
            freq,
            spectrum,
            "Frequency [Hz]",
            self._spectrum_ylabel(mode, yscale),
            **kwargs,
        )

        ax.set_xscale(xscale)

        if yscale == "log":
            ax.set_yscale("log")

        return ax

    def _plot_phase_standard(
        self,
        ax: Axes,
        xscale: Scale = "linear",
        **kwargs: Any,
    ) -> Axes:
        """Draw the (masked, unwrapped) phase spectrum on ``ax``."""
        freq, _, phase = self._compute_spectrum("amplitude")

        assert phase is not None  # "amplitude" always provides the phase

        ax = self._draw(
            ax, freq, phase, "Frequency [Hz]", "Phase [deg]", **kwargs
        )
        ax.set_xscale(xscale)

        return ax

    def _plot_spectrum_amplitude(
        self,
        mag_ax: Axes,
        phase_ax: Axes,
        xscale: Scale = "linear",
        yscale: SpectrumScale = "linear",
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

    def _plot_spectrum_amplitude_figure(
        self,
        with_scope: bool,
        xscale: Scale = "linear",
        yscale: SpectrumScale = "linear",
        **kwargs: Any,
    ) -> Figure:
        """Build the stacked magnitude/phase figure."""
        fig, axes, panel_ax = scope_subplots(
            2,
            1,
            with_scope=with_scope,
            # the scope panel needs the extra width; without it, stick to
            # the matplotlib default size
            figsize=_SIGNAL_FIGSIZE if with_scope else None,
            sharex=True,
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

        if panel_ax is not None:
            AmplitudeSpectrumScope(fig, mag_ax, phase_ax, panel_ax)

        return fig

    def _plot_spectrum_scope(
        self,
        xscale: Scale = "linear",
        yscale: SpectrumScale = "linear",
        mode: SpectrumMode = "psd_welch",
        **kwargs: Any,
    ) -> Figure:
        if mode == "amplitude":
            return self._plot_spectrum_amplitude_figure(
                True, xscale=xscale, yscale=yscale, **kwargs
            )

        fig, axes, panel_ax = scope_subplots(figsize=_SIGNAL_SPECTRUM_FIGSIZE)
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
        with_scope: bool | None = None,
        xscale: Scale = "linear",
        yscale: SpectrumScale = "linear",
        mode: SpectrumMode = "psd_welch",
        **kwargs: Any,
    ) -> Figure | Axes:
        """Plot the spectrum of the signal.

        With ``mode="amplitude"`` both the magnitude and the phase are
        shown, stacked vertically.

        As for :meth:`plot`, passing ``ax`` turns the scope off, since a
        scope owns its whole figure.
        """
        with_scope = ax is None if with_scope is None else with_scope

        if with_scope:
            if ax is not None:
                raise ValueError(
                    f"{self.name}: a scope needs its own figure and "
                    "cannot draw on the passed 'ax'. Pass either 'ax' "
                    "or with_scope=True, not both."
                )

            return self._plot_spectrum_scope(
                xscale=xscale, yscale=yscale, mode=mode, **kwargs
            )

        if mode == "amplitude":
            if ax is not None:
                raise ValueError(
                    "mode='amplitude' draws magnitude and phase and "
                    "therefore creates its own figure: 'ax' is not allowed."
                )

            return self._plot_spectrum_amplitude_figure(
                False, xscale=xscale, yscale=yscale, **kwargs
            )

        return self._plot_spectrum_standard(
            ax=ax, xscale=xscale, yscale=yscale, mode=mode, **kwargs
        )
