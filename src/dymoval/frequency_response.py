"""Nonparametric frequency-response models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from matplotlib.figure import Figure

from ._figure import _AX_HEIGHT, _AX_WIDTH, Layout, scope_subplots
from .signal import SCALES, Scale

__all__ = ["FrequencyResponse"]

_DB_FLOOR = np.finfo(float).tiny


@dataclass(frozen=True)
class FrequencyResponse:
    """Frequency-response data estimated from input/output measurements.

    ``response[k, i, j]`` is the complex response from input ``j`` to
    output ``i`` at ``frequency[k]``. Frequencies are angular frequencies
    in radians per dataset time unit.
    """

    frequency: np.ndarray
    response: np.ndarray
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    input_units: tuple[str | None, ...]
    output_units: tuple[str | None, ...]
    time_unit: str | None
    noise_spectrum: np.ndarray
    coherence: np.ndarray | None
    response_std: np.ndarray | None
    noise_spectrum_std: np.ndarray
    window_size: int

    def __post_init__(self) -> None:
        arrays = {
            "frequency": self.frequency,
            "response": self.response,
            "noise_spectrum": self.noise_spectrum,
            "noise_spectrum_std": self.noise_spectrum_std,
        }
        if self.coherence is not None:
            arrays["coherence"] = self.coherence
        if self.response_std is not None:
            arrays["response_std"] = self.response_std

        for name, value in arrays.items():
            if not isinstance(value, np.ndarray):
                raise TypeError(f"'{name}' must be a numpy array")

        if self.frequency.ndim != 1 or len(self.frequency) == 0:
            raise ValueError("'frequency' must be a non-empty 1-D array")

        if not np.all(np.isfinite(self.frequency)):
            raise ValueError("'frequency' must contain only finite values")

        if np.any(np.diff(self.frequency) <= 0.0):
            raise ValueError("'frequency' must be strictly increasing")

        expected = (
            len(self.frequency),
            len(self.output_names),
            len(self.input_names),
        )
        if not self.input_names or not self.output_names:
            raise ValueError(
                "At least one input and one output name are required"
            )

        if self.response.shape != expected:
            raise ValueError(
                f"'response' has shape {self.response.shape}, expected {expected}"
            )

        expected_noise = (
            len(self.frequency),
            len(self.output_names),
            len(self.output_names),
        )
        for name, value in (
            ("noise_spectrum", self.noise_spectrum),
            ("noise_spectrum_std", self.noise_spectrum_std),
        ):
            if value.shape != expected_noise:
                raise ValueError(
                    f"'{name}' has shape {value.shape}, "
                    f"expected {expected_noise}"
                )

        if self.coherence is not None and self.coherence.shape != (
            len(self.frequency),
        ):
            raise ValueError(
                f"'coherence' has shape {self.coherence.shape}, "
                f"expected {(len(self.frequency),)}"
            )

        if (
            self.response_std is not None
            and self.response_std.shape != expected
        ):
            raise ValueError(
                f"'response_std' has shape {self.response_std.shape}, "
                f"expected {expected}"
            )

        if len(self.input_units) != len(self.input_names):
            raise ValueError("'input_units' must match 'input_names'")

        if len(self.output_units) != len(self.output_names):
            raise ValueError("'output_units' must match 'output_names'")

        if (
            isinstance(self.window_size, bool)
            or not isinstance(self.window_size, int)
            or self.window_size < 1
        ):
            raise ValueError("'window_size' must be a positive integer")

    @property
    def frequency_hz(self) -> np.ndarray:
        """Return the frequency vector in cycles per dataset time unit."""
        return self.frequency / (2.0 * np.pi)

    def frf(
        self, frequencies: Sequence[float] | np.ndarray | None = None
    ) -> np.ndarray:
        """Return the response, optionally at new angular frequencies.

        New frequency points are linearly interpolated in the real and
        imaginary components. Values outside the estimated interval are
        ``nan``; frequency-response data cannot safely be extrapolated.
        """
        if frequencies is None:
            return self.response.copy()

        requested = np.atleast_1d(np.asarray(frequencies, dtype=float))
        if requested.ndim != 1:
            raise ValueError("'frequencies' must be a 1-D array")

        if not np.all(np.isfinite(requested)):
            raise ValueError("'frequencies' must contain only finite values")

        flat = self.response.reshape(len(self.frequency), -1)
        interpolated = np.empty((len(requested), flat.shape[1]), dtype=complex)

        for channel in range(flat.shape[1]):
            values = flat[:, channel]
            interpolated[:, channel] = np.interp(
                requested,
                self.frequency,
                values.real,
                left=np.nan,
                right=np.nan,
            ) + 1j * np.interp(
                requested,
                self.frequency,
                values.imag,
                left=np.nan,
                right=np.nan,
            )

        return interpolated.reshape(
            len(requested), len(self.output_names), len(self.input_names)
        )

    def plot(
        self,
        *,
        show_coherence: bool = True,
        xscale: Scale = "log",
        layout: Layout = "constrained",
        ax_height: float = _AX_HEIGHT,
        ax_width: float = _AX_WIDTH,
        **kwargs: Any,
    ) -> Figure:
        """Plot the estimated response as magnitude and phase.

        Every input/output channel is overlaid on a Bode-style magnitude and
        phase plot. For a SISO estimate, ``show_coherence=True`` adds a third
        subplot containing the magnitude-squared coherence.

        Parameters
        ----------
        show_coherence:
            Plot magnitude-squared coherence when it is available.
        xscale:
            Frequency-axis scale, either ``"linear"`` or ``"log"``.
        layout:
            Matplotlib layout engine.
        ax_height, ax_width:
            Height of each subplot and width of the figure, in inches.
        **kwargs:
            Forwarded to :meth:`matplotlib.axes.Axes.plot`.

        Returns
        -------
        matplotlib.figure.Figure
            The figure. This method never calls ``show()``.
        """
        if xscale not in SCALES:
            raise ValueError(
                f"Invalid xscale: {xscale!r}. Allowed: {list(SCALES)}"
            )

        plot_coherence = show_coherence and self.coherence is not None
        nrows = 3 if plot_coherence else 2
        fig, axes, _ = scope_subplots(
            nrows,
            with_scope=False,
            figsize=(ax_width, ax_height * nrows + 1),
            layout=layout,
            squeeze=False,
            sharex=True,
        )
        magnitude_ax, phase_ax = axes[:2]

        for output_index, output_name in enumerate(self.output_names):
            for input_index, input_name in enumerate(self.input_names):
                values = self.response[:, output_index, input_index]
                label = f"{output_name} / {input_name}"
                line_kwargs = dict(kwargs)
                line_kwargs.setdefault("label", label)

                magnitude = 20.0 * np.log10(
                    np.maximum(np.abs(values), _DB_FLOOR)
                )
                phase = np.full(len(values), np.nan)
                finite = np.isfinite(values)
                phase[finite] = np.rad2deg(np.unwrap(np.angle(values[finite])))

                magnitude_ax.plot(self.frequency, magnitude, **line_kwargs)
                phase_ax.plot(self.frequency, phase, **line_kwargs)

        time_unit = self.time_unit or "time unit"
        xlabel = f"Angular frequency [rad/{time_unit}]"
        magnitude_ax.set_ylabel("Magnitude [dB]")
        phase_ax.set_ylabel("Phase [deg]")

        for axis in axes:
            axis.set_xscale(xscale)
            axis.grid(True, which="both")

        if plot_coherence:
            coherence_ax = axes[2]
            assert self.coherence is not None
            coherence_kwargs = dict(kwargs)
            coherence_kwargs.pop("label", None)
            coherence_ax.plot(
                self.frequency,
                self.coherence,
                label="coherence",
                **coherence_kwargs,
            )
            coherence_ax.set_ylabel("Coherence")
            coherence_ax.set_ylim(0.0, 1.05)

        axes[-1].set_xlabel(xlabel)
        magnitude_ax.legend()

        return fig
