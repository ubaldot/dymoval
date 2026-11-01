"""Frequency-response estimation, results, and plotting.

FrequencyResponse stores SISO or MIMO responses and their diagnostics,
supports interpolation, channel selection, and Bode plotting.
Private numerical helpers implement the Blackman-Tukey estimator used by
Dataset.spa; dataset orchestration remains in dymoval.dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from matplotlib.figure import Figure

from ._figure import Layout, _subplot_size, scope_subplots
from .signal import SCALES, Scale
from .utils import _readonly_view

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
        self.validate()

    def validate(self) -> None:
        """Raise if the frequency-response data is internally inconsistent."""
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

        if self.coherence is not None and (
            len(self.input_names) != 1 or len(self.output_names) != 1
        ):
            raise ValueError("'coherence' is defined only for SISO responses")

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

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return ``(n_frequencies, n_outputs, n_inputs)``."""
        return self.response.shape

    @property
    def n_frequencies(self) -> int:
        """Number of estimated frequency points."""
        return len(self.frequency)

    @property
    def n_outputs(self) -> int:
        """Number of output channels."""
        return len(self.output_names)

    @property
    def n_inputs(self) -> int:
        """Number of input channels."""
        return len(self.input_names)

    @property
    def frequency_unit(self) -> str:
        """Angular-frequency unit used by ``frequency``."""
        return f"rad/{self.time_unit or 'time unit'}"

    @property
    def response_units(self) -> tuple[tuple[str | None, ...], ...]:
        """Return output/input units indexed as ``[output][input]``."""
        return tuple(
            tuple(
                None
                if output_unit is None or input_unit is None
                else f"{output_unit}/{input_unit}"
                for input_unit in self.input_units
            )
            for output_unit in self.output_units
        )

    @property
    def frequency_view(self) -> np.ndarray:
        """Return a zero-copy, read-only frequency view."""
        return _readonly_view(self.frequency)

    @property
    def response_view(self) -> np.ndarray:
        """Return a zero-copy, read-only response view."""
        return _readonly_view(self.response)

    def select(
        self,
        *,
        inputs: Sequence[str] | str | None = None,
        outputs: Sequence[str] | str | None = None,
    ) -> "FrequencyResponse":
        """Return a response containing only selected channels.

        Output-noise spectra remain those estimated while conditioning on the
        original full input set; channel selection does not re-estimate them.
        """

        def indices(
            selected: Sequence[str] | str | None,
            available: tuple[str, ...],
            kind: str,
        ) -> tuple[int, ...]:
            names = (
                available
                if selected is None
                else (
                    (selected,)
                    if isinstance(selected, str)
                    else tuple(selected)
                )
            )
            if not names:
                raise ValueError(f"At least one {kind} is required")
            missing = [name for name in names if name not in available]
            if missing:
                raise KeyError(
                    f"{kind.capitalize()} channel(s) {missing} not found. "
                    f"Available: {list(available)}"
                )
            if len(set(names)) != len(names):
                raise ValueError(
                    f"{kind.capitalize()} channels must be unique"
                )
            return tuple(available.index(name) for name in names)

        input_indices = indices(inputs, self.input_names, "input")
        output_indices = indices(outputs, self.output_names, "output")
        input_index_array = np.asarray(input_indices, dtype=int)
        output_index_array = np.asarray(output_indices, dtype=int)
        frequency_indices = np.arange(self.n_frequencies)
        response = self.response[
            np.ix_(
                frequency_indices,
                output_index_array,
                input_index_array,
            )
        ]
        response_std = (
            None
            if self.response_std is None
            else self.response_std[
                np.ix_(
                    frequency_indices,
                    output_index_array,
                    input_index_array,
                )
            ]
        )
        noise_indices = np.ix_(
            frequency_indices,
            output_index_array,
            output_index_array,
        )

        return FrequencyResponse(
            frequency=self.frequency.copy(),
            response=response,
            input_names=tuple(self.input_names[i] for i in input_indices),
            output_names=tuple(self.output_names[i] for i in output_indices),
            input_units=tuple(self.input_units[i] for i in input_indices),
            output_units=tuple(self.output_units[i] for i in output_indices),
            time_unit=self.time_unit,
            noise_spectrum=self.noise_spectrum[noise_indices],
            coherence=(
                self.coherence.copy()
                if self.coherence is not None
                and self.n_inputs == self.n_outputs == 1
                and len(input_indices) == len(output_indices) == 1
                else None
            ),
            response_std=response_std,
            noise_spectrum_std=self.noise_spectrum_std[noise_indices],
            window_size=self.window_size,
        )

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
        layout: Layout | None = None,
        ax_height: float | None = None,
        ax_width: float | None = None,
        autosize: bool = False,
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
            Matplotlib layout engine. ``None`` inherits rc settings.
        autosize:
            Use panel-aware sizing instead of rc defaults. Explicit dimensions
            take precedence.
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
            figsize=_subplot_size(
                nrows,
                autosize=autosize,
                ax_width=ax_width,
                ax_height=ax_height,
            ),
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


def _biased_covariance(
    x: np.ndarray, z: np.ndarray, max_lag: int
) -> np.ndarray:
    """Return biased cross-covariances for lags zero through ``max_lag``."""
    n = len(x)
    result = np.empty((max_lag + 1, x.shape[1], z.shape[1]))

    for lag in range(max_lag + 1):
        result[lag] = x[lag:].T @ z[: n - lag] / n

    return result


def _windowed_spectrum(
    positive: np.ndarray,
    negative: np.ndarray,
    window: np.ndarray,
    normalized_frequency: np.ndarray,
    sampling_period: float,
) -> np.ndarray:
    """Transform positive and negative covariance lags."""
    lags = np.arange(1, len(window))
    phase = np.exp(-1j * np.outer(normalized_frequency, lags))
    weighted_phase = phase * window[1:]

    result = np.broadcast_to(
        window[0] * positive[0],
        (len(normalized_frequency), *positive.shape[1:]),
    ).astype(complex, copy=True)
    result += np.einsum(
        "fl,lij->fij", weighted_phase, positive[1:], optimize=True
    )
    result += np.einsum(
        "fl,lij->fij",
        weighted_phase.conj(),
        negative[1:],
        optimize=True,
    )
    return sampling_period * result


def _clamp_psd(matrix: np.ndarray) -> np.ndarray:
    """Project a stack of Hermitian matrices onto the PSD cone."""
    result = np.empty_like(matrix)

    for index, value in enumerate(matrix):
        hermitian = (value + value.conj().T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
        result[index] = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ (
            eigenvectors.conj().T
        )

    return result


def _estimate_spa(
    inputs: np.ndarray,
    outputs: np.ndarray,
    *,
    sampling_period: float,
    frequencies: np.ndarray,
    window_size: int,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    input_units: tuple[str | None, ...],
    output_units: tuple[str | None, ...],
    time_unit: str | None,
) -> FrequencyResponse:
    """Estimate a Blackman-Tukey frequency response."""
    n = len(inputs)
    lags = np.arange(window_size + 1, dtype=float)
    window = 0.5 * (1.0 + np.cos(np.pi * lags / window_size))
    normalized_frequency = frequencies * sampling_period

    r_uu = _biased_covariance(inputs, inputs, window_size)
    r_yy = _biased_covariance(outputs, outputs, window_size)
    r_yu = _biased_covariance(outputs, inputs, window_size)
    r_uy = _biased_covariance(inputs, outputs, window_size)

    phi_u = _windowed_spectrum(
        r_uu,
        r_uu.transpose(0, 2, 1),
        window,
        normalized_frequency,
        sampling_period,
    )
    phi_y = _windowed_spectrum(
        r_yy,
        r_yy.transpose(0, 2, 1),
        window,
        normalized_frequency,
        sampling_period,
    )
    phi_yu = _windowed_spectrum(
        r_yu,
        r_uy.transpose(0, 2, 1),
        window,
        normalized_frequency,
        sampling_period,
    )

    phi_u = (phi_u + phi_u.conj().transpose(0, 2, 1)) / 2.0
    phi_y = (phi_y + phi_y.conj().transpose(0, 2, 1)) / 2.0

    response = np.full(
        (len(frequencies), outputs.shape[1], inputs.shape[1]),
        np.nan + 1j * np.nan,
    )
    noise = phi_y.copy()
    valid = np.ones(len(frequencies), dtype=bool)

    for index, input_spectrum in enumerate(phi_u):
        scale = np.linalg.norm(input_spectrum, ord=2)
        if scale == 0.0 or not np.isfinite(scale):
            valid[index] = False
            continue

        if np.linalg.cond(input_spectrum) >= 1.0 / np.finfo(float).eps:
            valid[index] = False
            continue

        response[index] = np.linalg.solve(input_spectrum.T, phi_yu[index].T).T
        noise[index] = phi_y[index] - response[index] @ phi_yu[index].conj().T

    noise = _clamp_psd(noise)

    coherence: np.ndarray | None = None
    response_std: np.ndarray | None = None
    window_energy = window[0] ** 2 + 2.0 * np.sum(window[1:] ** 2)
    # Spectral-density and response estimates have different asymptotic
    # variance formulas, hence the factor two appears only here.
    noise_std = np.sqrt(2.0 * window_energy / n) * np.abs(noise)

    if inputs.shape[1] == 1 and outputs.shape[1] == 1:
        numerator = np.abs(phi_yu[:, 0, 0]) ** 2
        denominator = np.real(phi_y[:, 0, 0] * phi_u[:, 0, 0])
        coherence = np.zeros(len(frequencies))
        usable = valid & (denominator > np.finfo(float).eps)
        coherence[usable] = np.clip(
            numerator[usable] / denominator[usable], 0.0, 1.0
        )

        response_std = np.full_like(response.real, np.inf)
        coherent = usable & (coherence > np.finfo(float).eps)
        response_std[coherent, 0, 0] = (
            np.sqrt(window_energy / n)
            * np.abs(response[coherent, 0, 0])
            * np.sqrt((1.0 - coherence[coherent]) / coherence[coherent])
        )

    return FrequencyResponse(
        frequency=frequencies.copy(),
        response=response,
        input_names=input_names,
        output_names=output_names,
        input_units=input_units,
        output_units=output_units,
        time_unit=time_unit,
        noise_spectrum=noise,
        coherence=coherence,
        response_std=response_std,
        noise_spectrum_std=noise_std,
        window_size=window_size,
    )
