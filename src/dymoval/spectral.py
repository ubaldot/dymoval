"""Internal nonparametric spectral estimators."""

from __future__ import annotations

import numpy as np

from .frequency_response import FrequencyResponse


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
