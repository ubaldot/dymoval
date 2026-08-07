"""Tests for nonparametric frequency-response estimation."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure
from scipy.signal import lfilter

from dymoval import Dataset, FrequencyResponse, Signal


def _linear_dataset(gain: np.ndarray, seed: int = 42) -> Dataset:
    rng = np.random.default_rng(seed)
    n = 4096
    time = np.arange(n) * 0.01
    values = rng.standard_normal((n, gain.shape[1]))
    outputs = values @ gain.T

    return Dataset.from_signals(
        inputs=[
            Signal(f"u{i}", values[:, i], time, "V")
            for i in range(values.shape[1])
        ],
        outputs=[
            Signal(f"y{i}", outputs[:, i], time, "m")
            for i in range(outputs.shape[1])
        ],
    )


def test_spa_siso_recovers_static_gain() -> None:
    estimate = _linear_dataset(np.array([[2.5]])).spa(window_size=30)

    assert isinstance(estimate, FrequencyResponse)
    assert estimate.response.shape == (128, 1, 1)
    assert estimate.noise_spectrum.shape == (128, 1, 1)
    assert estimate.coherence is not None
    np.testing.assert_allclose(estimate.response[:, 0, 0], 2.5, atol=1e-10)
    np.testing.assert_allclose(estimate.coherence, 1.0, atol=1e-10)
    assert estimate.frequency[-1] == pytest.approx(np.pi / 0.01)
    np.testing.assert_allclose(
        estimate.frequency_hz, estimate.frequency / (2.0 * np.pi)
    )


def test_spa_mimo_recovers_channel_matrix() -> None:
    gain = np.array([[2.0, -0.5], [0.25, 3.0]])
    estimate = _linear_dataset(gain).spa(window_size=30)

    assert estimate.response.shape == (128, 2, 2)
    assert estimate.coherence is None
    np.testing.assert_allclose(
        estimate.response,
        np.broadcast_to(gain, estimate.response.shape),
        atol=1e-10,
    )


def test_spa_recovers_dynamic_response_orientation() -> None:
    rng = np.random.default_rng(7)
    n = 8192
    sampling_period = 0.01
    time = np.arange(n) * sampling_period
    input_values = rng.standard_normal(n)
    output_values = lfilter([0.0, 1.0], [1.0, -0.7], input_values)
    frequencies = np.array([5.0, 20.0, 50.0])
    dataset = Dataset.from_signals(
        inputs=[Signal("u", input_values, time)],
        outputs=[Signal("y", output_values, time)],
    )

    estimate = dataset.spa(
        frequencies=frequencies,
        window_size=100,
    )
    z = np.exp(1j * frequencies * sampling_period)
    expected = 1.0 / (z - 0.7)

    np.testing.assert_allclose(estimate.response[:, 0, 0], expected, atol=0.06)


def test_spa_selects_inputs_and_outputs() -> None:
    estimate = _linear_dataset(np.array([[1.0, 0.0], [0.0, 1.0]])).spa(
        inputs="u1", outputs="y1"
    )

    assert estimate.input_names == ("u1",)
    assert estimate.output_names == ("y1",)
    np.testing.assert_allclose(estimate.response[:, 0, 0], 1.0, atol=1e-10)


def test_frf_returns_and_interpolates_response() -> None:
    frequency = np.array([1.0, 2.0, 3.0])
    response = np.array([1.0 + 1.0j, 2.0 + 3.0j, 3.0 + 5.0j])[:, None, None]
    model = FrequencyResponse(
        frequency=frequency,
        response=response,
        input_names=("u",),
        output_names=("y",),
        input_units=("V",),
        output_units=("m",),
        time_unit="s",
        noise_spectrum=np.zeros((3, 1, 1)),
        coherence=np.ones(3),
        response_std=np.zeros((3, 1, 1)),
        noise_spectrum_std=np.zeros((3, 1, 1)),
        window_size=2,
    )

    np.testing.assert_array_equal(model.frf(), response)
    interpolated = model.frf([0.5, 1.5, 3.5])
    assert np.isnan(interpolated[0, 0, 0])
    assert interpolated[1, 0, 0] == pytest.approx(1.5 + 2.0j)
    assert np.isnan(interpolated[2, 0, 0])


def test_frequency_response_validates_diagnostic_shapes() -> None:
    with pytest.raises(ValueError, match="noise_spectrum"):
        FrequencyResponse(
            frequency=np.array([1.0]),
            response=np.ones((1, 1, 1), dtype=complex),
            input_names=("u",),
            output_names=("y",),
            input_units=("V",),
            output_units=("m",),
            time_unit="s",
            noise_spectrum=np.zeros((1, 2, 2)),
            coherence=np.ones(1),
            response_std=np.zeros((1, 1, 1)),
            noise_spectrum_std=np.zeros((1, 1, 1)),
            window_size=1,
        )


@pytest.mark.plots
def test_plot_siso_includes_coherence() -> None:
    estimate = _linear_dataset(np.array([[2.5]])).spa()

    fig = estimate.plot()

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3
    assert [axis.get_ylabel() for axis in fig.axes] == [
        "Magnitude [dB]",
        "Phase [deg]",
        "Coherence",
    ]
    assert fig.axes[-1].get_xlabel() == "Angular frequency [rad/s]"
    assert all(axis.get_xscale() == "log" for axis in fig.axes)
    assert fig.axes[0].get_legend().get_texts()[0].get_text() == "y0 / u0"


@pytest.mark.plots
def test_plot_mimo_draws_every_channel() -> None:
    estimate = _linear_dataset(np.array([[2.0, -0.5], [0.25, 3.0]])).spa()

    fig = estimate.plot(xscale="linear")

    assert len(fig.axes) == 2
    assert all(len(axis.get_lines()) == 4 for axis in fig.axes)
    assert all(axis.get_xscale() == "linear" for axis in fig.axes)


@pytest.mark.plots
def test_plot_can_hide_coherence() -> None:
    estimate = _linear_dataset(np.array([[1.0]])).spa()

    assert len(estimate.plot(show_coherence=False).axes) == 2


def test_plot_rejects_unknown_scale() -> None:
    estimate = _linear_dataset(np.array([[1.0]])).spa()

    with pytest.raises(ValueError, match="xscale"):
        estimate.plot(xscale="banana")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"window_size": 0}, ValueError),
        ({"window_size": True}, ValueError),
        ({"frequencies": []}, ValueError),
        ({"frequencies": [2.0, 1.0]}, ValueError),
        ({"inputs": "missing"}, KeyError),
        ({"outputs": []}, ValueError),
    ],
)
def test_spa_rejects_invalid_arguments(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    dataset = _linear_dataset(np.array([[1.0]]))

    with pytest.raises(error):
        dataset.spa(**kwargs)  # type: ignore[arg-type]
