# -*- coding: utf-8 -*-
"""Shared fixtures for the numpy-based dymoval core."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from dymoval import Dataset, Signal  # noqa: E402

N = 512
DT = 0.01


@pytest.fixture
def time() -> np.ndarray:
    return np.arange(N) * DT


@pytest.fixture
def signal(time: np.ndarray) -> Signal:
    return Signal(
        name="u1",
        values=np.sin(2 * np.pi * 5 * time) + 1.0,
        time=time,
        unit="V",
        time_unit="s",
    )


@pytest.fixture
def dataset(time: np.ndarray) -> Dataset:
    u1 = Signal(
        name="u1",
        values=np.sin(2 * np.pi * 5 * time),
        time=time,
        unit="V",
    )
    y0 = Signal(
        name="y0",
        values=np.cos(2 * np.pi * 2 * time),
        time=time,
        unit="m",
    )
    y1 = Signal(
        name="y1",
        values=np.cos(2 * np.pi * 7 * time),
        time=time,
        unit="m",
    )

    return Dataset.from_signals(inputs=[u1], outputs=[y0, y1])


@pytest.fixture
def sine_dataset() -> Dataset:
    """Port of the legacy ``sine_dataframe`` fixture.

    101 samples, ``Ts = 0.1``, three inputs and four outputs built from
    sine waves at 2, 2.4 and 4.8 Hz.
    """
    t = np.linspace(0, 10, 101)

    w1, w2, w3 = (2 * np.pi * f for f in (2.0, 2.4, 4.8))

    inputs = [
        Signal("u1", 2 + np.sin(w1 * t) + np.sin(w2 * t), t, "kPa"),
        Signal("u2", 2 + np.sin(w2 * t), t, "bar"),
        Signal("u3", 2 + np.sin(w3 * t), t, "deg"),
    ]
    outputs = [
        Signal("y1", 2 + np.sin(w1 * t) + np.sin(w3 * t), t, "deg"),
        Signal("y2", 1 + np.sin(w3 * t), t, "rad/s"),
        Signal(
            "y3",
            2 + np.sin(w1 * t) + np.sin(w2 * t) + 3 * np.sin(w3 * t),
            t,
            "V",
        ),
        Signal("y4", np.sin(w1 * t) - np.sin(w2 * t) - np.sin(w3 * t), t, "A"),
    ]

    return Dataset.from_signals(inputs=inputs, outputs=outputs)


@pytest.fixture
def ones_dataset() -> Dataset:
    """Port of the legacy ``constant_ones_dataframe`` fixture."""
    t = np.linspace(0, 1, 10)
    ones = np.ones_like(t)

    inputs = [
        Signal("u1", ones.copy(), t, "m"),
        Signal("u2", ones.copy(), t, "m/s"),
        Signal("u3", ones.copy(), t, "bar"),
    ]
    outputs = [
        Signal("y1", ones.copy(), t, "deg"),
        Signal("y2", ones.copy(), t, "m/s**2"),
        Signal("y3", ones.copy(), t, "V"),
    ]

    return Dataset.from_signals(inputs=inputs, outputs=outputs)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ============================================================
# Validation fixtures (ported from the legacy pandas suite)
# ============================================================
DATASET_KINDS = ["MIMO", "SISO", "SIMO", "MISO"]


def _slice_for(kind: str, inputs: list, outputs: list) -> tuple[list, list]:
    """Keep only the first input and/or output according to ``kind``."""
    if kind in ("SISO", "SIMO"):
        inputs = inputs[:1]

    if kind in ("SISO", "MISO"):
        outputs = outputs[:1]

    return inputs, outputs


@pytest.fixture(params=DATASET_KINDS)
def good_dataset(request) -> tuple:
    """Port of the legacy ``good_dataframe`` fixture.

    100 samples, ``Ts = 0.1``, normally distributed values.
    Returns ``(dataset, u_names, y_names, u_units, y_units, kind)``.
    """
    kind = request.param

    rng = np.random.default_rng(42)
    t = np.arange(100) * 0.1

    u_names = ["u1", "u2", "u3"]
    u_units = ["kPa", "degC", "m/s"]
    y_names = ["y1", "y2"]
    y_units = ["kPa", "m/s**2"]

    inputs = [
        Signal(name, rng.standard_normal(t.size), t.copy(), unit)
        for name, unit in zip(u_names, u_units)
    ]
    outputs = [
        Signal(name, rng.standard_normal(t.size), t.copy(), unit)
        for name, unit in zip(y_names, y_units)
    ]

    inputs, outputs = _slice_for(kind, inputs, outputs)

    ds = Dataset.from_signals(inputs=inputs, outputs=outputs)

    return (
        ds,
        [s.name for s in inputs],
        [s.name for s in outputs],
        [s.unit for s in inputs],
        [s.unit for s in outputs],
        kind,
    )


@pytest.fixture(params=DATASET_KINDS)
def good_signals(request) -> tuple:
    """Port of the legacy ``good_signals_no_nans`` fixture.

    Returns ``(input_signals, output_signals, kind)`` with 50 samples and
    ``Ts = 0.1``.
    """
    kind = request.param

    rng = np.random.default_rng(7)
    t = np.arange(50) * 0.1

    inputs = [
        Signal(name, rng.random(t.size), t.copy(), unit)
        for name, unit in zip(["u1", "u2", "u3"], ["m/s", "%", "degC"])
    ]
    outputs = [
        Signal(name, rng.random(t.size), t.copy(), unit)
        for name, unit in zip(
            ["y1", "y2", "y3", "y4"], ["m/s", "deg", "degC", "kPa"]
        )
    ]

    inputs, outputs = _slice_for(kind, inputs, outputs)

    return inputs, outputs, kind


@pytest.fixture
def correlation_tensors() -> tuple:
    """Port of the legacy ``correlation_tensors`` fixture.

    The expected values were pre-computed with Matlab.
    """
    x0 = np.array(
        [
            0.1419,
            0.4218,
            0.9157,
            0.7922,
            0.9595,
            0.8361,
            0.1023,
            0.1927,
            0.6123,
        ]
    )
    x1 = np.array(
        [
            0.6557,
            0.0357,
            0.8491,
            0.9340,
            0.6787,
            0.1826,
            0.8167,
            0.4615,
            0.3742,
        ]
    )
    X = np.array([x0, x1]).T

    y0 = np.array(
        [
            0.7577,
            0.7431,
            0.3922,
            0.6555,
            0.1712,
            0.1435,
            0.5682,
            0.1445,
            0.4324,
        ]
    )
    y1 = np.array(
        [
            0.7060,
            0.0318,
            0.2769,
            0.0462,
            0.0971,
            0.4523,
            0.2245,
            0.5677,
            0.4234,
        ]
    )
    Y = np.array([y0, y1]).T

    X_bandwidths = [5, 10]
    Y_bandwidths = [40, 6]
    sampling_period = 0.01

    Rx0y0_expected = np.array(
        [
            -0.00630122,
            0.17174685,
            -0.36912014,
            -0.29761134,
            -0.0755604,
            -0.40431233,
            0.19567388,
            0.66457636,
            0.24006238,
            0.20621649,
            -0.02004455,
        ]
    )
    Rx1y0_expected = np.array(
        [
            -0.30059329,
            0.07846654,
            0.14153643,
            -0.29767643,
            -0.23247292,
            0.1310837,
            -0.0044224,
            0.08695495,
            0.49633477,
            -0.08933459,
            -0.11328822,
        ]
    )
    Rx0y1_expected = np.array(
        [
            0.11068929,
            0.21744757,
            0.45582407,
            -0.01375399,
            -0.22855556,
            -0.44652205,
            -0.52630036,
            0.03978512,
            0.26546066,
            0.27176197,
            0.35630626,
        ]
    )
    Rx1y1_expected = np.array(
        [
            0.29732113,
            -0.01706344,
            0.13960017,
            0.07897259,
            -0.1073099,
            -0.12048405,
            -0.42777734,
            0.08356703,
            0.08822852,
            0.34948741,
            -0.28269522,
        ]
    )

    # Computed by taking into account the bandwidths and sampling periods
    Rx0y1_expected_partial = np.array([0.11068929, -0.44652205, 0.35630626])
    Rx1y1_expected_partial = np.array([0.29732113, -0.12048405, -0.28269522])

    return (
        Rx0y0_expected,
        Rx1y0_expected,
        Rx0y1_expected,
        Rx0y1_expected_partial,
        Rx1y1_expected,
        Rx1y1_expected_partial,
        X,
        Y,
        X_bandwidths,
        Y_bandwidths,
        sampling_period,
    )
