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
