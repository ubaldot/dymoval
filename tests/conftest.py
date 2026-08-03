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


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")
