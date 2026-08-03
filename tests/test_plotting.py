# -*- coding: utf-8 -*-
"""Tests for the multi-dataset plotting helpers."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from dymoval import (
    Dataset,
    plot_compare,
    plot_dataset,
    plot_spectrum_compare,
)


def _plot_axes(fig: Figure) -> list:
    return [ax for ax in fig.axes if ax.get_lines()]


class Test_plot_dataset:
    @pytest.mark.plots
    def test_delegates_to_dataset(self, dataset: Dataset) -> None:
        fig = plot_dataset(dataset, "u1", with_scope=False)

        assert isinstance(fig, Figure)
        assert len(_plot_axes(fig)) == 1


class Test_plot_compare:
    @pytest.mark.plots
    def test_one_line_per_dataset(self, dataset: Dataset) -> None:
        fig = plot_compare(
            dataset,
            dataset.remove_mean(),
            labels=["raw", "centered"],
            with_scope=False,
        )

        axes = _plot_axes(fig)

        assert len(axes) == 3
        assert all(len(ax.get_lines()) == 2 for ax in axes)

        labels = [line.get_label() for line in axes[0].get_lines()]
        assert labels == ["u1 (raw)", "u1 (centered)"]

    @pytest.mark.plots
    def test_default_labels(self, dataset: Dataset) -> None:
        fig = plot_compare(dataset, dataset.detrend(), with_scope=False)
        labels = [line.get_label() for line in _plot_axes(fig)[0].get_lines()]

        assert labels == ["u1 (ds0)", "u1 (ds1)"]

    @pytest.mark.plots
    def test_explicit_names(self, dataset: Dataset) -> None:
        fig = plot_compare(
            dataset, dataset.detrend(), names=["y0"], with_scope=False
        )

        assert len(_plot_axes(fig)) == 1

    @pytest.mark.plots
    def test_missing_signal(self, dataset: Dataset) -> None:
        with pytest.raises(KeyError):
            plot_compare(
                dataset, dataset.detrend(), names=["nope"], with_scope=False
            )

    @pytest.mark.plots
    def test_wrong_number_of_labels(self, dataset: Dataset) -> None:
        with pytest.raises(ValueError):
            plot_compare(
                dataset, dataset.detrend(), labels=["a"], with_scope=False
            )

    @pytest.mark.plots
    def test_needs_two_datasets(self, dataset: Dataset) -> None:
        with pytest.raises(ValueError):
            plot_compare(dataset, with_scope=False)

    @pytest.mark.plots
    def test_bad_type(self, dataset: Dataset) -> None:
        with pytest.raises(TypeError):
            plot_compare(dataset, 42, with_scope=False)  # type: ignore[arg-type]

    @pytest.mark.plots
    def test_alignment(self, dataset: Dataset) -> None:
        shifted = dataset.resample(dataset.time()[10:])
        fig = plot_compare(dataset, shifted, align=True, with_scope=False)

        lines = _plot_axes(fig)[0].get_lines()

        assert np.allclose(lines[0].get_xdata(), lines[1].get_xdata())

    @pytest.mark.plots
    def test_no_overlap(self, dataset: Dataset) -> None:
        other = dataset.resample(dataset.time() + 1e6)

        with pytest.raises(ValueError):
            plot_compare(dataset, other, with_scope=False)

    @pytest.mark.plots
    def test_no_common_signals(self, dataset: Dataset, time) -> None:
        from dymoval import Signal

        other = Dataset.from_signals(
            inputs=[Signal(name="zzz", values=np.zeros_like(time), time=time)]
        )

        with pytest.raises(ValueError):
            plot_compare(dataset, other, with_scope=False)

    @pytest.mark.plots
    def test_scope_is_attached(self, dataset: Dataset) -> None:
        fig = plot_compare(dataset, dataset.detrend(), with_scope=True)

        assert len(fig._scopes) == 1


class Test_plot_spectrum_compare:
    @pytest.mark.plots
    @pytest.mark.parametrize("mode", ["power", "psd", "psd_welch"])
    def test_modes(self, dataset: Dataset, mode: str) -> None:
        fig = plot_spectrum_compare(
            dataset,
            dataset.detrend(),
            mode=mode,  # type: ignore[arg-type]
            with_scope=False,
        )

        assert len(_plot_axes(fig)) == 3

    @pytest.mark.plots
    def test_amplitude_is_rejected(self, dataset: Dataset) -> None:
        with pytest.raises(ValueError):
            plot_spectrum_compare(dataset, dataset.detrend(), mode="amplitude")

    @pytest.mark.plots
    def test_invalid_mode(self, dataset: Dataset) -> None:
        with pytest.raises(ValueError):
            plot_spectrum_compare(
                dataset,
                dataset.detrend(),
                mode="banana",  # type: ignore[arg-type]
            )

    @pytest.mark.plots
    def test_scope_is_attached(self, dataset: Dataset) -> None:
        fig = plot_spectrum_compare(
            dataset, dataset.detrend(), with_scope=True
        )

        assert len(fig._scopes) == 1
