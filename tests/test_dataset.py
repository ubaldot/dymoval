# -*- coding: utf-8 -*-
"""Tests for :class:`dymoval.dataset.Dataset`."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from dymoval import SPECTRUM_MODES, Dataset, Signal


def _sig(name: str, time: np.ndarray, value: float = 1.0) -> Signal:
    return Signal(
        name=name, values=np.full_like(time, value), time=time, unit="V"
    )


# ============================================================
# Construction / validation
# ============================================================
class Test_construction:
    def test_from_signals(self, dataset: Dataset) -> None:
        assert dataset.input_names() == ["u1"]
        assert dataset.output_names() == ["y0", "y1"]
        assert dataset.names() == ["u1", "y0", "y1"]
        assert len(dataset) == 3

    def test_from_dict(self, time: np.ndarray) -> None:
        ds = Dataset.from_dict(
            {"inputs": [_sig("u", time)], "outputs": [_sig("y", time)]}
        )

        assert "u" in ds and "y" in ds

    def test_from_dict_bad_key(self, time: np.ndarray) -> None:
        with pytest.raises(ValueError):
            Dataset.from_dict({"banana": [_sig("u", time)]})

    def test_from_dict_bad_type(self) -> None:
        with pytest.raises(TypeError):
            Dataset.from_dict({"inputs": [42]})  # type: ignore[list-item]

    def test_duplicate_name(self, time: np.ndarray) -> None:
        with pytest.raises(ValueError):
            Dataset.from_signals(inputs=[_sig("u", time), _sig("u", time)])

    def test_duplicate_across_inputs_outputs(self, time: np.ndarray) -> None:
        with pytest.raises(ValueError):
            Dataset.from_signals(
                inputs=[_sig("u", time)], outputs=[_sig("u", time)]
            )

    def test_empty(self) -> None:
        with pytest.raises(ValueError):
            Dataset.from_signals()

    def test_missing_time(self, time: np.ndarray) -> None:
        bad = Signal(name="u", values=np.zeros_like(time))

        with pytest.raises(ValueError):
            Dataset.from_signals(inputs=[bad])

    def test_length_mismatch(self, time: np.ndarray) -> None:
        short = time[:10]

        with pytest.raises(ValueError):
            Dataset.from_signals(
                inputs=[_sig("u", time)], outputs=[_sig("y", short)]
            )

    def test_sampling_period_mismatch(self, time: np.ndarray) -> None:
        with pytest.raises(ValueError):
            Dataset.from_signals(
                inputs=[_sig("u", time)], outputs=[_sig("y", 2 * time)]
            )

    def test_not_aligned(self, time: np.ndarray) -> None:
        with pytest.raises(ValueError):
            Dataset.from_signals(
                inputs=[_sig("u", time)], outputs=[_sig("y", time + 100.0)]
            )

    def test_outputs_only(self, time: np.ndarray) -> None:
        ds = Dataset.from_signals(outputs=[_sig("y", time)])

        assert ds.time() is not None
        assert ds.input_names() == []


# ============================================================
# Access
# ============================================================
class Test_access:
    def test_getitem(self, dataset: Dataset) -> None:
        assert dataset["u1"].name == "u1"
        assert dataset["y1"].name == "y1"

    def test_getitem_missing(self, dataset: Dataset) -> None:
        with pytest.raises(KeyError):
            dataset["nope"]

    def test_time_and_dt(self, dataset: Dataset) -> None:
        assert len(dataset.time()) == len(dataset["u1"])
        assert np.isclose(dataset.get_sampling_period(), 0.01)

    def test_select_signals(self, dataset: Dataset) -> None:
        assert len(dataset._select_signals([])) == 3
        assert [s.name for s in dataset._select_signals(["y1"])] == ["y1"]

        with pytest.raises(KeyError):
            dataset._select_signals(["nope"])


# ============================================================
# Processing
# ============================================================
class Test_processing:
    def test_copy_is_deep(self, dataset: Dataset) -> None:
        other = dataset.copy()
        other["u1"].values[0] = 12345.0

        assert dataset["u1"].values[0] != 12345.0

    def test_remove_mean(self, dataset: Dataset) -> None:
        out = dataset.remove_mean()

        for sig in out.all_signals().values():
            assert np.isclose(np.mean(sig.values), 0.0, atol=1e-12)

    def test_remove_constant_scalar(self, dataset: Dataset) -> None:
        out = dataset.remove_constant(1.0)

        for name, sig in out.all_signals().items():
            assert np.allclose(sig.values, dataset[name].values - 1.0)

    def test_remove_constant_mapping(self, dataset: Dataset) -> None:
        out = dataset.remove_constant({"u1": 2.0})

        assert np.allclose(out["u1"].values, dataset["u1"].values - 2.0)
        assert np.allclose(out["y0"].values, dataset["y0"].values)

    def test_remove_constant_unknown_signal(self, dataset: Dataset) -> None:
        with pytest.raises(KeyError):
            dataset.remove_constant({"nope": 1.0})

    def test_detrend(self, dataset: Dataset) -> None:
        out = dataset.detrend()

        assert set(out.names()) == set(dataset.names())

    def test_resample(self, dataset: Dataset) -> None:
        new_time = dataset.time()[::2]
        out = dataset.resample(new_time)

        assert len(out.time()) == len(new_time)

    def test_meta_is_copied(self, time: np.ndarray) -> None:
        ds = Dataset.from_signals(inputs=[_sig("u", time)], meta={"a": [1]})
        out = ds.copy()

        assert out.meta == {"a": [1]}
        assert out.meta is not ds.meta


# ============================================================
# Align / pipe
# ============================================================
class Test_align:
    def test_intersection(self, dataset: Dataset) -> None:
        shifted = dataset.resample(dataset.time()[10:])
        a, b = dataset.align(shifted)

        assert np.allclose(a.time(), b.time())

    def test_union(self, dataset: Dataset) -> None:
        other = dataset.resample(dataset.time() + 1.0)
        a, b = dataset.align(other, how="union")

        assert len(a.time()) >= len(dataset.time())
        assert np.allclose(a.time(), b.time())

    def test_invalid_mode(self, dataset: Dataset) -> None:
        with pytest.raises(ValueError):
            dataset.align(dataset, how="banana")

    def test_no_overlap(self, dataset: Dataset) -> None:
        other = dataset.resample(dataset.time() + 1e6)

        with pytest.raises(ValueError):
            dataset.align(other)

    def test_pipe(self, dataset: Dataset) -> None:
        out = dataset.pipe(lambda ds: ds.remove_mean())

        assert isinstance(out, Dataset)

    def test_pipe_bad_return(self, dataset: Dataset) -> None:
        with pytest.raises(TypeError):
            dataset.pipe(lambda ds: 42)


# ============================================================
# Frequency domain
# ============================================================
class Test_spectrum:
    def test_fft(self, dataset: Dataset) -> None:
        out = dataset.fft()

        assert set(out) == set(dataset.names())
        for freq, y in out.values():
            assert freq.shape == y.shape

    def test_spectrum(self, dataset: Dataset) -> None:
        out = dataset.spectrum("psd_welch")

        assert set(out) == set(dataset.names())


# ============================================================
# Grouping
# ============================================================
class Test_normalize_groups:
    def test_default_is_one_group_per_signal(self, dataset: Dataset) -> None:
        assert dataset._normalize_groups(()) == [("u1",), ("y0",), ("y1",)]

    def test_mixed_str_and_tuple(self, dataset: Dataset) -> None:
        groups = dataset._normalize_groups([("u1", "y1"), "y0"])

        assert groups == [("u1", "y1"), ("y0",)]

    def test_bad_type(self, dataset: Dataset) -> None:
        with pytest.raises(TypeError):
            dataset._normalize_groups([42])  # type: ignore[list-item]

    def test_unknown_signal(self, dataset: Dataset) -> None:
        with pytest.raises(KeyError):
            dataset._normalize_groups(["nope"])

    def test_empty_group(self, dataset: Dataset) -> None:
        with pytest.raises(ValueError):
            dataset._normalize_groups([()])


# ============================================================
# Plotting
# ============================================================
class Test_plot:
    @pytest.mark.plots
    @pytest.mark.parametrize("with_scope", [False, True])
    def test_grouping_layout(self, dataset: Dataset, with_scope: bool) -> None:
        fig = dataset.plot(("u1", "y1"), "y0", with_scope=with_scope)

        assert isinstance(fig, Figure)

        plot_axes = [ax for ax in fig.axes if ax.get_lines()]

        assert len(plot_axes) == 2
        assert len(plot_axes[0].get_lines()) == 2
        assert len(plot_axes[1].get_lines()) == 1

    @pytest.mark.plots
    def test_single_group(self, dataset: Dataset) -> None:
        fig = dataset.plot("u1", with_scope=False)

        assert len([ax for ax in fig.axes if ax.get_lines()]) == 1

    @pytest.mark.plots
    def test_output_is_green_when_alone(self, dataset: Dataset) -> None:
        fig = dataset.plot("y0", with_scope=False)
        line = fig.axes[0].get_lines()[0]

        assert line.get_color() == "green"

    @pytest.mark.plots
    def test_scope_is_attached(self, dataset: Dataset) -> None:
        fig = dataset.plot(with_scope=True)

        assert len(fig._scopes) == 1

    @pytest.mark.plots
    def test_unknown_signal(self, dataset: Dataset) -> None:
        with pytest.raises(KeyError):
            dataset.plot("nope", with_scope=False)

    @pytest.mark.plots
    def test_plot_xy(self, dataset: Dataset) -> None:
        ax = dataset.plot_xy("u1", "y1")

        assert isinstance(ax, Axes)

        line = ax.get_lines()[0]
        assert np.allclose(line.get_xdata(), dataset["u1"].values)
        assert np.allclose(line.get_ydata(), dataset["y1"].values)
        assert ax.get_xlabel() == "u1 [V]"
        assert ax.get_ylabel() == "y1 [m]"

    @pytest.mark.plots
    def test_plot_xy_unknown(self, dataset: Dataset) -> None:
        with pytest.raises(KeyError):
            dataset.plot_xy("u1", "nope")


class Test_plot_spectrum:
    @pytest.mark.plots
    @pytest.mark.parametrize("mode", SPECTRUM_MODES)
    @pytest.mark.parametrize("with_scope", [False, True])
    def test_modes(
        self, dataset: Dataset, mode: str, with_scope: bool
    ) -> None:
        fig = dataset.plot_spectrum(
            ("u1", "y1"),
            "y0",
            mode=mode,  # type: ignore[arg-type]
            with_scope=with_scope,
        )

        assert isinstance(fig, Figure)

    @pytest.mark.plots
    def test_invalid_mode(self, dataset: Dataset) -> None:
        with pytest.raises(ValueError):
            dataset.plot_spectrum(mode="banana")  # type: ignore[arg-type]

    @pytest.mark.plots
    def test_amplitude_layout_is_two_rows_per_group(
        self, dataset: Dataset
    ) -> None:
        fig = dataset.plot_spectrum(
            ("u1", "y1"), "y0", mode="amplitude", with_scope=False
        )

        plot_axes = [ax for ax in fig.axes if ax.get_lines()]

        assert len(plot_axes) == 4  # 2 groups * (magnitude + phase)
        assert plot_axes[1].get_ylabel() == "Phase [deg]"
        assert plot_axes[3].get_ylabel() == "Phase [deg]"

    @pytest.mark.plots
    def test_amplitude_scope_one_per_group(self, dataset: Dataset) -> None:
        fig = dataset.plot_spectrum(
            ("u1", "y1"), "y0", mode="amplitude", with_scope=True
        )

        assert len(fig._scopes) == 2

        # all the group scopes share the very same panel text artist
        texts = {id(scope.info_text) for scope in fig._scopes}
        assert len(texts) == 1
