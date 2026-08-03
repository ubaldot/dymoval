# -*- coding: utf-8 -*-
"""Tests for :class:`dymoval.dataset.Dataset`."""

from __future__ import annotations

import matplotlib.pyplot as plt
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

    def test_too_few_samples(self) -> None:
        t = np.array([0.0])
        bad = Signal(name="u", values=np.zeros(1), time=t)

        with pytest.raises(ValueError):
            Dataset.from_signals(inputs=[bad])

    def test_time_unit_must_be_consistent(self, time: np.ndarray) -> None:
        u = Signal(name="u", values=np.zeros_like(time), time=time)
        y = Signal(
            name="y",
            values=np.zeros_like(time),
            time=time,
            time_unit="ms",
        )

        with pytest.raises(ValueError):
            Dataset.from_signals(inputs=[u], outputs=[y])

    def test_outputs_only(self, time: np.ndarray) -> None:
        ds = Dataset.from_signals(outputs=[_sig("y", time)])

        assert ds.time() is not None
        assert ds.input_names() == []


# ============================================================
# The constructor itself stays strict
# ============================================================
class Test_constructor_is_strict:
    def test_length_mismatch(self, time: np.ndarray) -> None:
        with pytest.raises(ValueError):
            Dataset(
                inputs={"u": _sig("u", time)},
                outputs={"y": _sig("y", time[:10])},
            )

    def test_sampling_period_mismatch(self, time: np.ndarray) -> None:
        with pytest.raises(ValueError):
            Dataset(
                inputs={"u": _sig("u", time)},
                outputs={"y": _sig("y", 2 * time)},
            )

    def test_not_aligned(self, time: np.ndarray) -> None:
        with pytest.raises(ValueError):
            Dataset(
                inputs={"u": _sig("u", time)},
                outputs={"y": _sig("y", time + 100.0)},
            )

    def test_key_must_match_signal_name(self, time: np.ndarray) -> None:
        with pytest.raises(ValueError):
            Dataset(inputs={"wrong": _sig("u", time)})


# ============================================================
# Harmonization on construction (was _fix_sampling_periods)
# ============================================================
class Test_harmonization:
    def test_different_sampling_periods(self) -> None:
        t_fast = np.arange(0, 10, 0.01)
        t_slow = np.arange(0, 10, 0.1)

        u = Signal(name="u", values=np.sin(t_fast), time=t_fast)
        y = Signal(name="y", values=np.cos(t_slow), time=t_slow)

        ds = Dataset.from_signals(inputs=[u], outputs=[y])

        # everybody is brought to the SLOWEST sampling period
        assert np.isclose(ds.get_sampling_period(), 0.1)
        assert len(ds["u"]) == len(ds["y"])
        assert np.allclose(ds["y"].values, np.cos(ds.time()), atol=1e-9)

    def test_non_integer_ratio_is_supported(self) -> None:
        # the legacy implementation had to EXCLUDE such signals
        t1 = np.arange(0, 10, 0.01)
        t2 = np.arange(0, 10, 0.017)

        u = Signal(name="u", values=np.sin(t1), time=t1)
        y = Signal(name="y", values=np.sin(t2), time=t2)

        ds = Dataset.from_signals(inputs=[u], outputs=[y])

        assert np.isclose(ds.get_sampling_period(), 0.017)
        assert set(ds.names()) == {"u", "y"}

    def test_explicit_target_sampling_period(self, time: np.ndarray) -> None:
        ds = Dataset.from_signals(
            inputs=[_sig("u", time)],
            outputs=[_sig("y", time)],
            target_sampling_period=0.05,
        )

        assert np.isclose(ds.get_sampling_period(), 0.05)

    @pytest.mark.parametrize("target", [-0.1, 0.0, "potato", True])
    def test_invalid_target(self, time: np.ndarray, target: object) -> None:
        with pytest.raises(ValueError):
            Dataset.from_signals(
                inputs=[_sig("u", time)],
                target_sampling_period=target,  # type: ignore[arg-type]
            )

    def test_time_vector_is_preserved_when_already_aligned(
        self, dataset: Dataset, time: np.ndarray
    ) -> None:
        assert np.allclose(dataset.time(), time)
        assert len(dataset.time()) == len(time)

    def test_grid_spans_the_common_interval(self) -> None:
        t1 = np.arange(0.0, 10.0, 0.1)
        t2 = np.arange(3.0, 20.0, 0.1)

        u = Signal(name="u", values=np.sin(t1), time=t1)
        y = Signal(name="y", values=np.sin(t2), time=t2)

        ds = Dataset.from_signals(inputs=[u], outputs=[y])

        assert np.isclose(ds.time()[0], 3.0)
        assert ds.time()[-1] <= 9.9 + 1e-9

    def test_no_overlap(self) -> None:
        t1 = np.arange(0.0, 1.0, 0.1)
        t2 = np.arange(100.0, 101.0, 0.1)

        u = Signal(name="u", values=np.sin(t1), time=t1)
        y = Signal(name="y", values=np.sin(t2), time=t2)

        with pytest.raises(ValueError):
            Dataset.from_signals(inputs=[u], outputs=[y])


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
# Name / export
# ============================================================
class Test_name:
    def test_default_is_empty(self, dataset: Dataset) -> None:
        assert dataset.name == ""
        assert repr(dataset).startswith("Dataset: ")

    def test_name_is_shown_in_repr(self, time: np.ndarray) -> None:
        ds = Dataset.from_signals(inputs=[_sig("u", time)], name="My rig")

        assert ds.name == "My rig"
        assert repr(ds).startswith("Dataset 'My rig': ")

    @pytest.mark.parametrize(
        "op",
        [
            lambda ds: ds.copy(),
            lambda ds: ds.detrend(),
            lambda ds: ds.remove_signals("y0"),
            lambda ds: ds.add_output(
                Signal(
                    name="y9",
                    values=np.zeros_like(ds.time()),
                    time=ds.time(),
                )
            ),
        ],
    )
    def test_name_survives_operations(self, dataset: Dataset, op) -> None:
        named = Dataset(
            inputs=dataset.inputs, outputs=dataset.outputs, name="rig"
        )

        assert op(named).name == "rig"

    @pytest.mark.plots
    def test_name_is_the_figure_title(self, dataset: Dataset) -> None:
        named = Dataset(
            inputs=dataset.inputs, outputs=dataset.outputs, name="rig"
        )

        for fig in (
            named.plot(with_scope=False),
            named.plot_coverage(),
            named.plot_spectrum(with_scope=False),
        ):
            assert fig._suptitle is not None
            assert fig._suptitle.get_text() == "rig"

        assert dataset.plot(with_scope=False)._suptitle is None


class Test_export_to_mat:
    def test_roundtrip(self, dataset: Dataset, tmp_path) -> None:
        from scipy.io import loadmat

        target = tmp_path / "ds.mat"
        dataset.export_to_mat(str(target))

        assert target.exists()

        loaded = loadmat(str(target), simplify_cells=True)

        assert np.allclose(loaded["TIME"], dataset.time())
        assert set(loaded["INPUT"]) == {"u1"}
        assert set(loaded["OUTPUT"]) == {"y0", "y1"}

        stored = loaded["INPUT"]["u1"]
        assert np.allclose(stored["values"], dataset["u1"].values)
        assert stored["unit"] == (dataset["u1"].unit or "")
        assert np.isclose(
            stored["sampling_period"], dataset.get_sampling_period()
        )


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
        out = dataset.spectrum(mode="psd_welch")

        assert set(out) == set(dataset.names())

    def test_fft_selection(self, dataset: Dataset) -> None:
        assert set(dataset.fft("u1", "y1")) == {"u1", "y1"}

    def test_spectrum_selection(self, dataset: Dataset) -> None:
        assert set(dataset.spectrum("y0", mode="amplitude")) == {"y0"}

    def test_unknown_name_raises(self, dataset: Dataset) -> None:
        with pytest.raises(KeyError):
            dataset.fft("potato")

        with pytest.raises(KeyError):
            dataset.spectrum("potato")


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
        ax = dataset.plot_xy("u1", "y1", ax=plt.subplots()[1])

        assert isinstance(ax, Axes)

        line = ax.get_lines()[0]
        assert np.allclose(line.get_xdata(), dataset["u1"].values)
        assert np.allclose(line.get_ydata(), dataset["y1"].values)
        assert ax.get_xlabel() == "u1 [V]"
        assert ax.get_ylabel() == "y1 [m]"

    @pytest.mark.plots
    def test_plot_xy_returns_figure(self, dataset: Dataset) -> None:
        fig = dataset.plot_xy("u1", "y1")

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1

    @pytest.mark.plots
    def test_plot_xy_many_pairs(self, dataset: Dataset) -> None:
        fig = dataset.plot_xy(("u1", "y0"), ("u1", "y1"), ("y0", "y1"))

        assert isinstance(fig, Figure)
        # factorize(3) over-allocates a 2x2 grid: the spare axes is removed
        assert len(fig.axes) == 3

    @pytest.mark.plots
    def test_plot_xy_defaults_to_zip(self, dataset: Dataset) -> None:
        fig = dataset.plot_xy()

        assert isinstance(fig, Figure)
        assert len(fig.axes) == min(
            len(dataset.input_names()), len(dataset.output_names())
        )

    @pytest.mark.plots
    def test_plot_xy_bad_args(self, dataset: Dataset) -> None:
        with pytest.raises(TypeError):
            dataset.plot_xy("u1", "y0", "y1")

        with pytest.raises(TypeError):
            dataset.plot_xy(("u1", "y0", "y1"))  # type: ignore[arg-type]

        with pytest.raises(ValueError):
            dataset.plot_xy(("u1", "y0"), ("u1", "y1"), ax=plt.subplots()[1])

    @pytest.mark.plots
    def test_plot_xy_unknown(self, dataset: Dataset) -> None:
        with pytest.raises(KeyError):
            dataset.plot_xy("u1", "nope")


class Test_plot_geometry:
    """The ``layout`` / ``ax_height`` / ``ax_width`` knobs."""

    @pytest.mark.plots
    def test_figsize(self, dataset: Dataset) -> None:
        fig = dataset.plot(with_scope=False, ax_height=3.0, ax_width=8.0)

        width, height = fig.get_size_inches()

        assert width == pytest.approx(8.0)
        assert height == pytest.approx(3.0 * len(dataset.names()) + 1)

    @pytest.mark.plots
    def test_figsize_defaults_unchanged(self, dataset: Dataset) -> None:
        fig = dataset.plot(with_scope=False)

        width, height = fig.get_size_inches()

        assert width == pytest.approx(10.0)
        assert height == pytest.approx(2.0 * len(dataset.names()) + 1)

    @pytest.mark.plots
    @pytest.mark.parametrize(
        "layout", ["constrained", "compressed", "tight", "none"]
    )
    def test_layouts(self, dataset: Dataset, layout: str) -> None:
        fig = dataset.plot(with_scope=False, layout=layout)  # type: ignore[arg-type]

        engine = fig.get_layout_engine()

        if layout == "none":
            assert engine is None
        else:
            assert engine is not None

    @pytest.mark.plots
    def test_bad_layout(self, dataset: Dataset) -> None:
        with pytest.raises(ValueError):
            dataset.plot(with_scope=False, layout="nope")  # type: ignore[arg-type]

    @pytest.mark.plots
    def test_coverage_figsize(self, dataset: Dataset) -> None:
        fig = dataset.plot_coverage(ax_height=2.5, ax_width=6.0)

        width, height = fig.get_size_inches()

        assert width == pytest.approx(6.0)
        assert height == pytest.approx(2.5 * len(dataset.names()) + 1)

    @pytest.mark.plots
    def test_spectrum_figsize(self, dataset: Dataset) -> None:
        fig = dataset.plot_spectrum(
            with_scope=False, ax_height=2.5, ax_width=6.0
        )

        width, height = fig.get_size_inches()

        assert width == pytest.approx(6.0)
        assert height == pytest.approx(2.5 * len(dataset.names()) + 1)


class Test_plot_styling:
    """The ``color_input`` / ``color_output`` knobs and ``**kwargs``."""

    @pytest.mark.plots
    def test_semantic_colors(self, dataset: Dataset) -> None:
        fig = dataset.plot(
            "u1", "y0", with_scope=False, color_input="red", color_output="k"
        )

        assert fig.axes[0].get_lines()[0].get_color() == "red"
        assert fig.axes[1].get_lines()[0].get_color() == "k"

    @pytest.mark.plots
    def test_kwargs_forwarded(self, dataset: Dataset) -> None:
        fig = dataset.plot("u1", with_scope=False, linestyle="--", alpha=0.25)

        line = fig.axes[0].get_lines()[0]

        assert line.get_linestyle() == "--"
        assert line.get_alpha() == pytest.approx(0.25)

    @pytest.mark.plots
    def test_grouped_signals_use_the_cycle(self, dataset: Dataset) -> None:
        # semantic colors must not collapse two overlaid signals into one
        fig = dataset.plot(
            ("u1", "y0"),
            with_scope=False,
            color_input="red",
            color_output="red",
        )

        lines = fig.axes[0].get_lines()

        assert lines[0].get_color() != lines[1].get_color()

    @pytest.mark.plots
    def test_spectrum_kwargs_forwarded(self, dataset: Dataset) -> None:
        fig = dataset.plot_spectrum("u1", with_scope=False, linestyle=":")

        assert fig.axes[0].get_lines()[0].get_linestyle() == ":"


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
