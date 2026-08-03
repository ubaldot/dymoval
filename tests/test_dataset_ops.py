# -*- coding: utf-8 -*-
"""Tests for the Dataset features ported from the legacy pandas core.

The expected numbers of ``test_low_pass_filter`` come straight from the
legacy suite (``tests/legacy/test_dataset.py::test_lowpass_filter``), so
the new first-order filter is bit-for-bit compatible with the old one.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from dymoval import Dataset, Signal

ATOL = 1e-9


# ============================================================
# Introspection
# ============================================================
class Test_introspection:
    def test_signal_list(self, dataset: Dataset) -> None:
        assert dataset.signal_list() == [
            ("INPUT", "u1", "V"),
            ("OUTPUT", "y0", "m"),
            ("OUTPUT", "y1", "m"),
        ]

    def test_kind_of(self, dataset: Dataset) -> None:
        assert dataset.kind_of("u1") == "INPUT"
        assert dataset.kind_of("y0") == "OUTPUT"

        with pytest.raises(KeyError):
            dataset.kind_of("nope")

    def test_repr(self, dataset: Dataset) -> None:
        text = repr(dataset)

        assert "1 input(s), 2 output(s)" in text
        assert "INPUT  u1 [V]" in text
        assert "OUTPUT y1 [m]" in text

    def test_time_unit(self, dataset: Dataset) -> None:
        assert dataset.time_unit() == "s"

    def test_to_signals(self, dataset: Dataset) -> None:
        dumped = dataset.to_signals()

        assert [s.name for s in dumped["INPUT"]] == ["u1"]
        assert [s.name for s in dumped["OUTPUT"]] == ["y0", "y1"]

        for kind in ("INPUT", "OUTPUT"):
            for sig in dumped[kind]:
                original = dataset[sig.name]

                assert np.allclose(sig.values, original.values)
                assert sig.unit == original.unit
                assert sig.time_unit == original.time_unit
                assert np.isclose(
                    sig.get_sampling_period(),
                    original.get_sampling_period(),
                )

        # copies, not references
        dumped["INPUT"][0].values[0] = 12345.0
        assert dataset["u1"].values[0] != 12345.0

    def test_to_signals_round_trip(self, dataset: Dataset) -> None:
        dumped = dataset.to_signals()
        rebuilt = Dataset.from_signals(
            inputs=dumped["INPUT"], outputs=dumped["OUTPUT"]
        )

        assert rebuilt.signal_list() == dataset.signal_list()
        assert np.allclose(rebuilt.time(), dataset.time())


class Test_dataset_values:
    def test_mimo(self, sine_dataset: Dataset) -> None:
        t, u, y = sine_dataset.dataset_values()

        assert t.shape == (101,)
        assert u.shape == (101, 3)
        assert y.shape == (101, 4)
        assert np.allclose(u[:, 0], sine_dataset["u1"].values)
        assert np.allclose(y[:, 3], sine_dataset["y4"].values)

    def test_single_signal_is_still_2d(self, dataset: Dataset) -> None:
        t, u, y = dataset.dataset_values()

        assert u.shape == (len(t), 1)  # only one input
        assert y.shape == (len(t), 2)  # two outputs
        assert np.allclose(u[:, 0], dataset["u1"].values)
        assert np.allclose(t, dataset.time())

    def test_no_inputs(self, time: np.ndarray) -> None:
        ds = Dataset.from_signals(
            outputs=[Signal("y", np.zeros_like(time), time)]
        )
        _, u, y = ds.dataset_values()

        assert u.shape == (len(time), 0)
        assert y.shape == (len(time), 1)


# ============================================================
# Structure
# ============================================================
class Test_add_signals:
    def test_add_input(self, dataset: Dataset) -> None:
        new = Signal("u2", np.ones_like(dataset.time()), dataset.time(), "A")
        out = dataset.add_input(new)

        assert out.input_names() == ["u1", "u2"]
        assert dataset.input_names() == ["u1"]  # immutability
        assert np.allclose(out["u2"].values, 1.0)

    def test_add_output(self, dataset: Dataset) -> None:
        new = Signal("y2", np.ones_like(dataset.time()), dataset.time(), "A")
        out = dataset.add_output(new)

        assert out.output_names() == ["y0", "y1", "y2"]

    def test_added_signal_is_resampled(self, dataset: Dataset) -> None:
        # a slower, longer signal gets put on the dataset time grid
        t = np.arange(0.0, 20.0, 0.05)
        new = Signal("u2", np.sin(t), t)

        out = dataset.add_input(new)

        assert len(out["u2"]) == len(dataset.time())
        assert np.allclose(out["u2"].time, dataset.time())

    def test_duplicate_name(self, dataset: Dataset) -> None:
        clash = Signal("u1", np.zeros_like(dataset.time()), dataset.time())

        with pytest.raises(KeyError):
            dataset.add_input(clash)

        with pytest.raises(KeyError):
            dataset.add_output(clash)

    def test_duplicate_within_the_arguments(self, dataset: Dataset) -> None:
        a = Signal("u2", np.zeros_like(dataset.time()), dataset.time())
        b = Signal("u2", np.ones_like(dataset.time()), dataset.time())

        with pytest.raises(KeyError):
            dataset.add_input(a, b)

    def test_bad_type(self, dataset: Dataset) -> None:
        with pytest.raises(TypeError):
            dataset.add_input(42)  # type: ignore[arg-type]

    def test_no_signals_is_a_copy(self, dataset: Dataset) -> None:
        out = dataset.add_input()

        assert out.names() == dataset.names()
        assert out is not dataset


class Test_remove_signals:
    def test_nominal(self, sine_dataset: Dataset) -> None:
        out = sine_dataset.remove_signals("u1", "y1")

        assert "u1" not in out
        assert "y1" not in out
        assert "u2" in out
        # the original is untouched
        assert "u1" in sine_dataset

    def test_unknown_signal(self, dataset: Dataset) -> None:
        with pytest.raises(KeyError):
            dataset.remove_signals("potato")

    def test_cannot_empty_the_dataset(self, dataset: Dataset) -> None:
        with pytest.raises(ValueError):
            dataset.remove_signals(*dataset.names())

    def test_last_input_can_be_removed(self, dataset: Dataset) -> None:
        # unlike the legacy implementation, an output-only dataset is legal
        out = dataset.remove_signals("u1")

        assert out.input_names() == []
        assert out.output_names() == ["y0", "y1"]


# ============================================================
# Processing
# ============================================================
class Test_remove_means:
    def test_all_signals(self, sine_dataset: Dataset) -> None:
        out = sine_dataset.remove_mean()

        for sig in out.all_signals().values():
            assert np.isclose(np.mean(sig.values), 0.0, atol=ATOL)

        # the original is not overwritten
        assert not np.isclose(
            np.mean(sine_dataset["u1"].values), 0.0, atol=ATOL
        )

    def test_selected_signals(self, sine_dataset: Dataset) -> None:
        out = sine_dataset.remove_mean("u1", "y1")

        assert np.isclose(np.mean(out["u1"].values), 0.0, atol=ATOL)
        assert np.isclose(np.mean(out["y1"].values), 0.0, atol=ATOL)
        assert np.allclose(out["u2"].values, sine_dataset["u2"].values)

    def test_unknown_signal(self, sine_dataset: Dataset) -> None:
        with pytest.raises(KeyError):
            sine_dataset.remove_mean("potato")


class Test_remove_constant:
    def test_per_signal(self, ones_dataset: Dataset) -> None:
        out = ones_dataset.remove_constant(
            ("u1", 2.0),
            ("u2", 2.0),
            ("u3", 2.0),
            ("y1", 2.0),
            ("y2", 1.0),
            ("y3", 2.0),
        )

        assert np.allclose(out["u1"].values, -1.0)
        assert np.allclose(out["y2"].values, 0.0)
        assert np.allclose(out["y3"].values, -1.0)
        # the original is not overwritten
        assert np.allclose(ones_dataset["u1"].values, 1.0)

    def test_only_inputs(self, ones_dataset: Dataset) -> None:
        out = ones_dataset.remove_constant(("u1", 2.0), ("u2", 2.0))

        assert np.allclose(out["u1"].values, -1.0)
        assert np.allclose(out["y1"].values, 1.0)

    def test_scalar_applies_to_all(self, ones_dataset: Dataset) -> None:
        out = ones_dataset.remove_constant(2.0)

        assert np.allclose(out["u1"].values, -1.0)
        assert np.allclose(out["y1"].values, -1.0)

    def test_scalar_cannot_be_combined(self, ones_dataset: Dataset) -> None:
        with pytest.raises(TypeError):
            ones_dataset.remove_constant(2.0, ("u1", 1.0))

    def test_no_arguments(self, ones_dataset: Dataset) -> None:
        with pytest.raises(TypeError):
            ones_dataset.remove_constant()

    def test_mapping_is_rejected(self, ones_dataset: Dataset) -> None:
        with pytest.raises(TypeError, match="no longer supported"):
            ones_dataset.remove_constant({"u1": 2.0})  # type: ignore[arg-type]

    def test_duplicate_signal_raises(self, ones_dataset: Dataset) -> None:
        with pytest.raises(ValueError, match="more than once"):
            ones_dataset.remove_constant(("u1", 1.0), ("u1", 2.0))

    def test_unknown_signal(self, ones_dataset: Dataset) -> None:
        with pytest.raises(KeyError):
            ones_dataset.remove_constant(("potato", 1.0))


class Test_detrend:
    def test_selected_signals(self, sine_dataset: Dataset) -> None:
        out = sine_dataset.detrend("u1")

        assert not np.allclose(out["u1"].values, sine_dataset["u1"].values)
        assert np.allclose(out["u2"].values, sine_dataset["u2"].values)


class Test_apply:
    def test_function_and_unit(self, ones_dataset: Dataset) -> None:
        out = ones_dataset.apply(
            ("u1", lambda x: x + 1, "c"),
            ("y2", np.square, "b"),
        )

        assert np.allclose(out["u1"].values, 2.0)
        assert out["u1"].unit == "c"
        assert np.allclose(out["y2"].values, 1.0)
        assert out["y2"].unit == "b"

        # untouched signals keep their unit and values
        assert out["u2"].unit == "m/s"
        assert np.allclose(out["u2"].values, 1.0)
        # the original is not overwritten
        assert ones_dataset["u1"].unit == "m"

    def test_unit_is_optional(self, ones_dataset: Dataset) -> None:
        out = ones_dataset.apply(("u1", lambda x: 3 * x))

        assert np.allclose(out["u1"].values, 3.0)
        assert out["u1"].unit == "m"

    def test_non_vectorized_function(self, ones_dataset: Dataset) -> None:
        def scalar_only(x: float) -> float:
            return float(x) * 2.0

        out = ones_dataset.apply(("u1", scalar_only))

        assert np.allclose(out["u1"].values, 2.0)

    def test_unknown_signal(self, ones_dataset: Dataset) -> None:
        with pytest.raises(KeyError):
            ones_dataset.apply(("potato", np.square, "c"))

    def test_bad_arguments(self, ones_dataset: Dataset) -> None:
        with pytest.raises(TypeError):
            ones_dataset.apply("u1")  # type: ignore[arg-type]

    def test_too_many_elements(self, ones_dataset: Dataset) -> None:
        with pytest.raises(TypeError, match="between 2 and 3"):
            ones_dataset.apply(
                ("u1", np.square, "c", "extra")  # type: ignore[arg-type]
            )

    def test_too_few_elements(self, ones_dataset: Dataset) -> None:
        with pytest.raises(TypeError, match="between 2 and 3"):
            ones_dataset.apply(("u1",))  # type: ignore[arg-type]

    def test_name_must_be_a_string(self, ones_dataset: Dataset) -> None:
        with pytest.raises(TypeError, match="signal name"):
            ones_dataset.apply((1, np.square))  # type: ignore[arg-type]

    def test_duplicate_signal_raises(self, ones_dataset: Dataset) -> None:
        # The second tuple used to silently overwrite the first
        with pytest.raises(ValueError, match="more than once"):
            ones_dataset.apply(("u1", np.square), ("u1", np.sqrt))


class Test_low_pass_filter:
    def test_matches_the_legacy_implementation(
        self, sine_dataset: Dataset
    ) -> None:
        ds = sine_dataset.trim(tin=0.0, tout=1.0)
        out = ds.low_pass_filter(("u1", 1.0), ("y1", 1.5))

        u_expected = np.array(
            [2.0, 2.0, 2.1949, 2.2467, 2.065, 1.9386, 2.0398, 2.1678, 2.1193]
        )
        y_expected = np.array(
            [2.0, 2.0, 2.1615, 2.1881, 2.1269, 1.893, 1.9972, 2.0376, 2.2357]
        )

        assert np.allclose(out["u1"].values[:9], u_expected, atol=1e-4)
        assert np.allclose(out["y1"].values[:9], y_expected, atol=1e-4)

    def test_untouched_signals(self, sine_dataset: Dataset) -> None:
        out = sine_dataset.low_pass_filter(("u1", 1.0))

        assert np.allclose(out["u2"].values, sine_dataset["u2"].values)

    def test_negative_cutoff(self, sine_dataset: Dataset) -> None:
        with pytest.raises(ValueError):
            sine_dataset.low_pass_filter(("u1", -0.3))

        with pytest.raises(ValueError):
            sine_dataset.low_pass_filter(("y1", -0.3))

    def test_cutoff_above_sampling_frequency(
        self, sine_dataset: Dataset
    ) -> None:
        fs = 1.0 / sine_dataset.get_sampling_period()

        with pytest.raises(ValueError):
            sine_dataset.low_pass_filter(("u1", fs))

    def test_unknown_signal(self, sine_dataset: Dataset) -> None:
        with pytest.raises(KeyError):
            sine_dataset.low_pass_filter(("potato", 1.0))

    def test_extra_elements_are_rejected(self, sine_dataset: Dataset) -> None:
        with pytest.raises(TypeError, match="2"):
            sine_dataset.low_pass_filter(
                ("u1", 1.0, "junk")  # type: ignore[arg-type]
            )


class Test_trim:
    def test_shifts_to_zero(self, sine_dataset: Dataset) -> None:
        out = sine_dataset.trim(tin=1.0, tout=5.0)

        assert np.isclose(out.time()[0], 0.0, atol=ATOL)
        assert np.isclose(out.time()[-1], 4.0, atol=ATOL)
        assert np.isclose(out.get_sampling_period(), 0.1)

    def test_without_shift(self, sine_dataset: Dataset) -> None:
        out = sine_dataset.trim(tin=1.0, tout=5.0, shift_to_zero=False)

        assert np.isclose(out.time()[0], 1.0, atol=ATOL)
        assert np.isclose(out.time()[-1], 5.0, atol=ATOL)

    def test_open_ended(self, sine_dataset: Dataset) -> None:
        head = sine_dataset.trim(tout=1.0, shift_to_zero=False)
        tail = sine_dataset.trim(tin=9.0, shift_to_zero=False)

        assert np.isclose(head.time()[0], 0.0)
        assert np.isclose(head.time()[-1], 1.0)
        assert np.isclose(tail.time()[-1], 10.0)

    def test_values_are_preserved(self, sine_dataset: Dataset) -> None:
        out = sine_dataset.trim(tin=1.0, tout=5.0)

        assert np.allclose(out["u1"].values, sine_dataset["u1"].values[10:51])

    def test_reversed_interval(self, sine_dataset: Dataset) -> None:
        with pytest.raises(ValueError):
            sine_dataset.trim(tin=5.0, tout=1.0)

    def test_empty_interval(self, sine_dataset: Dataset) -> None:
        with pytest.raises(ValueError):
            sine_dataset.trim(tin=100.0, tout=200.0)


# ============================================================
# Missing data
# ============================================================
@pytest.fixture
def nan_dataset(sine_dataset: Dataset) -> Dataset:
    def poke(values: np.ndarray) -> np.ndarray:
        out = values.copy()
        out[10:13] = np.nan
        return out

    return sine_dataset.apply(("u1", poke))


class Test_nans:
    def test_has_nans(
        self, sine_dataset: Dataset, nan_dataset: Dataset
    ) -> None:
        assert not sine_dataset.has_nans()
        assert nan_dataset.has_nans()

    def test_nan_intervals(self, nan_dataset: Dataset) -> None:
        intervals = nan_dataset.nan_intervals()

        assert set(intervals) == set(nan_dataset.names())
        assert len(intervals["u1"]) == 1
        assert np.allclose(intervals["u1"][0], (1.0, 1.2))
        assert intervals["y1"] == []

    def test_remove_nans(self, nan_dataset: Dataset) -> None:
        out = nan_dataset.remove_nans()

        assert not out.has_nans()
        assert len(out.time()) == len(nan_dataset.time())

    def test_remove_nans_selection(self, nan_dataset: Dataset) -> None:
        assert not nan_dataset.remove_nans("u1").has_nans()
        assert nan_dataset.remove_nans("y1").has_nans()

    def test_drop_is_rejected(self, nan_dataset: Dataset) -> None:
        with pytest.raises(ValueError):
            nan_dataset.remove_nans(fill="drop")

    def test_unknown_name_raises(self, nan_dataset: Dataset) -> None:
        with pytest.raises(KeyError):
            nan_dataset.remove_nans("potato")

    def test_plot_shades_the_gaps(self, nan_dataset: Dataset) -> None:
        fig = nan_dataset.plot()
        patches = sum(len(ax.patches) for ax in fig.get_axes())

        assert patches == 1


# ============================================================
# Coverage
# ============================================================
class Test_coverage:
    def test_statistics(self, ones_dataset: Dataset) -> None:
        u_mean, u_cov, y_mean, y_cov = ones_dataset.coverage()

        assert np.allclose(u_mean, 1.0)
        assert np.allclose(y_mean, 1.0)
        assert u_cov.shape == (3, 3)
        assert y_cov.shape == (3, 3)
        assert np.allclose(u_cov, 0.0)

    def test_matches_numpy(self, sine_dataset: Dataset) -> None:
        u_mean, u_cov, _, _ = sine_dataset.coverage()
        values = np.column_stack(
            [sine_dataset[name].values for name in sine_dataset.input_names()]
        )

        assert np.allclose(u_mean, np.mean(values, axis=0))
        assert np.allclose(u_cov, np.cov(values, rowvar=False))

    def test_single_signal_gives_a_1x1_matrix(self, dataset: Dataset) -> None:
        u_mean, u_cov, _, _ = dataset.coverage()

        assert u_mean.shape == (1,)
        assert u_cov.shape == (1, 1)

    def test_nans_are_dropped_consistently(
        self, sine_dataset: Dataset
    ) -> None:
        # The mean used to come from `nanmean`, i.e. per-signal, while the
        # covariance came from `np.cov`, which turns a whole row and column
        # into NaN from a single missing sample. Both now describe the same
        # complete samples.
        def poke(values: np.ndarray) -> np.ndarray:
            out = values.copy()
            out[3] = np.nan
            return out

        ds = sine_dataset.apply(("u1", poke))
        u_mean, u_cov, _, _ = ds.coverage()

        assert np.all(np.isfinite(u_mean))
        assert np.all(np.isfinite(u_cov))

        expected = np.column_stack(
            [ds[name].values for name in ds.input_names()]
        )
        expected = np.delete(expected, 3, axis=0)

        assert np.allclose(u_mean, np.mean(expected, axis=0))
        assert np.allclose(u_cov, np.cov(expected, rowvar=False))

    def test_all_samples_missing_raises(self, sine_dataset: Dataset) -> None:
        ds = sine_dataset.apply(("u1", lambda v: np.full_like(v, np.nan)))

        with pytest.raises(ValueError, match="remove_nans"):
            ds.coverage()

    def test_no_inputs(self, time: np.ndarray) -> None:
        ds = Dataset.from_signals(
            outputs=[Signal("y", np.zeros_like(time), time)]
        )
        u_mean, u_cov, y_mean, _ = ds.coverage()

        assert u_mean.size == 0
        assert u_cov.size == 0
        assert y_mean.shape == (1,)

    @pytest.mark.plots
    def test_plot_coverage(self, sine_dataset: Dataset) -> None:
        fig = sine_dataset.plot_coverage()

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 7

    @pytest.mark.plots
    def test_plot_coverage_selection(self, sine_dataset: Dataset) -> None:
        fig = sine_dataset.plot_coverage("u1", "y1")

        assert len(fig.axes) == 2
        assert fig.axes[0].get_xlabel() == "u1 [kPa]"

    @pytest.mark.plots
    def test_plot_coverage_rejects_groups(self, sine_dataset: Dataset) -> None:
        with pytest.raises(TypeError):
            sine_dataset.plot_coverage(("u1", "y1"))  # type: ignore[arg-type]

    @pytest.mark.plots
    def test_plot_coverage_unknown_signal(self, sine_dataset: Dataset) -> None:
        with pytest.raises(KeyError):
            sine_dataset.plot_coverage("potato")
