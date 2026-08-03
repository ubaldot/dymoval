# -*- coding: utf-8 -*-
"""Tests for :class:`dymoval.signal.Signal`."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from dymoval import SPECTRUM_MODES, Signal


# ============================================================
# Construction
# ============================================================
class Test_construction:
    def test_nominal(self, signal: Signal) -> None:
        assert signal.name == "u1"
        assert len(signal) == len(signal.values)
        assert signal.unit == "V"
        assert signal.time_unit == "s"

    def test_values_must_be_ndarray(self) -> None:
        with pytest.raises(TypeError):
            Signal(name="u", values=[1, 2, 3])  # type: ignore[arg-type]

    def test_values_must_be_1d(self) -> None:
        with pytest.raises(ValueError):
            Signal(name="u", values=np.zeros((2, 2)))

    def test_time_must_be_ndarray(self) -> None:
        with pytest.raises(TypeError):
            Signal(
                name="u",
                values=np.zeros(3),
                time=[0, 1, 2],  # type: ignore[arg-type]
            )

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            Signal(name="u", values=np.zeros(3), time=np.zeros(4))

    def test_time_is_optional(self) -> None:
        sig = Signal(name="u", values=np.zeros(3))
        assert sig.time is None


# ============================================================
# Sampling period
# ============================================================
class Test_sampling_period:
    def test_nominal(self, signal: Signal) -> None:
        assert np.isclose(signal.get_sampling_period(), 0.01)

    def test_no_time(self) -> None:
        with pytest.raises(ValueError):
            Signal(name="u", values=np.zeros(3)).get_sampling_period()

    def test_not_increasing(self) -> None:
        sig = Signal(
            name="u", values=np.zeros(3), time=np.array([0.0, 2.0, 1.0])
        )
        with pytest.raises(ValueError):
            sig.get_sampling_period()

    def test_non_uniform(self) -> None:
        sig = Signal(
            name="u", values=np.zeros(3), time=np.array([0.0, 1.0, 3.0])
        )
        with pytest.raises(ValueError):
            sig.get_sampling_period()

    def test_too_short(self) -> None:
        sig = Signal(name="u", values=np.zeros(1), time=np.zeros(1))
        with pytest.raises(ValueError):
            sig.get_sampling_period()


# ============================================================
# Processing
# ============================================================
class Test_processing:
    def test_copy_is_deep(self, signal: Signal) -> None:
        other = signal.copy()
        other.values[0] = 12345.0

        assert signal.values[0] != 12345.0
        assert other.time is not signal.time

    def test_remove_mean(self, signal: Signal) -> None:
        out = signal.remove_mean()

        assert np.isclose(np.mean(out.values), 0.0)
        # immutability
        assert not np.isclose(np.mean(signal.values), 0.0)
        assert out.name == signal.name
        assert out.unit == signal.unit

    def test_remove_constant(self, signal: Signal) -> None:
        out = signal.remove_constant(2.0)

        assert np.allclose(out.values, signal.values - 2.0)

    def test_detrend(self, time: np.ndarray) -> None:
        sig = Signal(name="u", values=3.0 * time + 1.0, time=time)

        assert np.allclose(sig.detrend().values, 0.0, atol=1e-9)

    def test_resample(self, signal: Signal) -> None:
        assert signal.time is not None
        new_time = signal.time[::2]
        out = signal.resample(new_time)

        assert len(out) == len(new_time)
        assert np.allclose(out.values, signal.values[::2])

    def test_resample_without_time(self) -> None:
        sig = Signal(name="u", values=np.zeros(4))

        with pytest.raises(ValueError):
            sig.resample(np.arange(4.0))


# ============================================================
# apply / low_pass_filter / trim
# ============================================================
class Test_apply:
    def test_vectorized(self, signal: Signal) -> None:
        out = signal.apply(np.square, unit="V^2")

        assert np.allclose(out.values, signal.values**2)
        assert out.unit == "V^2"
        assert out.name == signal.name

    def test_unit_is_kept_by_default(self, signal: Signal) -> None:
        assert signal.apply(lambda x: 2 * x).unit == "V"

    def test_scalar_function(self, signal: Signal) -> None:
        out = signal.apply(lambda x: float(x) + 1.0)

        assert np.allclose(out.values, signal.values + 1.0)

    def test_shape_changing_function_falls_back(self, signal: Signal) -> None:
        # wrong output shape => element-wise fallback, which here fails too
        with pytest.raises((TypeError, IndexError)):
            signal.apply(lambda x: x[:2])

    def test_reducing_function_is_applied_element_wise(
        self, signal: Signal
    ) -> None:
        # np.sum() would collapse the array, so the element-wise fallback
        # kicks in and np.sum(scalar) is the identity
        out = signal.apply(np.sum)

        assert np.allclose(out.values, signal.values)


class Test_low_pass_filter:
    def test_recursion(self) -> None:
        t = np.arange(0.0, 1.0, 0.1)
        sig = Signal(name="u", values=np.arange(10.0), time=t)

        out = sig.low_pass_filter(1.0)

        fs = 10.0
        alpha = 1.0 / fs
        expected = np.empty(10)
        expected[0] = 0.0

        for k in range(9):
            expected[k + 1] = (1 - alpha) * expected[k] + alpha * k

        assert np.allclose(out.values, expected)
        assert out.name == "u"

    def test_dc_gain_is_one(self) -> None:
        t = np.arange(0.0, 100.0, 0.1)
        sig = Signal(name="u", values=np.full_like(t, 5.0), time=t)

        assert np.allclose(sig.low_pass_filter(1.0).values, 5.0)

    def test_attenuates_high_frequencies(self) -> None:
        t = np.arange(0.0, 10.0, 0.01)
        sig = Signal(
            name="u", values=np.sin(2 * np.pi * 40 * t), time=t, unit="V"
        )

        out = sig.low_pass_filter(1.0)

        assert np.max(np.abs(out.values)) < 0.2

    @pytest.mark.parametrize("cutoff", [0.0, -1.0])
    def test_non_positive_cutoff(self, signal: Signal, cutoff: float) -> None:
        with pytest.raises(ValueError):
            signal.low_pass_filter(cutoff)

    def test_cutoff_above_sampling_frequency(self, signal: Signal) -> None:
        fs = 1.0 / signal.get_sampling_period()

        with pytest.raises(ValueError):
            signal.low_pass_filter(fs)


class Test_trim:
    def test_nominal(self) -> None:
        t = np.arange(0.0, 1.0, 0.1)
        sig = Signal(name="u", values=np.arange(10.0), time=t)

        out = sig.trim(0.2, 0.5)

        assert np.allclose(out.values, [2.0, 3.0, 4.0, 5.0])
        assert np.isclose(out.time[0], 0.0)

    def test_without_shift(self) -> None:
        t = np.arange(0.0, 1.0, 0.1)
        sig = Signal(name="u", values=np.arange(10.0), time=t)

        out = sig.trim(0.2, 0.5, shift_to_zero=False)

        assert np.isclose(out.time[0], 0.2)

    def test_open_ended(self) -> None:
        t = np.arange(0.0, 1.0, 0.1)
        sig = Signal(name="u", values=np.arange(10.0), time=t)

        assert len(sig.trim(tout=0.5)) == 6
        assert len(sig.trim(tin=0.5)) == 5

    def test_reversed(self, signal: Signal) -> None:
        with pytest.raises(ValueError):
            signal.trim(2.0, 1.0)

    def test_empty(self, signal: Signal) -> None:
        with pytest.raises(ValueError):
            signal.trim(100.0, 200.0)

    def test_without_time(self) -> None:
        sig = Signal(name="u", values=np.zeros(4))

        with pytest.raises(ValueError):
            sig.trim(0.0, 1.0)


# ============================================================
# Missing data
# ============================================================
class Test_nans:
    @staticmethod
    def _sig(values: list[float]) -> Signal:
        return Signal(
            "s", np.array(values), time=np.arange(len(values), dtype=float)
        )

    def test_no_nans(self) -> None:
        sig = self._sig([1.0, 2.0, 3.0])

        assert not sig.has_nans()
        assert sig.nan_intervals() == []

    def test_single_interval(self) -> None:
        sig = self._sig([1.0, np.nan, np.nan, 4.0])

        assert sig.has_nans()
        assert sig.nan_intervals() == [(1.0, 2.0)]

    def test_isolated_nan_is_degenerate(self) -> None:
        assert self._sig([1.0, np.nan, 3.0]).nan_intervals() == [(1.0, 1.0)]

    def test_several_intervals(self) -> None:
        sig = self._sig([np.nan, 2.0, np.nan, np.nan, 5.0, np.nan])

        assert sig.nan_intervals() == [(0.0, 0.0), (2.0, 3.0), (5.0, 5.0)]

    def test_intervals_without_time_are_samples(self) -> None:
        sig = Signal("s", np.array([1.0, np.nan, 3.0]))

        assert sig.nan_intervals() == [(1.0, 1.0)]

    def test_interpolate(self) -> None:
        out = self._sig([1.0, np.nan, np.nan, 4.0]).remove_nans()

        assert not out.has_nans()
        np.testing.assert_allclose(out.values, [1.0, 2.0, 3.0, 4.0])

    def test_interpolate_holds_the_edges(self) -> None:
        out = self._sig([np.nan, 2.0, 3.0, np.nan]).remove_nans()

        np.testing.assert_allclose(out.values, [2.0, 2.0, 3.0, 3.0])

    def test_drop(self) -> None:
        out = self._sig([1.0, np.nan, 3.0]).remove_nans(fill="drop")

        np.testing.assert_allclose(out.values, [1.0, 3.0])
        assert out.time is not None
        np.testing.assert_allclose(out.time, [0.0, 2.0])

    def test_all_nans_raises(self) -> None:
        with pytest.raises(ValueError):
            self._sig([np.nan, np.nan]).remove_nans()

    def test_bad_fill_raises(self) -> None:
        with pytest.raises(ValueError):
            self._sig([1.0, np.nan]).remove_nans(fill="banana")  # type: ignore[arg-type]

    def test_shading(self) -> None:
        ax = self._sig([1.0, np.nan, np.nan, 4.0])._plot_standard()

        assert len(ax.patches) == 1

    def test_shading_can_be_disabled(self) -> None:
        sig = self._sig([1.0, np.nan, np.nan, 4.0])

        assert len(sig._plot_standard(shade_nans=False).patches) == 0


# ============================================================
# Frequency domain
# ============================================================
class Test_spectrum:
    def test_fft_shape(self, signal: Signal) -> None:
        freq, y = signal.fft()

        assert freq.shape == y.shape
        assert len(freq) == len(signal) // 2 + 1
        assert np.iscomplexobj(y)

    @pytest.mark.parametrize("mode", SPECTRUM_MODES)
    def test_modes(self, signal: Signal, mode: str) -> None:
        freq, spectrum = signal.spectrum(mode)  # type: ignore[arg-type]

        assert freq.shape == spectrum.shape
        assert np.all(spectrum >= 0.0)

    def test_invalid_mode(self, signal: Signal) -> None:
        with pytest.raises(ValueError):
            signal.spectrum("banana")  # type: ignore[arg-type]

    def test_amplitude_peak(self, time: np.ndarray) -> None:
        sig = Signal(name="u", values=np.sin(2 * np.pi * 5 * time), time=time)
        freq, magnitude = sig.spectrum("amplitude")

        assert np.isclose(freq[int(np.argmax(magnitude))], 5.0, atol=0.5)

    def test_phase_is_masked_and_in_degrees(self, signal: Signal) -> None:
        _, magnitude, phase = signal._compute_spectrum("amplitude")

        assert phase is not None
        assert np.any(np.isnan(phase))  # negligible magnitudes are masked
        assert np.all(np.isnan(phase[magnitude <= 0.05 * magnitude.max()]))

        finite = phase[~np.isnan(phase)]
        assert np.all(np.abs(finite) <= 360.0 * len(phase))

    def test_only_amplitude_returns_phase(self, signal: Signal) -> None:
        for mode in ("power", "psd", "psd_welch"):
            _, _, phase = signal._compute_spectrum(mode)  # type: ignore[arg-type]
            assert phase is None


# ============================================================
# Plotting
# ============================================================
class Test_plot:
    @pytest.mark.plots
    def test_standard_returns_axes(self, signal: Signal) -> None:
        ax = signal.plot(with_scope=False)

        assert isinstance(ax, Axes)
        assert len(ax.get_lines()) == 1
        assert ax.get_lines()[0]._signal is signal

    @pytest.mark.plots
    def test_without_time_uses_samples(self) -> None:
        sig = Signal(name="u", values=np.arange(10.0))
        ax = signal_ax = sig.plot(with_scope=False)

        assert signal_ax.get_xlabel() == "samples"
        assert isinstance(ax, Axes)

    @pytest.mark.plots
    def test_scope_returns_figure(self, signal: Signal) -> None:
        fig = signal.plot(with_scope=True)

        assert isinstance(fig, Figure)
        assert len(fig._scopes) == 1

    @pytest.mark.plots
    def test_passing_ax_draws_on_it(self, signal: Signal) -> None:
        _, ax = plt.subplots()
        out = signal.plot(ax=ax)

        assert out is ax
        assert len(ax.get_lines()) == 1

    @pytest.mark.plots
    def test_passing_ax_and_scope_raises(self, signal: Signal) -> None:
        _, ax = plt.subplots()

        with pytest.raises(ValueError):
            signal.plot(ax=ax, with_scope=True)

        with pytest.raises(ValueError):
            signal.plot_spectrum(ax=ax, with_scope=True)

    @pytest.mark.plots
    def test_spectrum_passing_ax_draws_on_it(self, signal: Signal) -> None:
        _, ax = plt.subplots()
        out = signal.plot_spectrum(ax=ax)

        assert out is ax
        assert len(ax.get_lines()) == 1

    @pytest.mark.plots
    def test_bad_scale_raises(self, signal: Signal) -> None:
        with pytest.raises(ValueError):
            signal.plot_spectrum(yscale="banana")  # type: ignore[arg-type]

        with pytest.raises(ValueError):
            signal.plot_spectrum(xscale="banana")  # type: ignore[arg-type]

    @pytest.mark.plots
    @pytest.mark.parametrize("mode", SPECTRUM_MODES)
    @pytest.mark.parametrize("with_scope", [False, True])
    def test_plot_spectrum(
        self, signal: Signal, mode: str, with_scope: bool
    ) -> None:
        out = signal.plot_spectrum(mode=mode, with_scope=with_scope)  # type: ignore[arg-type]

        if mode == "amplitude" or with_scope:
            assert isinstance(out, Figure)
        else:
            assert isinstance(out, Axes)

    @pytest.mark.plots
    def test_amplitude_has_magnitude_and_phase(self, signal: Signal) -> None:
        fig = signal.plot_spectrum(mode="amplitude", with_scope=False)

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2
        assert fig.axes[1].get_ylabel() == "Phase [deg]"

    @pytest.mark.plots
    def test_amplitude_rejects_ax(self, signal: Signal) -> None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()

        with pytest.raises(ValueError):
            signal.plot_spectrum(ax=ax, mode="amplitude", with_scope=False)

    @pytest.mark.plots
    def test_db_scale(self, signal: Signal) -> None:
        ax = signal.plot_spectrum(mode="psd", yscale="db", with_scope=False)

        assert isinstance(ax, Axes)
        assert ax.get_ylabel() == "Power [dB]"
