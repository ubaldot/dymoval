# -*- coding: utf-8 -*-
"""Tests for :class:`dymoval.signal.Signal`."""

from __future__ import annotations

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
