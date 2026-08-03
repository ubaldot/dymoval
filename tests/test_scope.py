# -*- coding: utf-8 -*-
"""Tests for the interactive scopes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dymoval import Dataset, Signal
from dymoval.scope import AmplitudeSpectrumScope, _fmt


def _click(scope, ax, x, y):  # type: ignore[no-untyped-def]
    scope._on_click(SimpleNamespace(inaxes=ax, xdata=x, ydata=y))


def _press(scope, key="r"):  # type: ignore[no-untyped-def]
    scope._on_key(SimpleNamespace(key=key))


def _plot_axes(fig):  # type: ignore[no-untyped-def]
    return [ax for ax in fig.axes if ax.get_lines()]


# ============================================================
# Formatting helper
# ============================================================
class Test_fmt:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (1.5, "1.500"),
            (0.0, "0.000"),
            (np.nan, "n/a"),
            (None, "n/a"),
        ],
    )
    def test_values(self, value: float, expected: str) -> None:
        assert _fmt(value) == expected

    def test_scientific_for_extreme_values(self) -> None:
        assert "e" in _fmt(1e-6)
        assert "e" in _fmt(1e9)

    def test_unit_is_appended(self) -> None:
        assert _fmt(1.0, "V") == "1.000 V"
        assert _fmt(1.0, "") == "1.000"


# ============================================================
# Lifetime and shared panel
# ============================================================
class Test_lifetime:
    @pytest.mark.plots
    def test_scopes_are_stored_on_the_figure(self, signal: Signal) -> None:
        fig = signal.plot(with_scope=True)

        assert isinstance(fig._scopes, list)
        assert len(fig._scopes) == 1
        assert fig._scopes[0].fig is fig

    @pytest.mark.plots
    def test_single_shared_text_artist(self, dataset: Dataset) -> None:
        fig = dataset.plot_spectrum(
            "u1", "y0", mode="amplitude", with_scope=True
        )
        scopes = fig._scopes

        assert len(scopes) == 2

        panel_ax = scopes[0].panel_ax
        texts = [t for t in panel_ax.texts]

        assert len(texts) == 1
        assert all(scope.info_text is texts[0] for scope in scopes)

    @pytest.mark.plots
    def test_axes_are_normalized_to_list(self, dataset: Dataset) -> None:
        fig = dataset.plot(with_scope=True)
        scope = fig._scopes[0]

        assert isinstance(scope.axes, list)
        assert len(scope.axes) == 3


# ============================================================
# Click behavior
# ============================================================
class Test_click:
    @pytest.mark.plots
    def test_first_click_reports_one_point(self, signal: Signal) -> None:
        fig = signal.plot(with_scope=True)
        scope = fig._scopes[0]
        ax = scope.axes[0]

        _click(scope, ax, 0.1, signal.values[10])

        assert len(scope.clicks) == 1
        assert len(scope.cursor_lines) == 1
        assert "Select second point" in scope.info_text.get_text()

    @pytest.mark.plots
    def test_second_click_reports_statistics(self, signal: Signal) -> None:
        fig = signal.plot(with_scope=True)
        scope = fig._scopes[0]
        ax = scope.axes[0]

        _click(scope, ax, 0.1, signal.values[10])
        _click(scope, ax, 0.5, signal.values[50])

        text = scope.info_text.get_text()

        assert len(scope.clicks) == 2
        assert "RMS" in text
        assert "\u0394t" in text

    @pytest.mark.plots
    def test_only_two_selections_are_kept(self, signal: Signal) -> None:
        fig = signal.plot(with_scope=True)
        scope = fig._scopes[0]
        ax = scope.axes[0]

        for x in (0.1, 0.2, 0.3, 0.4):
            _click(scope, ax, x, 0.0)

        assert len(scope.clicks) == 2
        assert len(scope.cursor_lines) == 2
        assert len(scope.cursor_points) == 2

    @pytest.mark.plots
    def test_click_outside_is_ignored(self, signal: Signal) -> None:
        fig = signal.plot(with_scope=True)
        scope = fig._scopes[0]

        _click(scope, None, 0.1, 0.0)
        _click(scope, scope.axes[0], None, 0.0)

        assert scope.clicks == []

    @pytest.mark.plots
    def test_selected_line_is_highlighted(self, dataset: Dataset) -> None:
        fig = dataset.plot(("u1", "y1"), with_scope=True)
        scope = fig._scopes[0]
        ax = _plot_axes(fig)[0]

        _click(scope, ax, 0.1, dataset["u1"].values[10])

        widths = [line.get_linewidth() for line in ax.get_lines()[:2]]

        assert 3 in widths
        assert scope.current_line.get_linewidth() == 3

    @pytest.mark.plots
    def test_nearest_line_is_selected(self, dataset: Dataset) -> None:
        fig = dataset.plot(("u1", "y1"), with_scope=True)
        scope = fig._scopes[0]
        ax = _plot_axes(fig)[0]

        _click(scope, ax, 0.13, dataset["y1"].values[13])

        assert scope.current_line._signal.name == "y1"

    @pytest.mark.plots
    def test_statistics_over_the_window(self, signal: Signal) -> None:
        fig = signal.plot(with_scope=True)
        scope = fig._scopes[0]

        _click(scope, scope.axes[0], 0.0, signal.values[0])
        _click(scope, scope.axes[0], 1.0, signal.values[100])

        assert signal.time is not None
        mask = (signal.time >= 0.0) & (signal.time <= 1.0)
        expected = (
            np.min(signal.values[mask]),
            np.max(signal.values[mask]),
            np.sqrt(np.mean(signal.values[mask] ** 2)),
        )

        assert np.allclose(scope._compute_stats(0.0, 1.0), expected)


# ============================================================
# Reset
# ============================================================
class Test_reset:
    @pytest.mark.plots
    def test_reset_clears_state(self, signal: Signal) -> None:
        fig = signal.plot(with_scope=True)
        scope = fig._scopes[0]

        _click(scope, scope.axes[0], 0.1, 0.0)
        _press(scope)

        assert scope.clicks == []
        assert scope.cursor_lines == []
        assert scope.cursor_points == []
        assert scope.current_line is None
        assert "Reset" in scope.info_text.get_text()

    @pytest.mark.plots
    def test_reset_restores_linewidths(self, dataset: Dataset) -> None:
        fig = dataset.plot(("u1", "y1"), with_scope=True)
        scope = fig._scopes[0]
        ax = _plot_axes(fig)[0]

        original = [line.get_linewidth() for line in ax.get_lines()]

        _click(scope, ax, 0.1, dataset["u1"].values[10])
        _press(scope)

        assert [line.get_linewidth() for line in ax.get_lines()] == original

    @pytest.mark.plots
    def test_reset_is_global(self, dataset: Dataset) -> None:
        fig = dataset.plot_spectrum(
            "u1", "y0", mode="amplitude", with_scope=True
        )
        scope_a, scope_b = fig._scopes

        _click(scope_a, scope_a.mag_ax, 5.0, 1.0)
        _click(scope_b, scope_b.mag_ax, 2.0, 1.0)

        assert scope_a.clicks and scope_b.clicks

        # pressing 'r' on one scope must reset every scope of the figure
        _press(scope_a)

        assert scope_a.clicks == []
        assert scope_b.clicks == []

    @pytest.mark.plots
    def test_other_keys_are_ignored(self, signal: Signal) -> None:
        fig = signal.plot(with_scope=True)
        scope = fig._scopes[0]

        _click(scope, scope.axes[0], 0.1, 0.0)
        _press(scope, key="q")

        assert len(scope.clicks) == 1


# ============================================================
# Spectrum scopes
# ============================================================
class Test_spectrum_scope:
    @pytest.mark.plots
    def test_frequency_symbols(self, signal: Signal) -> None:
        fig = signal.plot_spectrum(mode="psd_welch", with_scope=True)
        scope = fig._scopes[0]
        ax = _plot_axes(fig)[0]

        _click(scope, ax, 5.0, 0.0)

        assert "f1" in scope.info_text.get_text()

    @pytest.mark.plots
    def test_amplitude_scope_links_magnitude_and_phase(
        self, signal: Signal
    ) -> None:
        fig = signal.plot_spectrum(mode="amplitude", with_scope=True)
        scope = fig._scopes[0]

        assert isinstance(scope, AmplitudeSpectrumScope)

        _click(scope, scope.mag_ax, 5.0, 0.0)

        # one cursor on the magnitude axes, one on the phase axes
        assert len(scope.cursor_lines) == 2
        assert {line.axes for line in scope.cursor_lines} == {
            scope.mag_ax,
            scope.phase_ax,
        }

        x_positions = {line.get_xdata()[0] for line in scope.cursor_lines}
        assert len(x_positions) == 1

        text = scope.info_text.get_text()
        assert "|A1|" in text
        assert "\u2220A1" in text

    @pytest.mark.plots
    def test_clicking_phase_axes_uses_magnitude_line(
        self, signal: Signal
    ) -> None:
        fig = signal.plot_spectrum(mode="amplitude", with_scope=True)
        scope = fig._scopes[0]

        _click(scope, scope.phase_ax, 5.0, 0.0)

        assert scope.current_line.axes is scope.mag_ax

    @pytest.mark.plots
    def test_amplitude_scope_two_points(self, signal: Signal) -> None:
        fig = signal.plot_spectrum(mode="amplitude", with_scope=True)
        scope = fig._scopes[0]

        _click(scope, scope.mag_ax, 5.0, 0.0)
        _click(scope, scope.mag_ax, 10.0, 0.0)

        text = scope.info_text.get_text()

        assert len(scope.clicks) == 2
        assert len(scope.phases) == 2
        assert "\u0394f" in text
        assert "\u0394\u2220A" in text

        # two selections => two cursors per axes
        assert len(scope.cursor_lines) == 4

    @pytest.mark.plots
    def test_amplitude_scope_reset_clears_phases(self, signal: Signal) -> None:
        fig = signal.plot_spectrum(mode="amplitude", with_scope=True)
        scope = fig._scopes[0]

        _click(scope, scope.mag_ax, 5.0, 0.0)
        _press(scope)

        assert scope.phases == []

    @pytest.mark.plots
    def test_cursors_are_not_selectable(self, signal: Signal) -> None:
        fig = signal.plot(with_scope=True)
        scope = fig._scopes[0]
        ax = scope.axes[0]

        _click(scope, ax, 0.1, signal.values[10])
        _click(scope, ax, 0.2, signal.values[20])

        # the cursor artifacts must never become the current selection
        assert scope.current_line._signal is signal
