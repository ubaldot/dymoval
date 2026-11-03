# -*- coding: utf-8 -*-
"""Tests for the ValidationSession class."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib import cycler
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba

import dymoval as dmv
from dymoval.signal import Signal
from dymoval.validation import validate_models


def test_gaussian_residual_envelope_scores_thresholds_and_plots() -> None:
    rng = np.random.default_rng(72)
    u = rng.normal(size=(200, 1))
    y = rng.normal(size=(200, 2))
    error = rng.normal(size=(200, 2))
    error[1:] += 0.6 * error[:-1]
    parameters = {"y0": (0.95, 0.1, 2.0), "y1": (0.8, 0.2, 4.0)}
    vs = validate_models(
        u, y, y - error, sampling_period=0.5,
        Ree_envelope=parameters,
    )
    expected = []
    for ii, name in enumerate(vs.dataset.output_names()):
        entry = vs.Ree["Sim_0"].R[ii, ii]
        mask = entry.lags != 0
        b0, floor, scale = parameters[name]
        limit = floor + (b0 - floor) * np.exp(
            -(entry.lags[mask] * 0.5 / scale)**2
        )
        expected.append(np.max(np.abs(entry.values[mask]) / limit))
    score = max(expected)
    assert vs.validation_statistics["Sim_0"]["Ree_envelope"] == pytest.approx(score)
    local = vs.Ree["Sim_0"].correlation_score()[1]
    ordinary = validate_models(u, y, y - error, sampling_period=0.5, Ree_envelope=None)
    assert ordinary.validation_statistics["Sim_0"]["Ree_correlation"] == pytest.approx(
        max(local[0, 0], local[1, 1])
    )
    assert ordinary.validation_statistics["Sim_0"]["Ree_cross_correlation"] == pytest.approx(
        max(local[0, 1], local[1, 0])
    )
    assert "Ree_correlation" not in vs.validation_thresholds
    assert ordinary.validation_thresholds["Ree_cross_correlation"] == 0.5
    assert vs.validation_statistics["Sim_0"]["Ree_cross_correlation"] == pytest.approx(
        max(local[0, 1], local[1, 0])
    )
    assert vs.validation_statistics["Sim_0"]["Ree_correlation"] == pytest.approx(
        max(local[0, 0], local[1, 1])
    )
    assert vs.validation_thresholds["Ree_envelope"] == 1
    vs.validation_thresholds = {"Ree_envelope": score}
    assert vs.outcome["Sim_0"] == "PASS"
    vs.validation_thresholds = {"Ree_envelope": score * 0.9}
    assert vs.outcome["Sim_0"] == "FAIL"
    assert "Gaussian limits" in repr(vs)
    figures = vs.plot_residuals(plot_input=False)
    try:
        axes = figures[0].axes
        for index in (0, 3):
            shaded = [c for c in axes[index].collections
                      if c.get_label() == "Gaussian acceptance limits"]
            assert len(shaded) == 1
            vertices = shaded[0].get_paths()[0].vertices
            assert np.max(vertices[:, 1]) == pytest.approx(
                parameters[f"y{index // 3}"][0]
            )
        for index in (1, 2):
            assert not any(c.get_label() == "Gaussian acceptance limits"
                           for c in axes[index].collections)
        assert not any(c.get_label() == "Gaussian acceptance limits"
                       for ax in figures[1].axes for c in ax.collections)
    finally:
        for figure in figures:
            plt.close(figure)
    trimmed = vs.trim(10, 70)
    assert np.isfinite(trimmed.validation_statistics["Sim_0"]["Ree_envelope"])
    assert vs.drop_simulations("Sim_0").validation_statistics == {}


@pytest.mark.parametrize("gaussian", [True, False])
def test_correlation_threshold_reference_bands(gaussian) -> None:
    rng = np.random.default_rng(73)
    u = rng.normal(size=(100, 2))
    y = rng.normal(size=(100, 2))
    vs = validate_models(
        u, y, y + rng.normal(size=y.shape), sampling_period=1,
        Ree_envelope="auto" if gaussian else None,
    )
    thresholds = {
        "Ruu_correlation": 0.21,
        "Rue_correlation": 0.32,
        "Ree_cross_correlation": 0.43,
        "Ree_envelope" if gaussian else "Ree_correlation": 1 if gaussian else 0.54,
    }
    vs.validation_thresholds = thresholds
    label = "Score threshold reference (not a pointwise limit)"

    def check_bands(figures, expected):
        for figure, limits in zip(figures, expected):
            for ax, limit in zip(figure.axes, limits):
                bands = [patch for patch in ax.patches if patch.get_label() == label]
                assert len(bands) == (0 if limit is None else 1)
                if limit is not None:
                    vertices = bands[0].get_patch_transform().transform(
                        bands[0].get_path().vertices
                    )
                    assert vertices[:, 1].min() == pytest.approx(-limit)
                    assert vertices[:, 1].max() == pytest.approx(limit)
                    assert label in [text.get_text() for text in ax.get_legend().texts]

    diagonal = None if gaussian else 0.54
    figures = vs.plot_residuals()
    try:
        check_bands(figures, [
            [0.21] * 4, [diagonal, 0.43, 0.43, diagonal], [0.32] * 4
        ])
    finally:
        for figure in figures:
            plt.close(figure)

    # Changed and omitted checks must be reflected in newly created plots.
    vs.validation_thresholds = {"Rue_correlation": 0.17}
    figures = vs.plot_residuals()
    try:
        check_bands(figures, [[None] * 4, [None] * 4, [0.17] * 4])
    finally:
        for figure in figures:
            plt.close(figure)


def test_shared_envelope_single_output_and_opt_in() -> None:
    rng = np.random.default_rng(73)
    u = rng.normal(size=(100, 1))
    y = rng.normal(size=(100, 1))
    simulated = y + rng.normal(size=y.shape)
    baseline = validate_models(u, y, simulated, sampling_period=1, Ree_envelope=None)
    assert "Ree_envelope" not in baseline.validation_statistics["Sim_0"]
    vs = validate_models(
        u, y, simulated, sampling_period=1, Ree_envelope=(1, 1, 2),
        validation_thresholds={"Ree_envelope": 1},
    )
    assert "Ree_cross_correlation" not in vs.validation_statistics["Sim_0"]
    assert "N/A" in repr(vs)
    with pytest.raises(ValueError, match="multiple outputs"):
        vs.validation_thresholds = {"Ree_cross_correlation": 0.5}
    assert "Residual auto-correlation (Gaussian envelope ratio)" in repr(vs)
    assert vs.outcome["Sim_0"] == "PASS"
    integer_y = np.arange(100, dtype=float)[:, None]
    constant_residual = validate_models(
        u, integer_y, integer_y + 1, sampling_period=1,
        Ree_envelope=(1, 0.1, 2),
        validation_thresholds={"Ree_envelope": 1},
    )
    assert np.isinf(
        constant_residual.validation_statistics["Sim_0"]["Ree_envelope"]
    )
    assert constant_residual.outcome["Sim_0"] == "FAIL"
    with pytest.raises(ValueError, match="requires an envelope"):
        baseline.validation_thresholds = {"Ree_envelope": 1}
    with pytest.raises(ValueError, match="nonzero lag"):
        validate_models(
            u, y, simulated, sampling_period=1, Ree_envelope=(1, 0.1, 2),
            Ree_nlags=np.array([[1]]),
        )


@pytest.mark.parametrize("parameters", [
    (0.9, 0, 2), (0.5, 0.8, 2), (1.1, 0.1, 2),
    (0.9, 0.1, 0), (0.9, 0.1, np.inf), (0.9, 0.1), {},
])
def test_invalid_residual_envelope(parameters) -> None:
    u = np.random.default_rng(74).normal(size=(100, 1))
    with pytest.raises(ValueError, match="Ree_envelope"):
        validate_models(
            u, u, u + 0.1, sampling_period=1, Ree_envelope=parameters
        )


@pytest.mark.parametrize("sampling_period", [0.01, 2.0])
def test_default_envelope_scales_with_sampling_period(sampling_period) -> None:
    rng = np.random.default_rng(77)
    u = rng.normal(size=(200, 1))
    y = rng.normal(size=(200, 1))
    simulated = y + rng.normal(size=y.shape)
    automatic = validate_models(
        u, y, simulated, sampling_period=sampling_period
    )
    explicit = validate_models(
        u, y, simulated, sampling_period=sampling_period,
        Ree_envelope=(0.95, 0.1, 5 * sampling_period),
    )
    assert automatic.validation_statistics == explicit.validation_statistics
    assert automatic.outcome == explicit.outcome
    assert automatic.validation_thresholds["Ree_envelope"] == 1
    assert "Ree_correlation" not in automatic.validation_thresholds


def _sim_labels(n_outputs: int) -> list[str]:
    return [f"my_y{ii + 1}" for ii in range(n_outputs)]


def _stack(signals) -> np.ndarray:
    return np.column_stack([sig.values for sig in signals])


# ============================================================
# ValidationSession, nominal
# ============================================================
@pytest.mark.parametrize("statistic", ["max", "mean"])
def test_residual_rmse_report_and_lifecycle(statistic: str) -> None:
    time = np.arange(100, dtype=float)
    rng = np.random.default_rng(42)
    u = rng.normal(size=100)
    measured = np.column_stack([u, 2 * u])
    errors = rng.normal(size=(100, 2)) * [0.1, 2.0]
    errors[:50] *= 3
    ds = dmv.Dataset.from_signals(
        inputs=[Signal("u", u, time, "V")],
        outputs=[
            Signal("speed", measured[:, 0], time, "rad/s"),
            Signal("current", measured[:, 1], time, "A"),
        ],
    )
    empty = dmv.ValidationSession("RMSE", ds, rmse_statistic=statistic, Ree_envelope=None)
    vs = empty.append_simulation(
        "model", ["speed", "current"], measured + errors
    )
    expected = np.sqrt(np.mean(errors**2, axis=0))
    aggregate = np.max if statistic == "max" else np.mean
    assert vs.validation_statistics["model"]["rmse"] == pytest.approx(
        aggregate(expected)
    )
    report = repr(vs)
    assert f"Residual RMSE ({statistic})" in report
    assert f"{aggregate(expected):.4f}" in report
    assert set(vs.validation_statistics["model"]) == (
        set(dmv.VALIDATION_KEYS) - {"Ree_envelope"}
    )
    assert "rmse" not in vs.validation_thresholds

    trimmed = vs.trim(50, 99)
    assert trimmed.validation_statistics["model"]["rmse"] == pytest.approx(
        aggregate(np.sqrt(np.mean(errors[50:]**2, axis=0)))
    )
    assert vs.drop_simulations("model").validation_statistics == {}
    assert vs.clear().validation_statistics == {}
    assert "model" in vs.validation_statistics


def test_residual_rmse_threshold_and_invalid_statistic() -> None:
    rng = np.random.default_rng(99)
    u = rng.normal(size=(100, 1))
    y = 2 * u
    simulated = y + rng.normal(size=y.shape)
    expected = float(dmv.rmse(y, simulated)[0])
    for threshold, outcome in [(expected + 0.1, "PASS"), (expected, "FAIL")]:
        vs = validate_models(
            u, y, simulated, sampling_period=1.0,
            validation_thresholds={"rmse": threshold},
        )
        assert vs.outcome["Sim_0"] == outcome
    with pytest.raises(ValueError, match="rmse_statistic"):
        validate_models(
            u, y, simulated, sampling_period=1.0, rmse_statistic="min"
        )


class Test_ValidationSession_nominal:
    def test_init(self, good_dataset: tuple) -> None:
        ds, u_names, y_names, _, _, _ = good_dataset

        vs = dmv.ValidationSession("my_validation", ds)

        assert vs.name == "my_validation"
        assert vs.dataset is ds
        assert vs.simulations_names == []
        assert vs.validation_statistics == {}
        assert vs.Ruu.kind == "auto-correlation"

    def test_init_requires_inputs_and_outputs(self, time) -> None:
        y1 = Signal("y1", np.arange(len(time), dtype=float), time, "m")
        ds = dmv.Dataset(inputs={}, outputs={"y1": y1})

        with pytest.raises(ValueError):
            dmv.ValidationSession("my_validation", ds)

    def test_init_with_args(self, good_dataset: tuple) -> None:
        ds, u_names, y_names, _, _, _ = good_dataset

        p = len(u_names)
        q = len(y_names)

        u_nlags_oversized = np.array(
            [[5, 3, 2, 5], [6, 4, 4, 8], [8, 10, 7, 22]]
        )
        eps_nlags_oversized = np.array(
            [[5, 12, 99], [8, 30, 21], [11, 11, 22]]
        )
        ueps_nlags_oversized = np.array(
            [[10, 20, 30], [32, 33, 45], [21, 8, 9]]
        )

        vs = dmv.ValidationSession(
            "my_validation",
            ds,
            Ruu_nlags=u_nlags_oversized,
            Ree_nlags=eps_nlags_oversized,
            Rue_nlags=ueps_nlags_oversized,
        )

        np.testing.assert_array_equal(vs._Ruu.nlags, u_nlags_oversized[:p, :p])
        np.testing.assert_array_equal(
            vs._Ree.nlags, eps_nlags_oversized[:q, :q]
        )
        np.testing.assert_array_equal(
            vs._Rue.nlags, ueps_nlags_oversized[:p, :q]
        )

    def test_init_with_args_raise(self, good_dataset: tuple) -> None:
        ds, _, _, _, _, kind = good_dataset

        if kind != "MIMO":
            pytest.skip("undersized nlags only detectable in the MIMO case")

        with pytest.raises(IndexError):
            dmv.ValidationSession(
                "my_validation", ds, Ree_nlags=np.array([5, 8])
            )

        with pytest.raises(IndexError):
            dmv.ValidationSession(
                "my_validation", ds, Ruu_nlags=np.array([[5, 3], [6, 4]])
            )

        with pytest.raises(IndexError):
            dmv.ValidationSession(
                "my_validation", ds, Rue_nlags=np.array([[10], [32], [21]])
            )

    def test_append_drop_and_clear(self, good_dataset: tuple) -> None:
        ds, _, y_names, _, y_units, _ = good_dataset

        vs = dmv.ValidationSession("my_validation", ds)

        n = len(ds.time())
        q = len(y_names)
        rng = np.random.default_rng(1)

        sim1_name = "Model 1"
        sim1_labels = _sim_labels(q)
        sim1_values = rng.random((n, q))

        vs = vs.append_simulation(sim1_name, sim1_labels, sim1_values)

        assert sim1_name in vs.simulations_names
        assert sim1_name in vs.Ree
        assert sim1_name in vs.Rue
        assert sim1_name in vs.validation_statistics
        np.testing.assert_allclose(
            sim1_values, vs.simulations_values[sim1_name]
        )

        sim2_name = "Model 2"
        sim2_labels = _sim_labels(q)
        sim2_values = _stack(ds.outputs.values()) + rng.random((n, 1))

        vs = vs.append_simulation(sim2_name, sim2_labels, sim2_values)

        assert sim2_name in vs.simulations_names
        np.testing.assert_allclose(
            sim2_values, vs.simulations_values[sim2_name]
        )

        # ---- introspection ----
        assert sorted([sim1_name, sim2_name]) == sorted(vs.simulations_names)

        expected_signals = list(zip(sim1_labels, y_units))
        assert expected_signals == vs.simulation_signals_list(sim1_name)
        assert expected_signals == vs.simulation_signals_list(sim2_name)

        # ---- drop ----
        vs = vs.drop_simulations(sim1_name)

        assert sim1_name not in vs.simulations_names
        assert sim1_name not in vs.Ree
        assert sim1_name not in vs.Rue
        assert sim1_name not in vs.validation_statistics

        # ---- re-add then clear ----
        vs = vs.append_simulation(sim1_name, sim1_labels, sim1_values)
        vs = vs.clear()

        assert vs.simulations_names == []
        assert vs.simulations_values == {}
        assert list(vs.Ree) == []
        assert list(vs.Rue) == []
        assert vs.validation_statistics == {}

    def test_repr(self, good_dataset: tuple) -> None:
        ds, _, y_names, _, _, _ = good_dataset

        vs = dmv.ValidationSession("my_validation", ds)

        # no simulation yet
        assert "my_validation" in repr(vs)

        rng = np.random.default_rng(2)
        q = len(y_names)
        vs = vs.append_simulation(
            "Model 1", _sim_labels(q), rng.random((len(ds.time()), q))
        )

        text = repr(vs)
        assert "R-squared fit (%)" in text
        assert "Input auto-correlation (abs_mean-max)" in text
        assert "Residual auto-correlation (Gaussian envelope ratio)" in text
        assert "Input-residual cross-correlation (abs_mean-max)" in text
        assert "lower means less correlation" in text
        assert (
            "Input auto-correlation describes excitation, not model quality."
            in text
        )
        assert "Model 1" in text

    def test_repr_ignore_input(self, good_dataset: tuple) -> None:
        ds, _, y_names, _, _, _ = good_dataset

        vs = dmv.ValidationSession("my_validation", ds, ignore_input=True)

        assert "Ruu_correlation" not in vs.validation_thresholds

        rng = np.random.default_rng(3)
        q = len(y_names)
        vs = vs.append_simulation(
            "Model 1", _sim_labels(q), rng.random((len(ds.time()), q))
        )

        assert "Input ignored: True" in repr(vs)

    def test_trim(self, good_dataset: tuple) -> None:
        ds, _, y_names, _, _, _ = good_dataset

        vs = dmv.ValidationSession("my_validation", ds)

        n = len(ds.time())
        q = len(y_names)
        rng = np.random.default_rng(4)

        vs = vs.append_simulation(
            "Model 1", _sim_labels(q), rng.random((n, q))
        )

        # act
        vs = vs.trim(tin=1.0, tout=5.0)

        time = vs.dataset.time()

        assert np.isclose(0.0, time[0], atol=1e-9)
        assert np.isclose(4.0, time[-1], atol=1e-9)

        sim_signals = vs.simulations["Model 1"]

        for sig in sim_signals:
            assert sig.time is not None
            np.testing.assert_allclose(sig.time, time)

    def test_change_threshold(self, good_signals: tuple) -> None:
        inputs, outputs, _ = good_signals

        rng = np.random.default_rng(5)
        measured = _stack(outputs)
        sim_good = measured + rng.uniform(0.0, 1e-4, size=measured.shape)

        vs = validate_models(inputs, outputs, simulated_out=sim_good, Ree_envelope=None)

        assert list(vs.outcome.values()) == ["PASS"]

        vs.validation_thresholds = {
            "Ruu_correlation": 0.0,
            "r2": 110.0,
            "Ree_correlation": 0.0,
            "Rue_correlation": 0.0,
        }

        assert list(vs.outcome.values()) == ["FAIL"]

    def test_change_threshold_raise(self, good_signals: tuple) -> None:
        inputs, outputs, _ = good_signals

        rng = np.random.default_rng(6)
        measured = _stack(outputs)
        sim_good = measured + rng.uniform(0.0, 1e-4, size=measured.shape)

        vs = validate_models(inputs, outputs, simulated_out=sim_good)

        with pytest.raises(KeyError):
            vs.validation_thresholds = {"Ruu_correlationssss": 0.0}

        with pytest.raises(ValueError):
            vs.validation_thresholds = {"Ruu_correlation": -1.0}

        # An empty mapping means "nothing to check", which would make
        # every simulation pass unconditionally.
        with pytest.raises(ValueError):
            vs.validation_thresholds = {}

    def test_constructor_validates_thresholds(
        self, good_dataset: tuple
    ) -> None:
        # The constructor used to assign the mapping straight to the
        # attribute, skipping the checks the setter performs.
        ds = good_dataset[0]

        with pytest.raises(KeyError):
            dmv.ValidationSession(
                "my_validation",
                ds,
                validation_thresholds={"potato": 0.5},
            )

        with pytest.raises(ValueError):
            dmv.ValidationSession(
                "my_validation",
                ds,
                validation_thresholds={"r2": -1.0},
            )

        with pytest.raises(ValueError):
            dmv.ValidationSession(
                "my_validation", ds, validation_thresholds={}
            )

    def test_simulation_signals_list_raise(self, good_dataset: tuple) -> None:
        ds, _, y_names, _, _, _ = good_dataset

        vs = dmv.ValidationSession("my_validation", ds)

        # no simulation at all
        with pytest.raises(KeyError):
            vs.simulation_signals_list("potato")

        rng = np.random.default_rng(8)
        q = len(y_names)
        vs = vs.append_simulation(
            "Model 1", _sim_labels(q), rng.random((len(ds.time()), q))
        )

        with pytest.raises(KeyError):
            vs.simulation_signals_list("potato")

        # A list used to be accepted, but only its first element was used
        with pytest.raises(TypeError):
            vs.simulation_signals_list(["Model 1"])  # type: ignore[arg-type]


# ============================================================
# ValidationSession, simulation validation
# ============================================================
class Test_ValidationSession_sim_validation:
    @pytest.fixture
    def session(self, good_dataset: tuple) -> tuple:
        ds, _, y_names, _, _, _ = good_dataset

        return (
            dmv.ValidationSession("my_validation", ds),
            len(ds.time()),
            len(y_names),
        )

    def test_existing_sim_raise(self, session: tuple) -> None:
        vs, n, q = session
        rng = np.random.default_rng(9)

        values = rng.random((n, q))
        vs = vs.append_simulation("Model 1", _sim_labels(q), values)

        with pytest.raises(ValueError):
            vs.append_simulation("Model 1", _sim_labels(q), values)

    def test_too_many_signals_raise(self, session: tuple) -> None:
        vs, n, q = session
        rng = np.random.default_rng(10)

        labels = _sim_labels(q) + ["potato"]

        with pytest.raises(IndexError):
            vs.append_simulation("Model 1", labels, rng.random((n, q + 1)))

    def test_duplicate_names_raise(self, session: tuple) -> None:
        vs, n, q = session
        rng = np.random.default_rng(11)

        if q < 2:
            pytest.skip("duplicated names need at least two outputs")

        labels = ["my_y1"] * q

        with pytest.raises(ValueError):
            vs.append_simulation("Model 1", labels, rng.random((n, q)))

    def test_mismatch_labels_values_raise(self, session: tuple) -> None:
        vs, n, q = session
        rng = np.random.default_rng(12)

        with pytest.raises(IndexError):
            vs.append_simulation(
                "Model 1", _sim_labels(q), rng.random((n, q + 1))
            )

    def test_too_many_values_raise(self, session: tuple) -> None:
        vs, n, q = session
        rng = np.random.default_rng(13)

        with pytest.raises(IndexError):
            vs.append_simulation(
                "Model 1", _sim_labels(q), rng.random((n + 1, q + 1))
            )

    def test_values_not_ndarray_raise(self, session: tuple) -> None:
        vs, _, q = session

        with pytest.raises(ValueError):
            vs.append_simulation("Model 1", _sim_labels(q), "potato")

    def test_ydata_too_short_raise(self, session: tuple) -> None:
        vs, _, q = session
        rng = np.random.default_rng(14)

        with pytest.raises(IndexError):
            vs.append_simulation("Model 1", _sim_labels(q), rng.random((2, q)))

    def test_drop_simulations_raise(self, session: tuple) -> None:
        vs, n, q = session
        rng = np.random.default_rng(15)

        vs = vs.append_simulation(
            "Model 1", _sim_labels(q), rng.random((n, q))
        )

        with pytest.raises(KeyError):
            vs.drop_simulations("potato")

    def test_cheating_raise(self, good_dataset: tuple) -> None:
        ds, _, y_names, _, _, _ = good_dataset

        vs = dmv.ValidationSession("my_validation", ds)

        with pytest.raises(ValueError):
            vs.append_simulation(
                "Cheater",
                _sim_labels(len(y_names)),
                _stack(ds.outputs.values()),
            )


# ============================================================
# Plots
# ============================================================
class Test_Plots:
    def test_missing_simulations_preserve_requested_order(
        self, dataset: dmv.Dataset
    ) -> None:
        measured = _stack(dataset.outputs.values())
        vs = dmv.ValidationSession("selection", dataset).append_simulation(
            "known", _sim_labels(measured.shape[1]), measured + 0.1
        )

        with pytest.raises(KeyError) as exc:
            vs.plot_simulations(["missing-z", "known", "missing-a"])

        assert "['missing-z', 'missing-a']" in str(exc.value)

    @pytest.mark.parametrize("kind", ["simulations", "residuals"])
    @pytest.mark.parametrize(
        "prop_cycle, expected_colors",
        [
            (
                cycler(color=["#123456", "#abcdef"]),
                ["#123456", "#abcdef", "#123456"],
            ),
            (cycler(linestyle=["-", "--"]), ["black"] * 3),
        ],
    )
    def test_validation_plots_follow_color_cycle(
        self, dataset: dmv.Dataset, kind: str, prop_cycle, expected_colors
    ) -> None:
        measured = _stack(dataset.outputs.values())
        rng = np.random.default_rng(42)
        vs = dmv.ValidationSession("styled", dataset)
        for i in range(3):
            vs = vs.append_simulation(
                f"Model {i}",
                _sim_labels(measured.shape[1]),
                measured + rng.normal(size=measured.shape),
            )

        with plt.rc_context(
            {"axes.prop_cycle": prop_cycle, "text.usetex": False}
        ):
            if kind == "simulations":
                figures = (vs.plot_simulations(with_scope=False),)
            else:
                figures = vs.plot_residuals(plot_input=False)

            try:
                for fig in figures:
                    for ax in fig.axes:
                        if kind == "simulations":
                            colors = [
                                line.get_color()
                                for line in ax.lines
                                if line.get_label().startswith("Model ")
                            ]
                        else:
                            colors = [
                                stem.markerline.get_color()
                                for stem in ax.containers
                            ]
                        assert [to_rgba(c) for c in colors] == [
                            to_rgba(c) for c in expected_colors
                        ]
                    fig.canvas.draw()
            finally:
                for fig in figures:
                    plt.close(fig)

    @pytest.mark.plots
    @pytest.mark.parametrize("with_scope", [False, True])
    def test_plot_simulations_honours_layout(
        self, good_dataset: tuple, with_scope: bool
    ) -> None:
        ds, _, y_names, _, _, _ = good_dataset

        vs = dmv.ValidationSession("my_validation", ds)
        n = len(ds.time())
        q = len(y_names)
        rng = np.random.default_rng(16)
        vs = vs.append_simulation(
            "Model 1", _sim_labels(q), rng.random((n, q))
        )

        fig = vs.plot_simulations(layout="tight", with_scope=with_scope)
        engine = fig.get_layout_engine()

        assert engine is not None
        assert type(engine).__name__ == "TightLayoutEngine"

        plt.close("all")

    @pytest.mark.plots
    def test_plot_simulations_rejects_bad_dataset(
        self, good_dataset: tuple
    ) -> None:
        # An unrecognised value used to fall through both `in`/`out` checks
        # and quietly produce a simulations-only plot.
        ds, _, y_names, _, _, _ = good_dataset

        vs = dmv.ValidationSession("my_validation", ds)
        q = len(y_names)
        rng = np.random.default_rng(17)
        vs = vs.append_simulation(
            "Model 1", _sim_labels(q), rng.random((len(ds.time()), q))
        )

        with pytest.raises(ValueError, match="'dataset' must be one of"):
            vs.plot_simulations(dataset="banana")  # type: ignore[arg-type]

        plt.close("all")

    @pytest.mark.plots
    def test_xcorrelation_plot(self, correlation_tensors: tuple) -> None:
        X = correlation_tensors[6]
        Y = correlation_tensors[7]

        x0 = X[:, 0]
        y0 = Y[:, 0]

        for args in ((x0, y0), (X, y0), (x0, Y), (X, Y), (X, X)):
            fig = dmv.XCorrelation("foo", *args).plot()
            assert fig is not None
            plt.close("all")

    @pytest.mark.plots
    def test_validation_plots(self, good_dataset: tuple) -> None:
        ds, _, y_names, _, _, _ = good_dataset

        vs = dmv.ValidationSession("my_validation", ds)

        n = len(ds.time())
        q = len(y_names)
        rng = np.random.default_rng(16)

        vs = vs.append_simulation(
            "Model 1", _sim_labels(q), rng.random((n, q))
        )
        vs = vs.append_simulation(
            "Model 2",
            [f"your_y{ii + 1}" for ii in range(q)],
            _stack(ds.outputs.values()) + rng.random((n, 1)),
        )

        # ---- plot_simulations ----
        for kwargs in (
            {},
            {"dataset": "both"},
            {"dataset": "out"},
            {"dataset": "in"},
            {"with_scope": False},
        ):
            fig = vs.plot_simulations(**kwargs)
            assert fig is not None
            plt.close("all")

        fig = vs.plot_simulations("Model 2", dataset="out")
        assert fig is not None
        plt.close("all")

        fig = vs.plot_simulations(["Model 1", "Model 2"], dataset="in")
        assert fig is not None
        plt.close("all")

        with pytest.raises(KeyError):
            vs.plot_simulations("potato")

        # ---- plot_residuals ----
        figs = vs.plot_residuals()
        assert len(figs) == 3
        plt.close("all")

        figs = vs.plot_residuals("Model 1")
        assert len(figs) == 3
        plt.close("all")

        figs = vs.plot_residuals(["Model 1", "Model 2"], plot_input=False)
        assert len(figs) == 2
        plt.close("all")

        with pytest.raises(KeyError):
            vs.plot_residuals("potato")

        # ---- empty session ----
        vs = vs.clear()

        with pytest.raises(KeyError):
            vs.plot_simulations()

        with pytest.raises(KeyError):
            vs.plot_residuals()


# ============================================================
# validate_models
# ============================================================
class Test_validate_models:
    def test_list_of_signals_arg(self, good_signals: tuple) -> None:
        inputs, outputs, _ = good_signals

        rng = np.random.default_rng(18)
        measured = _stack(outputs)
        q = len(outputs)
        n = measured.shape[0]

        sim_good = measured + rng.uniform(0.0, 1e-4, size=measured.shape)
        sim_bad = rng.random((n, q))
        sim_bad2 = rng.random((n, q))

        vs = validate_models(
            inputs, outputs, simulated_out=[sim_good, sim_bad, sim_bad2],
            Ree_envelope=None,
        )

        assert list(vs.outcome.values()) == ["PASS", "FAIL", "FAIL"]

        np.testing.assert_allclose(
            _stack(vs.dataset.inputs.values()), _stack(inputs)
        )
        np.testing.assert_allclose(
            _stack(vs.dataset.outputs.values()), measured
        )

    def test_ndarrays_args(self, good_signals: tuple) -> None:
        inputs, outputs, _ = good_signals

        u_values = _stack(inputs)
        y_values = _stack(outputs)
        sampling_period = inputs[0].get_sampling_period()

        rng = np.random.default_rng(19)
        n, q = y_values.shape

        sim_good = y_values + rng.uniform(0.0, 1e-4, size=y_values.shape)
        sim_bad = rng.random((n, q))

        vs = validate_models(
            u_values,
            y_values,
            simulated_out=[sim_good, sim_bad],
            sampling_period=sampling_period,
        )

        np.testing.assert_allclose(
            _stack(vs.dataset.inputs.values()), u_values
        )
        np.testing.assert_allclose(
            _stack(vs.dataset.outputs.values()), y_values
        )
        assert vs.dataset.input_names() == [
            f"u{ii}" for ii in range(u_values.shape[1])
        ]
        assert vs.dataset.output_names() == [f"y{ii}" for ii in range(q)]

    def test_impossible_to_pass(self, good_signals: tuple) -> None:
        inputs, outputs, _ = good_signals

        u_values = _stack(inputs)
        y_values = _stack(outputs)
        sampling_period = inputs[0].get_sampling_period()

        rng = np.random.default_rng(20)
        n, q = y_values.shape

        sim_good = y_values + rng.uniform(0.0, 1e-4, size=y_values.shape)
        sim_bad = rng.random((n, q))

        vs = validate_models(
            u_values,
            y_values,
            simulated_out=[sim_good, sim_bad],
            sampling_period=sampling_period,
            validation_thresholds={
                "Ruu_correlation": 0.35,
                "r2": 100.0,  # impossible to achieve
                "Ree_correlation": 0.35,
                "Rue_correlation": 0.35,
            },
        )

        assert list(vs.outcome.values()) == ["FAIL", "FAIL"]

    def test_sampling_period_raise(self, good_signals: tuple) -> None:
        inputs, outputs, _ = good_signals

        u_values = _stack(inputs)
        y_values = _stack(outputs)

        rng = np.random.default_rng(21)

        with pytest.raises(TypeError):
            validate_models(
                u_values,
                y_values,
                simulated_out=rng.random(y_values.shape),
            )

    def test_wrong_ndim_raise(self, good_signals: tuple) -> None:
        inputs, outputs, _ = good_signals

        u_values = _stack(inputs)
        y_values = _stack(outputs)

        with pytest.raises(IndexError):
            validate_models(
                u_values[:, 0], y_values, np.zeros_like(y_values), 0.1
            )

        with pytest.raises(IndexError):
            validate_models(
                u_values, y_values[:, 0], np.zeros_like(y_values), 0.1
            )

    def test_wrong_type_raise(self, good_signals: tuple) -> None:
        inputs, outputs, _ = good_signals

        y_values = _stack(outputs)

        with pytest.raises(ValueError):
            validate_models(["potato"], outputs, np.zeros_like(y_values), 0.1)

    def test_wrong_simulation_shape_raise(self, good_signals: tuple) -> None:
        inputs, outputs, _ = good_signals

        with pytest.raises(ValueError):
            validate_models(inputs, outputs, np.zeros((3, 3)))
