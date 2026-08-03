# -*- coding: utf-8 -*-
"""Tests for the ValidationSession class."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib import pyplot as plt

import dymoval as dmv
from dymoval.config import ATOL
from dymoval.signal import Signal
from dymoval.validation import validate_models


def _sim_labels(n_outputs: int) -> list[str]:
    return [f"my_y{ii + 1}" for ii in range(n_outputs)]


def _stack(signals) -> np.ndarray:
    return np.column_stack([sig.values for sig in signals])


# ============================================================
# ValidationSession, nominal
# ============================================================
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
        assert "R-Squared (%)" in text
        assert "Model 1" in text

    def test_repr_ignore_input(self, good_dataset: tuple) -> None:
        ds, _, y_names, _, _, _ = good_dataset

        vs = dmv.ValidationSession("my_validation", ds, ignore_input=True)

        assert "Ruu_whiteness" not in vs.validation_thresholds

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

        assert np.isclose(0.0, time[0], atol=ATOL)
        assert np.isclose(4.0, time[-1], atol=ATOL)

        sim_signals = vs.simulations["Model 1"]

        for sig in sim_signals:
            assert sig.time is not None
            np.testing.assert_allclose(sig.time, time)

    def test_change_threshold(self, good_signals: tuple) -> None:
        inputs, outputs, _ = good_signals

        rng = np.random.default_rng(5)
        measured = _stack(outputs)
        sim_good = measured + rng.uniform(0.0, 1e-4, size=measured.shape)

        vs = validate_models(inputs, outputs, simulated_out=sim_good)

        assert list(vs.outcome.values()) == ["PASS"]

        vs.validation_thresholds = {
            "Ruu_whiteness": 0.0,
            "r2": 110.0,
            "Ree_whiteness": 0.0,
            "Rue_whiteness": 0.0,
        }

        assert list(vs.outcome.values()) == ["FAIL"]

    def test_change_threshold_raise(self, good_signals: tuple) -> None:
        inputs, outputs, _ = good_signals

        rng = np.random.default_rng(6)
        measured = _stack(outputs)
        sim_good = measured + rng.uniform(0.0, 1e-4, size=measured.shape)

        vs = validate_models(inputs, outputs, simulated_out=sim_good)

        with pytest.raises(KeyError):
            vs.validation_thresholds = {"Ruu_whitenessssss": 0.0}

        with pytest.raises(ValueError):
            vs.validation_thresholds = {"Ruu_whiteness": -1.0}

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
            inputs, outputs, simulated_out=[sim_good, sim_bad, sim_bad2]
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
                "Ruu_whiteness": 0.35,
                "r2": 100.0,  # impossible to achieve
                "Ree_whiteness": 0.35,
                "Rue_whiteness": 0.35,
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
