import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

import dymoval as dmv
from dymoval.config import ATOL
from dymoval.dataset import Signal
from dymoval.validation import compute_statistic, validate_models


class Test_ClassValidationNominal:
    def test_init(self, good_dataframe: pd.DataFrame) -> None:
        # Nominal data
        df, u_names, y_names, _, _, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.Dataset(
            name_ds, df, u_names, y_names, full_time_interval=True
        )
        name_vs = "my_validation"
        vs = dmv.ValidationSession(name_vs, ds)

        # Check that the passed dataset is correctly stored.
        # Main DataFrame
        assert all(vs.dataset.dataset == ds.dataset)

        for ii in range(4):  # Size of coverage
            assert all(vs.dataset.coverage[ii] == ds.coverage[ii])

    def test_init_with_args(self, good_dataframe: pd.DataFrame) -> None:
        # Nominal data
        df, u_names, y_names, _, _, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.Dataset(
            name_ds, df, u_names, y_names, full_time_interval=True
        )

        p = len(ds.dataset["INPUT"].columns.get_level_values("names"))
        q = len(ds.dataset["OUTPUT"].columns.get_level_values("names"))

        name_vs = "my_validation"

        u_nlags_wrong_size = np.array(
            [[5, 3, 2, 5], [6, 4, 4, 8], [8, 10, 7, 22]]
        )
        eps_nlags_wrong_size = np.array(
            [[5, 12, 99], [8, 30, 21], [11, 11, 22]]
        )
        ueps_nlags_wrong_size = np.array(
            [[10, 20, 30], [32, 33, 45], [21, 8, 9]]
        )
        vs = dmv.ValidationSession(
            name_vs,
            ds,
            Ruu_nlags=u_nlags_wrong_size,
            Ree_nlags=eps_nlags_wrong_size,
            Rue_nlags=ueps_nlags_wrong_size,
        )

        expected_u_nlags = u_nlags_wrong_size[:p, :p]
        expected_eps_nlags = eps_nlags_wrong_size[:q, :q]
        expected_ueps_nlags = ueps_nlags_wrong_size[:p, :q]

        np.testing.assert_array_equal(vs._Ruu_nlags, expected_u_nlags)
        np.testing.assert_array_equal(vs._Ree_nlags, expected_eps_nlags)
        np.testing.assert_array_equal(vs._Rue_nlags, expected_ueps_nlags)

    def test_init_with_args_raise(self, good_dataframe: pd.DataFrame) -> None:
        # Nominal data
        df, u_names, y_names, _, _, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.Dataset(
            name_ds, df, u_names, y_names, full_time_interval=True
        )

        name_vs = "my_validation"
        # Lags specified only for some residuals
        if fixture == "MIMO":
            eps_nlags_wrong_size = np.array([5, 8])

            with pytest.raises(IndexError):
                _ = dmv.ValidationSession(
                    name_vs,
                    ds,
                    Ree_nlags=eps_nlags_wrong_size,
                )

            # Lags specified only for some inputs
            u_nlags_wrong_size = np.array([[5, 3], [6, 4]])
            with pytest.raises(IndexError):
                _ = dmv.ValidationSession(
                    name_vs,
                    ds,
                    Ruu_nlags=u_nlags_wrong_size,
                )

            # Lags specified only for some inputs
            ueps_nlags_wrong_size = np.array([[10], [32], [21]])
            with pytest.raises(IndexError):
                _ = dmv.ValidationSession(
                    name_vs,
                    ds,
                    Rue_nlags=ueps_nlags_wrong_size,
                )

    def test_random_walk(self, good_dataframe: pd.DataFrame) -> None:
        df, u_names, y_names, _, y_units, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.Dataset(
            name_ds, df, u_names, y_names, full_time_interval=True
        )

        name_vs = "my_validation"
        vs = dmv.ValidationSession(name_vs, ds)

        if isinstance(y_units, str):
            y_units = [y_units]
        # ==================================
        # Append simulations test
        # ==================================

        # Add one model
        sim1_name = "Model 1"
        sim1_labels = ["my_y1", "my_y2"]  # The fixture has two outputs
        if fixture == "SISO" or fixture == "MISO":
            sim1_labels = [sim1_labels[0]]
        sim1_values = np.random.rand(
            len(df.iloc[:, 0].values), len(sim1_labels)
        )

        vs = vs.append_simulation(sim1_name, sim1_labels, sim1_values)
        # At least the names are there...
        assert sim1_name in vs.simulations_values.columns.get_level_values(
            "sim_names"
        )
        assert sim1_name in vs._Ree_tensor.keys()
        assert sim1_name in vs._Rue_tensor.keys()
        assert sim1_name in vs._validation_statistics.columns

        np.testing.assert_allclose(
            sim1_values, vs.simulations_values[sim1_name]
        )

        # # Add second model
        sim2_name = "Model 2"
        sim2_labels = ["my_y1", "my_y2"]  # The fixture has two outputs
        if fixture == "SISO" or fixture == "MISO":
            # You only have one output
            sim2_labels = [sim1_labels[0]]
        sim2_values = vs.dataset.dataset["OUTPUT"].values + np.random.rand(
            len(vs.dataset.dataset["OUTPUT"].values), 1
        )

        vs = vs.append_simulation(sim2_name, sim2_labels, sim2_values)
        # At least the names are there...
        assert sim2_name in vs.simulations_values.columns.get_level_values(
            "sim_names"
        )
        assert sim2_name in vs._Ree_tensor.keys()
        assert sim2_name in vs._Rue_tensor.keys()
        assert sim2_name in vs._validation_statistics.columns

        np.testing.assert_allclose(
            sim2_values, vs.simulations_values[sim2_name]
        )

        # ===============================================
        # Test simulation_signals_list and list_simulations
        # ============================================
        expected_sims = [sim1_name, sim2_name]

        assert sorted(expected_sims) == sorted(vs.simulations_names)

        expected_signals1 = list(zip(sim1_labels * len(y_units), y_units))
        expected_signals2 = list(zip(sim2_labels * len(y_units), y_units))

        assert sorted(expected_signals1) == sorted(
            vs.simulation_signals_list(sim1_name)
        )
        assert sorted(expected_signals2) == sorted(
            vs.simulation_signals_list(sim2_name)
        )

        # ==================================
        # drop_simulation sim
        # ==================================
        vs = vs.drop_simulations(sim1_name)
        # At least the names are nt there any longer.
        assert sim1_name not in vs.simulations_values.columns.get_level_values(
            "sim_names"
        )
        assert sim1_name not in vs._Ree_tensor.keys()
        assert sim1_name not in vs._Rue_tensor.keys()
        assert sim1_name not in vs._validation_statistics.columns

        # ============================================
        # Re-add sim and then clear.
        # ============================================
        vs = vs.append_simulation(sim1_name, sim1_labels, sim1_values)

        vs = vs.clear()

        assert [] == list(vs.simulations_values.columns)
        assert [] == list(vs._Ree_tensor.keys())
        assert [] == list(vs._Rue_tensor.keys())
        assert [] == list(vs._validation_statistics.columns)

    def test_trim(self, good_dataframe: pd.DataFrame) -> None:
        df, u_names, y_names, _, y_units, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.Dataset(
            name_ds, df, u_names, y_names, full_time_interval=True
        )

        name_vs = "my_validation"
        vs = dmv.ValidationSession(name_vs, ds)

        if isinstance(y_units, str):
            y_units = [y_units]
        # ==================================
        # Append simulations test
        # ==================================

        # Add one model
        sim1_name = "Model 1"
        sim1_labels = ["my_y1", "my_y2"]  # The fixture has two outputs
        if fixture == "SISO" or fixture == "MISO":
            sim1_labels = [sim1_labels[0]]
        sim1_values = np.random.rand(
            len(df.iloc[:, 0].values), len(sim1_labels)
        )

        vs = vs.append_simulation(sim1_name, sim1_labels, sim1_values)

        # # Add second model
        sim2_name = "Model 2"
        sim2_labels = ["my_y1", "my_y2"]  # The fixture has two outputs
        if fixture == "SISO" or fixture == "MISO":
            # You only have one output
            sim2_labels = [sim1_labels[0]]
        sim2_values = vs.dataset.dataset["OUTPUT"].values + np.random.rand(
            len(vs.dataset.dataset["OUTPUT"].values), 1
        )

        vs = vs.append_simulation(sim2_name, sim2_labels, sim2_values)

        # Expected value.
        # If you remove a mean from a signal, then the mean of the reminder
        # signal must be zero.
        expected_tin = 0.0  # as per fixture
        expected_tout = 4.0  # as per fixture

        # act
        vs = vs.trim(tin=1.0, tout=5.0)

        # Evaluate
        assert np.isclose(expected_tin, vs.dataset.dataset.index[0], atol=ATOL)
        assert np.isclose(
            expected_tout, vs.dataset.dataset.index[-1], atol=ATOL
        )

        assert np.isclose(
            expected_tin, vs.simulations_values.index[0], atol=ATOL
        )
        assert np.isclose(
            expected_tout, vs.simulations_values.index[-1], atol=ATOL
        )

    def test_change_threshold(
        self,
        good_signals_no_nans: list[Signal],
        tmp_path: str,
    ) -> None:
        (
            signal_list,
            u_names,
            y_names,
            u_units,
            y_units,
            fixture,
        ) = good_signals_no_nans

        # List of signals
        dataset_in = [s for s in signal_list if s["name"] in u_names]
        dataset_out = [s for s in signal_list if s["name"] in y_names]
        # sim_gppd is a list of 1D array
        small_perturbation = np.random.uniform(
            low=0.0,
            high=1e-4,
            size=(dataset_out[0]["samples"].size, len(y_names)),
        )
        sim_good = np.array(
            [
                s["samples"] + w
                for s, w in zip(dataset_out, small_perturbation.T)
            ]
        ).T

        # Override if MISO or SISO
        if fixture == "SISO" or fixture == "SIMO":
            dataset_in = [dataset_in[0]]

        if fixture == "MISO" or fixture == "SISO":
            dataset_out = [dataset_out[0]]

        #  act
        vs = validate_models(
            dataset_in,
            dataset_out,
            simulated_out=sim_good,
        )

        expected_outcome = ["PASS"]
        assert list(vs.outcome.values()) == expected_outcome

        # Change threshold now
        impossible_threshold = {
            "Ruu_whiteness": 0.0,
            "r2": 110,
            "Ree_whiteness": 0.0,
            "Rue_whiteness": 0.0,
        }
        vs.validation_thresholds = impossible_threshold
        expected_outcome = ["FAIL"]
        assert list(vs.outcome.values()) == expected_outcome

    def test_change_threshold_raise(
        self,
        good_signals_no_nans: list[Signal],
        tmp_path: str,
    ) -> None:
        (
            signal_list,
            u_names,
            y_names,
            u_units,
            y_units,
            fixture,
        ) = good_signals_no_nans

        # List of signals
        dataset_in = [s for s in signal_list if s["name"] in u_names]
        dataset_out = [s for s in signal_list if s["name"] in y_names]
        # sim_gppd is a list of 1D array
        small_perturbation = np.random.uniform(
            low=0.0,
            high=1e-4,
            size=(dataset_out[0]["samples"].size, len(y_names)),
        )
        sim_good = np.array(
            [
                s["samples"] + w
                for s, w in zip(dataset_out, small_perturbation.T)
            ]
        ).T

        # Override if MISO or SISO
        if fixture == "SISO" or fixture == "SIMO":
            dataset_in = [dataset_in[0]]

        if fixture == "MISO" or fixture == "SISO":
            dataset_out = [dataset_out[0]]

        #  act
        vs = validate_models(
            dataset_in,
            dataset_out,
            simulated_out=sim_good,
        )

        expected_outcome = ["PASS"]
        assert list(vs.outcome.values()) == expected_outcome

        # Change threshold now
        impossible_threshold = {
            "Ruu_whitenesssssss": 0.0,
            "r2": 110,
            "Ree_whiteness": 0.0,
            "Rue_whiteness": 0.0,
        }
        with pytest.raises(KeyError):
            vs.validation_thresholds = impossible_threshold

        # Change threshold now
        impossible_threshold = {
            "Ruu_whiteness": -1.0,
            "r2": 110,
            "Ree_whiteness": 0.0,
            "Rue_whiteness": 0.0,
        }

        with pytest.raises(ValueError):
            vs.validation_thresholds = impossible_threshold

    def test_get_sim_signal_list_and_statistics_raise(
        self, good_dataframe: pd.DataFrame
    ) -> None:
        df, u_names, y_names, _, _, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.Dataset(
            name_ds, df, u_names, y_names, full_time_interval=True
        )
        name_vs = "my_validation"
        vs = dmv.ValidationSession(name_vs, ds)

        # simulation not in the list
        with pytest.raises(KeyError):
            vs.simulation_signals_list("potato")

        # Another test with one model
        sim1_name = "Model 1"
        sim1_labels = ["my_y1", "my_y2"]  # The fixture has two outputs
        if fixture == "SISO" or fixture == "MISO":
            sim1_labels = [sim1_labels[0]]
        sim1_values = np.random.rand(
            len(df.iloc[:, 0].values), len(sim1_labels)
        )

        # Same sim nmane
        vs = vs.append_simulation(sim1_name, sim1_labels, sim1_values)

        # Search for a non-existing simulation
        with pytest.raises(KeyError):
            vs.simulation_signals_list("potato")


class Test_ClassValidationNominal_sim_validation:
    def test_existing_sim_raise(self, good_dataframe: pd.DataFrame) -> None:
        df, u_names, y_names, _, _, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.Dataset(
            name_ds, df, u_names, y_names, full_time_interval=True
        )
        name_vs = "my_validation"
        vs = dmv.ValidationSession(name_vs, ds)

        # Add one model
        sim1_name = "Model 1"
        sim1_labels = ["my_y1", "my_y2"]  # The fixture has two outputs
        if fixture == "SISO" or fixture == "MISO":
            sim1_labels = [sim1_labels[0]]
        sim1_values = np.random.rand(
            len(df.iloc[:, 0].values), len(sim1_labels)
        )

        # Same sim nmane
        vs = vs.append_simulation(sim1_name, sim1_labels, sim1_values)
        with pytest.raises(ValueError):
            vs.append_simulation(sim1_name, sim1_labels, sim1_values)

    def test_too_many_signals_raise(
        self, good_dataframe: pd.DataFrame
    ) -> None:
        df, u_names, y_names, _, _, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.Dataset(
            name_ds, df, u_names, y_names, full_time_interval=True
        )

        name_vs = "my_validation"
        vs = dmv.ValidationSession(name_vs, ds)

        # Add one model
        sim1_name = "Model 1"
        sim1_labels = [
            "my_y1",
            "my_y2",
            "potato",
        ]  # The fixture has two outputs
        sim1_values = np.random.rand(
            len(df.iloc[:, 0].values), len(sim1_labels)
        )

        # Same sim nmane
        with pytest.raises(IndexError):
            vs.append_simulation(sim1_name, sim1_labels, sim1_values)

    def test_duplicate_names_raise(self, good_dataframe: pd.DataFrame) -> None:
        df, u_names, y_names, _, _, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.Dataset(
            name_ds, df, u_names, y_names, full_time_interval=True
        )

        name_vs = "my_validation"
        vs = dmv.ValidationSession(name_vs, ds)

        # Add one model
        sim1_name = "Model 1"
        sim1_labels = ["my_y1", "my_y1"]  # The fixture has two outputs
        if fixture == "SISO" or fixture == "MISO":
            sim1_labels = [sim1_labels[0]]
        sim1_values = np.random.rand(
            len(df.iloc[:, 0].values), len(sim1_labels)
        )

        # Same sim nmane
        if fixture == "SIMO" or fixture == "MIMO":
            with pytest.raises(ValueError):
                vs.append_simulation(sim1_name, sim1_labels, sim1_values)

    def test_mismatch_labels_values_raise(
        self, good_dataframe: pd.DataFrame
    ) -> None:
        df, u_names, y_names, _, _, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.Dataset(
            name_ds, df, u_names, y_names, full_time_interval=True
        )

        name_vs = "my_validation"
        vs = dmv.ValidationSession(name_vs, ds)

        # Add one model
        sim1_name = "Model 1"
        sim1_labels = ["my_y1", "my_y2"]  # The fixture has two outputs
        if fixture == "SISO" or fixture == "MISO":
            sim1_labels = [sim1_labels[0]]
        sim1_values = np.random.rand(
            len(df.iloc[:, 0].values), len(sim1_labels) + 1
        )

        # Same sim nmane
        with pytest.raises(IndexError):
            vs.append_simulation(sim1_name, sim1_labels, sim1_values)

    def test_too_many_values_raise(self, good_dataframe: pd.DataFrame) -> None:
        df, u_names, y_names, _, _, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.Dataset(
            name_ds, df, u_names, y_names, full_time_interval=True
        )

        name_vs = "my_validation"
        vs = dmv.ValidationSession(name_vs, ds)

        # Add one model
        sim1_name = "Model 1"
        sim1_labels = ["my_y1", "my_y2"]  # The fixture has two outputs
        if fixture == "SISO" or fixture == "MISO":
            sim1_labels = [sim1_labels[0]]
        sim1_values = np.random.rand(
            len(df.iloc[:, 0].values) + 1, len(sim1_labels) + 1
        )

        # Same sim nmane
        with pytest.raises(IndexError):
            vs.append_simulation(sim1_name, sim1_labels, sim1_values)

    def test_values_not_ndarray_raise(
        self, good_dataframe: pd.DataFrame
    ) -> None:
        df, u_names, y_names, _, _, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.Dataset(
            name_ds, df, u_names, y_names, full_time_interval=True
        )

        name_vs = "my_validation"
        vs = dmv.ValidationSession(name_vs, ds)

        # Add one model
        sim1_name = "Model 1"
        sim1_labels = ["my_y1", "my_y2"]  # The fixture has two outputs
        if fixture == "SISO" or fixture == "MISO":
            sim1_labels = [sim1_labels[0]]
        sim1_values = "potato"

        # Same sim nmane
        with pytest.raises(ValueError):
            vs.append_simulation(sim1_name, sim1_labels, sim1_values)

    def test_ydata_too_short_raise(self, good_dataframe: pd.DataFrame) -> None:
        df, u_names, y_names, _, _, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.Dataset(
            name_ds, df, u_names, y_names, full_time_interval=True
        )

        name_vs = "my_validation"
        vs = dmv.ValidationSession(name_vs, ds)

        # Add one model
        sim1_name = "Model 1"
        sim1_labels = ["my_y1", "my_y2"]  # The fixture has two outputs
        if fixture == "SISO" or fixture == "MISO":
            sim1_labels = [sim1_labels[0]]

        # Short data
        sim1_values = np.random.rand(2, 1)

        # Same sim nmane
        with pytest.raises(IndexError):
            vs.append_simulation(sim1_name, sim1_labels, sim1_values)

    def test_drop_simulations_raise(
        self, good_dataframe: pd.DataFrame
    ) -> None:
        df, u_names, y_names, _, _, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.Dataset(
            name_ds, df, u_names, y_names, full_time_interval=True
        )

        # Create validation session.
        name_vs = "my_validation"
        vs = dmv.ValidationSession(name_vs, ds)

        # Add one model
        sim1_name = "Model 1"
        sim1_labels = ["my_y1", "my_y2"]  # The fixture has two outputs
        if fixture == "SISO" or fixture == "MISO":
            sim1_labels = [sim1_labels[0]]
        sim1_values = np.random.rand(
            len(df.iloc[:, 0].values), len(sim1_labels)
        )

        vs = vs.append_simulation(sim1_name, sim1_labels, sim1_values)

        with pytest.raises(ValueError):
            vs.drop_simulations("potato")


class Test_Plots:
    @pytest.mark.plots
    def test_plots(self, good_dataframe: pd.DataFrame, tmp_path: str) -> None:
        # ===========================
        # XCorrelation Plot
        # ===========================
        x1 = np.array([0.1419, 0.4218, 0.9157, 0.7922, 0.9595])
        x2 = np.array([0.6557, 0.0357, 0.8491, 0.9340, 0.6787])
        X = np.array([x1, x2]).T

        y1 = np.array([0.7577, 0.7431, 0.3922, 0.6555, 0.1712])
        y2 = np.array([0.7060, 0.0318, 0.2769, 0.0462, 0.0971])
        Y = np.array([y1, y2]).T

        XCorr_actual = dmv.XCorrelation("foo", x1, y1)
        _ = XCorr_actual.plot()
        plt.close("all")

        XCorr_actual = dmv.XCorrelation("foo", x1, Y)
        _ = XCorr_actual.plot()
        plt.close("all")

        XCorr_actual = dmv.XCorrelation("foo", X, y1)
        _ = XCorr_actual.plot()
        plt.close("all")

        XCorr_actual = dmv.XCorrelation("foo", X, Y)
        _ = XCorr_actual.plot()
        plt.close("all")

        # ===========================
        # ValidationSession Plot
        # ===========================

        df, u_names, y_names, _, _, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.Dataset(
            name_ds, df, u_names, y_names, full_time_interval=True
        )

        print(ds.dataset)

        name_vs = "my_validation"
        vs = dmv.ValidationSession(name_vs, ds)
        print(vs.simulations_values)

        # Add one model
        sim1_name = "Model 1"
        sim1_labels = ["my_y1", "my_y2"]  # The fixture has two outputs
        if fixture == "SISO" or fixture == "MISO":
            sim1_labels = [sim1_labels[0]]
        sim1_values = np.random.rand(
            len(df.iloc[:, 0].values), len(sim1_labels)
        )
        vs = vs.append_simulation(sim1_name, sim1_labels, sim1_values)

        # Add a second
        sim2_name = "Model 2"
        sim2_labels = ["your_y1", "your_y2"]  # The fixture has two outputs
        if fixture == "SISO" or fixture == "MISO":
            sim2_labels = [sim2_labels[0]]
        sim2_values = vs.dataset.dataset["OUTPUT"].values + np.random.rand(
            len(vs.dataset.dataset["OUTPUT"].values), 1
        )
        vs = vs.append_simulation(sim2_name, sim2_labels, sim2_values)
        print(vs.simulations_values)

        # =============================
        # plot simulations
        # =============================
        _ = vs.plot_simulations()
        plt.close("all")

        _ = vs.plot_simulations(dataset="both")
        plt.close("all")

        # Test plot - filtered
        _ = vs.plot_simulations("Model 2", dataset="out")
        plt.close("all")

        # Test plot - all the options
        _ = vs.plot_simulations(["Model 1", "Model 2"], dataset="in")
        plt.close("all")

        # =============================
        # plot simulations raises
        # =============================
        # Test plot - filtered wrong
        with pytest.raises(KeyError):
            _ = vs.plot_simulations("potato")
        # Test plot - filtered wrong
        vs = vs.clear()
        with pytest.raises(KeyError):
            _ = vs.plot_simulations()
        with pytest.raises(KeyError):
            _ = vs.plot_simulations("potato")

        # =============================
        # plot residuals
        # =============================
        vs = vs.append_simulation(sim1_name, sim1_labels, sim1_values)
        vs = vs.append_simulation(sim2_name, sim2_labels, sim2_values)

        _, _, _ = vs.plot_residuals()
        plt.close("all")

        _, _, _ = vs.plot_residuals("Model 1")
        plt.close("all")

        _, _, _ = vs.plot_residuals(["Model 1", "Model 2"])
        plt.close("all")

        _, _, _ = vs.plot_residuals(["Model 1", "Model 2"])
        plt.close("all")

        # =============================
        # plot residuals raises
        # =============================
        with pytest.raises(KeyError):
            _, _, _ = vs.plot_residuals("potato")

        # Empty simulation list
        vs = vs.clear()
        with pytest.raises(KeyError):
            _, _, _ = vs.plot_residuals()


class Test_XCorrelation:
    def test_initializer(self, correlation_tensors) -> None:
        # Just test that it won't run any error
        # Next, remove randoms with known values.
        (
            Rx0y0_expected,
            Rx1y0_expected,
            Rx0y1_expected,
            _,
            Rx1y1_expected,
            _,
            X,
            Y,
            X_bandwidths,
            Y_bandwidths,
            sampling_period,
        ) = correlation_tensors
        # x0y0_whiteness_expected = 0.38281031914047287
        lags_expected_long = np.arange(-5, 6)

        x0 = X[:, 0].T
        y0 = Y[:, 0].T

        # SISO
        XCorr_actual = dmv.XCorrelation("foo", x0, y0)
        R_actual = XCorr_actual.R

        np.testing.assert_allclose(
            R_actual[0, 0].values, Rx0y0_expected, atol=1e-3
        )
        np.testing.assert_allclose(R_actual[0, 0].lags, lags_expected_long)

        # SIMO
        XCorr_actual = dmv.XCorrelation("foo", x0, Y)
        R_actual = XCorr_actual.R

        np.testing.assert_allclose(
            R_actual[0, 0].values, Rx0y0_expected, atol=1e-3
        )
        np.testing.assert_allclose(
            R_actual[0, 1].values, Rx0y1_expected, atol=1e-3
        )

        np.testing.assert_allclose(R_actual[0, 0].lags, lags_expected_long)
        np.testing.assert_allclose(R_actual[0, 1].lags, lags_expected_long)

        # MISO
        XCorr_actual = dmv.XCorrelation("foo", X, y0)
        R_actual = XCorr_actual.R

        np.testing.assert_allclose(
            R_actual[0, 0].values, Rx0y0_expected, atol=1e-3
        )
        np.testing.assert_allclose(
            R_actual[1, 0].values, Rx1y0_expected, atol=1e-3
        )

        np.testing.assert_allclose(R_actual[0, 0].lags, lags_expected_long)
        np.testing.assert_allclose(R_actual[1, 0].lags, lags_expected_long)

        # MIMO
        XCorr_actual = dmv.XCorrelation("foo", X, Y)
        R_actual = XCorr_actual.R

        np.testing.assert_allclose(
            R_actual[0, 0].values, Rx0y0_expected, atol=1e-3
        )
        np.testing.assert_allclose(
            R_actual[0, 1].values, Rx0y1_expected, atol=1e-3
        )
        np.testing.assert_allclose(
            R_actual[1, 0].values, Rx1y0_expected, atol=1e-3
        )
        np.testing.assert_allclose(
            R_actual[1, 1].values, Rx1y1_expected, atol=1e-3
        )

        np.testing.assert_allclose(R_actual[0, 0].lags, lags_expected_long)
        np.testing.assert_allclose(R_actual[0, 1].lags, lags_expected_long)
        np.testing.assert_allclose(R_actual[1, 0].lags, lags_expected_long)
        np.testing.assert_allclose(R_actual[1, 1].lags, lags_expected_long)
        assert XCorr_actual.kind == "cross-correlation"

        # Name switch test
        XCorr_actual = dmv.XCorrelation("foo", X, X)
        assert XCorr_actual.kind == "auto-correlation"

    def test_initializer_with_bandwidth_args(
        self, correlation_tensors
    ) -> None:
        # Just test that it won't run any error
        # Next, remove randoms with known values.
        (
            Rx0y0_expected,
            Rx1y0_expected,
            Rx0y1_expected,
            Rx0y1_expected_partial,
            Rx1y1_expected,
            Rx1y1_expected_partial,
            X,
            Y,
            X_bandwidths,
            Y_bandwidths,
            sampling_period,
        ) = correlation_tensors
        # x0y0_whiteness_expected = 0.38281031914047287
        lags_expected_short_x0y1 = np.arange(-1, 2)
        lags_expected_short_x1y1 = np.arange(-1, 2)
        lags_expected_long = np.arange(-5, 6)

        # We only consider the MIMO case
        XCorr_actual = dmv.XCorrelation(
            "foo", X, Y, None, X_bandwidths, Y_bandwidths, sampling_period
        )
        R_actual = XCorr_actual.R

        np.testing.assert_allclose(
            R_actual[0, 0].values, Rx0y0_expected, atol=1e-3
        )
        np.testing.assert_allclose(
            R_actual[0, 1].values, Rx0y1_expected_partial, atol=1e-3
        )
        np.testing.assert_allclose(
            R_actual[1, 0].values, Rx1y0_expected, atol=1e-3
        )
        np.testing.assert_allclose(
            R_actual[1, 1].values, Rx1y1_expected_partial, atol=1e-3
        )

        np.testing.assert_allclose(R_actual[0, 0].lags, lags_expected_long)
        np.testing.assert_allclose(
            R_actual[0, 1].lags, lags_expected_short_x0y1
        )
        np.testing.assert_allclose(R_actual[1, 0].lags, lags_expected_long)
        np.testing.assert_allclose(
            R_actual[1, 1].lags, lags_expected_short_x1y1
        )
        assert XCorr_actual.kind == "cross-correlation"

    def test_initializer_with_not_all_args_passed(
        self, correlation_tensors
    ) -> None:
        # Not all arguments are passed
        # We only consider the MIMO case
        (
            Rx0y0_expected,
            Rx1y0_expected,
            Rx0y1_expected,
            Rx0y1_expected_partial,
            Rx1y1_expected,
            Rx1y1_expected_partial,
            X,
            Y,
            X_bandwidths,
            Y_bandwidths,
            sampling_period,
        ) = correlation_tensors
        XCorr_actual = dmv.XCorrelation(
            "foo",
            X,
            Y,
            None,
            X_bandwidths=X_bandwidths,
            sampling_period=sampling_period,
        )
        R_actual = XCorr_actual.R
        lags_expected_long = np.arange(-5, 6)

        np.testing.assert_allclose(
            R_actual[0, 0].values, Rx0y0_expected, atol=1e-3
        )
        np.testing.assert_allclose(
            R_actual[0, 1].values, Rx0y1_expected, atol=1e-3
        )
        np.testing.assert_allclose(
            R_actual[1, 0].values, Rx1y0_expected, atol=1e-3
        )
        np.testing.assert_allclose(
            R_actual[1, 1].values, Rx1y1_expected, atol=1e-3
        )

        np.testing.assert_allclose(R_actual[0, 0].lags, lags_expected_long)
        np.testing.assert_allclose(R_actual[0, 1].lags, lags_expected_long)
        np.testing.assert_allclose(R_actual[1, 0].lags, lags_expected_long)
        np.testing.assert_allclose(R_actual[1, 1].lags, lags_expected_long)
        assert XCorr_actual.kind == "cross-correlation"

    def test_initializer_with_nlags_arg(self, correlation_tensors) -> None:
        # Not all arguments are passed
        # We only consider the MIMO case
        (
            Rx0y0_expected,
            Rx1y0_expected,
            Rx0y1_expected,
            Rx0y1_expected_partial,
            Rx1y1_expected,
            Rx1y1_expected_partial,
            X,
            Y,
            X_bandwidths,
            Y_bandwidths,
            sampling_period,
        ) = correlation_tensors

        nlags = np.array([[5, 3], [6, 4]])
        XCorr_actual = dmv.XCorrelation(
            name="foo",
            X=X,
            Y=Y,
            nlags=nlags,
            X_bandwidths=X_bandwidths,
            Y_bandwidths=Y_bandwidths,
            sampling_period=sampling_period,
        )
        R_actual = XCorr_actual.R

        lags_expected_x0y0 = np.arange(-2, 3)
        lags_expected_x1y0 = np.arange(-3, 4)
        lags_expected_x0y1 = np.arange(-1, 2)
        lags_expected_x1y1 = np.arange(-1, 2)

        np.testing.assert_allclose(R_actual[0, 0].lags, lags_expected_x0y0)
        np.testing.assert_allclose(R_actual[0, 1].lags, lags_expected_x0y1)
        np.testing.assert_allclose(R_actual[1, 0].lags, lags_expected_x1y0)
        np.testing.assert_allclose(R_actual[1, 1].lags, lags_expected_x1y1)

        # TODO: check also if the values are correctly picked.

    def test_initializer_with_wrong_params(self, correlation_tensors) -> None:
        # Just test that it won't run any error
        # Next, remove randoms with known values.
        (
            Rx0y0_expected,
            Rx1y0_expected,
            Rx0y1_expected,
            Rx0y1_expected_partial,
            Rx1y1_expected,
            Rx1y1_expected_partial,
            X,
            Y,
            X_bandwidths,
            Y_bandwidths,
            sampling_period,
        ) = correlation_tensors

        # Bandwidths
        X_bandwidths = np.array([2])
        with pytest.raises(IndexError):
            _ = dmv.XCorrelation(
                "foo", X, Y, None, X_bandwidths, Y_bandwidths, sampling_period
            )

    def test_estimate_whiteness(self, correlation_tensors) -> None:
        (
            Rx0y0_expected,
            Rx1y0_expected,
            Rx0y1_expected,
            Rx0y1_expected_partial,
            Rx1y1_expected,
            Rx1y1_expected_partial,
            X,
            Y,
            X_bandwidths,
            Y_bandwidths,
            sampling_period,
        ) = correlation_tensors

        # Test 1: if the local_weights are all one, then the whiteness
        # estimate with and without weights shall be the same
        RXY = dmv.XCorrelation("foo", X, Y)
        w, W = RXY.estimate_whiteness()

        global_weights = np.ones((2, 2))
        local_weights = np.empty(RXY.R.shape, dtype=np.ndarray)
        local_weights[0, 0] = np.ones(11)
        local_weights[0, 1] = np.ones(11)
        local_weights[1, 0] = np.ones(11)
        local_weights[1, 1] = np.ones(11)
        w_weighted, W_weighted = RXY.estimate_whiteness(
            local_weights=local_weights, global_weights=global_weights
        )

        np.testing.assert_allclose(w, w_weighted)
        np.testing.assert_allclose(W, W_weighted)

    def test_estimate_whiteness_raise(self, correlation_tensors) -> None:
        (
            Rx0y0_expected,
            Rx1y0_expected,
            Rx0y1_expected,
            Rx0y1_expected_partial,
            Rx1y1_expected,
            Rx1y1_expected_partial,
            X,
            Y,
            X_bandwidths,
            Y_bandwidths,
            sampling_period,
        ) = correlation_tensors

        RXY = dmv.XCorrelation("foo", X, Y)

        # Wrong statistics name
        with pytest.raises(ValueError):
            RXY.estimate_whiteness(local_statistic="potato")

        # Wrong number of local weights
        with pytest.raises(IndexError):
            RXY.estimate_whiteness(local_weights=np.array([1, 2]))

        # Wrong length of individual local weight element
        local_weights = np.empty(RXY.R.shape, dtype=np.ndarray)
        local_weights[0, 0] = np.ones(11)
        local_weights[0, 1] = np.ones(3)
        local_weights[1, 0] = np.ones(13)
        local_weights[1, 1] = np.ones(6)  # This is wrong
        with pytest.raises(IndexError):
            RXY.estimate_whiteness(local_weights=local_weights)

        # Wrong number of global weights
        with pytest.raises(IndexError):
            RXY.estimate_whiteness(global_weights=np.array([1, 2]))

    @pytest.mark.plots
    def test_plot(self, correlation_tensors) -> None:
        # Next, remove randoms with known values.
        (
            _,
            _,
            _,
            _,
            _,
            _,
            X,
            Y,
            _,
            _,
            _,
        ) = correlation_tensors

        x0 = X[:, 0].T
        y0 = Y[:, 0].T

        _ = dmv.XCorrelation("foo", x0, y0).plot()
        _ = dmv.XCorrelation("foo", X, y0).plot()
        _ = dmv.XCorrelation("foo", x0, Y).plot()
        _ = dmv.XCorrelation("foo", X, Y).plot()
        _ = dmv.XCorrelation("foo", X, X).plot()
        plt.close("all")


class Test_rsquared:
    def test_rsquared_nominal(self) -> None:
        # Just test that it won't run any error
        # Next, remove randoms with known values.

        y1 = np.array(
            [
                0,
                0.5878,
                0.9511,
                0.9511,
                0.5878,
                0.0000,
                -0.5878,
                -0.9511,
                -0.9511,
                -0.5878,
                -0.0000,
            ]
        )

        y2 = np.array(
            [
                0,
                0.7053,
                1.1413,
                1.1413,
                0.7053,
                0.0000,
                -0.7053,
                -1.1413,
                -1.1413,
                -0.7053,
                -0.0000,
            ]
        )

        y1Calc = np.array(
            [
                0.1403,
                0.8620,
                1.0687,
                1.1633,
                0.9208,
                0.2390,
                -0.4537,
                -0.8314,
                -0.7700,
                -0.4187,
                0.1438,
            ]
        )

        y2Calc = np.array(
            [
                0.2233,
                1.0024,
                1.3110,
                1.3130,
                0.7553,
                0.0098,
                -0.5893,
                -1.0143,
                -0.8798,
                -0.3226,
                0.3743,
            ]
        )

        rsquared_expected_SISO = 91.2775
        rsquared_expected_MIMO = np.array([91.27830428, 91.89543218])

        rsquared_actual_SISO = dmv.rsquared(y1, y1Calc)
        rsquared_actual_MIMO = dmv.rsquared(
            np.array([y1, y2]).T, np.array([y1Calc, y2Calc]).T
        )

        assert np.isclose(
            rsquared_expected_SISO, rsquared_actual_SISO, atol=ATOL
        )

        np.testing.assert_allclose(
            rsquared_expected_MIMO, rsquared_actual_MIMO, atol=ATOL
        )

    @pytest.mark.parametrize(
        "y_values,y_sim_values",
        [
            (np.random.rand(10), np.random.rand(5)),
            (np.random.rand(8, 3), np.random.rand(10)),
            (np.random.rand(10), np.random.rand(10, 3)),
            (np.random.rand(5, 1), np.random.rand(10, 4)),
            (np.random.rand(8, 3), np.random.rand(4, 4)),
            (np.random.rand(10, 4), np.random.rand(15, 3)),
        ],
    )
    def test_rsquared_raise(
        self, y_values: np.ndarray, y_sim_values: np.ndarray
    ) -> None:
        # Just test that it won't run any error
        # Next, remove randoms with known values.
        with pytest.raises(IndexError):
            dmv.rsquared(y_values, y_sim_values)


class Test_whiteness_level_function:
    def test_whiteness_level(self) -> None:
        x1 = np.array([0.1419, -0.4218, 0.9157, -0.7922, 0.9595])
        whiteness_expected = 0.3579244755541881
        whiteness_matrix_expected = np.array([[0.3579244755541881]])

        whiteness_actual, whiteness_matrix_actual = dmv.whiteness_level(x1)

        assert np.isclose(whiteness_expected, whiteness_actual, atol=ATOL)
        np.testing.assert_allclose(
            whiteness_matrix_expected, whiteness_matrix_actual
        )


class Test_validate_models:
    def test_list_of_signals_arg(
        self,
        good_signals_no_nans: list[Signal],
        tmp_path: str,
    ) -> None:
        # You should just get a plot.

        (
            signal_list,
            u_names,
            y_names,
            u_units,
            y_units,
            fixture,
        ) = good_signals_no_nans

        # List of signals
        dataset_in = [s for s in signal_list if s["name"] in u_names]
        dataset_out = [s for s in signal_list if s["name"] in y_names]
        # sim_gppd is a list of 1D array
        small_perturbation = np.random.uniform(
            low=0.0,
            high=1e-4,
            size=(dataset_out[0]["samples"].size, len(y_names)),
        )
        sim_good = np.array(
            [
                s["samples"] + w
                for s, w in zip(dataset_out, small_perturbation.T)
            ]
        ).T

        # It is a (N,q) array
        sim_bad = np.random.random(
            (len(y_names), dataset_out[0]["samples"].size)
        ).T
        sim_bad2 = np.random.random(
            (len(y_names), dataset_out[0]["samples"].size)
        ).T

        # Override if MISO or SISO
        if fixture == "SISO" or fixture == "SIMO":
            dataset_in = [dataset_in[0]]

        if fixture == "MISO" or fixture == "SISO":
            dataset_out = [dataset_out[0]]
            # sim_good = sim_good[:,0]
            sim_bad = sim_bad[:, 0][:, np.newaxis]
            sim_bad2 = sim_bad2[:, 0][:, np.newaxis]

        #  act
        vs = validate_models(
            dataset_in,
            dataset_out,
            simulated_out=[sim_good, sim_bad, sim_bad2],
        )

        expected_outcome = ["PASS", "FAIL", "FAIL"]
        assert list(vs.outcome.values()) == expected_outcome

        # Tests if dataset is correctly stored in the ValidationSession
        # instance
        expected_stored_in = vs.dataset.dataset["INPUT"].to_numpy()
        expected_stored_out = vs.dataset.dataset["OUTPUT"].to_numpy()

        dataset_in_samples = np.column_stack(
            [s["samples"] for s in dataset_in]
        )
        dataset_out_samples = np.column_stack(
            [s["samples"] for s in dataset_out]
        )
        assert np.allclose(expected_stored_in, dataset_in_samples, atol=1e-4)
        assert np.allclose(
            expected_stored_out,
            dataset_out_samples,
            atol=1e-4,
        )

    def test_ndarrays_args(
        self,
        good_signals_no_nans: list[Signal],
        tmp_path: str,
    ) -> None:
        # You should just get a plot.

        (
            signal_list,
            u_names,
            y_names,
            u_units,
            y_units,
            fixture,
        ) = good_signals_no_nans

        input_size = {
            "SISO": 1,
            "SIMO": 1,
            "MISO": len(u_names),
            "MIMO": len(u_names),
        }
        output_size = {
            "SISO": 1,
            "SIMO": len(y_names),
            "MISO": 1,
            "MIMO": len(y_names),
        }

        p = input_size[fixture]
        q = output_size[fixture]

        # List of arrays
        dataset_in = np.array(
            [s["samples"] for s in signal_list if s["name"] in u_names]
        ).T[:, 0:p]
        dataset_out = np.array(
            [s["samples"] for s in signal_list if s["name"] in y_names]
        ).T[:, 0:q]
        sampling_period = signal_list[0]["sampling_period"]
        N = dataset_out.shape[0]

        small_perturbation = np.random.uniform(
            low=0.0,
            high=1e-4,
            size=(N, q),
        )
        sim_good = dataset_out + small_perturbation

        # It is a (N,q) array
        sim_bad = np.random.random((N, q))
        sim_bad2 = np.random.random((N, q))

        # act
        vs = validate_models(
            dataset_in,
            dataset_out,
            simulated_out=[sim_good, sim_bad, sim_bad2],
            sampling_period=sampling_period,
        )

        np.testing.assert_allclose(
            vs.dataset.dataset["INPUT"].to_numpy(), dataset_in, atol=1e-4
        )
        np.testing.assert_allclose(
            vs.dataset.dataset["OUTPUT"].to_numpy(),
            dataset_out,
            atol=1e-4,
        )

    def test_impossible_to_pass(
        self,
        good_signals_no_nans: list[Signal],
        tmp_path: str,
    ) -> None:
        (
            signal_list,
            u_names,
            y_names,
            u_units,
            y_units,
            fixture,
        ) = good_signals_no_nans

        input_size = {
            "SISO": 1,
            "SIMO": 1,
            "MISO": len(u_names),
            "MIMO": len(u_names),
        }
        output_size = {
            "SISO": 1,
            "SIMO": len(y_names),
            "MISO": 1,
            "MIMO": len(y_names),
        }

        p = input_size[fixture]
        q = output_size[fixture]
        # List of arrays
        dataset_in = np.array(
            [s["samples"] for s in signal_list if s["name"] in u_names]
        ).T[:, 0:p]
        dataset_out = np.array(
            [s["samples"] for s in signal_list if s["name"] in y_names]
        ).T[:, 0:q]
        sampling_period = signal_list[0]["sampling_period"]
        N = dataset_out.shape[0]

        small_perturbation = np.random.uniform(
            low=0.0,
            high=1e-4,
            size=(N, q),
        )
        sim_good = dataset_out + small_perturbation

        # It is a (N,q) array
        sim_bad = np.random.random((N, q))
        sim_bad2 = np.random.random((N, q))

        validation_thresholds_dict = {
            "Ruu_whiteness": 0.35,
            "r2": 100,  # This is impossible to achieve
            "Ree_whiteness": 0.35,
            "Rue_whiteness": 0.35,
        }

        # act
        vs = validate_models(
            dataset_in,
            dataset_out,
            simulated_out=[sim_good, sim_bad, sim_bad2],
            sampling_period=sampling_period,
            validation_thresholds=validation_thresholds_dict,
        )

        # Test will fail on r2 because the bad simulation are random noise
        # anyway (they have low autocorrelation values)
        expected_outcome = ["FAIL", "FAIL", "FAIL"]
        assert list(vs.outcome.values()) == expected_outcome

    def test_sampling_period_raise(
        self,
        good_signals_no_nans: list[Signal],
        tmp_path: str,
    ) -> None:
        # You should just get a plot.

        (
            signal_list,
            u_names,
            y_names,
            u_units,
            y_units,
            fixture,
        ) = good_signals_no_nans

        input_size = {
            "SISO": 1,
            "SIMO": 1,
            "MISO": len(u_names),
            "MIMO": len(u_names),
        }
        output_size = {
            "SISO": 1,
            "SIMO": len(y_names),
            "MISO": 1,
            "MIMO": len(y_names),
        }

        p = input_size[fixture]
        q = output_size[fixture]
        # List of arrays
        dataset_in = np.array(
            [s["samples"] for s in signal_list if s["name"] in u_names]
        ).T[:, 0:p]
        dataset_out = np.array(
            [s["samples"] for s in signal_list if s["name"] in y_names]
        ).T[:, 0:q]
        N = dataset_out.shape[0]

        small_perturbation = np.random.uniform(
            low=0.0,
            high=1e-4,
            size=(N, q),
        )
        sim_good = dataset_out + small_perturbation

        # It is a (N,q) array
        sim_bad = np.random.random((N, q))
        sim_bad2 = np.random.random((N, q))

        # act: sampling_period missing
        with pytest.raises(TypeError):
            _ = validate_models(
                dataset_in,
                dataset_out,
                simulated_out=[sim_good, sim_bad, sim_bad2],
            )


class Test_Compute_Statistics:
    def test_compute_statistics(
        self,
    ) -> None:
        test_data = np.array(
            [
                0.49065828,
                0.1754277,
                -0.37027646,
                0.26591682,
                -0.62597191,
                0.89125522,
                -0.14112183,
                -0.16938656,
                0.31309603,
                0.2876763,
            ]
        )

        expected_mean = 0.11172735900000001
        expected_abs_mean = 0.37307871099999995
        expected_inf = 0.89125522
        expected_quad = 0.18949074921298226
        expected_std = 0.420722885595575

        assert np.isclose(
            compute_statistic(data=test_data, statistic="mean"), expected_mean
        )
        assert np.isclose(
            compute_statistic(data=test_data, statistic="abs_mean"),
            expected_abs_mean,
        )
        assert np.isclose(
            compute_statistic(data=test_data, statistic="max"), expected_inf
        )
        assert np.isclose(
            compute_statistic(data=test_data, statistic="quadratic"),
            expected_quad,
        )

        assert np.isclose(
            compute_statistic(data=test_data, statistic="std"), expected_std
        )

        # ============  weighted statistics ==============
        # Parameters for the Gaussian shape
        a = 1  # Amplitude
        mu = 4.5  # Center of the Gaussian (mean)
        sigma = 1  # Standard deviation

        # Generate x values (indices)
        x = np.arange(10)

        # Calculate Gaussian weights
        weights = a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

        expected_mean_weighted = 0.10053961584752258
        expected_abs_mean_weighted = 0.5967800071293234
        expected_inf_weighted = 0.89125522
        expected_quad_weighted = 0.1228105035864878
        expected_std_weighted = 0.6498192687767278

        assert np.isclose(
            compute_statistic(
                data=test_data, statistic="mean", weights=weights
            ),
            expected_mean_weighted,
        )
        assert np.isclose(
            compute_statistic(
                data=test_data, statistic="abs_mean", weights=weights
            ),
            expected_abs_mean_weighted,
        )
        assert np.isclose(
            compute_statistic(
                data=test_data, statistic="max", weights=weights
            ),
            expected_inf_weighted,
        )
        assert np.isclose(
            compute_statistic(
                data=test_data, statistic="quadratic", weights=weights
            ),
            expected_quad_weighted,
        )
        assert np.isclose(
            compute_statistic(
                data=test_data, statistic="std", weights=weights
            ),
            expected_std_weighted,
        )

    def test_compute_statistics_raise(
        self,
    ) -> None:
        X = np.ones(10)
        X_bad = np.ones((2, 2))
        weights_bad1 = np.ones(15)
        weights_bad2 = np.ones((2, 2))
        weights_bad3 = -np.ones((2, 2))
        statistuc_bad = "quadraticcccc"

        # X shall be a 1-D array
        with pytest.raises(IndexError):
            dmv.validation.compute_statistic(X_bad)

        # weights shall have the same length as data
        with pytest.raises(IndexError):
            dmv.validation.compute_statistic(X, weights=weights_bad1)

        # weights shall be a 1-D array
        with pytest.raises(IndexError):
            dmv.validation.compute_statistic(X, weights=weights_bad2)

        # weights shall be all positive
        with pytest.raises(ValueError):
            dmv.validation.compute_statistic(X, weights=weights_bad3)

        # weights shall be all positive
        with pytest.raises(ValueError):
            dmv.validation.compute_statistic(X, statistic=statistuc_bad)
