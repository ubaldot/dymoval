import pytest
import pandas as pd
import dymoval as dmv
import numpy as np
from matplotlib import pyplot as plt
from dymoval.config import ATOL
from dymoval.dataset import Signal
from dymoval.validation import validate_models


class Test_ClassValidationNominal:
    def test_init(self, good_dataframe: pd.DataFrame) -> None:
        # Nominal data
        df, u_names, y_names, _, _, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.Dataset(name_ds, df, u_names, y_names, full_time_interval=True)
        name_vs = "my_validation"
        vs = dmv.ValidationSession(name_vs, ds)

        # Check that the passed dataset is correctly stored.
        # Main DataFrame
        assert all(vs.dataset.dataset == ds.dataset)

        for ii in range(4):  # Size of coverage
            assert all(vs.dataset.coverage[ii] == ds.coverage[ii])

        # Version with less lags
        expected_nlags = 11
        vs = dmv.ValidationSession(name_vs, ds, nlags=expected_nlags)
        assert vs._nlags == expected_nlags

        # Version with less lags
        expected_nlags = 11
        vs = dmv.ValidationSession(
            name_vs,
            ds,
            nlags=8,
            acorr_local_weights=np.ones(11),
            xcorr_local_weights=np.ones(22),
        )
        assert vs._nlags == expected_nlags

    def test_random_walk(self, good_dataframe: pd.DataFrame) -> None:
        df, u_names, y_names, _, y_units, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.Dataset(name_ds, df, u_names, y_names, full_time_interval=True)

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
        assert sim1_name in vs.simulations_values().columns.get_level_values(
            "sim_names"
        )
        assert sim1_name in vs._auto_correlation_tensors.keys()
        assert sim1_name in vs._cross_correlation_tensors.keys()
        assert sim1_name in vs._validation_results.columns

        assert np.allclose(sim1_values, vs.simulations_values()[sim1_name])

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
        assert sim2_name in vs.simulations_values().columns.get_level_values(
            "sim_names"
        )
        assert sim2_name in vs._auto_correlation_tensors.keys()
        assert sim2_name in vs._cross_correlation_tensors.keys()
        assert sim2_name in vs._validation_results.columns

        assert np.allclose(sim2_values, vs.simulations_values()[sim2_name])

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
        assert (
            sim1_name
            not in vs.simulations_values().columns.get_level_values("sim_names")
        )
        assert sim1_name not in vs._auto_correlation_tensors.keys()
        assert sim1_name not in vs._cross_correlation_tensors.keys()
        assert sim1_name not in vs._validation_results.columns

        # ============================================
        # Re-add sim and then clear.
        # ============================================
        vs = vs.append_simulation(sim1_name, sim1_labels, sim1_values)

        vs = vs.clear()

        assert [] == list(vs.simulations_values().columns)
        assert [] == list(vs._auto_correlation_tensors.keys())
        assert [] == list(vs._cross_correlation_tensors.keys())
        assert [] == list(vs._validation_results.columns)

    def test_trim(self, good_dataframe: pd.DataFrame) -> None:
        df, u_names, y_names, _, y_units, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.Dataset(name_ds, df, u_names, y_names, full_time_interval=True)

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
            expected_tin, vs.simulations_values().index[0], atol=ATOL
        )
        assert np.isclose(
            expected_tout, vs.simulations_values().index[-1], atol=ATOL
        )

    def test_get_sim_signal_list_and_statistics_raise(
        self, good_dataframe: pd.DataFrame
    ) -> None:
        df, u_names, y_names, _, _, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.Dataset(name_ds, df, u_names, y_names, full_time_interval=True)
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


class Test_ClassValidatioNominal_sim_validation:
    def test_existing_sim_raise(self, good_dataframe: pd.DataFrame) -> None:
        df, u_names, y_names, _, _, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.Dataset(name_ds, df, u_names, y_names, full_time_interval=True)
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

    def test_too_many_signals_raise(self, good_dataframe: pd.DataFrame) -> None:
        df, u_names, y_names, _, _, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.Dataset(name_ds, df, u_names, y_names, full_time_interval=True)

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
        ds = dmv.Dataset(name_ds, df, u_names, y_names, full_time_interval=True)

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
        ds = dmv.Dataset(name_ds, df, u_names, y_names, full_time_interval=True)

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
        ds = dmv.Dataset(name_ds, df, u_names, y_names, full_time_interval=True)

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
        ds = dmv.Dataset(name_ds, df, u_names, y_names, full_time_interval=True)

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
        ds = dmv.Dataset(name_ds, df, u_names, y_names, full_time_interval=True)

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

    def test_drop_simulations_raise(self, good_dataframe: pd.DataFrame) -> None:
        df, u_names, y_names, _, _, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.Dataset(name_ds, df, u_names, y_names, full_time_interval=True)

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
        ds = dmv.Dataset(name_ds, df, u_names, y_names, full_time_interval=True)

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
            Rx1y1_expected,
            Rx2y1_expected,
            Rx1y2_expected,
            Rx2y2_expected,
            X,
            Y,
        ) = correlation_tensors
        x1y1_whiteness_expected = 1.5419e-17
        lags_expected = np.arange(-4, 5)

        x1 = X.T[0]
        y1 = Y.T[0]

        # Call dymoval function
        # OBS! It works only if NUM_DECIMALS = 4, like in Matlab

        # SISO
        XCorr_actual = dmv.XCorrelation("foo", x1, y1)
        Rxy_actual = XCorr_actual.values
        lags_actual = XCorr_actual.lags

        assert np.allclose(Rxy_actual[:, 0, 0], Rx1y1_expected, atol=1e-3)
        assert np.allclose(lags_actual, lags_expected)
        # Check whiteness only for SISO
        assert np.isclose(x1y1_whiteness_expected, XCorr_actual.whiteness)

        # SIMO
        XCorr_actual = dmv.XCorrelation("foo", x1, Y)
        Rxy_actual = XCorr_actual.values
        lags_actual = XCorr_actual.lags

        assert np.allclose(Rxy_actual[:, 0, 0], Rx1y1_expected, atol=1e-3)
        assert np.allclose(Rxy_actual[:, 0, 1], Rx1y2_expected, atol=1e-3)
        assert np.allclose(lags_actual, lags_expected)

        # MISO
        XCorr_actual = dmv.XCorrelation("foo", X, y1)
        Rxy_actual = XCorr_actual.values
        lags_actual = XCorr_actual.lags

        assert np.allclose(Rxy_actual[:, 0, 0], Rx1y1_expected, atol=1e-3)
        assert np.allclose(Rxy_actual[:, 1, 0], Rx2y1_expected, atol=1e-3)
        assert np.allclose(lags_actual, lags_expected)

        # MIMO
        XCorr_actual = dmv.XCorrelation("foo", X, Y)
        Rxy_actual = XCorr_actual.values
        lags_actual = XCorr_actual.lags

        assert np.allclose(Rxy_actual[:, 0, 0], Rx1y1_expected, atol=1e-3)
        assert np.allclose(Rxy_actual[:, 0, 1], Rx1y2_expected, atol=1e-3)
        assert np.allclose(Rxy_actual[:, 1, 0], Rx2y1_expected, atol=1e-3)
        assert np.allclose(Rxy_actual[:, 1, 1], Rx2y2_expected, atol=1e-3)
        assert np.allclose(lags_actual, lags_expected)
        assert XCorr_actual.kind == "cross-correlation"

    def test_initializer_acorr(self, correlation_tensors) -> None:
        X = correlation_tensors[4]
        lags_expected = np.arange(-4, 5)

        x1 = X.T[0]
        Rx1x1_expected = np.array(
            [
                -0.31803681,
                -0.28972141,
                -0.16957768,
                0.27733591,
                1.0,
                0.27733591,
                -0.16957768,
                -0.28972141,
                -0.31803681,
            ]
        )
        x1x1_whiteness_expected = 0.125

        # Act
        XCorr_actual = dmv.XCorrelation("foo", x1, x1)
        Rxx_actual = XCorr_actual.values
        lags_actual = XCorr_actual.lags

        # Assert
        assert np.allclose(Rxx_actual[:, 0, 0], Rx1x1_expected, atol=1e-3)
        assert np.allclose(lags_actual, lags_expected)
        # Check whiteness only for SISO
        assert np.isclose(x1x1_whiteness_expected, XCorr_actual.whiteness)

    def test_nlags(self, correlation_tensors) -> None:

        (
            Rx1y1_expected,
            Rx2y1_expected,
            Rx1y2_expected,
            Rx2y2_expected,
            X,
            Y,
        ) = correlation_tensors

        # Only MIMO test
        nlags = 5
        half_lags = nlags // 2
        is_odd = 1 if nlags % 2 == 1 else 0
        lags_expected_1 = np.arange(-half_lags, half_lags + is_odd)

        mid_point = Rx1y1_expected.shape[0] // 2
        Rx1y1_expected_1 = Rx1y1_expected[
            mid_point - half_lags : mid_point + half_lags + is_odd
        ]
        Rx1y2_expected_1 = Rx1y2_expected[
            mid_point - half_lags : mid_point + half_lags + is_odd
        ]
        Rx2y1_expected_1 = Rx2y1_expected[
            mid_point - half_lags : mid_point + half_lags + is_odd
        ]
        Rx2y2_expected_1 = Rx2y2_expected[
            mid_point - half_lags : mid_point + half_lags + is_odd
        ]

        # Act
        XCorr_actual = dmv.XCorrelation("foo", X, Y, nlags=nlags)
        Rxy_actual = XCorr_actual.values
        lags_actual = XCorr_actual.lags

        assert np.allclose(Rxy_actual[:, 0, 0], Rx1y1_expected_1, atol=1e-3)
        assert np.allclose(Rxy_actual[:, 0, 1], Rx1y2_expected_1, atol=1e-3)
        assert np.allclose(Rxy_actual[:, 1, 0], Rx2y1_expected_1, atol=1e-3)
        assert np.allclose(Rxy_actual[:, 1, 1], Rx2y2_expected_1, atol=1e-3)
        assert np.allclose(lags_actual, lags_expected_1)
        assert Rxy_actual.shape[0] == len(lags_expected_1)

    def test_local_weights_to_lags(self, correlation_tensors) -> None:

        (
            Rx1y1_expected,
            Rx2y1_expected,
            Rx1y2_expected,
            Rx2y2_expected,
            X,
            Y,
        ) = correlation_tensors

        # ==========================
        # Setup
        local_weights = np.ones(4)
        half_nlags = local_weights.size // 2
        is_odd = 1 if local_weights.size % 2 == 1 else 0

        lags_expected_2 = np.arange(-half_nlags, half_nlags + is_odd)
        mid_point = Rx1y1_expected.shape[0] // 2
        Rx1y1_expected_2 = Rx1y1_expected[
            mid_point - half_nlags : mid_point + half_nlags + is_odd
        ]
        Rx1y2_expected_2 = Rx1y2_expected[
            mid_point - half_nlags : mid_point + half_nlags + is_odd
        ]
        Rx2y1_expected_2 = Rx2y1_expected[
            mid_point - half_nlags : mid_point + half_nlags + is_odd
        ]
        Rx2y2_expected_2 = Rx2y2_expected[
            mid_point - half_nlags : mid_point + half_nlags + is_odd
        ]

        # Act
        XCorr_actual = dmv.XCorrelation(
            "foo", X, Y, local_weights=local_weights
        )
        Rxy_actual = XCorr_actual.values
        lags_actual = XCorr_actual.lags

        # Assert
        assert np.allclose(Rxy_actual[:, 0, 0], Rx1y1_expected_2, atol=1e-3)
        assert np.allclose(Rxy_actual[:, 0, 1], Rx1y2_expected_2, atol=1e-3)
        assert np.allclose(Rxy_actual[:, 1, 0], Rx2y1_expected_2, atol=1e-3)
        assert np.allclose(Rxy_actual[:, 1, 1], Rx2y2_expected_2, atol=1e-3)
        assert np.allclose(lags_actual, lags_expected_2)

    def test_local_and_global_weights_to_lags(
        self, correlation_tensors
    ) -> None:

        (
            Rx1y1_expected,
            Rx2y1_expected,
            Rx1y2_expected,
            Rx2y2_expected,
            X,
            Y,
        ) = correlation_tensors
        # ==================================================
        # Setup
        nlags = 2
        local_weights = np.ones(7)
        half_nlags = local_weights.size // 2
        is_odd = 1 if local_weights.size % 2 == 1 else 0

        lags_expected = np.arange(-half_nlags, half_nlags + is_odd)
        mid_point = Rx1y1_expected.shape[0] // 2
        Rx1y1_expected = Rx1y1_expected[
            mid_point - half_nlags : mid_point + half_nlags + is_odd
        ]
        Rx1y2_expected = Rx1y2_expected[
            mid_point - half_nlags : mid_point + half_nlags + is_odd
        ]
        Rx2y1_expected = Rx2y1_expected[
            mid_point - half_nlags : mid_point + half_nlags + is_odd
        ]
        Rx2y2_expected = Rx2y2_expected[
            mid_point - half_nlags : mid_point + half_nlags + is_odd
        ]

        # Act
        XCorr_actual = dmv.XCorrelation(
            "foo", X, Y, local_weights=local_weights, nlags=nlags
        )
        Rxy_actual = XCorr_actual.values
        lags_actual = XCorr_actual.lags

        # Assert
        assert np.allclose(Rxy_actual[:, 0, 0], Rx1y1_expected, atol=1e-3)
        assert np.allclose(Rxy_actual[:, 0, 1], Rx1y2_expected, atol=1e-3)
        assert np.allclose(Rxy_actual[:, 1, 0], Rx2y1_expected, atol=1e-3)
        assert np.allclose(Rxy_actual[:, 1, 1], Rx2y2_expected, atol=1e-3)
        assert np.allclose(lags_actual, lags_expected)


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
        rsquared_expected_MIMO = 92.6092

        rsquared_actual_SISO = dmv.rsquared(y1, y1Calc)
        rsquared_actual_MIMO = dmv.rsquared(
            np.array([y1, y2]).T, np.array([y1Calc, y2Calc]).T
        )

        assert np.isclose(
            rsquared_expected_SISO, rsquared_actual_SISO, atol=ATOL
        )

        assert np.isclose(
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


class Test_whiteness:
    def test_whiteness_level(self) -> None:

        x1 = np.array([0.1419, 0.4218, 0.9157, 0.7922, 0.9595])
        whiteness_expected = 0.125

        whiteness_actual, _, _ = dmv.whiteness_level(x1)

        assert np.isclose(whiteness_expected, whiteness_actual, atol=ATOL)


class Test_Validate_Models:
    def test_list_of_signals(
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

        dataset_in = [s for s in signal_list if s["name"] in u_names]
        dataset_out = [s for s in signal_list if s["name"] in y_names]
        sampling_period = dataset_in[0]["sampling_period"]

        small_perturbation = np.random.uniform(
            low=0.0,
            high=1e-3,
            size=(dataset_out[0]["samples"].size, len(y_names)),
        )
        sim_good = [
            s["samples"] + w for s, w in zip(dataset_out, small_perturbation.T)
        ]

        # It is a (N,q) array
        sim_bad = np.random.random(
            (len(y_names), dataset_out[0]["samples"].size)
        ).T
        sim_bad2 = np.random.random(
            (len(y_names), dataset_out[0]["samples"].size)
        ).T

        # Override if MISO or SISO
        if fixture == "SISO" or fixture == "MISO":
            dataset_in = dataset_in[0]["samples"]
            dataset_out = dataset_out[0]["samples"]

            sim_good = sim_good[0]
            sim_bad = sim_bad[:, 0]
            sim_bad2 = sim_bad2[:, 0]

        # %% act
        global_outcome, vs, validation_thresholds_dict = validate_models(
            dataset_in,
            dataset_out,
            sampling_period,
            sim_good,
            sim_bad,
            sim_bad2,
        )
        expected_outcome = ["PASS", "FAIL", "FAIL"]

        assert global_outcome == expected_outcome
