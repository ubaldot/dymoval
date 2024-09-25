import pytest
import pandas as pd
import dymoval as dmv
import numpy as np
from matplotlib import pyplot as plt
from dymoval.config import ATOL


class Test_ClassValidationNominal:
    def test_init(self, good_dataframe: pd.DataFrame) -> None:
        # Nominal data
        df, u_names, y_names, _, _, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_names, y_names, full_time_interval=True
        )
        name_vs = "my_validation"
        vs = dmv.ValidationSession(name_vs, ds)

        # Check that the passed Dataset is correctly stored.
        # Main DataFrame
        assert all(vs.Dataset.dataset == ds.dataset)

        for ii in range(4):  # Size of coverage
            assert all(vs.Dataset.coverage[ii] == ds.coverage[ii])

    def test_random_walk(self, good_dataframe: pd.DataFrame) -> None:
        df, u_names, y_names, _, y_units, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
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
        assert sim1_name in vs.simulations_results.columns.get_level_values(
            "sim_names"
        )
        assert sim1_name in vs.auto_correlation_tensors.keys()
        assert sim1_name in vs.cross_correlation_tensors.keys()
        assert sim1_name in vs.validation_results.columns

        assert np.allclose(sim1_values, vs.simulations_results[sim1_name])

        # # Add second model
        sim2_name = "Model 2"
        sim2_labels = ["my_y1", "my_y2"]  # The fixture has two outputs
        if fixture == "SISO" or fixture == "MISO":
            # You only have one output
            sim2_labels = [sim1_labels[0]]
        sim2_values = vs.Dataset.dataset["OUTPUT"].values + np.random.rand(
            len(vs.Dataset.dataset["OUTPUT"].values), 1
        )

        vs = vs.append_simulation(sim2_name, sim2_labels, sim2_values)
        # At least the names are there...
        assert sim2_name in vs.simulations_results.columns.get_level_values(
            "sim_names"
        )
        assert sim2_name in vs.auto_correlation_tensors.keys()
        assert sim2_name in vs.cross_correlation_tensors.keys()
        assert sim2_name in vs.validation_results.columns

        assert np.allclose(sim2_values, vs.simulations_results[sim2_name])

        # ===============================================
        # Test simulation_signals_list and list_simulations
        # ============================================
        expected_sims = [sim1_name, sim2_name]

        assert sorted(expected_sims) == sorted(vs.simulations_names())

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
        vs = vs.drop_simulation(sim1_name)
        # At least the names are nt there any longer.
        assert (
            sim1_name
            not in vs.simulations_results.columns.get_level_values(
                "sim_names"
            )
        )
        assert sim1_name not in vs.auto_correlation_tensors.keys()
        assert sim1_name not in vs.cross_correlation_tensors.keys()
        assert sim1_name not in vs.validation_results.columns

        # ============================================
        # Re-add sim and then clear.
        # ============================================
        vs = vs.append_simulation(sim1_name, sim1_labels, sim1_values)

        vs = vs.clear()

        assert [] == list(vs.simulations_results.columns)
        assert [] == list(vs.auto_correlation_tensors.keys())
        assert [] == list(vs.cross_correlation_tensors.keys())
        assert [] == list(vs.validation_results.columns)

    def test_trim(self, good_dataframe: pd.DataFrame) -> None:
        df, u_names, y_names, _, y_units, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
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
        sim2_values = vs.Dataset.dataset["OUTPUT"].values + np.random.rand(
            len(vs.Dataset.dataset["OUTPUT"].values), 1
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
        assert np.isclose(
            expected_tin, vs.Dataset.dataset.index[0], atol=ATOL
        )
        assert np.isclose(
            expected_tout, vs.Dataset.dataset.index[-1], atol=ATOL
        )

        assert np.isclose(
            expected_tin, vs.simulations_results.index[0], atol=ATOL
        )
        assert np.isclose(
            expected_tout, vs.simulations_results.index[-1], atol=ATOL
        )

    def test_get_sim_signal_list_and_statistics_raise(
        self, good_dataframe: pd.DataFrame
    ) -> None:
        df, u_names, y_names, _, _, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
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


class Test_ClassValidatioNominal_sim_validation:
    def test_existing_sim_raise(self, good_dataframe: pd.DataFrame) -> None:
        df, u_names, y_names, _, _, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
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
        ds = dmv.dataset.Dataset(
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

    def test_duplicate_names_raise(
        self, good_dataframe: pd.DataFrame
    ) -> None:
        df, u_names, y_names, _, _, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
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
        ds = dmv.dataset.Dataset(
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

    def test_too_many_values_raise(
        self, good_dataframe: pd.DataFrame
    ) -> None:
        df, u_names, y_names, _, _, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
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
        ds = dmv.dataset.Dataset(
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

    def test_ydata_too_short_raise(
        self, good_dataframe: pd.DataFrame
    ) -> None:
        df, u_names, y_names, _, _, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
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

    def test_drop_simulation_raise(
        self, good_dataframe: pd.DataFrame
    ) -> None:
        df, u_names, y_names, _, _, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
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
            vs.drop_simulation("potato")


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
        ds = dmv.dataset.Dataset(
            name_ds, df, u_names, y_names, full_time_interval=True
        )

        print(ds.dataset)

        name_vs = "my_validation"
        vs = dmv.ValidationSession(name_vs, ds)
        print(vs.simulations_results)

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
        sim2_values = vs.Dataset.dataset["OUTPUT"].values + np.random.rand(
            len(vs.Dataset.dataset["OUTPUT"].values), 1
        )
        vs = vs.append_simulation(sim2_name, sim2_labels, sim2_values)
        print(vs.simulations_results)

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

        _, _ = vs.plot_residuals()
        plt.close("all")

        _, _ = vs.plot_residuals("Model 1")
        plt.close("all")

        _, _ = vs.plot_residuals(["Model 1", "Model 2"])
        plt.close("all")

        _, _ = vs.plot_residuals(["Model 1", "Model 2"])
        plt.close("all")

        # =============================
        # plot residuals raises
        # =============================
        with pytest.raises(KeyError):
            _, _ = vs.plot_residuals("potato")

        # Empty simulation list
        vs = vs.clear()
        with pytest.raises(KeyError):
            _, _ = vs.plot_residuals()


class Test_XCorrelation:
    def test_initializer(self) -> None:
        # Just test that it won't run any error
        # Next, remove randoms with known values.

        x1 = np.array([0.1419, 0.4218, 0.9157, 0.7922, 0.9595])
        x2 = np.array([0.6557, 0.0357, 0.8491, 0.9340, 0.6787])
        X = np.array([x1, x2]).T

        y1 = np.array([0.7577, 0.7431, 0.3922, 0.6555, 0.1712])
        y2 = np.array([0.7060, 0.0318, 0.2769, 0.0462, 0.0971])
        Y = np.array([y1, y2]).T

        # Expected values pre-computed with Matlab
        Rx1y1_expected = np.array(
            [
                0.5233,
                0.0763,
                -0.1363,
                -0.2526,
                -0.8181,
                0.0515,
                0.1090,
                0.2606,
                0.1864,
            ]
        )
        x1y1_whiteness_expected = 1.5419e-17

        Rx1y2_expected = np.array(
            [
                0.1702,
                0.3105,
                -0.0438,
                0.0526,
                -0.6310,
                -0.5316,
                0.2833,
                0.0167,
                0.3730,
            ]
        )

        Rx2y1_expected = np.array(
            [
                -0.0260,
                0.6252,
                -0.4220,
                0.0183,
                -0.3630,
                -0.3462,
                0.2779,
                0.2072,
                0.0286,
            ]
        )

        Rx2y2_expected = np.array(
            [
                -0.0085,
                0.1892,
                0.2061,
                -0.2843,
                0.1957,
                -0.8060,
                0.1135,
                0.3371,
                0.0573,
            ]
        )
        lags_expected = np.arange(-4, 5)

        # Call dymoval function
        # OBS! It works only if NUM_DECIMALS = 4, like in Matlab

        # SISO
        XCorr_actual = dmv.XCorrelation("foo", x1, y1)
        Rxy_actual = XCorr_actual.values
        lags_actual = XCorr_actual.lags

        assert np.allclose(Rxy_actual[:, 0, 0], Rx1y1_expected, atol=1e-4)
        assert np.allclose(lags_actual, lags_expected)
        # Check whiteness only for SISO
        assert np.isclose(x1y1_whiteness_expected, XCorr_actual.whiteness)

        # SIMO
        XCorr_actual = dmv.XCorrelation("foo", x1, Y)
        Rxy_actual = XCorr_actual.values
        lags_actual = XCorr_actual.lags

        assert np.allclose(Rxy_actual[:, 0, 0], Rx1y1_expected, atol=1e-4)
        assert np.allclose(Rxy_actual[:, 0, 1], Rx1y2_expected, atol=1e-4)
        assert np.allclose(lags_actual, lags_expected)

        # MISO
        XCorr_actual = dmv.XCorrelation("foo", X, y1)
        Rxy_actual = XCorr_actual.values
        lags_actual = XCorr_actual.lags

        assert np.allclose(Rxy_actual[:, 0, 0], Rx1y1_expected, atol=1e-4)
        assert np.allclose(Rxy_actual[:, 1, 0], Rx2y1_expected, atol=1e-4)
        assert np.allclose(lags_actual, lags_expected)

        # MIMO
        XCorr_actual = dmv.XCorrelation("foo", X, Y)
        Rxy_actual = XCorr_actual.values
        lags_actual = XCorr_actual.lags

        assert np.allclose(Rxy_actual[:, 0, 0], Rx1y1_expected, atol=1e-4)
        assert np.allclose(Rxy_actual[:, 0, 1], Rx1y2_expected, atol=1e-4)
        assert np.allclose(Rxy_actual[:, 1, 0], Rx2y1_expected, atol=1e-4)
        assert np.allclose(Rxy_actual[:, 1, 1], Rx2y2_expected, atol=1e-4)
        assert np.allclose(lags_actual, lags_expected)
        assert XCorr_actual.kind == "cross-correlation"

        # ===========================
        # Autocorrelation
        # ===========================

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
        x1x1_whiteness_expected = -1.85e-17

        # Act
        XCorr_actual = dmv.XCorrelation("foo", x1, x1)
        Rxx_actual = XCorr_actual.values
        lags_actual = XCorr_actual.lags

        # Assert
        assert np.allclose(Rxx_actual[:, 0, 0], Rx1x1_expected, atol=1e-4)
        assert np.allclose(lags_actual, lags_expected)
        # Check whiteness only for SISO
        assert np.isclose(x1x1_whiteness_expected, XCorr_actual.whiteness)
        # ==========================
        # Reduced number of lags
        # ========================
        # Only MIMO test
        nlags = 3
        lags_expected = np.arange(-nlags, nlags + 1)
        mid_point = Rx1y1_expected.shape[0] // 2
        Rx1y1_expected = Rx1y1_expected[
            mid_point - nlags : mid_point + nlags + 1
        ]
        Rx1y2_expected = Rx1y2_expected[
            mid_point - nlags : mid_point + nlags + 1
        ]
        Rx2y1_expected = Rx2y1_expected[
            mid_point - nlags : mid_point + nlags + 1
        ]
        Rx2y2_expected = Rx2y2_expected[
            mid_point - nlags : mid_point + nlags + 1
        ]

        XCorr_actual = dmv.XCorrelation("foo", X, Y, nlags=nlags)
        Rxy_actual = XCorr_actual.values
        lags_actual = XCorr_actual.lags

        assert np.allclose(Rxy_actual[:, 0, 0], Rx1y1_expected, atol=1e-4)
        assert np.allclose(Rxy_actual[:, 0, 1], Rx1y2_expected, atol=1e-4)
        assert np.allclose(Rxy_actual[:, 1, 0], Rx2y1_expected, atol=1e-4)
        assert np.allclose(Rxy_actual[:, 1, 1], Rx2y2_expected, atol=1e-4)
        assert np.allclose(lags_actual, lags_expected)
        assert Rxy_actual.shape[0] == len(lags_expected)


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
        whiteness_expected = -1.85e-17

        whiteness_actual, _ = dmv.whiteness_level(x1)

        assert np.isclose(whiteness_expected, whiteness_actual, atol=ATOL)
