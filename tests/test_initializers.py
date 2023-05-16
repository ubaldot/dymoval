# -*- coding: utf-8 -*-
"""
The tests herein included check that the object Dataset is correctly instantiated and has
(i.e. the attributes correctly) set when the input Signal/DataFrame format is OK.

When the Signal/DataFrame format is NOK, the utils functions 'dataframe_validation'
and 'signal validation' raise exceptions.
Such validation functions are tested in test_utils.py.
"""
import pytest
import dymoval as dmv
from dymoval.dataset import Signal
import numpy as np
from fixture_data import *  # NOQA


class TestInitializerFromSignals:
    # Check that the object Dataset is correctly built when the Signal format
    # is good.
    def test_nominal(self, good_signals: list[Signal]) -> None:
        # Nominal data
        (
            signal_list,
            input_signal_names,
            output_signal_names,
            input_signal_units,
            output_signal_units,
            fixture,
        ) = good_signals

        target_sampling_period = 0.1
        ds = dmv.Dataset(
            "potato",
            signal_list,
            input_signal_names,
            output_signal_names,
            target_sampling_period=target_sampling_period,
            full_time_interval=True,
            verbosity=1,
        )
        if fixture == "SISO":  # Convert string to list
            input_signal_names = [input_signal_names]
            output_signal_names = [output_signal_names]
            input_signal_units = [input_signal_units]
            output_signal_units = [output_signal_units]
        if fixture == "MISO":
            output_signal_names = [output_signal_names]
            output_signal_units = [output_signal_units]
        if fixture == "SIMO":  # Convert string to list
            input_signal_names = [input_signal_names]
            input_signal_units = [input_signal_units]
        expected_input_names = input_signal_names
        expected_output_names = output_signal_names
        actual_input_names = ds.dataset["INPUT"].columns.get_level_values(
            "names"
        )
        actual_output_names = ds.dataset["OUTPUT"].columns.get_level_values(
            "names"
        )
        # Check that names are well placed
        assert sorted(expected_input_names) == sorted(actual_input_names)
        assert sorted(expected_output_names) == sorted(actual_output_names)

        expected_input_units = input_signal_units
        expected_output_units = output_signal_units
        actual_input_units = ds.dataset["INPUT"].columns.get_level_values(
            "units"
        )
        actual_output_units = ds.dataset["OUTPUT"].columns.get_level_values(
            "units"
        )
        # Check that units are well placed
        assert sorted(expected_input_units) == sorted(actual_input_units)
        assert sorted(expected_output_units) == sorted(actual_output_units)

        # Time index starts from 0.0
        assert np.isclose(ds.dataset.index[0], 0.0, atol=ATOL)
        assert ds.dataset.index.name == ("Time", "s")
        # Check that time vector is periodic
        assert all(
            np.isclose(x, target_sampling_period, atol=ATOL)
            for x in np.diff(ds.dataset.index)
        )
        assert np.isclose(
            ds.dataset.index[1] - ds.dataset.index[0],
            target_sampling_period,
            atol=ATOL,
        )
        # Assert nan_intervals. Take only the first in/out names so you can
        # also work with SISO.
        # Input
        expected_num_nans_u1 = 2  # From fixture
        actual_num_nans_u1 = len(ds._nan_intervals["u1"])
        assert actual_num_nans_u1 == expected_num_nans_u1

        expected_nan1_time_interval = [0.5, 2.4]  # From fixture
        expected_nan2_time_interval = [6.5, 8.4]  # From fixture
        expected_nan_intervals = [
            expected_nan1_time_interval,
            expected_nan2_time_interval,
        ]

        for ii in range(0, expected_num_nans_u1):
            assert np.isclose(
                expected_nan_intervals[ii][0],
                ds._nan_intervals["u1"][ii][0],
                atol=ATOL,
            )
            assert np.isclose(
                expected_nan_intervals[ii][1],
                ds._nan_intervals["u1"][ii][-1],
                atol=ATOL,
            )
        # Output
        expected_num_nans_y1 = 1  # From fixture
        actual_num_nans_y1 = len(ds._nan_intervals["y1"])
        assert actual_num_nans_y1 == expected_num_nans_y1

        expected_nan_interval = [5.0, 8.4]  # From fixture

        assert np.isclose(
            expected_nan_interval[0], ds._nan_intervals["y1"][0][0], atol=ATOL
        )
        assert np.isclose(
            expected_nan_interval[1], ds._nan_intervals["y1"][0][-1], atol=ATOL
        )

        # assert sampling period
        assert np.isclose(ds.sampling_period, target_sampling_period, atol=ATOL)

    def test_no_leftovers(self, good_signals: list[Signal]) -> None:
        # Nominal data
        (
            signal_list,
            input_signal_names,
            output_signal_names,
            input_signal_units,
            output_signal_units,
            fixture,
        ) = good_signals

        target_sampling_period = 0.0001
        with pytest.raises(IndexError):
            _ = dmv.Dataset(
                "potato",
                signal_list,
                input_signal_names,
                output_signal_names,
                target_sampling_period=target_sampling_period,
                full_time_interval=True,
            )


class TestInitializerFromDataframe:
    # Check that the object Dataset is correctly built when the Signal format
    # is good.
    def test_nominal(self, good_dataframe: pd.DataFrame) -> None:
        # Nominal data
        (
            df_expected,
            expected_input_names,
            expected_output_names,
            expected_input_units,
            expected_output_units,
            fixture,
        ) = good_dataframe

        print("df_expected cols = ", df_expected.columns)

        sampling_period = 0.1  # from good_dataframe fixture
        if fixture == "SISO":
            expected_input_names = [expected_input_names]
            expected_output_names = [expected_output_names]
            expected_input_units = [expected_input_units]
            expected_output_units = [expected_output_units]
        if fixture == "MISO":
            expected_output_names = [expected_output_names]
            expected_output_units = [expected_output_units]
        if fixture == "SIMO":
            expected_input_names = [expected_input_names]
            expected_input_units = [expected_input_units]

        print("expected input names = ", expected_input_names)
        # Act
        ds = dmv.dataset.Dataset(
            "potato",
            df_expected,
            expected_input_names,
            expected_output_names,
            full_time_interval=True,
        )

        # assert names
        actual_input_names = ds.dataset["INPUT"].columns.get_level_values(
            "names"
        )
        actual_output_names = ds.dataset["OUTPUT"].columns.get_level_values(
            "names"
        )
        # Names are well placed
        assert sorted(expected_input_names) == sorted(actual_input_names)
        assert sorted(expected_output_names) == sorted(actual_output_names)

        # Assert units
        actual_input_units = ds.dataset["INPUT"].columns.get_level_values(
            "units"
        )
        actual_output_units = ds.dataset["OUTPUT"].columns.get_level_values(
            "units"
        )
        # Names are well placed
        assert sorted(expected_input_units) == sorted(actual_input_units)
        assert sorted(expected_output_units) == sorted(actual_output_units)

        # assert sampling period
        assert np.isclose(ds.sampling_period, sampling_period, atol=ATOL)

    def test_large_dataframe(self, good_dataframe: pd.DataFrame) -> None:
        # Nominal data
        (
            df_expected,
            expected_input_names,
            expected_output_names,
            expected_input_units,
            expected_output_units,
            fixture,
        ) = good_dataframe

        print("df_expected cols = ", df_expected.columns)

        # We only test MIMO case as it seems the most relevant
        if fixture == "MIMO":
            # Arrange
            u_names_test = ["u3", "u1"]
            y_names_test = "y2"

            cols = [
                ("u3", "m/s"),
                ("u1", "kPa"),
                ("y2", "m/s**2"),
            ]
            expected_vals = df_expected.loc[:, cols].to_numpy()

            expected_cols = [
                ("INPUT", "u3", "m/s"),
                ("INPUT", "u1", "kPa"),
                ("OUTPUT", "y2", "m/s**2"),
            ]

            # Act
            ds = dmv.dataset.Dataset(
                "potato",
                df_expected,
                u_names_test,
                y_names_test,
                full_time_interval=True,
            )

            # Assert if df has been filtered and ordered according to user needs
            assert list(ds.dataset.columns) == expected_cols
            assert np.allclose(expected_vals, ds.dataset.to_numpy(), atol=ATOL)

    @pytest.mark.parametrize(
        # Each test is ((tin,tout),(tin_expected,tout_expected))
        "test_input, expected",
        [((4, 8), (0.0, 4.0)), ((3.2, 3.8), (0.0, 0.5)), ((None, 5), (0.0, 5))],
    )
    def test_nominal_tin_tout(
        self,
        good_dataframe: pd.DataFrame,
        test_input: list[float],
        expected: list[float],
    ) -> None:
        # Nominal data
        (
            df_expected,
            expected_input_names,
            expected_output_names,
            _,
            _,
            _,
        ) = good_dataframe
        tin = test_input[0]
        tout = test_input[1]
        expected_tin = expected[0]
        expected_tout = expected[1]
        ds = dmv.dataset.Dataset(
            "potato",
            df_expected,
            expected_input_names,
            expected_output_names,
            tin=tin,
            tout=tout,
        )
        actual_tin = ds.dataset.index[0]
        actual_tout = ds.dataset.index[-1]

        # Names are well placed
        assert np.isclose(actual_tin, expected_tin, atol=ATOL)
        assert np.isclose(actual_tout, expected_tout, atol=ATOL)


class TestInitializerWrongInputData:
    # def test_nominal_missing_tout(self, good_dataframe: pd.DataFrame) -> None:
    #     # Nominal data
    #     df, u_names, y_names, _ = good_dataframe
    #     tin = 0.5
    #     with pytest.raises(Exception):
    #         dmv.dataset.Dataset(
    #             "potato",
    #             df,
    #             u_names,
    #             y_names,
    #             tin=tin,
    #         )

    def test_nominal_tin_ge_tout(self, good_dataframe: pd.DataFrame) -> None:
        # Nominal data
        df, u_names, y_names, _, _, _ = good_dataframe
        # tin > tout
        tin = 0.5
        tout = 0.1
        with pytest.raises(ValueError):
            dmv.dataset.Dataset(
                "potato", df, u_names, y_names, tin=tin, tout=tout
            )

    def test_nominal_wrong_type(self, good_dataframe: pd.DataFrame) -> None:
        # Nominal data
        df, u_names, y_names, _, _, _ = good_dataframe
        # tin > tout
        tin = 0.1
        tout = 0.5
        with pytest.raises(TypeError):
            dmv.dataset.Dataset(
                "potato", "string_type", u_names, y_names, tin=tin, tout=tout
            )


class TestOtherStuff:
    def test__str__(self, good_dataframe: pd.DataFrame) -> None:
        # Nominal data
        df, u_names, y_names, _, _, _ = good_dataframe
        # tin > tout
        tin = 0.1
        tout = 0.5
        ds = dmv.dataset.Dataset(
            "potato",
            df,
            u_names,
            y_names,
            tin=tin,
            tout=tout,
        )

        expected_string = "Dymoval dataset called 'potato'."
        assert ds.__str__() == expected_string
