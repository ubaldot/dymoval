# -*- coding: utf-8 -*-

import pytest
import dymoval as dmv
import numpy as np
from copy import deepcopy
import pandas as pd

# For more info on parametrized fixtures, look here:
# https://www.youtube.com/watch?v=aQH7hyJn-No

dataset_type = ["MIMO", "SISO", "SIMO", "MISO"]


# ============================================
# All good test Signals (good)
# ============================================
def generate_good_signals(fixture_type):
    nan_thing = np.empty(200)
    nan_thing[:] = np.nan

    # Signals creation
    input_signal_names = ["u1", "u2", "u3"]
    input_sampling_periods = [0.01, 0.1, 0.1]
    input_signal_values = [
        np.hstack(
            (np.random.rand(50), nan_thing, np.random.rand(400), nan_thing)
        ),
        np.hstack(
            (
                np.random.rand(20),
                nan_thing[0:5],
                np.random.rand(30),
                nan_thing,
            )
        ),
        np.hstack((np.random.rand(80), nan_thing, np.random.rand(100))),
    ]

    input_signal_units = ["m/s", "%", "°C"]

    in_lst = []
    for ii, val in enumerate(input_signal_names):
        temp_in: dmv.Signal = {
            "name": val,
            "samples": input_signal_values[ii],
            "signal_unit": input_signal_units[ii],
            "sampling_period": input_sampling_periods[ii],
            "time_unit": "s",
        }
        in_lst.append(deepcopy(temp_in))

    # Output signal
    output_signal_names = ["y1", "y2", "y3", "y4"]
    output_sampling_periods = [0.1, 0.1, 0.1, 0.1]
    output_signal_values = [
        np.hstack(
            (np.random.rand(50), nan_thing, np.random.rand(100), nan_thing)
        ),
        np.hstack(
            (
                np.random.rand(100),
                nan_thing[0:50],
                np.random.rand(150),
                nan_thing,
            )
        ),
        np.hstack(
            (
                np.random.rand(10),
                nan_thing[0:105],
                np.random.rand(50),
                nan_thing,
            )
        ),
        np.hstack(
            (
                np.random.rand(20),
                nan_thing[0:85],
                np.random.rand(60),
                nan_thing,
            )
        ),
    ]

    output_signal_units = ["m/s", "deg", "°C", "kPa"]
    out_lst = []
    for ii, val in enumerate(output_signal_names):
        # This is the syntax for defining a dymoval signal
        temp_out: dmv.Signal = {
            "name": val,
            "samples": output_signal_values[ii],
            "signal_unit": output_signal_units[ii],
            "sampling_period": output_sampling_periods[ii],
            "time_unit": "s",
        }
        out_lst.append(deepcopy(temp_out))
    signal_list = [*in_lst, *out_lst]
    first_output_idx = len(input_signal_names)

    # Adjust based on fixtures
    if fixture_type == "SISO":
        # Slice signal list
        # Pick u1 and y1
        signal_list = [signal_list[0], signal_list[first_output_idx]]
        input_signal_names = input_signal_names[0]
        output_signal_names = output_signal_names[0]
        input_signal_units = input_signal_units[0]
        output_signal_units = output_signal_units[0]
    if fixture_type == "MISO":
        signal_list = [
            *signal_list[:first_output_idx],
            signal_list[first_output_idx],
        ]
        output_signal_names = output_signal_names[0]
        output_signal_units = output_signal_units[0]
    if fixture_type == "SIMO":
        signal_list = [signal_list[0], *signal_list[first_output_idx:]]
        input_signal_names = input_signal_names[0]
        input_signal_units = input_signal_units[0]
    return (
        signal_list,
        input_signal_names,
        output_signal_names,
        input_signal_units,
        output_signal_units,
        fixture_type,
    )


@pytest.fixture(params=dataset_type)
def good_signals(request):  # type: ignore
    fixture_type = request.param
    # General case (MIMO)
    return generate_good_signals(fixture_type)


def generate_good_signals_no_nans(fixture_type):  # type: ignore
    # %% Signals creation
    input_signal_names = ["u1", "u2", "u3"]
    input_sampling_periods = [0.1, 0.1, 0.1]
    input_signal_values = np.concatenate(
        [np.random.rand(50), np.random.rand(50), np.random.rand(50)]
    ).reshape(3, 50)

    input_signal_units = ["m/s", "%", "°C"]

    in_lst = []
    for ii, val in enumerate(input_signal_names):
        temp_in: dmv.Signal = {
            "name": val,
            "samples": input_signal_values[ii],
            "signal_unit": input_signal_units[ii],
            "sampling_period": input_sampling_periods[ii],
            "time_unit": "s",
        }
        in_lst.append(deepcopy(temp_in))

    # Output signal
    output_signal_names = ["y1", "y2", "y3", "y4"]
    output_sampling_periods = [0.1, 0.1, 0.1, 0.1]
    output_signal_values = np.concatenate(
        [
            np.random.rand(50),
            np.random.rand(50),
            np.random.rand(50),
            np.random.rand(50),
        ]
    ).reshape(4, 50)

    output_signal_units = ["m/s", "deg", "°C", "kPa"]
    out_lst = []
    for ii, val in enumerate(output_signal_names):
        # This is the syntax for defining a dymoval signal
        temp_out: dmv.Signal = {
            "name": val,
            "samples": output_signal_values[ii],
            "signal_unit": output_signal_units[ii],
            "sampling_period": output_sampling_periods[ii],
            "time_unit": "s",
        }
        out_lst.append(deepcopy(temp_out))
    signal_list = [*in_lst, *out_lst]
    first_output_idx = len(input_signal_names)

    # %%Adjust based on fixtures
    if fixture_type == "SISO":
        # Slice signal list
        # Pick u1 and y1
        signal_list = [signal_list[0], signal_list[first_output_idx]]
        input_signal_names = input_signal_names[0]
        output_signal_names = output_signal_names[0]
        input_signal_units = input_signal_units[0]
        output_signal_units = output_signal_units[0]
    if fixture_type == "MISO":
        signal_list = [
            *signal_list[:first_output_idx],
            signal_list[first_output_idx],
        ]
        output_signal_names = output_signal_names[0]
        output_signal_units = output_signal_units[0]
    if fixture_type == "SIMO":
        signal_list = [signal_list[0], *signal_list[first_output_idx:]]
        input_signal_names = input_signal_names[0]
        input_signal_units = input_signal_units[0]
    return (
        signal_list,
        input_signal_names,
        output_signal_names,
        input_signal_units,
        output_signal_units,
        fixture_type,
    )


@pytest.fixture(params=dataset_type)
def good_signals_no_nans(request):  # type: ignore
    fixture_type = request.param
    # General case (MIMO)
    return generate_good_signals_no_nans(fixture_type)


# ============================================
# Good DataFrame
# ============================================
def generate_good_dataframe(fixture_type):
    # Create a dummy dataframe
    num_samples = 100
    sampling_period = 0.1
    idx = np.arange(num_samples) * sampling_period
    u_names = ["u1", "u2", "u3"]
    y_names = ["y1", "y2"]
    u_units = ["kPa", "°C", "m/s"]
    y_units = ["kPa", "m/s**2"]
    u_cols = list(zip(u_names, u_units))
    y_cols = list(zip(y_names, y_units))
    cols_name = u_cols + y_cols
    df = pd.DataFrame(
        np.random.randn(num_samples, len(cols_name)),
        index=idx,
        columns=cols_name,
    )
    df.index.name = ("Time", "s")

    if fixture_type == "SISO":
        # Slice signal list
        u_names = u_names[0]
        u_units = u_units[0]
        y_names = y_names[0]
        y_units = y_units[0]
        u_cols = u_cols[0]
        y_cols = y_cols[0]
        cols = [u_cols, y_cols]
    if fixture_type == "MISO":
        # Slice signal list
        y_names = y_names[0]
        y_units = y_units[0]
        y_cols = y_cols[0]
        cols = [*u_cols, y_cols]
    if fixture_type == "SIMO":
        # Slice signal list
        u_names = u_names[0]
        u_units = u_units[0]
        u_cols = u_cols[0]
        cols = [u_cols, *y_cols]
    if fixture_type == "MIMO":
        cols = [*u_cols, *y_cols]
    df = df.loc[:, cols]
    df.columns = df.columns.to_flat_index()
    return df, u_names, y_names, u_units, y_units, fixture_type


@pytest.fixture(params=dataset_type)
def good_dataframe(request):  # type: ignore
    fixture_type = request.param
    return generate_good_dataframe(fixture_type)


def generate_sine_dataframe(fixture_type):
    Ts = 0.1
    N = 100
    t = np.linspace(0, Ts * N, N + 1)

    c1 = 2
    c2 = 3
    c3 = 1

    f1 = 2
    w1 = 2 * np.pi * f1
    f2 = 2.4
    w2 = 2 * np.pi * f2
    f3 = 4.8
    w3 = 2 * np.pi * f3

    u_names = ["u1", "u2", "u3"]
    u_units = ["kPa", "bar", "deg"]
    u_cols = list(zip(u_names, u_units))
    u_values = [
        c1 + np.sin(w1 * t) + np.sin(w2 * t),
        c1 + np.sin(w2 * t),
        c1 + np.sin(w3 * t),
    ]

    y_names = ["y1", "y2", "y3", "y4"]
    y_units = ["deg", "rad/s", "V", "A"]
    y_cols = list(zip(y_names, y_units))
    y_values = [
        c1 + np.sin(w1 * t) + np.sin(w3 * t),
        c3 + np.sin(w3 * t),
        c1 + np.sin(w1 * t) + np.sin(w2 * t) + c2 * np.sin(w3 * t),
        np.sin(w1 * t) - np.sin(w2 * t) - np.sin(w3 * t),
    ]

    data = np.vstack((np.asarray(u_values), np.asarray(y_values))).transpose()

    cols_name = u_cols + y_cols
    df = pd.DataFrame(index=t, columns=cols_name, data=data)
    df.index.name = ("Time", "s")

    if fixture_type == "SISO":
        # Slice signal list
        u_names = u_names[0]
        u_units = u_units[0]
        y_names = y_names[0]
        y_units = y_units[0]
        u_cols = u_cols[0]
        y_cols = y_cols[0]
        cols = [u_cols, y_cols]
    if fixture_type == "MISO":
        # Slice signal list
        y_names = y_names[0]
        y_units = y_units[0]
        y_cols = y_cols[0]
        cols = [*u_cols, y_cols]
    if fixture_type == "SIMO":
        # Slice signal list
        u_names = u_names[0]
        u_units = u_units[0]
        u_cols = u_cols[0]
        cols = [u_cols, *y_cols]
    if fixture_type == "MIMO":
        cols = [*u_cols, *y_cols]
    df = df.loc[:, cols]
    df.columns = df.columns.to_flat_index()
    return df, u_names, y_names, u_units, y_units, fixture_type


@pytest.fixture(params=dataset_type)
def sine_dataframe(request):  # type: ignore
    fixture_type = request.param
    return generate_sine_dataframe(fixture_type)


def generate_constant_ones_dataframes(fixture_type):
    N = 10
    idx = np.linspace(0, 1, N)
    u_names = ["u1", "u2", "u3"]
    u_units = ["m", "m/s", "bar"]
    u_cols = list(zip(u_names, u_units))

    y_names = ["y1", "y2", "y3"]
    y_units = ["deg", "m/s**2", "V"]
    y_cols = list(zip(y_names, y_units))

    cols_name = u_cols + y_cols
    values = np.ones((N, 6))
    df = pd.DataFrame(index=idx, columns=cols_name, data=values)
    df.index.name = ("Time", "s")

    if fixture_type == "SISO":
        # Slice signal list
        u_names = u_names[0]
        u_units = u_units[0]
        y_names = y_names[0]
        y_units = y_units[0]
        u_cols = u_cols[0]
        y_cols = y_cols[0]
        cols = [u_cols, y_cols]
    if fixture_type == "MISO":
        # Slice signal list
        y_names = y_names[0]
        y_units = y_units[0]
        y_cols = y_cols[0]
        cols = [*u_cols, y_cols]
    if fixture_type == "SIMO":
        # Slice signal list
        u_names = u_names[0]
        u_units = u_units[0]
        u_cols = u_cols[0]
        cols = [u_cols, *y_cols]
    if fixture_type == "MIMO":
        cols = [*u_cols, *y_cols]
    df = df.loc[:, cols]
    df.columns = df.columns.to_flat_index()
    return df, u_names, y_names, u_units, y_units, fixture_type


@pytest.fixture(params=dataset_type)
def constant_ones_dataframe(request):  # type: ignore
    # Dataframe of all ones.
    fixture_type = request.param
    return generate_constant_ones_dataframes(fixture_type)


def generate_correlation_tensor():
    x0 = np.array([0.1419, 0.4218, 0.9157, 0.7922, 0.9595])
    x1 = np.array([0.6557, 0.0357, 0.8491, 0.9340, 0.6787])
    X = np.array([x0, x1]).T

    y0 = np.array([0.7577, 0.7431, 0.3922, 0.6555, 0.1712])
    y1 = np.array([0.7060, 0.0318, 0.2769, 0.0462, 0.0971])
    Y = np.array([y0, y1]).T

    X_bandwidths = [5, 10]
    Y_bandwidths = [300, 6]
    sampling_period = 0.01

    # Expected values pre-computed with Matlab
    # Same for all tests
    # Sampling at 100 Hz on a signal with bandwidth at most 10 Hz, you have a
    # big step when downsampling the correlation tensor
    Rx0y0_expected = np.array(
        [
            0.52328964,
            0.07624233,
            -0.13628327,
            -0.25258694,
            -0.81807756,
            0.05148857,
            0.10895678,
            0.26055195,
            0.18641852,
        ]
    )

    Rx1y0_expected = np.array(
        [
            -0.02600611,
            0.62518479,
            -0.42208101,
            0.01827536,
            -0.36291478,
            -0.34616707,
            0.27791823,
            0.20718844,
            0.02860214,
        ]
    )

    Rx0y1_expected = np.array(
        [
            0.1702623,
            0.31046205,
            -0.0438847,
            0.05271451,
            -0.63104697,
            -0.53157813,
            0.28330452,
            0.01671614,
            0.37305028,
        ]
    )

    Rx0y1_expected_partial = np.array([0.31046205, -0.63104697, 0.01671614])

    Rx1y1_expected = np.array(
        [
            -0.00846159,
            0.18921953,
            0.20599152,
            -0.28433695,
            0.19567986,
            -0.80598905,
            0.11347958,
            0.33718008,
            0.057237,
        ]
    )

    Rx1y1_expected_partial = np.array([0.18921953, 0.19567986, 0.33718008])

    return (
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
    )


@pytest.fixture(params=dataset_type)
def correlation_tensors(request):  # type: ignore
    # Dataframe of all ones.
    # fixture_type = request.param
    return generate_correlation_tensor()
