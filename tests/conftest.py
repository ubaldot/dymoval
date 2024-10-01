# -*- coding: utf-8 -*-

import pytest
import dymoval as dmv
import numpy as np
from copy import deepcopy
import pandas as pd

from dymoval.dataset import Signal
from typing import Tuple, List, Literal, Iterable

# For more info on parametrized fixtures, look here:
# https://www.youtube.com/watch?v=aQH7hyJn-No

dataset_type = ["MIMO", "SISO", "SIMO", "MISO"]


# ============================================
# All good test Signals (good)
# ============================================
def generate_good_signals(
    fixture_type,
) -> Tuple[
    List[Signal],
    List[str] | str,
    List[str] | str,
    List[str] | str,
    List[str] | str,
    Literal[*dataset_type],
]:
    nan_thing = np.empty(200)
    nan_thing[:] = np.nan

    # Signals creation
    input_signal_names: str | List[str] = ["u1", "u2", "u3"]
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

    input_signal_units: str | List[str] = ["m/s", "%", "°C"]

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
    output_signal_names: str | List[str] = ["y1", "y2", "y3", "y4"]
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

    output_signal_units: str | List[str] = ["m/s", "deg", "°C", "kPa"]
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


def generate_good_signals_no_nans(fixture_type: str) -> Tuple[
    List[Signal],
    List[str] | str,
    List[str] | str,
    List[str] | str,
    List[str] | str,
    Literal[*dataset_type],
]:
    # %% Signals creation
    input_signal_names: str | List[str] = ["u1", "u2", "u3"]
    input_sampling_periods = [0.1, 0.1, 0.1]
    input_signal_values = np.concatenate(
        [np.random.rand(50), np.random.rand(50), np.random.rand(50)]
    ).reshape(3, 50)

    input_signal_units: str | List[str] = ["m/s", "%", "°C"]

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
    output_signal_names: str | List[str] = ["y1", "y2", "y3", "y4"]
    output_sampling_periods = [0.1, 0.1, 0.1, 0.1]
    output_signal_values = np.concatenate(
        [
            np.random.rand(50),
            np.random.rand(50),
            np.random.rand(50),
            np.random.rand(50),
        ]
    ).reshape(4, 50)

    output_signal_units: str | List[str] = ["m/s", "deg", "°C", "kPa"]
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
def generate_good_dataframe(fixture_type) -> Tuple[
    pd.DataFrame,
    List[str] | str,
    List[str] | str,
    List[str] | str,
    List[str] | str,
    Literal[*dataset_type],
]:
    # Create a dummy dataframe
    num_samples = 100
    sampling_period = 0.1
    idx = np.arange(num_samples) * sampling_period
    u_names: str | List[str] = ["u1", "u2", "u3"]
    y_names: str | List[str] = ["y1", "y2"]
    u_units: str | List[str] = ["kPa", "°C", "m/s"]
    y_units: str | List[str] = ["kPa", "m/s**2"]
    u_cols: Iterable[str] = list(zip(u_names, u_units))
    y_cols: Iterable[str] = list(zip(y_names, y_units))
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


def generate_sine_dataframe(fixture_type) -> Tuple[
    pd.DataFrame,
    List[str] | str,
    List[str] | str,
    List[str] | str,
    List[str] | str,
    Literal[*dataset_type],
]:
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

    u_names: str | List[str] = ["u1", "u2", "u3"]
    u_units: str | List[str] = ["kPa", "bar", "deg"]
    u_cols = list(zip(u_names, u_units))
    u_values = [
        c1 + np.sin(w1 * t) + np.sin(w2 * t),
        c1 + np.sin(w2 * t),
        c1 + np.sin(w3 * t),
    ]

    y_names: str | List[str] = ["y1", "y2", "y3", "y4"]
    y_units: str | List[str] = ["deg", "rad/s", "V", "A"]
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


def generate_constant_ones_dataframes(fixture_type) -> Tuple[
    pd.DataFrame,
    List[str] | str,
    List[str] | str,
    List[str] | str,
    List[str] | str,
    Literal[dataset_type],
]:

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


def generate_correlation_tensor() -> (
    Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]
):
    x1 = np.array([0.1419, 0.4218, 0.9157, 0.7922, 0.9595])
    x2 = np.array([0.6557, 0.0357, 0.8491, 0.9340, 0.6787])
    X = np.array([x1, x2]).T

    y1 = np.array([0.7577, 0.7431, 0.3922, 0.6555, 0.1712])
    y2 = np.array([0.7060, 0.0318, 0.2769, 0.0462, 0.0971])
    Y = np.array([y1, y2]).T

    # Expected values pre-computed with Matlab
    # Same for all tests
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

    return (
        Rx1y1_expected,
        Rx2y1_expected,
        Rx1y2_expected,
        Rx2y2_expected,
        X,
        Y,
    )


@pytest.fixture(params=dataset_type)
def correlation_tensors(request):  # type: ignore
    # Dataframe of all ones.
    # fixture_type = request.param
    return generate_correlation_tensor()
