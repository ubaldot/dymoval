from copy import deepcopy
import dymoval as dmv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# This script is intended to test plots in a non-interactive environments,
# such as plain python. You should both:
#  1. python -m plain_python_test # Blocking plots
#  2. python and then run
#       >>> script_path = ./manual_test/plain_python_test
#       >>> with open(script_path) as file:
#           ...  exec(file.read())
#
# Test 2. should be non-blocking


def main():
    plt.ioff()
    matplotlib.use("TkAgg")
    # matplotlib.use("qtagg")

    # Set test data
    nan_thing = np.empty(200)
    nan_thing[:] = np.nan

    input_signal_names = ["u1", "u2", "u3"]
    input_sampling_periods = [0.01, 0.1, 0.1]
    input_signal_values = [
        np.hstack(
            (
                10 * np.random.rand(50),
                nan_thing,
                5 * np.random.rand(400),
                nan_thing,
            )
        ),
        np.hstack(
            (
                15 * np.random.rand(20),
                nan_thing[0:5],
                30 * np.random.rand(30),
                nan_thing,
            )
        ),
        np.hstack((np.random.rand(80), nan_thing, np.random.rand(100))),
    ]

    input_signal_units = ["m/s", "%", "°C"]
    #
    in_lst = []
    for ii, val in enumerate(input_signal_names):
        temp_in: dmv.Signal = {
            "name": val,
            "values": input_signal_values[ii],
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
            "values": output_signal_values[ii],
            "signal_unit": output_signal_units[ii],
            "sampling_period": output_sampling_periods[ii],
            "time_unit": "s",
        }
        out_lst.append(deepcopy(temp_out))
    signal_list = [*in_lst, *out_lst]

    # Get a dataset
    ds = dmv.Dataset(
        "mydataset",
        signal_list,
        input_signal_names,
        output_signal_names,
        target_sampling_period=0.1,
        overlap=True,
    )

    # %%

    ds.plot()
    # plt.pause(1)

    # This shall raise because there are NaNs
    # ds.plot_spectrum()

    # %%
    ds = ds.remove_NaNs()

    ds.plot()


if __name__ == "__main__":
    main()
