# -*- coding: utf-8 -*-
# ===========================================================================
# In this tutorial we show the main functionalities of dymoval.
# The code is divided in code-cell blocks so you can run it piece-by-piece
# and analyze the result.
# ===========================================================================


from copy import deepcopy
import dymoval as dmv
from tests.conftest import (
    generate_correlation_tensor,
    generate_good_dataframe,
    generate_good_signals_no_nans,
)
from dymoval.validation import (
    XCorrelation,
    ValidationSession,
    whiteness_level,
)

import numpy as np
import matplotlib

matplotlib.use("qtagg")
matplotlib.pyplot.ioff()
# You should just get a plot.
(
    signal_list,
    u_names,
    y_names,
    u_units,
    y_units,
    fixture,
) = generate_good_signals_no_nans("MIMO")


# dmv.validate_signals
dmv.validate_signals(*signal_list)

ds = dmv.Dataset(
    "my_dataset",
    signal_list,
    u_names,
    y_names,
    target_sampling_period=0.1,
    overlap=True,
    tin=0,
    tout=8,
    verbosity=1,
)

# Once the dymoval Dataset is created, it is possible to handle with the NaN:s
# in a number of ways
ds = ds.remove_NaNs()

# ax = ds.plot()

# ds.plot_coverage()

# Other methods of the class Dataset are self-explanatory.
#
# ===========================================================================
# validation module
# ===========================================================================
# Test XCorrelation constructor

results = generate_correlation_tensor()

Rx0y0_expected = results[0]
Rx0y1_expected = results[1]
Rx1y0_expected = results[2]
Rx0y1_expected_partial = results[3]
Rx1y1_expected = results[4]
Rx1y1_expected_partial = results[5]
X = results[6]
Y = results[7]
X_bandwidths = results[8]
Y_bandwidths = results[9]
sampling_period = results[10]


x0 = X[:, 0].T
y0 = Y[:, 0].T

RXY_full = XCorrelation(
    "RXY",
    X=X,
    Y=Y,
)

RXY_full_with_params = XCorrelation(
    "RXY",
    X=X,
    Y=Y,
    X_bandwidths=X_bandwidths,
    Y_bandwidths=Y_bandwidths,
    sampling_period=sampling_period,
)


nlags = np.array([[5, 3], [6, 4]])
RXY = XCorrelation(
    "RXY",
    X=X,
    Y=Y,
    nlags=nlags,
    X_bandwidths=X_bandwidths,
    Y_bandwidths=Y_bandwidths,
    sampling_period=sampling_period,
)

# local_weights = np.empty(RXY.R.shape, dtype=np.ndarray)
# local_weights[0, 0] = np.ones(11)
# local_weights[0, 1] = np.ones(3)
# local_weights[1, 0] = np.ones(13)
# local_weights[1, 1] = np.ones(5)
# w, W = RXY.estimate_whiteness(local_weights=local_weights)
# print(
#     RXY_full.estimate_whiteness(
#         local_statistic="quadratic", global_statistic="abs_mean"
#     )
# )
# %% ============ Test validation session with args ===============
fixture_type = "MIMO"
df, u_names, y_names, _, y_units, fixture = generate_good_dataframe(
    fixture_type
)

name_ds = "my_dataset"
ds1 = dmv.Dataset(name_ds, df, u_names, y_names, full_time_interval=True)

name_vs = "my_validation"
# u_nlags = np.array([[5, 3, 2, 5], [6, 4, 4, 8], [8, 10, 7, 22]])
# TODO Test when the size of the matrix is less that pxp
# u_nlags = np.array([[5, 3], [6, 4]])
# eps_nlags = np.array([[5, 12], [8, 30]])
# eps_nlags_wrong_size = np.array([[5, 12, 99], [8, 30, 21], [11, 11, 22]])
# ueps_nlags_wrong_size = np.array([[5, 12, 99], [8, 30, 21], [11, 11, 22]])
# ueps_nlags = np.array([[5, 12], [8, 11]])


local_weights = np.empty((3, 2), dtype=np.ndarray)
local_weights[0, 0] = np.ones(41)
local_weights[0, 1] = np.ones(41)
local_weights[1, 0] = np.ones(41)
local_weights[1, 1] = np.ones(41)
local_weights[2, 0] = np.ones(41)
local_weights[2, 1] = np.ones(22)
vs = ValidationSession(name_vs, ds1, ueps_xcorr_local_weights=local_weights)

# eps_nlags_wrong_size = np.array([[5], [8]])
# vs_weird = ValidationSession(
#     name_vs,
#     ds,
#     eps_acorr_nlags=eps_nlags_wrong_size,
# )


# %% Add one model
sim1_name = "Model 1"
sim1_labels = ["my_y1", "my_y2"]  # The fixture has two outputs
if fixture == "SISO" or fixture == "MISO":
    sim1_labels = [sim1_labels[0]]
sim1_values = np.random.rand(len(df.iloc[:, 0].values), len(sim1_labels))

vs = vs.append_simulation(sim1_name, sim1_labels, sim1_values)
# %%
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


R_trim = dmv.XCorrelation(
    "foo",
    signal_list[0]["samples"],
    signal_list[1]["samples"],
)

# Rue = dmv.XCorrelation(
#     "", signal_list[0]["samples"], signal_list[1]["samples"]
# )
# Rue.plot()


# %%
vs = vs.trim(dataset="out")

# %%
fig = vs.plot_simulations()

# fig.savefig("foo.png")

vs.validation_results

vs.clear()
