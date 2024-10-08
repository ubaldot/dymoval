import numpy as np
import matplotlib.pyplot as plt
import h5py
from copy import deepcopy
from itertools import product
import dymoval as dmv
import matplotlib
from dymoval_tutorial.DCMotorModel import DCMotor_dt
import control as ct

matplotlib.use("qtagg")
matplotlib.pyplot.ioff()

path_h5 = "/home/yt75534/dymoval/src/dymoval_tutorial/DCMotorLogs.h5"
# path_h5 = (
#     "/Users/ubaldot/Documents/dymoval/src/dymoval_tutorial/DCMotorLogs.h5"
# )

logs = h5py.File(
    # "/Users/ubaldot/Documents/dymoval/src/dymoval_tutorial/DCMotorLogs.h5",
    path_h5,
    "r",
)
logs["signals"].keys()

V = logs["signals/V"]
ia = logs["signals/ia"]
dot_theta = logs["signals/dot_theta"]

signal_list = []
for val in [V, ia, dot_theta]:
    temp: dmv.Signal = {
        "name": val.attrs["name"].decode("utf-8"),
        "samples": val[:],
        "signal_unit": val.attrs["unit"].decode("utf-8"),
        "sampling_period": val.attrs["period"][0],
        "time_unit": val.attrs["sampling_unit"].decode("utf-8"),
    }
    signal_list.append(deepcopy(temp))

signal_names = [s["name"] for s in signal_list]
u_names = signal_names[0]
y_names = signal_names[1:]


ds = dmv.Dataset(
    "DCMotor",
    signal_list,
    u_names,
    y_names,
    target_sampling_period=0.01,
    tin=40,
    tout=75.0,
)

B3 = 5
sampling_period = signal_list[0]["sampling_period"]
Fs = 1 / sampling_period
stepp = Fs / B3

X = dmv.XCorrelation(
    "foo",
    signal_list[0]["samples"][:-2],
    signal_list[0]["samples"][:-2],
    X_bandwidths=np.array([B3]),
    Y_bandwidths=np.array([B3]),
    sampling_period=sampling_period,
)

whiteness, R = X.estimate_whiteness()

# %%
cutoff = 5  # [Hz]
ds_filt = ds.low_pass_filter(
    (ds.signal_list()[0][1], cutoff), (ds.signal_list()[2][1], cutoff)
)
ds_filt.name = "Filtered"

(t, u, y) = ds_filt.dataset_values()

# Simulate model
res = ct.forced_response(DCMotor_dt, X0=[0.0, 0.0, 0.0], U=u)
y_sim = res.y.T[:, [0, 2]]

measured_signals = ds_filt.dump_to_signals()
measured_in = measured_signals["INPUT"]
measured_out = measured_signals["OUTPUT"]
vs = dmv.validation.validate_models(
    measured_in=measured_in,
    measured_out=measured_out,
    simulated_out=[y_sim],
    ignore_input=True,
    sys_time_constant=0.035,
)

data = 1000 * np.random.random(1000)
# %%
# trim the validation dataset between 1 and 35 seconds
vs = vs.trim(1, 35)

# %%
weights = 10 * np.ones(1000)
print(dmv.validation.compute_statistic(data, "quadratic", weights))


# %%
def autocorrelation(signal):  # type:ignore
    n = len(signal)
    mean = np.mean(signal)
    var = np.var(signal)

    # Compute the autocorrelation
    autocorr = np.correlate(signal - mean, signal - mean, mode="full")
    autocorr = autocorr[n - 1 :]  # Keep only the second half
    autocorr /= var * np.arange(n, 0, -1)  # Normalize
    return autocorr


# Compute the autocorrelation
motor_speed_np = vs.simulations_values()["Sim_0"]["MotorSpeed"].to_numpy()
autocorr_result = autocorrelation(motor_speed_np.flatten())

# Plot the autocorrelation
plt.figure(figsize=(10, 5))
plt.plot(autocorr_result)
plt.title("Autocorrelation")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.xlim(0, 100)  # Limit x-axis to the first 100 lags
plt.axhline(0, color="grey", lw=0.5, ls="--")  # Add a horizontal line at y=0
plt.grid()
plt.show()
