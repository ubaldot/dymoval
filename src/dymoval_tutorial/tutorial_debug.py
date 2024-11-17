# import numpy as np
from matplotlib.pylab import plt
import h5py
from copy import deepcopy
import numpy as np
import dymoval as dmv
import matplotlib
from pathlib import Path

from dymoval_tutorial.DCMotorModel import (
    DCMotor_model_dt,
)
import dymoval_tutorial.virtual_lab
from dymoval_tutorial.virtual_lab import Ts
import control as ct

matplotlib.use("qtagg")
plt.ioff()

dymoval_tutorial.virtual_lab

measurements_path = Path(__file__).parent
with h5py.File(measurements_path / "DCMotor_measurements.h5", "r") as logs:
    # logs = f["signals"].keys()
    V = logs["signals/V_measured"]
    Ia = logs["signals/Ia_measured"]
    motor_speed = logs["signals/motor_speed_measured"]

    signal_list = []
    for val in [V, Ia, motor_speed]:
        temp: dmv.Signal = {
            "name": val.attrs["name"],
            "samples": val[:],
            "signal_unit": val.attrs["unit"],
            "sampling_period": val.attrs["period"],
            "time_unit": val.attrs["sampling_unit"],
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
    target_sampling_period=Ts,
    tin=40,
    tout=70.0,
)

#
cutoff = 5  # [Hz]
ds_filt = ds.low_pass_filter(
    ("Supply_Voltage", cutoff),
    ("Motor_Speed", cutoff),
    ("Armature_Current", cutoff),
)
ds_filt.name = "Filtered"
(t, u, y) = ds_filt.dataset_values()
# (t, u, y) = ds.dataset_values()

dmv.compare_datasets(ds, ds_filt)
dmv.compare_datasets(ds.remove_means(), ds_filt.remove_means(), kind="power")

# Generated simulated out Model
res_sim = ct.forced_response(DCMotor_model_dt, X0=[0.0, 0.0, 0.0], U=u)
y_sim = res_sim.y.T

# Low-pass filter the output
lpf_ct = ct.ss(-2 * np.pi * cutoff, 2 * np.pi * cutoff, 1, 0)
lpf_dt = ct.c2d(lpf_ct, Ts)
y0_sim_filt = ct.forced_response(lpf_dt, U=y_sim[:, 0]).y.T
y1_sim_filt = ct.forced_response(lpf_dt, U=y_sim[:, 1]).y.T

y_sim_filt = np.hstack((y0_sim_filt, y1_sim_filt))

VS = dmv.validation.validate_models(
    measured_in=u,
    measured_out=y,
    simulated_out=[y_sim_filt, y_sim],
    sampling_period=Ts,
    U_bandwidths=cutoff,
    Y_bandwidths=[cutoff, cutoff],
).trim(1, 35)

print(VS)
# VS.plot_residuals()
VS.plot_simulations(dataset="out")
# %%
# Create a continuous-time low-pass filter (cutoff in rad/s)
omega_c = 2 * np.pi * cutoff  # Convert Hz to rad/s
s = ct.TransferFunction.s  # Laplace variable
H_continuous = omega_c / (s + omega_c)

# Convert the continuous-time transfer function to discrete-time (using zero-order hold method)
H_discrete = ct.c2d(H_continuous, 0.01, method="zoh")
res = ct.forced_response(H_discrete, X0=y_sim[1][0], U=y_sim[1])
y_sim_filt = np.vstack((y_sim[0], res.y[0].T)).T

# vs_unfiltered = dmv.ValidationSession("foo", ds, ignore_input=True)
# sim_labels = ["ia", "motor_speed"]
# vs_filtered = dmv.ValidationSession("foo", ds_filt, ignore_input=True)

# sim_name = "DCMotor_ss_model"
# vs_unfiltered = vs_unfiltered.append_simulation(sim_name, sim_labels, y_sim)
# vs_filtered = vs_filtered.append_simulation(sim_name, sim_labels, y_sim_filt)

# Test validate_models
measured_signals_filtered = ds_filt.dump_to_signals()
# measured_signals = ds_detrend.dump_to_signals()
measured_in_filtered = measured_signals_filtered["INPUT"]
measured_out_filtered = measured_signals_filtered["OUTPUT"]
sampling_period = measured_in_filtered[0]["sampling_period"]

vs_from_validate_models_filtered = dmv.validation.validate_models(
    measured_in=measured_in_filtered,
    measured_out=measured_out_filtered,
    # simulated_out=[y_sim_filt],
    simulated_out=[y_sim.T],
    sampling_period=sampling_period,
)
vs_from_validate_models_filtered_trimmed = (
    vs_from_validate_models_filtered.trim(1, 35)
)

measured_signals_unfiltered = ds.dump_to_signals()
# measured_signals = ds_detrend.dump_to_signals()
measured_in_unfiltered = measured_signals_unfiltered["INPUT"]
measured_out_unfiltered = measured_signals_unfiltered["OUTPUT"]

vs_from_validate_models_unfiltered = dmv.validation.validate_models(
    measured_in=measured_in_unfiltered,
    measured_out=measured_out_unfiltered,
    simulated_out=[y_sim.T],
    sampling_period=sampling_period,
)
vs_from_validate_models_unfiltered_trimmed = (
    vs_from_validate_models_unfiltered.trim(1, 35)
)
# outcome, vvs, _ = dmv.validation.validate_models(
#     measured_in=u,
#     measured_out=y,
#     simulated_out=[y_sim],
#     sampling_period=0.1,
#     ignore_input=True,
# )
