# import numpy as np
import matplotlib.pylab as plt
import h5py
from copy import deepcopy
import dymoval as dmv
import matplotlib
from pathlib import Path

from dymoval_tutorial.DCMotorModel import (
    DCMotor_model_dt,
)
from dymoval_tutorial.virtual_lab import Ts, DEBUG
import control as ct

matplotlib.use("qtagg")
plt.ioff()

if DEBUG:
    measurements_file = str(Path(__file__).parent / "DCMotor_measurements.h5")
else:
    # Be sure to be in ./dymoval/src/dymoval_tutorial folder
    measurements_file = "./DCMotor_measurements.h5"

# with h5py.File(measurements_file, "r") as logs:
#     # logs = f["signals"].keys()
#     V = logs["signals/V_measured"]
#     Ia = logs["signals/Ia_measured"]
#     motor_speed = logs["signals/motor_speed_measured"]

#     signal_list = []
#     for val in [V, Ia, motor_speed]:
#         temp: dmv.Signal = {
#             "name": val.attrs["name"],
#             "samples": val[:],
#             "signal_unit": val.attrs["unit"],
#             "sampling_period": val.attrs["period"],
#             "time_unit": val.attrs["sampling_unit"],
#         }
#         signal_list.append(deepcopy(temp))


#

logs = h5py.File("./DCMotor_measurements.h5", "r")
V = logs["signals/V_measured"]  # Measured supply voltage (input)
Ia = logs["signals/Ia_measured"]  # Measured current (output)
motor_speed = logs[
    "signals/motor_speed_measured"
]  # Measured motor rotational speed (output)

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

logs.close()

signal_list[0]["samples"]

#
_ = dmv.plot_signals(*signal_list)
#
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

# dmv.compare_datasets(ds, ds_filt)
# dmv.compare_datasets(ds.remove_means(), ds_filt.remove_means(), kind="power")

# Generated simulated out Model
res_sim = ct.forced_response(DCMotor_model_dt, X0=[0.0, 0.0, 0.0], U=u)
y_sim = res_sim.y.T

# Low-pass filter the output
# lpf_ct = ct.ss(-2 * np.pi * cutoff, 2 * np.pi * cutoff, 1, 0)
# lpf_dt = ct.c2d(lpf_ct, Ts)
# y0_sim_filt = ct.forced_response(lpf_dt, U=y_sim[:, 0]).y.T
# y1_sim_filt = ct.forced_response(lpf_dt, U=y_sim[:, 1]).y.T

# y_sim_filt = np.hstack((y0_sim_filt, y1_sim_filt))

measured_signals = ds_filt.dump_to_signals()
measured_in = measured_signals["INPUT"]
measured_out = measured_signals["OUTPUT"]
sampling_period = measured_in[0]["sampling_period"]

vs = dmv.validation.validate_models(
    # measured_in=u[:, np.newaxis],
    measured_in=measured_in,
    # measured_out=y,
    measured_out=measured_out,
    simulated_out=y_sim,
    sampling_period=Ts,
    U_bandwidths=cutoff,
    Y_bandwidths=[cutoff, cutoff],
)


y_sim2 = deepcopy(y_sim)
y_sim2[:, 0] = y_sim2[:, 0] * 0.82

vs = vs.append_simulation("Sim_1", y_names=["out0", "out1"], y_data=y_sim2)

vs_trimmed = vs.trim(1, 30)
print(vs_trimmed)
# vs.plot_residuals()
# vs.plot_simulations(dataset="out")
