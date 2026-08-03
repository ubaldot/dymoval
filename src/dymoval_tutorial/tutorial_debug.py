"""Script version of the tutorial notebook, handy for debugging.

Run it from anywhere:

    python -m dymoval_tutorial.tutorial_debug
"""

from copy import deepcopy
from pathlib import Path

import control as ct
import h5py
import matplotlib
import matplotlib.pylab as plt
import numpy as np

import dymoval as dmv
from dymoval_tutorial.DCMotorModel import DCMotor_model_dt
from dymoval_tutorial.virtual_lab import Ts

matplotlib.use("qtagg")
plt.ioff()

measurements_file = str(Path(__file__).parent / "DCMotor_measurements.h5")

# ===== Log data clean-up ==========================================
with h5py.File(measurements_file, "r") as logs:
    signal_list = []
    for key in ["V_measured", "Ia_measured", "motor_speed_measured"]:
        val = logs[f"signals/{key}"]
        samples = np.asarray(val[:], dtype=float)
        signal_list.append(
            dmv.Signal(
                name=str(val.attrs["name"]),
                values=samples,
                time=np.arange(len(samples)) * float(val.attrs["period"]),
                unit=str(val.attrs["unit"]),
                time_unit=str(val.attrs["sampling_unit"]),
            )
        )

_ = dmv.plot_signals(*signal_list)

# The signals are logged with different sampling periods and there is some
# missing data outside [40, 70] s.
signal_list = [s.trim(40.0, 70.0) for s in signal_list]

ds = dmv.Dataset.from_signals(
    inputs=signal_list[:1],
    outputs=signal_list[1:],
    target_sampling_period=Ts,
    meta={"name": "DCMotor"},
)

cutoff = 5  # [Hz]
ds_filt = ds.low_pass_filter(
    ("Supply_Voltage", cutoff),
    ("Motor_Speed", cutoff),
    ("Armature_Current", cutoff),
)

_ = dmv.plot_compare(ds, ds_filt, labels=["raw", "filtered"])
_ = dmv.plot_spectrum_compare(
    ds.remove_mean(),
    ds_filt.remove_mean(),
    labels=["raw", "filtered"],
    mode="power",
)

# ===== Validation =================================================
(t, u, y) = ds_filt.dataset_values()

res_sim = ct.forced_response(DCMotor_model_dt, X0=[0.0, 0.0, 0.0], U=u.T)
y_sim = res_sim.y.T

measured_signals = ds_filt.to_signals()

vs = dmv.validate_models(
    measured_in=measured_signals["INPUT"],
    measured_out=measured_signals["OUTPUT"],
    simulated_out=y_sim,
    sampling_period=Ts,
    U_bandwidths=cutoff,
    Y_bandwidths=[cutoff, cutoff],
)

y_sim2 = deepcopy(y_sim)
y_sim2[:, 0] = y_sim2[:, 0] * 0.82

vs = vs.append_simulation(
    "Sim_1", y_names=["Armature_Current", "Motor_Speed"], y_data=y_sim2
)

vs_trimmed = vs.trim(1, 30)
print(vs_trimmed)

_ = vs_trimmed.plot_simulations(dataset="out")
_ = vs_trimmed.plot_residuals()

plt.show()
