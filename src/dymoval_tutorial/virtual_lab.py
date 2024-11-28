"""
Given that we don't have a real electric motor, we use a virtual lab.
For the matter of explaining how dymoval works it suffices.

In this script the following happens:
    1. A nominal model is defined. We pretend that this is the real-world
       electric motor and we use it to generate real-world measurements
    2. We generate a reasonable low frequency voltage signal that is used to
       feed the electric motor
    3. Sensors are modeled:
        a. Both the input and output signals are sampled
        b. Some noise is added to make the sensors more realistic
        c. We drop some data to show that sensor may miss some data
        d. We save the sensor readings to a h5 file.
"""

import control as ct
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from typing import Any
import h5py
from copy import deepcopy
from scipy.signal import butter, lfilter
from pathlib import Path

matplotlib.use("qtagg")
plt.ioff()

DEBUG = False


def apply_higher_order_filter(
    data: np.ndarray,
    cutoff_freq: float | int,
    sampling_rate: float | int,
    order: int = 3,
) -> Any:
    nyquist = sampling_rate / 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normalized_cutoff, btype="low", analog=False)
    return lfilter(b, a, data)


def get_low_frequency_input_signal(
    sampling_rate: float | int = 1000,
    duration: float | int = 100,
    bandwidth: float | int = 1,
    mean: float | int = 10.0,
    std: float | int = 20.0,
    clip: bool = False,
) -> Any:
    # Generate white noise as the input signal
    num_samples = int(duration * sampling_rate)
    time = np.linspace(0, duration, num_samples + 1)[:-1]
    rng = np.random.default_rng()
    white_noise = rng.normal(loc=mean, scale=std, size=num_samples)

    # Generate a low_frequency signal from white noise
    low_freq_signal = apply_higher_order_filter(white_noise, bandwidth, sampling_rate)

    if clip:
        # clip negative values to 0 (Voltage always positive)
        low_freq_signal[low_freq_signal < 0] = 0

    return time, low_freq_signal


def get_ss_matrices(
    R: float, L: float, K: float, J: float, b: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A = np.array([[-R / L, 0, -K / L], [0, 0, 1], [K / J, 0, -b / J]])
    B = np.array([[1.0 / L], [0], [0]])
    C = np.array([[1.0, 0, 0], [0, 0, 9.5493]])
    D = np.array([[0], [0]])

    return A, B, C, D


# ====== Real system =====================
# This pretend to be the real electric-motor (unfortunately, we
# don't have one).

# States:
# x1: current
# x2: position
# x3: speed

# Outputs:
# y1 = current
# y2 = speed

# Nominal values
L = 1e-3
R = 1.0
J = 5e-5
b = 1e-4
K = 0.1

A, B, C, D = get_ss_matrices(R=R, L=L, K=K, J=J, b=b)
DCMotor_nominal_ct = ct.ss(A, B, C, D)

# ============ Virtual electric-motor done ======================

# =============== Run "real-world" experiments ==================
sampling_rate = 1000  # For the CT system [Hz]
duration = 100  # [s]
bandwidth = 0.5  # [Hz]
mean = 10.0  # [V]
std = 20.0  # [V]
# Get clean input to stimulate the "real" system
time, u = get_low_frequency_input_signal(
    sampling_rate=sampling_rate,
    duration=duration,
    bandwidth=bandwidth,
    mean=mean,
    std=std,
    clip=True,
)

# TODO: Simulate "real" plant
res_ct = ct.forced_response(DCMotor_nominal_ct, T=time, X0=[0.0, 0.0, 0.0], U=u)
# res_dt = ct.forced_response(
#     ct.c2d(DCMotor_nominal_ct, 1 / sampling_rate),
#     T=time,
#     X0=[0.0, 0.0, 0.0],
#     U=u,
# )

y = res_ct.y

# ==================== Sensor model =====================================
# Add noise to the measurements
rng = np.random.default_rng()
u_noisy = u + rng.uniform(low=-0.4, high=0.4, size=u.shape[0])
y0_noisy = y[0] + rng.uniform(low=-0.02, high=0.02, size=y[0].shape[0])
y1_noisy = y[1] + rng.uniform(low=-100, high=100, size=y[1].shape[0])

# u_noisy = u
# y0_noisy = y[0]
# y1_noisy = y[1]
# Sample the signals...
# Sampling period of the sensors. Ts is also used as time-period
# for model discretization
Ts = 0.01

# Sampling period for y1 = Motor_Speed
Ts_alt = 0.005

u_sampled = u_noisy[:: int(sampling_rate * Ts)]
y0_sampled = y0_noisy[:: int(sampling_rate * Ts)]
y1_sampled = y1_noisy[:: int(sampling_rate * Ts_alt)]
# y_sampled = np.vstack((y0_sampled, y1_sampled)).T
time_sampled = time[:: int(sampling_rate * Ts)]
time_sampled_alt = time[:: int(sampling_rate * Ts_alt)]

# ... and missing values...
# Input
u_measured = deepcopy(u_sampled)
y0_measured = deepcopy(y0_sampled)
y1_measured = deepcopy(y1_sampled)

for tin, tout in ((0, 15), (82, 91)):
    indices = np.where((time_sampled >= tin) & (time_sampled <= tout))
    u_measured[indices] = np.nan

# y0
for tin, tout in ((19, 15), (80, 92)):
    indices = np.where((time_sampled >= tin) & (time_sampled <= tout))
    y0_measured[indices] = np.nan

# y1
for tin, tout in ((76, 96),):
    indices = np.where((time_sampled_alt >= tin) & (time_sampled_alt <= tout))
    y1_measured[indices] = np.nan

# ============= Save log data to file ======================
# Our measurements are ready. Save it to file.
if DEBUG:
    measurements_file = str(Path(__file__).parent / "DCMotor_measurements.h5")
else:
    # Be sure in ./dymoval/stc/dymoval_tutorial folder
    measurements_file = "./DCMotor_measurements.h5"

if DEBUG or not Path(measurements_file).exists():
    with h5py.File(measurements_file, "w") as h5file:
        # Create a group named "signals"
        signals_group = h5file.create_group("signals")

        # Store the arrays under the specified names with attributes
        # V = signals_group.create_dataset("V", data=u_measured)
        V = signals_group.create_dataset("V", data=u)
        V.attrs["name"] = "Supply_Voltage"
        V.attrs["unit"] = "V"
        V.attrs["period"] = 1 / sampling_rate
        V.attrs["sampling_unit"] = "s"

        # Ia = signals_group.create_dataset("Ia", data=y0_measured)
        Ia = signals_group.create_dataset("Ia", data=y[0])
        Ia.attrs["name"] = "Armature_Current"
        Ia.attrs["unit"] = "A"
        Ia.attrs["period"] = 1 / sampling_rate
        Ia.attrs["sampling_unit"] = "s"

        motor_speed = signals_group.create_dataset(
            # "motor_speed", data=y1_measured
            "motor_speed",
            data=y[1],
        )
        motor_speed.attrs["name"] = "Motor_Speed"
        motor_speed.attrs["unit"] = "RPM"
        motor_speed.attrs["period"] = 1 / sampling_rate
        motor_speed.attrs["sampling_unit"] = "s"

        # Store the arrays under the specified names with attributes
        # V = signals_group.create_dataset("V", data=u_measured)
        V = signals_group.create_dataset("V_measured", data=u_measured)
        V.attrs["name"] = "Supply_Voltage"
        V.attrs["unit"] = "V"
        V.attrs["period"] = Ts
        V.attrs["sampling_unit"] = "s"

        # Ia = signals_group.create_dataset("Ia", data=y0_measured)
        Ia = signals_group.create_dataset("Ia_measured", data=y0_measured)
        Ia.attrs["name"] = "Armature_Current"
        Ia.attrs["unit"] = "A"
        Ia.attrs["period"] = Ts
        Ia.attrs["sampling_unit"] = "s"

        motor_speed = signals_group.create_dataset(
            "motor_speed_measured",
            data=y1_measured,
        )
        motor_speed.attrs["name"] = "Motor_Speed"
        motor_speed.attrs["unit"] = "RPM"
        motor_speed.attrs["period"] = Ts_alt
        motor_speed.attrs["sampling_unit"] = "s"
