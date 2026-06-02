# -*- coding: utf-8 -*-
"""
Manual test script for Dataset + Signal (NumPy version)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from dymoval.dataset import Dataset, Signal

plt.ioff()
matplotlib.use("qtagg")

# ============================================================
# CONFIGURATION
# ============================================================

fixture_type = "MIMO"  # ["MIMO", "SISO", "SIMO", "MISO"]

np.random.seed(0)


t = np.linspace(0, 1, 500)
y0 = np.sin(2 * np.pi * 5 * t) + 0.2 * np.random.randn(len(t))

s0 = Signal(
    name="y0",
    values=y0,
    time=t,
    unit="V",
    time_unit="s",
)

y1 = np.cos(2 * np.pi * 2 * t) + 0.2 * np.random.randn(len(t))
s1 = Signal(
    name="y1",
    values=y1,
    time=t,
    unit="V",
    time_unit="s",
)


u1 = np.cos(1 * np.pi * 0.5 * t) + 0.2 * np.random.randn(len(t))
s3 = Signal(
    name="u1",
    values=u1,
    time=t,
    unit="V",
    time_unit="s",
)

ds = Dataset.from_signals(inputs=[s3], outputs=[s1, s0])

# %% ============================================================
# SIGNAL GENERATION
# ============================================================


def make_signal(name, dt, values):  # type: ignore
    t = np.arange(len(values)) * dt
    return name, t, values


nan_block = np.full(200, np.nan)

# ---- Inputs
input_defs = [
    (
        "u1",
        0.01,
        np.hstack(
            (10 * np.random.rand(50), nan_block, 5 * np.random.rand(200))
        ),
    ),
    (
        "u2",
        0.05,
        np.hstack(
            (
                15 * np.random.rand(80),
                nan_block[:50],
                30 * np.random.rand(100),
            )
        ),
    ),
    (
        "u3",
        0.1,
        np.hstack((np.random.rand(100), nan_block, np.random.rand(80))),
    ),
]

# ---- Outputs
output_defs = [
    (
        "y1",
        0.1,
        np.hstack((np.random.rand(120), nan_block, np.random.rand(150))),
    ),
    (
        "y2",
        0.1,
        np.hstack((np.random.rand(200), nan_block[:80], np.random.rand(100))),
    ),
    (
        "y3",
        0.1,
        np.hstack((np.random.rand(50), nan_block[:120], np.random.rand(50))),
    ),
    (
        "y4",
        0.1,
        np.hstack((np.random.rand(70), nan_block[:60], np.random.rand(90))),
    ),
]

# ============================================================
# BUILD DATASET (WITH RESAMPLING)
# ============================================================

target_dt = 0.1
t_end = 40.0
common_time = np.arange(0, t_end, target_dt)


def build_dataset(defs):  # type: ignore
    data = {}

    for name, dt, values in defs:
        t = np.arange(len(values)) * dt

        # simple resampling (manual, mimics Dataset.resample testing)
        valid = ~np.isnan(values)
        if np.sum(valid) < 2:
            continue

        interp = np.interp(
            common_time,
            t[valid],
            values[valid],
        )

        data[name] = interp

    return Dataset(common_time, data)


ds_inputs = build_dataset(input_defs)
ds_outputs = build_dataset(output_defs)

# %% Merge datasets
ds = Dataset(
    time=common_time,
    data={**ds_inputs.data, **ds_outputs.data},
)

input_names = [k for k in ds_inputs.data]
output_names = [k for k in ds_outputs.data]

# %% ============================================================
# APPLY FIXTURE TYPE
# ============================================================

if fixture_type == "SISO":
    input_names = input_names[:1]
    output_names = output_names[:1]

elif fixture_type == "MISO":
    output_names = output_names[:1]

elif fixture_type == "SIMO":
    input_names = input_names[:1]

selected = input_names + output_names
ds = Dataset(common_time, {k: ds.data[k] for k in selected})

# ============================================================
# TEST: BASIC PLOT
# ============================================================

print("-> plot()")
ds.plot(*selected)

# overlay
ds.plot(*selected, overlay=True)

plt.pause(1)


# ============================================================
# TEST: COMPARE DATASETS (simulate output)
# ============================================================

print("-> compare")

# create synthetic "processed" dataset
ds_out = ds.detrend()

ds.plot_compare(ds_out, *selected)

plt.pause(1)


# ============================================================
# TEST: RESAMPLING
# ============================================================

print("-> resample")

new_time = np.arange(0, common_time[-1], 0.05)
ds_resampled = ds.resample(new_time)

ds.plot_compare(ds_resampled, *selected)

plt.pause(1)


# ============================================================
# TEST: FFT / SPECTRUM
# ============================================================

print("-> FFT")

spec = ds.fft()

fig, ax = plt.subplots()

for name, (freq, mag) in spec.items():
    ax.plot(freq, mag, label=name)

ax.set_title("Spectrum")
ax.legend()
plt.pause(1)


# ============================================================
# TEST: Interactive Scope (manual)
# ============================================================

print("-> scope")

scope = ds.scope(*selected)
# manual interaction (blocking GUI)


# ============================================================
# DONE
# ============================================================

print("All tests executed.")
