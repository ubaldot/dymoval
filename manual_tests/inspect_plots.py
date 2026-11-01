# -*- coding: utf-8 -*-
"""Visual inspection of plots and interactive scopes, outside pytest.

Run it with an interactive backend::

    python manual_tests\\inspect_plots.py

Click on the curves and press 'r' to reset every scope of a figure.
Use ``--no-show`` with an Agg backend for a noninteractive smoke check.
"""

import argparse

import matplotlib

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--backend", default="qtagg", help="Matplotlib backend (default: qtagg)."
)
parser.add_argument(
    "--no-show",
    action="store_true",
    help="Create and render plots without windows or graphical trimming.",
)
args = parser.parse_args()
matplotlib.use(args.backend)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import dymoval as dmv  # noqa: E402

# This visual checklist intentionally keeps many figures open together.
plt.rcParams["figure.max_open_warning"] = 0
np.random.seed(0)

# ============================================================
# Signals
# ============================================================
t = np.linspace(0, 1, 500)

u1 = dmv.Signal(
    name="u1",
    values=np.cos(2 * np.pi * 0.5 * t) + 0.2 * np.random.randn(len(t)),
    time=t,
    unit="V",
)
y0 = dmv.Signal(
    name="y0",
    values=np.sin(2 * np.pi * 5 * t) + 0.2 * np.random.randn(len(t)) + 3.0,
    time=t,
    unit="m",
)
y1 = dmv.Signal(
    name="y1",
    values=np.cos(2 * np.pi * 2 * t) + 0.2 * np.random.randn(len(t)),
    time=t,
    unit="m",
)

ds = dmv.Dataset.from_signals(inputs=[u1], outputs=[y0, y1])

# %% ============================================================
# Signal
# ============================================================
print("-> Signal.plot()")
y0.plot(with_scope=False)
y0.plot(with_scope=True)

print("-> Signal.plot_spectrum()")
for mode in dmv.SPECTRUM_MODES:
    y0.plot_spectrum(mode=mode, with_scope=True)

# ============================================================
# Processing
# ============================================================
print("-> remove_mean / remove_constant / detrend")
dmv.plot_compare(
    ds,
    ds.remove_mean(),
    ds.remove_constant(("y0", 3.0)),
    ds.detrend(),
    labels=["raw", "mean removed", "constant removed", "detrended"],
)

# ============================================================
# Dataset
# ============================================================
print("-> Dataset.plot() with grouping")
ds.plot(("u1", "y1"), "y0", with_scope=False)
ds.plot(("u1", "y1"), "y0", with_scope=True)

print("-> Dataset.plot_spectrum() with grouping")
for mode in dmv.SPECTRUM_MODES:
    ds.plot_spectrum(("u1", "y1"), "y0", mode=mode, with_scope=True)

print("-> Dataset.plot_xy()")
ds.plot_xy("u1", "y1")

print("-> Dataset.plot_coverage()")
ds.plot_coverage()
ds.plot_coverage("u1", "y1", nbins=50)

print("-> Dataset.trim() / low_pass_filter() / apply()")
dmv.plot_compare(
    ds,
    ds.low_pass_filter(("u1", 1.0), ("y1", 1.5)),
    ds.apply(("u1", np.abs)),
    labels=["raw", "low-pass filtered", "abs(u1)"],
)

# %% ============================================================
# Missing data
# ============================================================
print("-> NaN shading")
holed = ds.apply(("y0", lambda v: np.where((t > 0.3) & (t < 0.4), np.nan, v)))
print("   gaps:", holed.nan_intervals()["y0"])
dmv.plot_compare(
    holed,
    holed.remove_nans(),
    labels=["with gaps (shaded)", "interpolated"],
)

# ============================================================
# Comparison
# ============================================================
print("-> plot_spectrum_compare")
dmv.plot_spectrum_compare(ds, ds.detrend(), labels=["raw", "detrended"])

# ============================================================
# Validation
# ============================================================
print("-> ValidationSession")
vs = dmv.ValidationSession("manual", ds)

y_measured = np.column_stack([ds["y0"].values, ds["y1"].values])
vs = vs.append_simulation(
    "good model", ["y0", "y1"], y_measured + 0.05 * np.random.randn(len(t), 2)
)
vs = vs.append_simulation(
    "bad model", ["y0", "y1"], np.random.randn(len(t), 2)
)
print(vs)

print("-> ValidationSession.plot_simulations()")
vs.plot_simulations()
vs.plot_simulations(dataset="both")
vs.plot_simulations("good model", dataset="out", with_scope=False)

print("-> ValidationSession.plot_residuals()")
vs.plot_residuals()

print("-> FrequencyResponse.plot()")
ds.spa().plot()

print("-> Panel-aware sizing")
ds.plot(autosize=True)
y0.plot(autosize=True)

if args.no_show:
    for number in plt.get_fignums():
        plt.figure(number).canvas.draw()
    plt.close("all")
    print("All plots rendered. Interactive behavior was not checked.")
else:
    print("All plots created. Inspect legends, labels, and scope panels.")
    print("Click curves, press 'r' to reset, then close the windows.")
    plt.show()

    # Zoom on the interval you want to keep, then close the window.
    print("-> Dataset.trim() graphical picking")
    print("   zoom on the interval to keep, then close the window")
    print("   trimmed to:", ds.trim(verbosity=1).time()[[0, -1]])

    print("-> ValidationSession.trim() graphical picking")
    vs.trim(verbosity=1)
