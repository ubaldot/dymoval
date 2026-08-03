# -*- coding: utf-8 -*-
"""Manual (interactive) smoke test for Signal, Dataset and the scopes.

Run it with an interactive backend::

    python manual_tests/test_plots_manual.py

Click on the curves and press 'r' to reset every scope of a figure.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("qtagg")

import dymoval as dmv  # noqa: E402

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

# ============================================================
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
    ds.remove_constant({"y0": 3.0}),
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

# ============================================================
# Comparison
# ============================================================
print("-> plot_spectrum_compare")
dmv.plot_spectrum_compare(ds, ds.detrend(), labels=["raw", "detrended"])

print("All plots created. Close the windows to exit.")
plt.show()
