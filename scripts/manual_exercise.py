"""
Manual exercise script for dymoval.

Generates a sine wave, wraps it in dymoval.Signal, compares dymoval FFT/spectrum
with numpy's FFT and checks Parseval's theorem. Run locally to eyeball numbers
or paste results here for review.

Usage: python scripts/manual_exercise.py
"""

import numpy as np

import dymoval as dmv
from dymoval import Signal


def main():
    N = 1024
    fs = 100.0
    t = np.arange(N) / fs
    f0 = 5.0

    # Pure sine of amplitude 1
    x = np.sin(2 * np.pi * f0 * t)

    sig = Signal(name="sine", values=x, time=t, unit="")

    # dymoval's fft (rfft normalised by N)
    freq, Y = sig.fft()

    # numpy equivalent (rfft then divide by N)
    Y_np = np.fft.rfft(x) / N

    # one-sided folding weights (same logic as dymoval)
    fold = np.full(len(Y), 2.0)
    fold[0] = 1.0
    if N % 2 == 0 and len(Y) > 1:
        fold[-1] = 1.0

    amplitude = np.abs(Y) * fold
    amplitude_np = np.abs(Y_np) * fold

    # locate closest frequency bin to f0
    idx = int(np.argmin(np.abs(freq - f0)))

    print(f"freq[{idx}] = {freq[idx]:.6f} Hz (target {f0} Hz)")
    print(f"dymoval amplitude at f0: {amplitude[idx]:.6g}")
    print(f"numpy amplitude at f0:   {amplitude_np[idx]:.6g}")

    # Parseval (mean-square) check:
    # With y = rfft(x)/N and one-sided folding, the sum(|y|^2 * fold)
    # equals the time-domain mean-square = sum(x**2) / N.
    energy_time_mean = np.sum(x ** 2) / N
    energy_freq = np.sum((np.abs(Y) ** 2) * fold)
    print(f"time-domain mean-square: {energy_time_mean:.8g}")
    print(f"frequency-domain (folded) energy: {energy_freq:.8g}")
    print(f"ratio time_mean/freq: {energy_time_mean/energy_freq:.12g}")

    # Also use the convenience spectrum() method (amplitude mode)
    f_s, amp_s = sig.spectrum(mode="amplitude")
    print(f"spectrum() amplitude at f0: {amp_s[idx]:.6g}")


if __name__ == "__main__":
    main()
