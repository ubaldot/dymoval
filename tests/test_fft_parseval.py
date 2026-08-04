import numpy as np
import dymoval as dmv
from dymoval import Signal


def test_fft_parseval_sine():
    N = 1024
    fs = 100.0
    t = np.arange(N) / fs
    f0 = 5.0

    x = np.sin(2 * np.pi * f0 * t)
    sig = Signal(name="sine", values=x, time=t, unit="")

    freq, Y = sig.fft()

    # folding weights as in dymoval
    fold = np.full(len(Y), 2.0)
    fold[0] = 1.0
    if N % 2 == 0 and len(Y) > 1:
        fold[-1] = 1.0

    time_mean = np.sum(x ** 2) / N
    freq_energy = np.sum((np.abs(Y) ** 2) * fold)

    assert np.isclose(time_mean, freq_energy, rtol=1e-12, atol=0.0)

    # amplitude check: after folding, peak amplitude should be close to 1
    amplitude = np.abs(Y) * fold
    idx = int(np.argmin(np.abs(freq - f0)))
    assert np.isclose(amplitude[idx], 1.0, rtol=1e-2)
