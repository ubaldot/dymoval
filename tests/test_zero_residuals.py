import numpy as np
from dymoval import rsquared
from dymoval.xcorrelation import whiteness_level


def test_zero_residuals_rsquared_and_whiteness():
    # create deterministic example where y == yhat
    N = 100
    t = np.arange(N)
    x = np.sin(2 * np.pi * 5 * t / 100.0)
    y = x.copy()
    yhat = y.copy()

    r2 = rsquared(x.reshape(-1,1), yhat.reshape(-1,1))[0]
    assert r2 == 100.0

    # whiteness on zero signal should return nan (undefined)
    w, W = whiteness_level(np.zeros((N, 1)))
    assert np.isnan(w)
