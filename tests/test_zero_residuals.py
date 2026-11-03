import numpy as np
from dymoval import rsquared
from dymoval.xcorrelation import correlation_score


def test_zero_residuals_rsquared_and_whiteness():
    # create deterministic example where y == yhat
    N = 100
    t = np.arange(N)
    x = np.sin(2 * np.pi * 5 * t / 100.0)
    y = x.copy()
    yhat = y.copy()

    r2 = rsquared(x.reshape(-1, 1), yhat.reshape(-1, 1))[0]
    assert r2 == 100.0

    # Ordinary correlation on a zero signal is undefined.
    w, W = correlation_score(np.zeros((N, 1)))
    assert np.isnan(w)
