import numpy as np
import dymoval as dmv
from dymoval import rsquared


def test_rsquared_simple():
    rng = np.random.default_rng(0)
    N = 1000
    t = np.arange(N)
    x = np.sin(2 * np.pi * 5 * t / 100.0)
    # y is noisy measurement
    y = x + 0.01 * rng.standard_normal(N)
    # yhat is a slightly scaled prediction
    yhat = 0.98 * x

    # dymoval returns percentage
    r2_pct = rsquared(x.reshape(-1, 1), yhat.reshape(-1, 1))[0]

    sse = np.sum((x - yhat) ** 2)
    sst = np.sum((x - np.mean(x)) ** 2)
    expected_pct = (1.0 - sse / sst) * 100.0

    assert np.isclose(r2_pct, expected_pct, rtol=1e-12)


def test_rsquared_constant_raises():
    x = np.ones(100)
    y = np.ones(100)
    try:
        _ = rsquared(x.reshape(-1, 1), y.reshape(-1, 1))
        raised = False
    except ValueError:
        raised = True
    assert raised
