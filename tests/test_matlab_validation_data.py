from pathlib import Path

import numpy as np
from dymoval import rsquared, XCorrelation


def test_matlab_validation_data_rsquared_and_xcorr():
    fixtures = Path(__file__).parent / "fixtures"
    x = np.loadtxt(fixtures / "x.csv", delimiter=",")
    y = np.loadtxt(fixtures / "y.csv", delimiter=",")
    yhat = np.loadtxt(fixtures / "yhat.csv", delimiter=",")

    # R^2 (dymoval returns percent)
    r2 = rsquared(x.reshape(-1,1), yhat.reshape(-1,1))[0]
    # Expected value observed from running the script
    expected_r2 = 98.9999944401071
    assert abs(r2 - expected_r2) < 1e-9

    Rxy = XCorrelation('test', x, y)
    vals = Rxy.R[0,0].values
    lags = Rxy.R[0,0].lags
    idx = int(np.argmax(np.abs(vals)))
    peak = vals[idx]
    lag = lags[idx]

    expected_peak = 0.9975435717363579
    expected_lag = 0

    assert np.isclose(peak, expected_peak, atol=1e-12)
    assert lag == expected_lag
