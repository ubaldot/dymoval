import numpy as np
from dymoval import rsquared, XCorrelation

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare MATLAB CSV outputs with dymoval."
    )
    parser.add_argument("x", help="Path to x.csv")
    parser.add_argument("y", help="Path to y.csv")
    parser.add_argument("yhat", help="Path to yhat.csv")
    args = parser.parse_args()

    x = np.loadtxt(args.x, delimiter=",")
    y = np.loadtxt(args.y, delimiter=",")
    yhat = np.loadtxt(args.yhat, delimiter=",")

    # ensure shapes
    x = x.reshape(-1)
    y = y.reshape(-1)
    yhat = yhat.reshape(-1)

    r2 = rsquared(x.reshape(-1, 1), yhat.reshape(-1, 1))[0]
    print("R2 (percent) =", r2)

    Rxy = XCorrelation("test", x, y)
    vals = Rxy.R[0, 0].values
    lags = Rxy.R[0, 0].lags
    idx = int(np.argmax(np.abs(vals)))
    peak = vals[idx]
    lag = lags[idx]
    print("peak =", peak, "at lag =", lag)
