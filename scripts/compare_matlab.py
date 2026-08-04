import numpy as np
import dymoval as dmv
from dymoval import rsquared, XCorrelation

x = np.loadtxt('x.csv', delimiter=',')
y = np.loadtxt('y.csv', delimiter=',')
yhat = np.loadtxt('yhat.csv', delimiter=',')

# ensure shapes
x = x.reshape(-1)
y = y.reshape(-1)
yhat = yhat.reshape(-1)

r2 = rsquared(x.reshape(-1,1), yhat.reshape(-1,1))[0]
print('R2 (percent) =', r2)

Rxy = XCorrelation('test', x, y)
vals = Rxy.R[0,0].values
lags = Rxy.R[0,0].lags
idx = int(np.argmax(np.abs(vals)))
peak = vals[idx]
lag = lags[idx]
print('peak =', peak, 'at lag =', lag)
