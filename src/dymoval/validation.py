"""Module containing everything related to validation."""

# The following is needed when there are methods that return instance of the
# class itself.
# TODO If you remove python 3.10 remove typing_extensions as Self in typing is
# part of the standard python package starting from 3.11
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self  # noqa

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import matplotlib
import numpy as np
import pandas as pd
import scipy.signal as signal
from matplotlib import pyplot as plt

from .config import (
    COLORMAP,
    SIGNAL_KEYS,
    XCORR_STATISTIC_TYPE,
    R2_Statistic_type,
    XCorr_Statistic_type,
    is_latex_installed,
)
from .dataset import Dataset, Signal, validate_signals
from .utils import (
    difference_lists_of_str,
    factorize,
    is_interactive_shell,
    obj2list,
)

__all__ = [
    "XCorrelation",
    "rsquared",
    "whiteness_level",
    "compute_statistic",
    "validate_models",
    "ValidationSession",
]


# Util for defining XCorrelation elements.
# XCorrelation dataclass is a matrix of Rxy elements.
class _rxy(NamedTuple):
    values: np.ndarray
    lags: np.ndarray


@dataclass
class XCorrelation:
    # You have to manually write the type in the docstrings
    # and you have to exclude them in the :automodule:
    r"""Cross-correlation of two signals `X` and `Y`.

    The signals can be MIMO and shall have dimension
    :math:`N\times p` and :math:`N\times q`, respectively.

    If `X = Y` then it return the normalized auto-correlation of `X`.
    If additional arguments are passed, then either
    `X_Bandwidth`, `Y_Bandwidth` and `sampling_period` are passed or
    none of them.

    The cross-correlation functions are stored in the attribute `R` which is
    an array where the `(i, j)`-th element
    is the cross-correlation function between the `i`-th signal of
    `X` and the `j`-th signal of `Y`. The cross-correlation functions
    are `NamedTuple` s with attributes
    `values` and `lags`.


    Parameters
    ----------
    name:
        The XCorrelation object name.
    X:
        MIMO signal realizations expressed as :math:`N\times p` array
        of `N` observations of `p` signals.
    Y:
        MIMO signal realizations expressed as :math:`N\times q` array
        of `N` observations of `q` signals.
    nlags:
        :math:`p \times q` array where the `(i, j)`-th element represents
        the number of lags
        of the cross-correlation function associated to the `i`-th signal of
        `X` with the `j`-th signal of `Y`.
    X_bandwidths:
        1-D array representing the bandwidths of each signal in  `X`.
        `X_bandwidths[i]` corresponds to the bandwidth of signal `X[i]`.
    Y_bandwidths:
        1-D array representing the bandwidths of each signal in `Y`.
        `Y_bandwidths[i]` corresponds to the bandwidth of signal `Y[i]`.
    sampling_period:
        Sampling period of the signals `X` and `Y`.

    Example
    -------
    >>> import dymoval as dmv
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> X = rng.uniform(low=-1, high=1, size=(10,3))
    >>> Y = rng.normal(size=(10,4))
    >>> lags = np.array([[10,8,20,12],[8 ,6 ,2 ,10],[20, 12, 8, 8]])
    >>> Rxy = dmv.XCorrelation("foo", X, Y, nlags=lags)
    # Cross-correlation between the first element of X (1D time-series) and
    # the third element of Y (1D time-series).
    >>> Rxy.R[0,2].lags
        array([-9, -8, -7, -6, -5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5,  6,
        7,
        8,  9])
    >>> Rxy.R[0,2].values
        array([-0.06377225, -0.0083634 ,  0.14850791,  0.06379516,
        -0.16405862,
               -0.24074438,  0.14147755,  0.06538316, -0.26679362,
               0.14813509,
                0.64887265,  0.22247482, -0.4785613 , -0.30908332,
                0.12834458,
               -0.08259541, -0.27451256,  0.25320947,  0.06828447])
    """

    def __init__(
        self,
        name: str,
        X: np.ndarray,
        Y: np.ndarray,
        nlags: np.ndarray | None = None,
        X_bandwidths: np.ndarray | float | None = None,
        Y_bandwidths: np.ndarray | float | None = None,
        sampling_period: float | None = None,
    ) -> None:
        # =========================================
        # Attributes
        # =========================================
        self.name: str = name
        """XCorrelation object name."""

        # R is a matrix where each element is rij(\tau).
        # The range of \tau may change as it depends on the sampling_period
        # and the bandwidth of a given signal.
        self._R = self._init_R(
            X=X,
            Y=Y,
            nlags=nlags,
            X_bandwidths=X_bandwidths,
            Y_bandwidths=Y_bandwidths,
            sampling_period=sampling_period,
        )
        """XCorrelation tensor."""

        if np.array_equal(X, Y):
            self._kind = "auto-correlation"
        else:
            self._kind = "cross-correlation"

    def _init_R(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        nlags: np.ndarray | None = None,
        X_bandwidths: np.ndarray | float | None = None,
        Y_bandwidths: np.ndarray | float | None = None,
        sampling_period: float | None = None,
    ) -> np.ndarray:
        # The initialization consists in computing the following
        #  1. full x-correation
        #  2. downsample
        #  3. trim based on the lags needed (you don't need N observations
        #     lags)

        # Downsampling happens only if user pass all the bandwidths and the
        # sampling_period. Trim happens anyway.
        passed_all_arguments = (
            X_bandwidths is not None
            and Y_bandwidths is not None
            and sampling_period is not None
        )

        if X.ndim == 1:
            X = X.reshape(len(X), 1)
        if Y.ndim == 1:
            Y = Y.reshape(len(Y), 1)
        p = X.shape[1]
        q = Y.shape[1]

        # Some input check
        if isinstance(X_bandwidths, (int, float)):
            X_bandwidths = np.array([X_bandwidths])
        if isinstance(Y_bandwidths, (float, int)):
            Y_bandwidths = np.array([Y_bandwidths])

        if isinstance(X_bandwidths, np.ndarray):
            if X_bandwidths.size != p:
                raise IndexError(
                    "The number of elements of 'X_bandwidths' must be "
                    f"equal to {p}"
                )

        if isinstance(Y_bandwidths, np.ndarray):
            if Y_bandwidths.size != q:
                raise IndexError(
                    "The number of elements of 'Y_bandwidths' must be "
                    f"equal to {q}"
                )
        # nlags
        if nlags is not None:
            if (
                not isinstance(nlags, np.ndarray)
                or nlags.shape[0] < p
                or nlags.shape[1] < q
            ):
                raise IndexError(f"'nlags' shall be a {p}x{q} array.")
            else:
                nlags_from_user = nlags[0:p, 0:q]
        else:
            # Default 20 lags
            nlags_from_user = 10 * np.ones((p, q))

        # Let's preserve some immutability
        R_full = np.empty((p, q), dtype=_rxy)
        R_downsampled = np.empty((p, q), dtype=_rxy)
        R_trimmed = np.empty((p, q), dtype=_rxy)

        for ii in range(p):
            for jj in range(q):
                # Adjust the number of lags
                lags_full = signal.correlation_lags(
                    len(X[:, ii]), len(Y[:, jj])
                )
                nlags_full = lags_full.size

                # Classic correlation definition from Probability.
                # Rxy_values = E[(X-mu_x)^T(Y-mu_y))]/(sigma_x*sigma_y),
                # check normalized cross-correlation for stochastic processes
                # on Wikipedia.
                # Nevertheless, the cross-correlation is in-fact the same as
                # E[].
                # More specifically, the cross-correlation generate a sequence
                # [E[XY(\tau=0))] E[XY(\tau=1))], ...,E[XY(\tau=N))]] and this
                # is
                # the reason why in the computation below we use
                # signal.correlation.
                #
                # Another way of seeing it, is that to secure that the
                # cross-correlation
                # is always between -1 and 1, we "normalize" the observations
                # X and Y
                # Google for "Standard score"
                #
                # At the end, for each pair (ii,jj) you have Rxy_values =
                # r_{x_ii,y_jj}(\tau), therefore
                # for each (ii,jj) we compute a correlation.
                values_full = signal.correlate(
                    (X[:, ii] - np.mean(X[:, ii])) / np.std(X[:, ii]),
                    (Y[:, jj] - np.mean(Y[:, jj])) / np.std(Y[:, jj]),
                ) / min(len(X), len(Y))

                R_full[ii, jj] = _rxy(values_full, lags_full)

                # ------ Downsampling -------------------
                # Close measurements are naturally correlated, and
                # therefore the auto-correlation function (ACF) would have
                # high values around lag = 0. The idea is to downsample the
                # cross-correlation tensor so that we check for similarities
                # for longer delays
                #
                # 1 lag = step * sampling_period
                #
                # Rxy_values -> Rxy_values_downsampled -> Rxy_values_trimmed
                # based on nlags
                #
                # We extraxt X_B3 and Y_B3 to easy debug
                if passed_all_arguments:
                    assert X_bandwidths is not None
                    assert Y_bandwidths is not None
                    assert sampling_period is not None
                    # We take the maximum bandwidth to have the less step
                    # The 2 is because of Nyquist criteria, i.e.1 <= step <=
                    # Fs/(2*B3) but cannot be too long, e.g. you cannot have a
                    # step of 15 if the total number of lags are 10. Min 3
                    # lags.
                    bandwidth_max = max(X_bandwidths[ii], Y_bandwidths[jj])
                    Fs = 1 / sampling_period
                    if Fs < 2 * bandwidth_max:
                        raise ValueError(
                            "Nyquist criteria violated. "
                            f"Sampling frequency is {Fs} whereas "
                            f"some signal bandwidth is {bandwidth_max}"
                        )
                    step = int(Fs // (2.0 * bandwidth_max))
                else:
                    # We downsample with step 1 (= no downsampling). TODO
                    # Could be
                    # refactored
                    step = 1

                # print(f"step = {step}")
                # We won't take less than 3 lags
                # Saturate the steps based on number of observations
                nlags_min = 3
                step = max(1, min(step, nlags_full // nlags_min))
                indices_downsampled = np.where(lags_full % step == 0)[0]

                values_downsampled = R_full[ii, jj].values[indices_downsampled]

                lags_downsampled = lags_full[indices_downsampled] // step

                R_downsampled[ii, jj] = _rxy(
                    values_downsampled, lags_downsampled
                )

                # ----------- Trim ---------------
                # Create the half vectors for lags selection
                n = nlags_from_user[ii, jj] // 2
                nlags_trimmed = int(min(n, lags_downsampled[-1]))

                indices_trimmed = np.where(
                    (-nlags_trimmed <= lags_downsampled)
                    & (lags_downsampled <= nlags_trimmed)
                )[0]
                # Trim based on the number of lags
                values_trimmed = R_downsampled[ii, jj].values[indices_trimmed]
                lags_trimmed = R_downsampled[ii, jj].lags[indices_trimmed]

                R_trimmed[ii, jj] = _rxy(values_trimmed, lags_trimmed)

        return R_trimmed

    def __repr__(self) -> str:
        # Include basic information about the object
        repr_str = (
            f"name: {self.name}\n"
            f"type: {self.kind}\n"
            f"R shape: {self.R.shape}\n"
        )

        return repr_str

    # ========== read-only attributes ====================
    @property
    def R(self) -> np.ndarray:
        r"""Auto- or cross-correlation array.

        It is a :math:`p \times q` array where the :math:`(i, j)`-th
        element represent the auto- or cross-correlation function of the
        :math:`i`-th component of the argument `X` and the
        :math:`j`-th component of the argument `Y`.


        Each element of such an array is a `NamedTuple` object with
        attributes `values` and `lags`.
        """
        return self._R

    @property
    def kind(self) -> str:
        """Kind of the XCorrelation object.

        It can be `"auto-correlation"` or `"cross-correlation"`.
        """
        return self._kind

    def estimate_whiteness(
        self,
        local_statistic: XCorr_Statistic_type = "abs_mean",
        local_weights: (
            np.ndarray | None
        ) = None,  # shall be p*q matrix where each element is a 1-D array.
        global_statistic: XCorr_Statistic_type = "max",
        global_weights: np.ndarray | None = None,  # Shall be a p*q matrix
    ) -> tuple[float, np.ndarray]:
        r"""Return the whiteness estimate based on the selected statistics.


        Parameters
        ----------
        local_statistic:
            Statistic type for each `(i,j)` cross-correlation function of
            :py:attr:`~dymoval.validation.XCorrelation.R` array.

        local_weights:
            Weights associated with the value of each `(i, j)` element of
            :py:attr:`~dymoval.validation.XCorrelation.R`.
            It must have the same shape of
            :py:attr:`~dymoval.validation.XCorrelation.R`.

        global_statistic:
            Statistic used to estimate the whiteness of the flattened
            :math:`p \times q` array after the whiteness of each element of
            :py:attr:`~dymoval.validation.XCorrelation.R` is estimated.

        global_weights:
            Weights associated with each element of the resulting
            :math:`p \times q` array.  It shall be a :math:`p \times q` array.

        Returns
        -------
        whiteness_estimate:
            The overall whiteness estimate.
        whiteness_matrix:
            A :math:`p \times q` array where the `(i, j)`-th
            element is the statistic computed for the `(i, j)`-th
            cross-correlation function of
            :py:attr:`~dymoval.validation.XCorrelation.R`.

        Example
        -------
        >>> #  Assume that RXY is a XCorrelation instance
        >>> local_weights = np.empty(RXY.R.shape, dtype=np.ndarray)
        >>> local_weights[0, 0] = np.ones(11)
        >>> local_weights[0, 1] = np.ones(3)
        >>> local_weights[1, 0] = np.ones(13)
        >>> local_weights[1, 1] = np.ones(6)
        >>> w, W = RXY.estimate_whiteness(local_weights=local_weights)
        """

        # MAIN whiteness level =================================
        R = self.R
        p = R.shape[0]  # Number of rows
        q = R.shape[1]  # Number of columns

        # ---- statistics type is correct ----
        if (
            local_statistic not in XCORR_STATISTIC_TYPE
            or global_statistic not in XCORR_STATISTIC_TYPE
        ):
            raise ValueError(
                f"Statistic type must be in {XCORR_STATISTIC_TYPE}"
            )

        # -------------- Validation of locals and global weights from user
        if local_weights is None:
            W_local = np.empty((p, q), dtype=np.ndarray)
            for ii in range(p):
                for jj in range(q):
                    W_local[ii, jj] = np.ones(len(R[ii, jj].lags))
        else:
            # TODO a bit flaky test because we only check the element in
            # position [0,0]
            # Check that the number of lags and weights are the same and that
            # each element is a np.ndarray
            if local_weights.shape != (p, q) or not isinstance(
                local_weights[0, 0], np.ndarray
            ):
                raise IndexError(
                    "'local_weights' must have the same shape of "
                    "'R' and each element must be a np.ndarray."
                )
            for ii in range(p):
                for jj in range(q):
                    if len(local_weights[ii, jj]) != len(R[ii, jj].lags):
                        raise IndexError(
                            "Number of lags and number of weights "
                            "must be the same.\n"
                            f"In index {ii, jj} you have "
                            f" {len(R[ii, jj].lags)} lags and "
                            f"{len(local_weights[ii, jj])} weights."
                        )
            # if num_weights is equal to num_lags go ahead
            W_local = local_weights

        # fix global weights
        if global_weights is not None and global_weights.shape != (p, q):
            raise IndexError(f"'global_weights' must be a {p}x{q} np.ndarray.")
        else:
            W_global = (
                np.ones(p * q) if global_weights is None else global_weights
            )

        # ------------ compute statistics -------------------------
        # Build the R_matrix by computing the statistic of each scalar
        # cross-correlation (local)
        whiteness_matrix = np.zeros((p, q))
        for ii in range(p):
            for jj in range(q):
                if ii == jj and self.kind == "auto-correlation":
                    # Remove auto-correlation values at lag = 0
                    lag0_idx = np.nonzero(R[ii, jj].lags == 0)[0][0]
                    W = np.delete(W_local[ii, jj], lag0_idx)
                    rij_tau = np.delete(R[ii, jj].values, lag0_idx)
                else:
                    W = W_local[ii, jj]
                    rij_tau = R[ii, jj].values

                whiteness_matrix[ii, jj] = compute_statistic(
                    statistic=local_statistic,
                    weights=W,
                    data=rij_tau,
                )

        # Compute the overall statistic of the resulting matrix
        whiteness_estimate = compute_statistic(
            statistic=global_statistic,
            weights=W_global.flatten(),
            data=whiteness_matrix.flatten(),
        )

        return whiteness_estimate, whiteness_matrix

    def plot(self) -> matplotlib.figure.Figure:
        """Plot the :math:`p \times q` cross-correlation functions contained
        in :py:attr:`~dymoval.validation.XCorrelation.R`."""

        p = self.R.shape[0]
        q = self.R.shape[1]
        fig, ax = plt.subplots(p, q, squeeze=False)
        plt.setp(ax, ylim=(-1.2, 1.2))

        for ii in range(p):
            for jj in range(q):
                if is_latex_installed:
                    title_acorr = rf"$\hat r_{{\epsilon_{ii}\epsilon_{jj}}}$"
                    title_xcorr = rf"$\hat r_{{u_{ii}\epsilon_{jj}}}$"
                else:
                    title_acorr = rf"r_eps{ii}eps{jj}$"
                    title_xcorr = rf"r_u{ii}eps{jj}$"
                title = (
                    title_acorr
                    if self.kind == "auto-correlation"
                    else title_xcorr
                )
                ax[ii, jj].stem(
                    self.R[ii, jj].lags,
                    self.R[ii, jj].values,
                    label=self.name,
                )
                ax[ii, jj].grid(True)
                ax[ii, jj].set_xlabel("Lags")
                ax[ii, jj].set_title(title)
                if self.name != "":
                    ax[ii, jj].legend()
        fig.suptitle(f"{self.kind}")

        if is_interactive_shell():
            fig.show()
        else:
            plt.show()

        return fig


def compute_statistic(
    data: np.ndarray,
    statistic: XCorr_Statistic_type = "mean",
    weights: np.ndarray | None = None,
) -> float:
    r"""Compute the statistic of a sequence of numbers.

    The elements of `data` can be weighted through the `weights` array.

    If `data.shape` dimension is greater than 1 then `data` will be flatten
    to a 1-D array.
    The  return values are normalized such that the function always return
    values between 0 and 1, with the exclusion of the statistic `quadratic`
    that may return values greater than 1.0.

    The statistic `S` is computed as it follows. Let :math:`w_i` is the
    `i`-th element of `weights` and :math:`x_i` is the `i`-th element
    of `data`.

    **mean**
    This is the classic weighted mean value, computed as:

    .. math::
        S = \frac{\sum_{i=1}^N w_ix_i}{\sum_{i=1}^N w_i}

    **abs_mean**
    Mean of absolute values, computed as:

    .. math::
        S = \frac{\sum_{i=1}^N w_i|x_i|}{\sum_{i=1}^N w_i}

    **max**
    Max of absolute values, computed as:

    .. math::
        S = \max_i\{|x_i|\}

    **std**
    Standard deviation, computed as:

    .. math::
        S =\sqrt{\sum_{i=1}^N w_i(x_i - \bar x)^2}

    where :math:`\bar x` is the weighted mean value computed above.

    **quadratic**
    This is a generic quadratic form of the form:

    .. math::
        S = \frac{1}{N \|W\|_{\infty}}x^TWx
        = \frac{\sum_{i=1}^N w_i|x_i|}{N\max_i \{|w_i|\}}


    This is particular useful since many famous statistics, such as Ljung-Box,
    Box-Pierce, Lagrange Multiplier, etc., can be rewritten in the above form
    through an appropriate choice of the weights.

    Parameters
    ----------
    data:
        Array containing values for which the `statistic` shall be computed.
    statistic:
        Kind of statistic to be computed.
    weights:
        An array of weights associated with the values in `data`.
        More precisely, `weights[i]` correspond to `data[i]`.
    """

    if data.ndim > 1:
        raise IndexError("'data' must be a 1-D np.ndarray.")

    if weights is None:
        weights = np.ones(data.size)
    elif np.min(weights) < 0:
        raise ValueError("All weights must be positive.")
    elif weights.ndim > 1:
        raise IndexError("'weights' must be a 1-D np.ndarray.")
    elif data.size != weights.size:
        raise IndexError("'data' and 'weights' must have the same length.")

    if statistic == "quadratic":
        # It holds x'Wx < lambda_max(W) |x|^2 = max(W) given that W is
        # diagonal with all positive entries. Furthermore, it
        # hold np.max(weights) = lambda_max(W).
        # To avoid that the metric depends on the number of lags, we normalize
        # by n.

        quadratic_form = data.T @ np.diag(weights) @ data

        result = quadratic_form / (np.max(np.abs(weights)) * len(data))
        # If the weights are all the same, then the metric reduces to |x|²/n,
        # which is similar to the abs_norm with all the weights equal to 1.

    elif statistic == "abs_mean":
        # This is similar to RMS. By using l1 norm we are more gentle with
        # respect to outliers.
        result = np.sum(weights * np.abs(data)) / np.sum(weights)

    elif statistic == "max":
        # To secure <1 you have to divide by |x|_inf|W|_inf
        result = np.max(np.abs(weights.T * data)) / (
            np.linalg.norm(weights, ord=np.inf)
        )
    elif statistic == "mean":
        # This should always be less than 1
        result = weights.T @ data / np.sum(weights)
    elif statistic == "std":
        # Compute the weighted average
        weighted_avg = weights.T @ data / np.sum(weights)
        # Compute the weighted variance
        weighted_variance = np.average(
            (data - weighted_avg) ** 2, weights=weights
        )
        result = np.sqrt(weighted_variance)  # Standard deviation
    else:
        raise ValueError(
            f"'statistic' must be one of [{XCORR_STATISTIC_TYPE}]"
        )
    return float(result)


def rsquared(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""
    Return the :math:`R^2` value of two signals.

    Signals can be MIMO.

    Parameters
    ----------
    x:
        First input signal. It must have shape :math:`N\times p`, where :math:`N` is the number
        of observation and :math:`p` is the signal dimension.
    y:
        Second input signal. It must have shape :math:`N\times p`, where :math:`N` is the number
        of observation and :math:`p` is the signal dimension.
    """

    if x.shape != y.shape:
        raise IndexError("Arguments must have the same shape.")
    eps = x - y
    # Compute r-square fit (%)
    x_mean = np.mean(x, axis=0)

    # Compute the R² index
    ss_res = np.sum(eps**2, axis=0)
    ss_tot = np.sum((x - x_mean) ** 2, axis=0)
    r2: np.ndarray = np.asarray((1.0 - ss_res / ss_tot) * 100)

    return r2


# TODO: Not happy with this
def whiteness_level(
    data: np.ndarray,
    data_bandwidths: np.ndarray | float | None = None,
    sampling_period: float | None = None,
    nlags: np.ndarray | None = None,
    local_statistic: XCorr_Statistic_type = "abs_mean",
    # shall be a p*q matrix where each element is a
    # 1D-array (like the lags)
    local_weights: np.ndarray | None = None,
    global_statistic: XCorr_Statistic_type = "max",
    global_weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    r"""Estimate the whiteness of the signal `data`.

    If `data` is a multivariate signal of shape :math:`p \times p`, then the
    whiteness is computed in two steps:

    #. The cross-correlation function for each :math:`(i, j)` pair of signal in
       `data` is computed, and their whiteness of is computed and arranged
       in a :math:`p \times p` array.

    #. The resulting :math:`p \times p` array is flattened and the
       overall signal whiteness is estimated.

    It returns the values computed in points 1. and 2.

    The whiteness is computed through
    :py:meth:`~dymoval.validation.compute_statistic`.

    Parameters
    ----------
    data:
        Signal samples.
    data_bandwidths:
        Signal bandwidth. If the signal is multivariate, then this specify the
        bandwidth of each of its component.
    sampling_period:
        Signal sampling period.
    nlags:
        Number of lags to be considered for the whiteness estimate
        computation. If the signal is multivariate with `p` components, then
        this must be a :math:`p\times p` array.
    local_statistic:
        Statistic to be used for estimate the whiteness of each `(i, j)`
        cross-correlation function.
    local_weights:
        Weights to be used for the whiteness estimation of each `(i, j)`
        cross-correlation function.
        It shall have the same size of
        :py:attr:`~dymoval.validation.XCorrelation.R`.

    global_statistic:
        Statistic to be used for estimate the whiteness of the resulting
        :math:`p \times q` array.
    global_weights:
        Weight of each element of the resulting :math:`p \times q` array
        for estimating the overall signal whiteness.
    """

    # Convert signals into XCorrelation tensors and compute the
    # whiteness_level

    Rxx = XCorrelation(
        "",
        X=data,
        Y=data,
        nlags=nlags,
        X_bandwidths=data_bandwidths,
        Y_bandwidths=data_bandwidths,
        sampling_period=sampling_period,
    )

    whiteness_estimate, whiteness_matrix = Rxx.estimate_whiteness(
        local_statistic=local_statistic,
        local_weights=local_weights,
        global_statistic=global_statistic,
        global_weights=global_weights,
    )

    del Rxx

    return whiteness_estimate, whiteness_matrix


@dataclass
class ValidationSession:
    # TODO: Save validation session.
    r"""The *ValidationSession* class is used to validate models against a
    given dataset.

    A *ValidationSession* object is instantiated from a :ref:`Dataset` object.
    A validation session *name* shall be also provided.

    Multiple simulation results can be appended to the same
    *ValidationSession* instance,
    but for each ValidationSession instance only a :ref:`Dataset` object is
    considered.

    Parameters
    ----------
    name:
        The `ValidationSession` object name.
    validation_dataset:
        The :py:class:`~dymoval.dataset.Dataset` object to be used for
        validation.
    U_bandwidths:
        1-D array representing the bandwidths of each signal in the input U.
        `U_bandwidths[i]` corresponds to the bandwidth of signal `U[i]`.
    Y_bandwidths:
        1-D array representing the bandwidths of each signal in the output Y.
        `Y_bandwidths[i]` corresponds to the bandwidth of signal `Y[i]`.
    validation_thresolds:
        Threshold used for validation. The `dict` keys shall be:

        - `"Ruu_whiteness""`

        - `"r2"`

        - `"Ree_whiteness""`

        - `"Rue_whiteness""`

    ignore_input:
        If `True` input auto-correlation is not considered in the
        validation.
    r2_statistic:
        Statistic to be used for computing the global :math:`R^2` in case of
        multiple output signals.
    Ruu_nlags:
        Number of lags for the input auto-correlation array `Ruu`.
    Ruu_local_statistic_type:
        Statistic used for estimating the whiteness of each element of the
        :py:class:`~dymoval.validation,XCorrelation` object associated to the
        input signal.
    Ruu_global_statistic_type:
        Statistic used for estimating the overall whiteness of the resulting
        :math:`p\times p` matrix after the whiteness of each element of `Ruu`
        has been computed.
    Ruu_local_weights:
        Weights associated to each element of `Ruu`. It must be a
        :math:`p\times p` array where each element is a 1-D array.
    Ruu_local_weights:
        Weights associated to the resulting matrix after the local statistics
        for each element or `Ruu` have been computed. It must be a
        :math:`p\times p` array.
    Ree_nlags:
        Number of lags for the residuals auto-correlation array `Ree`.
    Ree_local_statistic_type:
        Statistic used for estimating the whiteness of each element of the
        :py:class:`~dymoval.validation,XCorrelation` object associated to the
        residuals auto-correlation.
    Ree_global_statistic_type:
        Statistic used for estimating the overall whiteness of the resulting
        :math:`p\times p` matrix after the whiteness of each element of `Ree`
        has been computed.
    Ree_local_weights:
        Weights associated to each element of `Ree`. It must be a
        :math:`q\times q` array where each element is a 1-D array.
    Ree_local_weights:
        Weights associated to the resulting matrix after the local statistics
        for each element or `Ree` have been computed. It must be a
        :math:`q\times q` array.
    Rue_nlags:
        Number of lags for the input-residuals cross-correlation array `Rue`.
    Rue_local_statistic_type:
        Statistic used for estimating the whiteness of each element of the
        :py:class:`~dymoval.validation,XCorrelation` object associated to the
        input-residuals cross-correlation.
    Rue_global_statistic_type:
        Statistic used for estimating the overall whiteness of the resulting
        :math:`p\times q` matrix after the whiteness of each element of `Rue`
        has been computed.
    Rue_local_weights:
        Weights associated to each element of `Rue`. It must be a
        :math:`p\times q` array where each element is a 1-D array.
    Rue_local_weights:
        Weights associated to the resulting matrix after the local statistics
        for each element or `Rue` have been computed. It must be a
        :math:`p\times q` array.
    """

    def __init__(
        self,
        name: str,
        validation_dataset: Dataset,
        U_bandwidths: np.ndarray | float | None = None,
        Y_bandwidths: np.ndarray | float | None = None,
        # Model validation
        validation_thresholds: dict[str, float] | None = None,
        ignore_input: bool = False,
        # r2
        r2_statistic: Literal["min", "mean"] = "min",
        # The following are input to XCorrelation.estimate_whiteness()
        # method.
        # input auto-correlation
        Ruu_nlags: np.ndarray | None = None,
        Ruu_local_statistic_type: XCorr_Statistic_type = "abs_mean",
        Ruu_global_statistic_type: XCorr_Statistic_type = "max",
        Ruu_local_weights: np.ndarray | None = None,
        Ruu_global_weights: np.ndarray | None = None,
        # residuals auto-correlation
        Ree_nlags: np.ndarray | None = None,
        Ree_local_statistic_type: XCorr_Statistic_type = "abs_mean",
        Ree_global_statistic_type: XCorr_Statistic_type = "max",
        Ree_local_weights: np.ndarray | None = None,
        Ree_global_weights: np.ndarray | None = None,
        # input-residuals cross-Correlation
        Rue_nlags: np.ndarray | None = None,
        Rue_local_statistic_type: XCorr_Statistic_type = "abs_mean",
        Rue_global_statistic_type: XCorr_Statistic_type = "max",
        Rue_local_weights: np.ndarray | None = None,
        Rue_global_weights: np.ndarray | None = None,
    ) -> None:
        # Once you created a ValidationSession you should not change the
        # validation dataset.
        # Create another ValidationSession with another validation dataset
        # By using the constructors, you should have no types problems because
        # the check is done there.

        # =============================================
        # Class attributes
        # ============================================

        self._Dataset: Dataset = validation_dataset

        # Number of inputs
        self._p = len(
            self._Dataset.dataset["INPUT"].columns.get_level_values("names")
        )

        # Number of outputs
        self._q = len(
            self._Dataset.dataset["OUTPUT"].columns.get_level_values("names")
        )

        # Simulation based
        self.name: str = name  # The validation session name.
        """ValidationSession object name."""

        self._default_nlags = 41

        self._simulations_values: pd.DataFrame = pd.DataFrame(
            index=validation_dataset.dataset.index, columns=[[], [], []]
        )
        """The appended simulation results.
        This attribute is automatically set through
        :py:meth:`~dymoval.validation.ValidationSession.append_simulation`
        and it should be considered as a *read-only* attribute."""
        # Format: 'name_sim': r2
        self._r2_list: dict[str, np.ndarray] = {}
        self._r2: dict[str, float] = {}
        self._r2_statistic = r2_statistic

        # Input: Ruu
        self._Ruu: XCorrelation
        """The auto-correlation tensors.
        This attribute is automatically set
        and it should be considered as a *read-only* attribute."""

        # Format: 'name_sim': Ree
        self._Ree: dict[str, XCorrelation] = {}
        self._Ree_whiteness: dict[str, float] = {}
        self._Ree_whiteness_matrix: dict[str, np.ndarray] = {}
        """The auto-correlation tensors.
        This attribute is automatically set
        and it should be considered as a *read-only* attribute."""

        # Format: 'name_sim': Rue
        self._Rue: dict[str, XCorrelation] = {}
        self._Rue_whiteness: dict[str, float] = {}
        self._Rue_whiteness_matrix: dict[str, np.ndarray] = {}
        """The cross-correlation tensors.
        This attribute is automatically set
        and it should be considered as a *read-only* attribute."""

        # ------------------ Input --------------------
        self._U_bandwidths = U_bandwidths

        # Input nlags
        # Default 41 lags (20 negative and 20 positive)
        self._Ruu_nlags = np.full(
            (self._p, self._p), fill_value=self._default_nlags
        )
        if Ruu_nlags is not None:
            if Ruu_nlags.shape[0] < self._p or Ruu_nlags.shape[1] < self._p:
                raise IndexError(
                    f"'Ruu_nlags' shall be a {self._p}x{self._p} " "array."
                )
            else:
                self._Ruu_nlags = Ruu_nlags[0 : self._p, 0 : self._p]
        elif Ruu_local_weights is not None:
            # Iterate through the input array and count the number of lags
            for ii in range(self._p):
                for jj in range(self._p):
                    self._Ruu_nlags[ii, jj] = len(Ruu_local_weights[ii, jj])

        self._Ruu_local_statistic_type = Ruu_local_statistic_type
        self._Ruu_global_statistic_type = Ruu_global_statistic_type
        self._Ruu_local_weights = Ruu_local_weights
        self._Ruu_global_weights = Ruu_global_weights

        Ruu = XCorrelation(
            "Ruu",
            X=self._Dataset.dataset["INPUT"].to_numpy(),
            Y=self._Dataset.dataset["INPUT"].to_numpy(),
            nlags=self._Ruu_nlags,
            X_bandwidths=self._U_bandwidths,
            Y_bandwidths=self._U_bandwidths,
            sampling_period=self._Dataset.dataset.index[1]
            - self._Dataset.dataset.index[0],
        )

        self._Ruu_tensor = Ruu

        self._Ruu_whiteness, self._Ruu_whiteness_matrix = (
            Ruu.estimate_whiteness(
                local_statistic=self._Ruu_local_statistic_type,
                local_weights=self._Ruu_local_weights,
                global_statistic=self._Ruu_global_statistic_type,
                global_weights=self._Ruu_global_weights,
            )
        )

        # ------------ Residuals -----------------------------
        self._Y_bandwidths = Y_bandwidths

        # Residuals auto-correlation
        self._Ree_tensor: dict[str, XCorrelation] = {}

        # nlags
        self._Ree_nlags = np.full(
            (self._q, self._q), fill_value=self._default_nlags
        )
        if Ree_nlags is not None:
            if Ree_nlags.shape[0] < self._q or Ree_nlags.shape[1] < self._q:
                raise IndexError(
                    f"'Ree_nlags' shall be a {self._q}x{self._q} " " array."
                )
            else:
                self._Ree_nlags = Ree_nlags[0 : self._q, 0 : self._q]
        elif Ree_local_weights is not None:
            # Iterate through the input array and count the number of lags
            for ii in range(self._q):
                for jj in range(self._q):
                    self._Ree_nlags[ii, jj] = len(Ree_local_weights[ii, jj])

        self._Ree_local_statistic_type = Ree_local_statistic_type
        self._Ree_global_statistic_type = Ree_global_statistic_type
        self._Ree_local_weights = Ree_local_weights
        self._Ree_global_weights = Ree_global_weights

        # Input-Residuals cross-correlation
        self._Rue_tensor: dict[str, XCorrelation] = {}

        self._Rue_nlags = np.full(
            (self._p, self._q), fill_value=self._default_nlags
        )
        if Rue_nlags is not None:
            if Rue_nlags.shape[0] < self._p or Rue_nlags.shape[1] < self._q:
                raise IndexError(
                    f"'Rue_nlags' shall be a {self._p}x{self._q} " "array."
                )
            else:
                self._Rue_nlags = Rue_nlags[0 : self._p, 0 : self._q]
        elif Rue_local_weights is not None:
            # Iterate through the input array and count the number of lags
            for ii in range(self._p):
                for jj in range(self._q):
                    self._Rue_nlags[ii, jj] = len(Rue_local_weights[ii, jj])

        self._Rue_local_statistic_type = Rue_local_statistic_type
        self._Rue_global_statistic_type = Rue_global_statistic_type
        self._Rue_local_weights = Rue_local_weights
        self._Rue_global_weights = Rue_global_weights

        # Initialize validation results DataFrame.
        idx = [
            f"Input whiteness "
            f"({self._Ruu_local_statistic_type}-"
            f"{self._Ruu_global_statistic_type})",
            "R-Squared (%)",
            f"Residuals whiteness "
            f"({self._Ree_local_statistic_type}-"
            f"{self._Ree_global_statistic_type})",
            f"Input-Res whiteness "
            f"({self._Rue_local_statistic_type}-"
            f"{self._Rue_global_statistic_type})",
        ]
        self._validation_statistics: pd.DataFrame = pd.DataFrame(
            index=idx, columns=[]
        )

        # =========== Model validation =============================
        self._validation_thresholds = (
            self._get_validation_thresholds_default(
                ignore_input=ignore_input,
            )
            if validation_thresholds is None
            else validation_thresholds
        )

        self._ignore_input = ignore_input

        # Initialize PASS/FAIL
        self._outcome: dict[str, str] = {}
        """The validation results.
        This attribute is automatically set
        and it should be considered as a *read-only* attribute."""

    def __repr__(self) -> str:
        # Save existing settings
        # np_options = np.get_printoptions()
        # pd_options = pd.options.display.float_format

        # np.set_printoptions(precision=NUM_DECIMALS, suppress=True)
        # pd.options.display.float_format = lambda x: f"{x:.{NUM_DECIMALS}f}"
        outcomes_head = "         "
        outcomes_body = "Outcome: "
        for k, v in self._outcome.items():
            delta = len(k) - len(v)
            if delta >= 0:
                outcomes_head += f"{k}  "
                outcomes_body += f"{self._outcome[k]}" + " " * (delta + 2)
            else:
                outcomes_head += f"{k}" + " " * (delta + 2)
                outcomes_body += f"{self._outcome[k]}"

        # u acorr weights
        if self._Ruu_local_weights is None:
            Ruu_local_weights_str = (
                f"local weights: {self._Ruu_local_weights}\n"
            )
        else:
            Ruu_local_weights_str = (
                "local weights: Yes (see self._Ruu_local_weights)\n"
            )

        if self._Ruu_global_weights is None:
            Ruu_global_weights_str = (
                f"global weights: {self._Ruu_global_weights}\n"
            )
        else:
            Ruu_global_weights_str = (
                "global weights: Yes (see self._Ruu_global_weights)\n"
            )

        # u_nlags
        if np.all(self._Ruu_nlags.flatten() == self._Ruu_nlags.flatten()[0]):
            u_nlags_str = f"num lags: {self._Ruu_nlags[0, 0]}\n"
        else:
            u_nlags_str = f"num lags: \n{self._Ruu_nlags}\n"

        # Ignore input
        if self._ignore_input:
            inputs_acorr_str = f"Input ignored: {self._ignore_input}\n"
            Ruu_whiteness = ""
            validation_results = self._validation_statistics.iloc[2:, :]
        else:
            inputs_acorr_str = (
                f"Inputs auto-correlation\n"
                f"Statistic: "
                f"{self._Ruu_local_statistic_type}-"
                f"{self._Ruu_global_statistic_type}\n"
                + Ruu_local_weights_str
                + Ruu_global_weights_str
                + u_nlags_str
            )
            Ruu_whiteness = f"{self._validation_statistics.index[0]}: "
            "{self._validation_thresholds['Ruu_whiteness']} \n"
            validation_results = self._validation_statistics

        # eps_nlags
        if np.all(self._Ree_nlags.flatten() == self._Ree_nlags.flatten()[0]):
            eps_nlags_str = f"num lags: {self._Ree_nlags[0, 0]}\n"
        else:
            eps_nlags_str = f"num lags: \n{self._Ree_nlags}\n"

        # eps acorr weights
        if self._Ree_local_weights is None:
            Ree_local_weights_str = (
                f"local weights: {self._Ree_local_weights}\n"
            )
        else:
            Ree_local_weights_str = (
                "local weights: Yes (see self._Ree_local_weights)\n"
            )

        if self._Ree_global_weights is None:
            Ree_global_weights_str = (
                f"global weights: {self._Ree_global_weights}\n"
            )
        else:
            Ree_global_weights_str = (
                "global weights: Yes (see " "self._Ree_global_weights)\n"
            )

        # ueps_nlags
        if np.all(self._Rue_nlags.flatten() == self._Rue_nlags.flatten()[0]):
            ueps_nlags_str = f"num lags: {self._Rue_nlags[0, 0]}\n"
        else:
            ueps_nlags_str = f"num lags: \n{self._Rue_nlags}\n"

        # ueps xcorr weights
        if self._Rue_local_weights is None:
            Rue_local_weights_str = (
                f"local weights: {self._Rue_local_weights}\n"
            )
        else:
            Rue_local_weights_str = (
                "local weights: Yes (see " "self._Rue_local_weights)\n"
            )

        if self._Rue_global_weights is None:
            Rue_global_weights_str = (
                f"global weights: {self._Rue_global_weights}\n"
            )
        else:
            Rue_global_weights_str = (
                "global weights: Yes (see " "self._Rue_global_weights)\n"
            )

        repr_str = (
            f"Validation session name: {self.name}\n\n"
            f"Validation setup:\n----------------\n"
            + inputs_acorr_str
            + "\n"
            + f"Residuals auto-correlation:\n"
            f"Statistic: "
            f"{self._Ree_local_statistic_type}-"
            f"{self._Ree_global_statistic_type}\n"
            + Ree_local_weights_str
            + Ree_global_weights_str
            + eps_nlags_str
            + "\n"
            +
            #
            f"Input-residuals cross-correlation:\n"
            f"Statistic: "
            f"{self._Rue_local_statistic_type}-"
            f"{self._Rue_global_statistic_type}\n"
            + Rue_local_weights_str
            + Rue_global_weights_str
            + ueps_nlags_str
            + "\n"
            +
            #
            f"Validation results:\n-------------------\n"
            f"Thresholds: \n"
            f"{Ruu_whiteness}"
            f"{self._validation_thresholds['Ruu_whiteness']:.4f} \n"
            f"{self._validation_statistics.index[1]}: "
            f"{self._validation_thresholds['r2']:.4f} \n"
            f"{self._validation_statistics.index[2]}: "
            f"{self._validation_thresholds['Ree_whiteness']:.4f} \n"
            f"{self._validation_statistics.index[3]}: "
            f"{self._validation_thresholds['Rue_whiteness']:.4f} \n\n"
            "Actuals:\n"
            f"{validation_results}\n\n"
            f"{outcomes_head}\n"
            f"{outcomes_body}\n"
        )

        # try:
        #     if self.validation_results is not None:
        #         repr_str = repr(self.validation_results)
        #     if self.simulations_results is not None:
        #         repr_str = repr(self.simulations_values)
        # finally:
        #     np.set_printoptions(**np_options)
        #     pd.reset_option("display.float_format")

        return repr_str

    # ========== read-only attributes ====================

    @property
    def dataset(self) -> Dataset:
        """The reference :ref:`Dataset` object."""
        return self._Dataset

    @property
    def simulations_values(self) -> pd.DataFrame:
        """Simulated out values."""
        return self._simulations_values

    @property
    def simulations_names(self) -> list[str]:
        """Names of the stored simulations."""
        return list(self._simulations_values.columns.levels[0])

    @property
    def outcome(self) -> dict[str, str]:
        """Validation outcome.

        For each simulation return validation outcome.
        """
        return self._outcome

    @property
    def Ree(self) -> dict[str, XCorrelation]:
        """Residuals auto-correlation arrays."""
        return self._Ree

    @property
    def Ruu(self) -> XCorrelation:
        """Input auto-correlation arrays."""
        return self._Ruu

    @property
    def Rue(self) -> dict[str, XCorrelation]:
        """Input-residuals cross-correlation arrays."""
        return self._Rue

    @property
    def validation_thresholds(self) -> dict[str, float]:
        """Input-residuals cross-correlation arrays."""
        return self._validation_thresholds

    @validation_thresholds.setter
    def validation_thresholds(self, val: dict[str, float]) -> None:
        """Input-residuals cross-correlation arrays."""
        allowed_keys = self._get_validation_thresholds_default(
            ignore_input=False
        ).keys()
        for k, v in val.items():
            if k not in allowed_keys:
                raise KeyError(f"Keys must be {allowed_keys}.")
            if v < 0.0:
                raise ValueError("Thresholds must be positive.")

        self._validation_thresholds = val

        for sim_name in self.simulations_names:
            self._append_validation_statistics(sim_name=sim_name)

    @property
    def validation_statistics(self) -> Any:
        """Return the computed statistics for each simulation."""

        return self._validation_statistics

    def _get_validation_thresholds_default(
        self, ignore_input: bool
    ) -> dict[str, float]:
        validation_thresholds_default = {
            "Ruu_whiteness": 0.6,
            "r2": 35,
            "Ree_whiteness": 0.5,
            "Rue_whiteness": 0.5,
        }

        if ignore_input is True:
            del validation_thresholds_default["Ruu_whiteness"]
        return validation_thresholds_default

    def _compute_r2_statistic(
        self, r2_list: np.ndarray, statistic: R2_Statistic_type = "min"
    ) -> float:
        result: float = 0.0
        if statistic == "mean":
            result = np.mean(r2_list)
        elif statistic == "min":
            result = np.min(r2_list)
        else:
            raise ValueError("'r2_statistic' must be 'mean' or 'min'")

        return result

    def _append_validation_statistics(
        self,
        sim_name: str,
    ) -> None:
        # Extact dataset output values
        df_val = self._Dataset.dataset
        y_values = df_val["OUTPUT"].to_numpy()
        u_values = df_val["INPUT"].to_numpy()

        # Simulation results
        y_sim_values = self._simulations_values[sim_name].to_numpy()

        # Residuals
        eps = y_values - y_sim_values

        if np.allclose(eps, 0.0):
            raise ValueError(
                "Simulation outputs are identical to measured outputs. "
                "Are you cheating?"
            )

        # r2 value and r2 statistics
        r2 = rsquared(y_values, y_sim_values)
        self._r2_list[sim_name] = r2
        self._r2[sim_name] = self._compute_r2_statistic(r2, self._r2_statistic)

        # Residuals auto-correlation
        Ree = XCorrelation(
            "Ree",
            eps,
            eps,
            X_bandwidths=self._Y_bandwidths,
            Y_bandwidths=self._Y_bandwidths,
            nlags=self._Ree_nlags,
            sampling_period=self._Dataset.dataset.index[1]
            - self._Dataset.dataset.index[0],
        )

        self._Ree_tensor[sim_name] = Ree

        (
            self._Ree_whiteness[sim_name],
            self._Ree_whiteness_matrix[sim_name],
        ) = Ree.estimate_whiteness(
            local_statistic=self._Ree_local_statistic_type,
            local_weights=self._Ree_local_weights,
            global_statistic=self._Ree_global_statistic_type,
            global_weights=self._Ree_global_weights,
        )

        # Input-residals cross-correlation
        Rue = XCorrelation(
            "Rue",
            u_values,
            eps,
            X_bandwidths=self._Y_bandwidths,
            Y_bandwidths=self._Y_bandwidths,
            nlags=self._Rue_nlags,
            sampling_period=self._Dataset.dataset.index[1]
            - self._Dataset.dataset.index[0],
        )

        self._Rue_tensor[sim_name] = Rue

        (
            self._Rue_whiteness[sim_name],
            self._Rue_whiteness_matrix[sim_name],
        ) = Rue.estimate_whiteness(
            local_statistic=self._Rue_local_statistic_type,
            local_weights=self._Rue_local_weights,
            global_statistic=self._Rue_global_statistic_type,
            global_weights=self._Rue_global_weights,
        )

        # Append numerical values:
        self._validation_statistics[sim_name] = np.array(
            [
                self._Ruu_whiteness,
                self._r2[sim_name],
                self._Ree_whiteness[sim_name],
                self._Rue_whiteness[sim_name],
            ]
        )

        # Compute PASS/FAIL outcome
        local_outcome = []

        # Use shorter names for the dataframe, e.g. Ruu_whiteness instead of
        # 'Ruu whiteness (abs-mean)'
        df_new_indices = deepcopy(self._validation_statistics)
        df_new_indices.index = self._validation_thresholds.keys()
        validation_dict = df_new_indices[sim_name].to_dict()

        # Start the comparison
        for k in self._validation_thresholds.keys():
            if k != "r2":
                local_outcome.append(
                    validation_dict[k] < self._validation_thresholds[k]
                )
            else:
                local_outcome.append(
                    validation_dict[k] > self._validation_thresholds[k]
                )
        if all(local_outcome):
            self._outcome[sim_name] = "PASS"
        else:
            self._outcome[sim_name] = "FAIL"

    def _sim_list_validate(self) -> None:
        if not self.simulations_names:
            raise KeyError(
                "The simulations list looks empty. "
                "Check the available simulation names with "
                "'simulations_names()'"
            )

    def _simulation_validation(
        self, sim_name: str, y_names: Sequence[str], y_data: np.ndarray
    ) -> None:
        if len(y_names) != len(set(y_names)):
            raise ValueError("Signals name must be unique.")
        if (
            not self._simulations_values.empty
            and sim_name in self.simulations_names
        ):
            raise ValueError(
                f"Simulation name '{sim_name}' already exists. \n"
                "HINT: check the loaded simulations names with"
                "'simulations_names' method."
            )
        if len(set(y_names)) != len(
            set(self._Dataset.dataset["OUTPUT"].columns)
        ):
            raise IndexError(
                "The number of outputs of your simulation must be equal "
                "to the number of outputs in the dataset AND "
                "the name of each simulation output shall be unique."
            )
        if not isinstance(y_data, np.ndarray):
            raise ValueError(
                "The type the input signal values must be a " "numpy ndarray."
            )
        if len(y_names) not in y_data.shape:
            raise IndexError(
                "The number of labels and the number of signals "
                "must be the same."
            )
        if len(y_data) != len(self._Dataset.dataset["OUTPUT"].values):
            raise IndexError(
                "The length of the input signal must be equal "
                "to the length "
                "of the other signals in the Dataset."
            )

    def plot_simulations(
        self,
        # Cam be a positional or a keyword arg
        list_sims: str | list[str] | None = None,
        dataset: Literal["in", "out", "both"] | None = None,
        layout: Literal[
            "constrained", "compressed", "tight", "none"
        ] = "tight",
        ax_height: float = 1.8,
        ax_width: float = 4.445,
    ) -> matplotlib.figure.Figure:
        """Plot the stored simulation results.

        Possible values of the parameters describing the plot aesthetics,
        such as the `linecolor_input` or the `alpha_output`,
        are the same for the corresponding `matplotlib.axes.Axes.plot`.

        You are free to manipulate the returned figure as you want by using
        any method of the class `matplotlib.figure.Figure`.

        Please, refer to *matplotlib* docs for more info.


        Example
        -------
        >>> fig = vs.plot_simulations() # ds is a dymoval ValidationSession
        object
        # The following are methods of the class `matplotlib.figure.Figure`
        >>> fig.set_size_inches(10,5)
        >>> fig.set_layout_engine("constrained")
        >>> fig.savefig("my_plot.svg")


        Parameters
        ----------
        list_sims:
            List of simulation names.
        dataset:
            Specify whether the dataset shall be plotted.

            - *"in"*: dataset only the input signals of the dataset.
            - *"out"*: dataset only the output signals of the dataset.
            - *"both"*: dataset both the input and the output signals of the
              dataset.

        layout:
            Figure layout.
        ax_height:
            Approximative height (inches) of each subplot.
        ax_width:
            Approximative width (inches) of each subplot.
        """
        # TODO: could be refactored
        # It uses the left axis for the simulation results and the dataset
        # output.
        # If the dataset input is overlapped, then we use the right axes.
        # However, if the number of inputs "p" is greater than the number of
        # outputs "q", we use the left axes of the remaining p-q axes since
        # there is no need to create a pair of axes only for one extra signal.

        # ================================================================
        # Validate and arrange the plot setup.
        # ================================================================
        # check if the sim list is empty
        self._sim_list_validate()

        # Check the passed list of simulations is non-empty.
        # or that the passed name actually exist
        if not list_sims:
            list_sims = self.simulations_names
        else:
            list_sims = obj2list(list_sims)
            sim_not_found = difference_lists_of_str(
                list_sims, self.simulations_names
            )
            if sim_not_found:
                raise KeyError(
                    f"Simulation {sim_not_found} not found. "
                    "Check the available simulations names with "
                    "'simulations_namess()'"
                )

        # Now we start
        vs = self
        ds_val = self._Dataset
        df_val = ds_val.dataset
        df_sim = self._simulations_values
        p = len(df_val["INPUT"].columns.get_level_values("names"))
        q = len(df_val["OUTPUT"].columns.get_level_values("names"))
        # ================================================================
        # Arrange the figure
        # ================================================================
        # Arange figure
        fig = plt.figure()
        cmap = plt.get_cmap(COLORMAP)
        if dataset == "in" or dataset == "both":
            n = max(p, q)
        else:
            n = q
        nrows, ncols = factorize(n)
        grid = fig.add_gridspec(nrows, ncols)
        # Set a dummy initial axis
        axes = fig.add_subplot(grid[0])

        # ================================================================
        # Start the simulations plots
        # ================================================================
        # Iterate through all the simulations
        sims = list_sims
        for kk, sim in enumerate(sims):
            signals_units = vs.simulation_signals_list(sim)
            for ii, s in enumerate(signals_units):
                if kk > 0:
                    # Second (and higher) roundr of simulations
                    axes = fig.get_axes()[ii]
                else:
                    axes = fig.add_subplot(grid[ii], sharex=axes)
                # Actual plot
                # labels = columns names
                df_sim.droplevel(level="units", axis=1).loc[
                    :, (sim, s[0])
                ].plot(
                    subplots=True,
                    grid=True,
                    color=cmap(kk),
                    legend=True,
                    ylabel=f"({s[1]})",
                    xlabel=f"{df_val.index.name[0]} "
                    "({df_val.index.name[1]})",
                    ax=axes,
                )
            # At the end of the first iteration drop the dummmy axis
            if kk == 0:
                fig.get_axes()[0].remove()

        # Add output plots if requested
        # labels = columns names
        if dataset == "out" or dataset == "both":
            signals_units = df_val["OUTPUT"].columns
            for ii, s in enumerate(signals_units):
                axes = fig.get_axes()[ii]
                df_val.droplevel(level="units", axis=1).loc[
                    :, ("OUTPUT", s[0])
                ].plot(
                    subplots=True,
                    grid=True,
                    legend=True,
                    color="gray",
                    xlabel=f"{df_val.index.name[0]} "
                    f"({df_val.index.name[1]})",
                    ax=axes,
                )

        # Until now, all the axes are on the left side.
        # Due to that fig.get_axes() returns a list of all axes
        # and apparently it is not possible to distinguis between
        # left and right, it is therefore wise to keep track of the
        # axes on the left side.
        # Note that len(axes_l) = q, but only until now.
        # len(axes_l) will change if p>q as we will place the remaining
        # p-q inputs on the left side axes to save space.
        axes_l = fig.get_axes()

        # Get labels and handles needed for the legend in all the
        # axes on the left side.
        labels_l = []
        handles_l = []
        for axes in axes_l:
            handles, labels = axes.get_legend_handles_labels()
            labels_l.append(labels)
            handles_l.append(handles)

        # ===============================================
        # Input signal handling.
        # ===============================================

        if dataset == "in" or dataset == "both":
            signals_units = df_val["INPUT"].columns
            for ii, s in enumerate(signals_units):
                # Add a right axes to the existings "q" left axes
                if ii < q:
                    # If there are available axes, then
                    # add a secondary y_axis to it.
                    axes = fig.get_axes()[ii]
                    axes_right = axes.twinx()
                else:
                    # Otherwise, create a new "left" axis.
                    # We do this because there is no need of creating
                    # a pair of two new axes for just one signal
                    axes = fig.add_subplot(grid[ii], sharex=axes)
                    axes_right = axes
                    # Update the list of axis on the left
                    axes_l.append(axes)

                # Plot.
                df_val.droplevel(level="units", axis=1).loc[
                    :, ("INPUT", s[0])
                ].plot(
                    subplots=True,
                    color="gray",
                    linestyle="--",
                    ylabel=f"({s[1]})",
                    xlabel=f"{df_val.index.name[0]} "
                    f"({df_val.index.name[1]})",
                    ax=axes_right,
                )

                # get labels for legend
                handles, labels = axes_right.get_legend_handles_labels()

                # If there are enough axes, then add an entry to the
                # existing legend, otherwise append new legend for the
                # newly added axes.
                if ii < q:
                    labels_l[ii] += labels
                    handles_l[ii] += handles
                else:
                    labels_l.append(labels)
                    handles_l.append(handles)
                    axes_right.grid(True)

        # ====================================================
        # Shade NaN:s areas
        if dataset is not None:
            ds_val._shade_nans(fig.get_axes())

        # Write the legend by considering only the left axes
        for ii, ax in enumerate(axes_l):
            ax.legend(handles_l[ii], labels_l[ii])

        # Title
        fig.suptitle("Simulations results.")

        # Adjust fig size and layout
        # nrows = fig.get_axes()[0].get_gridspec().get_geometry()[0]
        # ncols = fig.get_axes()[0].get_gridspec().get_geometry()[1]
        fig.set_size_inches(ncols * ax_width, nrows * ax_height + 1.25)
        fig.set_layout_engine(layout)

        if is_interactive_shell():
            fig.show()
        else:
            plt.show()

        return fig

    # TODO:
    def trim(
        self: Self,
        tin: float | None = None,
        tout: float | None = None,
        verbosity: int = 0,
        **kwargs: Any,
    ) -> Self:
        """
        Trim the Validation session
        :py:class:`ValidationSession <dymoval.validation.ValidationSession>`
        object.

        If not `tin` or `tout` are passed, then the selection is
        made graphically.

        Parameters
        ----------
        *signals :
            Signals to be plotted in case of trimming from a plot.
        tin :
            Initial time of the desired time interval
        tout :
            Final time of the desired time interval.
        verbosity :
            Depending on its level, more or less info is displayed.
            The higher the value, the higher is the verbosity.
        **kwargs:
            kwargs to be passed to the
            :py:meth:`~dymoval.validation.ValidationSession.plot_simulations` method.

        """
        # This function is similar to Dataset.trim

        def _graph_selection(
            vs: Self,
            **kwargs: Any,
        ) -> tuple[float, float]:  # pragma: no cover
            # Select the time interval graphically
            # OBS! This part cannot be automatically tested because the it
            # require
            # manual action from the user (resize window).
            # Hence, you must test this manually.

            # Get axes from the plot and use them to extract tin and tout
            figure = vs.plot_simulations(**kwargs)
            axes = figure.get_axes()

            # Define the selection dictionary
            selection = {"tin": 0.0, "tout": vs._Dataset.dataset.index[-1]}

            def update_time_interval(ax):  # type:ignore
                time_interval = ax.get_xlim()
                selection["tin"], selection["tout"] = time_interval
                selection["tin"] = max(selection["tin"], 0.0)
                selection["tout"] = max(selection["tout"], 0.0)
                print(
                    f"Updated time interval: {selection['tin']} to "
                    f"{selection['tout']}"
                )

            # Connect the event handler to the xlim_changed event
            cid = axes[0].callbacks.connect(
                "xlim_changed", update_time_interval
            )
            fig = axes[0].get_figure()
            assert fig is not None

            fig.suptitle("Trim the simulation results.")

            # =======================================================
            # By using this while loop we never give back the control to the
            # prompt. In this way user is constrained to graphically select a
            # time interval or to close the figure window if it wants the
            # control back.
            # The "event loop" is a programming structure that waits for and
            # dispatch events and programs. An example below.
            # An alternative better solution is welcome!
            try:
                while fig in [plt.figure(num) for num in plt.get_fignums()]:
                    plt.pause(0.1)
            except Exception as e:
                print(f"An error occurred {e}")
                plt.close(fig)
            finally:
                plt.close(fig)

            # =======================================================
            axes[0].remove_callback(cid)
            tin_sel = selection["tin"]
            tout_sel = selection["tout"]

            return tin_sel, tout_sel

        # =============================================
        # Trim ValidationSession main function
        # The user can either pass the pair (tin,tout) or
        # he/she can select it graphically if nothing has passed
        # =============================================

        vs = deepcopy(self)
        # Check if info on (tin,tout) is passed
        if tin is None and tout is not None:
            tin_sel = vs._Dataset.dataset.index[0]
            tout_sel = tout
        # If only tin is passed, then set tout to the last time sample.
        elif tin is not None and tout is None:
            tin_sel = tin
            tout_sel = vs._Dataset.dataset.index[-1]
        elif tin is not None and tout is not None:
            tin_sel = tin
            tout_sel = tout
        else:  # pragma: no cover
            tin_sel, tout_sel = _graph_selection(self, **kwargs)

        if verbosity != 0:
            print(
                f"\n tin = {tin_sel}{vs._Dataset.dataset.index.name[1]} ",
                f" tout = {tout_sel}{vs._Dataset.dataset.index.name[1]}",
            )

        # Now you can trim the dataset and update all the
        # other time-related attributes
        vs._Dataset.dataset = (
            vs._Dataset.dataset.loc[tin_sel:tout_sel, :]  # type: ignore[misc]
        )
        vs._Dataset._nan_intervals = vs._Dataset._find_nan_intervals()
        vs._Dataset.coverage = vs._Dataset._find_dataset_coverage()

        # ... and shift everything such that tin = 0.0
        vs._Dataset._shift_dataset_tin_to_zero()
        vs._Dataset.dataset = vs._Dataset.dataset

        # Also trim the simulations
        vs._simulations_values = vs.simulations_values.loc[
            tin_sel:tout_sel, :  # type: ignore[misc]
        ]
        vs.simulations_values.index = vs._Dataset.dataset.index

        for sim_name in vs.simulations_names:
            vs._append_validation_statistics(sim_name)

        return vs

    def plot_residuals(
        self,
        list_sims: str | list[str] | None = None,
        *,
        plot_input: bool = True,
        layout: Literal[
            "constrained", "compressed", "tight", "none"
        ] = "tight",
        ax_height: float = 1.8,
        ax_width: float = 4.445,
    ) -> tuple[
        matplotlib.figure.Figure,
        matplotlib.figure.Figure,
        matplotlib.figure.Figure,
    ]:
        """Plot the residuals auto- and cross-correlation functions.

        Parameters
        ----------
        list_sims :
            List of simulations.
            If empty, all the simulations are plotted.
        layout:
            Figures layout.
        ax_height:
            Approximative height (inches) of each subplot.
        ax_width:
            Approximative width (inches) of each subplot.


        You are free to manipulate the returned figure as you want by using
        any
        method of the class `matplotlib.figure.Figure`.

        Please, refer to *matplotlib* docs for more info.


        Example
        -------
        >>> fig = vs.plot_residuals() # vs is a dymoval ValidationSession
        object
        # The following are methods of the class `matplotlib.figure.Figure`
        >>> fig.set_size_inches(10,5)
        >>> fig.set_layout_engine("constrained")
        >>> fig.savefig("my_plot.svg")
        """
        # Raises
        # ------
        # KeyError
        #     If the requested simulation list is empty.

        # Check if you have any simulation available
        self._sim_list_validate()
        if not list_sims:
            list_sims = self.simulations_names
        else:
            list_sims = obj2list(list_sims)
            sim_not_found = difference_lists_of_str(
                list_sims, self.simulations_names
            )
            if sim_not_found:
                raise KeyError(
                    f"Simulation {sim_not_found} not found. "
                    "Check the available simulations names with "
                    "'simulations_namess()'"
                )
        Ruu = self._Ruu_tensor
        Ree = self._Ree_tensor
        Rue = self._Rue_tensor

        # Get p
        k0 = list(Rue.keys())[0]
        p = Rue[k0].R.shape[0]

        # Get q
        k0 = list(Ree.keys())[0]
        q = Ree[k0].R.shape[0]

        # ===============================================================
        # Plot input auto-correlation
        # ===============================================================
        if plot_input:
            fig, ax = plt.subplots(p, p, squeeze=False)
            plt.setp(ax, ylim=(-1.2, 1.2))
            for ii in range(p):
                for jj in range(p):
                    if is_latex_installed:
                        title = rf"$\hat r_{{u_{ii}u_{jj}}}$"
                    else:
                        title = rf"r_u{ii}u_{jj}"
                    ax[ii, jj].stem(
                        Ruu.R[ii, jj].lags,
                        Ruu.R[ii, jj].values,
                        label=title,
                    )
                    ax[ii, jj].grid(True)
                    ax[ii, jj].set_xlabel("Lags")
                    ax[ii, jj].set_title(title)
                    ax[ii, jj].legend()
            fig.suptitle("Input auto-correlation")

            # Adjust fig size and layout
            # Walrus operator to make mypy happy. Alternatively, you could use
            # assert, see below.
            if (gs := fig.get_axes()[0].get_gridspec()) is not None:
                nrows, ncols = gs.get_geometry()
            fig.set_size_inches(ncols * ax_width, nrows * ax_height + 1.25)
            fig.set_layout_engine(layout)

        # ===============================================================
        # Plot residuals auto-correlation
        # ===============================================================
        cmap = plt.get_cmap(COLORMAP)
        fig1, ax1 = plt.subplots(q, q, squeeze=False)
        plt.setp(ax1, ylim=(-1.2, 1.2))
        for kk, sim_name in enumerate(list_sims):
            # Convert the RGBA color to hex format
            color_hex = matplotlib.colors.to_hex(cmap(kk))
            for ii in range(q):
                for jj in range(q):
                    if is_latex_installed:
                        title = rf"$\hat r_{{\epsilon_{ii}\epsilon_{jj}}}$"
                    else:
                        title = rf"r_eps{ii}eps_{jj}"
                    ax1[ii, jj].stem(
                        Ree[sim_name].R[ii, jj].lags,
                        Ree[sim_name].R[ii, jj].values,
                        label=sim_name,
                        linefmt=f"{color_hex}",
                    )
                    ax1[ii, jj].grid(True)
                    ax1[ii, jj].set_xlabel("Lags")
                    ax1[ii, jj].set_title(title)
                    ax1[ii, jj].legend()
        fig1.suptitle("Residuals auto-correlation")

        # Adjust fig size and layout
        # Walrus operator to make mypy happy. Alternatively, you could use
        # assert, see below.
        if (gs := fig1.get_axes()[0].get_gridspec()) is not None:
            nrows, ncols = gs.get_geometry()
        fig1.set_size_inches(ncols * ax_width, nrows * ax_height + 1.25)
        fig1.set_layout_engine(layout)

        # ===============================================================
        # Plot input-residuals cross-correlation
        # ===============================================================
        fig2, ax2 = plt.subplots(p, q, sharex=True, squeeze=False)
        plt.setp(ax2, ylim=(-1.2, 1.2))
        for kk, sim_name in enumerate(list_sims):
            # Convert the RGBA color to hex format
            color_hex = matplotlib.colors.to_hex(cmap(kk))
            for ii in range(p):
                for jj in range(q):
                    if is_latex_installed:
                        title = rf"$\hat r_{{u_{ii}\epsilon_{jj}}}$"
                    else:
                        title = rf"r_u{ii}eps{jj}"
                    ax2[ii, jj].stem(
                        Rue[sim_name].R[ii, jj].lags,
                        Rue[sim_name].R[ii, jj].values,
                        label=sim_name,
                        linefmt=f"{color_hex}",
                    )
                    ax2[ii, jj].grid(True)
                    ax2[ii, jj].set_xlabel("Lags")
                    ax2[ii, jj].set_title(title)
                    ax2[ii, jj].legend()
        fig2.suptitle("Input-residuals cross-correlation")

        # Adjust fig size and layout
        gs = fig2.get_axes()[0].get_gridspec()
        assert gs is not None
        nrows, ncols = gs.get_geometry()
        fig2.set_size_inches(ncols * ax_width, nrows * ax_height + 1.25)
        fig2.set_layout_engine(layout)

        if is_interactive_shell():
            fig.show()
            fig1.show()
            fig2.show()
        else:
            plt.show()

        return fig, fig1, fig2

    def simulation_signals_list(self, sim_name: str | list[str]) -> list[str]:
        """
        Return the signal name list of a given simulation.

        Parameters
        ----------
        sim_name :
            Simulation name.

        """
        self._sim_list_validate()
        return list(self._simulations_values[sim_name].columns)

    def clear(self) -> Self:
        """Remove all the stored simulation results in the current
        ValidationSession object."""

        vs_temp = deepcopy(self)
        sim_names = vs_temp.simulations_names
        for x in sim_names:
            vs_temp = vs_temp.drop_simulations(x)
        return vs_temp

    def append_simulation(
        self,
        sim_name: str,
        y_names: str | list[str],
        y_data: np.ndarray,
    ) -> Self:
        r"""
        Append simulation results.

        Parameters
        ----------
        sim_name :
            Simulation name.
        y_label :
            Simulation output signal names.
        y_data :
            Simulated out expressed as :math:`N\times q` array
            with `N` observations of `q` signals.
        """
        vs_temp = deepcopy(self)
        # df_sim = vs_temp.simulations_values

        y_names = obj2list(y_names)
        vs_temp._simulation_validation(sim_name, y_names, y_data)

        y_units = list(
            vs_temp._Dataset.dataset["OUTPUT"].columns.get_level_values(
                "units"
            )
        )

        # Initialize sim df
        df_sim = pd.DataFrame(
            data=y_data, index=vs_temp._Dataset.dataset.index
        )
        multicols = list(zip([sim_name] * len(y_names), y_names, y_units))
        df_sim.columns = pd.MultiIndex.from_tuples(
            multicols, names=["sim_names", "signal_names", "units"]
        )

        # Concatenate df_sim with the current sim results
        vs_temp._simulations_values = vs_temp.simulations_values.join(
            df_sim, how="right"
        ).rename_axis(df_sim.columns.names, axis=1)

        # Update residuals auto-correlation and cross-correlation attributes
        vs_temp._append_validation_statistics(sim_name)

        return vs_temp

    def drop_simulations(self, *sims: str) -> Self:
        """Drop simulation results from the validation session object.

        Parameters
        ----------
        *sims:
            Name of the simulations to be dropped.
        """

        vs_temp = deepcopy(self)
        vs_temp._sim_list_validate()

        for sim_name in sims:
            if sim_name not in vs_temp.simulations_names:
                raise ValueError(f"Simulation {sim_name} not found.")
            vs_temp._simulations_values = vs_temp.simulations_values.drop(
                sim_name, axis=1, level="sim_names"
            )
            vs_temp.simulations_values.columns = (
                vs_temp.simulations_values.columns.remove_unused_levels()
            )

            vs_temp._Ree_tensor.pop(sim_name)
            vs_temp._Rue_tensor.pop(sim_name)

            vs_temp._validation_statistics = (
                vs_temp._validation_statistics.drop(sim_name, axis=1)
            )

        return vs_temp


def validate_models(
    measured_in: np.ndarray | Sequence[Signal],
    measured_out: np.ndarray | list[Signal],
    simulated_out: np.ndarray | list[np.ndarray],
    sampling_period: float | None = None,
    **kwargs: Any,
) -> ValidationSession:
    r"""Validate models based on measured and simulated data.


    Parameters
    ----------
    measured_in:
        Real-world measurements data related to the input. If dtype is
        `np.ndarray`, then the shape must be :math:`N\times p`, where `N` is
        the number of observations and `p` is the number of inputs.
    measured_out:
        Real-world measurements data related to the output. If dtype is
        `np.ndarray`, then the shape must be :math:`N\times q`, where `N` is
        the number of observations and `q` is the number of outputs.
    simulated_out:
        Simulated output. The shape of the `np.ndarray` must be
        :math:`N\times q`, where `N` is the
        number of observations and `q` is the number of outputs.
    sampling_period:
        Signals sampling period.
    **kwargs:
        Keyword arguments passed to
        :py:class:`~dymoval.validation.ValidationSession` constructor.
    """

    def _dummy_signal_list(
        dataset: np.ndarray,  # Must be 2D array
        sampling_period: float,
        kind: Literal["in", "out"],
    ) -> list[Signal]:
        uy_label = "u" if kind == "in" else "y"
        signal_list = []
        for ii in range(dataset.shape[1]):
            tmp: Signal = {
                "name": f"{uy_label}{ii}",
                "samples": dataset[:, ii],  # Must be a 1D array
                "signal_unit": "NA",
                "sampling_period": sampling_period,
                "time_unit": "NA",
            }
            signal_list.append(deepcopy(tmp))
        return signal_list

    def _to_list_of_Signal(
        data: np.ndarray | Sequence[Signal],
        sampling_period: float,
        kind: Literal["in", "out"],
    ) -> list[Signal]:
        # Case 2D np.ndarray, convert to a list[Signal]
        if isinstance(data, np.ndarray):
            data_list = _dummy_signal_list(
                dataset=data, sampling_period=sampling_period, kind=kind
            )

        # Case list[Signal]
        elif isinstance(data, list) and all(
            isinstance(item, dict) and set(item.keys()) == set(SIGNAL_KEYS)
            for item in data
        ):
            # elif isinstance(data, list) and all(isinstance(item, dict) and
            # set(item.keys()) == set(SIGNAL_KEYS), for all items in data):
            data_list = data
        else:
            raise ValueError(
                "'measured_in' and 'measured_out' must be 2D-arrays "
                "or list of Signals."
            )

        return data_list

    # ======== MAIN ================
    # Sanity check
    if isinstance(measured_in, np.ndarray) and measured_in.ndim != 2:
        raise IndexError("'measured_in' shall be a Nxp np.ndarray.")

    if isinstance(measured_out, np.ndarray) and measured_out.ndim != 2:
        raise IndexError("'measured_out' shall be a Nxq np.ndarray.")

    # Gather sampling_period from Signals
    if (
        isinstance(measured_out, list)
        and isinstance(measured_out[0], dict)
        and set(measured_out[0].keys()) == set(SIGNAL_KEYS)
    ):
        sampling_period = measured_out[0]["sampling_period"]
    elif sampling_period is None:
        raise TypeError("'sampling_period' missing.")

    # Convert everything into list[Signal] to create a dataset object
    measured_in_list = _to_list_of_Signal(
        data=measured_in, sampling_period=sampling_period, kind="in"
    )
    measured_out_list = _to_list_of_Signal(
        data=measured_out, sampling_period=sampling_period, kind="out"
    )

    # Build Dataset instance and Validation instance
    input_labels = [s["name"] for s in measured_in_list]
    output_labels = [s["name"] for s in measured_out_list]
    signal_list = measured_in_list + measured_out_list
    validate_signals(*signal_list)
    ds = Dataset(
        "dummy",
        signal_list,
        input_labels,
        output_labels,
        full_time_interval=True,
    )

    # Create a ValidationSession object placeholder
    vs = ValidationSession("quick & dirty", ds, **kwargs)

    # ---- Fix simulated_out arg -----
    simulated_out_list = obj2list(simulated_out)
    # Fetch the number of observations
    N = measured_out_list[0]["samples"].shape[0]

    # Format check
    if all(
        sim.shape[0] != N or sim.shape[1] != len(measured_out_list)
        for sim in simulated_out_list
    ):
        raise ValueError(
            "'simulated_out' shall be a "
            f"{N}x{len(measured_out_list)} np.ndarray "
            f"or a list of {N}x{len(measured_out_list)} np.ndarray."
        )

    # Append simulated_outs
    for ii, sim in enumerate(simulated_out_list):
        sim_name = f"Sim_{ii}"
        vs = vs.append_simulation(
            sim_name=sim_name, y_names=output_labels, y_data=sim
        )

    return vs
