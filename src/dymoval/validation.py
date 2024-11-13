"""Module containing everything related to validation."""

# The following is needed when there are methods that return instance of the
# class itself.
# TODO If you remove python 3.10 remove typing_extensions as Self in typing is
# part of the standard python package starting from 3.11
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self  # noqa

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from copy import deepcopy
import scipy.signal as signal
from .config import (
    COLORMAP,
    Statistic_type,
    STATISTIC_TYPE,
    SIGNAL_KEYS,
    is_latex_installed,
)
from .utils import (
    is_interactive_shell,
    factorize,
    difference_lists_of_str,
    obj2list,
)

from .dataset import Dataset, Signal, validate_signals
from typing import List, Dict, Literal, Any, NamedTuple
from dataclasses import dataclass

__all__ = [
    "XCorrelation",
    "rsquared",
    "whiteness_level",
    "ValidationSession",
]


# def _get_nlags(
#     local_weights: np.ndarray | None = None,
#     nlags_from_user: int | None = None,
# ) -> int:
#     # If user explicitly set nlags, pick that one, if the user passed
#     # local weights pick the minumum, otherwise if user passed system
#     # dominant time constant pick k*T/Ts, otherwise if nothing is
#     # specified, just pick N//5, where N is the number of observations.
#     if nlags_from_user is not None:
#         nlags = nlags_from_user
#     elif local_weights is not None:
#         nlags = local_weights.size
#     else:  # all None
#         nlags = 20
#     return nlags


# Util for defining XCorrelation elements.
# XCorrelation dataclass is a matrix of Rxy elements.
class _rxy(NamedTuple):
    values: np.ndarray
    lags: np.ndarray


@dataclass
class XCorrelation:
    # You have to manually write the type in the docstrings
    # and you have to exclude them in the :automodule:
    """Cross-correlation of two MIMO signals  `X` and `Y`.

    If `X = Y` then it return the normalized auto-correlation of `X`.
    You must pass ``X_Bandwidth``, ``Y_Bandwidth`` and ``sampling_period`` or
    none of them.

    The cross-correlation functions are stored in the attribute `R` which is
    an array where the `(i, j)`-th element
    is the cross-correlation function between the `i`-th signal of
    `X` and the `j`-th signal of `Y`. The cross-correlation functions
    are ``NamedTuple`` s with attributes
    values (``np.ndarray``) and lags (``np.ndarray``).


    Parameters
    ----------
    name:
        The XCorrelation name.
    X:
        MIMO signal realizations expressed as `Nxp` 2-D array
        of `N` observations of `p` signals.
    Y:
        MIMO signal realizations expressed as `Nxq` 2-D array
        of `N` observations of `q` signals.
    nlags:
        `pxq` array where the `(i, j)`-th element represents the number of lags
        of the cross-correlation function associated to the `i`-th signal of
        `X` with the `j`-th signal of `Y`.
    X_bandwidths:
        1-D array representing the bandwidths of each signal in  `X`.
    Y_bandwidths:
        1-D array representing the bandwidths of each signal in `Y`.
    sampling_period:
        Sampling period of the signals X and Y.

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
        array([-9, -8, -7, -6, -5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5,  6,  7,
        8,  9])
    >>> Rxy.R[0,2].values
        array([-0.06377225, -0.0083634 ,  0.14850791,  0.06379516, -0.16405862,
               -0.24074438,  0.14147755,  0.06538316, -0.26679362,  0.14813509,
                0.64887265,  0.22247482, -0.4785613 , -0.30908332,  0.12834458,
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
        self.name = name

        # R is a matrix where each element is rij(\tau).
        # The range of \tau may change as it depends on the sampling_period
        # and the bandwidth of a given signal.
        self.R = self._init_R(
            X=X,
            Y=Y,
            nlags=nlags,
            X_bandwidths=X_bandwidths,
            Y_bandwidths=Y_bandwidths,
            sampling_period=sampling_period,
        )

        if np.array_equal(X, Y):
            self.kind = "auto-correlation"
        else:
            self.kind = "cross-correlation"

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
        #  3. trim based on the lags needed (you don't need N observations lags)

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
        if isinstance(X_bandwidths, float):
            X_bandwidths = np.array([X_bandwidths])
        if isinstance(Y_bandwidths, float):
            Y_bandwidths = np.array([Y_bandwidths])

        if isinstance(X_bandwidths, np.ndarray):
            if X_bandwidths.size != p:
                raise IndexError(
                    f"The number of elements of 'X_bandwidths' must be equal to {p}"
                )

        if isinstance(Y_bandwidths, np.ndarray):
            if Y_bandwidths.size != q:
                raise IndexError(
                    "The number of elements of 'Y_bandwidths' must be equal to {q}"
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
                # check normalized cross-correlation for stochastic processes on Wikipedia.
                # Nevertheless, the cross-correlation is in-fact the same as E[].
                # More specifically, the cross-correlation generate a sequence
                # [E[XY(\tau=0))] E[XY(\tau=1))], ...,E[XY(\tau=N))]] and this is
                # the reason why in the computation below we use signal.correlation.
                #
                # Another way of seeing it, is that to secure that the cross-correlation
                # is always between -1 and 1, we "normalize" the observations X and Y
                # Google for "Standard score"
                #
                # At the end, for each pair (ii,jj) you have Rxy_values = r_{x_ii,y_jj}(\tau), therefore
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
                    step = int(
                        1
                        // (
                            2.0
                            * sampling_period
                            * max(X_bandwidths[ii], Y_bandwidths[jj])
                        )
                    )
                else:
                    # We downsample with step 1 (= no downsampling). TODO Could be
                    # refactored
                    step = 1

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
            f"R shape: {self.R.shape}\n"  # Shows the dimensions of the tensor
        )

        return repr_str

    def estimate_whiteness(
        self,
        local_statistic: Statistic_type = "quadratic",
        local_weights: (
            np.ndarray | None
        ) = None,  # shall be p*q matrix where each element is a 1-D array.
        global_statistic: Statistic_type = "max",
        global_weights: np.ndarray | None = None,  # Shall be a p*q matrix
    ) -> tuple[float, np.ndarray]:
        """Return the whiteness estimates based on the selected statistics.

        Compute first the statistic for each `pxq` cross-correlation function,
        then compute the statistic of the resulting `pxq` array.

        The statistics are computed through the function
        :py:meth:`~dymoval.validation.compute_statistic`.

        Parameters
        ----------
        local_statistic:
            Statistic type for each `(i,j)` cross-correlation function of `XCorrelation.R` array.

        local_weights:
            Weights for each `(i, j)` element of `XCorrelation.R` array. It
            shall be a `pxq` array where each element is a 1-D array.

        global_statistic:
            Statistic type for each element of the resulting `pxq` array
            obtained by computing the statistics for each `pxq`
            cross-correlation functions.

        global_weights:
            Weights for each element of the resulting `pxq` array
            obtained by computing the statistics for each `pxq`
            cross-correlation functions.
            It shall be a `pxq` array.

        Returns
        -------
        whiteness_estimate:
            The overall whiteness estimate.
        whiteness_matrix:
            A `pxq` array where the `(i, j)`-th element is the statistic computed
            for the `(i, j)`-th cross-correlation function.

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
            local_statistic not in STATISTIC_TYPE
            or global_statistic not in STATISTIC_TYPE
        ):
            raise ValueError(f"Statistic type must be in {STATISTIC_TYPE}")

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
                            "Number of lags and number of weights must be the same.\n"
                            f"In index {ii, jj} you have {len(R[ii, jj].lags)} lags and "
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
        """Plot the `pxq` cross-correlation functions."""

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
    statistic: Statistic_type = "mean",
    weights: np.ndarray | None = None,
) -> float:
    """Compute the statistic of a sequence of numbers.

    The samples can be weighted through the `weights` array.

    If data.shape dimension is greater than 1 it will be flatten to a 1-D array.

    The  return values are normalized such that the function always return
    values between 0 and 1.
    This measure the normalized distance from the origin in a Euclidean space
    Define W based on the Ljung-Box statistic formula
    lags = np.arange(1, data.size + 1)
    W = np.diag(1 / (data.size - lags))
    normalization_factor = np.max(weights)

    Inf norm is not constrained between 0 and 1.


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

        result = quadratic_form / (np.max(weights) * len(data))
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
        raise ValueError(f"'statistic' must be one of [{STATISTIC_TYPE}]")
    return float(result)


def rsquared(x: np.ndarray, y: np.ndarray) -> float:
    """
    Return the :math:`R^2` value of two signals.

    Signals can be MIMO.

    Parameters
    ----------
    x:
        First input signal.
    y:
        Second input signal.

    Raises
    ------
    IndexError
        If x and y don't have the same number of samples.
    """

    if x.shape != y.shape:
        raise IndexError("Arguments must have the same shape.")
    eps = x - y
    # Compute r-square fit (%)
    x_mean = np.mean(x, axis=0)
    r2 = (
        1.0 - np.linalg.norm(eps, 2) ** 2 / np.linalg.norm(x - x_mean, 2) ** 2
    ) * 100
    return r2  # type: ignore


# TODO: Not happy with this
def whiteness_level(
    data: np.ndarray,
    data_bandwidths: np.ndarray | float | None = None,
    sampling_period: float | None = None,
    nlags: np.ndarray | None = None,
    local_statistic: Statistic_type = "quadratic",
    local_weights: (
        np.ndarray | None
    ) = None,  # shall be a p*q matrix where each element is a 1D-array (like the lags)
    global_statistic: Statistic_type = "max",
    global_weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Estimate the whiteness of the time-series `data`.

    If data is a MIMO signal, then first the auto-correlation functions of
    each pair of signal is computed. Then, for each auto-correlation function
    a statistic is computed according to `local_statistic` value, thus
    resulting in a `pxp` array. Finally, a statistic of the resulting array is
    computed based on the values of `global_statistic` and `global_weights`.

    See Also
    --------
    :py:meth:`~dymoval.validation.compute_statistic`
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
    """The *ValidationSession* class is used to validate models against a given dataset.

    A *ValidationSession* object is instantiated from a :ref:`Dataset` object.
    A validation session *name* shall be also provided.

    Multiple simulation results can be appended to the same *ValidationSession* instance,
    but for each ValidationSession instance only a :ref:`Dataset` object is considered.

    If the :ref:`Dataset` object changes,
    it is recommended to create a new *ValidationSession* instance.
    """

    def __init__(
        self,
        name: str,
        validation_dataset: Dataset,
        U_bandwidths: np.ndarray | float | None = None,
        Y_bandwidths: np.ndarray | float | None = None,
        # Model validation
        validation_thresholds: Dict[str, float] | None = None,
        ignore_input: bool = False,
        # The following are input to XCorrelation.estimate_whiteness() method.
        # input auto-correlation
        u_acorr_nlags: np.ndarray | None = None,
        u_acorr_local_statistic_type_1st: Statistic_type = "mean",
        u_acorr_global_statistic_type_1st: Statistic_type = "max",
        u_acorr_local_statistic_type_2nd: Statistic_type = "quadratic",
        u_acorr_global_statistic_type_2nd: Statistic_type = "max",
        u_acorr_local_weights: np.ndarray | None = None,
        u_acorr_global_weights: np.ndarray | None = None,
        # residuals auto-correlation
        eps_acorr_nlags: np.ndarray | None = None,
        eps_acorr_local_statistic_type_1st: Statistic_type = "mean",
        eps_acorr_global_statistic_type_1st: Statistic_type = "max",
        eps_acorr_local_statistic_type_2nd: Statistic_type = "quadratic",
        eps_acorr_global_statistic_type_2nd: Statistic_type = "max",
        eps_acorr_local_weights: np.ndarray | None = None,
        eps_acorr_global_weights: np.ndarray | None = None,
        # input-residuals cross-Correlation
        ueps_xcorr_nlags: np.ndarray | None = None,
        ueps_xcorr_local_statistic_type_1st: Statistic_type = "mean",
        ueps_xcorr_global_statistic_type_1st: Statistic_type = "max",
        ueps_xcorr_local_statistic_type_2nd: Statistic_type = "quadratic",
        ueps_xcorr_global_statistic_type_2nd: Statistic_type = "max",
        ueps_xcorr_local_weights: np.ndarray | None = None,
        ueps_xcorr_global_weights: np.ndarray | None = None,
    ) -> None:
        # Once you created a ValidationSession you should not change the validation dataset.
        # Create another ValidationSession with another validation dataset
        # By using the constructors, you should have no types problems because the check is done there.

        # =============================================
        # Class attributes
        # ============================================

        self._Dataset: Dataset = validation_dataset
        """The reference :ref:`Dataset` object."""

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
        self._default_nlags = 41

        self._simulations_values: pd.DataFrame = pd.DataFrame(
            index=validation_dataset.dataset.index, columns=[[], [], []]
        )
        """The appended simulation results.
        This attribute is automatically set through
        :py:meth:`~dymoval.validation.ValidationSession.append_simulation`
        and it should be considered as a *read-only* attribute."""

        # Input: Ruu
        self._u_auto_correlation_tensors: XCorrelation
        """The auto-correlation tensors.
        This attribute is automatically set
        and it should be considered as a *read-only* attribute."""

        # Format: 'name_sim': Ree
        self._eps_auto_correlation_tensors: dict[str, XCorrelation] = {}
        self._eps_acorr_whiteness_1st: dict[str, float] = {}
        self._eps_acorr_whiteness_matrix_1st: dict[str, np.ndarray] = {}
        self._eps_acorr_whiteness_2nd: dict[str, float] = {}
        self._eps_acorr_whiteness_matrix_2nd: dict[str, np.ndarray] = {}
        """The auto-correlation tensors.
        This attribute is automatically set
        and it should be considered as a *read-only* attribute."""

        # Format: 'name_sim': Rue
        self._ueps_cross_correlation_tensors: dict[str, XCorrelation] = {}
        self._ueps_xcorr_whiteness_1st: dict[str, float] = {}
        self._ueps_xcorr_whiteness_matrix_1st: dict[str, np.ndarray] = {}
        self._ueps_xcorr_whiteness_2nd: dict[str, float] = {}
        self._ueps_xcorr_whiteness_matrix_2nd: dict[str, np.ndarray] = {}
        """The cross-correlation tensors.
        This attribute is automatically set
        and it should be considered as a *read-only* attribute."""

        # ------------------ Input --------------------
        self._U_bandwidths = U_bandwidths

        # Input nlags
        # Default 41 lags (20 negative and 20 positive)
        self._u_acorr_nlags = np.full(
            (self._p, self._p), fill_value=self._default_nlags
        )
        if u_acorr_nlags is not None:
            if (
                u_acorr_nlags.shape[0] < self._p
                or u_acorr_nlags.shape[1] < self._p
            ):
                raise IndexError(
                    f"'u_acorr_nlags' shall be a {self._p}x{self._p} array."
                )
            else:
                self._u_acorr_nlags = u_acorr_nlags[0 : self._p, 0 : self._p]
        elif u_acorr_local_weights is not None:
            # Iterate through the input array and count the number of lags
            for ii in range(self._p):
                for jj in range(self._p):
                    self._u_acorr_nlags[ii, jj] = len(
                        u_acorr_local_weights[ii, jj]
                    )

        self._u_acorr_local_statistic_type_1st = (
            u_acorr_local_statistic_type_1st
        )
        self._u_acorr_global_statistic_type_1st = (
            u_acorr_global_statistic_type_1st
        )
        self._u_acorr_local_statistic_type_2nd = (
            u_acorr_local_statistic_type_2nd
        )
        self._u_acorr_global_statistic_type_2nd = (
            u_acorr_global_statistic_type_2nd
        )
        self._u_acorr_local_weights = u_acorr_local_weights
        self._u_acorr_global_weights = u_acorr_global_weights

        Ruu = XCorrelation(
            "Ruu",
            X=self._Dataset.dataset["INPUT"].to_numpy(),
            Y=self._Dataset.dataset["INPUT"].to_numpy(),
            nlags=self._u_acorr_nlags,
            X_bandwidths=self._U_bandwidths,
            Y_bandwidths=self._U_bandwidths,
            sampling_period=self._Dataset.dataset.index[1]
            - self._Dataset.dataset.index[0],
        )

        self._u_acorr_tensor = Ruu

        self._u_acorr_whiteness_1st, self._u_acorr_whiteness_matrix_1st = (
            Ruu.estimate_whiteness(
                local_statistic=self._u_acorr_local_statistic_type_1st,
                local_weights=self._u_acorr_local_weights,
                global_statistic=self._u_acorr_global_statistic_type_1st,
                global_weights=self._u_acorr_global_weights,
            )
        )

        self._u_acorr_whiteness_2nd, self._u_acorr_whiteness_matrix2ndd = (
            Ruu.estimate_whiteness(
                local_statistic=self._u_acorr_local_statistic_type_2nd,
                local_weights=self._u_acorr_local_weights,
                global_statistic=self._u_acorr_global_statistic_type_2nd,
                global_weights=self._u_acorr_global_weights,
            )
        )

        # ------------ Residuals -----------------------------
        self._Y_bandwidths = Y_bandwidths

        # Residuals auto-correlation
        self._eps_acorr_tensor: Dict[str, XCorrelation] = {}

        # nlags
        self._eps_acorr_nlags = np.full(
            (self._q, self._q), fill_value=self._default_nlags
        )
        if eps_acorr_nlags is not None:
            if (
                eps_acorr_nlags.shape[0] < self._q
                or eps_acorr_nlags.shape[1] < self._q
            ):
                raise IndexError(
                    f"'eps_acorr_nlags' shall be a {self._q}x{self._q} array."
                )
            else:
                self._eps_acorr_nlags = eps_acorr_nlags[
                    0 : self._q, 0 : self._q
                ]
        elif eps_acorr_local_weights is not None:
            # Iterate through the input array and count the number of lags
            for ii in range(self._q):
                for jj in range(self._q):
                    self._eps_acorr_nlags[ii, jj] = len(
                        eps_acorr_local_weights[ii, jj]
                    )

        self._eps_acorr_local_statistic_type_1st = (
            eps_acorr_local_statistic_type_1st
        )
        self._eps_acorr_global_statistic_type_1st = (
            eps_acorr_global_statistic_type_1st
        )
        self._eps_acorr_local_statistic_type_2nd = (
            eps_acorr_local_statistic_type_2nd
        )
        self._eps_acorr_global_statistic_type_2nd = (
            eps_acorr_global_statistic_type_2nd
        )
        self._eps_acorr_local_weights = eps_acorr_local_weights
        self._eps_acorr_global_weights = eps_acorr_global_weights

        # Input-Residuals cross-correlation
        self._ueps_xcorr_tensor: Dict[str, XCorrelation] = {}

        self._ueps_xcorr_nlags = np.full(
            (self._p, self._q), fill_value=self._default_nlags
        )
        if ueps_xcorr_nlags is not None:
            if (
                ueps_xcorr_nlags.shape[0] < self._p
                or ueps_xcorr_nlags.shape[1] < self._q
            ):
                raise IndexError(
                    f"'ueps_xcorr_nlags' shall be a {self._p}x{self._q} array."
                )
            else:
                self._ueps_xcorr_nlags = ueps_xcorr_nlags[
                    0 : self._p, 0 : self._q
                ]
        elif ueps_xcorr_local_weights is not None:
            # Iterate through the input array and count the number of lags
            for ii in range(self._p):
                for jj in range(self._q):
                    self._ueps_xcorr_nlags[ii, jj] = len(
                        ueps_xcorr_local_weights[ii, jj]
                    )

        self._ueps_xcorr_local_statistic_type_1st = (
            ueps_xcorr_local_statistic_type_1st
        )
        self._ueps_xcorr_global_statistic_type_1st = (
            ueps_xcorr_global_statistic_type_1st
        )
        self._ueps_xcorr_local_statistic_type_2nd = (
            ueps_xcorr_local_statistic_type_2nd
        )
        self._ueps_xcorr_global_statistic_type_2nd = (
            ueps_xcorr_global_statistic_type_2nd
        )
        self._ueps_xcorr_local_weights = ueps_xcorr_local_weights
        self._ueps_xcorr_global_weights = ueps_xcorr_global_weights

        # Initialize validation results DataFrame.
        idx = [
            f"Input whiteness ({self._u_acorr_local_statistic_type_1st}-{self._u_acorr_global_statistic_type_1st})",
            f"Input whiteness ({self._u_acorr_local_statistic_type_2nd}-{self._u_acorr_global_statistic_type_2nd})",
            "R-Squared (%)",
            f"Residuals whiteness ({self._eps_acorr_local_statistic_type_1st}-{self._eps_acorr_global_statistic_type_1st})",
            f"Residuals whiteness ({self._eps_acorr_local_statistic_type_2nd}-{self._eps_acorr_global_statistic_type_2nd})",
            f"Input-Res whiteness ({self._ueps_xcorr_local_statistic_type_1st}-{self._ueps_xcorr_global_statistic_type_1st})",
            f"Input-Res whiteness ({self._ueps_xcorr_local_statistic_type_2nd}-{self._ueps_xcorr_global_statistic_type_2nd})",
        ]
        self._validation_results: pd.DataFrame = pd.DataFrame(
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
        self._outcome: Dict[str, str] = {}
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
        if self._u_acorr_local_weights is None:
            u_acorr_local_weights_str = (
                f"local weights: {self._u_acorr_local_weights}\n"
            )
        else:
            u_acorr_local_weights_str = (
                "local weights: Yes (see self._u_acorr_local_weights)\n"
            )

        if self._u_acorr_global_weights is None:
            u_acorr_global_weights_str = (
                f"global weights: {self._u_acorr_global_weights}\n"
            )
        else:
            u_acorr_global_weights_str = (
                "global weights: Yes (see self._u_acorr_global_weights)\n"
            )

        # u_nlags
        if np.all(
            self._u_acorr_nlags.flatten() == self._u_acorr_nlags.flatten()[0]
        ):
            u_nlags_str = f"num lags: {self._u_acorr_nlags[0, 0]}\n"
        else:
            u_nlags_str = f"num lags: \n{self._u_acorr_nlags}\n"

        # Ignore input
        if self._ignore_input:
            inputs_acorr_str = f"Input ignored: {self._ignore_input}\n"
            Ruu_whiteness_1st = ""
            Ruu_whiteness_2nd = ""
            validation_results = self._validation_results.iloc[2:, :]
        else:
            inputs_acorr_str = (
                f"Inputs auto-correlation\n"
                f"1st statistic: {self._u_acorr_local_statistic_type_1st}-{self._u_acorr_global_statistic_type_1st}\n"
                f"2nd statistic: {self._u_acorr_local_statistic_type_2nd}-{self._u_acorr_global_statistic_type_2nd}\n"
                + u_acorr_local_weights_str
                + u_acorr_global_weights_str
                + u_nlags_str
            )
            Ruu_whiteness_1st = f"{self._validation_results.index[0]}: {self._validation_thresholds['Ruu_whiteness_1st']} \n"
            Ruu_whiteness_2nd = f"{self._validation_results.index[1]}: {self._validation_thresholds['Ruu_whiteness_2nd']} \n"
            validation_results = self._validation_results

        # eps_nlags
        if np.all(
            self._eps_acorr_nlags.flatten()
            == self._eps_acorr_nlags.flatten()[0]
        ):
            eps_nlags_str = f"num lags: {self._eps_acorr_nlags[0, 0]}\n"
        else:
            eps_nlags_str = f"num lags: \n{self._eps_acorr_nlags}\n"

        # eps acorr weights
        if self._eps_acorr_local_weights is None:
            eps_acorr_local_weights_str = (
                f"local weights: {self._eps_acorr_local_weights}\n"
            )
        else:
            eps_acorr_local_weights_str = (
                "local weights: Yes (see self._eps_acorr_local_weights)\n"
            )

        if self._eps_acorr_global_weights is None:
            eps_acorr_global_weights_str = (
                f"global weights: {self._eps_acorr_global_weights}\n"
            )
        else:
            eps_acorr_global_weights_str = (
                "global weights: Yes (see self._eps_acorr_global_weights)\n"
            )

        # ueps_nlags
        if np.all(
            self._ueps_xcorr_nlags.flatten()
            == self._ueps_xcorr_nlags.flatten()[0]
        ):
            ueps_nlags_str = f"num lags: {self._ueps_xcorr_nlags[0, 0]}\n"
        else:
            ueps_nlags_str = f"num lags: \n{self._ueps_xcorr_nlags}\n"

        # ueps xcorr weights
        if self._ueps_xcorr_local_weights is None:
            ueps_xcorr_local_weights_str = (
                f"local weights: {self._ueps_xcorr_local_weights}\n"
            )
        else:
            ueps_xcorr_local_weights_str = (
                "local weights: Yes (see self._ueps_xcorr_local_weights)\n"
            )

        if self._ueps_xcorr_global_weights is None:
            ueps_xcorr_global_weights_str = (
                f"global weights: {self._ueps_xcorr_global_weights}\n"
            )
        else:
            ueps_xcorr_global_weights_str = (
                "global weights: Yes (see self._ueps_xcorr_global_weights)\n"
            )

        repr_str = (
            f"Validation session name: {self.name}\n\n"
            f"Validation setup:\n----------------\n"
            + inputs_acorr_str
            + "\n"
            + f"Residuals auto-correlation:\n"
            f"1st statistic: {self._eps_acorr_local_statistic_type_1st}-{self._eps_acorr_global_statistic_type_1st}\n"
            f"2nd statistic: {self._eps_acorr_local_statistic_type_2nd}-{self._eps_acorr_global_statistic_type_2nd}\n"
            + eps_acorr_local_weights_str
            + eps_acorr_global_weights_str
            + eps_nlags_str
            + "\n"
            +
            #
            f"Input-residuals cross-correlation:\n"
            f"1st statistic: {self._ueps_xcorr_local_statistic_type_1st}-{self._ueps_xcorr_global_statistic_type_1st}\n"
            f"2nd statistic: {self._ueps_xcorr_local_statistic_type_2nd}-{self._ueps_xcorr_global_statistic_type_2nd}\n"
            + ueps_xcorr_local_weights_str
            + ueps_xcorr_global_weights_str
            + ueps_nlags_str
            + "\n"
            +
            #
            f"Validation results:\n-------------------\n"
            f"Thresholds: \n"
            f"{Ruu_whiteness_1st}"
            f"{Ruu_whiteness_2nd}"
            f"{self._validation_results.index[2]}: {self._validation_thresholds['r2']:.4f} \n"
            f"{self._validation_results.index[3]}: {self._validation_thresholds['Ree_whiteness_1st']:.4f} \n"
            f"{self._validation_results.index[4]}: {self._validation_thresholds['Ree_whiteness_2nd']:.4f} \n"
            f"{self._validation_results.index[5]}: {self._validation_thresholds['Rue_whiteness_1st']:.4f} \n"
            f"{self._validation_results.index[6]}: {self._validation_thresholds['Rue_whiteness_2nd']:.4f} \n\n"
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
        return self._Dataset

    @property
    def simulations_values(self) -> pd.DataFrame:
        return self._simulations_values

    @property
    def simulations_names(self) -> list[str]:
        """Return a list of names of the stored simulations."""
        return list(self._simulations_values.columns.levels[0])

    @property
    def outcome(self) -> Dict[str, str]:
        return self._outcome

    def _get_validation_thresholds_default(
        self, ignore_input: bool
    ) -> Dict[str, float]:

        validation_thresholds_default = {
            "Ruu_whiteness_1st": 0.6,
            "Ruu_whiteness_2nd": 0.6,
            "r2": 65,
            "Ree_whiteness_1st": 0.5,
            "Ree_whiteness_2nd": 0.5,
            "Rue_whiteness_1st": 0.5,
            "Rue_whiteness_2nd": 0.5,
        }

        if ignore_input is True:
            del validation_thresholds_default["Ruu_whiteness_1st"]
            del validation_thresholds_default["Ruu_whiteness_2nd"]
        return validation_thresholds_default

    def validation_values(self, sim_name: str) -> Any:
        keys = [
            "Ruu_whiteness_1st",
            "Ruu_whiteness_2nd",
            "r2",
            "Ree_whiteness_1st",
            "Ree_whiteness_2nd",
            "Rue_whiteness_1st",
            "Rue_whiteness_2nd",
        ]
        vals = self._validation_results[sim_name].to_numpy()
        return dict(zip(keys, vals))

    def _append_validation_results(
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
                "Simulation outputs are identical to measured outputs. Are you cheating?"
            )

        # rsquared and various statistics
        r2 = rsquared(y_values, y_sim_values)

        # Residuals auto-correlation
        Ree = XCorrelation(
            "Ree",
            eps,
            eps,
            X_bandwidths=self._Y_bandwidths,
            Y_bandwidths=self._Y_bandwidths,
            nlags=self._eps_acorr_nlags,
            sampling_period=self._Dataset.dataset.index[1]
            - self._Dataset.dataset.index[0],
        )

        self._eps_acorr_tensor[sim_name] = Ree

        (
            self._eps_acorr_whiteness_1st[sim_name],
            self._eps_acorr_whiteness_matrix_1st[sim_name],
        ) = Ree.estimate_whiteness(
            local_statistic=self._eps_acorr_local_statistic_type_1st,
            local_weights=self._eps_acorr_local_weights,
            global_statistic=self._eps_acorr_global_statistic_type_1st,
            global_weights=self._eps_acorr_global_weights,
        )

        (
            self._eps_acorr_whiteness_2nd[sim_name],
            self._eps_acorr_whiteness_matrix_2nd[sim_name],
        ) = Ree.estimate_whiteness(
            local_statistic=self._eps_acorr_local_statistic_type_2nd,
            local_weights=self._eps_acorr_local_weights,
            global_statistic=self._eps_acorr_global_statistic_type_2nd,
            global_weights=self._eps_acorr_global_weights,
        )

        # Input-residals cross-correlation
        Rue = XCorrelation(
            "Rue",
            u_values,
            eps,
            X_bandwidths=self._Y_bandwidths,
            Y_bandwidths=self._Y_bandwidths,
            nlags=self._ueps_xcorr_nlags,
            sampling_period=self._Dataset.dataset.index[1]
            - self._Dataset.dataset.index[0],
        )

        self._ueps_xcorr_tensor[sim_name] = Rue

        (
            self._ueps_xcorr_whiteness_1st[sim_name],
            self._ueps_xcorr_whiteness_matrix_1st[sim_name],
        ) = Rue.estimate_whiteness(
            local_statistic=self._ueps_xcorr_local_statistic_type_1st,
            local_weights=self._ueps_xcorr_local_weights,
            global_statistic=self._ueps_xcorr_global_statistic_type_1st,
            global_weights=self._ueps_xcorr_global_weights,
        )

        (
            self._ueps_xcorr_whiteness_2nd[sim_name],
            self._ueps_xcorr_whiteness_matrix_2nd[sim_name],
        ) = Rue.estimate_whiteness(
            local_statistic=self._ueps_xcorr_local_statistic_type_2nd,
            local_weights=self._ueps_xcorr_local_weights,
            global_statistic=self._ueps_xcorr_global_statistic_type_2nd,
            global_weights=self._ueps_xcorr_global_weights,
        )

        # Append numerical values:
        self._validation_results[sim_name] = np.array(
            [
                self._u_acorr_whiteness_1st,
                self._u_acorr_whiteness_2nd,
                r2,
                self._eps_acorr_whiteness_1st[sim_name],
                self._eps_acorr_whiteness_2nd[sim_name],
                self._ueps_xcorr_whiteness_1st[sim_name],
                self._ueps_xcorr_whiteness_2nd[sim_name],
            ]
        )

        # Compute PASS/FAIL outcome
        local_outcome = []
        validation_dict = self.validation_values(sim_name)
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
                "Check the available simulation names with 'simulations_names()'"
            )

    def _simulation_validation(
        self, sim_name: str, y_names: list[str], y_data: np.ndarray
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
                "The number of outputs of your simulation must be equal to "
                "the number of outputs in the dataset AND "
                "the name of each simulation output shall be unique."
            )
        if not isinstance(y_data, np.ndarray):
            raise ValueError(
                "The type the input signal values must be a numpy ndarray."
            )
        if len(y_names) not in y_data.shape:
            raise IndexError(
                "The number of labels and the number of signals must be the same."
            )
        if len(y_data) != len(self._Dataset.dataset["OUTPUT"].values):
            raise IndexError(
                "The length of the input signal must be equal to the length "
                "of the other signals in the Dataset."
            )

    def plot_simulations(
        self,
        # Cam be a positional or a keyword arg
        list_sims: str | list[str] | None = None,
        dataset: Literal["in", "out", "both"] | None = None,
        layout: Literal["constrained", "compressed", "tight", "none"] = "tight",
        ax_height: float = 1.8,
        ax_width: float = 4.445,
    ) -> matplotlib.figure.Figure:
        """Plot the stored simulation results.

        Possible values of the parameters describing the plot aesthetics,
        such as the *linecolor_input* or the *alpha_output*,
        are the same for the corresponding *plot* function of *matplotlib*.

        You are free to manipulate the returned figure as you want by using any
        method of the class `matplotlib.figure.Figure`.

        Please, refer to *matplotlib* docs for more info.


        Example
        -------
        >>> fig = vs.plot_simulations() # ds is a dymoval ValidationSession object
        # The following are methods of the class `matplotlib.figure.Figure`
        >>> fig.set_size_inches(10,5)
        >>> fig.set_layout_engine("constrained")
        >>> fig.savefig("my_plot.svg")


        Parameters
        ----------
        list_sims:
            List of simulation names.
        dataset:
            Specify whether the dataset shall be datasetped to the simulations results.

            - **in**: dataset only the input signals of the dataset.
            - **out**: dataset only the output signals of the dataset.
            - **both**: dataset both the input and the output signals of the dataset.

        layout:
            Figure layout.
        ax_height:
            Approximative height (inches) of each subplot.
        ax_width:
            Approximative width (inches) of each subplot.
        """
        # TODO: could be refactored
        # It uses the left axis for the simulation results and the dataset output.
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
                    "Check the available simulations names with 'simulations_namess()'"
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
                    xlabel=f"{df_val.index.name[0]} ({df_val.index.name[1]})",
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
                    xlabel=f"{df_val.index.name[0]} ({df_val.index.name[1]})",
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
                    xlabel=f"{df_val.index.name[0]} ({df_val.index.name[1]})",
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
        :py:class:`ValidationSession <dymoval.validation.ValidationSession>` object.

        If not *tin* or *tout* are passed, then the selection is
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
            :py:meth:`ValidationSession
            <dymoval.validation.ValidationSession.plot_simulations>` method.

        """
        # This function is similar to Dataset.trim

        def _graph_selection(
            vs: Self,
            **kwargs: Any,
        ) -> tuple[float, float]:  # pragma: no cover
            # Select the time interval graphically
            # OBS! This part cannot be automatically tested because the it require
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
                    f"Updated time interval: {selection['tin']} to {selection['tout']}"
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
                f"\n tin = {tin_sel}{vs._Dataset.dataset.index.name[1]}, tout = {tout_sel}{vs._Dataset.dataset.index.name[1]}"
            )

        # Now you can trim the dataset and update all the
        # other time-related attributes
        vs._Dataset.dataset = vs._Dataset.dataset.loc[tin_sel:tout_sel, :]  # type: ignore[misc]
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
            vs._append_validation_results(sim_name)

        return vs

    def plot_residuals(
        self,
        list_sims: str | list[str] | None = None,
        *,
        plot_input: bool = True,
        layout: Literal["constrained", "compressed", "tight", "none"] = "tight",
        ax_height: float = 1.8,
        ax_width: float = 4.445,
    ) -> tuple[
        matplotlib.figure.Figure,
        matplotlib.figure.Figure,
        matplotlib.figure.Figure,
    ]:
        """Plot the residuals.

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


        You are free to manipulate the returned figure as you want by using any
        method of the class `matplotlib.figure.Figure`.

        Please, refer to *matplotlib* docs for more info.


        Example
        -------
        >>> fig = vs.plot_residuals() # vs is a dymoval ValidationSession object
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
                    "Check the available simulations names with 'simulations_namess()'"
                )
        Ruu = self._u_acorr_tensor
        Ree = self._eps_acorr_tensor
        Rue = self._ueps_xcorr_tensor

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
        Return the signal name list of a given simulation result.

        Parameters
        ----------
        sim_name :
            Simulation name.

        Raises
        ------
        KeyError
            If the requested simulation is not in the simulation list.
        """
        self._sim_list_validate()
        return list(self._simulations_values[sim_name].columns)

    def clear(self) -> Self:
        """Clear all the stored simulation results."""
        vs_temp = deepcopy(self)
        sim_names = vs_temp.simulations_names
        for x in sim_names:
            vs_temp = vs_temp.drop_simulations(x)
        return vs_temp

    def append_simulation(
        self,
        sim_name: str,
        y_names: list[str],
        y_data: np.ndarray,
    ) -> Self:
        """
        Append simulation results.
        The results are stored in the
        :py:attr:`<dymoval.validation.ValidationSession.simulations_values>` attribute.

        The validation statistics are automatically computed and stored in the
        :py:attr:`<dymoval.validation.ValidationSession.validation_results>` attribute.

        Parameters
        ----------
        sim_name :
            Simulation name.
        y_label :
            Simulation output signal names.
        y_data :
            Signal realizations expressed as `Nxq` 2D array of type *float*
            with `N` observations of `q` signals.
        """
        vs_temp = deepcopy(self)
        # df_sim = vs_temp.simulations_values

        y_names = obj2list(y_names)
        vs_temp._simulation_validation(sim_name, y_names, y_data)

        y_units = list(
            vs_temp._Dataset.dataset["OUTPUT"].columns.get_level_values("units")
        )

        # Initialize sim df
        df_sim = pd.DataFrame(data=y_data, index=vs_temp._Dataset.dataset.index)
        multicols = list(zip([sim_name] * len(y_names), y_names, y_units))
        df_sim.columns = pd.MultiIndex.from_tuples(
            multicols, names=["sim_names", "signal_names", "units"]
        )

        # Concatenate df_sim with the current sim results
        vs_temp._simulations_values = vs_temp.simulations_values.join(
            df_sim, how="right"
        ).rename_axis(df_sim.columns.names, axis=1)

        # Update residuals auto-correlation and cross-correlation attributes
        vs_temp._append_validation_results(sim_name)

        return vs_temp

    def drop_simulations(self, *sims: str) -> Self:
        """Drop simulation results from the validation session.


        Parameters
        ----------
        *sims :
            Name of the simulations to be dropped.
        """
        # Raises
        # ------
        # KeyError
        #     If the simulations list is empty.
        # ValueError
        #     If the simulation name is not found.

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

            vs_temp._eps_acorr_tensor.pop(sim_name)
            vs_temp._ueps_xcorr_tensor.pop(sim_name)

            vs_temp._validation_results = vs_temp._validation_results.drop(
                sim_name, axis=1
            )

        return vs_temp


def validate_models(
    measured_in: np.ndarray | List[Signal] | List[np.ndarray],
    measured_out: np.ndarray | List[Signal] | List[np.ndarray],
    simulated_out: np.ndarray | List[np.ndarray],
    sampling_period: float | None = None,
    U_bandwidths: np.ndarray | float | None = None,
    Y_bandwidths: np.ndarray | float | None = None,
    **kwargs: Any,
) -> ValidationSession:
    """Validate models based on measured and simulated data.

    sampling_period will retrieved from Signal id List[Signal], otherwise must
    be explicitely passed.
    # TODO retrieve information from Signals (sampling period, etc). Ignore
    # input
    """

    def _dummy_signal_list(
        dataset: np.ndarray,  # Must be 2D array
        sampling_period: float,
        kind: Literal["in", "out"],
    ) -> List[Signal]:
        # Input must be a 2D-array, of size (N,p)

        uy = "u" if kind == "in" else "y"
        signal_list = []
        for ii in range(dataset.shape[1]):
            tmp: Signal = {
                "name": f"{uy}{ii}",
                "samples": dataset[:, ii],  # Must be a 1D array
                "signal_unit": "NA",
                "sampling_period": sampling_period,
                "time_unit": "NA",
            }
            signal_list.append(deepcopy(tmp))
        return signal_list

    def _to_list_of_Signal(
        data: List[np.ndarray] | np.ndarray | List[Signal],
        sampling_period: float,
        kind: Literal["in", "out"],
    ) -> List[Signal]:
        # Convert a np.ndarray or a List[np.ndarray] to a List[Signal]
        # It performs some checks to the input data.

        # Scalar case: 1D np.ndarray: stack it into a column array
        if isinstance(data, np.ndarray) and data.ndim == 1:
            data_list = _dummy_signal_list(
                dataset=data[:, np.newaxis],
                sampling_period=sampling_period,
                kind=kind,
            )

        # Case 2D np.ndarray, columns are samples
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            data_list = _dummy_signal_list(
                dataset=data, sampling_period=sampling_period, kind=kind
            )

        # Case List[nd.array]
        elif isinstance(data, list) and all(
            isinstance(item, np.ndarray) for item in data
        ):
            # Check if all arrays have the same length
            lengths = [len(item) for item in data]
            if len(set(lengths)) == 1:  # All lengths should be the same
                data_list = _dummy_signal_list(
                    dataset=np.column_stack(data),  # type: ignore
                    sampling_period=sampling_period,
                    kind=kind,
                )
            else:
                raise ValueError("All arrays must have the same length")

        # Case List[Signal]

        elif isinstance(data, list) and all(
            isinstance(item, dict) and set(item.keys()) == set(SIGNAL_KEYS)
            for item in data
        ):
            # elif isinstance(data, list) and all(isinstance(item, dict) and set(item.keys()) == set(SIGNAL_KEYS), for all items in data):
            data_list = data  # type: ignore
        else:
            raise ValueError(
                "'measured_in' and 'measured_out' must be 2D-arrays, list of 1D-arrays, or List of Signals"
            )

        return data_list

    # ======== MAIN ================
    # Gather sampling_period from Signals
    if (
        isinstance(measured_out, list)
        and isinstance(measured_out[0], dict)
        and set(measured_out[0].keys()) == set(SIGNAL_KEYS)
    ):
        sampling_period = measured_out[0]["sampling_period"]
    elif sampling_period is None:
        raise TypeError("'sampling_period' missing")

    # Convert everything into List[Signal]
    measured_in_list = _to_list_of_Signal(
        data=measured_in, sampling_period=sampling_period, kind="in"
    )
    measured_out_list = _to_list_of_Signal(
        data=measured_out, sampling_period=sampling_period, kind="out"
    )

    # Check if simulated_data is in the correct format.
    N = measured_out_list[0]["samples"].shape[0]
    if isinstance(simulated_out, np.ndarray) and simulated_out.shape[0] != N:
        raise ValueError(
            "'simulated_out' shall be a "
            f"{N}x{len(measured_out_list)} np.ndarray "
            f"or a list of 1-D np.ndarray of length {len(measured_out_list)}"
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

    # Create a ValidationSession object
    # TODO: Check here
    vs_kwargs = kwargs.copy()
    vs = ValidationSession("quick & dirty", ds, **vs_kwargs)

    # If only one simulation is passed, put it into a list
    simulated_out_list = obj2list(simulated_out)
    for ii, sim in enumerate(simulated_out_list):
        sim_name = f"Sim_{ii}"
        vs = vs.append_simulation(
            sim_name=sim_name, y_names=output_labels, y_data=sim
        )

    return vs
