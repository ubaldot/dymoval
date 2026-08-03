"""The :class:`XCorrelation` class.

``XCorrelation`` is the cross-correlation of two, possibly MIMO, signals.
Like :class:`dymoval.signal.Signal` it owns its own computation and its own
primitive plotting; the orchestration belongs to
:class:`dymoval.validation.ValidationSession`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import matplotlib
import numpy as np
import scipy.signal as signal
from matplotlib import pyplot as plt

from .config import (
    XCORR_STATISTIC_TYPE,
    XCorr_Statistic_type,
    is_latex_installed,
)
from .statistics import compute_statistic
from .utils import is_interactive_shell

__all__ = ["XCorrelation", "whiteness_level"]


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
            f"name: {self.name}\ntype: {self.kind}\nR shape: {self.R.shape}\n"
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
            :py:attr:`~dymoval.xcorrelation.XCorrelation.R` array.

        local_weights:
            Weights associated with the value of each `(i, j)` element of
            :py:attr:`~dymoval.xcorrelation.XCorrelation.R`.
            It must have the same shape of
            :py:attr:`~dymoval.xcorrelation.XCorrelation.R`.

        global_statistic:
            Statistic used to estimate the whiteness of the flattened
            :math:`p \times q` array after the whiteness of each element of
            :py:attr:`~dymoval.xcorrelation.XCorrelation.R` is estimated.

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
            :py:attr:`~dymoval.xcorrelation.XCorrelation.R`.

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
        in :py:attr:`~dymoval.xcorrelation.XCorrelation.R`."""

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
    :py:meth:`~dymoval.statistics.compute_statistic`.

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
        :py:attr:`~dymoval.xcorrelation.XCorrelation.R`.

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
