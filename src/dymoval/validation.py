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
    str2list,
)

from .dataset import Dataset, Signal, validate_signals
from typing import List, Tuple, Dict, Literal, Any
from dataclasses import dataclass

__all__ = [
    "XCorrelation",
    "rsquared",
    "whiteness_level",
    "ValidationSession",
]


@dataclass
class XCorrelation:
    # You have to manually write the type in TypedDicts docstrings
    # and you have to exclude them in the :automodule:
    """Type used to store MIMO cross-correlations.


    Attributes
    ----------
    values: np.ndarray
        Values of the correlation tensor.
        It is a *Nxpxq* tensor, where *p* is the dimension of the first signals,
        *q* is the dimension of the second signal and *N* is the number of lags.

    Return the normalized cross-correlation of two MIMO signals.

    If X = Y then it return the normalized auto-correlation of X.

    Parameters
    ----------
    name :
        The XCorrelation name.
    X :
        MIMO signal realizations expressed as `Nxp` 2D array
        of `N` observations of `p` signals.
    Y :
        MIMO signal realizations expressed as `Nxq` 2D array
        of `N` observations of `q` signals.
    """

    def __init__(
        self,
        name: str,
        X: np.ndarray,
        Y: np.ndarray | None = None,
        nlags: int | None = None,
        local_statistic: Statistic_type = "mean",
        local_weights: np.ndarray | None = None,  # shall be a 1D vector
        global_statistic: Statistic_type = "max",
        global_weights: np.ndarray | None = None,
    ) -> None:

        def _init_tensor(
            X: np.ndarray,
            Y: np.ndarray,
            nlags: int | None = None,
        ) -> Any:
            if X.ndim == 1:
                X = X.reshape(len(X), 1)
            if Y.ndim == 1:
                Y = Y.reshape(len(Y), 1)
            p = X.shape[1]
            q = Y.shape[1]

            # Input validation
            if global_weights is not None:
                val = (
                    "number_of_outputs * number_of_outputs"
                    if (np.allclose(X, Y) or Y is None)
                    else "number_of_inputs * number_of_outputs"
                )
                if global_weights.size != p * q:
                    raise ValueError(
                        f"Length of 'global_weights' must be equal to '{val}'"
                    )

            all_lags = signal.correlation_lags(len(X), len(Y))
            Rxy = np.zeros([len(all_lags), p, q])
            for ii in range(p):
                for jj in range(q):
                    # Classic correlation definition from Probability.
                    # Rxy = E[(X-mu_x)^T(Y-mu_y))]/(sigma_x*sigma_y),
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
                    # At the end, for each pair (ii,jj) you have Rxy = r_{x_ii,y_jj}(\tau), therefore
                    # for each (ii,jj) we compute a correlation.
                    Rxy[:, ii, jj] = signal.correlate(
                        (X[:, ii] - np.mean(X[:, ii])) / np.std(X[:, ii]),
                        (Y[:, jj] - np.mean(Y[:, jj])) / np.std(Y[:, jj]),
                    ) / min(len(X), len(Y))

            # Trim the cross-correlation results if the user wants less lags
            if local_weights is not None:
                num_lags = (
                    local_weights.size
                    if local_weights.size < all_lags.size
                    else all_lags.size
                )
            elif nlags is not None:
                num_lags = nlags if nlags < all_lags.size else all_lags.size
            else:
                num_lags = all_lags.size

            # Actual trim
            half_lags = num_lags // 2
            is_odd = 1 if num_lags % 2 == 1 else 0
            lags: np.ndarray = np.arange(-half_lags, half_lags + is_odd)
            mid_point = Rxy.shape[0] // 2
            Rxy = Rxy[
                mid_point - half_lags : mid_point + half_lags + is_odd,
                :,
                :,
            ]

            return Rxy, lags

        def _whiteness_level(
            local_statistic: Statistic_type = "mean",
            local_weights: np.ndarray | None = None,  # shall be a 1D vector
            global_statistic: Statistic_type = "max",
            global_weights: np.ndarray | None = None,
        ) -> Any:

            # MAIN whiteness level =================================
            R = self.values
            num_lags = R.shape[0]  # Number of lags
            nrows = R.shape[1]  # Number of rows
            ncols = R.shape[2]  # Number of columns
            lags0_idx = np.nonzero(self.lags == 0)[0][0]

            # Fix locals weights
            W_local = (
                np.ones(num_lags) if local_weights is None else local_weights
            )

            # fix global weights
            W_global = (
                np.ones(nrows * ncols)
                if global_weights is None
                else global_weights
            )

            # Build the R_matrix by computing the statistic of each scalar
            # cross-correlation (local)
            R_matrix = np.zeros((nrows, ncols))
            for ii in range(nrows):
                for jj in range(ncols):
                    if ii == jj and self.kind == "auto-correlation":
                        # Remove auto-correlation values at lag = 0
                        W = np.delete(W_local, lags0_idx)
                        rij_tau = np.delete(R[:, ii, jj], lags0_idx)
                    else:
                        W = W_local
                        rij_tau = R[:, ii, jj]

                    R_matrix[ii, jj] = compute_statistic(
                        statistic=local_statistic,
                        weights=W,
                        data=rij_tau,
                    )

            # Compute the overall statistic of the resulting matrix
            whiteness_level = compute_statistic(
                statistic=global_statistic,
                weights=W_global,
                data=R_matrix,
            )
            return whiteness_level, R_matrix

        # =========================================
        # Attributes
        # =========================================
        self.name: str = name
        if Y is None or np.array_equal(X, Y):
            self.values, self.lags = _init_tensor(X, X, nlags)
            self.kind = "auto-correlation"
        else:
            self.values, self.lags = _init_tensor(X, Y, nlags)
            self.kind = "cross-correlation"

        self._local_statistics = local_statistic
        self._local_weights = local_weights
        self._global_statistics = global_statistic
        self._global_weights = global_weights
        self.whiteness, self.R_matrix = _whiteness_level(
            local_statistic=local_statistic,
            local_weights=local_weights,
            global_statistic=global_statistic,
            global_weights=global_weights,
        )

        """Lags of the cross-correlation.
        It is a vector of length *N*, where *N* is the number of lags."""

    def __repr__(self) -> str:
        # Save existing settings
        # np_options = np.get_printoptions()
        # pd_options = pd.options.display.float_format

        # np.set_printoptions(precision=NUM_DECIMALS, suppress=True)
        # pd.options.display.float_format = lambda x: f"{x:.{NUM_DECIMALS}f}"
        repr_str = (
            f"XCorrelation tensor name: {self.name}\n"
            f"type: {self.kind}\n"
            f"local statistic: {self._local_statistics}\n"
            f"local weights: {self._local_weights}\n"
            f"global statistic: {self._global_statistics}\n"
            f"global weights: {self._global_weights}\n"
            f"num lags: {self.lags.size}\n"
            f"whiteness: {self.whiteness}\n"
        )

        # try:
        #     if self.values is not None:
        #         repr_str = repr(self.values)
        #     if self.whiteness is not None:
        #         repr_str = repr(self.whiteness)
        #     if self.R_matrix is not None:
        #         repr_str = repr(self.R_matrix)
        # finally:
        #     np.set_printoptions(**np_options)
        # pd.reset_option("display.-float_format")

        return repr_str

    def plot(self) -> matplotlib.figure.Figure:
        p = self.values.shape[1]
        q = self.values.shape[2]
        fig, ax = plt.subplots(p, q, sharex=True, squeeze=False)
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
                ax[ii, jj].plot(
                    self.lags,
                    self.values[:, ii, jj],
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
    statistic: Statistic_type,
    weights: np.ndarray,
    data: np.ndarray,
) -> Any:
    """DOCSTRING

    If data.shape dimension is greater than 1 it will be flatten to a 1D array.

    """

    if len(data.shape) > 1:
        data = data.flatten()

    if weights is None:
        weights = np.ones(data.size)

    if statistic == "quadratic":
        result = data.T @ np.diag(weights) @ data / np.sum(weights)
    elif statistic == "max":
        result = np.max(np.abs(weights.T * data)) / np.sum(
            weights
        )  # Use weights if weights is provided
    elif statistic == "mean":
        result = np.sum(weights * data) / np.sum(weights)
    elif statistic == "std":
        # Calculate weighted mean
        weighted_mean_tmp = np.sum(weights * data) / np.sum(
            weights
        )  # No need for data.size
        # Calculate weighted variance
        weighted_variance = np.sum(
            weights * (data - weighted_mean_tmp) ** 2
        ) / np.sum(
            weights
        )  # No need for data.size
        result = np.sqrt(weighted_variance)  # Standard deviation
    else:
        raise ValueError(f"'statistic' must be one of [{STATISTIC_TYPE}]")
    return result


def rsquared(x: np.ndarray, y: np.ndarray) -> float:
    """
    Return the :math:`R^2` value of two signals.

    Signals can be MIMO.

    Parameters
    ----------
    x :
        First input signal.
    y :
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


def whiteness_level(
    data: np.ndarray,
    nlags: int | None = None,
    local_statistic: Statistic_type = "mean",
    local_weights: np.ndarray | None = None,  # shall be a 1D vector
    global_statistic: Statistic_type = "max",
    global_weights: np.ndarray | None = None,
) -> tuple[np.floating, np.ndarray, XCorrelation]:
    # Convert signals into XCorrelation tensors and compute the
    # whiteness_level

    # TODO Input validation
    # Check that the global_weights length is equal to p * q

    # Compute correlation tensor from input signals
    if local_weights is None:
        num_lags = None
    elif nlags is None:
        num_lags = local_weights.shape[0]
    else:
        num_lags = nlags

    Rxx = XCorrelation(
        "",
        X=data,
        Y=None,
        nlags=num_lags,
        local_statistic=local_statistic,
        local_weights=local_weights,
        global_statistic=global_statistic,
        global_weights=global_weights,
    )

    return Rxx.whiteness, Rxx.R_matrix, Rxx


@dataclass
class ValidationSession:
    # TODO: Save validation session.
    """The *ValidationSession* class is used to validate models against a given dataset.

    A *ValidationSession* object is instantiated from a :ref:`Dataset` object.
    A validation session *name* shall be also provided.

    Multiple simulation results can be appended to the same *ValidationSession* instance,
    but for each ValidationSession instance only a :ref:`Dataset` object is condsidered.

    If the :ref:`Dataset` object changes,
    it is recommended to create a new *ValidationSession* instance.
    """

    def __init__(
        self,
        name: str,
        validation_dataset: Dataset,
        nlags: int | None = None,
        input_nlags: int | None = None,
        # input auto-correlation
        input_local_statistic_name_1st: Statistic_type = "mean",
        input_global_statistic_name_1st: Statistic_type = "max",
        input_local_statistic_name_2nd: Statistic_type = "max",
        input_global_statistic_name_2nd: Statistic_type = "max",
        input_local_weights: np.ndarray | None = None,
        input_global_weights: np.ndarray | None = None,
        # residuals auto-correlation
        acorr_local_statistic_name_1st: Statistic_type = "mean",
        acorr_global_statistic_name_1st: Statistic_type = "max",
        acorr_local_statistic_name_2nd: Statistic_type = "max",
        acorr_global_statistic_name_2nd: Statistic_type = "max",
        acorr_local_weights: np.ndarray | None = None,
        acorr_global_weights: np.ndarray | None = None,
        # input-residuals cross-Correlation
        xcorr_local_statistic_name_1st: Statistic_type = "mean",
        xcorr_global_statistic_name_1st: Statistic_type = "max",
        xcorr_local_statistic_name_2nd: Statistic_type = "max",
        xcorr_global_statistic_name_2nd: Statistic_type = "max",
        xcorr_local_weights: np.ndarray | None = None,
        xcorr_global_weights: np.ndarray | None = None,
    ) -> None:
        # Once you created a ValidationSession you should not change the validation dataset.
        # Create another ValidationSession with another validation dataset
        # By using the constructors, you should have no types problems because the check is done there.

        # =============================================
        # Class attributes
        # ============================================

        self._Dataset: Dataset = validation_dataset
        """The reference :ref:`Dataset` object."""

        # Simulation based
        self.name: str = name  #: The validation session name.

        self._simulations_values: pd.DataFrame = pd.DataFrame(
            index=validation_dataset.dataset.index, columns=[[], [], []]
        )
        """The appended simulation results.
        This attribute is automatically set through
        :py:meth:`~dymoval.validation.ValidationSession.append_simulation`
        and it should be considered as a *read-only* attribute."""

        self._auto_correlation_tensors: dict[str, XCorrelation] = {}
        """The auto-correlation tensors.
        This attribute is automatically set
        and it should be considered as a *read-only* attribute."""

        self._cross_correlation_tensors: dict[str, XCorrelation] = {}
        """The cross-correlation tensors.
        This attribute is automatically set
        and it should be considered as a *read-only* attribute."""

        # =========== Input management =======================
        if input_local_weights is not None:
            self._input_nlags = input_local_weights.size
        elif input_nlags is not None:
            self._input_nlags = input_nlags
        else:
            n = self._Dataset.dataset.shape[0]
            self._input_nlags = signal.correlation_lags(n, n).size

        self._input_local_statistic_name_1st = input_local_statistic_name_1st
        self._input_global_statistic_name_1st = input_global_statistic_name_1st
        self._input_local_statistic_name_2nd = input_local_statistic_name_2nd
        self._input_global_statistic_name_2nd = input_global_statistic_name_2nd
        self._input_local_weights = input_local_weights
        self._input_global_weights = input_global_weights

        Ruu_whiteness_1st, _, Ruu = whiteness_level(
            self._Dataset.dataset["INPUT"].to_numpy(),
            nlags=self._input_nlags,
            local_statistic=self._input_local_statistic_name_1st,
            local_weights=self._input_local_weights,
            global_statistic=self._input_global_statistic_name_1st,
            global_weights=self._input_global_weights,
        )
        Ruu_whiteness_2nd = whiteness_level(
            self._Dataset.dataset["INPUT"].to_numpy(),
            nlags=self._input_nlags,
            local_statistic=self._input_local_statistic_name_2nd,
            local_weights=self._input_local_weights,
            global_statistic=self._input_global_statistic_name_2nd,
            global_weights=self._input_global_weights,
        )[0]

        self._input_acorr_tensor = Ruu
        self._input_acorr_whiteness_1st = Ruu_whiteness_1st
        self._input_acorr_whiteness_2nd = Ruu_whiteness_2nd

        # nlags: pick the minumum of the lengths between acorr_local_weights and
        # xcorr_local_weights.
        # If not specifies, take the argument nlags
        # If nothing specified, then just pick the full length lags
        if acorr_local_weights is None and xcorr_local_weights is not None:
            self._nlags = xcorr_local_weights.size
        elif acorr_local_weights is not None and xcorr_local_weights is None:
            self._nlags = acorr_local_weights.size
        elif (
            acorr_local_weights is not None and xcorr_local_weights is not None
        ):
            self._nlags = min(
                acorr_local_weights.size, xcorr_local_weights.size
            )
        elif nlags is not None:
            self._nlags = nlags
        else:
            n = self._Dataset.dataset.shape[0]
            self._nlags = signal.correlation_lags(n, n).size
        self._acorr_local_statistic_name_1st = acorr_local_statistic_name_1st
        self._acorr_global_statistic_name_1st = acorr_global_statistic_name_1st
        self._acorr_local_statistic_name_2nd = acorr_local_statistic_name_2nd
        self._acorr_global_statistic_name_2nd = acorr_global_statistic_name_2nd
        self._acorr_local_weights = acorr_local_weights
        self._acorr_global_weights = acorr_global_weights

        self._xcorr_local_statistic_name_1st = xcorr_local_statistic_name_1st
        self._xcorr_global_statistic_name_1st = xcorr_global_statistic_name_1st
        self._xcorr_local_statistic_name_2nd = xcorr_local_statistic_name_2nd
        self._xcorr_global_statistic_name_2nd = xcorr_global_statistic_name_2nd
        self._xcorr_local_weights = xcorr_local_weights
        self._xcorr_global_weights = xcorr_global_weights

        # Initialize validation results DataFrame.
        idx = [
            f"Input whiteness ({self._input_local_statistic_name_1st}-{self._input_global_statistic_name_1st})",
            f"Input whiteness ({self._input_local_statistic_name_2nd}-{self._input_global_statistic_name_2nd})",
            "R-Squared (%)",
            f"Residuals whiteness ({self._acorr_local_statistic_name_1st}-{self._acorr_global_statistic_name_1st})",
            f"Residuals whiteness ({self._acorr_local_statistic_name_2nd}-{self._acorr_global_statistic_name_2nd})",
            f"Input-Res whiteness ({self._xcorr_local_statistic_name_1st}-{self._xcorr_global_statistic_name_1st})",
            f"Input-Res whiteness ({self._xcorr_local_statistic_name_2nd}-{self._xcorr_global_statistic_name_2nd})",
        ]
        self._validation_results: pd.DataFrame = pd.DataFrame(
            index=idx, columns=[]
        )
        """The validation results.
        This attribute is automatically set
        and it should be considered as a *read-only* attribute."""

    def __repr__(self) -> str:
        # Save existing settings
        # np_options = np.get_printoptions()
        # pd_options = pd.options.display.float_format

        # np.set_printoptions(precision=NUM_DECIMALS, suppress=True)
        # pd.options.display.float_format = lambda x: f"{x:.{NUM_DECIMALS}f}"

        repr_str = (
            f"Validation session name: {self.name}\n\n"
            f"Validation setup:\n----------------\n"
            f"Input\n"
            f"1st statistic: {self._input_local_statistic_name_1st}-{self._input_global_statistic_name_1st}\n"
            f"2nd statistic: {self._input_local_statistic_name_2nd}-{self._input_global_statistic_name_2nd}\n"
            f"local weights: {self._input_local_weights}\n"
            f"global weights: {self._input_global_weights}\n"
            f"num lags: {self._input_nlags}\n\n"
            f"Residuals auto-correlation:\n"
            f"1st statistic: {self._acorr_local_statistic_name_1st}-{self._acorr_global_statistic_name_1st}\n"
            f"2nd statistic: {self._acorr_local_statistic_name_2nd}-{self._acorr_global_statistic_name_2nd}\n"
            f"local weights: {self._acorr_local_weights}\n"
            f"global weights: {self._acorr_global_weights}\n"
            f"num lags: {self._nlags}\n\n"
            f"Input-residuals cross-correlation:\n"
            f"1st statistic: {self._xcorr_local_statistic_name_1st}-{self._xcorr_global_statistic_name_1st}\n"
            f"2nd statistic: {self._xcorr_local_statistic_name_2nd}-{self._xcorr_global_statistic_name_2nd}\n"
            f"local weights: {self._xcorr_local_weights}\n"
            f"global weights: {self._xcorr_global_weights}\n"
            f"num lags: {self._nlags}\n\n"
            f"Validation results:\n-------------------\n"
            f"{self._validation_results}\n"
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

    # ========== All attributes are read-only ====================
    @property
    def dataset(self) -> Dataset:
        return self._Dataset

    def simulations_values(self) -> pd.DataFrame:
        return self._simulations_values

    @property
    def simulations_names(self) -> list[str]:
        """Return a list of names of the stored simulations."""
        return list(self._simulations_values.columns.levels[0])

    # @property
    # def _input_acorr_tensors(self) -> XCorrelation:
    #     return self._input_acorr_tensor

    # @property
    # def _input_local_weights(self) -> np.ndarray | None:
    #     return self._input_local_weights

    # @property
    # def _input_global_weights(self) -> np.ndarray | None:
    #     return self._input_global_weights

    # @property
    # def input_nlags(self) -> int:
    #     return self._input_nlags

    # @property
    # def nlags(self) -> int:
    #     return self._nlags

    # @property
    # def _acorr_local_weights(self) -> np.ndarray | None:
    #     return self._acorr_local_weights

    # @property
    # def _acorr_global_weights(self) -> np.ndarray | None:
    #     return self._acorr_global_weights

    # @property
    # def _xcorr_local_weights(self) -> np.ndarray | None:
    #     return self._xcorr_local_weights

    # @property
    # def _xcorr_global_weights(self) -> np.ndarray | None:
    #     return self._xcorr_global_weights

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
        # Here I can do it at once
        Ree_whiteness_1st, _, Ree = whiteness_level(
            eps,
            nlags=self._nlags,
            local_statistic=self._acorr_local_statistic_name_1st,
            local_weights=self._acorr_local_weights,
            global_statistic=self._acorr_global_statistic_name_1st,
            global_weights=self._acorr_global_weights,
        )
        Ree_whiteness_2nd = whiteness_level(
            eps,
            nlags=self._nlags,
            local_statistic=self._acorr_local_statistic_name_2nd,
            local_weights=self._acorr_local_weights,
            global_statistic=self._acorr_global_statistic_name_2nd,
            global_weights=self._acorr_global_weights,
        )[0]
        # Here I cannot.
        Rue = XCorrelation(
            "Rue",
            u_values,
            eps,
            nlags=self._nlags,
            local_statistic=self._xcorr_local_statistic_name_1st,
            local_weights=self._xcorr_local_weights,
            global_statistic=self._xcorr_global_statistic_name_1st,
            global_weights=self._xcorr_global_weights,
        )

        Rue_whiteness_1st = Rue.whiteness
        Rue_whiteness_2nd = XCorrelation(
            "Rue",
            u_values,
            eps,
            nlags=self._nlags,
            local_statistic=self._xcorr_local_statistic_name_2nd,
            local_weights=self._xcorr_local_weights,
            global_statistic=self._xcorr_global_statistic_name_2nd,
            global_weights=self._xcorr_global_weights,
        ).whiteness

        self._auto_correlation_tensors[sim_name] = Ree
        self._cross_correlation_tensors[sim_name] = Rue
        self._validation_results[sim_name] = [
            self._input_acorr_whiteness_1st,
            self._input_acorr_whiteness_2nd,
            r2,
            Ree_whiteness_1st,
            Ree_whiteness_2nd,
            Rue_whiteness_1st,
            Rue_whiteness_2nd,
        ]

    def _sim_list_validate(self) -> None:
        if not self.simulations_names:
            raise KeyError(
                "The simulations list looks empty. "
                "Check the available simulation names with 'simulations_namess()'"
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
            list_sims = str2list(list_sims)
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
        sims = list(vs.simulations_names)
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
        vs._simulations_values = vs.simulations_values().loc[
            tin_sel:tout_sel, :  # type: ignore[misc]
        ]
        vs.simulations_values().index = vs._Dataset.dataset.index

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
            list_sims = str2list(list_sims)
            sim_not_found = difference_lists_of_str(
                list_sims, self.simulations_names
            )
            if sim_not_found:
                raise KeyError(
                    f"Simulation {sim_not_found} not found. "
                    "Check the available simulations names with 'simulations_namess()'"
                )
        Ruu = self._input_acorr_tensor
        Ree = self._auto_correlation_tensors
        Rue = self._cross_correlation_tensors

        # Get p
        k0 = list(Rue.keys())[0]
        p = Rue[k0].values[0, :, :].shape[0]

        # Get q
        k0 = list(Ree.keys())[0]
        q = Ree[k0].values[0, :, :].shape[0]

        # ===============================================================
        # Plot input auto-correlation
        # ===============================================================
        if plot_input:
            fig, ax = plt.subplots(p, p, sharex=True, squeeze=False)
            plt.setp(ax, ylim=(-1.2, 1.2))
            for ii in range(p):
                for jj in range(p):
                    if is_latex_installed:
                        title = rf"$\hat r_{{u_{ii}u_{jj}}}$"
                    else:
                        title = rf"r_u{ii}u_{jj}"
                    ax[ii, jj].plot(
                        Ruu.lags,
                        Ruu.values[:, ii, jj],
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
        fig1, ax1 = plt.subplots(q, q, sharex=True, squeeze=False)
        plt.setp(ax1, ylim=(-1.2, 1.2))
        for sim_name in list_sims:
            for ii in range(q):
                for jj in range(q):
                    if is_latex_installed:
                        title = rf"$\hat r_{{\epsilon_{ii}\epsilon_{jj}}}$"
                    else:
                        title = rf"r_eps{ii}eps_{jj}"
                    ax1[ii, jj].plot(
                        Ree[sim_name].lags,
                        Ree[sim_name].values[:, ii, jj],
                        label=sim_name,
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
        for sim_name in list_sims:
            for ii in range(p):
                for jj in range(q):
                    if is_latex_installed:
                        title = rf"$\hat r_{{u_{ii}\epsilon_{jj}}}$"
                    else:
                        title = rf"r_u{ii}eps{jj}"
                    ax2[ii, jj].plot(
                        Rue[sim_name].lags,
                        Rue[sim_name].values[:, ii, jj],
                        label=sim_name,
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

        y_names = str2list(y_names)
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
        vs_temp._simulations_values = (
            vs_temp.simulations_values()
            .join(df_sim, how="right")
            .rename_axis(df_sim.columns.names, axis=1)
        )

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
            vs_temp._simulations_values = vs_temp.simulations_values().drop(
                sim_name, axis=1, level="sim_names"
            )
            vs_temp.simulations_values().columns = (
                vs_temp.simulations_values().columns.remove_unused_levels()
            )

            vs_temp._auto_correlation_tensors.pop(sim_name)
            vs_temp._cross_correlation_tensors.pop(sim_name)

            vs_temp._validation_results = vs_temp._validation_results.drop(
                sim_name, axis=1
            )

        return vs_temp


def validate_models(
    dataset_in: np.ndarray | List[Signal] | List[np.ndarray],
    dataset_out: np.ndarray | List[Signal] | List[np.ndarray],
    sampling_period: float,
    *sims_out: np.ndarray | List[np.ndarray],
    validation_thresholds: Dict[str, float] | None = None,
) -> Tuple[List[Literal["PASS", "FAIL"]], ValidationSession, Dict[str, float]]:

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
            isinstance(item, np.ndarray) and item.ndim == 1 for item in data
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
                "'dataset_in' and 'dataset_out' must be 2D-arrays, list of 1D-arrays, or List of Signals"
            )

        return data_list

    # ======== MAIN ================
    if validation_thresholds is None:
        validation_thresholds_dict = {
            "Ruu_whiteness_1st": 0.35,
            "Ruu_whiteness_2nd": 0.5,
            "r2": 65,
            "Ree_whiteness_1st": 0.35,
            "Ree_whiteness_2nd": 0.55,
            "Rue_whiteness_1st": 0.35,
            "Rue_whiteness_2nd": 0.55,
        }
    elif isinstance(validation_thresholds, dict):
        validation_thresholds_dict = validation_thresholds
    else:
        raise TypeError("'validation thresholds' must be a dict")

    # Convert everything into List[Signal]
    dataset_in_list = _to_list_of_Signal(
        data=dataset_in, sampling_period=sampling_period, kind="in"
    )
    dataset_out_list = _to_list_of_Signal(
        data=dataset_out, sampling_period=sampling_period, kind="out"
    )

    # Build Dataset instance and Validation instance
    input_labels = [s["name"] for s in dataset_in_list]
    output_labels = [s["name"] for s in dataset_out_list]
    signal_list = dataset_in_list + dataset_out_list
    validate_signals(*signal_list)
    ds = Dataset(
        "dummy",
        signal_list,
        input_labels,
        output_labels,
        full_time_interval=True,
    )
    vs = ValidationSession("quick & dirty", ds)

    # Actual test
    global_outcome: List[Literal["PASS", "FAIL"]] = []
    for ii, sim in enumerate(sims_out):
        local_outcome = []
        sim = np.column_stack(sim) if isinstance(sim, list) else sim
        sim = sim[:, np.newaxis] if len(sim.shape) == 1 else sim
        sim_name = f"Sim_{ii}"
        vs = vs.append_simulation(
            sim_name=sim_name, y_names=output_labels, y_data=sim
        )
        validation_dict = vs.validation_values(sim_name)

        for k in validation_dict.keys():
            if k != "r2":
                local_outcome.append(
                    validation_dict[k] < validation_thresholds_dict[k]
                )
            else:
                local_outcome.append(
                    validation_dict[k] > validation_thresholds_dict[k]
                )
        if all(local_outcome):
            global_outcome.append("PASS")
        else:
            global_outcome.append("FAIL")

    return global_outcome, vs, validation_thresholds_dict
