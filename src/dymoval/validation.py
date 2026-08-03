"""The :class:`ValidationSession` class.

``ValidationSession`` orchestrates a :class:`dymoval.dataset.Dataset` and a
set of simulation results: it computes the validation statistics through
:mod:`dymoval.statistics` and :mod:`dymoval.xcorrelation` and it lays out
the validation plots.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Self

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from .config import (
    COLORMAP,
    R2_Statistic_type,
    XCorr_Statistic_type,
)
from .dataset import Dataset
from .scope import DatasetScope, Layout, _pick_time_interval, scope_subplots
from .signal import Signal
from .statistics import rsquared
from .utils import (
    difference_lists_of_str,
    factorize,
    obj2list,
)
from .xcorrelation import XCorrelation

__all__ = ["ValidationSession", "validate_models", "VALIDATION_KEYS"]


VALIDATION_KEYS: tuple[str, ...] = (
    "Ruu_whiteness",
    "r2",
    "Ree_whiteness",
    "Rue_whiteness",
)
"""Keys of the validation statistics and of the validation thresholds."""

_SIM_COLOR_FALLBACK = "gray"


def _stack(signals: Sequence[Signal]) -> np.ndarray:
    """Stack a sequence of signals into a ``N x p`` array."""
    return np.column_stack([sig.values for sig in signals])


@dataclass
class _XCorrSettings:
    r"""Everything driving one whiteness estimation.

    A :class:`ValidationSession` holds three of these, one per correlation
    (``Ruu``, ``Ree``, ``Rue``), which keeps the three otherwise identical
    blocks of logic from being written out three times.
    """

    name: str
    nlags: np.ndarray
    local_statistic: XCorr_Statistic_type
    global_statistic: XCorr_Statistic_type
    local_weights: np.ndarray | None
    global_weights: np.ndarray | None

    @classmethod
    def build(
        cls,
        name: str,
        nrows: int,
        ncols: int,
        default_nlags: int,
        nlags: np.ndarray | None,
        local_statistic: XCorr_Statistic_type,
        global_statistic: XCorr_Statistic_type,
        local_weights: np.ndarray | None,
        global_weights: np.ndarray | None,
    ) -> "_XCorrSettings":
        """Resolve the ``nrows x ncols`` number-of-lags array."""
        if nlags is not None:
            if (
                nlags.ndim != 2
                or nlags.shape[0] < nrows
                or nlags.shape[1] < ncols
            ):
                raise IndexError(
                    f"'{name}_nlags' shall be a {nrows}x{ncols} array."
                )

            resolved = np.asarray(nlags[0:nrows, 0:ncols])
        else:
            resolved = np.full((nrows, ncols), fill_value=default_nlags)

            if local_weights is not None:
                for ii in range(nrows):
                    for jj in range(ncols):
                        resolved[ii, jj] = len(local_weights[ii, jj])

        return cls(
            name=name,
            nlags=resolved,
            local_statistic=local_statistic,
            global_statistic=global_statistic,
            local_weights=local_weights,
            global_weights=global_weights,
        )

    @property
    def statistic_label(self) -> str:
        return f"{self.local_statistic}-{self.global_statistic}"

    def whiteness_of(self, R: XCorrelation) -> float:
        """Estimate the whiteness of ``R`` according to these settings."""
        estimate, _ = R.estimate_whiteness(
            local_statistic=self.local_statistic,
            local_weights=self.local_weights,
            global_statistic=self.global_statistic,
            global_weights=self.global_weights,
        )

        return estimate

    def _weights_str(self, kind: str, weights: np.ndarray | None) -> str:
        who = f"{self.name}_{kind}_weights"

        if weights is None:
            return f"{who}: None\n"

        return f"{who}: Yes (see self._{who})\n"

    def _nlags_str(self) -> str:
        flat = self.nlags.flatten()

        if np.all(flat == flat[0]):
            return f"num lags: {self.nlags[0, 0]}\n"

        return f"num lags: \n{self.nlags}\n"

    def summary(self, header: str) -> str:
        """The block describing these settings in the session ``__repr__``."""
        return (
            f"{header}\n"
            f"Statistic: {self.statistic_label}\n"
            + self._weights_str("local", self.local_weights)
            + self._weights_str("global", self.global_weights)
            + self._nlags_str()
        )


class ValidationSession:
    # TODO: Save validation session.
    r"""The *ValidationSession* class is used to validate models against a
    given dataset.

    A *ValidationSession* object is instantiated from a :ref:`Dataset` object.
    A validation session *name* shall be also provided.

    Multiple simulation results can be appended to the same
    *ValidationSession* instance, but for each *ValidationSession* instance
    only one :ref:`Dataset` object is considered.

    Simulation results are stored as
    :py:class:`~dymoval.signal.Signal` objects laid on the time vector of
    the validation dataset.

    Parameters
    ----------
    name:
        The `ValidationSession` object name.
    validation_dataset:
        The :py:class:`~dymoval.dataset.Dataset` object to be used for
        validation. It must have at least one input and one output.
    U_bandwidths:
        1-D array representing the bandwidths of each signal in the input U.
        `U_bandwidths[i]` corresponds to the bandwidth of signal `U[i]`.
    Y_bandwidths:
        1-D array representing the bandwidths of each signal in the output Y.
        `Y_bandwidths[i]` corresponds to the bandwidth of signal `Y[i]`.
    validation_thresholds:
        Thresholds used for validation. The `dict` keys shall be a subset of
        :py:data:`~dymoval.validation.VALIDATION_KEYS`.
    ignore_input:
        If `True` the input auto-correlation is not considered in the
        validation.
    r2_statistic:
        Statistic to be used for computing the global :math:`R^2` in case of
        multiple output signals.
    Ruu_nlags:
        Number of lags for the input auto-correlation array `Ruu`.
    Ruu_local_statistic_type:
        Statistic used for estimating the whiteness of each element of the
        :py:class:`~dymoval.xcorrelation.XCorrelation` object associated to the
        input signal.
    Ruu_global_statistic_type:
        Statistic used for estimating the overall whiteness of the resulting
        :math:`p\times p` matrix after the whiteness of each element of `Ruu`
        has been computed.
    Ruu_local_weights:
        Weights associated to each element of `Ruu`. It must be a
        :math:`p\times p` array where each element is a 1-D array.
    Ruu_global_weights:
        Weights associated to the resulting matrix after the local statistics
        for each element of `Ruu` have been computed. It must be a
        :math:`p\times p` array.
    Ree_nlags:
        Number of lags for the residuals auto-correlation array `Ree`.
    Ree_local_statistic_type:
        Statistic used for estimating the whiteness of each element of the
        :py:class:`~dymoval.xcorrelation.XCorrelation` object associated to the
        residuals auto-correlation.
    Ree_global_statistic_type:
        Statistic used for estimating the overall whiteness of the resulting
        :math:`q\times q` matrix after the whiteness of each element of `Ree`
        has been computed.
    Ree_local_weights:
        Weights associated to each element of `Ree`. It must be a
        :math:`q\times q` array where each element is a 1-D array.
    Ree_global_weights:
        Weights associated to the resulting matrix after the local statistics
        for each element of `Ree` have been computed. It must be a
        :math:`q\times q` array.
    Rue_nlags:
        Number of lags for the input-residuals cross-correlation array `Rue`.
    Rue_local_statistic_type:
        Statistic used for estimating the whiteness of each element of the
        :py:class:`~dymoval.xcorrelation.XCorrelation` object associated to the
        input-residuals cross-correlation.
    Rue_global_statistic_type:
        Statistic used for estimating the overall whiteness of the resulting
        :math:`p\times q` matrix after the whiteness of each element of `Rue`
        has been computed.
    Rue_local_weights:
        Weights associated to each element of `Rue`. It must be a
        :math:`p\times q` array where each element is a 1-D array.
    Rue_global_weights:
        Weights associated to the resulting matrix after the local statistics
        for each element of `Rue` have been computed. It must be a
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
        r2_statistic: R2_Statistic_type = "min",
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
        # validation dataset. Create another ValidationSession with another
        # validation dataset instead.

        # =============================================
        # Class attributes
        # =============================================
        if not validation_dataset.inputs or not validation_dataset.outputs:
            raise ValueError(
                "The validation dataset must have at least one input "
                "and one output."
            )

        self._Dataset: Dataset = validation_dataset

        # Number of inputs and outputs
        self._p = len(validation_dataset.inputs)
        self._q = len(validation_dataset.outputs)

        self._sampling_period = validation_dataset.get_sampling_period()

        self.name: str = name  # The validation session name.
        """ValidationSession object name."""

        self._default_nlags = 41

        # sim_name -> list of q Signal objects
        self._simulations: dict[str, list[Signal]] = {}
        """The appended simulation results.
        This attribute is automatically set through
        :py:meth:`~dymoval.validation.ValidationSession.append_simulation`
        and it should be considered as a *read-only* attribute."""

        # Format: 'name_sim': r2
        self._r2_statistic: R2_Statistic_type = r2_statistic

        # ------------------ Input --------------------
        self._U_bandwidths = U_bandwidths

        # Input nlags. Default 41 lags (20 negative and 20 positive)
        self._Ruu = _XCorrSettings.build(
            "Ruu",
            nrows=self._p,
            ncols=self._p,
            default_nlags=self._default_nlags,
            nlags=Ruu_nlags,
            local_statistic=Ruu_local_statistic_type,
            global_statistic=Ruu_global_statistic_type,
            local_weights=Ruu_local_weights,
            global_weights=Ruu_global_weights,
        )

        u_values = _stack(list(validation_dataset.inputs.values()))

        self._Ruu_tensor = XCorrelation(
            "Ruu",
            X=u_values,
            Y=u_values,
            nlags=self._Ruu.nlags,
            X_bandwidths=self._U_bandwidths,
            Y_bandwidths=self._U_bandwidths,
            sampling_period=self._sampling_period,
        )

        self._Ruu_whiteness = self._Ruu.whiteness_of(self._Ruu_tensor)

        # ------------ Residuals -----------------------------
        self._Y_bandwidths = Y_bandwidths

        # Residuals auto-correlation
        self._Ree_tensor: dict[str, XCorrelation] = {}

        self._Ree = _XCorrSettings.build(
            "Ree",
            nrows=self._q,
            ncols=self._q,
            default_nlags=self._default_nlags,
            nlags=Ree_nlags,
            local_statistic=Ree_local_statistic_type,
            global_statistic=Ree_global_statistic_type,
            local_weights=Ree_local_weights,
            global_weights=Ree_global_weights,
        )

        # Input-Residuals cross-correlation
        self._Rue_tensor: dict[str, XCorrelation] = {}

        self._Rue = _XCorrSettings.build(
            "Rue",
            nrows=self._p,
            ncols=self._q,
            default_nlags=self._default_nlags,
            nlags=Rue_nlags,
            local_statistic=Rue_local_statistic_type,
            global_statistic=Rue_global_statistic_type,
            local_weights=Rue_local_weights,
            global_weights=Rue_global_weights,
        )

        # sim_name -> {statistic_key: value}
        self._validation_statistics: dict[str, dict[str, float]] = {}

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

    # ====================================================
    # Representation
    # ====================================================
    def _statistics_labels(self) -> dict[str, str]:
        return {
            "Ruu_whiteness": f"Input whiteness ({self._Ruu.statistic_label})",
            "r2": "R-Squared (%)",
            "Ree_whiteness": (
                f"Residuals whiteness ({self._Ree.statistic_label})"
            ),
            "Rue_whiteness": (
                f"Input-Res whiteness ({self._Rue.statistic_label})"
            ),
        }

    def _statistics_table(self, keys: Sequence[str]) -> str:
        """Render the validation statistics as a plain-text table."""
        labels = self._statistics_labels()
        sims = self.simulations_names

        if not sims:
            return "(no simulation appended)"

        label_width = max(len(labels[k]) for k in keys)
        col_width = max(12, *(len(s) for s in sims))

        lines = [
            " " * label_width + "".join(f"  {s:>{col_width}}" for s in sims)
        ]

        for k in keys:
            values = "".join(
                f"  {self._validation_statistics[s][k]:>{col_width}.4f}"
                for s in sims
            )
            lines.append(f"{labels[k]:<{label_width}}{values}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        outcomes_head = "         "
        outcomes_body = "Outcome: "
        for k, v in self._outcome.items():
            delta = len(k) - len(v)
            if delta >= 0:
                outcomes_head += f"{k}  "
                outcomes_body += f"{v}" + " " * (delta + 2)
            else:
                outcomes_head += f"{k}" + " " * (delta + 2)
                outcomes_body += f"{v}"

        keys = list(VALIDATION_KEYS)

        if self._ignore_input:
            inputs_acorr_str = f"Input ignored: {self._ignore_input}\n"
            keys.remove("Ruu_whiteness")
        else:
            inputs_acorr_str = self._Ruu.summary("Inputs auto-correlation")

        thresholds_str = "".join(
            f"{k}: {v:.4f} \n" for k, v in self._validation_thresholds.items()
        )

        return (
            f"Validation session name: {self.name}\n\n"
            f"Validation setup:\n----------------\n"
            + inputs_acorr_str
            + "\n"
            + self._Ree.summary("Residuals auto-correlation:")
            + "\n"
            + self._Rue.summary("Input-residuals cross-correlation:")
            + "\n"
            + "Validation results:\n-------------------\n"
            "Thresholds: \n"
            f"{thresholds_str}\n"
            "Actuals:\n"
            f"{self._statistics_table(keys)}\n\n"
            f"{outcomes_head}\n"
            f"{outcomes_body}\n"
        )

    # ========== read-only attributes ====================

    @property
    def dataset(self) -> Dataset:
        """The reference :ref:`Dataset` object."""
        return self._Dataset

    @property
    def simulations(self) -> dict[str, list[Signal]]:
        """The stored simulations as lists of
        :py:class:`~dymoval.signal.Signal` objects."""
        return self._simulations

    @property
    def simulations_values(self) -> dict[str, np.ndarray]:
        """Simulated output values, one :math:`N\\times q` array per
        simulation."""
        return {
            name: _stack(signals)
            for name, signals in self._simulations.items()
        }

    @property
    def simulations_names(self) -> list[str]:
        """Names of the stored simulations."""
        return list(self._simulations)

    @property
    def outcome(self) -> dict[str, str]:
        """Validation outcome.

        For each simulation return the validation outcome.
        """
        return self._outcome

    @property
    def Ree(self) -> dict[str, XCorrelation]:
        """Residuals auto-correlation arrays."""
        return self._Ree_tensor

    @property
    def Ruu(self) -> XCorrelation:
        """Input auto-correlation array."""
        return self._Ruu_tensor

    @property
    def Rue(self) -> dict[str, XCorrelation]:
        """Input-residuals cross-correlation arrays."""
        return self._Rue_tensor

    @property
    def validation_thresholds(self) -> dict[str, float]:
        """Thresholds used to compute the PASS/FAIL outcome."""
        return self._validation_thresholds

    @validation_thresholds.setter
    def validation_thresholds(self, val: dict[str, float]) -> None:
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
    def validation_statistics(self) -> dict[str, dict[str, float]]:
        """Return the computed statistics for each simulation."""
        return self._validation_statistics

    def _get_validation_thresholds_default(
        self, ignore_input: bool
    ) -> dict[str, float]:
        validation_thresholds_default = {
            "Ruu_whiteness": 0.6,
            "r2": 35.0,
            "Ree_whiteness": 0.5,
            "Rue_whiteness": 0.5,
        }

        if ignore_input is True:
            del validation_thresholds_default["Ruu_whiteness"]

        return validation_thresholds_default

    def _compute_r2_statistic(
        self, r2_list: np.ndarray, statistic: R2_Statistic_type = "min"
    ) -> float:
        if statistic == "mean":
            return float(np.mean(r2_list))

        if statistic == "min":
            return float(np.min(r2_list))

        raise ValueError("'r2_statistic' must be 'mean' or 'min'")

    def _append_validation_statistics(
        self,
        sim_name: str,
    ) -> None:
        # Dataset values
        u_values = _stack(list(self._Dataset.inputs.values()))
        y_values = _stack(list(self._Dataset.outputs.values()))

        # Simulation results
        y_sim_values = _stack(self._simulations[sim_name])

        # Residuals
        eps = y_values - y_sim_values

        if np.allclose(eps, 0.0):
            raise ValueError(
                "Simulation outputs are identical to measured outputs. "
                "Are you cheating?"
            )

        r2 = self._compute_r2_statistic(
            rsquared(y_values, y_sim_values), self._r2_statistic
        )

        # Residuals auto-correlation
        Ree = XCorrelation(
            "Ree",
            eps,
            eps,
            X_bandwidths=self._Y_bandwidths,
            Y_bandwidths=self._Y_bandwidths,
            nlags=self._Ree.nlags,
            sampling_period=self._sampling_period,
        )

        # Input-residuals cross-correlation
        Rue = XCorrelation(
            "Rue",
            u_values,
            eps,
            X_bandwidths=self._Y_bandwidths,
            Y_bandwidths=self._Y_bandwidths,
            nlags=self._Rue.nlags,
            sampling_period=self._sampling_period,
        )

        self._Ree_tensor[sim_name] = Ree
        self._Rue_tensor[sim_name] = Rue

        self._validation_statistics[sim_name] = {
            "Ruu_whiteness": self._Ruu_whiteness,
            "r2": r2,
            "Ree_whiteness": self._Ree.whiteness_of(Ree),
            "Rue_whiteness": self._Rue.whiteness_of(Rue),
        }

        # Compute PASS/FAIL outcome
        statistics = self._validation_statistics[sim_name]
        local_outcome = []

        for k, threshold in self._validation_thresholds.items():
            if k == "r2":
                local_outcome.append(statistics[k] > threshold)
            else:
                local_outcome.append(statistics[k] < threshold)

        self._outcome[sim_name] = "PASS" if all(local_outcome) else "FAIL"

    def _sim_list_validate(self) -> None:
        if not self.simulations_names:
            raise KeyError(
                "The simulations list looks empty. "
                "Check the available simulation names with "
                "'simulations_names'"
            )

    def _sims_to_plot(self, list_sims: str | list[str] | None) -> list[str]:
        self._sim_list_validate()

        if not list_sims:
            return self.simulations_names

        sims = obj2list(list_sims)
        sim_not_found = difference_lists_of_str(sims, self.simulations_names)

        if sim_not_found:
            raise KeyError(
                f"Simulation {sim_not_found} not found. "
                "Check the available simulations names with "
                "'simulations_names'"
            )

        return sims

    def _simulation_validation(
        self, sim_name: str, y_names: Sequence[str], y_data: np.ndarray
    ) -> None:
        if len(y_names) != len(set(y_names)):
            raise ValueError("Signals name must be unique.")

        if sim_name in self.simulations_names:
            raise ValueError(
                f"Simulation name '{sim_name}' already exists. \n"
                "HINT: check the loaded simulations names with"
                "'simulations_names' method."
            )

        if len(set(y_names)) != self._q:
            raise IndexError(
                "The number of outputs of your simulation must be equal "
                "to the number of outputs in the dataset AND "
                "the name of each simulation output shall be unique."
            )

        if not isinstance(y_data, np.ndarray):
            raise ValueError(
                "The type the input signal values must be a numpy ndarray."
            )

        if len(y_names) not in y_data.shape:
            raise IndexError(
                "The number of labels and the number of signals "
                "must be the same."
            )

        if len(y_data) != len(self._Dataset.time()):
            raise IndexError(
                "The length of the input signal must be equal "
                "to the length "
                "of the other signals in the Dataset."
            )

    # ====================================================
    # Simulations bookkeeping
    # ====================================================
    def simulation_signals_list(
        self, sim_name: str | list[str]
    ) -> list[tuple[str, str]]:
        """
        Return the ``(name, unit)`` list of a given simulation.

        Parameters
        ----------
        sim_name :
            Simulation name.
        """
        self._sim_list_validate()

        name = sim_name if isinstance(sim_name, str) else sim_name[0]

        if name not in self._simulations:
            raise KeyError(f"Simulation '{name}' not found.")

        return [(sig.name, sig.unit or "") for sig in self._simulations[name]]

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
        y_names :
            Simulation output signal names.
        y_data :
            Simulated output expressed as :math:`N\times q` array
            with `N` observations of `q` signals.
        """
        vs_temp = deepcopy(self)

        y_names = obj2list(y_names)
        vs_temp._simulation_validation(sim_name, y_names, y_data)

        y_units = [sig.unit or "" for sig in vs_temp._Dataset.outputs.values()]
        time = vs_temp._Dataset.time()
        time_unit = vs_temp._Dataset.time_unit()

        values = np.asarray(y_data).reshape(len(time), len(y_names))

        vs_temp._simulations[sim_name] = [
            Signal(
                name=name,
                values=np.asarray(values[:, ii], dtype=float),
                time=time.copy(),
                unit=y_units[ii],
                time_unit=time_unit,
            )
            for ii, name in enumerate(y_names)
        ]

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

            vs_temp._simulations.pop(sim_name)
            vs_temp._Ree_tensor.pop(sim_name)
            vs_temp._Rue_tensor.pop(sim_name)
            vs_temp._validation_statistics.pop(sim_name)
            vs_temp._outcome.pop(sim_name, None)

        return vs_temp

    # ====================================================
    # Plots
    # ====================================================
    def plot_simulations(
        self,
        # Can be a positional or a keyword arg
        list_sims: str | list[str] | None = None,
        dataset: Literal["in", "out", "both"] | None = None,
        layout: Layout = "tight",
        ax_height: float = 1.8,
        ax_width: float = 4.445,
        with_scope: bool = True,
    ) -> matplotlib.figure.Figure:
        """Plot the stored simulation results.

        One subplot per output is created. The measured outputs and the
        measured inputs of the validation dataset can be overlaid through
        the `dataset` argument. When more inputs than outputs are available
        the extra inputs get their own subplot.

        Example
        -------
        >>> fig = vs.plot_simulations() # vs is a dymoval ValidationSession
        >>> fig.set_size_inches(10,5)
        >>> fig.savefig("my_plot.svg")

        Parameters
        ----------
        list_sims:
            List of simulation names.
        dataset:
            Specify whether the dataset shall be plotted.

            - *"in"*: plot only the input signals of the dataset.
            - *"out"*: plot only the output signals of the dataset.
            - *"both"*: plot both the input and the output signals of the
              dataset.

        layout:
            Figure layout.
        ax_height:
            Approximative height (inches) of each subplot.
        ax_width:
            Approximative width (inches) of each subplot.
        with_scope:
            If `True` an interactive
            :py:class:`~dymoval.scope.DatasetScope` is attached.
        """
        sims = self._sims_to_plot(list_sims)

        ds = self._Dataset
        p = self._p
        q = self._q

        plot_in = dataset in ("in", "both")
        plot_out = dataset in ("out", "both")

        n = max(p, q) if plot_in else q
        nrows, ncols = factorize(n)

        # ================================================================
        # Arrange the figure
        # ================================================================
        fig, axes, panel_ax = scope_subplots(
            nrows, ncols, with_scope=with_scope, squeeze=False
        )

        # Only the first "n" axes are used
        for ax in axes[n:]:
            ax.remove()
        axes = axes[:n]

        cmap = plt.get_cmap(COLORMAP)

        # ================================================================
        # Simulations
        # ================================================================
        for kk, sim in enumerate(sims):
            color = cmap(kk % cmap.N)

            for ii, sig in enumerate(self._simulations[sim]):
                sig._plot_standard(
                    ax=axes[ii],
                    color=color,
                    label=f"{sim}: {sig.name}",
                )

        # ================================================================
        # Measured outputs
        # ================================================================
        if plot_out:
            for ii, sig in enumerate(ds.outputs.values()):
                sig._plot_standard(
                    ax=axes[ii],
                    color=_SIM_COLOR_FALLBACK,
                )

        # ================================================================
        # Measured inputs
        # ================================================================
        if plot_in:
            for ii, sig in enumerate(ds.inputs.values()):
                if ii < q:
                    # Overlay on a twin axes to keep the scales separated.
                    # Lines drawn there are invisible to the scope, which
                    # only looks at the axes it was given.
                    ax = axes[ii].twinx()
                else:
                    # No need of a pair of axes for a single extra signal
                    ax = axes[ii]

                sig._plot_standard(
                    ax=ax,
                    color=_SIM_COLOR_FALLBACK,
                    linestyle="--",
                )

                if ii >= q:
                    ax.legend()

        for ax in axes:
            ax.grid(True)
            ax.legend()

        fig.suptitle("Simulations results.")

        fig_width_inches = ncols * ax_width
        fig_height_inches = nrows * ax_height + 1.25
        fig.set_size_inches(fig_width_inches, fig_height_inches)

        if with_scope:
            assert panel_ax is not None
            DatasetScope(fig, axes, panel_ax)
        else:
            fig.set_layout_engine(layout)

        return fig

    def plot_residuals(
        self,
        list_sims: str | list[str] | None = None,
        *,
        plot_input: bool = True,
        layout: Layout = "tight",
        ax_height: float = 1.8,
        ax_width: float = 4.445,
    ) -> tuple[matplotlib.figure.Figure, ...]:
        """Plot the residuals auto- and cross-correlation functions.

        It returns the input auto-correlation figure (only when `plot_input`
        is `True`), the residuals auto-correlation figure and the
        input-residuals cross-correlation figure.

        Parameters
        ----------
        list_sims :
            List of simulations. If empty, all the simulations are plotted.
        plot_input:
            Whether the input auto-correlation shall be plotted.
        layout:
            Figures layout.
        ax_height:
            Approximative height (inches) of each subplot.
        ax_width:
            Approximative width (inches) of each subplot.
        """
        sims = self._sims_to_plot(list_sims)

        p = self._p
        q = self._q

        cmap = plt.get_cmap(COLORMAP)
        figs: list[matplotlib.figure.Figure] = []

        def _new_figure(
            nrows: int, ncols: int, title: str, sharex: bool = False
        ) -> np.ndarray:
            fig, axes = plt.subplots(
                nrows, ncols, squeeze=False, sharex=sharex
            )
            fig.suptitle(title)
            fig.set_size_inches(ncols * ax_width, nrows * ax_height + 1.25)
            fig.set_layout_engine(layout)
            figs.append(fig)

            return axes

        def _plot_per_simulation(
            tensors: dict[str, XCorrelation],
            nrows: int,
            ncols: int,
            title: str,
            x_symbol: str,
            y_symbol: str,
            sharex: bool = False,
        ) -> None:
            """One grid, one color per simulation."""
            axes = _new_figure(nrows, ncols, title, sharex=sharex)

            for kk, sim_name in enumerate(sims):
                tensors[sim_name]._plot_grid(
                    axes,
                    x_symbol=x_symbol,
                    y_symbol=y_symbol,
                    label=sim_name,
                    linefmt=matplotlib.colors.to_hex(cmap(kk % cmap.N)),
                )

        # ===============================================================
        # Input auto-correlation
        # ===============================================================
        if plot_input:
            axes = _new_figure(p, p, "Input auto-correlation")
            # A single curve per subplot, already named by the subplot
            # title: a legend would only repeat it.
            self._Ruu_tensor._plot_grid(axes, x_symbol="u", y_symbol="u")

        # ===============================================================
        # Residuals auto-correlation
        # ===============================================================
        _plot_per_simulation(
            self._Ree_tensor, q, q, "Residuals auto-correlation", "eps", "eps"
        )

        # ===============================================================
        # Input-residuals cross-correlation
        # ===============================================================
        _plot_per_simulation(
            self._Rue_tensor,
            p,
            q,
            "Input-residuals cross-correlation",
            "u",
            "eps",
            sharex=True,
        )

        return tuple(figs)

    # ====================================================
    # Trim
    # ====================================================
    def trim(
        self: Self,
        tin: float | None = None,
        tout: float | None = None,
        verbosity: int = 0,
        **kwargs: Any,
    ) -> Self:
        """
        Trim the
        :py:class:`ValidationSession <dymoval.validation.ValidationSession>`
        object.

        If neither `tin` nor `tout` are passed, then the selection is
        made graphically.

        Parameters
        ----------
        tin :
            Initial time of the desired time interval.
        tout :
            Final time of the desired time interval.
        verbosity :
            Depending on its level, more or less info is displayed.
            The higher the value, the higher is the verbosity.
        **kwargs:
            kwargs to be passed to the
            :py:meth:`~dymoval.validation.ValidationSession.plot_simulations`
            method.
        """

        # =============================================
        # Trim ValidationSession main function
        # =============================================
        vs = deepcopy(self)
        time = vs._Dataset.time()

        if tin is None and tout is None:  # pragma: no cover
            tin_sel, tout_sel = _pick_time_interval(
                self.plot_simulations(**kwargs),
                float(time[0]),
                float(time[-1]),
                title="Trim the simulation results.",
                verbosity=verbosity,
            )
        else:
            tin_sel = float(time[0]) if tin is None else tin
            tout_sel = float(time[-1]) if tout is None else tout

        if verbosity != 0:
            print(
                f"\n tin = {tin_sel}{vs._Dataset.time_unit()} ",
                f" tout = {tout_sel}{vs._Dataset.time_unit()}",
            )

        # Trim the dataset ...
        vs._Dataset = vs._Dataset.trim(tin_sel, tout_sel)

        # ... and the simulations, on the very same grid.
        vs._simulations = {
            sim_name: [
                sig.trim(tin_sel, tout_sel, shift_to_zero=True)
                for sig in signals
            ]
            for sim_name, signals in vs._simulations.items()
        }

        for sim_name in vs.simulations_names:
            vs._append_validation_statistics(sim_name)

        return vs


def validate_models(
    measured_in: np.ndarray | Sequence[Signal],
    measured_out: np.ndarray | Sequence[Signal],
    simulated_out: np.ndarray | Sequence[np.ndarray],
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
        Signals sampling period. It is mandatory when the measurements are
        passed as `np.ndarray`.
    **kwargs:
        Keyword arguments passed to
        :py:class:`~dymoval.validation.ValidationSession` constructor.
    """

    def _dummy_signal_list(
        values: np.ndarray,  # Must be a 2D array
        sampling_period: float,
        kind: Literal["in", "out"],
    ) -> list[Signal]:
        uy_label = "u" if kind == "in" else "y"
        time = np.arange(values.shape[0]) * sampling_period

        return [
            Signal(
                name=f"{uy_label}{ii}",
                values=np.asarray(values[:, ii], dtype=float),
                time=time.copy(),
                unit="NA",
                time_unit="NA",
            )
            for ii in range(values.shape[1])
        ]

    def _to_list_of_Signal(
        data: np.ndarray | Sequence[Signal],
        sampling_period: float,
        kind: Literal["in", "out"],
    ) -> list[Signal]:
        if isinstance(data, np.ndarray):
            return _dummy_signal_list(
                values=data, sampling_period=sampling_period, kind=kind
            )

        if isinstance(data, Sequence) and all(
            isinstance(item, Signal) for item in data
        ):
            return list(data)

        raise ValueError(
            "'measured_in' and 'measured_out' must be 2D-arrays "
            "or list of Signals."
        )

    # ======== MAIN ================
    # Sanity check
    if isinstance(measured_in, np.ndarray) and measured_in.ndim != 2:
        raise IndexError("'measured_in' shall be a Nxp np.ndarray.")

    if isinstance(measured_out, np.ndarray) and measured_out.ndim != 2:
        raise IndexError("'measured_out' shall be a Nxq np.ndarray.")

    # Gather the sampling period from the Signals, when available
    if not isinstance(measured_out, np.ndarray) and all(
        isinstance(item, Signal) for item in measured_out
    ):
        sampling_period = measured_out[0].get_sampling_period()
    elif sampling_period is None:
        raise TypeError("'sampling_period' missing.")

    measured_in_list = _to_list_of_Signal(
        data=measured_in, sampling_period=sampling_period, kind="in"
    )
    measured_out_list = _to_list_of_Signal(
        data=measured_out, sampling_period=sampling_period, kind="out"
    )

    output_labels = [s.name for s in measured_out_list]

    ds = Dataset.from_signals(
        inputs=measured_in_list, outputs=measured_out_list
    )

    # Create a ValidationSession object placeholder
    vs = ValidationSession("quick & dirty", ds, **kwargs)

    # ---- Fix simulated_out arg -----
    simulated_out_list = (
        [simulated_out]
        if isinstance(simulated_out, np.ndarray)
        else list(simulated_out)
    )

    # Fetch the number of observations
    N = len(ds.time())

    if any(
        sim.ndim != 2
        or sim.shape[0] != N
        or sim.shape[1] != len(measured_out_list)
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
