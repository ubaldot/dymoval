# mypy: disable_error_code = ["name-defined"]
# mypy: show_error_codes
"""Module containing everything related to datasets.
Here are defined special datatypes, classes and auxiliary functions to deal with datasets.
"""


import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import io, fft
from .config import *  # noqa
from .utils import *  # noqa
from typing import TypedDict, Optional, Union, Any, Literal


class Signal(TypedDict):
    """Signals are used to represent real-world signals and are used for
    instantiate :py:class:`Dataset <dymoval.dataset.Dataset>` class objects.
    """

    # TO DO: it does not render np.ndarray
    name: str  #: Signal name.
    values: np.ndarray  #: Signal values.
    signal_unit: str  #: Signal unit.
    sampling_period: float  #: Signal sampling period.
    time_unit: str  #: Signal sampling period.


class Dataset:
    """The *Dataset* class stores the signals that you want to use as dataset
    and it provides methods for analyzing and manipulating them.

    A signal list shall be passed to the initializer along with a list of signal labels
    (i.e. signal names), in order to differentiate which signal(s) shall be considered
    as input and which signal(s) shall be considered as output.

    The signals list can be either a list
    of dymoval :py:class:`Signal <dymoval.dataset.Signal>` type or a
    pandas DataFrame with a well-defined structure.

    See :py:meth:`~dymoval.dataset.signals_validation` and
    :py:meth:`~dymoval.dataset.dataframe_validation` for more
    information.


    Parameters
    ----------
    name:
        Dataset name.
    signal_list :
        Signals to be included in the :py:class:`Dataset <dymoval.dataset.Dataset>`.
        See :py:meth:`~dymoval.dataset.signals_validation` and
        :py:meth:`~dymoval.dataset.dataframe_validation` to figure out how
        the list of :py:class:`Signal <dymoval.dataset.Signal>` or the pandas
        DataFrame representing the dataset signals shall look like.
    u_labels :
        List of input signal names. Each signal name must be unique and must be
        contained in the signal_list.
    y_labels :
        List of input signal names. Each signal name must be unique and must be
        contained in the signal_list.
    target_sampling_period :
        This parameter is passed to :py:meth:`~dymoval.dataset.fix_sampling_period`.
    plot_raw :
        If *True*, then the :py:class:`Signal <dymoval.dataset.Signal>`
        contained in the *signal_list* will be plotted.
        This parameter only have effect when the *signal_list* parameter is
        a list of :py:class:`Signal <dymoval.dataset.Signal>` objects.

    tin :
        Initial time instant of the Dataset.
    tout :
        Final time instant of the Dataset.
    full_time_interval :
        If *True*, the Dataset time interval will be equal to the longest
        time interval among all of the signals included in the *signal_list*
        parameter.
        This is overriden if the parameters *tin* and *tout* are specified.
    overlap :
        If *True* it will overlap the input and output signals plots during the
        dataset time interval selection.
    """

    def __init__(
        self,
        name: str,
        signal_list: Union[list[Signal], pd.DataFrame],
        u_labels: Union[str, list[str]],
        y_labels: Union[str, list[str]],
        target_sampling_period: Optional[float] = None,
        plot_raw: Optional[bool] = False,
        tin: Optional[float] = None,
        tout: Optional[float] = None,
        full_time_interval: Optional[bool] = False,
        overlap: Optional[bool] = False,
    ) -> None:

        if all(isinstance(x, dict) for x in signal_list):
            attr_sign = self._new_dataset_from_signals(
                signal_list,
                u_labels,
                y_labels,
                target_sampling_period,
                plot_raw,
                tin,
                tout,
                full_time_interval,
                overlap,
            )
            df, nan_intervals, excluded_signals, dataset_coverage = attr_sign
        # Initialization by pandas DataFrame
        elif isinstance(signal_list, pd.DataFrame):
            attr_df = self._new_dataset_from_dataframe(
                signal_list,
                u_labels,
                y_labels,
                tin,
                tout,
                full_time_interval,
                overlap,
            )
            df, nan_intervals, dataset_coverage = attr_df
            excluded_signals = []
        else:
            raise TypeError(
                "Input signals must be Signal or pandas DataFrame type. \n",
                "A Signal type is a dict with the following fields: \n\n",
                "name: str \n",
                "values: 1D numpy.ndarray \n",
                "sampling_period: float.\n\n ",
            )
        # ================================================
        # Class attributes
        # ================================================
        # NOTE: You have to use the #: to add a doc description
        # in class attributes (see sphinx.ext.autodoc)

        self.name: str = name  #: Dataset name.
        self.dataset: pd.DataFrame = df  #: Actual dataset.
        self.coverage: pd.DataFrame = dataset_coverage  # Docstring below,
        """Coverage statistics in terms of mean value and variance"""
        self.information_level: float = 0.0  #: *Not implemented yet!*
        self._nan_intervals: Any = nan_intervals
        # TODO: Check if _excluded_signals it can be removed.
        self._excluded_signals: list[str] = excluded_signals

    def __str__(self) -> str:
        return f"Dymoval dataset called '{self.name}'.\nTo see the stored values use the method 'get_dataset()'."

    # ============= NOT READY =======================
    # def __eq__(self, other):
    #     if not isinstance(other, Dataset):
    #         # don't attempt to compare against unrelated types
    #         return NotImplemented
    #     return all(self.dataset.columns == other.dataset.columns)
    # and self._nan_intervals == other._nan_intervals
    # ==============================================

    # ==============================================
    #   Class methods
    # ==============================================
    def _shade_input_nans(
        self,
        df: pd.DataFrame,
        NaN_intervals: dict[str, dict[str, list[np.ndarray]]],
        axes: matplotlib.axes.Axes,
        u_labels: list[str],
        color: Optional[str] = "b",
    ) -> None:

        # u_labels are forwarded from other functions and are already
        # converted into a list[str].
        for ii, ax in enumerate(axes):
            input_name = u_labels[ii]
            for idx, val in enumerate(NaN_intervals["INPUT"][input_name]):
                if not val.size == 0:
                    ax.axvspan(min(val), max(val), color=color, alpha=0.2)

    def _shade_output_nans(
        self,
        df: pd.DataFrame,
        NaN_intervals: dict[str, dict[str, list[np.ndarray]]],
        axes: matplotlib.axes.Axes,
        y_labels: list[str],
        color: Optional[str] = "g",
    ) -> None:

        # y_labels are forwarded from other functions and are already
        # converted into a list[str].
        for ii, ax in enumerate(axes):
            output_name = y_labels[ii]
            for idx, val in enumerate(NaN_intervals["OUTPUT"][output_name]):
                if not val.size == 0:
                    ax.axvspan(min(val), max(val), color=color, alpha=0.2)

    def _init_dataset_coverage(
        self, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:

        u_mean = df["INPUT"].mean(axis=0)
        u_cov = df["INPUT"].cov()
        y_mean = df["OUTPUT"].mean(axis=0)
        y_cov = df["OUTPUT"].cov()

        return u_mean, u_cov, y_mean, y_cov

    def _init_nan_intervals(
        self, df: pd.DataFrame
    ) -> dict[str, list[np.ndarray]]:
        # Find index intervals (i.e. time intervals) where columns values
        # are NaN.
        # Run an example in the tutorial to see an example on how they are stored.
        sampling_period = df.index[1] - df.index[0]
        NaN_index = {}
        NaN_intervals = {}
        for x in list(df.columns):
            NaN_index[x] = df.loc[df[x].isnull().to_numpy()].index
            idx = np.where(~np.isclose(np.diff(NaN_index[x]), sampling_period))[
                0
            ]
            NaN_intervals[x] = np.split(NaN_index[x], idx + 1)
        return NaN_intervals

    def _init_dataset_time_interval(
        self,
        df: pd.DataFrame,
        NaN_intervals: dict[str, dict[str, list[np.ndarray]]],
        tin: Optional[float],
        tout: Optional[float],
        overlap: Optional[bool],
        full_time_interval: Optional[bool],
    ) -> tuple[pd.DataFrame, dict[str, dict[str, list[np.ndarray]]]]:
        # We have to trim the signals to have a meaningful dataset
        # This can be done both graphically or by passing tin and tout
        # if the user knows them before hand.

        # Number of inputs and outputs
        p = len(df["INPUT"].columns)
        q = len(df["OUTPUT"].columns)

        # Check if there is some argument.
        if tin and tout:
            tin_sel = np.round(tin, NUM_DECIMALS)  # noqa
            tout_sel = np.round(tout, NUM_DECIMALS)  # noqa
        elif full_time_interval:
            tin_sel = np.round(df.index[0], NUM_DECIMALS)  # noqa
            tout_sel = np.round(
                df.index[-1], NUM_DECIMALS  # noqa
            )  # noqa, type: ignore
        else:
            #  ===========================================================
            # The following code is needed because not all IDE:s
            # have interactive plot set to ON as default.
            #
            is_interactive = plt.isinteractive()
            plt.ion()
            #  ===========================================================

            if overlap:
                n = max(p, q)
                range_out = np.arange(0, q)
            else:
                n = p + q
                range_out = np.arange(p, p + q)
            nrows, ncols = factorize(n)  # noqa
            _, axes = plt.subplots(nrows, ncols, sharex=True, squeeze=False)
            axes = axes.flat
            df["INPUT"].plot(subplots=True, grid=True, color="b", ax=axes[0:p])
            df["OUTPUT"].plot(
                subplots=True, grid=True, color="g", ax=axes[range_out]
            )

            for ii in range((ncols - 1) * nrows, ncols * nrows):
                axes[ii].set_xlabel("Time")
            plt.suptitle(
                "Sampling time "
                f"= {np.round(df.index[1]-df.index[0],NUM_DECIMALS)}.\n"  # noqa
                "Select the dataset time interval by resizing "
                "the picture."
            )

            # Shade NaN areas
            self._shade_input_nans(
                df, NaN_intervals, axes[0:p], list(df["INPUT"].columns)
            )
            self._shade_output_nans(
                df, NaN_intervals, axes[range_out], list(df["OUTPUT"].columns)
            )

            # Figure closure handler
            # It can be better done perhaps.
            def close_event(event):  # type:ignore
                time_interval = np.round(
                    axes[0].get_xlim(), NUM_DECIMALS  # noqa
                )  # noqa
                close_event.tin, close_event.tout = time_interval
                close_event.tin = max(close_event.tin, 0.0)
                close_event.tout = max(close_event.tout, 0.0)
                if is_interactive:
                    plt.ion()
                else:
                    plt.ioff()

            close_event.tin = 0.0  # type:ignore
            close_event.tout = 0.0  # type:ignore
            fig = axes[-1].get_figure()
            cid = fig.canvas.mpl_connect("close_event", close_event)
            fig.canvas.draw()
            plt.show()

            # =======================================================
            # This is needed for Spyder to block the prompt while
            # the figure is opened.
            # An alternative better solution is welcome!
            try:
                while fig.number in plt.get_fignums():
                    plt.pause(0.1)
            except:  # noqa
                plt.close(fig.number)
                raise
            # =======================================================

            fig.canvas.mpl_disconnect(cid)
            tin_sel = close_event.tin  # type:ignore
            tout_sel = close_event.tout  # type:ignore
        print("\ntin = ", tin_sel, "tout =", tout_sel)

        # ===================================================================
        # Trim dataset and NaN intervals based on (tin,tout)
        # ===================================================================
        # Trim dataset
        df = df.loc[tin_sel:tout_sel, :]
        # Trim NaN_intevals
        for u_name in NaN_intervals["INPUT"].keys():
            # In the following, nan_chunks are time-interval.
            # Note! For a given signal, you may have many nan_chunks.
            for idx, nan_chunk in enumerate(NaN_intervals["INPUT"][u_name]):
                nan_chunk = np.round(nan_chunk, NUM_DECIMALS)  # noqa
                NaN_intervals["INPUT"][u_name][idx] = nan_chunk[
                    nan_chunk >= tin_sel
                ]
                NaN_intervals["INPUT"][u_name][idx] = nan_chunk[
                    nan_chunk <= tout_sel
                ]
        for y_name in NaN_intervals["OUTPUT"].keys():
            for idx, nan_chunk in enumerate(NaN_intervals["OUTPUT"][y_name]):
                nan_chunk = np.round(nan_chunk, NUM_DECIMALS)  # noqa
                NaN_intervals["OUTPUT"][y_name][idx] = nan_chunk[
                    nan_chunk >= tin_sel
                ]
                NaN_intervals["OUTPUT"][y_name][idx] = nan_chunk[
                    nan_chunk <= tout_sel
                ]
        return df, NaN_intervals

    def _shift_dataset_time_interval(
        self,
        df: pd.DataFrame,
        NaN_intervals: dict[str, dict[str, list[np.ndarray]]],
    ) -> tuple[pd.DataFrame, dict[str, dict[str, list[np.ndarray]]]]:
        # ===================================================================
        # Shift tin to zero.
        # ===================================================================
        tin_sel = df.index[0]
        timeVectorFromZero = df.index - tin_sel
        df.set_index(
            np.round(timeVectorFromZero, NUM_DECIMALS), inplace=True  # noqa
        )  # noqa

        # Shift also the NaN_intervals to zero.
        for k in NaN_intervals["INPUT"].keys():
            for idx, nan_chunk in enumerate(NaN_intervals["INPUT"][k]):
                nan_chunk_translated = nan_chunk - tin_sel
                NaN_intervals["INPUT"][k][idx] = np.round(
                    nan_chunk_translated, NUM_DECIMALS  # noqa
                )
                NaN_intervals["INPUT"][k][idx] = nan_chunk_translated[
                    nan_chunk_translated >= 0.0
                ]
        for k in NaN_intervals["OUTPUT"].keys():
            for idx, nan_chunk in enumerate(NaN_intervals["OUTPUT"][k]):
                nan_chunk_translated = nan_chunk - tin_sel
                NaN_intervals["OUTPUT"][k][idx] = np.round(
                    nan_chunk_translated, NUM_DECIMALS  # noqa
                )
                NaN_intervals["OUTPUT"][k][idx] = nan_chunk_translated[
                    nan_chunk_translated >= 0.0
                ]
        # Adjust the DataFrame accordingly
        df = df.round(decimals=NUM_DECIMALS)  # noqa

        return df, NaN_intervals

    def _new_dataset_from_dataframe(
        self,
        df: pd.DataFrame,
        u_labels: Union[str, list[str]],
        y_labels: Union[str, list[str]],
        tin: Optional[float] = None,
        tout: Optional[float] = None,
        full_time_interval: Optional[bool] = False,
        overlap: Optional[bool] = False,
    ) -> tuple[
        pd.DataFrame,
        dict[str, dict[str, list[np.ndarray]]],
        tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame],
    ]:

        # ==============================================================
        # This is the Dataset initializer when the signals are arranged
        # in a pandas DataFrame
        # ==============================================================

        # Arguments validation
        if bool(tin) != bool(tout):
            raise Exception("You cannot pass only 'tin' or only 'tout'.")
        if tin and tout and tin > tout:
            raise ValueError(
                f" Value of tin ( ={tin}) shall be smaller than the value of tout ( ={tout})."
            )
        # If the user passes a str cast into a list[str]
        u_labels = str2list(u_labels)  # noqa
        y_labels = str2list(y_labels)  # noqa

        # TODO: Start here for adding units
        dataframe_validation(df, u_labels, y_labels)

        # Add column index level with labels 'INPUT' and 'OUTPUT'
        df = df.loc[:, [*u_labels, *y_labels]]
        u_extended_labels = list(zip(["INPUT"] * len(u_labels), u_labels))
        y_extended_labels = list(zip(["OUTPUT"] * len(y_labels), y_labels))
        df.columns = pd.MultiIndex.from_tuples(
            [*u_extended_labels, *y_extended_labels]
        )

        # Initialize NaN intervals
        NaN_intervals = {
            "INPUT": self._init_nan_intervals(df["INPUT"]),
            "OUTPUT": self._init_nan_intervals(df["OUTPUT"]),
        }
        # Trim dataset time interval
        df, NaN_intervals = self._init_dataset_time_interval(
            df, NaN_intervals, tin, tout, overlap, full_time_interval
        )
        # Shift dataset tin to 0.0.
        df, NaN_intervals = self._shift_dataset_time_interval(df, NaN_intervals)

        # Initialize coverage region
        dataset_coverage = self._init_dataset_coverage(df)
        return df, NaN_intervals, dataset_coverage

    def _new_dataset_from_signals(
        self,
        signal_list: list[Signal],
        u_labels: Union[str, list[str]],
        y_labels: Union[str, list[str]],
        target_sampling_period: Optional[float],
        plot_raw: Optional[bool],
        tin: Optional[float],
        tout: Optional[float],
        full_time_interval: Optional[bool],
        overlap: Optional[bool],
    ) -> tuple[
        pd.DataFrame,
        dict[str, dict[str, list[np.ndarray]]],
        list[str],
        tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame],
    ]:

        # If the user passed a string, we need to convert into list
        u_labels = str2list(u_labels)  # noqa
        y_labels = str2list(y_labels)  # noqa

        # Arguments validation
        signals_validation(signal_list)

        if plot_raw:
            plot_signals(signal_list, u_labels, y_labels)

        # Try to align the sampling periods, whenever possible
        # Note! resampled_signals:list[Signals], whereas
        # excluded_signals: list[str]
        resampled_signals, excluded_signals = fix_sampling_periods(
            signal_list, target_sampling_period
        )

        # Check that you you have at least one input and one output
        # after re-sampling.
        input_leftovers = [
            u for u in resampled_signals if u["name"] in u_labels
        ]
        output_leftovers = [
            y for y in resampled_signals if y["name"] in y_labels
        ]

        # Check that you don't have zero inputs or zero outputs
        if not input_leftovers or not output_leftovers:
            raise IndexError(
                "Re-sampling issue. "
                "The current 'target_sampling_period' would lead "
                "to a dataset with zero inputs or zero outputs."
            )

        # After fix_sampling_periods call all the signals in the resampled_signals
        # have the same sampling period.
        # Ts is the current sampling period for all the signals.
        Ts = list(resampled_signals)[0]["sampling_period"]

        # Drop excluded signals from u_labels and y_labels
        u_labels = [
            x["name"] for x in resampled_signals if x["name"] in u_labels
        ]
        y_labels = [
            x["name"] for x in resampled_signals if x["name"] in y_labels
        ]

        # Trim the signals to have equal length and
        # then build the DataFrame for inizializing the Dataset class.
        nsamples = [len(x["values"]) for x in resampled_signals]
        max_idx = min(nsamples)

        # Create DataFrame with trimmed and re-sampled signals
        df = pd.DataFrame(
            index=np.arange(max_idx) * Ts, columns=[*u_labels, *y_labels]
        )
        df.index.name = "Time"
        for s in resampled_signals:
            df[s["name"]] = s["values"][0:max_idx]

        # Call the initializer from dataframe to get a Dataset object.
        df, nan_intervals, dataset_coverage = self._new_dataset_from_dataframe(
            df, u_labels, y_labels, tin, tout, full_time_interval, overlap
        )
        return df, nan_intervals, excluded_signals, dataset_coverage

    def _signals_exist(
        self,
        u_labels: Optional[Union[str, list[str]]],
        y_labels: Optional[Union[str, list[str]]],
    ) -> tuple[list[str], list[str]]:
        # This function check if the signals (labels) from the user
        # exist in the current dataset.
        df = self.dataset
        input_columns_list = list(df["INPUT"].columns)  # Index type -> list.
        output_columns_list = list(df["OUTPUT"].columns)  # Index type -> list.

        # Input: check if the signal names exist
        if not u_labels:
            u_labels = input_columns_list
        else:
            u_labels = str2list(u_labels)  # noqa
            input_not_found = list_belonging_check(  # noqa
                u_labels, input_columns_list
            )

            if input_not_found:
                raise KeyError(
                    f"Signal(s) {input_not_found} not found in the input signals dataset. "
                    "Use 'get_signal_list()' to get the list of all available signals. "
                )

        # Output: check if the signal names exist
        if not y_labels:
            y_labels = output_columns_list
        else:
            y_labels = str2list(y_labels)  # noqa
            output_not_found = list_belonging_check(  # noqa
                y_labels, output_columns_list
            )

            if output_not_found:
                raise KeyError(
                    f"Signal(s) {output_not_found} not found in the output signal dataset. "
                    "Use 'get_signal_list()' to get the list of all available signals "
                )

        return u_labels, y_labels

    def get_dataset_values(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the dataset values as a tuple of *numpy ndarrays* corresponding
        to the tuple *(t,u,y)*.

        Returns
        -------
        t:
            The dataset time interval.
        u:
            The values of the input signal.
        y:
            The values of the output signal.
        """
        return (
            self.dataset.index.to_numpy(),
            self.dataset["INPUT"].to_numpy(),
            self.dataset["OUTPUT"].to_numpy(),
        )

    # def get_dataset(self) -> pd.DataFrame:
    #     """Return the dataset values in a pandas DataFrame."""
    #     return self.dataset

    # def get_region_covered(
    #     self,
    # ) -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    #     # TODO :examples
    #     """Return the dataset coverage in histograms.

    #     Returns
    #     -------
    #     input mean:
    #         Input(s) mean values.
    #     input covariance matrix:
    #         Input(s) covariance matrix.
    #     output mean:
    #         Outputs(s) mean values.
    #     output covariance matrix:
    #         Outputs(s) covariance matrix.
    #     """
    #     return self.coverage

    # def get_information_level(self) -> Any:
    #     """*To be implemented!*"""
    #     return print("To be implemented!")

    def export_to_mat(self, filename: str) -> None:
        """
        Write the dataset in a *.mat* file.

        Parameters
        ----------
        filename:
            Target filename. The extension *.mat* is automatically appended.
        """

        (t, u, y) = self.get_dataset_values()
        u_labels = list(self.dataset["INPUT"].columns)
        y_labels = list(self.dataset["OUTPUT"].columns)

        u_dict = {
            "INPUT": {
                s: self.dataset["INPUT"].loc[:, s].to_numpy() for s in u_labels
            }
        }
        y_dict = {
            "OUTPUT": {
                s: self.dataset["OUTPUT"].loc[:, s].to_numpy() for s in y_labels
            }
        }
        time = {"TIME": t}
        dsdict = time | u_dict | y_dict

        io.savemat(filename, dsdict, oned_as="column", appendmat=True)

    def get_signal_list(self) -> list[tuple[str, str]]:
        """Return the list of signal names of the dataset."""
        return list(self.dataset.columns)

    def plot(
        self,
        *,
        u_labels: Optional[Union[str, list[str]]] = None,
        y_labels: Optional[Union[str, list[str]]] = None,
        overlap: Optional[bool] = False,
        line_color_input: Optional[str] = "b",
        linestyle_input: str = "-",
        alpha_input: float = 1.0,
        line_color_output: str = "g",
        linestyle_output: str = "-",
        alpha_output: float = 1.0,
        return_figure: bool = False,
    ) -> Optional[tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]]:
        """Plot the Dataset.

        Possible values for the parameters describing the line used in the plot
        (e.g. *line_color_input* , *alpha_output*. etc)
        are the same for the corresponding plot function in matplotlib.

        Parameters
        ----------
        u_labels:
            List of input signals.
        y_labels:
            List of output signals.
        overlap:
            If true *True* overlaps the input and the output signals plots.
        line_color_input:
            Line color for the input signals.
        linestyle_input:
            Line style for the input signals.
        alpha_input:
            Alpha channel value for the input signals.
        line_color_output:
            Line color for the output signals.
        linestyle_output:
            Line style for the output signals.
        alpha_output:
            Alpha channel value for the output signals.
        return_figure:
            If *True* it returns the figure parameters.
        """
        # Extract dataset
        df = self.dataset

        u_labels, y_labels = self._signals_exist(u_labels, y_labels)

        # Input-output length
        p = len(u_labels)
        q = len(y_labels)

        if overlap:
            n = max(p, q)
            range_out = np.arange(0, q)
        else:
            n = p + q
            range_out = np.arange(p, p + q)
        nrows, ncols = factorize(n)  # noqa
        fig, axes = plt.subplots(nrows, ncols, sharex=True, squeeze=False)
        axes = axes.flat
        df["INPUT"].loc[:, u_labels].plot(
            subplots=True,
            grid=True,
            color=line_color_input,
            linestyle=linestyle_input,
            alpha=alpha_input,
            ax=axes[0:p],
        )
        df["OUTPUT"].loc[:, y_labels].plot(
            subplots=True,
            grid=True,
            color=line_color_output,
            linestyle=linestyle_output,
            alpha=alpha_output,
            ax=axes[range_out],
        )

        for ii in range((ncols - 1) * nrows, ncols * nrows):
            axes[ii].set_xlabel("Time")  # Fix with  df.index.name
        plt.suptitle("Blue lines are input and green lines are output. ")

        self._shade_input_nans(
            self.dataset,
            self._nan_intervals,
            axes[0:p],
            u_labels,
            color=line_color_input,
        )
        self._shade_output_nans(
            self.dataset,
            self._nan_intervals,
            axes[range_out],
            y_labels,
            color=line_color_output,
        )
        if return_figure:
            return fig, axes
        else:
            return None

    def plot_coverage(
        self,
        *,
        u_labels: Optional[Union[str, list[str]]] = None,
        y_labels: Optional[Union[str, list[str]]] = None,
        nbins: Optional[int] = 100,
        line_color_input: Optional[str] = "b",
        alpha_input: Optional[float] = 1.0,
        line_color_output: Optional[str] = "g",
        alpha_output: Optional[float] = 1.0,
        return_figure: Optional[bool] = False,
    ) -> Optional[
        tuple[
            matplotlib.figure.Figure,
            matplotlib.axes.Axes,
            matplotlib.figure.Figure,
            matplotlib.axes.Axes,
        ]
    ]:
        """
        Plot the dataset coverage as histograms.

        Parameters
        ----------
        u_labels:
            List of input signals.
        y_labels:
            List of output signals.
        nbins:
            The number of bins.
        line_color_input:
            Line color for the input signals.
        alpha_input:
            Alpha channel value for the input signals.
        line_color_output:
            Line color for the output signals.
        alpha_output:
            Alpha channel value for the output signals.
        return_figure:
            If *True* it returns the figure parameters.
        """
        # Extract dataset
        df = self.dataset

        u_labels, y_labels = self._signals_exist(u_labels, y_labels)

        p = len(u_labels)
        nrows, ncols = factorize(p)  # noqa
        fig_in, axes_in = plt.subplots(nrows, ncols, sharex=True, squeeze=False)
        axes_in = axes_in.flat
        df["INPUT"].hist(
            grid=True,
            bins=nbins,
            color=line_color_input,
            alpha=alpha_input,
            ax=axes_in[0:p],
        )

        for ii in range(p):
            axes_in[ii].set_xlabel(u_labels[ii][1])
        plt.suptitle("Coverage region (input).")

        q = len(y_labels)
        nrows, ncols = factorize(q)  # noqa
        fig_out, axes_out = plt.subplots(
            nrows, ncols, sharex=True, squeeze=False
        )
        axes_out = axes_out.flat
        df["OUTPUT"].hist(
            grid=True,
            bins=nbins,
            color=line_color_output,
            alpha=alpha_output,
            ax=axes_out[0:q],
        )

        for ii in range(q):
            axes_out[ii].set_xlabel(y_labels[ii][1])
        plt.suptitle("Coverage region (output).")
        if return_figure:
            return fig_in, axes_in, fig_out, axes_out
        else:
            return None

    def plot_amplitude_spectrum(
        self,
        *,
        u_labels: Optional[Union[str, list[str]]] = None,
        y_labels: Optional[Union[str, list[str]]] = None,
        overlap: Optional[bool] = False,
        line_color_input: Optional[str] = "b",
        linestyle_input: Optional[str] = "-",
        alpha_input: Optional[float] = 1.0,
        line_color_output: Optional[str] = "g",
        linestyle_output: Optional[str] = "-",
        alpha_output: Optional[float] = 1.0,
        return_figure: Optional[bool] = False,
    ) -> Optional[
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, pd.DataFrame]
    ]:
        """
        Plot the amplitude spectrum of the dataset and return the FFT of the
        specified signals.

        If some signals have *NaN* values, then the FFT cannot be computed and
        an error is raised.


        Parameters
        ----------
        u_labels:
            List of input signals.
            If not specified, the FFT is performed over all the input signals.
        y_labels:
            List of output signals.
            If not specified, the FFT is performed over all the output signals.
        overlap:
            If true it overlaps the input and the output signals plots.
        line_color_input:
            Line color for the input signals.
        linestyle_input:
            Line style for the input signals.
        alpha_input:
            Alpha channel value for the input signals.
        line_color_output:
            Line color for the output signals.
        linestyle_output:
            Line style for the output signals.
        alpha_output:
            Alpha channel value for the output signals.
        return_figure:
            If *True* it returns the figure parameters.

        Raises
        ------
        ValueError
            Signal(s) have *NaN*:s values.

        Returns
        -------
        fig :
            Figure of the plot.
        axes :
            Axes of the plot.
        df_freq :
            FFT of the dataset.
        """

        u_labels, y_labels = self._signals_exist(u_labels, y_labels)
        # Input-output length
        p = len(u_labels)
        q = len(y_labels)

        # Compute FFT
        Ts = self.dataset.index[1] - self.dataset.index[0]
        N = len(self.dataset.index)  # number of samples
        # Remove 'INPUT' 'OUTPUT' columns level from dataframe
        df_temp = self.dataset.droplevel(level=0, axis=1)
        vals = np.abs(fft.rfftn(df_temp.loc[:, [*u_labels, *y_labels]], axes=0))

        if np.any(np.isnan(vals)):
            raise ValueError(
                f"Dataset '{self.name}' contains NaN:s. I Cannot plot the amplitude spectrum."
            )
        # Frequency axis
        f_bins = fft.rfftfreq(N, Ts)

        # Create a new Dataframe
        u_extended_labels = list(zip(["INPUT"] * len(u_labels), u_labels))
        y_extended_labels = list(zip(["OUTPUT"] * len(y_labels), y_labels))
        cols = pd.MultiIndex.from_tuples(
            [*u_extended_labels, *y_extended_labels]
        )

        df_freq = pd.DataFrame(data=vals, columns=cols, index=f_bins)

        # Start plot ritual
        if overlap:
            n = max(p, q)
            range_out = np.arange(0, q)
        else:
            n = p + q
            range_out = np.arange(p, p + q)
        nrows, ncols = factorize(n)  # noqa
        fig, axes = plt.subplots(
            nrows, ncols, sharex=True, sharey=True, squeeze=False
        )
        axes = axes.flat
        df_freq["INPUT"].loc[:, u_labels].plot(
            subplots=True,
            grid=True,
            color=line_color_input,
            linestyle=linestyle_input,
            alpha=alpha_input,
            ax=axes[0:p],
        )
        df_freq["OUTPUT"].loc[:, y_labels].plot(
            subplots=True,
            grid=True,
            color=line_color_output,
            linestyle=linestyle_output,
            alpha=alpha_output,
            ax=axes[range_out],
        )

        for ii in range((ncols - 1) * nrows, ncols * nrows):
            axes[ii].set_xlabel("Frequency")  # Fix with  df.index.name
        plt.suptitle(
            "Amplitude spectrum. \n Blue lines are input and green lines are output. "
        )
        if return_figure:
            return fig, axes, df_freq
        else:
            return None

    def remove_means(
        self,
        *,
        u_labels: Optional[Union[str, list[str]]] = None,
        y_labels: Optional[Union[str, list[str]]] = None,
    ) -> None:
        """
        Remove the mean value to the specified signals.

        Parameters
        ----------
        u_labels :
            List of the input signal names.
            If not specified, then the mean value is removed to all
            the input signals in the dataset.

        y_labels :
            List of the output signal names.
            If not specified, then the mean value is removed to all
            the output signals in the dataset.
        """
        # Arguments validation
        u_labels, y_labels = self._signals_exist(u_labels, y_labels)

        cols = list(zip(len(u_labels) * ["INPUT"], u_labels))
        self.dataset.loc[:, cols] = (
            self.dataset.loc[:, cols] - self.dataset.loc[:, cols].mean()
        )

        cols = list(zip(len(y_labels) * ["OUTPUT"], y_labels))
        self.dataset.loc[:, cols] = (
            self.dataset.loc[:, cols] - self.dataset.loc[:, cols].mean()
        )

        # means_input = self.dataset["INPUT"]
        # vals = (
        #     self.dataset.droplevel(level=0, axis=1)
        #     .loc[:, [*u_labels, *y_labels]]
        #     .to_numpy()
        # )

        self.dataset = np.round(self.dataset, NUM_DECIMALS)  # noqa

    def remove_offset(
        self,
        *,
        u_list: list[tuple[str, float]],
        y_list: list[tuple[str, float]],
    ) -> None:
        """
        Remove a specified offset to the list of specified signals.

        For each target signal a tuple of the form *(name,value)*
        shall be passed.
        The value specified in the *offset* parameter is removed from
        the signal with name *name*.

        For multiple signals, the tuple shall be arranged in a list.

        Parameters
        ----------
        u_list:
            List of tuples of the form *(name, offset)*.
            The *name* parameter must match the name of any input signal stored
            in the dataset.
        y_list:
            List of tuples of the form *(name, offset)*.
            The *name* parameter must match the name of any output signal stored
            in the dataset.
        """
        if u_list:
            if not isinstance(u_list, list):
                u_list = [u_list]
            u_labels = [u[0] for u in u_list]
            u_labels, y_labels = self._signals_exist(u_labels, None)
        if y_list:
            if not isinstance(y_list, list):
                y_list = [y_list]
            y_labels = [y[0] for y in y_list]
            u_labels, y_labels = self._signals_exist(None, y_labels)
        if u_list and y_list:
            if not isinstance(u_list, list):
                u_list = [u_list]
            if not isinstance(y_list, list):
                y_list = [y_list]
            u_labels = [u[0] for u in u_list]
            y_labels = [y[0] for y in y_list]
            u_labels, y_labels = self._signals_exist(u_labels, y_labels)
        # Input
        if u_list:
            u_offset = [u[1] for u in u_list]
            cols = list(zip(len(u_labels) * ["INPUT"], u_labels))
            self.dataset.loc[:, cols] = self.dataset.loc[:, cols].apply(
                lambda x: x.subtract(u_offset), axis=1
            )
            # for ii, u in enumerate(u_labels):
            #     self.dataset[("INPUT", u)] -= u_offset[ii]
        # Output
        if y_list:
            y_offset = [y[1] for y in y_list]
            cols = list(zip(len(y_labels) * ["OUTPUT"], y_labels))
            self.dataset.loc[:, cols] = self.dataset.loc[:, cols].apply(
                lambda x: x.subtract(y_offset), axis=1
            )

    def low_pass_filter(
        self,
        *,
        u_list: Optional[Union[str, list[str]]] = None,
        y_list: Optional[Union[str, list[str]]] = None,
    ) -> None:
        """
        Low-pass filter a list of specified signals.

        For each target signal a tuple of the form *(name, cutoff_frequency)*
        shall be passed as parameter.
        For multiple signals, the tuples shall be arranged in a list.

        The low-pass filter is first-order IIR filter.

        Parameters
        ----------
        u_list:
            List of tuples of the form *(name, cutoff_frequency)*.
            The *name* parameter must match the name of any input signal stored
            in the dataset.
            Such a signal is filtered with a low-pass
            filter whose cutoff frequency is specified by the *cutoff_frequency*
            parameter.
        y_list:
            List of tuples of the form *(name, cutoff_frequency)*.
            The *name* parameter must match the name of any output signal stored
            in the dataset.
            Such a signal is filtered with a low-pass
            filter whose cutoff frequency is specified by the *cutoff_frequency*
            parameter.
        """
        # TODO: inplace implementation!
        if u_list:
            u_list = str2list(u_list)  # noqa
            u_labels = [u[0] for u in u_list]
            u_labels, y_labels = self._signals_exist(u_labels, None)
        if y_list:
            y_list = str2list(y_list)  # noqa
            y_labels = [y[0] for y in y_list]
            u_labels, y_labels = self._signals_exist(None, y_labels)
        if u_list and y_list:
            u_list = str2list(u_list)  # noqa
            y_list = str2list(y_list)  # noqa
            u_labels = [u[0] for u in u_list]
            y_labels = [y[0] for y in y_list]
            u_labels, y_labels = self._signals_exist(u_labels, y_labels)
        # Sampling frequency
        fs = 1 / (self.dataset.index[1] - self.dataset.index[0])
        N = len(self.dataset.index)

        # List of all the requested input cutoff frequencies
        # INPUT
        if u_list:
            u_fc = [u[1] for u in u_list]
            for ii, u in enumerate(u_labels):
                # Low-pass filter implementatiom
                fc = u_fc[ii]
                u_filt = self.dataset[("INPUT", u)].to_numpy()
                y_filt = np.zeros(N)
                y_filt[0]
                for kk in range(0, N - 1):
                    y_filt[kk + 1] = (1.0 - fc / fs) * y_filt[kk] + (
                        fc / fs
                    ) * u_filt[kk]
                self.dataset.loc[:, ("INPUT", u)] = y_filt
        # OUTPUT
        # List of all the requested input cutoff frequencies
        if u_list:
            y_fc = [y[1] for y in u_list]
            for ii, y in enumerate(y_labels):
                fc = y_fc[ii]  # cutoff frequency
                # Low-pass filter implementatiom
                u_filt = self.dataset[("OUTPUT", y)].to_numpy()
                y_filt = np.zeros(N)
                y_filt[0]
                for kk in range(0, N - 1):
                    y_filt[kk + 1] = (1.0 - fc / fs) * y_filt[kk] + (
                        fc / fs
                    ) * u_filt[kk]
                self.dataset.loc[:, ("OUTPUT", y)] = y_filt
        # Round value
        self.dataset = np.round(self.dataset, NUM_DECIMALS)  # noqa

    def filter(self) -> Any:
        """To be implemented!"""
        print("Not implemented yet!")

    def replace_NaNs(
        self,
        method: Literal["interpolate", "fillna"] = "interpolate",
        fill_value: float = 0.0,
    ) -> None:
        """Replace NaN:s in the dataset.

        Parameters
        ----------
        method :
            Interpolation method.
        fill_value :
            When *method* = "fillna", then the *NaN*:s vales are filled with this value.
        """

        if method == "interpolate":
            self.dataset.interpolate(axis=1, inplace=True)
        elif method == "fillna":
            self.dataset.fillna(fill_value, inplace=True)
        else:
            print("Unknown method. Choose between 'interpolate' or 'fillna'")


# ====================================================
# Useful functions
# ====================================================


def signals_validation(signal_list: list[Signal]) -> None:
    """
    Perform a number of checks to verify that the passed
    list of :py:class:`Signals <dymoval.dataset.Dataset>`
    can be used to create a Dataset.

    Every :py:class:`Signal <dymoval.dataset.Signal>` in the *signal_list*
    parameter must have all the attributes adequately set.

    To figure how the attributes shall be set, look at the *RAISES* section below.

    Parameters
    ----------
    signal_list :
        List of signal to be checked.

    Raises
    ------
    ValueError
        If signal names are not unique, or if values is not a *1-D numpy ndarray*,
        or if sampling period must positive.
    KeyError
        If signal attributes are not found or not allowed.
    IndexError
        If signal have less than two samples.
    TypeError
        If values is not a *1-D numpy ndarray*, or if sampling period is not a *float*.
    """

    # Name unicity
    signal_names = [s["name"] for s in signal_list]
    if len(signal_names) > len(set(signal_names)):
        raise ValueError("Signal names are not unique")
    #
    ALLOWED_KEYS = Signal.__required_keys__  # type:ignore
    for sig in signal_list:
        keys = sig.keys()
        #
        # Existence
        not_found_keys = list_belonging_check(  # noqa
            list(ALLOWED_KEYS), list(keys)
        )
        if not_found_keys:
            raise KeyError(
                f"Key {not_found_keys} not found in signal {sig['name']}."
            )
        for key in keys:
            if key not in ALLOWED_KEYS or key is None:
                raise KeyError(
                    f"Key '{key}' is not allowed."
                    " Keys must be one of '{ALLOWED_KEYS}'"
                )
            if key == "values":
                cond = (
                    not isinstance(sig[key], np.ndarray) or sig[key].ndim != 1  # type: ignore
                )
                if cond:
                    raise TypeError("Key {key} must be 1-D numpy array'.")
                if sig[key].size < 2:  # type: ignore
                    raise IndexError(
                        "Signal {sig[name']} has only one sample.",
                        "A signal must have at least two samples.",
                    )
            if key == "sampling_period":
                if not isinstance(sig[key], float):  # type: ignore
                    raise TypeError(
                        "Key 'sampling_period' must be a positive float."
                    )
                if sig[key] < 0.0 or np.isclose(sig[key], 0.0):  # type: ignore
                    raise ValueError(
                        "Key 'sampling_period' must be a positive float."
                    )


def dataframe_validation(
    df: pd.DataFrame,
    u_labels: Union[str, list[str]],
    y_labels: Union[str, list[str]],
) -> None:
    """
    Check if a pandas Dataframe is suitable for instantiating
    a :py:class:`Dataset <dymoval.dataset.Dataset>` object.

    When the signals are sampled with the same period, then they can be arranged
    in a pandas DataFrame, where the index represents the common time vector
    and each column represent a signal.

    Once the signals are arranged in a DataFrame,
    it must be specified which signal(s) are the input through the *u_labels* and
    which signal(s) is the output through the  *y_labels* parameters.

    Furthermore, the candidate DataFrame shall meet the following requirements

    - only one index and columns levels are allowed (no *MultiIndex*),
    - each column shall correspond to one signal,
    - the column names must be unique,
    - each signal must have at least two samples (i.e. the DataFrame has at least two rows),
    - both the index values and the column values must be *float*,
    - each signal in the *u_labels* and *y_labels* must be equal to exactly, one column name of the input DataFrame

    Parameters
    ----------
    df :
        DataFrame to be validated.
    u_labels :
        List of input signal names.
    y_labels :
        List of output signal names.

    Raises
    ------
    IndexError
        If there is not at least one input and output signal,
        or if there is any signal that don't have at least two samples,
        or if there is any *MultiIndex* index.
    ValueError
        If the signal names are not unique,
        or if the DataFrame *Index* (time vector) values are not monotonically increasing,
        or if some input or output signal names don't correspond to any
        DataFrame column name.
    TypeError
        If the DataFrame values are not all *float*.
    """
    # ===================================================================
    # Checks performed:
    # 1. There is at least one input and one output,
    # 2. Signals names are unique,
    # 3. The DataFrame must have only one index and columns levels,
    # 4. Signals shall have at least two samples.
    # 5. Input and output names exist in the passed DataFrame,
    # 6. The index is made of monotonically increasing floats,
    # 7. The datatype of the DataFrame's elements are float.
    # ===================================================================

    u_labels = str2list(u_labels)  # noqa
    y_labels = str2list(y_labels)  # noqa
    # 1. Check that you have at least one input and one output
    if not u_labels or not y_labels:
        raise IndexError(
            "You need at least one input and one output signal."
            "Check 'u_labels' and 'y_labels'."
        )
    # 2. Check unicity of signal names
    if (
        len(u_labels) != len(set(u_labels))
        or len(y_labels) != len(set(y_labels))
        or (set(u_labels) & set(y_labels))  # Non empty intersection
    ):
        raise ValueError(
            "Signal names must be unique." "Check 'u_labels' and 'y_labels'."
        )
    # 3. The DataFrame must have only one index and columns levels
    if df.columns.nlevels > 1 or df.index.nlevels > 1:
        raise IndexError(
            "The number index levels must be one for both the index and the columns.",
            "The index shall represent a time vector of equi-distant time instants",
            "and each column shall correspond to one signal values.",
        )
    # 4. At least two samples
    if df.index.size < 2:
        raise IndexError("A signal needs at least two samples.")
    # 5. Check if u_labels and y_labels exist in the passed DataFrame
    input_not_found = list_belonging_check(u_labels, list(df.columns))  # noqa
    if input_not_found:
        raise ValueError(f"Input(s) {input_not_found} not found.")
    output_not_found = list_belonging_check(y_labels, list(df.columns))  # noqa
    # Check output
    if output_not_found:
        raise ValueError(f"Output(s) {output_not_found} not found.")
    # 6. The index is a 1D vector of monotonically increasing floats.
    sampling_period = df.index[1] - df.index[0]
    if not np.all(np.isclose(np.diff(df.index), sampling_period)):
        raise ValueError(
            "Index must be a 1D vector of monotonically",
            "equi-spaced, increasing floats.",
        )
    # 7. The dataframe elements are all floats
    if not all(df.dtypes == float):
        raise TypeError("Elements of the DataFrame must be float.")


def fix_sampling_periods(
    signal_list: list[Signal],
    target_sampling_period: Optional[float] = None,
) -> tuple[list[Signal], list[str]]:
    # TODO: Implement some other re-sampling mechanism.
    """
    Resample the :py:class:`Signals <dymoval.dataset.Signal>` in the *signal_list*.

    The signals are resampled either with the slowest sampling period as target,
    or towards the sampling period specified by the *target_sampling_period*
    parameter, if specified.

    Nevertheless, signals whose sampling period is not a divisor of the
    the target sampling period will not be resampled and a list with the names
    of such signals is returned.

    Parameters
    ----------
    signal_list :
        List of :py:class:`Signals <dymoval.dataset.Signal>` to be resampled.
    target_sampling_period :
        Target sampling period.

    Raises
    ------
    ValueError
        If the *target_sampling_period* value is not positive.

    Returns
    -------
    resampled_signals:
        List of :py:class:`Signal <dymoval.dataset.Signal>` with adjusted
        sampling period.
    excluded_signals:
        List of signal names that could not be resampled.
    """
    # ===========================================================
    # arguments Validation
    signals_validation(signal_list)
    #
    if target_sampling_period:
        cond = (
            not isinstance(target_sampling_period, float)
            or target_sampling_period < 0
            or np.isclose(target_sampling_period, 0.0)
        )
        if cond:
            raise ValueError("'target_sampling_period' must be positive.")
    # ==========================================================

    # Initialization
    excluded_signals = []

    # Downsample to the slowest period if target_sampling_period
    # is not given.
    if not target_sampling_period:
        target_sampling_period = 0.0
        for sig in signal_list:
            target_sampling_period = max(
                target_sampling_period, sig["sampling_period"]
            )
            print(f"target_sampling_period = {target_sampling_period}")
    # Separate sigs
    for sig in signal_list:
        N = target_sampling_period / sig["sampling_period"]
        # Check if N is integer
        if np.isclose(N, round(N)):
            sig["values"] = sig["values"][:: int(N)]
            sig["sampling_period"] = target_sampling_period
        else:
            excluded_signals.append(sig["name"])
    resampled_signals = [
        sig
        for sig in list(signal_list)
        if not (sig["name"] in excluded_signals)
    ]
    print(
        "\nre-sampled signals =",
        f"{[sig['name'] for sig in resampled_signals]}",
    )
    print(
        "excluded signals from dataset =" f"{[sig for sig in excluded_signals]}"
    )
    print(f"actual sampling period = {target_sampling_period}")

    return resampled_signals, excluded_signals


def plot_signals(
    signal_list: list[Signal],
    u_labels: Optional[Union[str, list[str]]] = None,
    y_labels: Optional[Union[str, list[str]]] = None,
) -> None:
    """
    Plot the :py:class:`Signals <dymoval.dataset.Signal>` of a signal list.

    Parameters
    ----------
    signal_list :
        List of :py:class:`Dataset <dymoval.dataset.Dataset>`.
    u_labels :
        Used for specifying which signals shall be considered as input.
    y_labels : Optional[Union[str, list[str]]], optional
        Used for specifying which signals shall be considered as output.

    Raises
    ------
    KeyError
        If there is any label specified in *u_labels* or *y_labels* that does not
        correspond to any :py:class:`Dataset <dymoval.dataset.Dataset>` name.
    """
    # Validate signals first
    signals_validation(signal_list)
    signal_names = [sig["name"] for sig in signal_list]
    if u_labels:
        u_labels = str2list(u_labels)  # noqa
        inputs_not_found = list_belonging_check(u_labels, signal_names)  # noqa
        if inputs_not_found:
            raise KeyError(
                f"Signal(s) {inputs_not_found} not in the signals list. "
                f"Available signals are {signal_names}."
            )
    if y_labels:
        y_labels = str2list(y_labels)  # noqa
        outputs_not_found = list_belonging_check(y_labels, signal_names)  # noqa
        if outputs_not_found:
            raise KeyError(
                f"Signal(s) {outputs_not_found} not in the signals list. "
                f"Available signals are {signal_names}."
            )
    # Plot raw signals
    n = len(signal_list)
    nrows, ncols = factorize(n)  # noqa
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, sharex=True)
    ax = ax.flat
    for ii, sig in enumerate(signal_list):
        timeline = np.linspace(
            0.0, len(sig["values"]) * sig["sampling_period"], len(sig["values"])
        )
        if u_labels and sig["name"] in u_labels:
            line_color = "blue"
            sig_label = (sig["name"], "input")
        elif y_labels and sig["name"] in y_labels:
            line_color = "green"
            sig_label = (sig["name"], "output")
        else:
            line_color = "k"
            sig_label = (sig["name"], "")
        ax[ii].plot(timeline, sig["values"], label=sig_label, color=line_color)
        ax[ii].text(
            0.8,
            0.8,
            f"$T_s$ = {sig['sampling_period']} s",
            bbox=dict(facecolor="yellow", alpha=0.8),
        )
        ax[ii].grid()

        # Write time only in the last row
        if ii >= (nrows - 1) * ncols:
            ax[ii].set_xlabel("Time")
        ax[ii].legend()
    plt.suptitle("Raw signals.")
    # return fig, ax


def compare_datasets(*datasets: Dataset, target: str = "all") -> None:
    # arguments validation
    # for ds in datasets:
    #     if not isinstance(ds, Dataset):
    #         raise TypeError("Input must be a dymoval Dataset type.")

    ALLOWED_TARGETS = ["time", "freq", "cov", "all"]
    if target not in ALLOWED_TARGETS:
        raise ValueError(
            f"Target {target} not valid. Allowed targets are {ALLOWED_TARGETS}"
        )
    # Find the maximum number of axes.
    p_max = 0
    q_max = 0
    for ii, ds in enumerate(datasets):
        p_max = max(p_max, len(ds.dataset["INPUT"].columns))
        q_max = max(q_max, len(ds.dataset["OUTPUT"].columns))
    cmap = plt.get_cmap(COLORMAP)  # noqa

    # ================================================================
    # Start the plot. Note how idx work as a filter to select signals
    # in a certain position in all the simulations.
    # ================================================================

    # =================================================================
    #  Time
    # =================================================================
    if target in ["all", "time"]:
        nrows, ncols = factorize(p_max)  # noqa
        fig, axes = plt.subplots(nrows, ncols, sharex=True, squeeze=False)
        axes = axes.flat
        for ii, ds in enumerate(datasets):
            p = len(ds.dataset["INPUT"].columns)
            # Scan simulation names.
            ds.dataset["INPUT"].plot(
                subplots=True,
                grid=True,
                color=cmap(ii),
                ax=axes[0:p],
            )

            ds._shade_input_nans(
                ds.dataset,
                ds._nan_intervals,
                axes[0:p],
                list(ds.dataset["INPUT"].columns),
                color=cmap(ii),
            )

            for ii in range((ncols - 1) * nrows, ncols * nrows):
                axes[ii].set_xlabel("Time")  # Fix with  df.index.name
        plt.suptitle("Input signals comparison with respect to time. ")

        # Plot the output signals
        nrows, ncols = factorize(q_max)  # noqa
        fig, axes = plt.subplots(nrows, ncols, sharex=True, squeeze=False)
        axes = axes.flat
        for ii, ds in enumerate(datasets):
            q = len(ds.dataset["OUTPUT"].columns)
            # Scan simulation names.
            ds.dataset["OUTPUT"].plot(
                subplots=True,
                grid=True,
                color=cmap(ii),
                ax=axes[0:q],
            )

            ds._shade_output_nans(
                ds.dataset,
                ds._nan_intervals,
                axes[0:q],
                list(ds.dataset["OUTPUT"].columns),
                color=cmap(ii),
            )

            for ii in range((ncols - 1) * nrows, ncols * nrows):
                axes[ii].set_xlabel("Time")  # Fix with  df.index.name
        plt.suptitle("Output signals comparison with respect to time. ")
    # =================================================================
    #  Coverage
    # =================================================================
    if target in ["all", "cov"]:
        # Plot the input signals
        nrows, ncols = factorize(p_max)  # noqa
        fig, axes = plt.subplots(nrows, ncols, sharex=True, squeeze=False)
        axes = axes.flat
        for ii, ds in enumerate(datasets):
            p = len(ds.dataset["INPUT"].columns)
            # Scan simulation names.
            ds.dataset["INPUT"].hist(
                grid=True,
                color=cmap(ii),
                ax=axes[0:p],
            )

            for ii in range((ncols - 1) * nrows, ncols * nrows):
                axes[ii].set_xlabel("Time")  # Fix with  df.index.name
        plt.suptitle("Input signals comparison with respect to coverage. ")

        # Plot the output signals
        nrows, ncols = factorize(q_max)  # noqa
        fig, axes = plt.subplots(nrows, ncols, sharex=True, squeeze=False)
        axes = axes.flat
        for ii, ds in enumerate(datasets):
            q = len(ds.dataset["OUTPUT"].columns)
            # Scan simulation names.
            ds.dataset["OUTPUT"].hist(
                grid=True,
                color=cmap(ii),
                ax=axes[0:q],
            )

            for ii in range((ncols - 1) * nrows, ncols * nrows):
                axes[ii].set_xlabel("Time")  # Fix with  df.index.name
        plt.suptitle("Output signals comparison with respect to coverage. ")
    # =================================================================
    #  Frequency
    # =================================================================
    if target in ["all", "freq"]:
        # Inputs
        nrows, ncols = factorize(p_max)  # noqa
        fig, axes = plt.subplots(nrows, ncols, sharex=True, squeeze=False)
        axes = axes.flat

        for ii, ds in enumerate(datasets):
            u_labels = None
            y_labels = None
            u_labels, y_labels = ds._signals_exist(u_labels, y_labels)
            p = len(u_labels)
            # Compute FFT
            Ts = ds.dataset.index[1] - ds.dataset.index[0]
            N = len(ds.dataset.index)  # number of samples
            # Remove 'INPUT' 'OUTPUT' columns level from dataframe
            df_temp = ds.dataset.droplevel(level=0, axis=1)
            vals = np.abs(
                fft.rfftn(df_temp.loc[:, [*u_labels, *y_labels]], axes=0)
            )

            if np.any(np.isnan(vals)):
                warnings.warn(  # noqa
                    f"Dataset '{ds.name}' contains NaN:s. I Cannot plot the amplitude spectrum.".format(
                        ds.name, ds.name
                    )
                )
            # Frequency axis
            f_bins = fft.rfftfreq(N, Ts)

            # Create a new Dataframe
            u_extended_labels = list(zip(["INPUT"] * len(u_labels), u_labels))
            y_extended_labels = list(zip(["OUTPUT"] * len(y_labels), y_labels))
            cols = pd.MultiIndex.from_tuples(
                [*u_extended_labels, *y_extended_labels]
            )

            df_freq = pd.DataFrame(data=vals, columns=cols, index=f_bins)
            df_freq["INPUT"].loc[:, u_labels].plot(
                subplots=True,
                grid=True,
                color=cmap(ii),
                ax=axes[0:p],
            )

            for ii in range((ncols - 1) * nrows, ncols * nrows):
                axes[ii].set_xlabel("Frequency")  # Fix with  df.index.name
            plt.suptitle("Input signals frequency content.")
        # # Outputs
        nrows, ncols = factorize(q_max)  # noqa
        fig, axes = plt.subplots(nrows, ncols, sharex=True, squeeze=False)
        axes = axes.flat
        for ii, ds in enumerate(datasets):
            u_labels = None
            y_labels = None
            u_labels, y_labels = ds._signals_exist(u_labels, y_labels)
            q = len(y_labels)
            # Compute FFT
            Ts = ds.dataset.index[1] - ds.dataset.index[0]
            N = len(ds.dataset.index)  # number of samples
            # Remove 'INPUT' 'OUTPUT' columns level from dataframe
            df_temp = ds.dataset.droplevel(level=0, axis=1)
            vals = np.abs(
                fft.rfftn(df_temp.loc[:, [*u_labels, *y_labels]], axes=0)
            )

            if np.any(np.isnan(vals)):
                warnings.warn(  # noqa
                    f"Dataset '{ds.name}' contains NaN:s. I Cannot plot the amplitude spectrum."
                )
            # Frequency axis
            f_bins = fft.rfftfreq(N, Ts)

            # Create a new Dataframe
            u_extended_labels = list(zip(["INPUT"] * len(u_labels), u_labels))
            y_extended_labels = list(zip(["OUTPUT"] * len(y_labels), y_labels))
            cols = pd.MultiIndex.from_tuples(
                [*u_extended_labels, *y_extended_labels]
            )

            df_freq = pd.DataFrame(data=vals, columns=cols, index=f_bins)
            df_freq["OUTPUT"].loc[:, y_labels].plot(
                subplots=True,
                grid=True,
                color=cmap(ii),
                ax=axes[0:q],
            )

            for ii in range((ncols - 1) * nrows, ncols * nrows):
                axes[ii].set_xlabel("Frequency")  # Fix with  df.index.name
            plt.suptitle("Outputs amplitude spectrum.")


# def analyze_inout_dataset(df):
# -	Remove trends. Cannot do.
# -	(Resample). One line of code with pandas