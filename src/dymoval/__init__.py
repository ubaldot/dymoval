"""dymoval: analyze measurement data and validate models."""

from .dataset import Dataset
from .plotting import plot_compare, plot_dataset, plot_spectrum_compare
from .scope import (
    AmplitudeSpectrumScope,
    BaseScope,
    DatasetScope,
    SignalScope,
    SpectrumScope,
)
from .signal import SPECTRUM_MODES, Signal
from .utils import (
    difference_lists_of_str,
    factorize,
    is_interactive_shell,
    obj2list,
    open_tutorial,
)

__all__ = [
    # core
    "Signal",
    "Dataset",
    "SPECTRUM_MODES",
    # plotting
    "plot_dataset",
    "plot_compare",
    "plot_spectrum_compare",
    # scopes
    "BaseScope",
    "SignalScope",
    "DatasetScope",
    "SpectrumScope",
    "AmplitudeSpectrumScope",
    # utils
    "is_interactive_shell",
    "factorize",
    "difference_lists_of_str",
    "obj2list",
    "open_tutorial",
]
