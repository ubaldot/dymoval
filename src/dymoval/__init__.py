"""dymoval: analyze measurement data and validate models."""

from .config import ATOL, COLORMAP
from .dataset import SIGNAL_KIND, Dataset
from .plotting import (
    plot_compare,
    plot_coverage_compare,
    plot_dataset,
    plot_signals,
    plot_spectrum_compare,
)
from .scope import (
    AmplitudeSpectrumScope,
    BaseScope,
    DatasetScope,
    SignalScope,
    SpectrumScope,
)
from .signal import SPECTRUM_MODES, Signal
from .statistics import compute_statistic, rsquared
from .utils import (
    difference_lists_of_str,
    factorize,
    is_interactive_shell,
    obj2list,
    open_tutorial,
)
from .validation import (
    VALIDATION_KEYS,
    ValidationSession,
    validate_models,
)
from .xcorrelation import XCorrelation, whiteness_level

__all__ = [
    # core
    "Signal",
    "Dataset",
    "SPECTRUM_MODES",
    "SIGNAL_KIND",
    # plotting
    "plot_signals",
    "plot_dataset",
    "plot_compare",
    "plot_coverage_compare",
    "plot_spectrum_compare",
    # scopes
    "BaseScope",
    "SignalScope",
    "DatasetScope",
    "SpectrumScope",
    "AmplitudeSpectrumScope",
    # validation
    "ValidationSession",
    "XCorrelation",
    "VALIDATION_KEYS",
    "validate_models",
    "compute_statistic",
    "rsquared",
    "whiteness_level",
    # utils
    "is_interactive_shell",
    "factorize",
    "difference_lists_of_str",
    "obj2list",
    "open_tutorial",
    # config
    "COLORMAP",
    "ATOL",
]
