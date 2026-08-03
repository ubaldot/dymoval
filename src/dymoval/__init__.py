"""dymoval: analyze measurement data and validate models."""

from .config import ATOL, COLORMAP
from .dataset import ALIGN_MODES, SIGNAL_KIND, AlignMode, Dataset, SignalKind
from .plotting import (
    plot_compare,
    plot_coverage_compare,
    plot_dataset,
    plot_signals,
    plot_spectrum_compare,
)
from .scope import (
    LAYOUTS,
    AmplitudeSpectrumScope,
    BaseScope,
    DatasetScope,
    Layout,
    SignalScope,
    SpectrumScope,
)
from .signal import (
    SCALES,
    SPECTRUM_MODES,
    SPECTRUM_SCALES,
    Scale,
    Signal,
    SpectrumMode,
    SpectrumScale,
)
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
    # type aliases and their runtime tuples
    "SpectrumMode",
    "SPECTRUM_MODES",
    "SignalKind",
    "SIGNAL_KIND",
    "Scale",
    "SCALES",
    "SpectrumScale",
    "SPECTRUM_SCALES",
    "AlignMode",
    "ALIGN_MODES",
    "Layout",
    "LAYOUTS",
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
