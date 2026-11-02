"""Public package interface for measurement analysis and model validation.

This module re-exports the stable classes, type aliases, statistics, plotting
helpers, and validation entry points intended for normal use. Implementation
details remain in their dedicated submodules.
"""

from .dataset import ALIGN_MODES, SIGNAL_KIND, AlignMode, Dataset, SignalKind
from .frf import FrequencyResponse
from .plotting import (
    plot_compare,
    plot_coverage_compare,
    plot_dataset,
    plot_signals,
    plot_spectrum_compare,
)
from .scope import (
    LAYOUTS,
    Layout,
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
from .statistics import compute_statistic, rmse, rsquared
from .utils import open_tutorial
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
    "FrequencyResponse",
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
    # validation
    "ValidationSession",
    "XCorrelation",
    "VALIDATION_KEYS",
    "validate_models",
    "compute_statistic",
    "rsquared",
    "rmse",
    "whiteness_level",
    # utils
    "open_tutorial",
]
