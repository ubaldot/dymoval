# Here you specify the public API
# __init__.py tell python that this is a package
from .utils import (
    open_tutorial,
)
from .dataset import (
    Signal,
    Dataset,
    change_axes_layout,
    validate_signals,
    validate_dataframe,
    plot_signals,
    compare_datasets,
)

from .validation import (
    XCorrelation,
    xcorr,
    rsquared,
    acorr_norm,
    xcorr_norm,
    ValidationSession,
)

from .config import (
    NUM_DECIMALS,
    COLORMAP,
)

# Package public API
__all__ = [
    "Signal",
    "Dataset",
    "change_axes_layout",
    "validate_signals",
    "validate_dataframe",
    "plot_signals",
    "compare_datasets",
    "XCorrelation",
    "xcorr",
    "rsquared",
    "acorr_norm",
    "xcorr_norm",
    "ValidationSession",
    "open_tutorial",
    "NUM_DECIMALS",
    "COLORMAP",
]
