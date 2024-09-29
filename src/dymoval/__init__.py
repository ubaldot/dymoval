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
    ValidationSession,
    rsquared,
    whiteness_level,
)

from .config import COLORMAP

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
    "rsquared",
    "whiteness_level",
    "ValidationSession",
    "open_tutorial",
    "COLORMAP",
]
