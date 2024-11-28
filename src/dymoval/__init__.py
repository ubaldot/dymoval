# Here you specify the public API
# __init__.py tell python that this is a package
from .config import COLORMAP
from .dataset import (
    Dataset,
    Signal,
    change_axes_layout,
    compare_datasets,
    plot_signals,
    validate_dataframe,
    validate_signals,
)
from .utils import (
    open_tutorial,
)
from .validation import (
    ValidationSession,
    XCorrelation,
    rsquared,
    whiteness_level,
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
    "rsquared",
    "whiteness_level",
    "ValidationSession",
    "open_tutorial",
    "COLORMAP",
]
