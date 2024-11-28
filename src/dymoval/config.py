"""Config file."""

import pathlib
import shutil
import typing
from typing import Any, Literal, TypeAlias

# Initialize the latex_installed variable
is_latex_installed = False


def check_latex_installed() -> bool:
    """Check if LaTeX is installed."""
    return shutil.which("pdflatex") is not None


# Set the variable based on the check
is_latex_installed = check_latex_installed()

# Constants exposed to the user: Defaults
config: dict[str, Any] = {
    "COLORMAP": "tab10",
    "ATOL": 1e-9,
    "IS_INTERACTIVE": None,
}

mapping_dict: dict[str, str] = {
    "color_map": "COLORMAP",
    "float_tolerance": "ATOL",
    "is_interactive": "IS_INTERACTIVE",
}

# TODO If you remove python 3.10 remove the dependency (also from pyproject) from tomli as tomlib is
# part of the standard python package starting from 3.11
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore

try:
    with open(
        pathlib.Path.home().joinpath(".dymoval/config.toml"), mode="rb"
    ) as fp:
        data = tomllib.load(fp)
    for k, val in data.items():
        config[mapping_dict[k]] = val
except FileNotFoundError:  # pragma: no cover
    pass


# locals().update(config)

COLORMAP: Any = config["COLORMAP"]
ATOL: float = config["ATOL"]
IS_INTERACTIVE: bool | None = config["IS_INTERACTIVE"]

# Internal constants
# TODO: with python => 3.13 add "type" to define TypeAlias
Signal_type: TypeAlias = Literal["INPUT", "OUTPUT"]
SIGNAL_KIND: list[Signal_type] = list(typing.get_args(Signal_type))

Spectrum_type: TypeAlias = Literal["amplitude", "power", "psd"]
SPECTRUM_KIND: list[Spectrum_type] = list(typing.get_args(Spectrum_type))

Allowed_keys_type: TypeAlias = Literal[
    "name", "samples", "signal_unit", "sampling_period", "time_unit"
]
SIGNAL_KEYS: list[Allowed_keys_type] = list(typing.get_args(Allowed_keys_type))

XCorr_Statistic_type: TypeAlias = Literal[
    "mean", "quadratic", "std", "max", "abs_mean"
]
XCORR_STATISTIC_TYPE: list[XCorr_Statistic_type] = list(
    typing.get_args(XCorr_Statistic_type)
)

R2_Statistic_type: TypeAlias = Literal["mean", "min"]
R2_STATISTIC_TYPE: list[R2_Statistic_type] = list(
    typing.get_args(R2_Statistic_type)
)
