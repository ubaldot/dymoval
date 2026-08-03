"""Config file."""

import pathlib
import shutil
import tomllib
import typing
from typing import Any, Literal, TypeAlias


def check_latex_installed() -> bool:
    """Check if LaTeX is installed."""
    return shutil.which("pdflatex") is not None


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

try:
    with open(
        pathlib.Path.home().joinpath(".dymoval/config.toml"), mode="rb"
    ) as fp:
        data = tomllib.load(fp)
    for k, val in data.items():
        config[mapping_dict[k]] = val
except FileNotFoundError:  # pragma: no cover
    pass


COLORMAP: Any = config["COLORMAP"]
ATOL: float = config["ATOL"]
IS_INTERACTIVE: bool | None = config["IS_INTERACTIVE"]

# Internal constants
# TODO: with python => 3.13 add "type" to define TypeAlias
XCorr_Statistic_type: TypeAlias = Literal[
    "mean", "quadratic", "std", "max", "abs_mean"
]
XCORR_STATISTIC_TYPE: list[XCorr_Statistic_type] = list(
    typing.get_args(XCorr_Statistic_type)
)

R2_Statistic_type: TypeAlias = Literal["mean", "min"]
