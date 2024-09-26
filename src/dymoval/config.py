"""Config file."""

import pathlib
from typing import Literal, cast, Any
import typing
import subprocess


# Initialize the latex_installed variable
latex_installed = False


def is_latex_installed() -> bool:
    """Check if LaTeX is installed by looking for pdflatex in PATH."""
    try:
        subprocess.run(
            ["pdflatex", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# Constants exposed to the user
config = {
    "NUM_DECIMALS": 4,
    "COLORMAP": "tab10",
}  # Defaults
mapping_dict = {
    "num_decimals": "NUM_DECIMALS",
    "color_map": "COLORMAP",
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

NUM_DECIMALS: int = cast(int, config["NUM_DECIMALS"])
COLORMAP: Any = config["COLORMAP"]

ATOL = 10**-NUM_DECIMALS

# Internal constants
Signal_type = Literal["INPUT", "OUTPUT"]
SIGNAL_KIND: list[Signal_type] = list(typing.get_args(Signal_type))

Spectrum_type = Literal["amplitude", "power", "psd"]
SPECTRUM_KIND: list[Spectrum_type] = list(typing.get_args(Spectrum_type))

Allowed_keys_type = Literal[
    "name", "samples", "signal_unit", "sampling_period", "time_unit"
]
SIGNAL_KEYS: list[Allowed_keys_type] = list(
    typing.get_args(Allowed_keys_type)
)

Statistic_type = Literal["mean", "quadratic", "std", "max"]
STATISTIC_TYPE: list[Statistic_type] = list(typing.get_args(Statistic_type))
