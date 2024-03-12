# -*- coding: utf-8 -*-
"""Module containing some useful functions."""

import numpy as np
from .config import *  # noqa
import scipy.signal as signal  # noqa
from typing import Any, Union
import sys
import os
import subprocess
from importlib import resources
import shutil
from pathlib import Path


def factorize(n: int) -> tuple[int, int]:
    r"""Find the smallest and closest integers *(a,b)* such that :math:`n \le ab`."""
    a = int(np.ceil(np.sqrt(n)))
    b = int(np.ceil(n / a))
    return a, b


def difference_lists_of_str(
    # Does it work only for strings?
    A: str | list[str],
    B: str | list[str],
) -> list[str]:
    r"""Return the strings contained in the list *A* but not in the list *B*.

    In set formalism, this function return a list representing the set difference
    :math:`A \backslash ( A \cap B)`.
    Note that the operation is not commutative.

    Parameters
    ----------
    A:
        list of strings A.
    B:
        list of strings B.
    """
    A = str2list(A)
    B = str2list(B)

    return list(set(A) - set(B))


def str2list(x: str | list[str]) -> list[str]:
    """
    Cast a *str* type to a *list[str]* type.

    If the input is alread a string, then it return it as-is.

    Parameters
    ----------
    x :
        Input string or list of strings.
    """
    if not isinstance(x, list):
        x = [x]
    return x


def _get_tutorial_files(dymoval_tutorial_folder: str) -> None:

    # Delete everything inside the destination folder if it exists
    if os.path.exists(dymoval_tutorial_folder):
        shutil.rmtree(dymoval_tutorial_folder)

    # Create the destination folder
    os.makedirs(dymoval_tutorial_folder, exist_ok=True)

    # Iterate over each file in the "tutorial" package and move it to the destination folder
    tutorial_contents = resources.contents("dymoval_tutorial")

    for content in tutorial_contents:
        if resources.is_resource("dymoval_tutorial", content):
            source_file = resources.files("dymoval_tutorial").joinpath(
                content
            )
            destination_file = os.path.join(dymoval_tutorial_folder, content)
            shutil.copyfile(str(source_file), destination_file)


def open_tutorial() -> Any:
    """Create a dymoval_tutorial folder containing all the files needed to
    run the tutorial. All you have to do is to run Jupyter notebook named
    dymoval_tutorial.ipynb. You need an app for opening .ipynb files.

    The content of the dymoval_tutorial folder will be overwritten every time this function is
    called.
    """

    home = str(Path.home())
    dymoval_tutorial_folder = os.path.join(home, "dymoval_tutorial")
    _get_tutorial_files(dymoval_tutorial_folder)

    if sys.platform == "win32":
        destination = dymoval_tutorial_folder + "\\dymoval_tutorial.ipynb"
        shell_process = subprocess.run(["explorer.exe", destination])
    else:
        destination = dymoval_tutorial_folder + "/dymoval_tutorial.ipynb"
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        shell_process = subprocess.run([opener, destination])

    return shell_process
