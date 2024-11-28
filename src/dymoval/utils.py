# -*- coding: utf-8 -*-
"""Module containing some useful functions."""

import os
import shutil
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from .config import IS_INTERACTIVE

__all__ = [
    "is_interactive_shell",
    "factorize",
    "difference_lists_of_str",
    "obj2list",
    "open_tutorial",
]


def is_interactive_shell() -> bool:
    if IS_INTERACTIVE is None:
        isinteractive = False
        try:
            from IPython import get_ipython

            if get_ipython():
                isinteractive = True
        except ImportError:
            pass
        if hasattr(sys, "ps1"):
            isinteractive = True
        return isinteractive
    else:
        return IS_INTERACTIVE


def factorize(n: int) -> tuple[int, int]:
    r"""
    Find the smallest and closest integers *(a,b)* such that :math:`n \le ab`.

    """
    a = int(np.ceil(np.sqrt(n)))
    b = int(np.ceil(n / a))
    return a, b


def difference_lists_of_str(
    # Does it work only for strings?
    A: str | list[str],
    B: str | list[str],
) -> list[str]:
    r"""
    Return the strings contained in the list `A` but not in the list `B`.

    In set formalism, this function returns a list representing the
    set difference :math:`A \backslash (A \cap B)`.
    Note that the operation is not commutative.

    Parameters
    ----------
    A : list of str
        First list of strings.
    B : list of str
        Second list of strings.

    Returns
    -------
    list of str
        The set difference of `A` and `B`.
    """

    A = obj2list(A)
    B = obj2list(B)

    return list(set(A) - set(B))


T = TypeVar("T")


def obj2list(x: T | list[T]) -> list[T]:
    """
    Convert an object `obj` into `list[obj]`.

    If `obj` is already a list, then it return it as-is.

    Parameters
    ----------
    x :
        Input object.
    """
    if not isinstance(x, list):
        x = [x]
    return x


def _get_tutorial_files(dymoval_tutorial_folder: Path) -> None:
    tutorial_site_package_dir = resources.files("dymoval_tutorial")

    # Iterate over each file in the "dymoval_tutorial" package and move it to the destination folder
    for file_path in tutorial_site_package_dir.iterdir():
        if file_path.is_file():
            destination_file = dymoval_tutorial_folder / file_path.name
            shutil.copyfile(str(file_path), str(destination_file))


def open_tutorial() -> Any:
    """Create a dymoval_tutorial folder containing all the files needed to
    run the tutorial in your `home` folder. All you have to do is to run Jupyter notebook named
    `dymoval_tutorial.ipynb`. You need an app for opening `.ipynb` files.

    The content of the *dymoval_tutorial* folder will be overwritten every time this function is
    called.
    """

    home = Path.home()
    dymoval_tutorial_folder = home / "dymoval_tutorial"

    # Delete everything inside the destination folder if it exists
    if os.path.exists(dymoval_tutorial_folder):
        shutil.rmtree(dymoval_tutorial_folder)

    # Create the destination folder
    os.makedirs(dymoval_tutorial_folder, exist_ok=True)

    _get_tutorial_files(dymoval_tutorial_folder)

    destination = dymoval_tutorial_folder / "dymoval_tutorial.ipynb"
    if sys.platform == "win32":
        shell_process = subprocess.run(["explorer.exe", str(destination)])
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        shell_process = subprocess.run([opener, str(destination)])

    return shell_process, destination
