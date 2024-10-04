# -*- coding: utf-8 -*-
"""Module containing some useful functions."""

import numpy as np

from typing import Any, List, TypeVar
import sys
import os
import subprocess
from importlib import resources
import shutil
from pathlib import Path

__all__ = [
    "is_interactive_shell",
    "factorize",
    "difference_lists_of_str",
    "obj2list",
    "open_tutorial",
]


def is_interactive_shell() -> bool:
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
    A = obj2list(A)
    B = obj2list(B)

    return list(set(A) - set(B))


T = TypeVar("T")


# TODO: rename obj2list
def obj2list(x: T | List[T]) -> List[T]:
    # def obj2list(x: Any | List[Any]) -> List[Any]:
    """
    Convert *obj* of type T into *list[obj]* of type T.

    If *obj* is already a list, then it return it as-is.

    Parameters
    ----------
    x :
        Input object.
    """
    if not isinstance(x, list):
        x = [x]
    return x


def _get_tutorial_files(dymoval_tutorial_folder: str) -> None:

    tutorial_site_package_dir = resources.files("dymoval_tutorial")

    # Iterate over each file in the "dymoval_tutorial" package and move it to the destination folder
    for file_path in tutorial_site_package_dir.iterdir():
        if file_path.is_file():
            destination_file = os.path.join(
                dymoval_tutorial_folder, file_path.name
            )
            shutil.copyfile(str(file_path), destination_file)


def open_tutorial() -> Any:
    """Create a dymoval_tutorial folder containing all the files needed to
    run the tutorialin your `home` folder. All you have to do is to run Jupyter notebook named
    dymoval_tutorial.ipynb. You need an app for opening .ipynb files.

    The content of the dymoval_tutorial folder will be overwritten every time this function is
    called.
    """

    home = str(Path.home())
    dymoval_tutorial_folder = os.path.join(home, "dymoval_tutorial")

    # Delete everything inside the destination folder if it exists
    if os.path.exists(dymoval_tutorial_folder):
        shutil.rmtree(dymoval_tutorial_folder)

    # Create the destination folder
    os.makedirs(dymoval_tutorial_folder, exist_ok=True)

    _get_tutorial_files(dymoval_tutorial_folder)

    if sys.platform == "win32":
        destination = dymoval_tutorial_folder + "\\dymoval_tutorial.ipynb"
        shell_process = subprocess.run(["explorer.exe", destination])
    else:
        destination = dymoval_tutorial_folder + "/dymoval_tutorial.ipynb"
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        shell_process = subprocess.run([opener, destination])

    return shell_process, destination
