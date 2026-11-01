# -*- coding: utf-8 -*-
"""Small package utilities that do not belong to a domain object.

The helpers here cover read-only array views, collection normalization,
and installation of tutorial resources. Signal processing, plotting behavior,
and validation logic remain in their own modules.
"""

import os
import shutil
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

__all__ = ["open_tutorial"]


def _readonly_view(array: np.ndarray) -> np.ndarray:
    """Return a zero-copy, read-only view of an array."""
    view = array.view()
    view.setflags(write=False)
    return view


T = TypeVar("T")


def _obj2list(x: T | list[T]) -> list[T]:
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
