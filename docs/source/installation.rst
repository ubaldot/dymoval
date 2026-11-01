#############
 Installation
#############

**************
 Installation
**************

By running

.. code::

   pip install dymoval

or

.. code::

   conda install -c conda-forge dymoval

everything should work fine.


****************************
 Installation from the repo
****************************

Clone the repo from the_repo_ and run

.. _the_repo: https://github.com/VolvoGroup/dymoval

.. code::

   cd /path/to/where/you/cloned/this/repo
   conda env update --name env_name --file environment.yml
   conda activate env_name
   pip install .

or

.. code::

   cd /path/to/where/you/cloned/this/repo
   pip install .

.. _github: https://github.com/ubaldot/dymoval

Developer checks
================
``tests`` contains automated regression tests, including FFT/Parseval and
bundled MATLAB reference data. ``scripts`` contains standalone developer
tools; ``compare_matlab.py`` prints metrics for external CSV files.

``manual_tests`` is reserved for human visual and interactive checks and is
excluded from pytest discovery. From the repository root, run:

.. code-block:: console

    python manual_tests\inspect_plots.py

The default Qt backend requires a Qt binding. Use ``--backend TkAgg`` or
another installed interactive backend if needed. Inspect subplot geometry,
labels, legends, NaN shading, and scope panels; click curves and press ``r``
to reset. After closing the plot windows, check graphical interval selection
by zooming and closing each trimming window.

To check figure construction and rendering without opening windows:

.. code-block:: console

    python manual_tests\inspect_plots.py --backend Agg --no-show

This smoke check does not replace visual inspection or interaction testing.
