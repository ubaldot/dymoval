Installation & configuration
============================

Installation
------------

By running

.. code-block::

   pip install dymoval

or

.. code-block::

   conda install dymoval

everything should work fine.


Installation from the repo
^^^^^^^^^^^^^^^^^^^^^^^^^^
Clone the repo from `the_repo`_  and run

.. _the_repo: https://github.com/VolvoGroup/dymoval


.. code-block::

    cd /path/to/where/you/cloned/this/repo
    conda env update --name env_name --file environment.yml
    conda activate env_name
    pip install .

or

.. code-block::

	cd /path/to/where/you/cloned/this/repo
	pip install .


.. _GitHub: https://github.com/ubaldot/dymoval


Configuration
-------------
The configuration of `dymoval` is fairly straightforward since there is only
one parameter that you can set.


.. confval:: color_map
    :type: str
    :default: "tab10"

    The used `matplotlib` color map. Check `Matplotlib` docs for possible values.

These parameters can be set through a :code:`~/.dymoval/config.toml`  file.
You have to create such a file manually.

A :code:`~/.dymoval/config.toml` could for example include the following content

.. code-block::

    color_map = "tab20"

Plots
^^^^^
Dymoval shall be able to recognize if you are working or not in an interactive
environment. It is however suggested to disable the matplotlib interactivity
with `plt.ioff()`.
