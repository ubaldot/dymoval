##############################
 Installation & configuration
##############################

**************
 Installation
**************

By running

.. code::

   pip install dymoval

or

.. code::

   conda install dymoval

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

****************************
Configuration
****************************
The configuration of `dymoval` is fairly straightforward since there are only
few parameter that you can set.


.. confval:: color_map
    :type: str
    :default: "tab10"

    The used `matplotlib` color map. Check `Matplotlib` docs for possible values.

.. confval:: float_tolerance
    :type: str
    :default: 1e-9

    Tolerance for `float` operations, such as `np.close()', etc.

.. confval:: is_interactive
    :type: bool | None
    :default: None

    The dymoval plot functions end with `fig.show()` in interactive
    environments such as `IPython`, and with `plt.show()` for non-interactive
    environments. If 'is_interactive` is `None`, then dymoval attempts to
    detect your environment automatically. Otherwise, you can force the
    behavior through this configuration parameter.

These parameters can be set through a :code:`~/.dymoval/config.toml`  file.
You have to create such a file manually.

A :code:`~/.dymoval/config.toml` could for example include the following content

.. code-block::

    color_map = "tab20"
    atol = 1-6

Plots
=====
Dymoval shall be able to recognize if you are working or not in an interactive
environment. It is however suggested to disable the matplotlib interactivity
with `plt.ioff()` if working with `IPython`.
