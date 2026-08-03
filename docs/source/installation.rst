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

****************************
Configuration
****************************
The configuration of `dymoval` is fairly straightforward since there are only
few parameter that you can set.

.. py:data:: color_map
   :type: str
   :value: "tab10"

   The used ``matplotlib`` color map. Check ``Matplotlib`` docs for possible values.
.. py:data:: float_tolerance
   :type: str
   :value: 1e-9

   Tolerance for ``float`` operations, such as ``np.isclose()``, etc.

.. py:data:: is_interactive
   :type: bool | None
   :value: None

    Whether `dymoval` should consider the current environment interactive.
    If `None`, the environment is auto-detected. This only backs the
    :py:func:`is_interactive_shell <dymoval.utils.is_interactive_shell>`
    helper: it does **not** influence plotting any more (see `Plots`_
    below).


These parameters can be set through a ``~/.dymoval/config.toml`` file. You
have to create such a file manually.

A ``~/.dymoval/config.toml`` could for example include the following content

.. code-block::

    color_map = "tab20"
    atol = 1-6

Plots
=====
The `dymoval` plotting functions **never** call ``show()``: they build the
figure and return it, leaving the display to you and to your `matplotlib`
backend. In a script this means calling ``plt.show()`` yourself; in a
notebook the returned figure is rendered by the inline backend, or you can
``display(fig)`` it explicitly.

Functions returning several figures (such as
:py:meth:`plot_residuals <dymoval.validation.ValidationSession.plot_residuals>`)
return a tuple, so:

.. code-block::

    figs = vs.plot_residuals()
    for fig in figs:
        display(fig)   # or plt.show() once, in a script

It is suggested to disable the `matplotlib` interactivity with
``plt.ioff()`` if working with `IPython`.
