##############################
 Installation & plotting
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

Plots
=====
Use Matplotlib styles, ``matplotlibrc``, or ``plt.rc_context()`` to customize
plot appearance. Validation plots use ``axes.prop_cycle`` to assign consistent
colors to simulations across subplots. Correlation titles use built-in
MathText; external LaTeX is only used when you enable ``text.usetex``.
There is no package-specific configuration file.

For example, apply styling locally without changing the rest of your session:

.. code-block:: python

    import matplotlib.pyplot as plt

    with plt.rc_context({
        "font.size": 12,
        "lines.linewidth": 1.5,
        "axes.prop_cycle": plt.cycler(color=["tab:blue", "tab:orange"]),
    }):
        fig = ds.plot(layout="tight", ax_width=12, ax_height=3)

By default, plot size and layout inherit Matplotlib's ``figure.figsize``,
``figure.constrained_layout.use``, and ``figure.autolayout`` settings.
Pass ``autosize=True`` to opt into dymoval's panel-aware sizing, which grows
with the subplot grid and is recommended for crowded scope-enabled plots:

.. code-block:: python

    fig = ds.plot(autosize=True, layout="constrained")

``autosize`` controls size only, not layout or styling. Explicit ``layout``,
``ax_width``, and ``ax_height`` arguments override the corresponding defaults.
With ``autosize=False``, a dimension you leave unspecified still follows
``figure.figsize``; with ``autosize=True`` it uses the panel sizing rule.
``Signal.plot()`` and ``Signal.plot_spectrum()`` accept ``figsize`` instead of
per-panel dimensions. Geometry options leave caller-supplied axes unchanged.
Use ``layout="none"`` to disable automatic layout, or adjust returned figures
with ``fig.set_layout_engine()`` and ``fig.set_size_inches()``. The relative
width of the scope information panel remains an internal structural default.

NumPy and SciPy do not provide a Matplotlib-style ``rcParams`` system.
NumPy's ``set_printoptions()`` controls array display, while ``seterr()`` and
``errstate()`` control floating-point error handling, not comparison
tolerances. Numerical tolerances and algorithm options are supplied to the
relevant functions explicitly.

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
