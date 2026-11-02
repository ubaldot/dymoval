#################
 Getting Started
#################

Let's walk through a small example: prepare some measurements, check a
simple model, and explore the system's frequency response.

**************************
 Prepare measurement data
**************************

We'll generate synthetic voltage and speed measurements so you can run this
example without downloading any data. In your own work, these arrays would
come from your experiment logs.

.. code-block:: python

   import numpy as np
   from scipy.signal import lfilter

   import dymoval as dmv

   rng = np.random.default_rng(42)
   sampling_period = 0.01
   time = np.arange(4096) * sampling_period
   voltage = rng.standard_normal(len(time))
   speed = lfilter([0.0, 0.2], [1.0, -0.8], voltage)

   # Pretend the speed sensor briefly stopped recording.
   speed[500:505] = np.nan

   ds = dmv.Dataset.from_signals(
       inputs=[dmv.Signal("voltage", voltage, time, "V")],
       outputs=[dmv.Signal("speed", speed, time, "rad/s")],
       name="motor experiment",
   )
   ds_clean = ds.remove_nans()
   ds_clean.validate()

Each :py:class:`~dymoval.signal.Signal` keeps its name, units, and time
samples alongside its values. A :py:class:`~dymoval.dataset.Dataset` groups
the input and output signals on a common time grid. Here, ``remove_nans()``
fills the short gap by interpolation and returns a new dataset, leaving the
original unchanged.

You can inspect the cleaned signals with:

.. code-block:: python

   fig = ds_clean.plot()

*************************
 Assess input excitation
*************************

Before comparing a model with the measurements, we can check the inputs
independently of any model:

.. code-block:: python

   L = 20  # Illustrative choice: 20 samples at the recorded sampling period.
   eigenvalues = ds_clean.input_gram_eigenvalues(window_length=L)
   print(f"Minimum input Gram eigenvalue: {eigenvalues[0]:.6g}")
   print(f"Maximum input Gram eigenvalue: {eigenvalues[-1]:.6g}")

The minimum eigenvalue describes the weakest excited combination of input
channels and delayed samples. Choose ``L`` to cover a time span relevant to
the dynamics you want to validate. Values depend on input amplitudes and
units: there is no universal acceptance threshold. The calculation uses raw
inputs, without centering or scaling; preprocess them first if appropriate.
Use ``ds_clean.input_gram_matrix(L)`` if you also need the averaged matrix.

This assesses input excitation, not overall dataset suitability. Inspect
the outputs too: does the applied input produce a clear response above
measurement noise? See :ref:`doe` for details. These diagnostics belong to
the dataset and are separate from the model-validation report below.

******************
 Validate a model
******************

Suppose our model is a simple first-order low-pass filter (LPF). We'll use
SciPy to simulate it, with a slightly different pole from the system that
generated the measurements. The important part is to feed it the
**same measured input**.

.. code-block:: python

   _, measured_input, measured_output = ds_clean.dataset_values()

   simulated_output = lfilter(
       [0.0, 0.2],
       [1.0, -0.78],
       measured_input[:, 0],
   )[:, None]

   vs = dmv.validate_models(
       measured_in=measured_input,
       measured_out=measured_output,
       simulated_out=simulated_output,
       sampling_period=sampling_period,
   )
   print(vs)

``dataset_values()`` returns time, inputs, and outputs. The input and output
arrays have one row per sample and one column per channel; ``[:, None]``
gives our simulated output that same two-dimensional shape.

The returned :py:class:`~dymoval.validation.ValidationSession` shows the
setup, scores, and outcome, including residual RMSE. RMSE measures the typical
residual magnitude; lower is better. Per-output values are aggregated using
``rmse_statistic="max"`` by default; ``"mean"`` is also available.
Access the aggregate as ``vs.validation_statistics["Sim_0"]["rmse"]``.
Use ``dmv.rmse(measured_output, simulated_output)`` for per-output values.
For multiple outputs, units and scaling should be comparable before
aggregating. No RMSE threshold is enabled by default; supplying
``validation_thresholds={"rmse": tolerance}`` checks that the aggregate is
strictly below your tolerance. As with other custom threshold dictionaries,
include any other checks you want to retain.
The results for this example look like:

.. code-block:: text

   Validation results:
   -------------------
   Thresholds:
   Ruu_correlation: 0.6000
   r2: 35.0000
   Ree_correlation: 0.5000
   Rue_correlation: 0.5000

   Actuals:
                                                           Sim_0
   Input auto-correlation (abs_mean-max)                   0.0157
   R-squared fit (%)                                     99.2913
   Residual RMSE (max)                                   0.0287
   Residual auto-correlation (abs_mean-max)                0.2676
   Input-residual cross-correlation (abs_mean-max)         0.0831

   Correlation scores: lower means less correlation.
   R-squared fit: higher is better.
   Residual RMSE: lower is better; output units/scaling must be comparable.
   Input auto-correlation describes excitation, not model quality.

            Sim_0
   Outcome: PASS

Our LPF passes the default thresholds for this experiment. Here is how to
read the scores:

-  **Input auto-correlation (optional)**: Lower values indicate whiter
   excitation, not necessarily a better model or experiment. Colored inputs
   can also be useful.
-  :math:`\mathbf{R^2}`: A good model should have this value as large as
   possible.
-  **Residual auto-correlation**: Values near zero suggest that the errors
   have little correlation across the evaluated lags.
-  **Input-residual cross-correlation**: Values near zero suggest little
   correlation between the inputs and the errors at the evaluated lags.

These are correlation summary scores, not formal statistical whiteness
tests. Their dictionary keys are ``Ruu_correlation``, ``Ree_correlation``,
and ``Rue_correlation``.

A pass is not a guarantee that the model works everywhere. Choose experiments
and thresholds that suit your application, and inspect the errors as well
as the fit. See :ref:`theory` for the calculations and
:ref:`model_validation` for more background.

We used SciPy here, but dymoval does not need to know how you produced
``simulated_output``. Results from another simulation tool work just as well.

*************************************************
 Extra: estimate and inspect a frequency response
*************************************************

You can also explore the measured dynamics without choosing a parametric
model. :py:meth:`Dataset.spa <dymoval.dataset.Dataset.spa>` estimates a
frequency response directly from the cleaned input and output signals:

.. code-block:: python

   response = ds_clean.spa(window_size=30)
   fig = response.plot()

The plot shows magnitude and phase, plus coherence for this single-input,
single-output example. Coherence can help you judge how well the input
explains the output at each frequency.

To select channels or inspect the numerical results:

.. code-block:: python

   motor_response = response.select(inputs="voltage", outputs="speed")
   print(motor_response.shape)           # (frequencies, outputs, inputs)
   print(motor_response.frequency_unit)  # rad/s
   print(motor_response.response_units)  # (("rad/s/V",),)

See the :doc:`reference_index/dataset` for SPA options and uncertainty
estimates.

******
 Plots
******

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
        fig = ds_clean.plot(layout="tight", ax_width=12, ax_height=3)

By default, plot size and layout inherit Matplotlib's ``figure.figsize``,
``figure.constrained_layout.use``, and ``figure.autolayout`` settings.
Pass ``autosize=True`` to opt into dymoval's panel-aware sizing, which grows
with the subplot grid and is recommended for crowded scope-enabled plots:

.. code-block:: python

    fig = ds_clean.plot(autosize=True, layout="constrained")

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

*********************************
 Enabling models CI/CD pipelines
*********************************

A traditional software development workflow consists of pushing software
changes to a repository where an automation server (like Jenkins or GitLab)
automatically assesses whether your changes can be integrated into the
codebase by executing some tests.

*Dymoval* allows you to do the same with models. All you need is a good
measurement dataset and a mechanism to run simulations on an automation
server. Then, you can use the *Dymoval* API to validate the model changes and
decide whether to merge or reject the proposed changes.

For example, after computing ``vs`` as above, make the pipeline fail if a
simulation does not pass:

.. code-block:: python

   assert all(result == "PASS" for result in vs.outcome.values()), vs

Keep the measurement data and validation settings fixed so that you can
compare model revisions consistently.

For a longer, hands-on example with motor measurements, copy the bundled
notebook into your home directory:

.. code-block:: python

   dmv.open_tutorial()

Open ``~/dymoval_tutorial/dymoval_tutorial.ipynb`` in Jupyter. Calling
``open_tutorial()`` again overwrites the files in that tutorial folder.
