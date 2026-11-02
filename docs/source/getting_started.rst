#################
 Getting Started
#################

Let's walk through a small example: prepare some measurements, check a
simple model, and explore the system's frequency response.

You do not need to be a statistics expert to run these checks and inspect
the plots. Interpreting them still requires knowledge of your experiment
and what the model will be used for. **PASS means the configured checks
passed on this dataset, not that the model is correct everywhere.** A FAIL
is a reason to investigate, not an automatic rejection. The defaults help
you start; they are not calibrated acceptance criteria.
See :ref:`model_ci_cd` before using the results as an automatic release gate.

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
   Ree_envelope: 1.0000
   Rue_correlation: 0.5000

   Actuals:
                                                           Sim_0
   Input auto-correlation (abs_mean-max)                   0.0157
   R-squared fit (%)                                     99.2913
   Residual RMSE (max)                                   0.0287
   Residual auto-correlation (Gaussian envelope ratio)     1.7550
   Residual cross-correlation (abs_mean-max)                 N/A
   Input-residual cross-correlation (abs_mean-max)         0.0831

   Correlation scores: lower means less correlation.
   R-squared fit: higher is better.
   Residual RMSE: lower is better; output units/scaling must be comparable.
   Input auto-correlation describes excitation, not model quality.

            Sim_0
   Outcome: FAIL

Our LPF has a high fit, but exceeds the starting Gaussian residual limit.
This illustrates why fit alone is not the whole assessment. Here is how to
read the scores:

-  **Input auto-correlation (optional)**: Lower values indicate less redundant
   excitation, not necessarily a better model or experiment. Colored inputs
   can also be useful.
-  :math:`\mathbf{R^2}`: A good model should have this value as large as
   possible.
-  **Residual RMSE**: Error magnitude in output units; lower is better.
   There is no default RMSE acceptance threshold.
-  **Residual auto-correlation**: The Gaussian ratio must be at or below 1
   to meet the configured envelope at every evaluated nonzero lag.
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

Customize delay-dependent residual limits
=========================================

Diagonal residual auto-correlation uses a Gaussian envelope by default:
``Ree_envelope="auto"`` selects ``(0.95, 0.1, 5*sampling_period)``.
The limits are starting settings, not universal criteria. Customize them
when you have justified tolerances for short- and long-delay correlation:

.. code-block:: python

   vs_envelope = dmv.validate_models(
       measured_in=measured_input,
       measured_out=measured_output,
       simulated_out=simulated_output,
       sampling_period=sampling_period,
       Ree_envelope=(0.95, 0.1, 0.05),
   )
   print(vs_envelope)
   residual_figures = vs_envelope.plot_residuals(plot_input=False)

These illustrative parameters are ``(short_delay_limit, long_delay_floor,
time_scale)``. Here the time scale is in seconds; it must be chosen from
acceptable error behavior, not adjusted until a model passes.

``Ree_envelope`` is the worst ratio of absolute diagonal residual
auto-correlation to its limit, excluding zero lag. A ratio at or below 1
passes the default envelope check. The residual plot shades the allowed
region on the diagonal only.

The report separates residual auto-correlation (diagonal) from residual
cross-correlation (off-diagonal). With an envelope, the diagonal report row
uses the Gaussian ratio and its default check replaces ``Ree_correlation``.
The ordinary diagonal statistic remains available under ``Ree_correlation``.
``Ree_cross_correlation`` always covers off-diagonal pairs; for this
single-output example it is omitted from statistics and thresholds and
displayed as N/A. Input-residual scores are unchanged.
Here the explicit parameters equal the automatic defaults. The ratio is
about 1.755, so the envelope check fails. That is not a universal verdict
on the LPF: the envelope expresses a different, application-specific
requirement.
Pass ``Ree_envelope=None`` to use the ordinary diagonal statistic instead.
Custom threshold dictionaries must include ``"Ree_envelope": 1.0`` if you
want this check. See :ref:`validation_session` for per-output envelopes,
reference-model baselines, and tuning guidance.

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

.. _model_ci_cd:

*********************************
 Enabling models CI/CD pipelines
*********************************

A traditional software development workflow consists of pushing software
changes to a repository where an automation server (like Jenkins or GitLab)
automatically assesses whether your changes can be integrated into the
codebase by executing some tests.

*Dymoval* can provide repeatable checks in a model-development pipeline.
However, putting a PASS/FAIL check in CI does not turn it into automatic
model certification. It becomes a useful **acceptance gate for a defined
use and operating region** only after you establish what the checks must
demonstrate.

Before enabling automatic acceptance:

* **Define the intended use and coverage.** Specify the operating conditions,
  relevant dynamics, outputs, and acceptable errors. Use several representative
  experiments, including important boundary conditions. Assess input excitation
  and measurement quality separately: a low correlation score alone does not
  establish dataset suitability.
* **Calibrate the checks.** Choose lag windows, aggregation statistics,
  Gaussian envelopes, and thresholds from application requirements and
  representative measurements, rather than accepting the defaults unchanged.
  Include an error-magnitude limit such as RMSE: normalized correlations
  cannot tell you whether the errors are acceptably small. RMSE has no
  default threshold and retains output units; mixed-unit outputs need
  separate limits or justified scaling.
* **Check the gate itself.** Test trusted models and known inadequate
  candidates. If a trusted reference helps define an envelope, allow for
  variation across repeated experiments. Freeze the criteria before assessing
  candidates on independent validation records; do not tune them merely to
  make a failing revision pass.
* **Make runs reproducible and complete.** Version datasets, preprocessing,
  simulation settings, validation settings, and dependencies. Keep them fixed
  across model comparisons. Require every planned experiment and simulation
  to finish; missing results, invalid data, and execution errors must fail
  the pipeline rather than silently skip checks.
* **Keep evidence and review exceptions.** Save scores, plots, settings, and
  model/data revisions as build artifacts. Investigate failures before changing
  limits, and review coverage when the intended use or operating conditions
  change. A threshold-based gate does not replace any required safety or
  regulatory assessment.

For a minimal pipeline check, after computing ``vs`` with your calibrated
settings, return a nonzero exit status if there are no results or any
simulation fails:

.. code-block:: python

   if not vs.outcome or any(result != "PASS" for result in vs.outcome.values()):
       print(vs)
       raise SystemExit(1)

This only checks the results present in one session. Your pipeline must
also verify that all expected experiments and model results are present.
The earlier LPF example deliberately fails its starting envelope check;
the snippet does not override that outcome. Custom threshold dictionaries
enable only their listed checks, so explicitly retain every required check,
including ``Ree_envelope``.

Unlike ``assert``, this check still runs when Python optimization is enabled.
Let simulation and validation exceptions fail the job too.

For a longer, hands-on example with motor measurements, copy the bundled
notebook into your home directory:

.. code-block:: python

   dmv.open_tutorial()

Open ``~/dymoval_tutorial/dymoval_tutorial.ipynb`` in Jupyter. Calling
``open_tutorial()`` again overwrites the files in that tutorial folder.
