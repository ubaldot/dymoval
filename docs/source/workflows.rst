######################
 End-to-end workflows
######################

The examples below show the main path through *dymoval*: clean measured
signals, inspect empirical dynamics, and validate a
simulation model.

Prepare measurement data
========================

Create named :py:class:`~dymoval.signal.Signal` objects and let
:py:class:`~dymoval.dataset.Dataset` enforce a common time grid:

.. code:: python

   import numpy as np
   from scipy.signal import lfilter

   import dymoval as dmv

   rng = np.random.default_rng(42)
   sampling_period = 0.01
   time = np.arange(4096) * sampling_period
   voltage = rng.standard_normal(len(time))
   speed = lfilter([0.0, 0.2], [1.0, -0.8], voltage)

   # A short sensor dropout.
   speed[500:505] = np.nan

   ds = dmv.Dataset.from_signals(
       inputs=[dmv.Signal("voltage", voltage, time, "V")],
       outputs=[dmv.Signal("speed", speed, time, "rad/s")],
       name="motor identification",
       meta={"campaign": "bench-2026"},
   )
   ds_clean = ds.remove_nans()
   ds_clean.validate()

Manipulation methods return a new object. For inspection without accidental
array mutation, use ``Signal.values_view``, ``Signal.time_view``,
``Dataset.time_view``, ``Dataset.input_signals``, and
``Dataset.output_signals``. These are shallow guards: they do not make nested
signals or other aliases immutable. Direct mutation of public dictionaries
and arrays is retained for compatibility but discouraged; call ``validate()``
if legacy code mutates them.

Estimate and inspect a frequency response
=========================================

:py:meth:`Dataset.spa <dymoval.dataset.Dataset.spa>` estimates a
Blackman--Tukey frequency response:

.. code:: python

   response = ds_clean.spa(window_size=30)

   print(response.shape)           # (frequencies, outputs, inputs)
   print(response.frequency_unit)  # rad/s
   print(response.response_units)  # (("rad/s/V",),)

   motor_response = response.select(
       inputs="voltage",
       outputs="speed",
   )
   figure = motor_response.plot()
   figure.savefig("motor-frequency-response.svg")

``response_std`` and ``noise_spectrum_std`` contain asymptotic standard
deviations. SISO estimates additionally expose magnitude-squared
``coherence``. All response arrays use frequency-first shapes:
``(n_frequencies, n_outputs, n_inputs)``.

Validate a model
================

Compare measured output with one or more simulated outputs:

.. code:: python

   measured_input = voltage[:, None]
   measured_output = np.nan_to_num(speed)[:, None]

   # Candidate model with a slightly different pole.
   simulated_output = lfilter(
       [0.0, 0.2],
       [1.0, -0.78],
       voltage,
   )[:, None]

   session = dmv.validate_models(
       measured_in=measured_input,
       measured_out=measured_output,
       simulated_out=simulated_output,
       sampling_period=sampling_period,
   )

   print(session.outcome)
   session.plot_simulations().savefig("simulation-comparison.svg")

The validation result is independent of the tool used to produce
``simulated_output``.
