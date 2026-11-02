.. _validation_session:

#####################
 Validate your model
#####################

You have measurements and a model simulated with the same inputs. Now comes
the useful question: how well does the model describe what happened?
Start by comparing the outputs, then look at the errors and their patterns.
Dymoval brings these checks together in a validation session.

************************
 Start with a comparison
************************

.. code:: python

   from dymoval import validate_models

   vs = validate_models(
       measured_in=u_meas,
       measured_out=y_meas,
       simulated_out=y_sim,
       sampling_period=sampling_period,
   )
   print(vs)
   fig = vs.plot_simulations()

The arrays must be aligned: ``u_meas`` has shape ``(N, p)``, while
``y_meas`` and ``y_sim`` have shape ``(N, q)``. Here, ``N`` is the number
of samples, ``p`` the number of inputs, and ``q`` the number of outputs.
You can also pass measured ``Signal`` objects to retain names and units.

The returned :py:class:`~dymoval.validation.ValidationSession` contains the
measurements, simulations, scores, thresholds, and outcomes. Compare another
model or parameter setting by appending its simulated outputs:

.. code:: python

   vs = vs.append_simulation(
       sim_name="Another model",
       y_names=vs.dataset.output_names(),
       y_data=y_sim2,
   )

.. _theory:

************************************
 Understand the residuals and delays
************************************

The *residuals* are simply the differences between measured and simulated
outputs:

.. math::

   \varepsilon_i(t) =
   y_{\mathrm{measured},i}(t)-y_{\mathrm{simulated},i}(t).

Their magnitude tells you how far the predictions are from the measurements.
Their correlation tells you whether the errors contain repeated or shared
patterns. These are different questions: small errors can still have
structure, and uncorrelated errors can still be too large to be useful.

Auto-correlation compares a signal with delayed copies of itself.
Cross-correlation compares two different signals at different delays.
Dymoval computes three arrays of correlation functions:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Array
     - Shape
     - What its entries compare
   * - ``Ruu``
     - ``p x p``
     - Input channels with themselves and with other input channels.
   * - ``Ree``
     - ``q x q``
     - Residual components with themselves and with other residual components.
   * - ``Rue``
     - ``p x q``
     - Each input channel with each residual component.

The diagonal of ``Ree`` contains scalar residual auto-correlations
:math:`\varepsilon_i` versus :math:`\varepsilon_i`. Its off-diagonal entries
are cross-correlations :math:`\varepsilon_i` versus :math:`\varepsilon_j`.
Every entry of ``Rue`` is a cross-correlation, including :math:`u_i` versus
:math:`\varepsilon_i`: matching indices do not make them the same signal.
The same diagonal/off-diagonal distinction applies to ``Ruu``.

What does one lag mean?
=======================

One lag is one sampling period :math:`T_s`. Lag :math:`k` therefore means
a physical delay of :math:`kT_s`. Dymoval does not automatically thin
correlation lags based on bandwidth.

Both ``XCorrelation`` and ``ValidationSession`` default to 41 points:
-20 through +20 sample offsets, limited by record length. To inspect delays
up to :math:`KT_s` in either direction, request ``2*K+1`` points per pair.
The session settings are ``Ruu_nlags`` (``p x p``), ``Ree_nlags``
(``q x q``), and ``Rue_nlags`` (``p x q``).

For example, with :math:`T_s=0.5` seconds, the following residual window
covers -10 to +10 seconds, provided enough data is available:

.. code:: python

   import numpy as np

   q = y_meas.shape[1]
   residual_windows = np.full((q, q), 41, dtype=int)
   # Pass Ree_nlags=residual_windows when creating the session.

Increasing the window extends the inspected delay range, not the spacing.
Plots and returned ``lags`` use sample offsets; multiply by :math:`T_s`
to obtain delays in the dataset's time units.

******************************
 How dymoval computes scores
******************************

Each simulation gets one scalar per reported statistic. Here is how the
measurements and correlation curves become those numbers.

Output fit and error magnitude
==============================

For each output, dymoval computes:

.. math::

   R_i^2 = 100\left(
       1-\frac{\sum_t\varepsilon_i(t)^2}
       {\sum_t(y_i(t)-\overline y_i)^2}\right),
   \qquad
   \mathrm{RMSE}_i =
       \sqrt{\frac{1}{N}\sum_t\varepsilon_i(t)^2}.

**R-squared** measures fit relative to the measured output's variation.
Higher is better; it can be negative and is undefined for a constant
reference output. **Residual RMSE** measures error magnitude in output
units. Lower is better, and a residual bias contributes because its mean
is not removed.

For multiple outputs, ``r2_statistic="min"`` selects the lowest fit and
``rmse_statistic="max"`` selects the largest RMSE by default. Both also
support ``"mean"``. Outputs need comparable units and scaling before
aggregating RMSE. Use :py:func:`~dymoval.statistics.rmse` directly if you
want the individual output values.

The session stores the aggregates under ``"r2"`` and ``"rmse"`` in
``validation_statistics``. There is no default RMSE threshold.

Ordinary correlation scores: local, then global
===============================================

Dymoval first subtracts each signal's mean and divides by its standard
deviation. It computes the correlation sums and normalizes by record
length, then keeps the requested lag window. For aligned, equal-length
signals, the resulting correlations lie between -1 and 1. Constant
channels have undefined normalized correlation.

Scoring happens in two stages:

1. **Local score:** summarize each channel pair's correlation curve over
   its evaluated lags. The default ``"abs_mean"`` is the mean absolute
   correlation. For diagonal self-correlations, the zero-lag peak is
   excluded; cross-correlations still include zero lag.
2. **Global score:** summarize the resulting array of local scores. The
   default ``"max"`` selects the largest, so the worst channel pair sets
   the reported score.

These settings are separate for ``Ruu``, ``Ree``, and ``Rue``. For example,
``Ree_local_statistic_type`` selects the statistic over lags, while
``Ree_global_statistic_type`` selects the statistic over channel pairs.
The corresponding ``*_local_weights`` and ``*_global_weights`` provide
weighting. See :py:meth:`~dymoval.xcorrelation.XCorrelation.correlation_score`
and :py:func:`~dymoval.statistics.compute_statistic` for the available
statistics and weight conventions.

The reported keys are ``"Ruu_correlation"``, ``"Ree_correlation"``,
``"Ree_cross_correlation"``, and ``"Rue_correlation"``.
``Ree_correlation`` aggregates diagonal residual auto-correlations only;
``Ree_cross_correlation`` aggregates off-diagonal residual cross-correlations.
With one output, there are no off-diagonal pairs: the latter statistic and
threshold are omitted, and the report displays N/A.

.. important::

   An ``abs_mean`` score depends on the lag window. Adding many near-zero
   correlations can dilute a short-delay peak without improving the model.
   ``max`` avoids this dilution but cannot distinguish one peak from
   sustained correlation. Choose and report the window, statistic, and
   threshold together, keeping them fixed across model comparisons.
   These summaries are not formal statistical tests.

Default diagonal score: Gaussian acceptance envelope
=====================================================

Sometimes you have reason to tolerate stronger residual correlation at
short delays, but want tighter limits at longer delays. An *acceptance
envelope* expresses that requirement without ignoring the central region.
It is a chosen limit, not the derivative of the correlation or a curve
fitted to each candidate model.

For each output, the Gaussian limit is

.. math::

   b(t)=b_\infty+(b_0-b_\infty)\exp[-(t/T_b)^2],
   \qquad 0<b_\infty\leq b_0\leq1,\quad T_b>0.

Dymoval compares each nonzero-lag diagonal correlation with its limit and
takes the worst ratio across delays and outputs:

.. math::

   S=\max_i\max_{k\ne0}
       \frac{|\hat\rho_{\varepsilon_i\varepsilon_i}(k)|}
       {b_i(|k|T_s)}.

**A score at or below 1 means every evaluated diagonal value meets its
limit.** A score above 1 identifies an exceedance. Adding near-zero
correlations cannot dilute a violation. This only assesses the chosen
window, not every possible delay.

Gaussian scoring is enabled by default with ``Ree_envelope="auto"``:
``b0=0.95``, ``b_inf=0.1``, and ``time_scale=5*Ts``. These are starting
settings, not calibrated acceptance limits. The width follows the sampling
period, not an inferred system or residual time constant; review it when
changing sampling rates.

Override the parameters when creating a session:

.. code:: python

   vs = validate_models(
       measured_in=u_meas,
       measured_out=y_meas,
       simulated_out=y_sim,
       sampling_period=sampling_period,
       Ree_envelope=(0.95, 0.1, 5.0),
   )
   figures = vs.plot_residuals()

The tuple is ``(b0, b_inf, time_scale)``, shared across outputs. Parameters
must be finite and satisfy the constraints above. The time scale uses
dataset time units, not lag indices. With array measurements it uses the
same time unit as ``sampling_period``.

For separate limits, supply every measured output name:

.. code:: python

   limits = {
       "temperature": (0.95, 0.1, 5.0),
       "flow": (0.8, 0.05, 2.0),
   }
   # Pass Ree_envelope=limits with correspondingly named measured Signals.
   # Array measurements instead get names y0, y1, and so on.

With an envelope enabled:

* ``Ree_envelope`` stores the maximum diagonal ratio. Its default
  acceptance threshold is 1, inclusive.
* The diagonal report row uses the Gaussian ratio instead of the ordinary
  ``Ree_correlation`` score. Its default check is replaced by the envelope
  check. The ordinary diagonal score remains available in the statistics.
* ``Ree_cross_correlation`` retains ordinary off-diagonal scoring.
* ``plot_residuals()`` shades the region between the positive and negative
  Gaussian limits on diagonal residual plots only. Zero lag is displayed
  but not scored.

The envelope ratio does not use ``Ree_local_weights``,
``Ree_global_weights``, or the ordinary aggregation settings. It always
takes the maximum. Each diagonal window must contain a nonzero lag;
undefined diagonal correlations produce an infinite ratio.
Input and input-residual scoring are unchanged.

Pass ``Ree_envelope=None`` to use ordinary diagonal correlation scoring.
Envelope ratios and ordinary correlation magnitudes are kept separate:
they should not be averaged together as if they meant the same thing.

*****************************
 Choose meaningful envelopes
*****************************

The parameters control different parts of the acceptance limit:

* **Increase** :math:`b_0` to allow more short-delay correlation.
* **Lower** :math:`b_\infty` to demand less long-delay correlation.
* **Increase** :math:`T_b` to let the limit stay permissive for longer.
  At :math:`t=T_b`, the allowance above the floor has fallen to
  :math:`1/e` of its initial value.

For illustration, :math:`b_0=0.95`, :math:`b_\infty=0.1`, and
:math:`T_b=5` seconds give limits of about 0.92 at one second and 0.12
at ten seconds. These are not recommended defaults.

.. figure:: ../figures/ResidualCorrelationEnvelope.svg
   :alt: A Gaussian limit allows correlation of 0.75 at one second
         but rejects correlation of 0.88 at ten seconds.
   :align: center
   :width: 100%

   Short delays are checked against permissive limits, not discarded.

A Gaussian is flatter near zero and tightens more sharply later than an
exponential. This can suit a rounded central cluster, but residual
correlations need not be Gaussian. The envelope expresses acceptable
behavior, not a universal model of the errors.

Choose limits from application requirements, sensor/filter knowledge, or
independent experiments. A trusted reference model is a useful baseline:
use several representative logs, allow for run-to-run variation, and
freeze the limits before evaluating candidates on separate data.
Keep sampling, preprocessing, operating conditions, and evaluated delays
comparable. Do not tune the envelope merely until a model passes.

A slow output does not automatically justify a wide envelope: its slow
response may cancel in the residuals. Account for finite-record estimation
fluctuations too. Retain an RMSE criterion, since identical normalized
correlation can accompany much larger errors.

What about cross-correlations?
===============================

Off-diagonal residual entries and all input-residual entries can retain
``abs_mean``, ``max``, or the other ordinary statistics, including zero lag.
When near-zero correlation is the target, a constant tolerance is a
reasonable starting point. ``max`` divided by that tolerance is equivalent
to a flat-envelope violation ratio; ``abs_mean`` instead measures average
correlation and can hide localized peaks.

Different outputs can share disturbances or sensor effects, so residual
cross-correlations need not vanish for accurate modeled dynamics.
Feedback and input-related disturbances can also complicate input-residual
correlation. Treat the scores accordingly rather than assuming every
nonzero correlation proves model error.

********************************
 Interpret results and PASS/FAIL
********************************

Fit and RMSE describe agreement and error magnitude. Residual correlation
reveals temporal or shared patterns; input-residual correlation can expose
missed input-output behavior. Input correlation describes excitation, not
model quality. Use :ref:`doe` to assess input richness and dataset suitability.

.. note::

   Low residual correlation is generally desirable, but not a universal
   requirement for a correct simulation model. Colored noise and unmeasured
   disturbances can produce correlated errors. These residuals are not
   necessarily one-step prediction errors from a model of both dynamics
   and noise. Clean, meaningful measurements and knowledge of the experiment
   are essential to interpreting the scores.

The default ordinary thresholds are:

.. list-table::
   :header-rows: 1

   * - Key
     - Passing condition
   * - ``Ruu_correlation``
     - Less than 0.6
   * - ``r2``
     - Greater than 35 (%)
   * - ``Ree_correlation``
     - Less than 0.5 (diagonal; only with ``Ree_envelope=None``)
   * - ``Ree_cross_correlation``
     - Less than 0.5 (off-diagonal; multiple outputs only)
   * - ``Rue_correlation``
     - Less than 0.5

RMSE is reported without a default threshold. The default diagonal check is
``Ree_envelope <= 1``. All configured checks must pass. These defaults are
starting settings, not universal suitability criteria.

``ignore_input=True`` omits the input check from the default thresholds.
With custom thresholds, omit ``Ruu_correlation`` yourself if you do not
want that check. Custom dictionaries enable only their listed checks;
include ``"Ree_envelope": 1.0`` explicitly to check an enabled envelope.

Inspect or reuse the results as plain dictionaries:

.. code:: python

   print(vs.validation_statistics)
   print(vs.validation_thresholds)
   print(vs.outcome)

To add an RMSE tolerance without dropping existing checks:

.. code:: python

   thresholds = dict(vs.validation_thresholds)
   thresholds["rmse"] = 0.1  # Illustrative: choose for your units/application.
   vs.validation_thresholds = thresholds

Setting thresholds recomputes existing outcomes. Ordinary correlation
and RMSE thresholds are strict upper bounds; R-squared uses a strict lower
bound. The envelope threshold is an inclusive upper bound.

Inspect the plots too:

.. code:: python

   fig = vs.plot_simulations()
   figures = vs.plot_residuals()

Deliver the results alongside the experiment's operating-region coverage
and acceptance settings. A PASS only describes these checks on this data.
It does not certify that the model is correct, nor does a FAIL necessarily
mean that the model is unusable. Beginners can use the scores and plots to
guide an investigation, but choosing acceptable errors and interpreting
residual patterns still needs knowledge of the application.
Before using outcomes for automatic acceptance, calibrate and freeze the
criteria, assess independent representative records, and verify that the
gate detects known inadequate models. See :ref:`model_ci_cd` for a practical
checklist.

*******************************
 The results are disappointing
*******************************

A failed check is a reason to investigate, not automatically to reject the
model or relax the threshold.

* **Check the measurements.** Verify timing alignment, sensor quality,
  actual actuator behavior, excitation, and operating conditions.
* **Be careful with filtering.** It may help separate relevant behavior
  from noise, but can hide errors or introduce correlation and phase shifts.
  Define the frequency range being validated and treat measured and
  simulated outputs consistently.
* **Inspect the delay range.** If correlation stays high across the plotted
  window, extend it to understand its persistence. Do not use the resulting
  dilution of ``abs_mean`` as evidence that the model improved.
* **Check offsets and trends.** Correlation diagnostics remove means but
  not trends. A constant bias can disappear from correlation while still
  contributing to RMSE. Preprocessing must match the model conventions.
* **Inspect the model response.** Incorrect initial conditions, timing
  differences, and missed dynamics can all leave residual patterns.
