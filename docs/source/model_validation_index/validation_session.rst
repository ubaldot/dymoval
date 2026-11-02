.. _validation_session:

#####################
 Validate your model
#####################

To validate models you can just run the following:

.. code::

   from dymoval.validation import validate_models

   vs = validate_models(
       measured_in=u_meas,
       measured_out=y_meas,
       simulated_out=y_sim,
       sampling_period=sampling_period
   )

where `y_sim` is the simulated out, `u_meas` is the measured input, `y_meas`
is the measured out arranged in :math:`N\times q`, :math:`N\times p` and
:math:`N\times q` arrays, respectively, where :math:`N` is the number of
observations sampled with period `sampling_period`, :math:`p` is the number of
inputs and :math:`q` is the number of outputs.

Correlation lags use one-sample spacing. Choose the lag-window sizes to cover
the physical delays relevant to your application. Signal bandwidth
arguments are no longer accepted; no automatic lag thinning is performed.

The function :py:meth:`~dymoval.validation.validate_models` returns a
:py:class:`~dymoval.validation.ValidationSession` object that stores the
validation outcome.

If you have other simulated data, coming from other models or from the same
model with different settings, then you can append them to the same
:py:class:`~dymoval.validation.ValidationSession` object. The evaluation is
done automatically:

.. code::

   # vs is a ValidationSession object
   vs = vs.append_simulation(sim_name='Sim_1', y_names=['out0'], y_data=y_sim2)
   print(vs)

The report shows the validation settings, thresholds, scores for each
simulation, and PASS/FAIL outcomes.

The same numbers are available programmatically through
:py:attr:`~dymoval.validation.ValidationSession.validation_statistics`,
:py:attr:`~dymoval.validation.ValidationSession.validation_thresholds` and
:py:attr:`~dymoval.validation.ValidationSession.outcome`, all of which are
plain dictionaries.

The report includes the following quantities:

-  *Input auto-correlation (optional)*,
-  :math:`R^2` fit of the measured and simulated outputs,
-  *Residual root mean squared error (RMSE)*,
-  *Residual auto-correlation*,
-  *Input-residual cross-correlation*,

You can finally visually inspect both the simulations results with the
:py:meth:`~dymoval.validation.ValidationSession.plot_simulations` method and
the residuals with the
:py:meth:`~dymoval.validation.ValidationSession.plot_residuals` method.

*******************************
 How to interpret the results?
*******************************

The scores describe different aspects of the experiment and the model:

-  **R-squared fit** measures agreement with the measured outputs relative
   to their variation. Higher is better, but a high value does not rule out
   missing dynamics.

-  **Residual RMSE** measures the magnitude of the errors
   :math:`\varepsilon = y_{\mathrm{measured}} - y_{\mathrm{simulated}}`.
   It is computed as :math:`\sqrt{\mathrm{mean}(\varepsilon^2)}` per output,
   without removing the residual mean, so bias contributes too. Lower is
   better; zero means a perfect match. Unlike R-squared, RMSE retains the
   output units.

-  **Input auto- and cross-correlation** describe repetition and redundancy
   in the excitation, not model quality. Low scores are generally desirable,
   but do not establish that all relevant behavior was exercised. Colored
   inputs can also be useful. See :ref:`doe` for input-excitation assessment.

-  **Residual auto-correlation:** a large score means the residuals contain
   structure that the model does not explain. This may indicate missing or
   incorrectly modeled dynamics, but it can also result from disturbances,
   nonlinearities, or measurement problems. Correlation describes patterns
   over time; RMSE describes error magnitude. Neither replaces the other.

-  **Input-residual cross-correlation:** a large score means the residuals
   remain correlated with the inputs. This can indicate that the model has
   not captured part of the input-output relationship. The cause may be missing
   dynamics, incorrect parameters, nonlinear behavior, or other model
   limitations; the correlation does not identify one particular state-space
   matrix as the cause.

Correlation scores summarize the selected lags and statistics: lower means
less correlation. They are not formal statistical tests. Low residual
correlation alone does not make a model suitable: acceptable fit and error
magnitude depend on what you want to use the model for.

.. note::

   Small residual correlations are generally desirable, but are not a
   universal requirement for a correct simulation model. Colored sensor
   noise or unmeasured disturbances can produce correlated residuals even
   when the modeled dynamics are accurate. These simulation residuals are
   not necessarily the one-step prediction errors of a model that also
   describes the noise.

   Clean, meaningful measurements are essential: check timing alignment,
   sensor quality, excitation, and operating conditions before attributing
   residual patterns to the model. Interpret correlation alongside output
   plots, RMSE, and knowledge of the experiment, rather than trying to make
   the score small at any cost.

Both R-squared and residual RMSE are stored as scalar values for each
simulation in ``vs.validation_statistics``, under ``"r2"`` and ``"rmse"``.
For multiple outputs, ``r2_statistic="min"`` selects the lowest fit by
default, while ``rmse_statistic="max"`` selects the largest per-output RMSE.
Both also support ``"mean"``. Outputs should have comparable units and
scaling before aggregating RMSE.

The correlation keys are ``"Ruu_correlation"``, ``"Ree_correlation"``, and
``"Rue_correlation"``. The report compares scores with the configured
thresholds: R-squared must be strictly above its threshold, and the other
scores strictly below theirs. All configured checks must pass.
Use ``ignore_input=True`` to disable the input-correlation check.

There is no RMSE threshold enabled by default, since an acceptable error
depends on the output units and application. To add one while retaining the
current checks:

.. code:: python

   thresholds = dict(vs.validation_thresholds)
   thresholds["rmse"] = 0.1  # Example only: choose a meaningful tolerance.
   vs.validation_thresholds = thresholds

Choose thresholds for your application, or use the underlying data to build
your own evaluation criteria.

In any case, it is important that you deliver your model along with the
validation results and the coverage region, so users know within which limits
they can trust the model.

Envelope-based residual auto-correlation score (proposed)
======================================================================

A single correlation threshold treats every delay alike. Sometimes you have
reason to tolerate stronger residual correlation at short delays, but want
to reject correlation that persists for longer. An *acceptance envelope*
expresses those different limits.

First, distinguish the entries of the residual correlation matrix
:math:`R_{\varepsilon\varepsilon}`, which has shape :math:`q\times q`:

* The diagonal entries compare each residual :math:`\varepsilon_i` with
  delayed copies of itself. These are scalar auto-correlations.
* The off-diagonal entries compare different residual components
  :math:`\varepsilon_i` and :math:`\varepsilon_j`. These are
  cross-correlations, even though they belong to the vector's
  auto-correlation matrix.

The proposed envelope applies only to the diagonal entries. The
off-diagonal entries can retain existing statistics such as ``abs_mean``
or ``max``. All entries of the input-residual matrix
:math:`R_{u\varepsilon}` are cross-correlations, including
:math:`u_i` versus :math:`\varepsilon_i`: matching indices do not make
them the same signal.

The envelope is a positive limit curve :math:`b(t)` that you choose, not a
curve fitted to the observed auto-correlation and not its derivative.
At each evaluated nonzero lag :math:`k`, compare the absolute normalized
residual auto-correlation with the limit at delay :math:`|k|T_s`:

.. math::

   |\hat\rho_\varepsilon(k)| \leq b(|k|T_s).

One scalar can summarize the worst violation:

.. math::

   S = \max_{k \ne 0}
       \frac{|\hat\rho_\varepsilon(k)|}{b(|k|T_s)}.

Here, :math:`S \leq 1` means every evaluated lag is within its limit;
:math:`S>1` means at least one exceeds it. Unlike an average, adding
near-zero correlations cannot dilute an existing violation. The result
only covers the evaluated delay window, not all possible delays.
The zero-lag auto-correlation peak is excluded, as in the existing scores.

How would you choose the envelope?
----------------------------------

A Gaussian-shaped limit with a positive floor is a useful starting point
when you have reason to tolerate a rounded central cluster:

.. math::

   b(t) = b_\infty + (b_0-b_\infty)\exp[-(t/T_b)^2],
   \qquad 0 < b_\infty \leq b_0 \leq 1,\quad T_b>0.

* :math:`b_0` sets the short-delay tolerance. Increasing it permits
  stronger correlation near zero, but does not exclude those delays.
* :math:`b_\infty` sets the long-delay tolerance. Lowering it makes
  persistent correlation harder to accept.
* :math:`T_b` controls how quickly the limit tightens. Increasing it
  tolerates short-delay correlation for longer. It is an acceptance
  time scale, not automatically the output's time constant. At
  :math:`t=T_b`, the allowance above the floor has fallen to
  :math:`1/e` of its initial value.

Compared with an exponential limit, this shape is flatter near zero and
tightens more sharply later. Neither shape is universally correct, and
residual auto-correlations need not be Gaussian. Choose a limit that
expresses acceptable behavior, not simply one that resembles the data.

For illustration only, take :math:`b_0=0.95`, :math:`b_\infty=0.1`, and
:math:`T_b=5` seconds. The limit is about 0.92 at one second and 0.12 at
ten seconds. Correlation of 0.75 at one second would meet its limit,
while correlation of 0.88 at ten seconds would not. These numbers are
not recommended defaults.

.. figure:: ../figures/ResidualCorrelationEnvelope.svg
   :alt: A decaying acceptance limit allows correlation of 0.75 at one
         second but rejects correlation of 0.88 at ten seconds.
   :align: center
   :width: 100%

   An acceptance envelope checks short delays rather than ignoring them,
   while imposing tighter limits at longer delays.

Choose the limits using application requirements, sensor/filter knowledge,
or independent reference experiments. Do not tune them merely until the
candidate model passes. A slow measured output alone does not justify a
permissive short-delay envelope: its residuals may have a different time
scale. Finite-record uncertainty matters too; an envelope below the normal
estimation fluctuations can reject useful models.

A trusted reference model can also provide an empirical baseline. If its
performance is known to be acceptable, use its diagonal residual
auto-correlations to help define the envelope. Prefer several representative
logs and include a margin for run-to-run variation rather than fitting a
limit tightly to one curve. Freeze the envelope before evaluating candidate
models on separate data, keeping sampling, preprocessing, operating
conditions, and evaluated delays comparable.

This asks whether a candidate's residual dependence stays within accepted
reference behavior, not whether its residuals are statistically white.
Retain an RMSE criterion too: normalized correlation can look identical
even when the candidate's errors are much larger.

Keep the envelope and maximum evaluated physical delay fixed when comparing
models under comparable experimental conditions. Inspect the full
correlation plot and RMSE alongside the score. Different outputs can need
different envelopes; a combined score could take the largest violation
across outputs without mixing their physical units.

This is an optional application-specific acceptance criterion, not a
universal requirement, a statistical confidence band, or a formal test.

Envelope scores are ratios to a limit, whereas existing correlation scores
summarize correlation magnitudes. Do not feed both directly into the same
global aggregation as though they had the same meaning. Assess them
separately, or divide each ordinary score by its own positive acceptance
threshold before combining violation ratios.

.. note::

   Envelope-based scoring is not implemented yet. The formula above defines
   a proposed criterion; it is not an available API option and does not
   change the existing scores or PASS/FAIL behavior.

What about the cross-correlation entries?
-----------------------------------------

The off-diagonal residual entries and all input-residual entries can keep
the current statistics, including zero lag. There is no self-correlation
peak that must be excluded for these pairs.

Different residual outputs can share disturbances or sensor effects, so
their cross-correlation need not vanish even for accurate modeled dynamics.
Treat these entries as diagnostics unless low cross-correlation is a
justified application requirement.

When residuals are expected to be uncorrelated with the inputs, the target
is approximately zero at every evaluated delay, including zero delay.
A constant acceptance limit is then a reasonable starting point: there is
no auto-correlation peak of one that must be excluded.

The existing ``abs_mean`` and ``max`` statistics remain useful summaries.
However, ``abs_mean`` can hide a localized cross-correlation peak and can
be diluted by extending the lag window. ``max`` checks the strongest peak;
dividing it by a constant positive limit is the constant-envelope version
of the score above.

Neither provides a universal threshold. Finite-data fluctuations depend
on record length and signal correlation, and feedback or disturbances
related to the inputs can complicate the interpretation. A flat
acceptance limit is a choice, not a guarantee of a flat observed
cross-correlation or a constant statistical uncertainty.

*******************************
 The results are disappointing
*******************************

When the validation results are bad does not necessarily mean that the model
is bad. It may be that the validation procedure needs some tweak. Here are few
things to check:

-  Noisy measurements can obscure the response you want to validate.
   Filtering may help, but can also hide model errors or introduce
   correlation and phase shifts. If you filter, define the frequency range
   you intend to validate and treat measured and simulated outputs
   consistently. Do not smooth the data merely to improve a score.

-  Slowly varying residuals can have large correlations at short delays.
   Inspect correlation against physical delay and choose a meaningful
   evaluation window. Do not automatically discard short delays: they may
   reveal model errors. Increasing the window can dilute an ``abs_mean``
   score with small correlations; keep the window fixed when comparing
   models, or use ``max`` to avoid that dilution.

-  The input signal has some trend or some large mean values or offset, etc..
   The normalized correlation diagnostics already subtract signal means;
   they do not automatically remove trends. Any preprocessing should be
   consistent with the model's input/output conventions. Residual offsets
   do not necessarily cancel: a bias contributes to RMSE even when it is
   removed by the correlation calculation.

.. _theory:

**********************************
 Some theory: what are residuals?
**********************************

The residuals, denoted as :math:`\varepsilon`, are simply the error between
the measured outputs and the simulated outputs, defined as :math:`\varepsilon
= y_{\mathrm{measured}} - y_{\mathrm{simulated}}`.

Residual correlation helps us look for structure left in the errors.
For a signal :math:`x(t), t=1,\dots,N`, auto-correlation measures similarity
with delayed copies of itself. The function :math:`r_{x,x}(k)` is
called *auto-correlation function (ACF)* of the signal :math:`x(t)` and it
represents how similar :math:`x(t)` is with delayed copies of itself at
different lags :math:`k\in \mathbb Z`. If instead of considering one signal we
consider two signals :math:`x(t)` and :math:`y(t)`, then we obtain the
*cross-correlation function (CCF)* :math:`r_{x,y}(k)` between :math:`x(t)` and
:math:`y(t)`.

The next question is: to how many seconds one lag corresponds to?

One lag corresponds to one sampling period :math:`T_s`, so lag :math:`k`
represents a physical delay :math:`kT_s`. Signals being compared must share
the same sampling period. No automatic bandwidth-based lag thinning is used.

To inspect delays up to :math:`KT_s` in either direction, request
``nlags=2*K+1`` for each channel pair. In a validation session, these sizes
are supplied as arrays through ``Ruu_nlags``, ``Ree_nlags``, and
``Rue_nlags``. For example, with :math:`T_s=0.5` seconds, a window of 41
points covers delays from -10 to +10 seconds, provided the record is long
enough. Returned ``lags`` remain integer sample offsets; multiply them by
:math:`T_s` to express delays in time units.

Increasing the window extends the inspected delay range; it does not
change the spacing or exclude the central region. A slow measured output
does not necessarily imply slow residuals: the modeled slow response may
cancel when taking their difference.

If residual correlation stays high across the displayed window, extending
it can reveal how long the pattern persists. However, an ``abs_mean`` score
depends on the window: adding many near-zero values can lower the score
without improving the model. The ``max`` statistic avoids this dilution,
but does not distinguish an isolated peak from persistent correlation.
Choose and report the window, statistic, and threshold together, keeping
them fixed across model comparisons. Neither statistic is a universal
model-quality test.

*Dymoval* also summarizes correlation for **multivariate signals** as
follows.

Let :math:`X` a signal of dimension :math:`p` with :math:`N` observations. The
correlation scoring of :math:`X` is performed in three steps:

#. That is, the `auto-/cross-correlation` functions :math:`r_{i,j}(k)` of each
   pair of components :math:`x_i, x_j \in X` for :math:`i,j = 1 \dots p` is
   computed and arranged in :math:`p\times p`
   :py:class:`~dymoval.xcorrelation.XCorrelation` object.

#. For each element :math:`r_{i,j}(k), i,j = 1 \dots p` of the
   :py:class:`~dymoval.xcorrelation.XCorrelation` object, correlation is
   summarized over the selected symmetric lag window (by default, 20
   negative lags, zero, and 20 positive lags). The default statistic is the *mean of
   the absolute value* of the realizations of the
   :py:class:`~dymoval.xcorrelation.XCorrelation` function. The results are
   arranged in a :math:`p\times p` array where each element is `float`.

#. Another statistic is finally computed on the resulting flattened array. By
   default, *dymoval* take the :math:`\max` element of such an array, which
   corresponds to the largest local correlation score.

*Dymoval* considers **normalized** correlation functions, whose values are
between :math:`-1.0` and :math:`1.0`. For an auto-correlation, the value at
zero lag is one for a nonconstant channel. These diagonal zero-lag peaks
are excluded from the score; cross-correlations between different channels
still include zero lag.
