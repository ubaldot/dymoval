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

For more accurate results, the bandwidths of the involved signals can be
passed to the *dymoval* functions.

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

*******************************
 The results are disappointing
*******************************

When the validation results are bad does not necessarily mean that the model
is bad. It may be that the validation procedure needs some tweak. Here are few
things to check:

-  The measurements dataset has noisy measurements. In that case you want to
   low-pass filter the dataset, but avoid to use tight cutoff frequencies
   because that would smooth the signal too much, possibly resulting in high
   ACF values. Also, it is worth nothing that the bandwidth of a signal
   downstream a first-order low-pass filter is, in general, not equal to the
   filter cutoff frequency.

-  The signals may be over-sampled. Consider estimating the signals'
   bandwidths and passing this information to *dymoval* functions. This lets
   *dymoval* evaluate the correlation on a bandwidth-appropriate lag grid.
   Alternatively, increase the lag window and inspect the correlation outside
   the region around zero lag, where the short sampling interval can dominate
   the result. When no bandwidth-based lag grid is used, each lag corresponds
   to one sampling period.

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

The answer is given by the `delay time` (or `lag time`) :math:`\tau \in
\mathbb{R}^+`. Without bandwidth information, *dymoval* uses the signal
sampling period :math:`T_s` as the spacing between adjacent lags, so lag
:math:`k` corresponds to a delay of approximately :math:`kT_s`. When bandwidth
information is supplied, *dymoval* uses a reduced lag grid with an effective
spacing of approximately :math:`\tau=1/(2B_x)`, where :math:`B_x` is the
bandwidth of :math:`X`.

This bandwidth-based spacing should not be confused with the maximum lag that
may be inspected. The Shannon-Nyquist condition limits the sampling interval
to :math:`T_s \leq 1/(2B_x)`; it does not prevent inspecting larger delays.
To study the system memory, the lag window can extend well beyond
:math:`1/(2B_x)`, for example by increasing ``nlags`` and checking whether
the correlation has decayed outside the dominant time constants.

In case of cross-correlation between two signals :math:`x` and :math:`y`, the
signals must have the same sampling period. Without bandwidth information,
the lag spacing is :math:`T_s`; with bandwidth information, it is
:math:`\tau \approx \min(1/(2B_x), 1/(2B_y))`.

This is a practical way to inspect delays on a bandwidth-related time
scale, not a guarantee that samples at those delays are uncorrelated.
Nyquist does not prescribe a correlation lag grid. The actual spacing is
an integer multiple of the sampling period, rounded down from the target
spacing and limited by record length. Only the evaluated correlation lags
are thinned; the input signals are not downsampled.

The grid can make oversampled data easier to assess, but may skip short-delay
features. Inspect the full-resolution correlation too when those features
matter, and do not compare aggregate scores from different grids as though
they measured exactly the same thing.

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
