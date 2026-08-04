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
   vs

   Validation session name: quick & dirty

   Validation setup:
   ----------------
   Inputs auto-correlation
   Statistic: abs_mean-max
   Ruu_local_weights: None
   Ruu_global_weights: None
   num lags: 41

   Residuals auto-correlation:
   Statistic: abs_mean-max
   Ree_local_weights: None
   Ree_global_weights: None
   num lags: 41

   Input-residuals cross-correlation:
   Statistic: abs_mean-max
   Rue_local_weights: None
   Rue_global_weights: None
   num lags: 41

   Validation results:
   -------------------
   Thresholds:
   Ruu_whiteness: 0.6000
   r2: 35.0000
   Ree_whiteness: 0.5000
   Rue_whiteness: 0.5000

   Actuals:
                                              Sim_0         Sim_1
   Input whiteness (abs_mean-max)            0.0367        0.0367
   R-Squared (%)                            99.8454       66.4326
   Residuals whiteness (abs_mean-max)        0.1515        0.3663
   Input-Res whiteness (abs_mean-max)        0.0795        0.1168

            Sim_0  Sim_1
   Outcome: PASS   FAIL

The same numbers are available programmatically through
:py:attr:`~dymoval.validation.ValidationSession.validation_statistics`,
:py:attr:`~dymoval.validation.ValidationSession.validation_thresholds` and
:py:attr:`~dymoval.validation.ValidationSession.outcome`, all of which are
plain dictionaries.

The default validation procedure evaluates the following quantities:

-  *Input whiteness (optional)*,
-  :math:`R^2` fit of the measured and simulated outputs,
-  *Residuals whiteness*,
-  *Input-Residuals whiteness*,

and compare them with some thresholds. If each individual quantity pass the
test, then the overall test is passed. It is however possible to access all
the validation information for defining custom evaluation criteria in a fairly
easy manner since it is possible to extract any kind on information from
:py:class:`~dymoval.validation.ValidationSession` objects.

You can finally visually inspect both the simulations results with the
:py:meth:`~dymoval.validation.ValidationSession.plot_simulations` method and
the residuals with the
:py:meth:`~dymoval.validation.ValidationSession.plot_residuals` method.

*******************************
 How to interpret the results?
*******************************

The :math:`R^2` index tells us how well the simulation results fit the
measurement data, whereas the residuals provide information about the dynamic
behavior of our model. More precisely:

-  If the input signal has a low whiteness value (i.e., as close to 0.0 as
   possible), it means that during the lab tests, the system was adequately
   stimulated, covering all aspects of the real system. This gives higher
   trust to our model if the other validation metrics are good.

-  If the residuals' whiteness is large, the residual sequence still contains
   structure that the model does not explain. This may indicate missing or
   incorrectly modeled dynamics, but it can also result from disturbances,
   nonlinearities, or other modeling assumptions. The magnitude of the
   residuals is a separate measure: large residual values indicate a poor
   pointwise fit, whereas large residual whiteness indicates remaining
   correlation over time. A large :math:`R^2` value only means that the model
   fits the measured outputs well overall; it does not rule out unmodeled
   dynamics.

-  If the input-residuals' whiteness level is large, the residuals remain
   correlated with the inputs. This indicates that the model has not captured
   some part of the input-output relationship. The cause may be missing
   dynamics, incorrect parameters, nonlinear behavior, or other model
   limitations; the correlation does not identify one particular state-space
   matrix as the cause.

For simulation models, which motivated the development of *dymoval*, we are
more interested in the dynamic behavior of models than the point-wise fit of
the data. Hence, even if the :math:`R^2` index is low, the model can still be
very useful in a simulation setting, provided that the residuals are white
enough.

The default validation process offered by Dymoval consists of comparing these
values with some adjustable thresholds. You can tune such thresholds depending
on how stringent you want to be with your model, but you can also fetch raw
data and build up the criteria you want.

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
   Consider removing possible trends, mean values, etc. from the input signals
   used in the :ref:`Dataset <Dataset>` object contained in the
   :ref:`ValidationSession <ValidationSession>` object. *You don't need to do
   it in the output signals because eventual trends or mean values are
   canceled out during the computation of the residuals*. However, to generate
   the simulated data the input signal shall be as close as possible to the
   input signal used in the test. Hence, you may consider **two distinct input
   signals**: one for feeding the model and a manipulated version of it for
   validation purpose that is included in the :ref:`ValidationSession
   <ValidationSession>` object.

-  *Stiff models*: *dymoval* can naturally cope with stiff models, but it is
   very important to exploit bandwidths information. However, you can ignore
   entries in the resulting validation matrix described in point 2. of the
   next Section when performing an overall assessment if they represents
   signal with significantly different bandwidths. This means that you should
   extract information from the :ref:`ValidationSession <ValidationSession>`
   object and build custom evaluation metrics.

.. _theory:

**********************************
 Some theory: what are residuals?
**********************************

The residuals, denoted as :math:`\varepsilon`, are simply the error between
the measured outputs and the simulated outputs, defined as :math:`\varepsilon
= y_{\mathrm{measured}} - y_{\mathrm{simulated}}`.

It is desirable for the residuals to be as `white` as possible.

In general, to examine the whiteness of a signal :math:`x(t), t=1,\dots,N`, we
study its similarity with some of its delayed copies. If such a similarity is
small for a sufficiently high number of *lags*, then we can say that the
signal :math:`x(t)` is somewhat *white*. The function :math:`r_{x,x}(k)` is
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
:math:`\tau = \min(1/(2B_x), 1/(2B_y))`.

*Dymoval* also performs whiteness analysis of **multivariate signals** as
follows.

Let :math:`X` a signal of dimension :math:`p` with :math:`N` observations. The
whiteness estimation of :math:`X` is performed in three steps:

#. That is, the `auto-/cross-correlation` functions :math:`r_{i,j}(k)` of each
   pair of components :math:`x_i, x_j \in X` for :math:`i,j = 1 \dots p` is
   computed and arranged in :math:`p\times p`
   :py:class:`~dymoval.xcorrelation.XCorrelation` object.

#. For each element :math:`r_{i,j}(k), i,j = 1 \dots p` of the
   :py:class:`~dymoval.xcorrelation.XCorrelation` object the whiteness is
   estimated by computing a statistic of its realizations for
   :math:`k=-n_{lags}, \dots, n_{lags}`, being :math:`n_{lags} >0` the number
   of lags considered (20 by default). The default statistic is the *mean of
   the absolute value* of the realizations of the
   :py:class:`~dymoval.xcorrelation.XCorrelation` function. The results are
   arranged in a :math:`p\times p` array where each element is `float`.

#. Another statistic is finally computed on the resulting flattened array. By
   default, *dymoval* take the :math:`\max` element of such an array, which
   correspond to the *worst-case* whiteness estimate.

*Dymoval* considers **normalized** correlation functions, whose values are
between :math:`-1.0` and :math:`1.0`. For an auto-correlation, the value at
zero lag is one and is excluded from the whiteness estimate; whiteness is
assessed from the non-zero lags.
