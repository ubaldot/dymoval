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
       sampling_period = sampling_period
   )

where `y_sim` is the simulated out, `u_meas` is the measured input, `y_meas`
is the measured out arranged in :math:`N\times q`, :math:`N\times p` and
:math:`N\times q` arrays, respectively, where :math:`N` is the number of
observations sampled with period `sampled_period`, :math:`p` is the number of
inputs and :math:`q` is the number of outputs.

For more accurate results, the bandwidths of the involved signals can be
passed to the *dymoval* functions.

The function :py:meth:`~dymoval.validation.validate_models` return a
:py:class:`~dymoval.validation.ValidationSession` object that store the
validation outcome.

If you have other simulated data coming from other models or from the same
model with different settings, then you can append them to the same
:py:class:`~dymoval.validation.ValidationSession` object. The evaluation is
done automatically:

.. code::

   # vs is a ValidationSession object
   vs = vs.append_simulation(name='Sim_1', y_names=['out0', 'out1'], y_data=y_sim2)
   vs

   Actuals:
                                           Sim_0       Sim_1
   Input whiteness (abs_mean-max)       0.367134    0.367134
   R-Squared (%)                       69.538448 -173.163879
   Residuals whiteness (abs_mean-max)   0.116023    0.146618
   Input-Res whiteness (abs_mean-max)   0.189000    0.263764

            Sim_0  Sim_1
   Outcome: PASS   FAIL

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

-  If the residuals' whiteness is large, it indicates that some dynamics have
   been poorly modeled, and therefore the model needs updates. In this case,
   if the :math:`R^2` value is large, it only means that your model is fitting
   well what has been modeled, but there are still underlying, non-modeled
   dynamics. If the model is of the form :math:`\dot x = Ax + Bu`, then the
   model between :math:`x` and :math:`\dot x`, namely the matrix :math:`A`,
   shall be revised.

-  If the input-residuals' whiteness level is large, it means that the
   input-output model needs improvements. If our model is of the form
   :math:`\dot x = Ax + Bu`, then the model between :math:`u` and :math:`\dot
   x`, namely the matrix :math:`B`, shall be revised.

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
   bandwidths and pass this information to *dymoval* functions.

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
\mathbb{R}^+` which in *dymoval* is equal to :math:`\tau = T_s`, being
:math:`T_s` the signal sampling period, or equal to :math:`\tau=1/2B_x`, where
:math:`B_x` is the bandwidth of :math:`X`, if the value of :math:`B_x` is
passed to *dymoval* API. It is not possible to take a larger lag time than
:math:`\tau=1/2B_x` otherwise the Nyquist-Shannon criteria would be violated.

In case of cross-correlation between two signals :math:`x` and :math:`y` the
`lag time` :math:`\tau` is equal to the sampling period :math:`T_s` of the
signals - that must be the same - or to :math:`\tau = \min(1/2B_x, 1/2B_y)` if
information about the bandwidths are passed to the *dymoval* API.

*Dymval* also performs whiteness analysis of **multivariate signals** as it
follows.

Let :math:`X` a signal of dimension :math:`p` with :math:`N` observations. The
whiteness estimation of :math:`X` is performed in three steps:

#. That is, the `auto-/cross-correlation` functions :math:`r_{i,j}(k)` of each
   pair of components :math:`x_i, x_j \in X` for :math:`i,j = 1 \dots p` is
   computed and arranged in :math:`p\times p`
   :py:class:`~dymoval.validation.XCorrelation` object.

#. For each element :math:`r_{i,j}(k), i,j = 1 \dots p` of the
   :py:class:`~dymoval.validation.XCorrelation` object the whiteness is
   estimated by computing a statistic of its realizations for
   :math:`k=-n_{lags}, \dots, n_{lags}`, being :math:`n_{lags} >0` the number
   of lags considered (20 by default). The default statistic is the *mean of
   the absolute value* of the realizations of the
   :py:class:`~dymoval.validation.XCorrelation` function. The results are
   arranged in a :math:`p\times p` array where each element is `float`.

#. Another statistic is finally computed on the resulting flattened array. By
   default, *dymoval* take the :math:`\max` element of such an array, which
   correspond to the *worst-case* whiteness estimate.

*Dymoval* consider **normalized** correlation functions, which means that the
values of the *ACF*:s and *CCF*:s are always **between 0.0 and 1.0**.
