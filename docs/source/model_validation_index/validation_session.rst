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

Dymoval validation procedure evaluates the following quantities:

-  R-square fit of the measured and simulated outputs,
-  Residuals whiteness,
-  Input-Residuals whiteness,

and compare them with some thresholds.

Assuming that The whiteness is computed in 3 steps:

#. The auto- and cross- correlation for each pair of component is computed.
   This results in a 2D array where each element is a XCorrelation object

#. For each XCorrlation element the whiteness is computed (default statistic
   is abs_mean), resulting in a qxq array, where each element is a float
   number

#. The resulting array is flattened and a statistic is computed on it (default
   ``max`` which is the "worst-case" element)

*********************
 What are residuals?
*********************

The residuals, denoted as :math:`\varepsilon`, are simply the error between
the measured outputs and the simulated outputs, defined as :math:`\varepsilon
= y_{\mathrm{measured}} - y_{\mathrm{simulated}}`. During residuals analysis,
we examine whether there is any correlation of the residuals with their
delayed copies. The value of the residuals with respect to different lags is
also called the auto-correlation function (ACF), in case you want to look it
up.

It is desirable for the residuals to be as white as possible, meaning that
their auto-correlation values are as close to 0.0 as possible for all lags. If
this is not the case, it indicates that there are underlying dynamics that
have not been modeled.

You can visually inspect both the simulations results with the
:py:meth:`~dymoval.validation.ValidationSession.plot_simulations` method and
the residuals with the
:py:meth:`~dymoval.validation.ValidationSession.plot_residuals` method.

The **coverage region** can be shown through the
:py:meth:`~dymoval.dataset.Dataset.plot_coverage()` of the stored
:ref:`Dataset <Dataset>`.

*******************************
 How to interpret the results?
*******************************

The R-squared index tells us how well the simulation results fit the
measurement data, whereas the residuals provide information about the dynamic
behavior of our model. More precisely:

-  If the input signal has a low whiteness value (i.e., as close to 0.0 as
   possible), it means that during the lab tests, the system was adequately
   stimulated, covering all aspects of the real motor. This gives higher trust
   to our model if the validation metrics are good.

-  If the residuals' whiteness is large, it indicates that some dynamics have
   been poorly modeled, and therefore the model needs updates. In this case,
   if the R-squared is large, it only means that your model is fitting well
   what has been modeled, but there are still many aspects not modeled. If our
   model is of the form :math:`\dot x = Ax + Bu`, then the model between
   :math:`x` and :math:`\dot x` shall be revised.

-  If the input-residuals' whiteness level is large, it means that the
   input-output model needs improvements. If our model is of the form
   :math:`\dot x = Ax + Bu`, then the model between :math:`u` and :math:`\dot
   x` shall be revised.

For simulation models, which motivated the development of Dymoval, we are more
interested in the dynamic behavior of models than the point-wise fit of the
data. Hence, even if the R-squared index is low, the model can still be very
useful in a simulation setting, provided that the residuals are white enough.

The default validation process offered by Dymoval consists of comparing these
values with some adjustable thresholds. You can tune such thresholds depending
on how stringent you want to be with your model, but you can also fetch raw
data and build up the criteria you want.

In any case, it is important that you deliver your model along with the
validation results and the coverage region, so users know within which limits
they can trust the model.

********************************
 The results are disappointing.
********************************
