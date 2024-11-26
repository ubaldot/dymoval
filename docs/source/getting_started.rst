#################
 Getting Started
#################

********************************
 Model validation in a nutshell
********************************

Model validation's job is to evaluate the quality of your models.

The process happens in four steps:

#. **Plan** some test to execute on the real system. The goal is to stimulate
   the real-world system with input signals that are random as possible. You
   should try to hit every corner of the system.

#. **Collect** the measured inputs and outputs of your real system,

#. **Simulate** your model with the same input that you used to stimulate your
   real system and log the simulated output,

#. **Evaluate** your model quality through Dymoval by feeding it with measured
   and simulated data.

The following figure summarize the process:

.. figure:: ./figures/ModelValidationDymoval.svg
   :scale: 50 %

   The model validation process.

If the results of the last point are good, then your model is good to go.

Operational speaking, suppose you want to validate a model and you have the
simulated out ``y_sim`` the measured input ``u_meas``, and the measured out
``y_meas`` arranged in :math:`N\times q`, :math:`N\times p` and :math:`N\times
q` ``np.ndarray``, respectively, where :math:`N` is the number of observations
sampled with period ``sampled_period``, :math:`p` is the number of inputs and
:math:`q` is the number of outputs. Just call the following function:

.. code::

   from dymoval.validation import validate_models

   validate_models(
       measured_in=u_meas,
       measured_out=y_meas,
       simulated_out=y_sim,
       sampling_period = sampling_period
   )

to get something like the following:

.. code::

   Input whiteness (abs_mean-max)      0.3532
   R-Squared (%)                      65.9009
   Residuals whiteness (abs_mean-max)  0.1087
   Input-Res whiteness (abs_mean-max)  0.2053

            My_Model
   Outcome: PASS

and are good to go. The model quality is evaluated according to the following
criteria:

-  **R-squared** index: A good model should have this as large as possible.

-  **Residuals auto-correlation**: A good model should have this as close to
   white noise as possible (Residuals whiteness nearly equal to 0.0).

-  **Input-residuals cross-correlation**: A good model should have this as
   close to white noise as possible (Input-residuals whiteness nearly equal to
   0.0).

Nevertheless, given that "*all models are wrong, but some are useful,*" we
cannot expect perfect figures. However, since we are interested in the dynamic
behavior of our models, residuals are somewhat more important than the
R-squared match (that does not mean the R-squared can be very bad!).

However, it is worth nothing that **it does not matter what simulation tool
you use**. Dymoval only look at the simulated output values and make an
evaluation versus the measurement data.

Now that you understand the process, you can gain hands-on experience with the
tutorial. If you want to learn more about model validation, how Dymoval
relates to it, and how to address issues when the results are disappointing,
feel free to check the :ref:`model_validation` section.

**********
 Tutorial
**********

The tutorial is tailored to mimic what happens in a typical development
environment where different teams are involved. You can start the tutorial by
typing the following lines in a Python console:

.. code::

   import dymoval as dmv
   dmv.open_tutorial()

The :py:meth:`~dymoval.utils.open_tutorial()` function creates a
``~/dymoval_tutorial`` folder containing all the files needed to run the
tutorial. All you have to do is to run Jupyter notebook named
``dymoval_tutorial.ipynb``. You need an app for opening ``.ipynb`` files.

The content of the ``dymoval_tutorial`` folder will be overwritten every time
this function is called.

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

***************************
 But why model validation?
***************************

Imagine you are developing an innovative product. At various stages, you need
to test it. Based on the test outcomes, you adjust your development direction.
This cycle of development and testing continues iteratively until you achieve
something deployable.

Typically, testing in the target environment—the real-world setting where
your product will ultimately be deployed—incurs costs in terms of money,
time, and often personal stress.

To alleviate these challenges, you can run your tests in a virtual environment
instead. If your work-product performs well in this virtual setting, it should
theoretically perform well in the real-world environment too.

However, this assumption holds true only if your virtual environment
accurately represents the target environment and behaves similarly. And this
is what model validation is all about.
