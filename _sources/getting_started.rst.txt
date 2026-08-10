#################
 Getting Started
#################

Suppose you want to validate a model and you have the simulated out ``y_sim``
the measured input ``u_meas``, and the measured out ``y_meas`` arranged in
:math:`N\times q`, :math:`N\times p` and :math:`N\times q` arrays,
respectively, where :math:`N` is the number of observations sampled with
period ``sampled_period``, :math:`p` is the number of inputs and :math:`q` is
the number of outputs. Just call the following function:

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

The model quality is evaluated according to the following criteria:

-  **Input whiteness (optional)**: A trustworthy model shall have this value
   is close to 0.0 (max value is 1.0).
-  :math:`\mathbf{R^2}`: A good model should have this value as large as
   possible.
-  **Residuals whiteness**: A good model should have this as close to 0.0 as
   possible (max value is 1.0).
-  **Input-residuals whiteness**: A good model should have this as close to
   0.0 as possible (max value is 1.0).

See :ref:`here <theory>` for more details how such metrics are computed.

Nevertheless, given that "*all models are wrong, but some are useful,*" we
cannot expect perfect figures. However, since we are interested in the dynamic
behavior of our models, residuals are somewhat more important than the
:math:`R^2` match (that does not mean the :math:`R^2` can be very bad!).

However, it is worth nothing that **it does not matter what simulation tool
you use**. *Dymoval* only look at the simulated output values and make an
evaluation versus the measurement data.

Now that you understand the process, you can gain hands-on experience with the
tutorial. If you want to learn more about model validation, how *dymoval*
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
