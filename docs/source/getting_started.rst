#################
 Getting Started
#################

********************************
 Model validation in a nutshell
********************************

Model validation's job is to evaluate the quality of your models.

The process happens in four steps:

#. **Plan** some test to execute on the real system. The goal is to
   stimulate the real-world system with input signals that are
   random as possible. You should try to hit every corner of the system, by
   feeding the system with as random input as possible (PRBS sequences, chirp
   signals, etc.).

#. **Collect** the measured inputs and outputs while executing the
   tests planned in the previous step on the real system,

#. **Simulate** your model with the same input that you used to stimulate the
   real system and log the simulated output,

#. **Evaluate** your model quality by comparing the simulated outputs with the
   measured outputs.

.. figure:: ./figures/ModelValidationDymoval.svg
   :scale: 50 %

   The model validation process.

My equation:

.. math::

      \int_{t_0}^t e^{-A(t-\tau)}u(\tau)\,d\tau

Dymoval only performs step 4. The model quality is evaluated
according to the following criteria:

-  **R-squared** index: A good model should have this as large as possible.
-  **Residuals auto-correlation**: A good model should have this as close to
   white noise as possible.
-  **Input-residuals cross-correlation**: A good model should have this as
   close to white noise as possible.

Just feed Dymoval with measurement data and simulated output, and you will get
the aforementioned metrics evaluated for free.

Nevertheless, given that "*all models are wrong, but some are useful,*" we
cannot expect perfect figures. However, since we are interested in the dynamic
behavior of the model, residuals are somewhat more important than the
R-squared match (which does not mean the R-squared can be very bad).

Now that you understand the process, you can gain hands-on experience with the
tutorial. If you want to learn more about model validation, how Dymoval
relates to it, and how to address issues when the results are disappointing,
feel free to check the :ref: `some_theory` section.

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

********************************
Enabling models CI/CD pipelines
********************************

A traditional software development workflow consists of pushing software
changes to a repository where an automation server (like Jenkins or GitLab)
automatically assesses whether your changes can be integrated into the
codebase by executing some tests.

*Dymoval* allows you to do the same with models. All you need is a good
measurement dataset and a mechanism to run simulations on an automation
server. Then, you can use the *Dymoval* API to validate the model changes and
decide whether to merge or reject the proposed changes.

..
   vim: set ts=3 tw=78:
