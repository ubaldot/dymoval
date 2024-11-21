#################
 Getting Started
#################

********************************
 Model validation in a nutshell
********************************

Model validation's job is to evaluate the quality of your models. To do that,
you need some measurement data and a model.

The process happens in four steps:

#. **Plan** some test to execute on the real system. The goal is to
   aggressively stimulate the real-world system with input signals that are
   random as possible. You should try to hit every corner of the system,

#. **Collect** the measured inputs and outputs while actually executing the
   tests planned in the previous step,

#. **Simulate** your model with the same input that you used to stimulate the
   real system and log the simulated output,

#. **Evaluate** your model quality by comparing the simulated outputs with the
   measured outputs.

.. figure:: ./figures/ModelValidationDymoval.svg
   :scale: 50 %

The model validation process.

Dymoval only performs step 4. The model quality evaluation is performed
according to the following criteria:

-  R-squared index: A good model should have this as large as possible.
-  Residuals auto-correlation: A good model should have this as close to zero
   as possible.
-  Input-residuals cross-correlation: A good model should have this as close
   to zero as possible.

Just feed Dymoval with measurement data and simulated output, and you will get
the aforementioned metrics evaluated for free.

Nevertheless, given that "*all models are wrong, but some are useful,*" we
cannot expect perfect figures. However, since we are interested in the dynamic
behavior of the model, residuals are somewhat more important than the
R-squared match (which does not mean the R-squared can be very bad).

Now that you understand the process, you can gain hands-on experience with the
tutorial. If you want to learn more about model validation, how Dymoval
relates to it, and how to address issues when the results are disappointing,
feel free to check the some_theory section.

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
`dymoval_tutorial` folder containing all the files needed to run the tutorial
in your `home` folder. All you have to do is to run Jupyter notebook named
`dymoval_tutorial.ipynb`. You need an app for opening `.ipynb` files.

The content of the `dymoval_tutorial` folder will be overwritten every time
this function is called.

CI/CD integration
=================

*Dymoval* can be used for unit-testing your models and therefore can be used
in development pipelines like those provided for example by e.g. Jenkins or
GitLab.

###########
 Unit-test
###########

The development of large models is typically done by breaking it down in
smaller components.

For example, if you are developing the model of a car, you may need to develop
the model of the engine, the model of the tires, the model of the gearbox and
then you integrate them.

However, smaller components are models themselves and therefore they can be
validated against some dataset through *Dymoval*. This means that you can use
*Dymoval* for unit-testing single components.

###################
 CI/CD integration
###################

A traditional software development workflow consists in pushing your software
changes towards a repo where there is some source automation server (like
Jenkins or GitLab) that automatically assess if your changes can be integrated
in the codebase or not.

Very often, the process of developing models goes along the same line: you
typically have a *Version Control System (VCS)* to track your model changes...

... but there are no automated mechanisms that test your model.

Checking that *"everything still works"* is typically done manually and if
your changes can be integrated or not is at reviewer discretion. Not robust,
nor democratic.

The ideal scenario would be to automatically test your model changes every
time they are pushed towards the remote repo, as it happens in traditional
software development. If on the one hand you are developing *models* - and
not, loosely speaking, *code* - on the other hand, testing a model just means
to *validate* it.

Here is where *Dymoval* comes into play: you can exploit its API to write
scripts that can be automatically executed by automation tools and you can
automatically get an answer if your changes can be integrated or not depending
if the validation metrics evaluation meet some criteria.

..
   vim: set ts=2 tw=78:
