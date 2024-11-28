..
   dymoval documentation master file, created by
   sphinx-quickstart on Wed Aug 31 12:11:21 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

####################################
 Dymoval (Dynamic Model Validation)
####################################

*************
 What is it?
*************

*Dymoval* is a Python package for *analyzing measurement data* and *validating
models*.

*Dymoval* validates models based only on *measurements* and *simulated data*,
and it is completely independent of the used modeling tool. This means that it
does not matter if a model has been developed with Simulink, Modelica, etc.,
*dymoval* will only look at the produced data from the model.

If you are tracking your models changes in a CI/CD environment, then *dymoval*
API can be easily used to run tests in Jenkins or GitHub Actions pipelines as
it enables unit-testing on models.

Finally, *dymoval* provides a number of functions for for handling
measurements data, addressing common issues such as noise, missing data, and
varying sampling intervals.

**************
 What is not.
**************

*Dymoval* **is not** a tool for developing models. You have to develop your
models with the tool you prefer.

It is not a tool for *System Identification* either (but we don't exclude it
can happen in the future ;-)).

*Dymoval* only checks if your models are good or not but you have to develop
your models by yourself in the environment that you prefer.

**************
 Why dymoval?
**************

Simulation results frequently deviate significantly from real-world
measurements, leading to a growing skepticism towards simulation models.
*Dymoval* is dedicated to bridging this gap, aiming to restore confidence in
simulation accuracy and reliability.

*Dymoval* specializes in model validation, offering robust solutions for a
variety of models, including MIMO (Multiple Input Multiple Output) and stiff
models, all in an easy and comprehensible manner. Additionally, *Dymoval*
provides a comprehensive toolbox designed to handle real-world measurement
data, which often comes with challenges such as noise, missing data, and
varying sampling intervals. This ensures that your models are not only
validated but also capable of accurately reflecting real-world conditions.

***************
 Main Features
***************

Measurements analysis and manipulation
======================================

-  Time and frequency analysis
-  Easy plotting
-  Missing data handling
-  Linear filtering
-  Means and offsets removal
-  Re-sampling
-  Physical units

Model validation
================

-  Validation metrics:
-  R-square fit
-  Residuals auto-correlation statistics
-  Input-Residuals cross-correlation statistics
-  Coverage region
-  MIMO models
-  Independence of the modeling tool used.
-  API suitable for model unit-tests

*******
 Index
*******

.. toctree::
   :maxdepth: 2

   installation
   getting_started
   model_validation
   reference
   api

####################
 Indices and tables
####################

-  :ref:`genindex`
-  :ref:`modindex`
-  :ref:`search`

..
   vim: set ts=2 tw=78:
