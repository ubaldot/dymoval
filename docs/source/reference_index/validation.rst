****************
Model Validation
****************

Auto- and cross-correlation functions are central to model validation, so
*dymoval* provides an :ref:`XCorrelation <XCorrelation>` class along with a
few statistics functions. On top of them, the :ref:`ValidationSession
<ValidationSession>` class collects the simulation results of your models and
tells you whether they pass validation.

.. _statistics:

Statistics functions
====================

.. currentmodule:: dymoval.statistics

.. autosummary::

   rsquared
   compute_statistic

.. _XCorrelation:

XCorrelation class
==================

.. currentmodule:: dymoval.xcorrelation

An :ref:`XCorrelation <XCorrelation>` object stores the normalized
cross-correlation of two (possibly multivariate) signals ``X`` and ``Y``. When
``X is Y`` it is an *auto*-correlation, and :py:attr:`kind
<dymoval.xcorrelation.XCorrelation.kind>` reports which of the two it is.

.. rubric:: Constructor

.. autosummary::

   XCorrelation

.. rubric:: Attributes
.. autosummary::

   XCorrelation.name
   XCorrelation.kind
   XCorrelation.R

.. rubric:: Methods
.. autosummary::

   XCorrelation.estimate_whiteness
   XCorrelation.plot

.. rubric:: Functions
.. autosummary::

   whiteness_level

.. _ValidationSession:

ValidationSession class
=======================

.. currentmodule:: dymoval.validation

A :ref:`ValidationSession <ValidationSession>` pairs one measurement
:ref:`Dataset <Dataset>` with any number of simulation results. Each appended
simulation is scored against the same set of thresholds, and
:py:attr:`outcome <dymoval.validation.ValidationSession.outcome>` reports
whether each of them passed.

.. rubric:: Constructor

.. autosummary::

   ValidationSession

.. rubric:: Attributes
.. autosummary::

   ValidationSession.name
   ValidationSession.dataset
   ValidationSession.outcome
   ValidationSession.validation_statistics
   ValidationSession.validation_thresholds
   ValidationSession.simulations
   ValidationSession.simulations_names
   ValidationSession.simulations_values
   ValidationSession.Ruu
   ValidationSession.Ree
   ValidationSession.Rue

.. rubric:: Methods
.. autosummary::

   ValidationSession.append_simulation
   ValidationSession.drop_simulations
   ValidationSession.plot_simulations
   ValidationSession.plot_residuals
   ValidationSession.simulation_signals_list
   ValidationSession.clear
   ValidationSession.trim

.. rubric:: Functions
.. autosummary::

   validate_models

..
   vim: set ts=3 tw=78:
