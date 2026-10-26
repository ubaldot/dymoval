****************
Model Validation
****************

Given the centrality of auto- and cross-correlation functions in model
validation, *dymoval* provides a class :ref:`XCorrelation
<XCorrelation>`. Furthermore, it also provides a :ref:`ValidationSession<ValidationSession>` to cope with simulation results.

.. _XCorrelation:

XCorrelation class
==================

.. currentmodule:: dymoval.validation

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

.. _ValidationSession:

ValidationSession class
=======================

.. currentmodule:: dymoval.validation

.. rubric:: Constructor

.. autosummary::

   ValidationSession

.. rubric:: Attributes
.. autosummary::

   ValidationSession.name
   ValidationSession.dataset
   ValidationSession.outcome
   ValidationSession.simulations_names
   ValidationSession.validation_thresholds
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

   rsquared
   compute_statistic
   whiteness_level
   validate_models
