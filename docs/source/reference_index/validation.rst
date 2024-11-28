****************
Model Validation
****************

A :ref:`ValidationSession<ValidationSession>` object stores a :ref:`Dataset <Dataset>` object that serves as a basis for validate your models.
Then, the user can append as many simulation results as he/she want to the same
:ref:`ValidationSession<ValidationSession>` object that automatically computes validation metrics
and store the results in the
:py:attr:`~.ValidationSession.validation_results` attribute.

.. warning::
   It is **discouraged** to change a :ref:`Dataset <Dataset>` object once it is an attribute of a :ref:`ValidationSession<ValidationSession>` object.
   This because the validation results will be jeopardized.

   If you want to change Dataset, then consider to create a new :ref:`ValidationSession<ValidationSession>` object.

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
