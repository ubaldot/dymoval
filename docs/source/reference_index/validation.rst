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
   rmse
   compute_statistic

.. _XCorrelation:

XCorrelation class
==================

.. currentmodule:: dymoval.xcorrelation

All correlation lags use one-sample spacing, with no automatic
bandwidth-based adjustment. The default window is -20 through +20 sample
offsets, limited by record length. Multiply offsets by the sampling period
to obtain physical delays. ``nlags`` controls the window size, not spacing.

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

   XCorrelation.correlation_score
   XCorrelation.plot

.. rubric:: Functions
.. autosummary::

   correlation_score
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

Statistics and acceptance limits
--------------------------------

``validation_statistics`` contains ``r2``, ``rmse``, ``Ruu_correlation``,
``Ree_correlation``, and ``Rue_correlation`` for each simulation, plus
``Ruu_cross_correlation`` and ``Ree_cross_correlation`` for multiple inputs
and outputs. The ordinary ``Ruu_correlation`` and ``Ree_correlation`` scores
cover diagonal pairs only; their corresponding ``*_cross_correlation``
scores cover off-diagonal pairs. RMSE is
computed per output, then aggregated with ``rmse_statistic="max"`` or
``"mean"``; outputs need comparable units/scaling. No RMSE threshold is
enabled by default.

Gaussian limits are enabled by default for both inputs and residuals using
``Ruu_envelope="auto"`` and ``Ree_envelope="auto"``. Each uses
``(0.95, 0.1, 5*Ts)`` as starting settings, not calibrated limits. Pass a
tuple or a mapping from every input/output name to a tuple to customize
limits independently. The third tuple value is a physical time in dataset
units. Each ``*_envelope`` score is the maximum nonzero-lag violation ratio,
with an inclusive default upper threshold of 1; it replaces the default
ordinary diagonal check. The ordinary ``*_correlation`` scores remain
available in statistics. Use ``Ruu_envelope=None`` or
``Ree_envelope=None`` independently to restore ordinary diagonal scoring.
Single-input and single-output cross-correlation are displayed as N/A and
omitted from statistics and thresholds.

``plot_residuals()`` shades Gaussian limits on input and residual diagonal
plots. Other correlation panels shade plus/minus their configured ordinary
score threshold, labelled as a reference rather than a pointwise limit.
Peaks outside a reference band do not necessarily fail an ``abs_mean`` check.
Omitted thresholds have no reference band. With either envelope set to
``None``, its diagonal panels use the ordinary threshold reference too.
Time scales use dataset time units; plotted lag offsets still
use samples. Custom threshold dictionaries enable only their listed checks.
R-squared uses a strict lower bound, ordinary correlation scores and RMSE
strict upper bounds, and the envelope ratio an inclusive upper bound.

See :ref:`validation_session` for the formulas, parameter constraints,
examples, and guidance on choosing meaningful tolerances.

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
