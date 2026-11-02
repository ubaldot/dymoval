####################################
 Dymoval (Dynamic Model Validation)
####################################

Dymoval is a Python package for exploring measurement data and checking how
well your models match the real world.

*************
 What is it?
*************

Dymoval helps you check your models without having to build your own
analysis tools.
The idea is simple: feed Dymoval with measurements and model ouptput data
and get a detailed report about your model quality in terms of a set of
comprehensive KPIs.
Your model can come from Simulink, Modelica, Python, or another tool. Dymoval
only needs the signals, not the model itself.

There is another challenge before you can even compare the results:
real-world logs can be messy. One sensor records faster than another,
some samples are missing, and noise makes it hard to see what is going on.
Dymoval helps you get these signals ready to work with, without having to
write the same cleanup code for every experiment.

Once the data is ready, you can plot measurements and simulations together,
check how closely they match, and look for patterns in the errors. A good
fit score is useful, but those patterns can reveal behavior your model
is still missing.

You can also estimate a frequency response directly from measurements to
explore how your system responds at different frequencies, without first
building a parametric model.

You can work in a notebook, inspect plots interactively, or turn your checks
into automated tests. That makes it easier to catch changes in model behavior
as you develop it, including in Jenkins or GitHub Actions.

****************
 What it is not
****************

Dymoval is not a replacement for your modeling tool. It does not build or
tune a parametric model for you, or run that model's simulations. Keep using
the tools you like for those tasks, then bring the results to dymoval.

Of course, matching one experiment does not mean a model will work
everywhere. Dymoval helps you check the data you have; you decide which
experiments and checks matter for your application.

***************
 Main Features
***************

Working with measurements
=========================

- Explore signals in time and frequency
- Estimate frequency responses with ``Dataset.spa()``
- Filter noise, align sampling times, fill gaps, and remove offsets or trends
- See which values your measurements cover with histograms and statistics
- Keep track of signal names and physical units
- Click on plotted curves to inspect them with interactive scopes

Checking your models
====================

- Compare several simulations against the same measurements
- Measure the fit with R-squared scores
- Look for patterns in errors using residual correlation checks
- Set your own thresholds for pass/fail checks
- Work with one or several inputs and outputs (SISO and MIMO)
- Run the same checks automatically when your model changes

Ready to try it?
If you don't know what Model Validation is take a look at
:doc:`model_validation`, otherwise jump straight to :doc:`getting_started` to
clean some data, check a simple model, and explore a frequency response.

*******
 Index
*******

.. toctree::
   :maxdepth: 2

   installation
   model_validation
   getting_started
   reference
   migrating
   api

####################
 Indices and tables
####################

-  :ref:`genindex`
-  :ref:`modindex`
-  :ref:`search`

..
   vim: set ts=2 tw=78:
