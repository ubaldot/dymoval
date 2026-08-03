*********************
Measurements handling
*********************

Measurement datasets are a central part of model validation. *Dymoval*
represents them with two objects:

- a :ref:`Signal <signal>`, which is one measured time-series, and
- a :ref:`Dataset <Dataset>`, which is a set of *aligned* signals split into
  inputs and outputs.

The typical workflow is to cast each logged channel into a :ref:`Signal
<signal>`, and then to group the resulting signals into a :ref:`Dataset
<Dataset>`.

.. _signal:

Signal class
============

.. currentmodule:: dymoval.signal

A :ref:`Signal <signal>` is an immutable dataclass holding one uniformly
sampled time-series.

.. rubric:: Constructor
.. autosummary::

   Signal

.. rubric:: Fields
.. autosummary::

   Signal.name
   Signal.values
   Signal.time
   Signal.unit
   Signal.time_unit

The ``time`` vector may be ``None``, in which case the signal is
sample-indexed. Every method below returns a **new** :ref:`Signal <signal>`:
nothing is ever modified in-place.

.. rubric:: Manipulation methods
.. autosummary::

   Signal.trim
   Signal.resample
   Signal.detrend
   Signal.remove_mean
   Signal.remove_constant
   Signal.low_pass_filter
   Signal.apply
   Signal.copy

.. rubric:: Analysis methods
.. autosummary::

   Signal.get_sampling_period
   Signal.fft
   Signal.spectrum

.. rubric:: Plotting methods
.. autosummary::

   Signal.plot
   Signal.plot_spectrum

.. _Dataset:

Dataset class
=============

.. currentmodule:: dymoval.dataset

A :ref:`Dataset <Dataset>` holds two mappings of :ref:`Signals <signal>` —
``inputs`` and ``outputs`` — that all share one and the same time vector.

Since validating a model requires measurement data, a :ref:`Dataset <Dataset>`
is also what you pass to a :ref:`ValidationSession <ValidationSession>`, where
it becomes the ``dataset`` attribute of the newly created object.

Building a Dataset
------------------

Prefer the two factories over the raw constructor: they accept signals that do
**not** yet share a time vector and resample them onto a common uniform grid.

.. autosummary::

   Dataset.from_signals
   Dataset.from_dict
   Dataset

By default the common grid uses the **largest** sampling period found among
the passed signals and spans only the time interval covered by all of them, so
that no extrapolation ever takes place. Pass ``target_sampling_period`` to
pick the grid yourself.

.. warning::

   Building a :ref:`Dataset <Dataset>` interpolates. Trim away leading and
   trailing ``NaN`` samples with :py:meth:`Signal.trim
   <dymoval.signal.Signal.trim>` **before** calling a factory, otherwise the
   ``NaN``\ s spread to the interpolated samples.

.. rubric:: Fields
.. autosummary::

   Dataset.inputs
   Dataset.outputs
   Dataset.meta
   Dataset.name

.. _datasetMethods:
.. rubric:: Access methods
.. autosummary::

   Dataset.names
   Dataset.input_names
   Dataset.output_names
   Dataset.all_signals
   Dataset.kind_of
   Dataset.signal_list
   Dataset.time
   Dataset.time_unit
   Dataset.get_sampling_period

.. rubric:: Manipulation methods
.. autosummary::

   Dataset.trim
   Dataset.resample
   Dataset.align
   Dataset.detrend
   Dataset.remove_mean
   Dataset.remove_constant
   Dataset.low_pass_filter
   Dataset.apply
   Dataset.pipe
   Dataset.add_input
   Dataset.add_output
   Dataset.remove_signals
   Dataset.copy

As for :ref:`Signals <signal>`, each of these returns a new :ref:`Dataset
<Dataset>`.

.. rubric:: Analysis methods
.. autosummary::

   Dataset.fft
   Dataset.spectrum
   Dataset.coverage

.. rubric:: Plotting methods
.. autosummary::

   Dataset.plot
   Dataset.plot_xy
   Dataset.plot_coverage
   Dataset.plot_spectrum

.. rubric:: Export methods
.. autosummary::

   Dataset.dataset_values
   Dataset.to_signals
   Dataset.export_to_mat

Plotting several objects at once
================================

.. currentmodule:: dymoval.plotting

The methods listed above plot one object. To plot loose signals, or to compare
several datasets against each other, use the module-level functions:

.. autosummary::

   plot_signals
   plot_dataset
   plot_compare
   plot_spectrum_compare
   plot_coverage_compare

:py:func:`plot_signals <dymoval.plotting.plot_signals>` is the only one that
does not require its signals to be aligned, which makes it the tool for
eyeballing raw logs *before* a :ref:`Dataset <Dataset>` exists.

.. _figure_geometry:

Controlling the figures
=======================

Every figure-returning plot function or method accepts the same three
geometry arguments:

``layout``
   The *matplotlib* layout engine: ``"constrained"`` (the default),
   ``"compressed"``, ``"tight"`` or ``"none"``.

``ax_height``
   Height, in inches, of each subplot.

``ax_width``
   Width, in inches, of the whole figure.

Any further keyword argument is forwarded verbatim to *matplotlib*, so the
usual ``linestyle``, ``alpha``, ``linewidth``, ... all work:

.. code::

   fig = ds.plot(("u1", "y1"), ax_height=3.0, layout="tight", linestyle="--")

On top of that, :py:meth:`Dataset.plot <dymoval.dataset.Dataset.plot>`,
:py:meth:`Dataset.plot_spectrum <dymoval.dataset.Dataset.plot_spectrum>` and
:py:meth:`Dataset.plot_coverage <dymoval.dataset.Dataset.plot_coverage>`
accept ``color_input`` and ``color_output``, the colors used when a subplot
holds a *single* signal. Subplots holding a group of signals always fall back
to the *matplotlib* color cycle, otherwise the overlaid curves would be
indistinguishable.

Since every function returns the figure and never calls ``show()``, anything
not covered by these arguments can still be done afterwards with the plain
*matplotlib* API::

   fig = ds.plot()
   fig.set_size_inches(10, 5)
   fig.savefig("measurements.svg")

.. _scopes:

Interactive scopes
==================

.. currentmodule:: dymoval.scope

Every plotting function accepts a ``with_scope`` argument, which defaults to
``True``. The figure is then split in two: the plots on the left and an info
panel on the right. Click on a curve to select it and read its name, unit and
statistics; click on a second point to get the delta between the two; press
``r`` to reset.

.. autosummary::

   BaseScope
   SignalScope
   DatasetScope
   SpectrumScope
   AmplitudeSpectrumScope

.. note::

   Scopes need an interactive *matplotlib* backend such as ``qtagg`` or
   ``widget``. With the ``inline`` backend, which is the default in many
   notebook setups, there is nothing to click on, so pass ``with_scope=False``
   to get a plain figure.

..
   vim: set ts=3 tw=78:
