*********************
Measurements handling
*********************

.. currentmodule:: dymoval.dataset

Measurement datasets are a central part in model validation and therefore we
designed the :ref:`Dataset` that offer a number of useful :ref:`methods
<datasetMethods>` to deal with them.

A typical workflow consists in casting your measurement data into dymoval
:ref:`Signal <signal>` objects and then use the created :ref:`Signals
<signal>` to instantiate a :ref:`Dataset <Dataset>` object.

.. _signal:

Signals
=======
Dymoval :ref:`Signal <signal>` are used to represent real-world signals.

.. currentmodule:: dymoval.dataset

Dymoval :ref:`Signals <signal>` are *Typeddict* with the following keys

.. rubric:: Keys
.. autosummary::

   Signal.name
   Signal.samples
   Signal.signal_unit
   Signal.sampling_period
   Signal.time_unit


.. rubric:: Functions

Dymoval offers few function for dealing with :ref:`Signals <signal>`:

.. autosummary::

   validate_signals
   plot_signals


.. _Dataset:

Dataset class
=============
The :ref:`Dataset` is used to store and manipulate measurement datasets.

Since to validate a model you need some measurement datasets, objects of this
class are used also to instantiate :ref:`ValidationSession
<ValidationSession>` objects, and the passed :ref:`Dataset <Dataset>` object
becomes an attribute of the newly created :ref:`ValidationSession
<ValidationSession>` object.

A :ref:`Dataset <Dataset>` object can be instantiated in two ways:

#. Through a list of dymoval :ref:`Signals <signal>` (see
   :py:meth:`~dymoval.dataset.validate_signals` )
#. Through a *pandas* DataFrame with a specific structure (see
   :py:meth:`~dymoval.dataset.validate_dataframe`)

.. currentmodule:: dymoval.dataset

.. rubric:: Constructor
.. autosummary::

   Dataset

.. rubric:: Attributes
.. autosummary::

   Dataset.name
   Dataset.dataset
   Dataset.coverage
   Dataset.sampling_period
   Dataset.excluded_signals

.. _datasetMethods:
.. rubric:: Manipulation methods
.. autosummary::

   Dataset.trim
   Dataset.fft
   Dataset.remove_means
   Dataset.detrend
   Dataset.remove_offset
   Dataset.low_pass_filter
   Dataset.apply
   Dataset.remove_NaNs
   Dataset.add_input
   Dataset.add_output
   Dataset.remove_signals

.. rubric:: Plotting methods
.. autosummary::

   Dataset.plot
   Dataset.plotxy
   Dataset.plot_coverage
   Dataset.plot_spectrum
   plot_signals
   change_axes_layout

.. rubric:: Other methods
.. autosummary::

   Dataset.dump_to_signals
   Dataset.dataset_values
   Dataset.export_to_mat
   Dataset.signal_list
   validate_dataframe
   validate_signals
   compare_datasets
..
   vim: set ts=3 tw=78:
