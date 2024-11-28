.. _simulate_model:

#####################
 Simulate your model
#####################

So far, you have collected measurement data and stored them in a :ref:`Dataset
<Dataset>` object.

Next, you need to collect simulated data. How? You should feed your model with
the input signals stored in the :ref:`Dataset <Dataset>` object that you
prepared.

To extract the input signals from your :ref:`Dataset <Dataset>` object, you
can use the method :py:meth:`~dymoval.dataset.Dataset.dataset_values` and then
use these signals to feed your model:

.. code::

   # Assuming `ds` is a measurements Dataset object
   (t, u_meas, y_meas) = ds.dataset_values()
   # Assume you are using a simulation tool that has a Python API with
   # a function called 'simulate_model()`
   y_sim = simulate_model(time = t, input= u_meas)

Alternatively, you can export your :ref:`Dataset <Dataset>` object in the
format you need and import it into your modeling tool. To facilitate this
task, Dymoval allows you to dump :ref:`Dataset <Dataset>` objects into
Dymoval.Signal objects through the method
:py:meth:`~dymoval.dataset.Dataset.dump_to_signals`.

.. code::

   # List containing Signal objects
   signal_list = ds.dump_to_signals()

However, you will then need to manually export these signals in an appropriate
format depending on your modeling tool. Given the popularity of Matlab,
:py:class:`~dymoval.dataset.Dataset` class has a
:py:meth:`~dymoval.dataset.Dataset.export_to_mat` method:

.. code::

   signal_list = ds.export_to_mat("./my_measurements.mat")

Once you have simulated your model, you should import the simulated data back
into Python. At this point, you are ready to validate your model.


.. note::

   Exporting/importing signals from/to Python to/from your modeling tool may
   be fairly annoying. For this reason, we recommend compiling your model into
   an FMU and using packages like pyfmu or fmpy to simulate your model
   directly from a Python environment, so you have everything in one place.

   Regardless of your modeling tool (Simulink, Dymola, GT-Power, etc.), you most
   likely have an option for compiling models into FMUs. Check the documentation
   of your modeling tool for more details.
