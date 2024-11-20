Simulate your model
===================

So far, you have collected measurement data and stored them in a
:ref:`Dataset <Dataset>` object.

Next, you need to collect simulated data. How? You should feed your model with
the input signals stored in the :ref:`Dataset <Dataset>` object that you
prepared.

To extract the input signals from your :ref:`Dataset <Dataset>` object, you
can use the
method  :py:meth:`~dymoval.dataset.Dataset.dataset_values` and then use these
signals to feed your model.

Alternatively, you can export your :ref:`Dataset <Dataset>` object in the
format you need and import it into your modeling tool. To facilitate this
task, Dymoval
allows you to dump :ref:`Dataset <Dataset>` objects into Dymoval.Signal
objects through
the method :py:meth:`~dymoval.dataset.Dataset.dump_to_signals`. However, you
will then need to manually export these signals in an appropriate format
depending on your modeling tool.

Once you have simulated your model, you should import the simulated data back
into Python. At this point, you are ready to validate your model, and guess
what? Dymoval is here to help with that!

Go to the next section to discover more.

.. note::
  Exporting/importing signals from/to Python to/from your modeling tool
  may be fairly annoying. For this reason, we recommend compiling your model
  into an FMU and using packages like pyfmu or fmpy to simulate your model
  directly from a Python environment, so you have everything in one place.

Regardless of your modeling tool (Simulink, Dymola, GT-Power, etc.), you most
likely have an option for compiling models into FMUs. Check the documentation
of your modeling tool for more details.

.. vim: set ts=2 tw=78:
