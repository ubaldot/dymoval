Create, analyze and manipulate a dataset
========================================

Measurement data formats depend on many aspects such as the specific
application domain, the logging system, the manufacturer, and so on. Due to
this variability, it is impossible to establish a unified measurements
format that fits every domain. Each domain has its own
requirements. Therefore, we need to find a solution.

Dymoval defines a :ref:`Signal<signal>` object type that is general enough to
capture all the aspects of a signal, regardless of its application domain.

As a first step to using Dymoval, each logged signal must be cast into a
Dymoval :ref:`Signal<signal>`. Once this is done, a list of such
:ref:`Signals<signal>` can be used to create a :ref:`Dataset<Dataset>`
object. This represents the measurement dataset against which the simulated
outputs will be evaluated.

However, when dealing with measurement datasets, several problems arise:

- Signals may be sampled at different rates.
- Data loggers may run continuously for hours, logging data even when nothing
  interesting is happening, resulting in large log files with little
  information.
- Logs are often affected by other issues such as noisy measurements,
  missing data, and so on.

Dymoval provides a number of functions for dealing with
:ref:`Dataset<Dataset>` objects, including re-sampling, plotting, frequency
analysis, filtering, and more.

Once you have created and adjusted a measurement :ref:`Dataset<Dataset>`
object, you are ready to simulate your model.

.. vim: set ts=2 tw=78:
