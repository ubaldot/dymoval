.. _doe:

#############################
 Design of Experiments (DoE)
#############################

When running experiments, it is important to stimulate the target environment
in a way that we can extract as much information as possible from it.

Good experiments shall stress the target environment as much as possible under
different conditions. This is important because a model is as trustworthy as
the measurements dataset used for validating it is *informative*.

   **Example**

   If we are developing the model of a car, we want to log sensors
   measurements while driving within a wide range of speeds, with different
   accelerations profiles, in different road and weather conditions and so on
   and so forth.

   If we log sensors measurements only when we are driving on a flat road and
   in the range 0-10 km/h and by doing exactly the same maneuvers over and
   over, then it would be hard to disagree on that the collected measurements
   dataset is poorly informative.

How can you design experiments that produce sufficiently informative
measurement datasets?

Ideally, the best would be to stimulate the system at every frequencies. Very
ideally with a white noise signal. A white noise signal could be mimic through
a *Psuedo-Random-Binary-Sequence (PRBS)* which is extremely easy to implement
on a digital platform. Alternatively, you could use a chirp signal or you
could just manually drive your real system as randomly as possible. Be sure to
hit all the corner cases.

Say that we have done with our experiments.
How to check if our signal was *white* enough? Well, we can analyze its
*auto-correlation* function. Dymoval has the
:py:class:`~dymoval.validation.XCorrelation` that can be used for this
purpose.

Assume that you have a signal ``u`` expressed as a :math:`N \times p`
``np.ndarray``, where :math:`N` is the number of samples and :math:`p` is the
dimension of the signal.

You can create and analyze the auto-correlation function of the signal ``u``
with the following:

.. code::

   from dymoval.correlation import XCorrelation

   Ruu = XCorrelation("Ruu", u, u)
   # Plot the auto-correlation function
   Ruu.plot()
   # Estimate whiteness
   Ruu.whiteness()

The whiteness level of the signal ``u`` is computed according to different
metrics, see :py:meth:`~dymoval.validation.compute_statistics`.

.. note::

   Sometimes it is not possible to stimulate our real-world system as randomly
   as possible. For example, if you are designing a space-shuttle I don't
   think you will get any approval for driving it randomly in the sky only for
   the purpose of collecting measurements. In this case, you can simply ignore
   this evaluation metric, and if you are validating your model you can pass
   the argument ``ignore_input=True`` to
   :py:meth:`~dymoval.validation.validate_models` or to
   :py:class:`~dymoval.validation.ValidationSession` class constructor.

**********************
 Over-sampled signals
**********************

If the signal under examination is overs-ampled, the whiteness results may be
inaccurate. This occurs because consecutive measurements are naturally
correlated when a sensor samples too quickly, leading to higher values in the
auto-correlation function.

However, by knowing the signal’s bandwidth and its sampling period, we can
determine whether a signal is over-sampled. If it is, we can adjust the lag
time of the auto-correlation function to be :math:`lag\_time = 1/(2B)`, where
:math:`B` is the signal bandwidth. This adjustment ensures that we are
analyzing the auto-correlation of the actual signal rather than consecutive
measurements. Dymoval handles this automatically, provided that the bandwidth
and sampling period are supplied to the constructor of the
:py:class:`~dymoval.validation.XCorrelation` class.

.. note::

   In future releases we plan to further provide measures (Cramer-Rao Theorem?
   Fisher Matrix?) on the information level contained in a given measurements
   dataset in within its coverage region.

*****************
 Coverage region
*****************

It is worth noting that a measurements dataset covering a fairly large region
won't necessarily imply *information richness.* This happens for example when
you take a wide range of values but you stimulate your target environment only
with constant inputs in within such a range. You would certainly have a
measurements dataset with a fairly large covered region but it would contain
little information.

   **Example**

   With reference to the example above, imagine driving the car only at
   constant speeds ranging from 0 to 180 km/h on a flat road, without ever
   accelerating or braking. For instance, you make a first run driving (and
   logging) data at a constant speed of 10 km/h, without accelerating or
   braking, and staying on a flat road. Then, you perform a second run at a
   constant speed of 20 km/h under the same conditions as the previous run,
   and so on, until reaching the final run at 180 km/h.

   Your measurements dataset will have a fairly large coverage region, but it
   will contain little information since all the runs were conducted at
   constant speeds without any acceleration or braking, and on a flat road.

Hence, at the end of this phase we should have both the coverage region and
the information richness of the input signal.
