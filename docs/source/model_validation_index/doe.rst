.. _doe:

#############################
 Design of Experiments (DoE)
#############################

When running experiments, it is important to stimulate the target environment
in a way that we extract as much information as possible from it.

Good experiments shall stress the target environment as much as possible under
different conditions to get a sufficiently *informative* measurement dataset.

   **Example**

   If we are developing the model of a car, we want to log sensors
   measurements while driving within a wide range of speeds, with different
   accelerations profiles, in different road and weather conditions and so on
   and so forth.

   If we log sensors measurements only when we are driving on a flat road and
   in the range 0-10 km/h and by doing exactly the same maneuvers over and
   over, then it would be hard to disagree on that the collected measurements
   dataset is poorly informative.

Ideally, the excitation should be **persistently exciting**: it should contain
enough independent variation over the frequency range that matters for the
system and the model. White noise is a common example of a persistently
exciting signal in the stochastic sense. In practice, a white-noise signal can
be approximated with a **Pseudo-Random-Binary-Sequence (PRBS)**, which is easy
to implement on digital platforms. A chirp or another carefully designed
multisine can also be appropriate when the frequency content needs to be
controlled explicitly. Regardless of the particular signal, the goal is to
exercise the relevant dynamics over a sufficiently broad range of operating
conditions.

Next, let's say that we have completed your experiments. To assess the quality
of the experiments, we can simply check how similar the input signal was to
white noise. This is done by analyzing the **auto-correlation** function of
our input signal. Dymoval has the :py:class:`~dymoval.xcorrelation.XCorrelation`
class that can be used for this purpose.

Assume that you have a signal ``u`` expressed as a :math:`N \times p` array,
where :math:`N` is the number of samples and :math:`p` is the dimension of the
signal.

You can create and analyze the auto-correlation function of the signal ``u``
with the following:

.. code::

   from dymoval import XCorrelation

   Ruu = XCorrelation("Ruu", u, u)
   # Plot the auto-correlation function
   Ruu.plot()
   # Estimate whiteness
   Ruu.estimate_whiteness()

The whiteness level of the signal ``u`` is computed according to different
metrics, see :py:meth:`~dymoval.statistics.compute_statistic`.

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

If a signal is over-sampled, consecutive measurements are often correlated
simply because the sampling interval is much shorter than the time scale of
the signal. This can make the auto-correlation appear large near zero lag,
even when the experiment contains useful excitation. The result may then
describe the sampling rate more than the signal's relevant dynamics.

There are two common ways to address this:

* compute the auto-correlation over a sufficiently long lag window and
  interpret the result outside the system's memory, that is, after the
  dominant correlation has decayed over the relevant time constants; or
* down-sample the signal so that consecutive samples are separated by a
  time scale appropriate for its bandwidth, while still satisfying the
  Shannon-Nyquist criterion.

*Dymoval* uses the second approach when bandwidth and sampling-period
information are supplied to :py:class:`~dymoval.xcorrelation.XCorrelation`.
It selects a lag time of approximately :math:`1/(2B)`, where :math:`B` is the
signal bandwidth, and evaluates the correlation on that reduced lag grid.
This avoids treating densely packed, naturally correlated samples as
independent evidence of poor excitation.

.. note::

   In future releases we plan to further provide measures (Cramer-Rao Theorem?
   Fisher Matrix?) on the information level contained in a given measurements
   dataset in within its coverage region.

*****************
 Coverage region
*****************

Other than having an informative measurement dataset in terms of signal
variation in terms of similarity to white-noise, we also want to cover as much
amplitude as possible during our experiments. Such an amplitude is referred as
coverage region.

It is worth mentioning that *size* of the coverage region and *whiteness*
within go hand in hand, as shown in the next example.

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
