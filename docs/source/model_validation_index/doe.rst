.. _doe:

#############################
 Design of Experiments (DoE)
#############################

Good experiments shall stress the target environment under
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

How can you tell whether your experiment provided enough information?
There is no single score that answers this for every model. Three useful
questions are:

* Did the inputs excite the dynamics and input directions you care about?
* Would shortcomings in the model show up in this experiment?
* Did the experiment cover the operating conditions where you will use it?

If you already have measurement logs, two useful ways to assess their
input excitation are:

* *Input auto- and cross-correlation:* inspect patterns over time and
  redundancy between input channels. These plots help explain the input's
  behavior, but a small aggregate correlation score does not guarantee
  sufficient excitation.
* *Lagged-input Gram-matrix eigenvalues:* check whether the input excites
  every direction for a chosen input-window length :math:`L`.
  A strictly positive minimum eigenvalue is equivalent to full rank of that
  finite-data matrix; its magnitude depends on amplitude and channel scaling.

These are related views of the input data, not two independent certificates
of experiment quality. A colored input can pass the Gram-matrix rank check
even when its auto-correlation is substantial.

Which should you use?

* Start with correlation plots when you want to understand the inputs:
  are they slowly varying or repetitive, and are some channels redundant?
  They are a useful first check and are available directly in dymoval.
* Use Gram-matrix eigenvalues when you need an excitation check for a
  particular input-window length :math:`L`, or want to compare portions of a log with consistent
  settings. They examine combinations of channels and delays together,
  rather than summarizing individual correlations. Dymoval does not
  currently provide this calculation.
* Use both when you want a numerical assessment and an explanation of
  the patterns behind it, especially with several input channels. If the
  weakest direction is poorly excited, correlation plots can help you
  investigate why.

Neither is a replacement for the other: a correlation plot is easier to
interpret, while the eigenvalues directly answer a rank/conditioning question
for the chosen time-window length *L*.
Substantial correlation does not automatically mean poor excitation, and a
small average correlation score does not guarantee a well-conditioned Gram
matrix.

If you can plan a new experiment, you can instead start with deliberately
designed excitation signals, such as PRBS, chirps, or multisines. These help
you control which dynamics you exercise, but the recorded signals still
need checking. The section "Planning a new experiment: choosing the inputs"
below discusses this design step.

.. note::

   Whichever method you use, inspect the measured inputs and outputs too.
   Check that the output response is clearly visible above measurement noise,
   the signals are not clipped, and actuators or sensors have not saturated
   unintentionally. Also check timing alignment, recording duration, relevant
   frequency content, and operating conditions. Output movement alone is not
   enough if it comes mainly from disturbances or sensor noise.

   Neither method replaces these checks. Saturation may be something you
   deliberately want to study, but it changes how you should interpret the
   results, particularly when assessing a linear model.

**************************************************
 Method 1: input auto- and cross-correlation
**************************************************

This method asks: *what time patterns and channel relationships are present
in the inputs?* Auto-correlation compares an input with delayed versions of
itself; cross-correlation compares different input channels at selected delays.

Start with the plots. Do the inputs vary on the time scales you want to study?
Input auto-correlation helps you spot slow variation, repeated patterns, or
narrow-band excitation. With multiple inputs, also check cross-correlation
between input channels: two inputs can each look like white noise but be
identical, so they do not excite independent directions.

Use :py:class:`~dymoval.xcorrelation.XCorrelation` to inspect both.

Assume that you have a signal ``u`` expressed as a :math:`N \times p` array,
where :math:`N` is the number of samples and :math:`p` is the dimension of the
signal.

You can create and analyze the auto-correlation function of the signal ``u``
with the following:

.. code::

   from dymoval import XCorrelation

   Ruu = XCorrelation("Ruu", u, u)
   # Diagonal plots: each input's auto-correlation.
   # Off-diagonal plots: cross-correlation between input channels.
   Ruu.plot()
   # Summarize correlation away from the zero-lag diagonal peak.
   Ruu.estimate_whiteness()

Despite its name, ``estimate_whiteness()`` returns a correlation summary:
lower values mean less correlation over the evaluated lags. The aggregation
can be selected through :py:meth:`~dymoval.statistics.compute_statistic`.
This is neither a formal statistical whiteness test nor a test of persistent
excitation.

Next, look at the outputs alongside the applied inputs. A large output range
alone is not enough: movement may come from disturbances or sensor noise
rather than the input. Look for a plausible response to the input changes,
and record long enough to see the relevant transients.

These are practical screening checks, not a certificate that the data is
informative. Colored inputs can be useful, while small correlation scores
do not guarantee that the input excites the dynamics you need.

.. note::

   Safety and operating constraints may limit the inputs you can apply.
   Do not aim for randomness at the expense of a safe, representative
   experiment. If input correlation is not a suitable pass/fail criterion,
   pass ``ignore_input=True`` to
   :py:meth:`~dymoval.validation.validate_models` or to
   :py:class:`~dymoval.validation.ValidationSession` class constructor.
   This disables the input-correlation pass/fail check; it does not establish
   that the experiment has sufficient excitation.

Over-sampled signals
====================

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

****************************************************
 Method 2: Lagged-input Gram-matrix eigenvalues
****************************************************

This method asks: *does the recorded input excite every combination of
channels and delays in a chosen lagged-input vector, and how weak is the
weakest direction?* A positive minimum eigenvalue establishes full rank
in exact arithmetic; in practice, interpret small eigenvalues relative to
the signal scaling and numerical precision. Eigenvalues alone do not tell
you which channels or delays cause a weak direction.

For an input vector :math:`U(t)\in\mathbb{R}^p`, form a vector containing
:math:`L` consecutive input samples:

.. math::

   \psi_L(t) =
   \begin{bmatrix}
   U(t) \\
   U(t-1) \\
   \vdots \\
   U(t-L+1)
   \end{bmatrix}.

For a record with samples indexed from :math:`0` to :math:`N-1`, its
finite-data Gram matrix is

.. math::

   G_L = \sum_{t=L-1}^{N-1}\psi_L(t)\psi_L(t)^\top.

This definition uses raw input values, so constant offsets contribute to
the matrix. If you want to assess variations around an operating point,
subtract that operating-point input before constructing the stacks.
Dymoval's normalized correlation diagnostics subtract signal means, so they
describe fluctuations rather than constant levels. Use a consistent
centering convention when comparing the two methods, and keep it fixed
across segments when scanning a log.

A bound :math:`G_L \succeq \alpha I`, with :math:`\alpha>0`, means there is
variation in every direction of this lagged input vector over the record.
The required input-window length :math:`L` depends on the time scales and input
patterns relevant to the behavior you want to validate;
the condition is not a universal test of model quality. For finite records,
full rank of this matrix is commonly called persistent excitation of order
:math:`L`. For an ongoing signal, a stronger condition requires excitation
to remain bounded below over successive segments, rather than occurring
only once near the start of the recording.

Finding useful portions of a log
==================================

You will often start by looking at the plots and trimming away periods where
nothing happens. An excitation score can complement that visual inspection.

First, distinguish two lengths:

* :math:`L` is the number of consecutive input samples in each stack
  :math:`\psi_L`.
* :math:`W` is the number of measurement samples in the segment you want
  to assess.

The stacks overlap: with :math:`N=100` and :math:`L=20`, use samples
1--20, 2--21, 3--22, and so on up to 81--100. That gives
:math:`N-L+1=81` outer products, not five non-overlapping contributions.
Each stack is ordered newest sample first, as in the definition above.

For the whole record, average those matrices and calculate the smallest
eigenvalue:

.. math::

   \overline G_L = \frac{G_L}{N-L+1},
   \qquad
   \alpha = \lambda_{\min}(\overline G_L).

A near-zero value means at least one lagged-input direction has little
excitation. Larger values mean stronger excitation in the weakest direction,
provided the sampling period, :math:`L`, and channel scaling stay fixed.
This is not an overall dataset-quality score or a universal pass/fail test.
The average makes comparisons less dependent on record length; without it,
adding samples cannot decrease the minimum eigenvalue of the accumulated
Gram matrix.

To see how excitation changes within a log, choose a fixed segment length
:math:`W` and move that segment forward one sample at a time. Intuitively,
treat each segment as a small dataset of its own: apply the same calculation
as for the whole record, replacing its length :math:`N` with :math:`W`.
Each score then uses :math:`W-L+1` overlapping stacks. The original
:math:`N` only determines how many segment positions you can examine:
:math:`N-W+1` when moving one sample at a time.

For a segment
starting at zero-based index :math:`x`, use only stacks fully inside it:

.. math::

   \overline G_{L,W}(x) =
   \frac{1}{W-L+1}
   \sum_{t=x+L-1}^{x+W-1}
   \psi_L(t)\psi_L(t)^\top,
   \qquad
   \alpha(x)=\lambda_{\min}(\overline G_{L,W}(x)),

with :math:`x=0,\ldots,N-W`.

For example, with :math:`N=100`, :math:`L=20`, and :math:`W=50`:

* Assess samples 1--50 using stacks 1--20, 2--21, ..., 31--50:
  a total of 31 contributions gives one score.
* Move to samples 2--51 and calculate a new score.
* Continue up to samples 51--100.

Plot these scores against the center time of each segment. This can reveal
weakly and strongly excited regions at the *beginning, middle, or end*
of the log, not just an inactive tail. It does not discard data automatically.
Each score summarizes :math:`W` samples, so a segment crossing a boundary
between active and inactive periods mixes both.

Selecting one such segment is equivalent to trimming the dataset to it and
calling its length :math:`N=W`. You only need the separate symbol
:math:`W` while comparing candidate segments.

For scanning, choose :math:`L<W<N`; :math:`W=N` is also valid but gives
only one score. With :math:`p` inputs the matrix has dimension :math:`pL`,
so full rank requires at least

.. math::

   W-L+1 \geq pL,
   \qquad\text{or equivalently}\qquad
   W \geq (p+1)L-1.

For one input and :math:`L=20`, that means :math:`W\geq39`. This is only
a necessary sample-count condition, not a guarantee of full rank; use more
samples for a stable assessment.

There is no need to test every possible start/end pair. Start with one
meaningful segment duration, or a few durations. When moving a fixed-length
segment, update its sum by subtracting the outer product leaving it and
adding the one entering it, then recompute the smallest eigenvalue.

Keep channel scaling consistent across the log: normalizing each segment
separately could make barely moving segments appear strongly excited.
Higher amplitudes can also raise the score. Choose what to keep alongside
input/output plots, noise levels, relevant transients, and operating
conditions, rather than maximizing this number alone.

There is no universal eigenvalue threshold that works for every experiment.
Dymoval does not currently compute these Gram-matrix eigenvalues.

****************************************************
 Planning a new experiment: choosing the inputs
****************************************************

When you can choose the inputs, you do not have to rely on whatever variation
happens to be present in an existing log. You can design signals to exercise
the dynamics you want to validate:

* **Noise-like inputs: PRBS and random signals** provide broad-band excitation
  when their timing and duration are chosen appropriately. A PRBS is a
  deterministic, repeatable sequence switching between two levels. A
  maximum-length sequence has small periodic auto-correlation at nonzero lags
  within one period, at its switching grid. Random signals can instead use
  Gaussian, uniform, or binary amplitude distributions. Independent,
  zero-mean random samples are white in expectation, not exactly uncorrelated
  in every finite record. PRBS repetition produces correlation peaks at its
  period; holding samples or filtering either type introduces temporal
  correlation.
* **Chirps** sweep through a frequency range, letting you explore the response
  at different frequencies over time. They need not resemble white noise or
  produce a small auto-correlation score.
* **Multisines** combine selected frequencies. They let you concentrate
  excitation where it matters, but the frequency selection determines which
  dynamics you can check.

Generating random inputs
========================

You do not need Python on the device running the experiment. On a
microcontroller, you can generate pseudo-random samples in C/C++, or replay
a sequence generated beforehand on a PC.

PRBS is often a convenient choice for embedded experiments. A
linear-feedback shift register (LFSR) generates it with little memory and
computation, using suitable feedback taps and a valid initial state. Map its
binary output to two safe input levels and choose a switching interval and
sequence length that exercise the dynamics of interest. Saving the initial
state and generator settings makes the sequence repeatable.

A general-purpose pseudo-random generator can provide other amplitude
distributions. Whether the resulting input resembles white noise depends
on its statistical properties and update timing, not the programming
language. Holding samples between updates or filtering them introduces
temporal correlation.

PRBS is a useful practical default, not the only choice: chirps and
multisines can be preferable when you need more control over the excited
frequencies or actuator motion.

For PC-based generation, NumPy can produce repeatable pseudo-random samples.
For example, generate two channels of bounded, uniform noise around an
operating point, which you could then export for replay on the device:

.. code-block:: python

   import numpy as np

   rng = np.random.default_rng(42)
   sampling_period = 0.01
   time = np.arange(10000) * sampling_period
   operating_point = np.array([5.0, 3.0])
   amplitude = np.array([0.5, 0.2])
   inputs = operating_point + amplitude * rng.uniform(
       -1.0, 1.0, size=(len(time), 2)
   )

Each column uses separate draws from the generator. The fixed seed makes
the example reproducible; it does not make the two channels identical.
The numerical levels above are illustrative, not safe settings for a
particular plant. Check amplitude and rate limits before applying any signal.

Use ``rng.standard_normal((len(time), 2))`` for unit-standard-deviation
Gaussian samples, or ``rng.choice([-1.0, 1.0], size=(len(time), 2))`` for
random binary samples. Gaussian samples are unbounded; a random binary
sequence is not necessarily a designed maximum-length PRBS.

If the system needs slower excitation, choose an appropriate update interval
or use a SciPy low-pass filter to shape the bandwidth. The resulting signal
will be colored, which is not inherently a problem. Inspect the final
signal's spectrum, amplitude, and rate of change, not just the generator's
original samples.

Checking the applied inputs
==============================

For multiple inputs, design the channels together. Using the same PRBS or
random signal on every channel does not provide independent excitation.
Choose suitable sequences or frequency allocations and check their
relationships over the actual record; low cross-correlation is not automatic.

Choose amplitudes, switching intervals or sweep rates, duration, and operating
points for your system and application. A well-designed signal makes useful
excitation more likely, but does not guarantee a suitable validation dataset:
actuator limits, measurement noise, disturbances, and the system response
still matter.

Methods 1 and 2 can assess existing logs *and* check the measurements from
these planned experiments. Signal design and data assessment complement each
other; neither replaces inspection of the recorded inputs and outputs.

*****************
 Coverage region
*****************

Useful excitation is only part of the picture. Your experiments should also
cover the signal values and operating conditions relevant to your application,
within safe limits. This is the *coverage region*.

Broad coverage does not automatically mean that the experiment reveals the
system's dynamics, as shown in the next example.

   **Example**

   With reference to the example above, imagine driving the car only at
   constant speeds ranging from 0 to 180 km/h on a flat road, without ever
   accelerating or braking. For instance, you make a first run driving (and
   logging) data at a constant speed of 10 km/h, without accelerating or
   braking, and staying on a flat road. Then, you perform a second run at a
   constant speed of 20 km/h under the same conditions as the previous run,
   and so on, until reaching the final run at 180 km/h.

   Your measurement dataset will have a fairly large coverage region and can
   reveal steady-state behavior. However, it will contain little information
   about acceleration and transient behavior, since all the runs were
   conducted at constant speeds without any acceleration or braking, and on
   a flat road.

At the end of this phase, check both operating-region coverage and whether
the variations in the data reveal the dynamics you want to model. Neither
input whiteness nor a broad range of amplitudes alone is enough.
