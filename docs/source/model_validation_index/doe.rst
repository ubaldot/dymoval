.. _doe:

#############################
 Design of Experiments (DoE)
#############################

The goal of Design of Experiments (DoE) is to find suitable experiments to
expose as much as possible of the system under examination.

Good experiments shall stress the target environment under
different conditions to get a sufficiently *informative* measurement dataset.


   **Example**

   If we are developing the model of a car, we want to log sensor
   measurements while driving within a wide range of speeds, with different
   acceleration profiles, in different road and weather conditions and so on
   and so forth.

   If we log sensor measurements only when we are driving on a flat road and
   in the range 0-10 km/h and by doing exactly the same maneuvers over and
   over, the dataset would tell us little about behavior outside those
   conditions.

We may have two different situations: some logged measurements may be
available and we want to assess if they are rich enough for model
validation purposes or we have to run some experiments from scratch and we
wonder how to select input signals to obtain a sufficiently rich measurement
dataset.

*******************************
 Measurement logs are available
*******************************

How can you tell whether an existing measurements log contains enough information?
In fact, there is no single answer for every model, but there
are three useful questions to ask:

* Did the inputs excite the dynamics and input directions we care about?
* Would shortcomings in the model show up in this experiment?
* Did the experiment cover the operating conditions where you will use it?

To assess the input-excitation part of these questions, we can rely on two
related tools:

1. *Input auto- and cross-correlation:* inspect relationships between delayed
   input samples and redundancy between input channels. If the correlation is
   low, it suggests that the input excitation is less redundant in the dataset,
   which is desirable.

2. *Lagged-input Gram-matrix eigenvalues:* check whether any combination of
   input channels and their delayed samples remains unexcited over the
   assessed record, using stacks of :math:`L` consecutive input samples.
   A positive minimum eigenvalue means that none does; its size indicates
   how strongly the weakest combination is excited.

Neither is a replacement for the other: a correlation plot is easier to
interpret, while the eigenvalues show whether combinations of input channels
and delayed samples are missing or barely present.
Substantial correlation does not automatically mean poor excitation, and a
small average correlation score does not guarantee a well-conditioned Gram
matrix.

Nevertheless, whichever method you use, we should check that the output response is clearly visible above measurement
noise, the signals are not clipped, and actuators or sensors have not
saturated unintentionally. Also check timing alignment, recording duration,
relevant frequency content, and operating conditions. Output movement alone is
not enough if it comes mainly from disturbances or sensor noise.
Neither method replaces these checks. Saturation may be something you
deliberately want to study, but it changes how you should interpret the
results, particularly when assessing a linear model.

Let's see more in detail each of the two methods.


Method 1: input auto- and cross-correlation
===========================================

This method asks: *what time patterns and channel relationships are present
in the inputs?* Auto-correlation compares an input with delayed versions of
itself; cross-correlation compares different input channels at selected delays.
It helps you spot slow variation, repeated patterns, or
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
   Ruu.correlation_score()

``correlation_score()`` returns a correlation summary:
lower values mean less correlation over the evaluated lags. The aggregation
can be selected through :py:meth:`~dymoval.statistics.compute_statistic`.
This is neither a formal statistical whiteness test nor a test of persistent
excitation.

.. Next, look at the outputs alongside the applied inputs. A large output range
.. alone is not enough: movement may come from disturbances or sensor noise
.. rather than the input. Look for a plausible response to the input changes,
.. and record long enough to see the relevant transients.

These are practical screening checks, not a certificate that the data is
informative. Colored inputs can be useful, while small correlation scores
do not guarantee that the input excites the dynamics you need.

Sampling and correlation
---------------------------

If a signal is over-sampled, consecutive measurements are often correlated
simply because the sampling interval is much shorter than the time scale of
the signal. This can make the auto-correlation appear large near zero lag,
even when the experiment contains useful excitation. The result may then
describe the sampling rate more than the signal's relevant dynamics.

Inspect the correlation against physical delay, not just lag index, over
a window relevant to the behavior you want to validate. Correlated input
samples do not automatically mean the experiment is unsuitable, and
excluding delays within a system time constant can hide useful information.
If you deliberately resample a log, preserve the relevant dynamics and
apply appropriate anti-alias filtering before reducing the sampling rate.

*Dymoval* evaluates correlations at one-sample spacing; it does not
automatically change the lag grid based on bandwidth. Choose a meaningful
physical delay range and inspect the correlation before summarizing it.
Keep that range fixed when comparing scores: adding many low-correlation
lags can dilute an average score.

Method 2: Lagged-input Gram-matrix eigenvalues
====================================================

This method asks: *does the recorded input excite every combination of
channels and their delayed samples?* A positive minimum eigenvalue means
that every such direction is excited over the assessed record. Its magnitude
indicates how strongly the weakest direction is excited, for the chosen
input-window length :math:`L` and channel scaling.

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
energy in every direction of this lagged input vector over the record.
The required input-window length :math:`L` depends on the time scales and input
patterns relevant to the behavior you want to validate;
the condition is not a universal test of model quality. For finite records,
full rank of this matrix is commonly called persistent excitation of order
:math:`L`. For an ongoing signal, a stronger condition requires excitation
to remain bounded below over successive segments, rather than occurring
only once near the start of the recording.

Finding useful portions of a log
----------------------------------

You will often start by looking at the plots and trimming away periods where
nothing happens. An excitation score can complement that visual inspection.

First, distinguish two lengths:

* :math:`L` is the number of consecutive input samples in each stack
  :math:`\psi_L`.
* :math:`W` is the number of measurement samples in the segment you want
  to assess.

.. figure:: ../figures/LaggedInputWindows.svg
   :alt: A full log of N samples contains an assessment segment of W samples,
         with overlapping stacks of L samples fully inside that segment.
   :align: center
   :width: 100%

   The full log, one assessment segment, and its overlapping input stacks.
   The diagram runs forward in time; each stack is stored newest sample first.

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
Use :py:meth:`~dymoval.dataset.Dataset.input_gram_matrix` to compute the
averaged whole-record matrix, or
:py:meth:`~dymoval.dataset.Dataset.input_gram_eigenvalues` to obtain its
eigenvalues directly:

.. code:: python

   L = 20  # Choose a stack length appropriate for the dynamics of interest.
   gram = ds.input_gram_matrix(L)
   eigenvalues = ds.input_gram_eigenvalues(L)  # Ascending order.
   weakest_excitation = eigenvalues[0]
   print(f"Weakest input excitation (L={L}): {weakest_excitation:.6g}")

The method uses the inputs in dataset order, without centering or scaling.
It requires finite input values. The outputs do not enter this computation.
For a selected segment, trim the dataset first and call the same method.

Input-excitation assessment belongs to the dataset and is independent of
model predictions. It is requested separately from
:py:class:`~dymoval.validation.ValidationSession` and does not change its
reports, validation thresholds, or PASS/FAIL outcomes.

**************************
 Planning a new experiment
**************************

When you have to set up an experiment, the main question is often related to
how we should stimulate the system under examination, or, in other words, what
input signal to use.
Fortunately, there are common signals that are used to exercise
the dynamics you want to validate:

* **Noise-like inputs: PRBS and random signals** A
  *Pseudo-Random-Binary-Sequence* (PRBS) is a deterministic, repeatable
  sequence switching between two levels that resembles band-limited white
  noise. It provides broad excitation over a useful frequency band.
  Random signals can instead use Gaussian, uniform, or binary amplitude
  distributions. Independent, zero-mean random samples are white in
  expectation, not exactly uncorrelated in every finite record.
* **Chirps** sweep through a frequency range, letting you explore the response
  at different frequencies over time. They need not resemble white noise or
  produce a small auto-correlation score.
* **Multisines** combine selected frequencies. They let you concentrate
  excitation where it matters, but the frequency selection determines which
  dynamics you can check.


These signals are useful starting points for designing informative
experiments, but their settings matter. Choose their frequency content,
amplitudes, and relationships between input channels carefully, then check
the recorded inputs using the methods above. Also check that the outputs
respond clearly above measurement noise.

.. Methods 1 and 2 can assess existing logs *and* check the measurements from
.. these planned experiments. Signal design and data assessment complement each
.. other; neither replaces inspection of the recorded inputs and outputs.

.. note::

   Safety and operating constraints may limit the inputs you can apply.
   Do not aim for randomness at the expense of a safe, representative
   experiment. If input correlation is not a suitable pass/fail criterion,
   pass ``ignore_input=True`` to
   :py:meth:`~dymoval.validation.validate_models` or to
   :py:class:`~dymoval.validation.ValidationSession` class constructor.
   This disables the input-correlation pass/fail check; it does not establish
   that the experiment has sufficient excitation.

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
low input correlation nor a broad range of amplitudes alone is enough.
