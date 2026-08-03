.. _migrating:

##########################
 Migrating from 0.9 to 1.0
##########################

Release 1.0 is a rewrite of the *dymoval* core. The package no longer
depends on ``pandas``: a :ref:`Signal <signal>` is now a small ``numpy``
dataclass and a :ref:`Dataset <Dataset>` a pair of signal mappings sharing
one time vector, instead of a wrapper around a multi-indexed
``DataFrame``.

This page maps the old API onto the new one. The validation half of the
package — :ref:`ValidationSession <ValidationSession>`,
:py:func:`validate_models <dymoval.validation.validate_models>`,
:py:class:`XCorrelation <dymoval.xcorrelation.XCorrelation>` — is
essentially unchanged; almost everything below concerns signals, datasets
and plotting.

*********************
 Signals are objects
*********************

A :ref:`Signal <signal>` used to be a ``TypedDict``. It is now a
dataclass with a ``time`` vector rather than a ``sampling_period``, which
is what makes non-uniformly sampled measurements representable:

.. code::

   # 0.9
   sig = {
       "name": "speed",
       "samples": values,
       "signal_unit": "m/s",
       "sampling_period": 0.1,
       "time_unit": "s",
   }

   # 1.0
   sig = dmv.Signal(
       name="speed",
       values=values,
       time=np.arange(len(values)) * 0.1,
       unit="m/s",
       time_unit="s",
   )

============================ ==========================================
0.9                          1.0
============================ ==========================================
``sig["samples"]``           ``sig.values``
``sig["signal_unit"]``       ``sig.unit``
``sig["sampling_period"]``   ``sig.get_sampling_period()``
``len(sig["samples"])``      ``len(sig)``
============================ ==========================================

``time`` may be left to ``None``, in which case the signal is
sample-indexed.

Signals also grew the whole processing toolbox that used to live on the
:ref:`Dataset <Dataset>` only: ``trim``, ``resample``, ``detrend``,
``remove_mean``, ``remove_constant``, ``low_pass_filter``, ``apply``,
``fft``, ``spectrum``, ``remove_nans``, ``plot``, ``plot_spectrum``.

*********************************
 Datasets are built by factories
*********************************

The constructor no longer sorts signals into inputs and outputs by name.
Pass them explicitly to a factory:

.. code::

   # 0.9
   ds = dmv.Dataset(
       "my dataset",
       signal_list,
       u_names=["voltage"],
       y_names=["current", "speed"],
       target_sampling_period=0.1,
   )

   # 1.0
   ds = dmv.Dataset.from_signals(
       inputs=[voltage],
       outputs=[current, speed],
       target_sampling_period=0.1,
       name="my dataset",
   )

:py:meth:`Dataset.from_dict <dymoval.dataset.Dataset.from_dict>` takes
the same information as ``{"inputs": [...], "outputs": [...]}``.

The factories resample onto a common grid and *validate*, which is why
``validate_signals`` and ``validate_dataframe`` are gone: an invalid set
of signals now simply raises. The ``tin``/``tout``/``full_time_interval``
constructor arguments are gone too — build the dataset, then
:py:meth:`trim <dymoval.dataset.Dataset.trim>` it.

Renamed and reshaped members
============================

===================================== =========================================
0.9                                   1.0
===================================== =========================================
``ds.dataset`` (a ``DataFrame``)      ``ds.inputs`` / ``ds.outputs``, or
                                      :py:meth:`ds.dataset_values()
                                      <dymoval.dataset.Dataset.dataset_values>`
``ds.coverage`` (a ``DataFrame``)     :py:meth:`ds.coverage()
                                      <dymoval.dataset.Dataset.coverage>`
``ds.remove_means()``                 :py:meth:`ds.remove_mean()
                                      <dymoval.dataset.Dataset.remove_mean>`
``ds.remove_offset(("u1", 3.0))``     :py:meth:`ds.remove_constant(("u1", 3.0))
                                      <dymoval.dataset.Dataset.remove_constant>`
``ds.remove_NaNs()``                  :py:meth:`ds.remove_nans()
                                      <dymoval.dataset.Dataset.remove_nans>`
``ds.plotxy()``                       :py:meth:`ds.plot_xy()
                                      <dymoval.dataset.Dataset.plot_xy>`
``ds.dump_to_signals()``              :py:meth:`ds.to_signals()
                                      <dymoval.dataset.Dataset.to_signals>`
``ds.fft()`` → ``DataFrame``          ``{name: (freq, values)}``
``ds.apply(("u1", f, "V^2"))``        unchanged, but ``f`` receives a
                                      ``numpy`` array
===================================== =========================================

:py:meth:`ds.dataset_values() <dymoval.dataset.Dataset.dataset_values>`
returns ``(time, inputs, outputs)`` as plain arrays and is the direct
replacement for reaching into the old ``DataFrame``. ``inputs`` and
``outputs`` are always 2-D, with shape ``(n_samples, n_signals)``, even
for a SISO dataset: use ``u[:, 0]`` where you would have got a 1-D array.

The helpers ``factorize``, ``difference_lists_of_str`` and ``obj2list``
were internal plumbing that happened to be exported. They are now
private; ``numpy`` and the standard library cover what they did.

**********************
 Spectra are rescaled
**********************

The magnitude spectra changed value, so numbers read off a 0.9 plot will
not match a 1.0 one.

:py:meth:`Signal.fft <dymoval.signal.Signal.fft>` is normalised by the
number of samples. 0.9 did this too, but the 1.0 rewrite lost it along
the way, which made the amplitudes grow with the length of the record.

On top of that, ``amplitude``, ``power`` and ``psd`` now fold the
negative half of the spectrum onto the positive half. 0.9 intended to do
this — the code and its comments are explicit about it — but the line
meant to do the folding was label-based slicing on a frequency index and
quietly did nothing.

The practical consequences are worth knowing:

-  a sine of amplitude :math:`A` peaks at :math:`A` in ``amplitude``
   mode, not at :math:`A/2` and not at :math:`AN/2`;
-  ``power`` sums, and ``psd`` integrates, to the mean square of the
   signal, so Parseval's theorem holds;
-  DC and, for an even number of samples, the Nyquist bin are not
   doubled, because neither has a mirror image.

``psd_welch`` was already correct and is unchanged, so it is the mode to
compare against if you want to check the others.

*************************
 Overlapping is grouping
*************************

The ``overlap=True`` flag, which paired the *i*-th input with the *i*-th
output, is replaced by explicit grouping: any tuple of names passed to a
plotting method is drawn on one and the same axes.

.. code::

   # 0.9
   ds.plot(overlap=True)

   # 1.0
   ds.plot(("u1", "y1"), ("u2", "y2"))

This works for :py:meth:`plot <dymoval.dataset.Dataset.plot>` and
:py:meth:`plot_spectrum <dymoval.dataset.Dataset.plot_spectrum>`, and
lets you overlap *any* signals, not just same-index pairs.

Line styling
============

The ``linecolor_input``, ``linestyle_fg``, ``alpha_fg``,
``linecolor_output``, ``linestyle_bg``, ``alpha_bg`` arguments collapsed
into ``color_input`` / ``color_output`` plus ``**kwargs`` forwarded
straight to ``matplotlib``:

.. code::

   # 0.9
   ds.plot(linecolor_input="blue", linestyle_fg="--", alpha_fg=0.5)

   # 1.0
   ds.plot(color_input="blue", linestyle="--", alpha=0.5)

The ``layout``, ``ax_height`` and ``ax_width`` arguments are unchanged
and are now available on *every* plotting function — see
:ref:`figure_geometry`.

*******************
 Comparing datasets
*******************

``compare_datasets(*datasets, kind=...)`` is split into one function per
kind, which removes the guesswork about which arguments apply:

``compare_datasets(a, b)``, ``compare_datasets(a, b, kind="time")``
   :py:func:`plot_compare(a, b) <dymoval.plotting.plot_compare>`

``compare_datasets(a, b, kind="coverage")``
   :py:func:`plot_coverage_compare(a, b)
   <dymoval.plotting.plot_coverage_compare>`

``compare_datasets(a, b, kind="power")`` (or ``"amplitude"``, ``"psd"``)
   :py:func:`plot_spectrum_compare(a, b, mode="power")
   <dymoval.plotting.plot_spectrum_compare>`

All of them accept ``labels=[...]`` to name the compared datasets in the
legend.

*********
 Spectra
*********

The ``kind`` argument became ``mode``, and a fourth mode was added:

.. code::

   # 0.9
   ds.plot_spectrum(kind="psd")

   # 1.0
   ds.plot_spectrum(mode="psd")

Allowed values are ``"amplitude"``, ``"power"``, ``"psd"`` and
``"psd_welch"``. Note that :py:meth:`Dataset.spectrum
<dymoval.dataset.Dataset.spectrum>` takes signal names positionally, so
``mode`` must be passed by keyword: ``ds.spectrum("u1", mode="psd")``.

``xscale`` and ``yscale`` are validated too: ``xscale`` accepts
``"linear"`` and ``"log"``, ``yscale`` also accepts ``"db"``. An
unsupported value used to be silently ignored and now raises.

**************
 Plot output
**************

Plotting functions **never** call ``show()`` any more: they return the
``Figure`` (or the tuple of figures, for
:py:meth:`plot_residuals
<dymoval.validation.ValidationSession.plot_residuals>`). Call
``matplotlib.pyplot.show()`` yourself when you want a window.

Consequently the ``is_interactive`` configuration key no longer
influences plotting.

Conversely, most plots now carry an interactive **scope**: click a curve
to inspect it, press ``r`` to reset. Pass ``with_scope=False`` to opt
out. A scope owns its whole figure, so :py:meth:`Signal.plot
<dymoval.signal.Signal.plot>` and :py:meth:`Signal.plot_spectrum
<dymoval.signal.Signal.plot_spectrum>` turn it off automatically when
you hand them an ``ax`` to draw on, and refuse to do both at once.

******************
 Removed for good
******************

============================= ==============================================
Removed                       Why / what to use
============================= ==============================================
``validate_signals``          the factories validate and raise
``validate_dataframe``        ``DataFrame`` input is gone entirely
``change_axes_layout``        it had no call site and discarded the contents
                              of the axes it re-laid-out. Use
                              ``fig.clear(); fig.subplots(nrows, ncols)``
``Dataset.excluded_signals``  the factories interpolate rather than exclude
``scope``, ``scope_compare``, superseded by ``with_scope=True``
``InteractiveScope``,
``plot_multi``
============================= ==============================================
