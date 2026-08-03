# Changelog

All notable changes to this project are documented in this file.

## [1.0.0]

A rewrite of the *dymoval* core. **This release is not backwards
compatible**: see the [migration
guide](https://ubaldot.github.io/dymoval/migrating.html) for a
point-by-point mapping of the old API onto the new one.

### Removed

- **The `pandas` dependency.** `Signal` and `Dataset` are now built
  directly on `numpy`. The runtime dependencies are `numpy`, `scipy`,
  `matplotlib`, `control` and `h5py`.
- `validate_signals` and `validate_dataframe`: the `Dataset` factories
  validate their input and raise on invalid data.
- Construction of a `Dataset` from a `DataFrame`.
- `change_axes_layout`: it had no call site and it discarded the contents
  of the axes it re-laid-out. `fig.clear(); fig.subplots(nrows, ncols)`
  is the native replacement.
- `Dataset.excluded_signals`: the factories resample rather than exclude.
- `compare_datasets`, replaced by three explicit functions.
- The legacy `scope`, `scope_compare`, `InteractiveScope` and
  `plot_multi` helpers, superseded by `with_scope=True`.

### Changed

- `Signal` is a dataclass, not a `TypedDict`, and carries an explicit
  `time` vector rather than a `sampling_period`, so non-uniformly sampled
  measurements can be represented and then resampled.
- A `Dataset` is built with `Dataset.from_signals(inputs=..., outputs=...)`
  or `Dataset.from_dict(...)` instead of the name-matching constructor.
- `overlap=True` is replaced by explicit grouping: any tuple of names
  passed to a plotting method is drawn on the same axes, e.g.
  `ds.plot(("u1", "y1"))`.
- The `linecolor_*` / `linestyle_*` / `alpha_*` plotting arguments
  collapsed into `color_input` / `color_output` plus `**kwargs`
  forwarded to `matplotlib`.
- `plot_spectrum`'s `kind` argument is now `mode`, and `xscale`/`yscale`
  are validated instead of silently falling back to linear.
- Plotting functions never call `show()`; they return the `Figure`. The
  `is_interactive` configuration key no longer influences plotting.
- `Signal.plot(ax=...)` and `Signal.plot_spectrum(ax=...)` draw on the
  passed axes instead of silently creating a scope figure of their own.
- `Dataset.trim` picks the interval graphically when neither `tin` nor
  `tout` is given; `shift_to_zero`, `show` and `verbosity` are
  keyword-only.
- `ValidationSession.drop_simulations` raises `KeyError` rather than
  `ValueError` for an unknown simulation, matching the rest of the
  package.
- Renames: `remove_means` → `remove_mean`, `remove_offset` →
  `remove_constant`, `remove_NaNs` → `remove_nans`, `plotxy` →
  `plot_xy`, `dump_to_signals` → `to_signals`.
- `Dataset.fft` returns `{name: (freq, values)}` instead of a
  `DataFrame`; `Dataset.coverage` and `Dataset.dataset_values` return
  `numpy` arrays.
- `Dataset.dataset_values` always returns 2-D `(n_samples, n_signals)`
  arrays. It used to flatten to 1-D when the dataset held a single input
  (resp. output), so downstream code had to handle both ranks.
- `Dataset.remove_constant` takes `(name, constant)` tuples, like its
  siblings `apply` and `low_pass_filter`, instead of a
  `{name: constant}` mapping. A lone scalar still applies to every
  signal.
- `Dataset.apply` and `Dataset.low_pass_filter` reject tuples of the
  wrong length instead of silently ignoring the extra elements.
- `factorize`, `difference_lists_of_str` and `obj2list` are no longer
  exported: they were internal plumbing.

### Fixed

- **Spectra were not normalised.** `Signal.fft` and `Dataset.fft` now
  divide by the number of samples, as 0.9 documented and did, and the
  `amplitude`, `power` and `psd` modes fold the negative frequencies
  onto the positive ones. A sine of amplitude `A` now reads `A` in
  `amplitude` mode instead of `A * N / 2`, and `power` sums (`psd`
  integrates) to the mean square of the signal. 0.9 meant to fold too,
  but its `df_freq.loc[1:-1] *= 2` was label-based slicing on a
  frequency index and silently did nothing. `psd_welch` was already
  correct and is unchanged.
- **The input-residuals cross-correlation used the wrong bandwidths.**
  `Rue` correlates the *input* with the residuals but passed the output
  bandwidths for both axes. It raised `IndexError` whenever the number
  of inputs differed from the number of outputs, and downsampled
  incorrectly when they happened to match.
- `ValidationSession.__init__` assigned `validation_thresholds` straight
  to the attribute, so, unlike the property setter, it accepted unknown
  keys and negative values. An empty mapping is now rejected too: it
  made every simulation PASS unconditionally.
- `compute_statistic` chained its argument checks with `elif`, so only
  the first one ever ran: a 2-D `weights`, or one of the wrong length,
  went through unnoticed as long as it had no negative entry. All-zero
  and non-finite weights are rejected as well, since every statistic
  divides by the sum or the maximum of the weights.
- `rsquared` raises instead of returning `nan`/`-inf` when the reference
  signal is constant and there is therefore no variance to explain.
- `Dataset.coverage` computed the mean with `nanmean` but the covariance
  with `np.cov`, which propagates a single missing sample across a whole
  row and column. Both now use the samples that are complete for every
  signal.
- `XCorrelation` silently ignored `X_bandwidths`/`Y_bandwidths` unless
  the two of them *and* `sampling_period` were given. Supplying only
  part of them now raises.
- `ValidationSession.simulation_signals_list` accepted a list but only
  ever looked at its first element.
- `ValidationSession.plot_simulations` silently treated an unrecognised
  `dataset` value as `None`.
- `Dataset.apply`, `remove_constant` and `low_pass_filter` raise on a
  repeated signal name instead of quietly keeping the last one.

### Added

- `Signal` gained the whole processing toolbox that used to live on
  `Dataset` only: `trim`, `resample`, `detrend`, `remove_mean`,
  `remove_constant`, `low_pass_filter`, `apply`, `fft`, `spectrum`,
  `plot`, `plot_spectrum`.
- Interactive **scopes** on most plots: click a curve to inspect it,
  press `r` to reset. Opt out with `with_scope=False`.
- Missing-data handling: `has_nans`, `nan_intervals` and `remove_nans`
  on both `Signal` and `Dataset`. Gaps are shaded in every time plot.
- `psd_welch` spectrum mode.
- `layout`, `ax_height` and `ax_width` on every plotting function, and
  `Dataset.plot_xy(*pairs)` accepting arbitrary signal pairs.
- Graphical `tin`/`tout` picking in `Dataset.trim`, shared with
  `ValidationSession.trim`.
- `Dataset.align`, `Dataset.pipe`, and per-signal selection in
  `Dataset.fft` and `Dataset.spectrum`.
- The `SpectrumMode`, `SignalKind`, `Scale`, `SpectrumScale`, `AlignMode`
  and `Layout` type aliases are exported from the top-level package,
  alongside the runtime tuples of their allowed values.
- A dedicated `statistics`, `xcorrelation`, `signal`, `scope` and
  `plotting` module, replacing the two monolithic ones.

### Unchanged

- `ValidationSession`, `validate_models`, `XCorrelation`,
  `whiteness_level`, `compute_statistic` and `rsquared` keep their
  signatures and their numerical behaviour.
