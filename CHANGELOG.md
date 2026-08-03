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
- `plot_spectrum`'s `kind` argument is now `mode`.
- Plotting functions never call `show()`; they return the `Figure`. The
  `is_interactive` configuration key no longer influences plotting.
- Renames: `remove_means` → `remove_mean`, `remove_offset` →
  `remove_constant`, `remove_NaNs` → `remove_nans`, `plotxy` →
  `plot_xy`, `dump_to_signals` → `to_signals`.
- `Dataset.fft` returns `{name: (freq, values)}` instead of a
  `DataFrame`; `Dataset.coverage` and `Dataset.dataset_values` return
  `numpy` arrays.

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
- A dedicated `statistics`, `xcorrelation`, `signal`, `scope` and
  `plotting` module, replacing the two monolithic ones.

### Unchanged

- `ValidationSession`, `validate_models`, `XCorrelation`,
  `whiteness_level`, `compute_statistic` and `rsquared` keep their
  signatures and their numerical behaviour.
