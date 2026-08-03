# dymoval Development Handover

## Project Goal

A lightweight signal-processing and visualization framework built on
NumPy/SciPy/Matplotlib (**no pandas**) around two core abstractions:

```python
Signal
Dataset
```

---

# Core Design Rule

```text
Signal
    computation
    primitive plotting (one signal on one axes)

Dataset
    orchestration
    grouping
    layout

Scope
    interaction
```

`Dataset` reuses `Signal` internals

```python
sig._plot_standard(...)
sig._plot_spectrum_standard(...)
sig._plot_phase_standard(...)
sig._plot_spectrum_amplitude(...)
```

All three `Signal._plot_*_standard` primitives funnel through a single
private `Signal._draw(ax, x, y, xlabel, ylabel, **kwargs)`: it is the only
place where a `Signal` touches matplotlib, and the only place that tags the
line with the `line._signal` back-reference the scopes rely on.

and **never** calls the public plotting methods internally.

The same rule holds one level up: `XCorrelation` owns the primitives
`_plot_standard(ax, ii, jj)` (one stem plot on one axes) and
`_plot_grid(axes, x_symbol, y_symbol)` (a `p x q` grid), and both
`XCorrelation.plot()` and `ValidationSession.plot_residuals()` are built
on top of them.

Two further conventions:

- **Every figure that may carry a scope is created by
  `dymoval.scope.scope_subplots(nrows, ncols, *, with_scope, figsize,
  **subplots_kwargs)`**, which returns `(fig, axes, panel_ax)` with a
  *flat* `axes` list and `panel_ax is None` when `with_scope=False`. The
  caller only attaches the scope class it needs. This replaced ten
  copy-pasted `plt.figure(...) / fig.subfigures(...) / add_subplot() /
  set_anchor("N")` blocks.
- **Plotting functions never call `show()`** (nor `fig.show()`), they only
  return the `Figure`. Displaying is the caller's / the backend's job;
  this also keeps the test suite warning-free.

---

# Module map

| module        | status | content                                             |
| ------------- | ------ | --------------------------------------------------- |
| `signal.py`   | new    | `Signal`                                             |
| `dataset.py`  | new    | `Dataset`                                            |
| `scope.py`    | new    | `BaseScope` / `SignalScope` / `DatasetScope` / `SpectrumScope` / `AmplitudeSpectrumScope`, plus the `scope_subplots()` figure factory |
| `plotting.py` | new    | multi-object helpers only                            |
| `statistics.py` | new  | `compute_statistic`, `rsquared`                      |
| `xcorrelation.py` | new | `XCorrelation`, `whiteness_level`                   |
| `validation.py` | new  | `ValidationSession`, `validate_models`               |
| `utils.py`, `config.py` | kept | unchanged                                  |

The package is now **pandas-free**: `dataset_old.py`, the legacy
`tests/legacy/` suite and the `pandas` / `mpl-measurements` dependencies
have all been removed. `src/dymoval_tutorial/` (the notebook and
`tutorial_debug.py`) has been rewritten on the new API and the notebook is
now shipped without stored outputs.

`utils_internal.py` (never imported), the `R2_STATISTIC_TYPE` config entry
and the `tomli` dependency (unreachable: `requires-python >= 3.11` always
provides `tomllib`) have been dropped as dead code.

---

# Signal

```python
@dataclass
class Signal:
    name: str
    values: np.ndarray
    time: np.ndarray | None = None
    unit: str | None = None
    time_unit: str | None = "s"
```

All the processing methods are **immutable**: they return a new `Signal`.

```python
copy()
detrend()
remove_mean()            # subtracts the signal mean
remove_constant(value)   # subtracts a user-defined constant
apply(func, unit=None)   # vectorized call, element-wise fallback
low_pass_filter(cutoff)  # first-order IIR
trim(tin, tout, shift_to_zero=True)
resample(new_time)
get_sampling_period()

fft()                    # (freq, complex one-sided spectrum)
spectrum(mode)           # (freq, spectrum)

plot(ax=None, with_scope=True)
plot_spectrum(ax=None, with_scope=True, xscale, yscale, mode)
```

Every plotted line carries `line._signal`, so scopes can recover the
signal name and units.

`low_pass_filter` reproduces the legacy first-order recursion exactly:

```text
alpha = cutoff / fs
y[0]   = u[0]
y[k+1] = (1 - alpha) * y[k] + alpha * u[k]
```

with `0 < cutoff < fs`.

---

# Dataset

```python
inputs: dict[str, Signal]
outputs: dict[str, Signal]
meta: dict | None
```

## Validation

- non empty, no duplicated names across inputs/outputs
- all signals have `time` and at least 2 samples
- same sampling period, same length, same time vector
- one single `time_unit` shared by every signal

The plain `Dataset(...)` constructor is **strict**: it never repairs
anything, it only validates.

## Construction

```python
Dataset.from_signals(inputs=[...], outputs=[...], target_sampling_period=None)
Dataset.from_dict({"inputs": [...], "outputs": [...]}, target_sampling_period=None)
```

The **factories** harmonize: signals living on different grids are
resampled (interpolated) onto a common uniform grid

```text
dt   = target_sampling_period or max(sampling periods)
span = [max(t_start), min(t_end)]      # intersection => no extrapolation
```

`Dataset._common_time()` returns `None` when the signals already agree, so
the original time array is preserved bit-exactly.

> This replaces the legacy `_fix_sampling_periods()`, which decimated
> (`values[::N]`) only when the ratio was an integer and **excluded** the
> other signals. Non-integer ratios are now supported and
> `excluded_signals` no longer exists.

## API

```python
# introspection
time_unit()
kind_of(name)              # "INPUT" | "OUTPUT"
signal_list()              # [(kind, name, unit), ...]
to_signals()               # {kind: list[Signal]}
dataset_values()           # (time, U, Y) as plain ndarrays
export_to_mat(filename)    # scipy.io.savemat, TIME + INPUT/OUTPUT structs
repr(ds)

# structure
add_input(*signals)        # resampled onto the dataset time grid
add_output(*signals)
remove_signals(*names)     # may not empty the dataset

# processing (all immutable)
copy()
detrend(*names)            # no name => every signal
remove_mean(*names)
remove_constant(value | {name: value})
apply((name, func[, unit]), ...)
low_pass_filter((name, fc), ...)
trim(tin=None, tout=None, shift_to_zero=True)
resample(new_time)
align(other, how="intersection" | "union")
pipe(func)

fft()
spectrum(mode)

# coverage
coverage()                 # (u_mean, u_cov, y_mean, y_cov)
plot_coverage(*names, nbins=100, ...)

# plotting
plot(*groups, with_scope=True, color_input, color_output, **kwargs)
plot_xy(*pairs, ax=None, **kwargs)   # no pair => zip(inputs, outputs)
plot_spectrum(*groups, with_scope=True, xscale, yscale, mode, **kwargs)
```

`SIGNAL_KIND = ("INPUT", "OUTPUT")` is exported by the package.

`align()` always returns a **uniform** time vector built with the sampling
period of `self`, so the resulting datasets stay valid.

## Figure geometry

Every figure-returning plot function takes the same three knobs:

| argument    | meaning                                    |
| ----------- | ------------------------------------------ |
| `layout`    | matplotlib layout engine, `"constrained"` (default), `"compressed"`, `"tight"` or `"none"`. Validated by `scope_subplots`. |
| `ax_height` | height, in inches, of **each subplot**     |
| `ax_width`  | width, in inches, of the **figure**        |

The resulting `figsize` is always `(ax_width, ax_height * n + 1)`, where `n`
is the number of *groups* (the amplitude spectrum uses `+ 2` because each
group owns two rows). Defaults are `_AX_WIDTH = 10.0` / `_AX_HEIGHT = 2.0`
(`7.0` / `1.8` for the coverage histograms) and were chosen to reproduce the
previously hard-coded sizes **exactly**.

Line styling goes through `**kwargs`, forwarded verbatim to
`matplotlib.axes.Axes.plot`. On top of that `Dataset.plot`/`plot_spectrum`
accept `color_input`/`color_output`, the *semantic* colors used when a
subplot holds a single signal; `None` means "use the matplotlib cycle",
which is the default for inputs. Groups of two or more signals always use
the cycle, otherwise the overlaid curves would be indistinguishable.

### Deliberate differences from the legacy implementation

| legacy                                     | new                                       |
| ------------------------------------------ | ----------------------------------------- |
| `remove_offset`                            | `remove_constant`                          |
| `remove_signals` required ≥1 input **and** ≥1 output | only forbids emptying the dataset |
| `add_signal` truncated / NaN-padded        | `add_input`/`add_output` resample          |
| `_fix_sampling_periods` + `excluded_signals` | factory harmonization by interpolation   |

---

# Grouping

Signals passed as a tuple are overlaid on the same subplot:

```python
ds.plot(("u1", "y1"), "y0")
```

```text
subplot 1:  u1 + y1
subplot 2:  y0
```

Grouping is implemented once in `Dataset._normalize_groups()` and reused
by every plotting method (time domain and spectrum). When a group holds a
single *output* signal it is drawn in green; otherwise the matplotlib
color cycle is used.

---

# Spectrum

Modes (`dymoval.SPECTRUM_MODES`):

```python
"amplitude"   # magnitude AND phase
"power"       # |FFT|^2
"psd"         # |FFT|^2 / (n * fs)
"psd_welch"   # scipy.signal.welch, hann, nperseg=min(256, n)  <- default
```

`yscale="db"` uses `20*log10` for `amplitude` and `10*log10` otherwise.

## Phase processing

```python
phase = np.unwrap(np.angle(y)) * 180 / np.pi
phase = np.where(mag > 0.05 * mag.max(), phase, np.nan)
```

Phase is meaningless where the magnitude is negligible, hence the mask.

## Layouts

`Signal`, `mode="amplitude"`:

```text
Magnitude
Phase
| panel
```

`Dataset`, `mode="amplitude"`, N groups → `2 * N` rows:

```text
group1 magnitude
group1 phase
group2 magnitude
group2 phase
| panel
```

One `AmplitudeSpectrumScope` **per group**, so the cursor is local to the
group but magnitude and phase stay linked. All the group scopes write to
the same shared panel; the last clicked one wins (intentional).

---

# Validation

Validation follows the very same "one object, one module" convention as
`Signal` / `Dataset`:

| module            | content                              | role                                        |
| ----------------- | ------------------------------------ | ------------------------------------------- |
| `statistics.py`   | `compute_statistic`, `rsquared`      | pure numpy metrics, no dymoval objects      |
| `xcorrelation.py` | `XCorrelation`, `whiteness_level`    | the leaf object: computation + own plotting |
| `validation.py`   | `ValidationSession`, `validate_models` | the orchestrator, over `Dataset` + simulations |

The dependency graph is strictly layered, exactly like
`signal.py` ← `dataset.py`:

```text
statistics.py  ←  xcorrelation.py  ←  validation.py
                                   ↖  dataset.py, signal.py, scope.py
```

`whiteness_level` lives in `xcorrelation.py` rather than in
`statistics.py` because it is a thin wrapper around
`XCorrelation.estimate_whiteness`; putting it with the other metrics would
have made the two modules mutually dependent.

Everything is still re-exported from the package root, so `dmv.rsquared`,
`dmv.XCorrelation`, `dmv.whiteness_level`, `dmv.compute_statistic` are
unchanged. Only the *submodule* paths moved
(`dymoval.validation.XCorrelation` → `dymoval.xcorrelation.XCorrelation`).

The algorithms themselves were already numpy-based and are untouched; the
port only replaced the *storage* and the *plotting*.

```python
XCorrelation(name, X, Y, nlags, X_bandwidths, Y_bandwidths, sampling_period)
compute_statistic(data, statistic, weights)
rsquared(x, y)
whiteness_level(data, ...)
ValidationSession(name, validation_dataset, ...)
validate_models(measured_in, measured_out, simulated_out, sampling_period, ...)
```

## What changed

| legacy                                        | new                                            |
| --------------------------------------------- | ---------------------------------------------- |
| `_simulations_values` : `pd.DataFrame`         | `_simulations` : `dict[str, list[Signal]]`      |
| `simulations_values` → `pd.DataFrame`          | → `dict[str, np.ndarray]` (`N x q` each)        |
| `_validation_statistics` : `pd.DataFrame`      | `dict[sim_name, dict[key, float]]`              |
| `ds.dataset["INPUT"].to_numpy()`               | `np.column_stack([s.values for s in ds.inputs.values()])` |
| `mpl_measurements.InteractiveScope`            | `DatasetScope`                                  |
| pandas-rendered `__repr__`                     | hand-rolled plain-text table                    |
| `plot_residuals()` → always 3 figures (crashed with `plot_input=False`) | returns 2 or 3 figures |

The statistic keys are exported as
`dymoval.VALIDATION_KEYS = ("Ruu_whiteness", "r2", "Ree_whiteness",
"Rue_whiteness")`; they are also the keys of `validation_thresholds`.

A `ValidationSession` now **requires at least one input and one output**
(the input auto-correlation `Ruu` would be undefined otherwise).

## Plotting

```python
vs.plot_simulations(list_sims=None, dataset=None, layout, ax_height,
                    ax_width, with_scope=True)
vs.plot_residuals(list_sims=None, *, plot_input=True, ...)
```

`plot_simulations` draws one subplot per output through
`sig._plot_standard(...)`, overlays the measured outputs in gray and the
measured inputs on a `twinx()` axes (extra inputs, when `p > q`, get their
own subplot). The `twinx` axes are deliberately **not** handed to the
scope, so only the simulations and the measured outputs are selectable.

`trim()` reuses `Dataset.trim()` and applies the very same trimming to the
stored simulation signals, then recomputes every statistic.

## Internals

The three correlation flavours (`Ruu`, `Ree`, `Rue`) used to be described
by ~20 parallel `_Ruu_*` / `_Ree_*` / `_Rue_*` attributes. They are now
folded into a single `_XCorrSettings` dataclass, instantiated three times
as `self._Ruu`, `self._Ree`, `self._Rue`. It owns the lag resolution
(`build`), the statistic label, the whiteness computation
(`whiteness_of`) and the `__repr__` fragments (`summary`). The write-only
shadow bookkeeping (`_r2_list`, `_Ree_whiteness`, `_*_whiteness_matrix`, ...)
has been deleted: the statistics live only in `_validation_statistics`.

---

# Scope architecture

```python
BaseScope           # panel, cursors, highlighting, reset, statistics
├── SignalScope     # single axes, single signal
└── DatasetScope    # several axes
    └── SpectrumScope             # frequency abscissa, no time statistics
        └── AmplitudeSpectrumScope  # magnitude axes + phase axes
```

`self.axes` is always normalized to `list[Axes]`.

## Lifetime

```python
fig._scopes  # list of every scope attached to the figure
```

## Shared panel

One text artist per panel axes:

```python
panel_ax._shared_info_text
```

Never create a new text object per scope.

## Global reset

Pressing `r` resets **all** the scopes in `fig._scopes`: cursors, cursor
points, clicks and highlighting. The key handler is connected only once
per figure (`fig._scope_key_connected`).

## Line picking

The closest line is resolved in **display** coordinates, so the very
different x/y scales do not bias the selection. Artists created by a scope
are tagged with `_scope_artifact` and are never selectable.

---

# Multi-object plotting (`plotting.py`)

```python
plot_signals(*signals, with_scope=True)             # loose, possibly unaligned Signals
plot_dataset(ds, *groups, with_scope=True)          # alias of ds.plot
plot_compare(ref, *others, names, labels, align, with_scope)
plot_spectrum_compare(ref, *others, ..., mode)      # mode != "amplitude"
plot_coverage_compare(ref, *others, names, labels, nbins, alpha, histtype, align)
```

`plot_signals` is the only way to eyeball raw logs *before* a `Dataset`
exists: it does not require the signals to share a time vector. A tuple of
signals is drawn on a single subplot.

`plot_compare` and `plot_spectrum_compare` are thin wrappers over the
private `_compare(...)` skeleton, which owns the dataset checking, label
resolution, alignment, figure creation and scope attachment. The only
per-flavour parts are the `draw(ax, signal, label)` callable, the scope
class and the optional `finish(ax, name)` touch-up.

---

# Missing data

`NaN` handling lives in the primitive layer and is mirrored by the
orchestrator, like every other processing method.

```python
sig.has_nans()                     # bool
sig.nan_intervals()                # [(t_start, t_end), ...], closed
sig.remove_nans(fill="interpolate")  # or fill="drop"
```

- `nan_intervals` detects the runs of `NaN` with
  `np.flatnonzero(np.diff(padded_mask))` and returns **closed** intervals
  on the time vector, falling back to sample indices when `time is None`.
  A lone `NaN` gives a degenerate interval whose bounds are equal.
- `fill="interpolate"` uses `np.interp`, which conveniently holds the
  end values constant outside the valid range — exactly what is wanted
  for leading/trailing `NaN`s, which have no neighbour on one side.
- `fill="drop"` deletes the samples. This breaks the uniform sampling,
  so **`Dataset.remove_nans` rejects it** with a `ValueError`: dropping
  samples of one signal and not of another would destroy the common time
  vector that defines a `Dataset`. Drop on the single `Signal`s before
  building the `Dataset`.
- `remove_nans` raises if *every* sample is `NaN`, and returns a copy
  (not `self`) when there is nothing to do.

On the `Dataset` side, `remove_nans(*names, fill=...)` goes through the
usual `_map` helper so the selection semantics match `detrend` and
friends, while `nan_intervals()` returns `{name: [(start, end), ...]}`.

**Shading.** `Signal._shade_nans(ax, color, alpha=0.2)` is a primitive
that `axvspan`s each interval, and `Signal._plot_standard` calls it by
default (`shade_nans=True`) using the color of the line it has just
drawn. Since every time-domain plot in the package — `Signal.plot`,
`Dataset.plot`, `plot_signals`, `plot_dataset`, `plot_compare`,
`ValidationSession.plot_simulations` — funnels through
`_plot_standard`, gaps are shaded everywhere for free. The legacy code
instead matched lines to signals by substring-searching the legend
labels, which was fragile; that is gone.

---

# Graphical time-interval picking

`scope._pick_time_interval(fig, tin, tout, title, verbosity)` is the
single implementation: it connects an `xlim_changed` callback on the
first axes, blocks on `plt.pause(0.1)` until the figure is closed, and
returns the last x-limits seen (clipped at 0), or the passed defaults if
the user never zoomed. It is `# pragma: no cover` — it needs a human.

Both `Dataset.trim` and `ValidationSession.trim` call it when **neither**
`tin` **nor** `tout` is given. `Dataset.trim`'s signature is

```python
trim(tin=None, tout=None, *names, shift_to_zero=True, verbosity=0, **kwargs)
```

`names`/`kwargs` only select and style what is displayed while picking.
Note `tin`/`tout` stay **positional** because `ValidationSession.trim`
calls `vs._Dataset.trim(tin_sel, tout_sel)`, and `shift_to_zero` became
keyword-only.

---

# Removed APIs

```python
scope()
scope_compare()
InteractiveScope
plot_multi()
```

Interactive behavior is now integrated through
`plot(..., with_scope=True)` and `plot_spectrum(..., with_scope=True)`.
There is **no** reference to `InteractiveScope` left anywhere in the
package.

---

# Known gaps vs. the legacy version

Everything from the legacy version has been ported except the items below,
which are **deliberately absent** and are not mentioned anywhere in the
docs.

| legacy feature                                | status |
| --------------------------------------------- | ------ |
| `change_axes_layout(fig, nrows, ncols)`       | **not ported, by decision.** It had zero call sites in the legacy tree, and it was lossy: it `remove()`d every existing axes — throwing their contents away — before adding an empty grid. The modern one-liner is `fig.clear(); axes = fig.subplots(nrows, ncols)`, or `fig.subplot_mosaic` for anything fancier. |
| `Dataset.excluded_signals`                    | dropped on purpose: the factories interpolate instead of excluding. |
| `validate_signals`, `validate_dataframe`, DataFrame construction | dropped on purpose: the factories validate. |
| `overlap=True`                                | replaced by tuple grouping, e.g. `ds.plot(("u1", "y1"))`. |
| `compare_datasets(kind=...)`                  | split into `plot_compare` / `plot_spectrum_compare` / `plot_coverage_compare`. |
| `plotxy(*signal_pairs)`, `layout`/`ax_height`/`ax_width`, line styling | **ported** — see *Figure geometry* above. |
| `Dataset.remove_NaNs`, `_nan_intervals`, NaN shading | **ported** — see *Missing data* above. |
| graphical `tin`/`tout` picking in `Dataset.trim` | **ported** — shared with `ValidationSession.trim` via `scope._pick_time_interval`. |
| `Dataset.fft(*signals)` per-signal selection  | **ported**, and `spectrum(*names, mode=...)` too. |

---

# Docs

Sphinx sources live in `docs/source`. Build with:

```bash
cd docs
python -m sphinx -b html source _build/html -E --keep-going
```

`docs/_build/` is untracked and must never be committed. The reference
pages are hand-written `autosummary` lists, so **any rename or new public
method must be mirrored** in `docs/source/reference_index/dataset.rst` or
`docs/source/reference_index/validation.rst`. `docs/source/api_index/api.rst`
is a plain `automodule` dump of all nine modules and needs no maintenance
beyond adding a new module.

---

# Tests

```bash
pytest tests -q -m "not open_tutorial"   # 420 tests (~11 s)
pytest tests -m "not plots"              # skip the plotting tests
ruff format ./src ./tests && ruff check ./src ./tests
mypy ./src/dymoval
```

> `tests/test_utils.py::Test_open_tutorial` launches VSCode and takes
> ~100 s; deselect it with `-m "not open_tutorial"`.

Files:

```text
tests/conftest.py         # Agg backend + Signal/Dataset/validation fixtures
tests/test_signal.py
tests/test_dataset.py     # construction, validation, harmonization
tests/test_dataset_ops.py # structure editing, processing, coverage
tests/test_scope.py       # synthetic click/key events
tests/test_plotting.py
tests/test_statistics.py  # compute_statistic, rsquared
tests/test_xcorrelation.py # XCorrelation, whiteness_level
tests/test_validation.py  # ValidationSession, validate_models
tests/test_utils.py
```

The fixtures `sine_dataset` and `ones_dataset` are ports of the legacy
`sine_dataframe` / `constant_ones_dataframe`, and `good_dataset`,
`good_signals` and `correlation_tensors` are ports of `good_dataframe`,
`good_signals_no_nans` and `correlation_tensors`. All the numerical
expectations of the legacy suite (the `low_pass_filter` reference values,
the Matlab-computed cross-correlations, the `compute_statistic` and
`rsquared` references) are reused verbatim.

`manual_tests/test_plots_manual.py` is the **only** interactive smoke
script; it walks every plot the package can draw, plus NaN shading and
the two graphical `trim` pickers (which no automated test can reach).
`test_manual.py` and `test_plain_python.py` were deleted: they still
spoke the legacy API (`validate_signals`, `overlap=True`, dict-signals)
and `norecursedirs = ["manual_tests"]` had been hiding the breakage from
pytest.

---

# Near-Term Roadmap

## Validation enhancements

- attach a scope to `plot_residuals` (stem plots need a dedicated scope)
- persist / reload a `ValidationSession`

## Spectrum enhancements

- peak detection / dominant frequency display
- phase at peak
- harmonic markers

## Future ideas

- Bode plots
- transfer-function estimation
- coherence analysis
- dataset compare with scope on the frequency axis
