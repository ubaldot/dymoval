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

and **never** calls the public plotting methods internally.

---

# Module map

| module        | status | content                                             |
| ------------- | ------ | --------------------------------------------------- |
| `signal.py`   | new    | `Signal`                                             |
| `dataset.py`  | new    | `Dataset`                                            |
| `scope.py`    | new    | `BaseScope` / `SignalScope` / `DatasetScope` / `SpectrumScope` / `AmplitudeSpectrumScope` |
| `plotting.py` | new    | multi-dataset helpers only                           |
| `utils.py`, `config.py` | kept | unchanged                                  |
| `validation.py` | legacy | still pandas + `mpl_measurements`                  |
| `dataset_old.py` | legacy | pandas implementation, kept for reference         |

The legacy pandas test-suite lives in `tests/legacy/` and is excluded from
collection (`norecursedirs` in `pyproject.toml`).

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
to_signals()               # flat list[Signal]
dataset_values()           # (time, U, Y) as plain ndarrays
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
plot(*groups, with_scope=True)
plot_xy(x_name, y_name, ax=None)
plot_spectrum(*groups, with_scope=True, xscale, yscale, mode)
```

`SIGNAL_KIND = ("INPUT", "OUTPUT")` is exported by the package.

`align()` always returns a **uniform** time vector built with the sampling
period of `self`, so the resulting datasets stay valid.

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

# Multi-dataset plotting (`plotting.py`)

```python
plot_dataset(ds, *groups, with_scope=True)          # alias of ds.plot
plot_compare(ref, *others, names, labels, align, with_scope)
plot_spectrum_compare(ref, *others, ..., mode)      # mode != "amplitude"
```

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
The new core has **no** reference to `InteractiveScope`; the remaining
occurrences live only in the legacy `validation.py` / `dataset_old.py`.

---

# Tests

```bash
pytest tests -q                     # 232 tests
pytest tests -m "not plots"         # skip the plotting tests
ruff format ./src ./tests && ruff check ./src ./tests
mypy ./src/dymoval/{signal,dataset,scope,plotting,__init__}.py
```

Files:

```text
tests/conftest.py         # Agg backend + Signal/Dataset fixtures
tests/test_signal.py
tests/test_dataset.py     # construction, validation, harmonization
tests/test_dataset_ops.py # structure editing, processing, coverage
tests/test_scope.py       # synthetic click/key events
tests/test_plotting.py
tests/test_utils.py
tests/legacy/             # pandas-era suite, not collected
```

The fixtures `sine_dataset` and `ones_dataset` are ports of the legacy
`sine_dataframe` / `constant_ones_dataframe`, so the numerical
expectations of the legacy suite (notably the `low_pass_filter` reference
values) are reused verbatim.

`manual_tests/test_plots_manual.py` is the interactive smoke script.

---

# Near-Term Roadmap

## Port `validation.py`

The last pandas consumer. It also depends on `mpl_measurements`
(`InteractiveScope`) which must be replaced by the new scopes. Once done:

- delete `src/dymoval/dataset_old.py` and `tests/legacy/`
- drop `pandas` and `mpl-measurements` from `pyproject.toml`
- refresh `src/dymoval_tutorial/` (still pandas-based)

## Spectrum enhancements

- peak detection / dominant frequency display
- phase at peak
- harmonic markers

## Future ideas

- Bode plots
- transfer-function estimation
- coherence analysis
- dataset compare with scope on the frequency axis
