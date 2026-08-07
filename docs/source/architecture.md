dymoval — Architecture overview

Goal
----
Provide a concise rationale for the library structure and the main design
choices so that future maintainers can reason about trade-offs.

High-level structure
--------------------
- Signal: numerical primitives (time-domain ops, FFT/spectrum, filtering) and
  primitive plotting helpers tied to one signal instance. Low-level, fast,
  and focused on array operations.

- Dataset: orchestration of aligned Signal objects. Responsible for grouping,
  resampling, dataset-wide plotting, and convenience methods that operate on
  multiple signals consistently (e.g., dataset_values, fft across a dataset,
  and nonparametric input/output frequency-response estimation).

- FrequencyResponse: a nonparametric model estimated from a Dataset. It stores
  MIMO response and noise spectra and evaluates the response at requested
  frequencies without extrapolating.

- ValidationSession / XCorrelation / statistics: model evaluation and
  statistical diagnostics. The numerical kernels use NumPy/SciPy;
  ValidationSession and XCorrelation also expose plotting methods.

- spectral: internal Blackman--Tukey estimation used by Dataset.spa. It is
  separate from the FrequencyResponse result object and its plotting API.

- _figure / scope / plotting: the presentation infrastructure. _figure owns
  generic figure construction, layout types, and geometry defaults; scope
  owns click-to-inspect interaction; plotting contains high-level functions
  that combine several Signals or Datasets.

Design decisions & trade-offs
-----------------------------
- Separation of concerns: numerical kernels are kept separate from figure
  construction and interaction. Domain objects deliberately provide
  convenient plot methods, so Signal, Dataset, FrequencyResponse,
  XCorrelation, and ValidationSession depend on the presentation layer at
  their API boundary. Their numerical methods do not depend on interactive
  scope state.

- Copy-on-transform API: manipulation methods return new Signal or Dataset
  instances and leave the caller unchanged. The dataclasses and their NumPy
  arrays are not deeply immutable, so callers should treat their fields as
  owned data rather than mutate them in place.

- Conservative public API: most internal helper functions and plotting scope
  classes are intentionally private. The __init__ exports only the stable
  surface: Signal, Dataset, validation primitives, plotting convenience
  functions and core statistics. This reduces the surface area for
  compatibility guarantees and eases future refactors.

- Performance vs. clarity: routines like FFT and cross-correlation favor
  predictable conventions (normalization, one-sided folding) over ambiguous
  choices. Tests document numerical expectations (Parseval consistency,
  one-sided amplitude/power semantics).

- Defensive checks: many runtime validations protect users from mis-using
  the API (shape mismatches, sampling-period checks, inconsistent
  bandwidth arguments). This increases robustness at the cost of verbosity.

Recommendations for maintainers
-------------------------------
- Keep numerical kernels dependency-light (NumPy/SciPy only) and richly
  tested; keep Matplotlib imports at domain plotting boundaries.
- Prefer making helpers private rather than exported — evolve public API in
  small, documented steps when strictly necessary.
- Keep generic figure construction in _figure and interactive behavior in
  scope. Do not add numerical algorithms to either module.
- Dataset and ValidationSession are orchestration-heavy. Extract cohesive
  algorithms or presentation infrastructure when adding features rather than
  splitting them into inheritance-based mixins solely to reduce file length.

Contact
-------
For questions about design rationale or proposed refactors, review the
commit history and tests that anchor numerical semantics (fft/spectrum and
validation tests are the most important references).