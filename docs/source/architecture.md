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
  multiple signals consistently (e.g., dataset_values, fft across a dataset).

- ValidationSession / XCorrelation / statistics: model evaluation primitives
  and statistical diagnostics. They depend only on numpy/scipy and are
  designed to be testable without plotting or interactive dependencies.

- scope & plotting: interactive matplotlib layer kept separate from numeric
  code. Scope implements figure geometry and click-to-inspect features. The
  plotting module contains high-level convenience functions that combine
  Signals/Datasets into multi-axes figures.

Design decisions & trade-offs
-----------------------------
- Separation of concerns: numeric code (Signal, Dataset, statistics) is
  independent from interactive plotting (scope, plotting). This reduces the
  risk of UI-side changes breaking numerical behaviour and simplifies testing.

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
- Keep numeric code dependency-light (numpy/scipy only) and richly tested.
- Prefer making helpers private rather than exported — evolve public API in
  small, documented steps when strictly necessary.
- When simplifying plotting, keep scope separate and small; collapse only
  helpers that are genuinely single-use.

Contact
-------
For questions about design rationale or proposed refactors, review the
commit history and tests that anchor numerical semantics (fft/spectrum and
validation tests are the most important references).