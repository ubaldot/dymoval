"""The :class:`Dataset` class.

A ``Dataset`` is a set of *aligned* :class:`dymoval.signal.Signal`, split
into inputs and outputs.

``Dataset`` is responsible for orchestration, grouping and layout. All the
actual computation and the primitive plotting are delegated to ``Signal``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Self, Sequence, get_args

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .scope import (
    AmplitudeSpectrumScope,
    DatasetScope,
    SpectrumScope,
    scope_subplots,
)
from .signal import Signal, SpectrumMode, _check_mode

__all__ = ["Dataset", "SIGNAL_KIND"]

#: color used when a single *output* signal is plotted alone
_OUTPUT_COLOR = "green"
_INPUT_COLOR = "blue"

SignalKind = Literal["INPUT", "OUTPUT"]
SIGNAL_KIND: tuple[str, ...] = get_args(SignalKind)

Group = tuple[str, ...]


@dataclass
class Dataset:
    """Aligned input/output signals."""

    inputs: dict[str, Signal] = field(default_factory=dict)
    outputs: dict[str, Signal] = field(default_factory=dict)
    meta: dict[str, Any] | None = None

    # ====================================================
    # Initialization / validation
    # ====================================================
    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        all_signals = list(self.all_signals().values())

        if not all_signals:
            raise ValueError("Dataset cannot be empty")

        overlap = set(self.inputs) & set(self.outputs)
        if overlap:
            raise ValueError(
                f"Duplicate names across inputs/outputs: {sorted(overlap)}"
            )

        for name, sig in self.all_signals().items():
            if name != sig.name:
                raise ValueError(
                    f"Key '{name}' does not match signal name '{sig.name}'"
                )

        time_units = {sig.time_unit for sig in all_signals}
        if len(time_units) > 1:
            raise ValueError(
                f"All the signals must share the same time unit, "
                f"got {sorted(str(u) for u in time_units)}"
            )

        ref = all_signals[0]

        if ref.time is None:
            raise ValueError(f"{ref.name}: signals must have time defined")

        ref_dt = ref.get_sampling_period()

        for sig in all_signals[1:]:
            if sig.time is None:
                raise ValueError(f"{sig.name}: missing time")

            if len(sig.time) != len(ref.time):
                raise ValueError(f"{sig.name}: time length mismatch")

            dt = sig.get_sampling_period()

            if not np.isclose(dt, ref_dt):
                raise ValueError(
                    f"Sampling period mismatch: {sig.name} has dt={dt}, "
                    f"expected {ref_dt}"
                )

            if not np.allclose(sig.time, ref.time):
                raise ValueError(f"{sig.name} is not aligned")

    # ================================================
    # Harmonization
    # ================================================
    @staticmethod
    def _common_time(
        signals: Sequence[Signal], target_sampling_period: float | None
    ) -> np.ndarray | None:
        """Return the time vector every signal must share.

        ``None`` means "the signals already agree, leave them alone".
        The grid spans the time interval covered by **all** the signals,
        so that resampling never extrapolates.
        """
        for sig in signals:
            if sig.time is None:
                raise ValueError(f"{sig.name}: missing time")

            if len(sig.time) < 2:
                raise ValueError(f"{sig.name}: at least two samples required")

        if target_sampling_period is not None:
            if (
                isinstance(target_sampling_period, bool)
                or not isinstance(target_sampling_period, (int, float))
                or target_sampling_period <= 0
            ):
                raise ValueError(
                    "'target_sampling_period' must be a positive number"
                )

        periods = [sig.get_sampling_period() for sig in signals]
        ref_time = signals[0].time
        assert ref_time is not None

        already_aligned = all(
            sig.time is not None
            and len(sig.time) == len(ref_time)
            and np.allclose(sig.time, ref_time)
            for sig in signals
        )

        if already_aligned and (
            target_sampling_period is None
            or np.isclose(target_sampling_period, periods[0])
        ):
            return None

        # Downsample everybody to the slowest signal, unless the user
        # asked for a specific sampling period.
        dt = (
            float(np.max(periods))
            if target_sampling_period is None
            else float(target_sampling_period)
        )

        t_start = max(float(sig.time[0]) for sig in signals)  # type: ignore[index]
        t_end = min(float(sig.time[-1]) for sig in signals)  # type: ignore[index]

        n_samples = int(np.floor((t_end - t_start) / dt)) + 1

        if n_samples < 2:
            raise ValueError(
                "The signals do not share a long enough time interval "
                f"(from {t_start} to {t_end} with dt={dt})"
            )

        return t_start + np.arange(n_samples) * dt

    # ================================================
    # Constructors
    # ================================================
    @classmethod
    def from_dict(
        cls,
        data: dict[str, Sequence[Signal]],
        meta: dict[str, Any] | None = None,
        target_sampling_period: float | None = None,
    ) -> "Dataset":
        """Build a ``Dataset`` from ``{"inputs": [...], "outputs": [...]}``.

        Signals that do not already share a time vector are resampled on a
        common uniform grid. By default the grid uses the **largest**
        sampling period (the slowest signal) and spans the time interval
        covered by every signal, so that no extrapolation takes place.
        Pass ``target_sampling_period`` to choose the grid explicitly.
        """
        allowed = {"inputs", "outputs"}

        if not set(data).issubset(allowed):
            raise ValueError(f"Only keys {allowed} are allowed")

        def to_dict(signals: Iterable[Signal]) -> dict[str, Signal]:
            result: dict[str, Signal] = {}

            for sig in signals:
                if not isinstance(sig, Signal):
                    raise TypeError("All elements must be Signal instances")

                if sig.name in result:
                    raise ValueError(f"Duplicate signal name: {sig.name}")

                result[sig.name] = sig

            return result

        inputs = to_dict(data.get("inputs", []))
        outputs = to_dict(data.get("outputs", []))
        all_signals = [*inputs.values(), *outputs.values()]

        if not all_signals:
            raise ValueError("Dataset cannot be empty")

        new_time = cls._common_time(all_signals, target_sampling_period)

        if new_time is not None:
            inputs = {k: v.resample(new_time) for k, v in inputs.items()}
            outputs = {k: v.resample(new_time) for k, v in outputs.items()}

        return cls(inputs=inputs, outputs=outputs, meta=meta)

    @classmethod
    def from_signals(
        cls,
        inputs: Sequence[Signal] | None = None,
        outputs: Sequence[Signal] | None = None,
        meta: dict[str, Any] | None = None,
        target_sampling_period: float | None = None,
    ) -> "Dataset":
        return cls.from_dict(
            {"inputs": list(inputs or []), "outputs": list(outputs or [])},
            meta=meta,
            target_sampling_period=target_sampling_period,
        )

    # ================================================
    # Access helpers
    # ================================================
    def __getitem__(self, key: str) -> Signal:
        try:
            return self.all_signals()[key]
        except KeyError:
            raise KeyError(f"Signal '{key}' not found") from None

    def __contains__(self, key: object) -> bool:
        return key in self.all_signals()

    def __len__(self) -> int:
        return len(self.inputs) + len(self.outputs)

    def all_signals(self) -> dict[str, Signal]:
        return {**self.inputs, **self.outputs}

    def names(self) -> list[str]:
        return list(self.all_signals())

    def input_names(self) -> list[str]:
        return list(self.inputs)

    def output_names(self) -> list[str]:
        return list(self.outputs)

    def time(self) -> np.ndarray:
        time = next(iter(self.all_signals().values())).time
        assert time is not None  # guaranteed by _validate()
        return time

    def get_sampling_period(self) -> float:
        return next(iter(self.all_signals().values())).get_sampling_period()

    def time_unit(self) -> str | None:
        return next(iter(self.all_signals().values())).time_unit

    def kind_of(self, name: str) -> SignalKind:
        """Return whether ``name`` is an ``"INPUT"`` or an ``"OUTPUT"``."""
        if name in self.inputs:
            return "INPUT"

        if name in self.outputs:
            return "OUTPUT"

        raise KeyError(f"Signal '{name}' not found")

    def signal_list(self) -> list[tuple[SignalKind, str, str]]:
        """Return ``[(kind, name, unit), ...]``, inputs first."""
        groups: list[tuple[SignalKind, dict[str, Signal]]] = [
            ("INPUT", self.inputs),
            ("OUTPUT", self.outputs),
        ]

        return [
            (kind, sig.name, sig.unit or "")
            for kind, signals in groups
            for sig in signals.values()
        ]

    def to_signals(self) -> dict[SignalKind, list[Signal]]:
        """Return ``{"INPUT": [...], "OUTPUT": [...]}`` with copies."""
        return {
            "INPUT": [sig.copy() for sig in self.inputs.values()],
            "OUTPUT": [sig.copy() for sig in self.outputs.values()],
        }

    def dataset_values(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(time, inputs, outputs)`` as numpy arrays.

        ``inputs`` and ``outputs`` are 2-D ``(n_samples, n_signals)``,
        except when the dataset holds a single input (resp. output), in
        which case the corresponding array is 1-D.
        """

        def stack(signals: dict[str, Signal]) -> np.ndarray:
            if not signals:
                return np.empty((len(self.time()), 0))

            values = [sig.values for sig in signals.values()]

            return values[0] if len(values) == 1 else np.column_stack(values)

        return self.time(), stack(self.inputs), stack(self.outputs)

    def __repr__(self) -> str:
        lines = [
            f"Dataset: {len(self.inputs)} input(s), "
            f"{len(self.outputs)} output(s), "
            f"{len(self.time())} samples, "
            f"dt = {self.get_sampling_period():.6g} {self.time_unit()}"
        ]

        for kind, name, unit in self.signal_list():
            lines.append(f"  {kind:<6} {name} [{unit}]")

        return "\n".join(lines)

    # ================================================
    # Internal helpers
    # ================================================
    def _select_signals(self, names: Sequence[str]) -> list[Signal]:
        all_signals = self.all_signals()

        if not names:
            return list(all_signals.values())

        self._check_names(names)

        return [all_signals[name] for name in names]

    def _check_names(self, names: Sequence[str]) -> None:
        not_found = [name for name in names if name not in self.all_signals()]

        if not_found:
            raise KeyError(
                f"Signal(s) {sorted(not_found)} not found. "
                f"Available: {self.names()}"
            )

    def _map(
        self,
        func: Callable[[Signal], Signal],
        names: Sequence[str] | None = None,
    ) -> Self:
        """Apply ``func`` to the selected signals, return a new ``Dataset``.

        ``names=None`` (or empty) applies ``func`` to every signal.
        """
        if names:
            self._check_names(names)
            selected = set(names)
        else:
            selected = set(self.all_signals())

        def apply(sig: Signal) -> Signal:
            return func(sig) if sig.name in selected else sig.copy()

        return type(self)(
            inputs={k: apply(v) for k, v in self.inputs.items()},
            outputs={k: apply(v) for k, v in self.outputs.items()},
            meta=deepcopy(self.meta),
        )

    def _pairs(
        self, pairs: Sequence[tuple[str, Any]], what: str
    ) -> dict[str, Any]:
        """Validate and flatten ``(name, value)`` argument tuples."""
        result: dict[str, Any] = {}

        for item in pairs:
            if not isinstance(item, tuple) or len(item) < 2:
                raise TypeError(
                    f"Arguments must be ('signal_name', {what}) tuples"
                )

            result[item[0]] = item[1:] if len(item) > 2 else item[1]

        self._check_names(list(result))

        return result

    # ================================================
    # Structure
    # ================================================
    def _add_signals(self, kind: SignalKind, *signals: Signal) -> Self:
        if not signals:
            return self.copy()

        existing = set(self.all_signals())
        seen: set[str] = set()

        for sig in signals:
            if not isinstance(sig, Signal):
                raise TypeError("All elements must be Signal instances")

            if sig.name in existing or sig.name in seen:
                raise KeyError(f"Signal '{sig.name}' already exists")

            seen.add(sig.name)

        new_time = self.time()
        added = {sig.name: sig.resample(new_time) for sig in signals}

        inputs = {k: v.copy() for k, v in self.inputs.items()}
        outputs = {k: v.copy() for k, v in self.outputs.items()}

        if kind == "INPUT":
            inputs.update(added)
        else:
            outputs.update(added)

        return type(self)(
            inputs=inputs, outputs=outputs, meta=deepcopy(self.meta)
        )

    def add_input(self, *signals: Signal) -> Self:
        """Return a new ``Dataset`` with extra input signals.

        The added signals are resampled on the dataset time vector.
        """
        return self._add_signals("INPUT", *signals)

    def add_output(self, *signals: Signal) -> Self:
        """Return a new ``Dataset`` with extra output signals."""
        return self._add_signals("OUTPUT", *signals)

    def remove_signals(self, *names: str) -> Self:
        """Return a new ``Dataset`` without the given signals."""
        self._check_names(names)

        removed = set(names)

        if removed >= set(self.all_signals()):
            raise ValueError(
                "Cannot remove every signal: the dataset would be empty"
            )

        return type(self)(
            inputs={
                k: v.copy() for k, v in self.inputs.items() if k not in removed
            },
            outputs={
                k: v.copy()
                for k, v in self.outputs.items()
                if k not in removed
            },
            meta=deepcopy(self.meta),
        )

    # ================================================
    # Copy / processing
    # ================================================
    def copy(self) -> Self:
        return self._map(lambda sig: sig.copy())

    def detrend(self, *names: str) -> Self:
        return self._map(lambda sig: sig.detrend(), names)

    def remove_mean(self, *names: str) -> Self:
        """Subtract the mean of the selected signals (all by default)."""
        return self._map(lambda sig: sig.remove_mean(), names)

    def remove_constant(self, value: float | dict[str, float]) -> Self:
        """Subtract a constant from every signal.

        ``value`` is either a scalar applied to all the signals, or a
        ``{signal_name: constant}`` mapping. Signals missing from the
        mapping are left untouched.
        """
        if isinstance(value, dict):
            self._check_names(list(value))

            return self._map(
                lambda sig: sig.remove_constant(value.get(sig.name, 0.0))
            )

        return self._map(lambda sig: sig.remove_constant(value))

    def apply(self, *signal_function_unit: tuple[Any, ...]) -> Self:
        """Apply a function to the given signals.

        Each argument is a ``(name, func)`` or ``(name, func, new_unit)``
        tuple::

            ds.apply(("u1", np.square, "V^2"), ("y1", lambda x: 2 * x))
        """
        spec = self._pairs(signal_function_unit, "func[, unit]")

        def transform(sig: Signal) -> Signal:
            item = spec[sig.name]

            if isinstance(item, tuple):
                func, unit = item[0], item[1]
            else:
                func, unit = item, None

            return sig.apply(func, unit)

        return self._map(transform, list(spec))

    def low_pass_filter(self, *signals_cutoffs: tuple[str, float]) -> Self:
        """Low-pass filter the given signals.

        Each argument is a ``(name, cutoff_hz)`` tuple::

            ds.low_pass_filter(("u1", 10.0), ("y1", 2.5))
        """
        cutoffs = self._pairs(signals_cutoffs, "cutoff")

        return self._map(
            lambda sig: sig.low_pass_filter(float(cutoffs[sig.name])),
            list(cutoffs),
        )

    def trim(
        self,
        tin: float | None = None,
        tout: float | None = None,
        shift_to_zero: bool = True,
    ) -> Self:
        """Keep the samples with ``tin <= time <= tout``.

        ``None`` means "from the beginning" / "until the end". When
        ``shift_to_zero`` is set the resulting time vector starts at 0.
        """
        return self._map(
            lambda sig: sig.trim(tin, tout, shift_to_zero=shift_to_zero)
        )

    def resample(self, new_time: np.ndarray) -> Self:
        return self._map(lambda sig: sig.resample(new_time))

    def align(
        self, other: "Dataset", how: str = "intersection"
    ) -> tuple[Self, "Dataset"]:
        """Resample ``self`` and ``other`` on a common time vector.

        The resulting time vector is always uniformly sampled with the
        sampling period of ``self``, so that both datasets stay valid.
        """
        if how not in ("intersection", "union"):
            raise ValueError(
                f"Invalid align mode: {how!r}. "
                "Allowed: ['intersection', 'union']"
            )

        t1 = self.time()
        t2 = other.time()
        dt = self.get_sampling_period()

        if how == "intersection":
            t_start = max(t1[0], t2[0])
            t_end = min(t1[-1], t2[-1])

            if t_end < t_start:
                raise ValueError("Datasets do not overlap in time")
        else:
            t_start = min(t1[0], t2[0])
            t_end = max(t1[-1], t2[-1])

        n_samples = int(np.floor((t_end - t_start) / dt)) + 1

        if n_samples < 2:
            raise ValueError("Datasets do not overlap in time")

        new_time = t_start + np.arange(n_samples) * dt

        return self.resample(new_time), other.resample(new_time)

    # ================================================
    # Frequency domain
    # ================================================
    def fft(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return ``{name: (freq, complex spectrum)}``."""
        return {name: sig.fft() for name, sig in self.all_signals().items()}

    def spectrum(
        self, mode: SpectrumMode = "psd_welch"
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return ``{name: (freq, spectrum)}`` for the requested mode."""
        return {
            name: sig.spectrum(mode)
            for name, sig in self.all_signals().items()
        }

    # ================================================
    # Pipeline
    # ================================================
    def pipe(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        result = func(self, *args, **kwargs)

        if not isinstance(result, Dataset):
            raise TypeError("Pipeline must return Dataset")

        return result

    # ================================================
    # Grouping
    # ================================================
    def _normalize_groups(
        self, names: Sequence[str | tuple[str, ...]]
    ) -> list[Group]:
        """Turn the ``*names`` varargs into a list of groups.

        Signals belonging to the same group are overlaid on one subplot.
        """
        if not names:
            groups: list[Group] = [(name,) for name in self.all_signals()]
        else:
            groups = []

            for item in names:
                if isinstance(item, str):
                    groups.append((item,))
                elif isinstance(item, (tuple, list)):
                    groups.append(tuple(item))
                else:
                    raise TypeError("Arguments must be str or tuple[str, ...]")

        for group in groups:
            if not group:
                raise ValueError("Empty group")

            self._check_names(group)

        return groups

    def _plot_group(self, ax: Axes, group: Group) -> None:
        """Draw one group of signals on ``ax``."""
        all_signals = self.all_signals()

        # semantic coloring only makes sense when a single signal is drawn
        use_default_colors = len(group) > 1

        for name in group:
            sig = all_signals[name]

            if not use_default_colors and name in self.outputs:
                sig._plot_standard(ax=ax, color=_OUTPUT_COLOR)
            else:
                sig._plot_standard(ax=ax)

        ax.legend()
        ax.grid(True)

    # ================================================
    # Time-domain plotting
    # ================================================
    def _plot(self, *names: str | tuple[str, ...], with_scope: bool) -> Figure:
        groups = self._normalize_groups(names)

        fig, axes, panel_ax = scope_subplots(
            len(groups),
            with_scope=with_scope,
            figsize=(10, 2.0 * len(groups) + 1),
            sharex=True,
        )

        for ax, group in zip(axes, groups):
            self._plot_group(ax, group)

        if panel_ax is not None:
            DatasetScope(fig, axes, panel_ax)

        return fig

    def plot(
        self,
        *names: str | tuple[str, ...],
        with_scope: bool = True,
    ) -> Figure:
        """Plot the signals, one subplot per group."""
        return self._plot(*names, with_scope=with_scope)

    # ================================================
    # x/y plotting
    # ================================================
    def plot_xy(
        self,
        x_name: str,
        y_name: str,
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot ``y_name`` against ``x_name``."""
        x_sig = self[x_name]
        y_sig = self[y_name]

        if ax is None:
            _, ax = plt.subplots()

        kwargs.setdefault("label", f"{y_name} vs {x_name}")

        ax.plot(x_sig.values, y_sig.values, **kwargs)

        ax.set_xlabel(x_sig._ylabel())
        ax.set_ylabel(y_sig._ylabel())
        ax.grid(True)
        ax.legend()

        return ax

    # ================================================
    # Coverage
    # ================================================
    @staticmethod
    def _stats(
        signals: dict[str, Signal],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not signals:
            return np.empty(0), np.empty((0, 0))

        values = np.column_stack([sig.values for sig in signals.values()])

        mean = np.nanmean(values, axis=0)
        cov = np.atleast_2d(np.cov(values, rowvar=False))

        return mean, cov

    def coverage(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(u_mean, u_cov, y_mean, y_cov)``.

        Means are 1-D arrays of length ``n_signals``; covariances are the
        corresponding ``(n_signals, n_signals)`` matrices.
        """
        u_mean, u_cov = self._stats(self.inputs)
        y_mean, y_cov = self._stats(self.outputs)

        return u_mean, u_cov, y_mean, y_cov

    def plot_coverage(
        self,
        *names: str,
        nbins: int = 100,
        color_input: str = _INPUT_COLOR,
        color_output: str = _OUTPUT_COLOR,
        alpha: float = 1.0,
        histtype: Literal["bar", "barstacked", "step", "stepfilled"] = "bar",
    ) -> Figure:
        """Plot the histogram of the signal values, one subplot each.

        Coverage plots cannot be overlapped, hence groups (tuples) are
        not accepted here.
        """
        overlapped = [name for name in names if not isinstance(name, str)]

        if overlapped:
            raise TypeError(
                f"It seems that you are trying to overlap {overlapped}. "
                "Coverage plots cannot be overlapped."
            )

        selected = list(names) if names else self.names()
        self._check_names(selected)

        fig, axes, _ = scope_subplots(
            len(selected),
            with_scope=False,
            figsize=(7, 1.8 * len(selected) + 1),
        )

        for ax, name in zip(axes, selected):
            sig = self[name]
            is_input = name in self.inputs

            ax.hist(
                sig.values,
                bins=nbins,
                color=color_input if is_input else color_output,
                alpha=alpha,
                histtype=histtype,
                label=name,
            )

            ax.set_xlabel(sig._ylabel())
            ax.set_ylabel("count")
            ax.grid(True)
            ax.legend()

        return fig

    # ================================================
    # Frequency-domain plotting
    # ================================================
    def _plot_spectrum_group(
        self,
        ax: Axes,
        group: Group,
        xscale: str,
        yscale: str,
        mode: SpectrumMode,
    ) -> None:
        all_signals = self.all_signals()

        for name in group:
            all_signals[name]._plot_spectrum_standard(
                ax=ax, xscale=xscale, yscale=yscale, mode=mode
            )

        ax.legend()
        ax.grid(True)

    def _plot_spectrum_amplitude_group(
        self,
        mag_ax: Axes,
        phase_ax: Axes,
        group: Group,
        xscale: str,
        yscale: str,
    ) -> None:
        all_signals = self.all_signals()

        for name in group:
            all_signals[name]._plot_spectrum_amplitude(
                mag_ax=mag_ax,
                phase_ax=phase_ax,
                xscale=xscale,
                yscale=yscale,
            )

        mag_ax.legend()
        mag_ax.grid(True)
        phase_ax.grid(True)

    # ------------------------------------------------
    # amplitude (magnitude + phase) layouts
    # ------------------------------------------------
    def _plot_spectrum_amplitude(
        self,
        *names: str | tuple[str, ...],
        xscale: str = "linear",
        yscale: str = "linear",
        with_scope: bool = False,
    ) -> Figure:
        groups = self._normalize_groups(names)

        fig, axes, panel_ax = scope_subplots(
            2 * len(groups),
            with_scope=with_scope,
            figsize=(10, 2.0 * len(groups) + 2),
            sharex=True,
        )

        # One scope per group so that magnitude and phase are linked and
        # the cursor stays local to the group. All the scopes share the
        # same panel: the last clicked one wins.
        for i, group in enumerate(groups):
            mag_ax, phase_ax = axes[2 * i], axes[2 * i + 1]

            self._plot_spectrum_amplitude_group(
                mag_ax=mag_ax,
                phase_ax=phase_ax,
                group=group,
                xscale=xscale,
                yscale=yscale,
            )

            if panel_ax is not None:
                AmplitudeSpectrumScope(fig, mag_ax, phase_ax, panel_ax)

        return fig

    # ------------------------------------------------
    # magnitude-only layouts
    # ------------------------------------------------
    def _plot_spectrum(
        self,
        *names: str | tuple[str, ...],
        with_scope: bool,
        xscale: str = "linear",
        yscale: str = "linear",
        mode: SpectrumMode = "psd_welch",
    ) -> Figure:
        if mode == "amplitude":
            return self._plot_spectrum_amplitude(
                *names,
                xscale=xscale,
                yscale=yscale,
                with_scope=with_scope,
            )

        groups = self._normalize_groups(names)

        fig, axes, panel_ax = scope_subplots(
            len(groups),
            with_scope=with_scope,
            figsize=(10, 2.0 * len(groups) + 1),
            sharex=True,
        )

        for ax, group in zip(axes, groups):
            self._plot_spectrum_group(ax, group, xscale, yscale, mode)

        if panel_ax is not None:
            SpectrumScope(fig, axes, panel_ax)

        return fig

    def plot_spectrum(
        self,
        *names: str | tuple[str, ...],
        with_scope: bool = True,
        xscale: str = "linear",
        yscale: str = "linear",
        mode: SpectrumMode = "psd_welch",
    ) -> Figure:
        """Plot the spectra of the signals, one subplot per group.

        With ``mode="amplitude"`` each group gets a magnitude *and* a
        phase subplot.
        """
        _check_mode(mode)

        return self._plot_spectrum(
            *names,
            with_scope=with_scope,
            xscale=xscale,
            yscale=yscale,
            mode=mode,
        )
