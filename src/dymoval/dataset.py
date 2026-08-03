"""The :class:`Dataset` class.

A ``Dataset`` is a set of *aligned* :class:`dymoval.signal.Signal`, split
into inputs and outputs.

``Dataset`` is responsible for orchestration, grouping and layout. All the
actual computation and the primitive plotting are delegated to ``Signal``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Self, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .scope import AmplitudeSpectrumScope, DatasetScope, SpectrumScope
from .signal import SPECTRUM_MODES, Signal, SpectrumMode

__all__ = ["Dataset"]

#: color used when a single *output* signal is plotted alone
_OUTPUT_COLOR = "green"

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
    # Constructors
    # ================================================
    @classmethod
    def from_dict(
        cls,
        data: dict[str, Sequence[Signal]],
        meta: dict[str, Any] | None = None,
    ) -> "Dataset":
        """Build a ``Dataset`` from ``{"inputs": [...], "outputs": [...]}``."""
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

        return cls(
            inputs=to_dict(data.get("inputs", [])),
            outputs=to_dict(data.get("outputs", [])),
            meta=meta,
        )

    @classmethod
    def from_signals(
        cls,
        inputs: Sequence[Signal] | None = None,
        outputs: Sequence[Signal] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> "Dataset":
        return cls.from_dict(
            {"inputs": list(inputs or []), "outputs": list(outputs or [])},
            meta=meta,
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

    # ================================================
    # Internal helpers
    # ================================================
    def _select_signals(self, names: Sequence[str]) -> list[Signal]:
        all_signals = self.all_signals()

        if not names:
            return list(all_signals.values())

        for name in names:
            if name not in all_signals:
                raise KeyError(f"Signal '{name}' not found")

        return [all_signals[name] for name in names]

    def _map(self, func: Callable[[Signal], Signal]) -> Self:
        """Apply ``func`` to every signal and return a new ``Dataset``."""
        return type(self)(
            inputs={k: func(v) for k, v in self.inputs.items()},
            outputs={k: func(v) for k, v in self.outputs.items()},
            meta=deepcopy(self.meta),
        )

    # ================================================
    # Copy / processing
    # ================================================
    def copy(self) -> Self:
        return self._map(lambda sig: sig.copy())

    def detrend(self) -> Self:
        return self._map(lambda sig: sig.detrend())

    def remove_mean(self) -> Self:
        """Subtract the mean of every signal."""
        return self._map(lambda sig: sig.remove_mean())

    def remove_constant(self, value: float | dict[str, float]) -> Self:
        """Subtract a constant from every signal.

        ``value`` is either a scalar applied to all the signals, or a
        ``{signal_name: constant}`` mapping. Signals missing from the
        mapping are left untouched.
        """
        if isinstance(value, dict):
            unknown = set(value) - set(self.all_signals())

            if unknown:
                raise KeyError(f"Unknown signals: {sorted(unknown)}")

            return self._map(
                lambda sig: sig.remove_constant(value.get(sig.name, 0.0))
            )

        return self._map(lambda sig: sig.remove_constant(value))

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

        all_signals = self.all_signals()

        for group in groups:
            if not group:
                raise ValueError("Empty group")

            for name in group:
                if name not in all_signals:
                    raise KeyError(f"Signal '{name}' not found")

        return groups

    @staticmethod
    def _make_axes(subfig: Any, nrows: int, **kwargs: Any) -> list[Axes]:
        axes = subfig.subplots(nrows, 1, **kwargs)
        return list(np.atleast_1d(axes).ravel())

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
    def _plot_standard(self, *names: str | tuple[str, ...]) -> Figure:
        groups = self._normalize_groups(names)

        fig = plt.figure(
            constrained_layout=True, figsize=(10, 2.0 * len(groups) + 1)
        )
        axes = self._make_axes(fig, len(groups), sharex=True)

        for ax, group in zip(axes, groups):
            self._plot_group(ax, group)

        return fig

    def _plot_scope(self, *names: str | tuple[str, ...]) -> Figure:
        groups = self._normalize_groups(names)

        fig = plt.figure(
            constrained_layout=True, figsize=(10, 2.0 * len(groups) + 1)
        )
        subfigs = fig.subfigures(1, 2, width_ratios=[3.8, 1.2])

        axes = self._make_axes(subfigs[0], len(groups), sharex=True)

        for ax, group in zip(axes, groups):
            self._plot_group(ax, group)

        panel_ax = subfigs[1].add_subplot()
        panel_ax.set_anchor("N")

        DatasetScope(fig, axes, panel_ax)

        return fig

    def plot(
        self,
        *names: str | tuple[str, ...],
        with_scope: bool = True,
    ) -> Figure:
        """Plot the signals, one subplot per group."""
        if with_scope:
            return self._plot_scope(*names)

        return self._plot_standard(*names)

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
    ) -> Figure:
        groups = self._normalize_groups(names)

        fig = plt.figure(
            constrained_layout=True, figsize=(10, 2.0 * len(groups) + 2)
        )
        axes = self._make_axes(fig, 2 * len(groups), sharex=True)

        for i, group in enumerate(groups):
            self._plot_spectrum_amplitude_group(
                mag_ax=axes[2 * i],
                phase_ax=axes[2 * i + 1],
                group=group,
                xscale=xscale,
                yscale=yscale,
            )

        return fig

    def _plot_spectrum_amplitude_scope(
        self,
        *names: str | tuple[str, ...],
        xscale: str = "linear",
        yscale: str = "linear",
    ) -> Figure:
        groups = self._normalize_groups(names)

        fig = plt.figure(
            constrained_layout=True, figsize=(10, 2.0 * len(groups) + 2)
        )
        subfigs = fig.subfigures(1, 2, width_ratios=[3.8, 1.2])

        axes = self._make_axes(subfigs[0], 2 * len(groups), sharex=True)

        panel_ax = subfigs[1].add_subplot()
        panel_ax.set_anchor("N")

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

            AmplitudeSpectrumScope(fig, mag_ax, phase_ax, panel_ax)

        return fig

    # ------------------------------------------------
    # magnitude-only layouts
    # ------------------------------------------------
    def _plot_spectrum_standard(
        self,
        *names: str | tuple[str, ...],
        xscale: str = "linear",
        yscale: str = "linear",
        mode: SpectrumMode = "psd_welch",
    ) -> Figure:
        if mode == "amplitude":
            return self._plot_spectrum_amplitude(
                *names, xscale=xscale, yscale=yscale
            )

        groups = self._normalize_groups(names)

        fig = plt.figure(
            constrained_layout=True, figsize=(10, 2.0 * len(groups) + 1)
        )
        axes = self._make_axes(fig, len(groups), sharex=True)

        for ax, group in zip(axes, groups):
            self._plot_spectrum_group(ax, group, xscale, yscale, mode)

        return fig

    def _plot_spectrum_scope(
        self,
        *names: str | tuple[str, ...],
        xscale: str = "linear",
        yscale: str = "linear",
        mode: SpectrumMode = "psd_welch",
    ) -> Figure:
        if mode == "amplitude":
            return self._plot_spectrum_amplitude_scope(
                *names, xscale=xscale, yscale=yscale
            )

        groups = self._normalize_groups(names)

        fig = plt.figure(
            constrained_layout=True, figsize=(10, 2.0 * len(groups) + 1)
        )
        subfigs = fig.subfigures(1, 2, width_ratios=[3.8, 1.2])

        axes = self._make_axes(subfigs[0], len(groups), sharex=True)

        for ax, group in zip(axes, groups):
            self._plot_spectrum_group(ax, group, xscale, yscale, mode)

        panel_ax = subfigs[1].add_subplot()
        panel_ax.set_anchor("N")

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
        if mode not in SPECTRUM_MODES:
            raise ValueError(
                f"Invalid mode: {mode!r}. Allowed: {list(SPECTRUM_MODES)}"
            )

        if with_scope:
            return self._plot_spectrum_scope(
                *names, xscale=xscale, yscale=yscale, mode=mode
            )

        return self._plot_spectrum_standard(
            *names, xscale=xscale, yscale=yscale, mode=mode
        )
