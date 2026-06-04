from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Self

import numpy as np
import matplotlib.pyplot as plt

from mpl_measurements import InteractiveScope

from .signal import Signal
from .scope import DatasetScope


@dataclass
class Dataset:
    inputs: dict[str, Signal]
    outputs: dict[str, Signal]
    meta: dict[str, Any] | None = None

    # ====================================================
    # Initialization / validation
    # ====================================================
    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        all_signals = list(self.inputs.values()) + list(self.outputs.values())

        if not all_signals:
            raise ValueError("Dataset cannot be empty")

        ref_time = all_signals[0].time

        if ref_time is None:
            raise ValueError("Signals must have time defined")

        for sig in all_signals:
            if sig.time is None:
                raise ValueError(f"{sig.name}: missing time")

            if len(sig.time) != len(ref_time):
                raise ValueError("Time length mismatch")

            if not np.allclose(sig.time, ref_time):
                raise ValueError("Signals are not aligned")

    # ================================================
    # Constructor
    # ================================================
    @classmethod
    def from_dict(cls, data: dict[str, list[Signal]]) -> "Dataset":
        """
        Build Dataset from user-friendly dictionary.

        Example:
            {
                "inputs": [Signal, ...],
                "outputs": [Signal, ...]
            }
        """

        allowed = {"inputs", "outputs"}

        if not set(data).issubset(allowed):
            raise ValueError(f"Only keys {allowed} are allowed")

        inputs_list = data.get("inputs", [])
        outputs_list = data.get("outputs", [])

        # ---- validate types
        for sig in inputs_list + outputs_list:
            if not isinstance(sig, Signal):
                raise TypeError("All elements must be Signal instances")

        # ---- enforce unique names
        def to_dict(lst):
            result = {}
            for sig in lst:
                if sig.name in result:
                    raise ValueError(f"Duplicate signal name: {sig.name}")
                result[sig.name] = sig
            return result

        inputs = to_dict(inputs_list)
        outputs = to_dict(outputs_list)

        # ---- global uniqueness check
        overlap = set(inputs) & set(outputs)
        if overlap:
            raise ValueError(
                f"Duplicate names across inputs/outputs: {overlap}"
            )

        # ---- sampling period check
        all_signals = list(inputs.values()) + list(outputs.values())

        if not all_signals:
            raise ValueError("No signals provided")

        ref_dt = all_signals[0].get_sampling_period()

        for sig in all_signals[1:]:
            dt = sig.get_sampling_period()

            if not np.isclose(dt, ref_dt):
                raise ValueError(
                    f"Sampling period mismatch: {sig.name} has dt={dt}, expected {ref_dt}"
                )

        # ---- alignment check
        ref_time = all_signals[0].time

        for sig in all_signals:
            if not np.allclose(sig.time, ref_time):
                raise ValueError(f"{sig.name} is not aligned")

        return cls(inputs=inputs, outputs=outputs)

    @classmethod
    def from_signals(cls, inputs=None, outputs=None):
        return cls.from_dict(
            {
                "inputs": inputs or [],
                "outputs": outputs or [],
            }
        )

    # ====================================================
    # Access helpers
    # ====================================================
    def __getitem__(self, key: str) -> Signal:
        if key in self.inputs:
            return self.inputs[key]
        if key in self.outputs:
            return self.outputs[key]
        raise KeyError(f"Signal '{key}' not found")

    def all_signals(self) -> dict[str, Signal]:
        return {**self.inputs, **self.outputs}

    def input_names(self) -> list[str]:
        return list(self.inputs.keys())

    def output_names(self) -> list[str]:
        return list(self.outputs.keys())

    def time(self) -> np.ndarray:
        return next(iter(self.inputs.values())).time

    # ====================================================
    # Internal selection
    # ====================================================
    def _select_signals(self, names):
        all_signals = self.all_signals()

        if not names:
            return list(all_signals.values())

        selected = []
        for name in names:
            if name not in all_signals:
                raise KeyError(f"Signal '{name}' not found")
            selected.append(all_signals[name])

        return selected

    # ====================================================
    # Copy
    # ====================================================
    def copy(self) -> Self:
        return Dataset(
            inputs={k: v.copy() for k, v in self.inputs.items()},
            outputs={k: v.copy() for k, v in self.outputs.items()},
            meta=deepcopy(self.meta),
        )

    # ====================================================
    # Processing
    # ====================================================
    def detrend(self) -> Self:
        return Dataset(
            inputs={k: v.detrend() for k, v in self.inputs.items()},
            outputs={k: v.detrend() for k, v in self.outputs.items()},
            meta=deepcopy(self.meta),
        )

    def fft(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {name: sig.fft() for name, sig in self.all_signals().items()}

    def resample(self, new_time: np.ndarray) -> Self:
        return Dataset(
            inputs={k: v.resample(new_time) for k, v in self.inputs.items()},
            outputs={
                k: v.resample(new_time) for k, v in self.outputs.items()
            },
            meta=deepcopy(self.meta),
        )

    def align(self, other: "Dataset", how: str = "intersection"):
        t1 = self.time()
        t2 = other.time()

        if how == "intersection":
            t_min = max(t1[0], t2[0])
            t_max = min(t1[-1], t2[-1])
            new_time = t1[(t1 >= t_min) & (t1 <= t_max)]

        elif how == "union":
            new_time = np.unique(np.concatenate([t1, t2]))

        else:
            raise ValueError("Invalid align mode")

        return self.resample(new_time), other.resample(new_time)

    # ====================================================
    # Pipeline
    # ====================================================
    def pipe(self, func, *args, **kwargs) -> "Dataset":
        result = func(self, *args, **kwargs)
        if not isinstance(result, Dataset):
            raise TypeError("Pipeline must return Dataset")
        return result

    # ====================================================
    # Interactive
    # ====================================================
    def scope(self, *names: str) -> InteractiveScope:
        signals = self._select_signals(names)
        return InteractiveScope(signals)

    def scope_compare(
        self, other: "Dataset", *names: str
    ) -> InteractiveScope:
        ds1, ds2 = self.align(other)

        if not names:
            names = tuple(set(ds1.all_signals()) & set(ds2.all_signals()))

        signals = []

        for name in names:
            signals.append(ds1[name])
            signals.append(ds2[name])

        return InteractiveScope(signals)

    # ====================================================
    # Plotting
    # ====================================================
    def _normalize_groups(self, names):
        # Needed to be able to overlap plots when passing tuples
        if not names:
            return [(name,) for name in self.all_signals()]

        groups = []
        for item in names:
            if isinstance(item, str):
                groups.append((item,))
            elif isinstance(item, tuple):
                groups.append(item)
            else:
                raise TypeError("Arguments must be str or tuple[str,...]")

        return groups

    def _plot_standard(self, *names: str | tuple[str, ...]):

        # Prepare figure
        groups = self._normalize_groups(names)
        fig, axes = plt.subplots(len(groups), 1, sharex=True)

        if len(groups) == 1:
            axes = [axes]

        # ====================================================
        # Plot groups
        # ====================================================
        all_signals = self.all_signals()

        for ax, group in zip(axes, groups):
            plotted_signals = []

            use_default_colors = len(group) > 1  # ✅ key logic

            for name in group:
                if name not in all_signals:
                    raise KeyError(f"Signal '{name}' not found")

                sig = all_signals[name]

                if use_default_colors:
                    # ✅ let matplotlib handle color cycle
                    sig.plot(ax=ax)
                else:
                    # ✅ semantic coloring (only for single signal)
                    if name in self.outputs:
                        sig._plot_standard(ax=ax, color="green")
                    else:
                        sig._plot_standard(ax=ax)

                plotted_signals.append(sig)

            ax.legend([s.name for s in plotted_signals])
            ax.grid(True)

        return fig

    def _plot_scope(self, *names):
        # Figure + layout
        groups = self._normalize_groups(names)
        n_groups = len(groups)

        # fig = plt.figure(
        #     constrained_layout=True,
        #     figsize=(10, 2.5 * n_groups),  # ✅ adaptive height
        # )

        # panel_ratio = min(2.5, 1.5 + 0.3 * (n_groups - 1))

        # subfigs = fig.subfigures(
        #     1,
        #     2,
        #     width_ratios=[4, panel_ratio],  # ✅ adaptive width
        # )

        fig = plt.figure(constrained_layout=True, figsize=(10, 5))
        subfigs = fig.subfigures(1, 2, width_ratios=[3.8, 1.2])

        axes = subfigs[0].subplots(n_groups, 1, sharex=True)

        if n_groups == 1:
            axes = [axes]

        # ====================================================
        # Plot groups
        # ====================================================
        all_signals = self.all_signals()

        for ax, group in zip(axes, groups):
            use_default_colors = len(group) > 1

            for name in group:
                if name not in all_signals:
                    raise KeyError(f"Signal '{name}' not found")

                sig = all_signals[name]

                if use_default_colors:
                    sig._plot_standard(ax=ax)
                else:
                    if name in self.outputs:
                        sig._plot_standard(ax=ax, color="green")
                    else:
                        sig._plot_standard(ax=ax)

            ax.legend(group)
            ax.grid(True)

        # ====================================================
        # Panel + scope
        # ====================================================
        panel_ax = subfigs[1].add_subplot()
        panel_ax.set_anchor("N")  # ✅ better vertical alignment

        DatasetScope(fig, axes, panel_ax)

        return fig

    def plot(self, *names, with_scope: bool = True, **kwargs):
        if with_scope:
            return self._plot_scope(*names, **kwargs)
        return self._plot_standard(*names, **kwargs)
