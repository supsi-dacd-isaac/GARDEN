"""Simple array standardization helpers for emulator training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import ClosedLoopWindowedArrays, WindowedArrays


@dataclass(frozen=True)
class StandardScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, axis: int | tuple[int, ...]) -> "StandardScaler":
        mean = np.nanmean(values, axis=axis, keepdims=False).astype(np.float32)
        scale = np.nanstd(values, axis=axis, keepdims=False).astype(np.float32)
        scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return (values * self.scale + self.mean).astype(np.float32)


@dataclass(frozen=True)
class WindowScalers:
    metadata: StandardScaler
    inputs: StandardScaler
    target: StandardScaler


def fit_window_scalers(windows: WindowedArrays) -> WindowScalers:
    return WindowScalers(
        metadata=StandardScaler.fit(windows.metadata, axis=0),
        inputs=StandardScaler.fit(windows.inputs, axis=(0, 1)),
        target=StandardScaler.fit(windows.targets, axis=(0, 1)),
    )


def transform_windows(windows: WindowedArrays, scalers: WindowScalers) -> WindowedArrays:
    return WindowedArrays(
        profile_ids=windows.profile_ids,
        start_indices=windows.start_indices,
        metadata=scalers.metadata.transform(windows.metadata),
        inputs=scalers.inputs.transform(windows.inputs),
        targets=scalers.target.transform(windows.targets),
        initial_temperature=scalers.target.transform(windows.initial_temperature),
    )


def transform_closed_loop_windows(
    windows: ClosedLoopWindowedArrays,
    scalers: WindowScalers,
) -> ClosedLoopWindowedArrays:
    target_mean = np.asarray(scalers.target.mean, dtype=np.float32).reshape(-1)
    target_scale = np.asarray(scalers.target.scale, dtype=np.float32).reshape(-1)
    initial_temperature = (
        (windows.initial_temperature - target_mean[:1]) / target_scale[:1]
    ).astype(np.float32)
    return ClosedLoopWindowedArrays(
        profile_ids=windows.profile_ids,
        start_indices=windows.start_indices,
        metadata=scalers.metadata.transform(windows.metadata),
        inputs=scalers.inputs.transform(windows.inputs),
        targets=scalers.target.transform(windows.targets),
        initial_temperature=initial_temperature,
    )


def inverse_target(values: np.ndarray, scalers: WindowScalers) -> np.ndarray:
    return scalers.target.inverse_transform(values)
