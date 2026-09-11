"""Training orchestration for all models in the comparison registry."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from neural_building_emulator.columns import CLOSED_LOOP_TARGET_COLUMNS, TARGET_COLUMN
from neural_building_emulator.data import (
    ClosedLoopDatasetSplits,
    ClosedLoopProfile,
    ClosedLoopWindowedArrays,
    BuildingDatasetSplits,
    BuildingProfile,
    SplitConfig,
    WindowConfig,
    WindowedArrays,
    load_closed_loop_result_splits,
    load_result_splits,
    make_closed_loop_windows,
    make_windows,
    to_closed_loop_profiles,
    to_profiles,
)
from neural_building_emulator.scaling import (
    WindowScalers,
    fit_window_scalers,
    transform_closed_loop_windows,
    transform_windows,
)

from .artifacts import save_lstm_artifact
from .config import ExperimentConfig
from .legacy import train_legacy
from .models import AutoregressiveLSTM
from .registry import ModelSpec, get_model_spec

Windows = WindowedArrays | ClosedLoopWindowedArrays
Profiles = list[BuildingProfile] | list[ClosedLoopProfile]
Splits = BuildingDatasetSplits | ClosedLoopDatasetSplits


def rotating_window_start_offset(epoch: int, epochs: int, stride: int) -> int:
    if epochs <= 1 or stride <= 1:
        return 0
    return int(round((epoch - 1) * (stride - 1) / (epochs - 1)))


def _subset_windows(windows: Windows, max_windows: int, seed: int) -> Windows:
    count = windows.targets.shape[0]
    if max_windows <= 0 or count <= max_windows:
        return windows
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(count, size=max_windows, replace=False))
    return replace(
        windows,
        profile_ids=windows.profile_ids[indices],
        start_indices=windows.start_indices[indices],
        metadata=windows.metadata[indices],
        inputs=windows.inputs[indices],
        targets=windows.targets[indices],
        initial_temperature=windows.initial_temperature[indices],
    )


def _batches(
    windows: Windows,
    *,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    indices = np.arange(windows.targets.shape[0])
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        yield (
            windows.metadata[selected],
            windows.inputs[selected],
            windows.initial_temperature[selected],
            windows.targets[selected],
        )


def _prepare_data(
    config: ExperimentConfig,
    spec: ModelSpec,
) -> tuple[Splits, Profiles, Profiles, Windows, Windows, WindowScalers]:
    split_config = SplitConfig(
        dataset_path=Path(config.dataset_path),
        max_profiles=config.max_profiles,
        test_fraction=config.test_fraction,
        seed=config.seed,
        heat_input_normalization=config.heat_input_normalization,  # type: ignore[arg-type]
        hp_power_area_normalization=config.hp_power_area_normalization,  # type: ignore[arg-type]
    )
    window_config = WindowConfig(
        sequence_length=config.sequence_length,
        stride=config.stride,
        target_alignment="next_step" if spec.task == "q_to_t" else "same_time",
    )
    if spec.task == "q_to_t":
        splits = load_result_splits(split_config, heating_mode=config.heating_mode)
        train_profiles = to_profiles(
            splits.train,
            splits.heating_mode,
            splits.heat_input_normalization,
            splits.input_feature_mode,
            splits.heating_regime_window_steps,
            hp_power_area_normalization=splits.hp_power_area_normalization,
        )
        test_profiles = to_profiles(
            splits.test,
            splits.heating_mode,
            splits.heat_input_normalization,
            splits.input_feature_mode,
            splits.heating_regime_window_steps,
            hp_power_area_normalization=splits.hp_power_area_normalization,
        )
        raw_train = make_windows(train_profiles, window_config)
        raw_test = make_windows(test_profiles, window_config) if test_profiles else raw_train
        scalers = fit_window_scalers(raw_train)
        train_windows = transform_windows(raw_train, scalers)
        test_windows = transform_windows(raw_test, scalers)
    else:
        splits = load_closed_loop_result_splits(split_config)
        train_profiles = to_closed_loop_profiles(
            splits.train,
            hp_power_area_normalization=splits.hp_power_area_normalization,
            metadata_columns=splits.metadata_columns,
        )
        test_profiles = to_closed_loop_profiles(
            splits.test,
            hp_power_area_normalization=splits.hp_power_area_normalization,
            metadata_columns=splits.metadata_columns,
        )
        raw_train = make_closed_loop_windows(train_profiles, window_config)
        raw_test = (
            make_closed_loop_windows(test_profiles, window_config)
            if test_profiles
            else raw_train
        )
        scalers = fit_window_scalers(raw_train)
        train_windows = transform_closed_loop_windows(raw_train, scalers)
        test_windows = transform_closed_loop_windows(raw_test, scalers)
    return splits, train_profiles, test_profiles, train_windows, test_windows, scalers


def _initial_output(
    initial_temperature: jnp.ndarray,
    *,
    output_dim: int,
    target_mean: jnp.ndarray,
    target_scale: jnp.ndarray,
) -> jnp.ndarray:
    if output_dim == 1:
        return initial_temperature
    normalized_zero_powers = -target_mean[1:] / target_scale[1:]
    repeated = jnp.broadcast_to(
        normalized_zero_powers,
        (initial_temperature.shape[0], output_dim - 1),
    )
    return jnp.concatenate([initial_temperature, repeated], axis=-1)


def _output_weights(config: ExperimentConfig, output_dim: int) -> jnp.ndarray:
    values = config.lstm.output_weights
    if output_dim == 1:
        values = values[:1]
    if len(values) != output_dim:
        raise ValueError(
            f"LSTM output_weights must contain {output_dim} values for this task, got {len(values)}"
        )
    if any(weight < 0.0 for weight in values) or sum(values) <= 0.0:
        raise ValueError("LSTM output weights must be non-negative with at least one positive value")
    weights = jnp.asarray(values, dtype=jnp.float32)
    return weights / jnp.sum(weights)


def _build_optimizer(config: ExperimentConfig) -> optax.GradientTransformation:
    transforms: list[optax.GradientTransformation] = []
    if config.optimizer.gradient_clip_norm > 0.0:
        transforms.append(optax.clip_by_global_norm(config.optimizer.gradient_clip_norm))
    transforms.append(optax.adam(config.optimizer.learning_rate))
    return optax.chain(*transforms)


def _make_steps(
    optimizer: optax.GradientTransformation,
    *,
    output_weights: jnp.ndarray,
    target_mean: jnp.ndarray,
    target_scale: jnp.ndarray,
):
    @eqx.filter_value_and_grad
    def loss_fn(model, metadata, inputs, initial_temperature, targets):
        initial = _initial_output(
            initial_temperature,
            output_dim=model.output_dim,
            target_mean=target_mean,
            target_scale=target_scale,
        )
        predictions = jax.vmap(model)(metadata, inputs, initial)
        channel_mse = jnp.mean((predictions - targets) ** 2, axis=(0, 1))
        return jnp.sum(output_weights * channel_mse)

    @eqx.filter_jit
    def train_step(model, opt_state, metadata, inputs, initial_temperature, targets):
        loss, grads = loss_fn(model, metadata, inputs, initial_temperature, targets)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    return train_step


def _evaluate(
    model: AutoregressiveLSTM,
    windows: Windows,
    scalers: WindowScalers,
    *,
    batch_size: int,
) -> dict[str, float]:
    target_mean = jnp.asarray(scalers.target.mean)
    target_scale = jnp.asarray(scalers.target.scale)
    sum_squared = np.zeros(model.output_dim, dtype=np.float64)
    sum_absolute = np.zeros(model.output_dim, dtype=np.float64)
    count = 0
    rng = np.random.default_rng(0)
    for metadata, inputs, initial_temperature, targets in _batches(
        windows,
        batch_size=batch_size,
        rng=rng,
        shuffle=False,
    ):
        initial = _initial_output(
            jnp.asarray(initial_temperature),
            output_dim=model.output_dim,
            target_mean=target_mean,
            target_scale=target_scale,
        )
        normalized = _predict_lstm_batch(
            model,
            jnp.asarray(metadata),
            jnp.asarray(inputs),
            initial,
        )
        prediction = np.asarray(normalized) * scalers.target.scale + scalers.target.mean
        truth = targets * scalers.target.scale + scalers.target.mean
        error = prediction - truth
        sum_squared += np.sum(error.astype(np.float64) ** 2, axis=(0, 1))
        sum_absolute += np.sum(np.abs(error.astype(np.float64)), axis=(0, 1))
        count += int(error.shape[0] * error.shape[1])
    rmse = np.sqrt(sum_squared / max(count, 1))
    mae = sum_absolute / max(count, 1)
    result: dict[str, float] = {
        "temperature_rmse_c": float(rmse[0]),
        "temperature_mae_c": float(mae[0]),
    }
    if model.output_dim == 3:
        result.update(
            qroom_rmse_w_m2=float(rmse[1]),
            qroom_mae_w_m2=float(mae[1]),
            pel_rmse_w_m2=float(rmse[2]),
            pel_mae_w_m2=float(mae[2]),
        )
    return result


@eqx.filter_jit
def _predict_lstm_batch(model, metadata, inputs, initial):
    return jax.vmap(model)(metadata, inputs, initial)


def _epoch_windows(
    *,
    config: ExperimentConfig,
    spec: ModelSpec,
    train_profiles: Profiles,
    scalers: WindowScalers,
    epoch: int,
) -> Windows:
    if not config.rotate_window_starts:
        offset = 0
    else:
        offset = rotating_window_start_offset(
            epoch,
            config.optimizer.epochs,
            config.stride,
        )
    window_config = WindowConfig(
        sequence_length=config.sequence_length,
        stride=config.stride,
        target_alignment="next_step" if spec.task == "q_to_t" else "same_time",
    )
    if spec.task == "q_to_t":
        raw = make_windows(train_profiles, window_config, start_offset=offset)
        return transform_windows(raw, scalers)
    raw = make_closed_loop_windows(train_profiles, window_config, start_offset=offset)
    return transform_closed_loop_windows(raw, scalers)


def train_lstm(config: ExperimentConfig, spec: ModelSpec) -> Path:
    splits, train_profiles, _, train_windows, test_windows, scalers = _prepare_data(config, spec)
    output_dim = int(train_windows.targets.shape[-1])
    model = AutoregressiveLSTM(
        metadata_dim=int(train_windows.metadata.shape[-1]),
        input_dim=int(train_windows.inputs.shape[-1]),
        output_dim=output_dim,
        hidden_dim=config.lstm.hidden_dim,
        metadata_hidden_dim=config.lstm.metadata_hidden_dim,
        metadata_depth=config.lstm.metadata_depth,
        bptt_truncate_steps=config.lstm.bptt_truncate_steps,
        key=jax.random.PRNGKey(config.seed),
    )
    optimizer = _build_optimizer(config)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    output_weights = _output_weights(config, output_dim)
    train_step = _make_steps(
        optimizer,
        output_weights=output_weights,
        target_mean=jnp.asarray(scalers.target.mean),
        target_scale=jnp.asarray(scalers.target.scale),
    )
    train_eval = _subset_windows(
        train_windows,
        config.optimizer.train_eval_max_windows,
        config.seed + 101,
    )
    test_eval = _subset_windows(
        test_windows,
        config.optimizer.train_eval_max_windows,
        config.seed + 102,
    )
    rng = np.random.default_rng(config.seed + 7)
    best_model = model
    best_epoch = 0
    best_metric = float("inf")
    best_train_metrics: dict[str, float] = {}
    best_test_metrics: dict[str, float] = {}
    wait = 0

    print(f"model={spec.name} backend=lstm task={spec.task}")
    print(
        f"train_profiles={len(splits.train_ids)} test_profiles={len(splits.test_ids)} "
        f"train_windows={train_windows.targets.shape[0]} test_windows={test_windows.targets.shape[0]}"
    )
    for epoch in range(1, config.optimizer.epochs + 1):
        epoch_start = time.perf_counter()
        epoch_train = _epoch_windows(
            config=config,
            spec=spec,
            train_profiles=train_profiles,
            scalers=scalers,
            epoch=epoch,
        )
        losses: list[float] = []
        for metadata, inputs, initial_temperature, targets in _batches(
            epoch_train,
            batch_size=config.optimizer.batch_size,
            rng=rng,
            shuffle=True,
        ):
            model, opt_state, loss = train_step(
                model,
                opt_state,
                jnp.asarray(metadata),
                jnp.asarray(inputs),
                jnp.asarray(initial_temperature),
                jnp.asarray(targets),
            )
            value = float(loss)
            if not np.isfinite(value):
                raise FloatingPointError(f"Non-finite LSTM loss at epoch {epoch}")
            losses.append(value)
        train_metrics = _evaluate(
            model,
            train_eval,
            scalers,
            batch_size=config.optimizer.batch_size,
        )
        test_metrics = _evaluate(
            model,
            test_eval,
            scalers,
            batch_size=config.optimizer.batch_size,
        )
        metric = test_metrics["temperature_rmse_c"]
        improved = metric < best_metric - config.optimizer.early_stopping_min_delta
        if improved:
            best_metric = metric
            best_epoch = epoch
            best_model = model
            best_train_metrics = train_metrics
            best_test_metrics = test_metrics
            wait = 0
        else:
            wait += 1
        elapsed_minutes = (time.perf_counter() - epoch_start) / 60.0
        extra = " checkpoint=best" if improved else f" checkpoint_wait={wait}"
        print(
            f"epoch={epoch:03d} train_loss={np.mean(losses):.6f} "
            f"train_rmse_c={train_metrics['temperature_rmse_c']:.4f} "
            f"test_rmse_c={test_metrics['temperature_rmse_c']:.4f} "
            f"elapsed_min={elapsed_minutes:.2f}{extra}"
        )
        patience = config.optimizer.early_stopping_patience
        if patience is not None and wait >= patience:
            print(f"early_stopping epoch={epoch} best_epoch={best_epoch}")
            break

    artifact_dir = Path(config.output_dir) / "artifacts" / spec.name / "selected"
    target_columns = [TARGET_COLUMN] if spec.task == "q_to_t" else CLOSED_LOOP_TARGET_COLUMNS
    return save_lstm_artifact(
        artifact_dir,
        model=best_model,
        scalers=scalers,
        spec=spec,
        experiment_config=config,
        input_columns=splits.input_columns,
        metadata_columns=splits.metadata_columns,
        target_columns=target_columns,
        selected_ids=splits.selected_ids,
        train_ids=splits.train_ids,
        test_ids=splits.test_ids,
        checkpoint_epoch=best_epoch,
        checkpoint_metric_value=best_metric,
        train_metrics=best_train_metrics,
        test_metrics=best_test_metrics,
    )


def train(config: ExperimentConfig) -> Path:
    """Train a registered model and return its selected artifact directory."""
    config.validate()
    spec = get_model_spec(config.model_name)
    if spec.backend == "legacy":
        return train_legacy(config, spec)
    return train_lstm(config, spec)
