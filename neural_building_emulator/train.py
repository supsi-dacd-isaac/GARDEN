"""Train a metadata-conditioned stable state-space building emulator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import plotly.graph_objects as go

from .columns import DISTURBANCE_COLUMNS, HEATING_INPUT_COLUMNS
from .data import (
    BuildingProfile,
    DEFAULT_DATASET_PATH,
    SplitConfig,
    WindowConfig,
    WindowedArrays,
    load_result_splits,
    make_windows,
    to_profiles,
)
from .metrics import regression_metrics
from .models import MetadataStateSpaceEmulator, spectral_radius
from .scaling import WindowScalers, fit_window_scalers, inverse_target, transform_windows

TargetMode = Literal["absolute", "delta", "residual"]
CheckpointMetric = Literal["auto", "train_rmse_c", "test_rmse_c"]


@dataclass(frozen=True)
class TrainConfig:
    dataset_path: Path = DEFAULT_DATASET_PATH
    max_profiles: int | None = 10
    test_fraction: float = 0.2
    seed: int = 13
    heating_mode: str = "A"
    sequence_length: int = 96
    stride: int = 96
    state_dim: int = 6
    hidden_dim: int = 64
    depth: int = 3
    schur_gamma: float = 0.995
    schur_mode: str = "near_identity"
    batch_size: int = 128
    epochs: int = 5
    learning_rate: float = 1e-3
    max_train_batches: int | None = None
    output_dir: Path = Path("output/neural_building_emulator")
    num_window_plots: int = 1
    num_full_profile_plots: int = 1
    monotonicity_weight: float = 0.0
    monotonicity_horizon: int | None = None
    monotonicity_features: tuple[str, ...] = ("heat", "outdoor_temperature", "solar")
    target_mode: TargetMode = "absolute"
    checkpoint_metric: CheckpointMetric = "auto"
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0


def minibatches(
    windows: WindowedArrays,
    *,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    n = windows.targets.shape[0]
    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, n, batch_size):
        batch_idx = indices[start : start + batch_size]
        yield (
            windows.metadata[batch_idx],
            windows.inputs[batch_idx],
            windows.initial_temperature[batch_idx],
            windows.targets[batch_idx],
        )


def predict_batch(
    model: MetadataStateSpaceEmulator,
    metadata: jnp.ndarray,
    inputs: jnp.ndarray,
    initial_temperature: jnp.ndarray,
    target_mode: TargetMode = "absolute",
) -> jnp.ndarray:
    raw_prediction = jax.vmap(model)(metadata, inputs, initial_temperature)
    return reconstruct_batch_temperature(raw_prediction, initial_temperature, target_mode)


def reconstruct_temperature(
    raw_prediction: jnp.ndarray,
    initial_temperature: jnp.ndarray,
    target_mode: TargetMode,
) -> jnp.ndarray:
    """Convert raw model outputs to normalized absolute temperatures."""
    if target_mode == "absolute":
        return raw_prediction
    if target_mode == "residual":
        initial = initial_temperature[jnp.newaxis, :]
        return initial + raw_prediction - raw_prediction[:1]
    if target_mode == "delta":
        initial = initial_temperature[jnp.newaxis, :]
        if raw_prediction.shape[0] == 1:
            return initial
        integrated = initial + jnp.cumsum(raw_prediction[:-1], axis=0)
        return jnp.concatenate([initial, integrated], axis=0)
    raise ValueError(f"Unknown target_mode {target_mode!r}")


def reconstruct_batch_temperature(
    raw_prediction: jnp.ndarray,
    initial_temperature: jnp.ndarray,
    target_mode: TargetMode,
) -> jnp.ndarray:
    if target_mode == "absolute":
        return raw_prediction
    return jax.vmap(lambda pred, init: reconstruct_temperature(pred, init, target_mode))(
        raw_prediction,
        initial_temperature,
    )


def resolve_monotonicity_feature_indices(
    columns: Sequence[str],
    feature_names: Sequence[str],
) -> tuple[int, ...]:
    """Map physics feature aliases to input-channel indices."""
    aliases = {
        "heat": columns[0],
        "heating": columns[0],
        "q_heat": columns[0],
        "heating_power": columns[0],
        "outdoor": DISTURBANCE_COLUMNS[0],
        "outdoor_temperature": DISTURBANCE_COLUMNS[0],
        "t_out": DISTURBANCE_COLUMNS[0],
        "solar": DISTURBANCE_COLUMNS[1],
        "solar_radiation": DISTURBANCE_COLUMNS[1],
        "g_solar": DISTURBANCE_COLUMNS[1],
    }
    column_indices = {column: index for index, column in enumerate(columns)}

    indices: list[int] = []
    for feature_name in feature_names:
        key = feature_name.strip().lower().replace("-", "_")
        if key not in aliases:
            valid = ", ".join(sorted(aliases))
            raise ValueError(
                f"Unknown monotonicity feature {feature_name!r}. Expected one of: {valid}"
            )
        column = aliases[key]
        if column not in column_indices:
            raise ValueError(
                f"Monotonicity feature {feature_name!r} maps to missing input column {column!r}"
            )
        index = column_indices[column]
        if index not in indices:
            indices.append(index)
    return tuple(indices)


def _impulse_response_monotonicity_penalty_one(
    model: MetadataStateSpaceEmulator,
    metadata: jnp.ndarray,
    feature_indices: tuple[int, ...],
    horizon: int,
    target_mode: TargetMode,
) -> jnp.ndarray:
    matrices = model.matrices(metadata)
    feature_idx = jnp.asarray(feature_indices)
    direct_response = matrices.d[:, feature_idx]

    if horizon == 1:
        responses = direct_response[jnp.newaxis, :, :]
    else:
        initial_state_response = matrices.b[:, feature_idx]

        def step(
            state_response: jnp.ndarray,
            _: None,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            output_response = matrices.c @ state_response
            next_state_response = matrices.a @ state_response
            return next_state_response, output_response

        _, future_responses = jax.lax.scan(
            step,
            initial_state_response,
            None,
            length=horizon - 1,
        )
        responses = jnp.concatenate(
            [direct_response[jnp.newaxis, :, :], future_responses],
            axis=0,
        )
    if target_mode == "delta":
        zero_response = jnp.zeros_like(responses[:1])
        responses = jnp.concatenate(
            [zero_response, jnp.cumsum(responses[:-1], axis=0)],
            axis=0,
        )
    return jnp.mean(jax.nn.relu(-responses) ** 2)


def impulse_response_monotonicity_penalty(
    model: MetadataStateSpaceEmulator,
    metadata: jnp.ndarray,
    feature_indices: tuple[int, ...],
    horizon: int,
    target_mode: TargetMode,
) -> jnp.ndarray:
    """Penalize negative d indoor-temperature / d input impulse responses."""
    penalties = jax.vmap(
        lambda row: _impulse_response_monotonicity_penalty_one(
            model,
            row,
            feature_indices,
            horizon,
            target_mode,
        )
    )(metadata)
    return jnp.mean(penalties)


@eqx.filter_value_and_grad
def loss_fn(
    model: MetadataStateSpaceEmulator,
    metadata: jnp.ndarray,
    inputs: jnp.ndarray,
    initial_temperature: jnp.ndarray,
    targets: jnp.ndarray,
    monotonicity_weight: float,
    monotonicity_feature_indices: tuple[int, ...],
    monotonicity_horizon: int,
    target_mode: TargetMode,
) -> jnp.ndarray:
    prediction = predict_batch(model, metadata, inputs, initial_temperature, target_mode)
    mse = jnp.mean((prediction - targets) ** 2)
    if monotonicity_weight <= 0.0 or not monotonicity_feature_indices:
        return mse
    monotonicity = impulse_response_monotonicity_penalty(
        model,
        metadata,
        monotonicity_feature_indices,
        monotonicity_horizon,
        target_mode,
    )
    return mse + monotonicity_weight * monotonicity


@eqx.filter_jit
def train_step(
    model: MetadataStateSpaceEmulator,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    metadata: jnp.ndarray,
    inputs: jnp.ndarray,
    initial_temperature: jnp.ndarray,
    targets: jnp.ndarray,
    monotonicity_weight: float,
    monotonicity_feature_indices: tuple[int, ...],
    monotonicity_horizon: int,
    target_mode: TargetMode,
) -> tuple[MetadataStateSpaceEmulator, optax.OptState, jnp.ndarray]:
    loss, grads = loss_fn(
        model,
        metadata,
        inputs,
        initial_temperature,
        targets,
        monotonicity_weight,
        monotonicity_feature_indices,
        monotonicity_horizon,
        target_mode,
    )
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def evaluate(
    model: MetadataStateSpaceEmulator,
    windows: WindowedArrays,
    scalers: WindowScalers,
    *,
    batch_size: int,
    target_mode: TargetMode,
) -> dict[str, float]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    rng = np.random.default_rng(0)
    for metadata, inputs, initial_temperature, target in minibatches(
        windows,
        batch_size=batch_size,
        rng=rng,
        shuffle=False,
    ):
        pred = predict_batch(
            model,
            jnp.asarray(metadata),
            jnp.asarray(inputs),
            jnp.asarray(initial_temperature),
            target_mode,
        )
        predictions.append(np.asarray(pred))
        targets.append(target)

    pred_norm = np.concatenate(predictions, axis=0)
    target_norm = np.concatenate(targets, axis=0)
    normalized = regression_metrics(pred_norm, target_norm)
    physical = regression_metrics(
        inverse_target(pred_norm, scalers),
        inverse_target(target_norm, scalers),
    )
    return {
        "loss": normalized.rmse**2,
        "rmse_c": physical.rmse,
        "mae_c": physical.mae,
        "nmae": physical.nmae,
        "bias_c": physical.bias,
    }


def resolve_checkpoint_metric(
    metric: CheckpointMetric,
    *,
    has_test_windows: bool,
) -> CheckpointMetric:
    if metric == "auto":
        return "test_rmse_c" if has_test_windows else "train_rmse_c"
    if metric == "test_rmse_c" and not has_test_windows:
        raise ValueError("checkpoint_metric='test_rmse_c' requires a non-empty test split")
    return metric


def checkpoint_metric_value(
    metric: CheckpointMetric,
    train_eval: dict[str, float],
    test_eval: dict[str, float] | None,
) -> float:
    if metric == "train_rmse_c":
        return train_eval["rmse_c"]
    if metric == "test_rmse_c" and test_eval is not None:
        return test_eval["rmse_c"]
    raise ValueError(f"Cannot compute checkpoint metric {metric!r}")


def predict_full_profile(
    model: MetadataStateSpaceEmulator,
    profile: BuildingProfile,
    scalers: WindowScalers,
    target_mode: TargetMode,
) -> np.ndarray:
    """Run one continuous rollout over a complete profile."""
    metadata = scalers.metadata.transform(profile.metadata)
    inputs = scalers.inputs.transform(profile.inputs)
    initial_temperature = scalers.target.transform(profile.target[0])
    raw_prediction = model(
        jnp.asarray(metadata),
        jnp.asarray(inputs),
        jnp.asarray(initial_temperature),
    )
    prediction = reconstruct_temperature(
        raw_prediction,
        jnp.asarray(initial_temperature),
        target_mode,
    )
    return inverse_target(np.asarray(prediction), scalers)


def evaluate_full_profiles(
    model: MetadataStateSpaceEmulator,
    profiles: list[BuildingProfile],
    scalers: WindowScalers,
    target_mode: TargetMode,
) -> dict[str, float]:
    """Evaluate one continuous rollout per full profile."""
    if not profiles:
        return {
            "profile_count": 0.0,
            "rmse_c": float("nan"),
            "mae_c": float("nan"),
            "nmae": float("nan"),
            "bias_c": float("nan"),
        }

    predictions = []
    targets = []
    for profile in profiles:
        predictions.append(predict_full_profile(model, profile, scalers, target_mode))
        targets.append(profile.target)

    metrics = regression_metrics(
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
    )
    return {
        "profile_count": float(len(profiles)),
        "rmse_c": metrics.rmse,
        "mae_c": metrics.mae,
        "nmae": metrics.nmae,
        "bias_c": metrics.bias,
    }


def _profile_by_id(profiles: list[BuildingProfile]) -> dict[int, BuildingProfile]:
    return {profile.profile_id: profile for profile in profiles}


def _evenly_spaced_indices(count: int, limit: int) -> list[int]:
    if count <= 0 or limit <= 0:
        return []
    n = min(count, limit)
    return [int(index) for index in np.linspace(0, limit - 1, n, dtype=np.int64)]


def _prediction_metrics_text(prediction: np.ndarray, target: np.ndarray) -> str:
    metrics = regression_metrics(prediction, target)
    return (
        f"RMSE={metrics.rmse:.3f} degC, "
        f"MAE={metrics.mae:.3f} degC, "
        f"bias={metrics.bias:.3f} degC"
    )


def _write_temperature_comparison(
    *,
    path: Path,
    title: str,
    datetimes: np.ndarray,
    simulated: np.ndarray,
    emulated: np.ndarray,
) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=datetimes,
            y=simulated.ravel(),
            mode="lines",
            name="Simulated",
            line={"color": "#1f77b4"},
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=datetimes,
            y=emulated.ravel(),
            mode="lines",
            name="Emulated",
            line={"color": "#d62728"},
        )
    )
    fig.update_layout(
        title=f"{title}<br><sup>{_prediction_metrics_text(emulated, simulated)}</sup>",
        xaxis_title="Time",
        yaxis_title="Indoor temperature [degC]",
        template="plotly_white",
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
        },
    )
    fig.write_html(path, include_plotlyjs=True)


def save_prediction_visualizations(
    model: MetadataStateSpaceEmulator,
    test_windows: WindowedArrays,
    test_profiles: list[BuildingProfile],
    scalers: WindowScalers,
    output_dir: Path,
    target_mode: TargetMode,
    num_window_plots: int,
    num_full_profile_plots: int,
) -> dict[str, Path]:
    """Save fixed-window and full-profile rollout plots."""
    if test_windows.targets.shape[0] == 0 or not test_profiles:
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    profiles_by_id = _profile_by_id(test_profiles)
    paths: dict[str, Path] = {}

    for plot_number, window_index in enumerate(
        _evenly_spaced_indices(num_window_plots, test_windows.targets.shape[0]),
        start=1,
    ):
        window_profile_id = int(test_windows.profile_ids[window_index])
        window_start = int(test_windows.start_indices[window_index])
        window_profile = profiles_by_id[window_profile_id]
        window_raw_prediction = model(
            jnp.asarray(test_windows.metadata[window_index]),
            jnp.asarray(test_windows.inputs[window_index]),
            jnp.asarray(test_windows.initial_temperature[window_index]),
        )
        window_prediction_norm = reconstruct_temperature(
            window_raw_prediction,
            jnp.asarray(test_windows.initial_temperature[window_index]),
            target_mode,
        )
        window_prediction = inverse_target(np.asarray(window_prediction_norm), scalers)
        window_target = inverse_target(test_windows.targets[window_index], scalers)
        window_end = window_start + window_prediction.shape[0]
        window_path = (
            output_dir
            / f"test_window_{plot_number:02d}_profile_{window_profile_id}_start_{window_start}.html"
        )
        _write_temperature_comparison(
            path=window_path,
            title=(
                f"{window_prediction.shape[0]}-step test rollout, "
                f"profile {window_profile_id}, start {window_start}"
            ),
            datetimes=window_profile.datetime[window_start:window_end],
            simulated=window_target,
            emulated=window_prediction,
        )
        paths[f"test_window_{plot_number:02d}"] = window_path

    for plot_number, profile_index in enumerate(
        _evenly_spaced_indices(num_full_profile_plots, len(test_profiles)),
        start=1,
    ):
        full_profile = test_profiles[profile_index]
        full_prediction = predict_full_profile(model, full_profile, scalers, target_mode)
        full_path = output_dir / f"test_full_profile_{plot_number:02d}_{full_profile.profile_id}.html"
        _write_temperature_comparison(
            path=full_path,
            title=f"Full-profile continuous rollout, profile {full_profile.profile_id}",
            datetimes=full_profile.datetime,
            simulated=full_profile.target,
            emulated=full_prediction,
        )
        paths[f"full_profile_{plot_number:02d}"] = full_path

    return paths


def sample_spectral_radius(
    model: MetadataStateSpaceEmulator,
    windows: WindowedArrays,
    n: int = 32,
) -> float:
    count = min(n, windows.metadata.shape[0])
    radii = []
    for metadata in windows.metadata[:count]:
        matrices = model.matrices(jnp.asarray(metadata))
        radii.append(float(spectral_radius(matrices.a)))
    return float(np.max(radii)) if radii else float("nan")


def run_training(config: TrainConfig) -> MetadataStateSpaceEmulator:
    if config.monotonicity_weight < 0.0:
        raise ValueError("monotonicity_weight must be non-negative")
    if config.target_mode not in ("absolute", "delta", "residual"):
        raise ValueError("target_mode must be 'absolute', 'delta', or 'residual'")
    if config.checkpoint_metric not in ("auto", "train_rmse_c", "test_rmse_c"):
        raise ValueError("checkpoint_metric must be 'auto', 'train_rmse_c', or 'test_rmse_c'")
    if config.early_stopping_patience is not None and config.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive or None")
    if config.early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be non-negative")
    if config.num_window_plots < 0:
        raise ValueError("num_window_plots must be non-negative")
    if config.num_full_profile_plots < 0:
        raise ValueError("num_full_profile_plots must be non-negative")

    split_config = SplitConfig(
        dataset_path=config.dataset_path,
        max_profiles=config.max_profiles,
        test_fraction=config.test_fraction,
        seed=config.seed,
    )
    splits = load_result_splits(split_config, heating_mode=config.heating_mode)
    window_config = WindowConfig(sequence_length=config.sequence_length, stride=config.stride)
    monotonicity_horizon = config.monotonicity_horizon or config.sequence_length
    if monotonicity_horizon < 1:
        raise ValueError("monotonicity_horizon must be positive")
    monotonicity_feature_indices = resolve_monotonicity_feature_indices(
        splits.input_columns,
        config.monotonicity_features,
    )
    train_profiles = to_profiles(splits.train, splits.heating_mode)
    test_profiles = to_profiles(splits.test, splits.heating_mode) if splits.test_ids else []
    train_windows = make_windows(train_profiles, window_config)
    test_windows = None
    if splits.test_ids:
        test_windows = make_windows(test_profiles, window_config)
    resolved_checkpoint_metric = resolve_checkpoint_metric(
        config.checkpoint_metric,
        has_test_windows=test_windows is not None,
    )

    scalers = fit_window_scalers(train_windows)
    train_windows = transform_windows(train_windows, scalers)
    if test_windows is not None:
        test_windows = transform_windows(test_windows, scalers)

    key = jax.random.PRNGKey(config.seed)
    model = MetadataStateSpaceEmulator(
        metadata_dim=train_windows.metadata.shape[-1],
        input_dim=train_windows.inputs.shape[-1],
        state_dim=config.state_dim,
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        schur_gamma=config.schur_gamma,
        schur_mode=config.schur_mode,  # type: ignore[arg-type]
        key=key,
    )
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    rng = np.random.default_rng(config.seed)

    print(f"heating_mode={splits.heating_mode}")
    print(f"target_mode={config.target_mode}")
    print(f"input_columns={list(splits.input_columns)}")
    print(f"train_profiles={len(splits.train_ids)} test_profiles={len(splits.test_ids)}")
    print(f"train_windows={train_windows.targets.shape[0]}")
    if test_windows is not None:
        print(f"test_windows={test_windows.targets.shape[0]}")
    print(f"checkpoint_metric={resolved_checkpoint_metric}")
    if config.early_stopping_patience is not None:
        print(
            "early_stopping=enabled "
            f"patience={config.early_stopping_patience} "
            f"min_delta={config.early_stopping_min_delta}"
        )
    if config.monotonicity_weight > 0.0:
        selected_columns = [splits.input_columns[index] for index in monotonicity_feature_indices]
        print(
            "monotonicity_regularization=enabled "
            f"weight={config.monotonicity_weight} "
            f"horizon={monotonicity_horizon} "
            f"input_columns={selected_columns}"
        )

    best_model = model
    best_epoch = 0
    best_metric = float("inf")
    best_train_eval: dict[str, float] | None = None
    best_test_eval: dict[str, float] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        losses = []
        for batch_idx, (metadata, inputs, initial_temperature, targets) in enumerate(
            minibatches(train_windows, batch_size=config.batch_size, rng=rng, shuffle=True),
            start=1,
        ):
            model, opt_state, loss = train_step(
                model,
                opt_state,
                optimizer,
                jnp.asarray(metadata),
                jnp.asarray(inputs),
                jnp.asarray(initial_temperature),
                jnp.asarray(targets),
                config.monotonicity_weight,
                monotonicity_feature_indices,
                monotonicity_horizon,
                config.target_mode,
            )
            losses.append(float(loss))
            if config.max_train_batches is not None and batch_idx >= config.max_train_batches:
                break

        train_eval = evaluate(
            model,
            train_windows,
            scalers,
            batch_size=config.batch_size,
            target_mode=config.target_mode,
        )
        message = (
            f"epoch={epoch:03d} train_loss={np.mean(losses):.6f} "
            f"train_rmse_c={train_eval['rmse_c']:.4f} "
            f"train_nmae={train_eval['nmae']:.4f} "
            f"rho_max={sample_spectral_radius(model, train_windows):.5f}"
        )
        if test_windows is not None:
            test_eval = evaluate(
                model,
                test_windows,
                scalers,
                batch_size=config.batch_size,
                target_mode=config.target_mode,
            )
            message += f" test_rmse_c={test_eval['rmse_c']:.4f} test_nmae={test_eval['nmae']:.4f}"
        metric_value = checkpoint_metric_value(
            resolved_checkpoint_metric,
            train_eval,
            test_eval if test_windows is not None else None,
        )
        if metric_value < best_metric - config.early_stopping_min_delta:
            best_model = model
            best_epoch = epoch
            best_metric = metric_value
            best_train_eval = train_eval
            best_test_eval = test_eval if test_windows is not None else None
            epochs_without_improvement = 0
            message += f" checkpoint=best_{resolved_checkpoint_metric}"
        else:
            epochs_without_improvement += 1
            message += f" checkpoint_wait={epochs_without_improvement}"
        print(message)

        if (
            config.early_stopping_patience is not None
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            print(
                "early_stopped "
                f"epoch={epoch} "
                f"best_epoch={best_epoch} "
                f"best_{resolved_checkpoint_metric}={best_metric:.4f}"
            )
            break

    model = best_model
    print(
        "selected_checkpoint "
        f"epoch={best_epoch} "
        f"{resolved_checkpoint_metric}={best_metric:.4f}"
    )
    if best_train_eval is not None:
        print(
            "selected_train_metrics "
            f"rmse_c={best_train_eval['rmse_c']:.4f} "
            f"nmae={best_train_eval['nmae']:.4f}"
        )
    if best_test_eval is not None:
        print(
            "selected_test_metrics "
            f"rmse_c={best_test_eval['rmse_c']:.4f} "
            f"nmae={best_test_eval['nmae']:.4f}"
        )

    if test_windows is not None:
        full_profile_eval = evaluate_full_profiles(
            model,
            test_profiles,
            scalers,
            config.target_mode,
        )
        print(
            "full_profile_test "
            f"profiles={int(full_profile_eval['profile_count'])} "
            f"rmse_c={full_profile_eval['rmse_c']:.4f} "
            f"mae_c={full_profile_eval['mae_c']:.4f} "
            f"nmae={full_profile_eval['nmae']:.4f} "
            f"bias_c={full_profile_eval['bias_c']:.4f}"
        )
        visualization_paths = save_prediction_visualizations(
            model,
            test_windows,
            test_profiles,
            scalers,
            config.output_dir,
            config.target_mode,
            config.num_window_plots,
            config.num_full_profile_plots,
        )
        for name, path in visualization_paths.items():
            print(f"saved_{name}_plot={path}")

    return model


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--max-profiles", type=int, default=10)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--heating-mode", choices=(*HEATING_INPUT_COLUMNS.keys(), "A", "B"), default="A")
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument(
        "--target-mode",
        choices=("absolute", "delta", "residual"),
        default="absolute",
        help=(
            "Output reconstruction mode: absolute, delta integration, or residual anchoring "
            "to the true initial temperature."
        ),
    )
    parser.add_argument("--state-dim", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--schur-gamma", type=float, default=0.995)
    parser.add_argument("--schur-mode", choices=("dense", "near_identity"), default="near_identity")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("output/neural_building_emulator"))
    parser.add_argument(
        "--num-window-plots",
        type=int,
        default=1,
        help="Number of held-out fixed-window HTML comparisons to save after training.",
    )
    parser.add_argument(
        "--num-full-profile-plots",
        type=int,
        default=1,
        help="Number of held-out full-profile HTML comparisons to save after training.",
    )
    parser.add_argument(
        "--checkpoint-metric",
        choices=("auto", "train_rmse_c", "test_rmse_c"),
        default="auto",
        help="Metric used to select the best in-memory checkpoint. Auto prefers test_rmse_c when available.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Stop after this many epochs without checkpoint-metric improvement. Disabled by default.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum checkpoint-metric improvement required to reset early-stopping patience.",
    )
    parser.add_argument(
        "--monotonicity-weight",
        type=float,
        default=0.0,
        help="Weight for non-negative heat/outdoor/solar impulse-response regularization.",
    )
    parser.add_argument(
        "--monotonicity-horizon",
        type=int,
        default=None,
        help="Impulse-response horizon for monotonicity regularization. Defaults to sequence length.",
    )
    parser.add_argument(
        "--monotonicity-features",
        nargs="+",
        default=["heat", "outdoor_temperature", "solar"],
        help="Input features to constrain. Aliases include heat, outdoor_temperature, and solar.",
    )
    args = parser.parse_args()
    return TrainConfig(
        dataset_path=args.dataset,
        max_profiles=args.max_profiles,
        test_fraction=args.test_fraction,
        seed=args.seed,
        heating_mode=args.heating_mode,
        sequence_length=args.sequence_length,
        stride=args.stride,
        target_mode=args.target_mode,
        state_dim=args.state_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        schur_gamma=args.schur_gamma,
        schur_mode=args.schur_mode,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_train_batches=args.max_train_batches,
        output_dir=args.output_dir,
        num_window_plots=args.num_window_plots,
        num_full_profile_plots=args.num_full_profile_plots,
        checkpoint_metric=args.checkpoint_metric,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        monotonicity_weight=args.monotonicity_weight,
        monotonicity_horizon=args.monotonicity_horizon,
        monotonicity_features=tuple(args.monotonicity_features),
    )


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
