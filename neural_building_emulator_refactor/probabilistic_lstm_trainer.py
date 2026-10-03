"""Matched trajectory training for the probabilistic closed-loop LSTM baseline."""

from __future__ import annotations

import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from neural_building_emulator.columns import CLOSED_LOOP_TARGET_COLUMNS
from neural_building_emulator.train import (
    energy_score,
    soft_optimistic_loss,
    variogram_score,
)

from .artifacts import save_probabilistic_lstm_artifact
from .config import ExperimentConfig, ProbabilisticLSTMConfig
from .gradient_logging import GradientUpdateLogger
from .models import (
    ProbabilisticAutoregressiveLSTM,
    VanillaProbabilisticAutoregressiveLSTM,
)
from .registry import ModelSpec
from .trainer import (
    _batches,
    _build_optimizer,
    _epoch_windows,
    _initial_output,
    _prepare_data,
    _subset_windows,
)


ProbabilisticLSTMModel = (
    ProbabilisticAutoregressiveLSTM | VanillaProbabilisticAutoregressiveLSTM
)


def _sample_batch_with_aux(
    model: ProbabilisticLSTMModel,
    metadata: jnp.ndarray,
    inputs: jnp.ndarray,
    initial: jnp.ndarray,
    key: jax.Array,
    *,
    num_particles: int,
    hp_scenario_mode: str,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
    keys = jax.random.split(key, metadata.shape[0])
    return jax.vmap(
        lambda row_metadata, row_inputs, row_initial, row_key: model.sample_with_aux(
            row_metadata,
            row_inputs,
            row_initial,
            key=row_key,
            num_particles=num_particles,
            sample_process_noise=True,
            hp_scenario_mode=hp_scenario_mode,
        )
    )(metadata, inputs, initial, keys)


def _loss_components(
    model: ProbabilisticAutoregressiveLSTM,
    metadata: jnp.ndarray,
    inputs: jnp.ndarray,
    initial: jnp.ndarray,
    targets: jnp.ndarray,
    key: jax.Array,
    settings: ProbabilisticLSTMConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    predictions, aux = _sample_batch_with_aux(
        model,
        metadata,
        inputs,
        initial,
        key,
        num_particles=settings.particles,
        hp_scenario_mode=settings.hp_training_mode,
    )
    energy = energy_score(
        predictions,
        targets,
        output_weights=settings.output_weights,
    )
    variogram = jnp.asarray(0.0, dtype=predictions.dtype)
    if settings.variogram_weight > 0.0:
        threshold_scaled = (
            settings.variogram_setpoint_threshold_c
            / max(float(model.input_scale[model.setpoint_input_index]), 1e-6)
        )
        variogram = settings.variogram_weight * variogram_score(
            predictions,
            targets,
            lags=settings.variogram_lags,
            power=settings.variogram_power,
            output_weights=settings.variogram_output_weights,
            constant_setpoint_inputs=(
                inputs if settings.variogram_constant_setpoint_only else None
            ),
            setpoint_delta_threshold=threshold_scaled,
        )
    softopt = jnp.asarray(0.0, dtype=predictions.dtype)
    if settings.softopt_weight > 0.0:
        softopt = settings.softopt_weight * soft_optimistic_loss(
            predictions,
            targets,
            temperature=settings.softopt_temperature,
            horizon_weight_power=0.0,
            output_weights=settings.output_weights,
        )

    target_mean = jnp.asarray(model.target_mean, dtype=targets.dtype)
    target_scale = jnp.asarray(model.target_scale, dtype=targets.dtype)
    pel_target = targets[:, :, 2] * target_scale[2] + target_mean[2]
    mode_target = (pel_target > 1e-6).astype(targets.dtype)
    mode_target_particles = mode_target[:, jnp.newaxis, :]
    pi, log_mu, log_sigma, p_start, p_stop, expected_active = aux

    hp_mode = jnp.asarray(0.0, dtype=predictions.dtype)
    if settings.hp_mode_loss_weight > 0.0 and inputs.shape[1] > 1:
        mean_start = jnp.mean(p_start[:, :, 1:], axis=1, keepdims=True)
        mean_stop = jnp.mean(p_stop[:, :, 1:], axis=1, keepdims=True)
        previous_mode = mode_target_particles[:, :, :-1]
        current_mode = mode_target_particles[:, :, 1:]
        availability = jax.vmap(jax.vmap(model._availability))(inputs)[
            :, jnp.newaxis, 1:
        ]
        eps = jnp.asarray(1e-5, dtype=predictions.dtype)
        mean_start = jnp.clip(mean_start, eps, 1.0 - eps)
        mean_stop = jnp.clip(mean_stop, eps, 1.0 - eps)
        start_risk = (1.0 - previous_mode) * availability
        stop_risk = previous_mode * availability
        start_nll = -(
            current_mode * jnp.log(mean_start)
            + (1.0 - current_mode) * jnp.log1p(-mean_start)
        )
        stop_target = 1.0 - current_mode
        stop_nll = -(
            stop_target * jnp.log(mean_stop)
            + (1.0 - stop_target) * jnp.log1p(-mean_stop)
        )
        start_loss = jnp.sum(start_risk * start_nll) / jnp.maximum(
            jnp.sum(start_risk), 1.0
        )
        stop_loss = jnp.sum(stop_risk * stop_nll) / jnp.maximum(
            jnp.sum(stop_risk), 1.0
        )
        hp_mode = settings.hp_mode_loss_weight * 0.5 * (
            start_loss + stop_loss
        )

    hp_active = jnp.asarray(0.0, dtype=predictions.dtype)
    if settings.hp_active_power_nll_weight > 0.0:
        active_mask = jnp.broadcast_to(mode_target_particles, log_mu.shape)
        log_target = jnp.log1p(jnp.maximum(pel_target, 0.0))[:, jnp.newaxis, :]
        nll = (
            0.5 * ((log_target - log_mu) / log_sigma) ** 2
            + jnp.log(log_sigma)
            + 0.5 * jnp.log(jnp.asarray(2.0 * np.pi, dtype=predictions.dtype))
        )
        hp_active = settings.hp_active_power_nll_weight * (
            jnp.sum(active_mask * nll) / jnp.maximum(jnp.sum(active_mask), 1.0)
        )

    hp_leakage = jnp.asarray(0.0, dtype=predictions.dtype)
    if settings.hp_inactive_leakage_weight > 0.0:
        inactive_mask = jnp.broadcast_to(1.0 - mode_target_particles, pi.shape)
        expected_pel = pi * expected_active
        hp_leakage = settings.hp_inactive_leakage_weight * (
            jnp.sum(inactive_mask * (expected_pel / target_scale[2]) ** 2)
            / jnp.maximum(jnp.sum(inactive_mask), 1.0)
        )

    components = jnp.stack(
        [energy, variogram, softopt, hp_mode, hp_active, hp_leakage]
    )
    return jnp.sum(components), components


def _vanilla_loss_components(
    model: VanillaProbabilisticAutoregressiveLSTM,
    metadata: jnp.ndarray,
    inputs: jnp.ndarray,
    initial: jnp.ndarray,
    targets: jnp.ndarray,
    key: jax.Array,
    settings: ProbabilisticLSTMConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Trajectory losses for the genuinely unstructured stochastic control."""
    predictions, _ = _sample_batch_with_aux(
        model,
        metadata,
        inputs,
        initial,
        key,
        num_particles=settings.particles,
        hp_scenario_mode=settings.hp_training_mode,
    )
    energy = energy_score(
        predictions,
        targets,
        output_weights=settings.output_weights,
    )
    variogram = jnp.asarray(0.0, dtype=predictions.dtype)
    if settings.variogram_weight > 0.0:
        threshold_scaled = (
            settings.variogram_setpoint_threshold_c
            / max(float(model.input_scale[model.setpoint_input_index]), 1e-6)
        )
        variogram = settings.variogram_weight * variogram_score(
            predictions,
            targets,
            lags=settings.variogram_lags,
            power=settings.variogram_power,
            output_weights=settings.variogram_output_weights,
            constant_setpoint_inputs=(
                inputs if settings.variogram_constant_setpoint_only else None
            ),
            setpoint_delta_threshold=threshold_scaled,
        )
    softopt = jnp.asarray(0.0, dtype=predictions.dtype)
    if settings.softopt_weight > 0.0:
        softopt = settings.softopt_weight * soft_optimistic_loss(
            predictions,
            targets,
            temperature=settings.softopt_temperature,
            horizon_weight_power=0.0,
            output_weights=settings.output_weights,
        )
    components = jnp.stack([energy, variogram, softopt])
    return jnp.sum(components), components


def _make_train_step(
    optimizer: optax.GradientTransformation,
    config: ExperimentConfig,
    *,
    vanilla: bool,
):
    settings = config.probabilistic_lstm
    skip_grad_norm_above = settings.skip_grad_norm_above
    loss_function = _vanilla_loss_components if vanilla else _loss_components
    value_and_grad = eqx.filter_value_and_grad(loss_function, has_aux=True)

    @eqx.filter_jit
    def train_step(model, opt_state, metadata, inputs, initial, targets, key):
        (loss, components), grads = value_and_grad(
            model, metadata, inputs, initial, targets, key, settings
        )
        grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_inexact_array))
        updates, candidate_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        candidate = eqx.apply_updates(model, updates)
        finite = jnp.isfinite(loss) & jnp.isfinite(grad_norm)
        if skip_grad_norm_above > 0.0:
            finite = finite & (grad_norm <= skip_grad_norm_above)
        model = jax.tree.map(
            lambda new, old: jnp.where(finite, new, old)
            if eqx.is_array(new)
            else new,
            candidate,
            model,
        )
        opt_state = jax.tree.map(
            lambda new, old: jnp.where(finite, new, old),
            candidate_opt_state,
            opt_state,
        )
        return model, opt_state, loss, components, grad_norm, finite

    return train_step


def _evaluate(
    model: ProbabilisticLSTMModel,
    windows,
    scalers,
    *,
    batch_size: int,
    num_particles: int,
    seed: int,
) -> dict[str, float]:
    sum_squared = np.zeros(3, dtype=np.float64)
    sum_absolute = np.zeros(3, dtype=np.float64)
    count = 0
    key = jax.random.PRNGKey(seed)
    rng = np.random.default_rng(seed)
    for metadata, inputs, initial_temperature, targets in _batches(
        windows,
        batch_size=batch_size,
        rng=rng,
        shuffle=False,
    ):
        key, batch_key = jax.random.split(key)
        initial = _initial_output(
            jnp.asarray(initial_temperature),
            output_dim=3,
            target_mean=jnp.asarray(scalers.target.mean),
            target_scale=jnp.asarray(scalers.target.scale),
        )
        particles, _ = _sample_batch_with_aux(
            model,
            jnp.asarray(metadata),
            jnp.asarray(inputs),
            initial,
            batch_key,
            num_particles=num_particles,
            hp_scenario_mode="expected",
        )
        normalized = np.asarray(jnp.mean(particles, axis=1))
        prediction = normalized * scalers.target.scale + scalers.target.mean
        truth = targets * scalers.target.scale + scalers.target.mean
        error = prediction.astype(np.float64) - truth.astype(np.float64)
        sum_squared += np.sum(error**2, axis=(0, 1))
        sum_absolute += np.sum(np.abs(error), axis=(0, 1))
        count += int(error.shape[0] * error.shape[1])
    rmse = np.sqrt(sum_squared / max(count, 1))
    mae = sum_absolute / max(count, 1)
    nrmse = rmse / np.asarray(scalers.target.scale, dtype=np.float64)
    return {
        "temperature_rmse_c": float(rmse[0]),
        "temperature_mae_c": float(mae[0]),
        "qroom_rmse_w_m2": float(rmse[1]),
        "qroom_mae_w_m2": float(mae[1]),
        "pel_rmse_w_m2": float(rmse[2]),
        "pel_mae_w_m2": float(mae[2]),
        "total_nrmse": float(np.sqrt(np.mean(nrmse**2))),
    }


def train_probabilistic_lstm(config: ExperimentConfig, spec: ModelSpec) -> Path:
    if spec.task != "closed_loop_hp" or not spec.probabilistic:
        raise ValueError("probabilistic LSTM trainer requires a closed-loop model")
    splits, train_profiles, _, train_windows, test_windows, scalers = _prepare_data(
        config, spec
    )
    settings = config.probabilistic_lstm
    input_columns = list(splits.input_columns)
    availability_index = (
        input_columns.index("space_heating_available")
        if "space_heating_available" in input_columns
        else -1
    )
    vanilla = spec.name == "closed_loop_hp_vanilla_probabilistic_lstm"
    common_model_args = dict(
        metadata_dim=int(train_windows.metadata.shape[-1]),
        input_dim=int(train_windows.inputs.shape[-1]),
        hidden_dim=settings.hidden_dim,
        metadata_hidden_dim=settings.metadata_hidden_dim,
        metadata_depth=settings.metadata_depth,
        latent_dim=settings.latent_dim,
        process_noise_mode=settings.process_noise,
        process_noise_init=settings.process_noise_init,
        bptt_truncate_steps=settings.bptt_truncate_steps,
        setpoint_input_index=0,
        input_scale=tuple(float(value) for value in scalers.inputs.scale),
        key=jax.random.PRNGKey(config.seed),
    )
    if vanilla:
        model: ProbabilisticLSTMModel = VanillaProbabilisticAutoregressiveLSTM(
            **common_model_args
        )
    else:
        model = ProbabilisticAutoregressiveLSTM(
            **common_model_args,
            availability_input_index=availability_index,
            dt_hours=0.25,
            setpoint_shock_timescales_hours=settings.setpoint_shock_timescales_hours,
            input_mean=tuple(float(value) for value in scalers.inputs.mean),
            target_mean=tuple(float(value) for value in scalers.target.mean),
            target_scale=tuple(float(value) for value in scalers.target.scale),
        )
    optimizer = _build_optimizer(config)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    train_step = _make_train_step(optimizer, config, vanilla=vanilla)
    train_eval = _subset_windows(
        train_windows, config.optimizer.train_eval_max_windows, config.seed + 101
    )
    # Match the structured trainer: throttle train diagnostics, but use every
    # held-out window for checkpoint selection.
    test_eval = test_windows
    rng = np.random.default_rng(config.seed + 7)
    key = jax.random.PRNGKey(config.seed + 19)
    best_model = model
    best_epoch = 0
    best_metric = float("inf")
    best_train_metrics: dict[str, float] = {}
    best_test_metrics: dict[str, float] = {}
    wait = 0

    print(f"model={spec.name} backend=lstm probabilistic=true task={spec.task}")
    print(
        f"train_profiles={len(splits.train_ids)} test_profiles={len(splits.test_ids)} "
        f"train_windows={train_windows.targets.shape[0]} "
        f"test_windows={test_windows.targets.shape[0]}"
    )
    print(
        f"particles={settings.particles} eval_particles={settings.eval_particles} "
        f"latent_dim={settings.latent_dim} process_noise={settings.process_noise} "
        f"structure={'vanilla' if vanilla else 'semi_structured_hp'}"
    )
    if vanilla:
        print(
            "vanilla_outputs=linear(Tin,Qroom,Pel) "
            "hp_specific_features=none hp_specific_losses=none"
        )
        component_names = ("energy", "variogram", "softopt")
    else:
        print(
            f"hp_training_mode={settings.hp_training_mode} "
            f"hp_scenario_mode={settings.hp_scenario_mode}"
        )
        component_names = (
            "energy",
            "variogram",
            "softopt",
            "hp_mode",
            "hp_active",
            "hp_leakage",
        )
    gradient_log = GradientUpdateLogger(config.output_dir, spec.name)
    print(f"gradient_update_log={gradient_log.path}")
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
        component_sums = np.zeros(len(component_names), dtype=np.float64)
        max_grad_norm = 0.0
        skipped = 0
        batches = 0
        for batch_index, batch in enumerate(
            _batches(
                epoch_train,
                batch_size=config.optimizer.batch_size,
                rng=rng,
                shuffle=True,
            )
        ):
            metadata, inputs, initial_temperature, targets = batch
            key, step_key = jax.random.split(key)
            initial = _initial_output(
                jnp.asarray(initial_temperature),
                output_dim=3,
                target_mean=jnp.asarray(scalers.target.mean),
                target_scale=jnp.asarray(scalers.target.scale),
            )
            model, opt_state, loss, components, grad_norm, accepted = train_step(
                model,
                opt_state,
                jnp.asarray(metadata),
                jnp.asarray(inputs),
                initial,
                jnp.asarray(targets),
                step_key,
            )
            value = float(loss)
            norm = float(grad_norm)
            is_accepted = bool(accepted)
            if is_accepted:
                reason = "accepted"
            elif not np.isfinite(value):
                reason = "nonfinite_loss"
            elif not np.isfinite(norm):
                reason = "nonfinite_gradient"
            else:
                reason = "grad_threshold"
            gradient_log.log(
                epoch=epoch,
                batch_index=batch_index,
                loss=value,
                grad_norm=norm,
                gradient_clip_norm=config.optimizer.gradient_clip_norm,
                skip_grad_norm_above=settings.skip_grad_norm_above,
                accepted=is_accepted,
                reason=reason,
            )
            if not np.isfinite(value):
                raise FloatingPointError(
                    f"Non-finite probabilistic LSTM loss at epoch {epoch}"
                )
            losses.append(value)
            component_sums += np.asarray(components, dtype=np.float64)
            max_grad_norm = max(max_grad_norm, norm if np.isfinite(norm) else float("inf"))
            skipped += int(not is_accepted)
            batches += 1
        train_metrics = _evaluate(
            model,
            train_eval,
            scalers,
            batch_size=config.optimizer.batch_size,
            num_particles=settings.eval_particles,
            seed=config.seed + epoch * 2,
        )
        test_metrics = _evaluate(
            model,
            test_eval,
            scalers,
            batch_size=config.optimizer.batch_size,
            num_particles=settings.eval_particles,
            seed=config.seed + epoch * 2 + 1,
        )
        metric = test_metrics["total_nrmse"]
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
        components_mean = component_sums / max(batches, 1)
        component_text = " ".join(
            f"loss_{name}={value:.6f}"
            for name, value in zip(component_names, components_mean)
        )
        checkpoint = "checkpoint=best" if improved else f"checkpoint_wait={wait}"
        print(
            f"epoch={epoch:03d} train_loss={np.mean(losses):.6f} "
            f"train_rmse_c={train_metrics['temperature_rmse_c']:.4f} "
            f"train_total_nrmse={train_metrics['total_nrmse']:.4f} "
            f"test_rmse_c={test_metrics['temperature_rmse_c']:.4f} "
            f"test_qroom_rmse_w_m2={test_metrics['qroom_rmse_w_m2']:.4f} "
            f"test_pel_rmse_w_m2={test_metrics['pel_rmse_w_m2']:.4f} "
            f"test_total_nrmse={metric:.4f} {component_text} "
            f"max_grad_norm={max_grad_norm:.3f} skipped_updates={skipped} "
            f"elapsed_min={elapsed_minutes:.2f} {checkpoint}",
            flush=True,
        )
        patience = config.optimizer.early_stopping_patience
        if patience is not None and wait >= patience:
            print(f"early_stopping epoch={epoch} best_epoch={best_epoch}")
            break

    gradient_log.close()
    artifact_dir = Path(config.output_dir) / "artifacts" / spec.name / "selected"
    return save_probabilistic_lstm_artifact(
        artifact_dir,
        model=best_model,
        scalers=scalers,
        spec=spec,
        experiment_config=config,
        input_columns=splits.input_columns,
        metadata_columns=splits.metadata_columns,
        target_columns=CLOSED_LOOP_TARGET_COLUMNS,
        selected_ids=splits.selected_ids,
        train_ids=splits.train_ids,
        test_ids=splits.test_ids,
        checkpoint_epoch=best_epoch,
        checkpoint_metric_value=best_metric,
        train_metrics=best_train_metrics,
        test_metrics=best_test_metrics,
    )
