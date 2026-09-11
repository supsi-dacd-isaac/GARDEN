"""Backend-neutral full-profile prediction."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from neural_building_emulator.data import BuildingProfile, ClosedLoopProfile
from neural_building_emulator.scaling import inverse_target
from neural_building_emulator.train import (
    full_profile_arrays,
    predict_closed_loop_full_profile,
    predict_full_profile,
    predict_probabilistic_closed_loop_full_profile,
    sample_full_profile_scenarios,
    sample_probabilistic_closed_loop_full_profile_scenarios,
)

from .artifacts import LoadedArtifact
from .models import AutoregressiveLSTM
from .profiles import artifact_value


@dataclass(frozen=True)
class PredictionResult:
    profile_id: int
    datetime: np.ndarray
    target: np.ndarray
    mean: np.ndarray
    scenarios: np.ndarray | None


def _lstm_initial_output(
    artifact: LoadedArtifact,
    initial_temperature_c: np.ndarray,
) -> jnp.ndarray:
    target_mean = np.asarray(artifact.scalers.target.mean, dtype=np.float32).reshape(-1)
    target_scale = np.asarray(artifact.scalers.target.scale, dtype=np.float32).reshape(-1)
    temperature = (np.asarray(initial_temperature_c).reshape(-1)[:1] - target_mean[:1]) / target_scale[:1]
    if artifact.model.output_dim == 1:
        return jnp.asarray(temperature)
    zero_powers = -target_mean[1:] / target_scale[1:]
    return jnp.asarray(np.concatenate([temperature, zero_powers]).astype(np.float32))


def _predict_lstm(
    artifact: LoadedArtifact,
    profile: BuildingProfile | ClosedLoopProfile,
) -> PredictionResult:
    model = artifact.model
    if not isinstance(model, AutoregressiveLSTM):
        raise TypeError(f"Expected AutoregressiveLSTM, got {type(model)!r}")
    metadata = artifact.scalers.metadata.transform(profile.metadata)
    if isinstance(profile, BuildingProfile):
        inputs, target, datetime, initial_temperature = full_profile_arrays(profile, "next_step")
    else:
        inputs = profile.inputs
        target = profile.targets
        datetime = profile.datetime
        initial_temperature = profile.initial_temperature[0]
    inputs_scaled = artifact.scalers.inputs.transform(inputs)
    initial = _lstm_initial_output(artifact, np.asarray(initial_temperature))
    normalized = model(jnp.asarray(metadata), jnp.asarray(inputs_scaled), initial)
    prediction = inverse_target(np.asarray(normalized), artifact.scalers)
    return PredictionResult(
        profile_id=profile.profile_id,
        datetime=np.asarray(datetime),
        target=np.asarray(target),
        mean=prediction,
        scenarios=None,
    )


def predict_profile(
    artifact: LoadedArtifact,
    profile: BuildingProfile | ClosedLoopProfile,
    *,
    key: jax.Array | None = None,
    num_scenarios: int = 0,
    num_eval_particles: int | None = None,
    hp_scenario_mode: str = "bernoulli",
    ventilation_rollout_mode: str = "recorded",
) -> PredictionResult:
    """Predict a complete autonomous trajectory from any registered artifact."""
    if artifact.backend == "lstm":
        if num_scenarios > 0:
            raise ValueError("The current LSTM baselines are deterministic and have no scenarios")
        return _predict_lstm(artifact, profile)

    assert artifact.legacy_artifact is not None
    legacy_metadata = artifact.legacy_artifact.metadata
    rng_key = jax.random.PRNGKey(0) if key is None else key
    if artifact.spec.task == "q_to_t":
        if not isinstance(profile, BuildingProfile):
            raise TypeError("Q-to-T artifact requires BuildingProfile")
        alignment = str(legacy_metadata.get("target_alignment", "same_time"))
        target_mode = str(legacy_metadata.get("target_mode", "absolute"))
        model_kind = str(legacy_metadata.get("model_kind", "deterministic"))
        _, target, datetime, _ = full_profile_arrays(profile, alignment)  # type: ignore[arg-type]
        scenarios = None
        if num_scenarios > 0:
            if not artifact.spec.probabilistic:
                raise ValueError("A deterministic Q-to-T artifact has no scenarios")
            scenarios = sample_full_profile_scenarios(
                artifact.model,
                profile,
                artifact.scalers,
                target_mode,  # type: ignore[arg-type]
                target_alignment=alignment,  # type: ignore[arg-type]
                key=rng_key,
                num_particles=num_scenarios,
            )
            mean = np.mean(scenarios, axis=0)
        else:
            mean = predict_full_profile(
                artifact.model,
                profile,
                artifact.scalers,
                target_mode,  # type: ignore[arg-type]
                target_alignment=alignment,  # type: ignore[arg-type]
                model_kind=model_kind,  # type: ignore[arg-type]
                key=rng_key,
                num_particles=(
                    int(num_eval_particles)
                    if num_eval_particles is not None
                    else int(artifact_value(artifact, "prob_eval_particles", 16))
                ),
            )
        return PredictionResult(profile.profile_id, np.asarray(datetime), target, mean, scenarios)

    if not isinstance(profile, ClosedLoopProfile):
        raise TypeError("Closed-loop artifact requires ClosedLoopProfile")
    if not artifact.spec.probabilistic:
        if ventilation_rollout_mode != "recorded":
            raise ValueError(
                "EnergyPlus-rule ventilation is currently implemented for probabilistic "
                "closed-loop artifacts"
            )
        if num_scenarios > 0:
            raise ValueError("A deterministic closed-loop artifact has no scenarios")
        mean = predict_closed_loop_full_profile(
            artifact.model,
            profile,
            artifact.scalers,
        )
        return PredictionResult(
            profile.profile_id,
            np.asarray(profile.datetime),
            np.asarray(profile.targets),
            mean,
            None,
        )
    scenarios = None
    if num_scenarios > 0:
        scenarios = sample_probabilistic_closed_loop_full_profile_scenarios(
            artifact.model,
            profile,
            artifact.scalers,
            key=rng_key,
            num_particles=num_scenarios,
            hp_scenario_mode=hp_scenario_mode,  # type: ignore[arg-type]
            ventilation_rollout_mode=ventilation_rollout_mode,  # type: ignore[arg-type]
            metadata_columns=tuple(legacy_metadata.get("metadata_columns", ())) or None,
        )
        mean = np.mean(scenarios, axis=0)
    else:
        mean = predict_probabilistic_closed_loop_full_profile(
            artifact.model,
            profile,
            artifact.scalers,
            key=rng_key,
            num_particles=(
                int(num_eval_particles)
                if num_eval_particles is not None
                else int(artifact_value(artifact, "prob_eval_particles", 16))
            ),
            ventilation_rollout_mode=ventilation_rollout_mode,  # type: ignore[arg-type]
            metadata_columns=tuple(legacy_metadata.get("metadata_columns", ())) or None,
        )
    return PredictionResult(
        profile.profile_id,
        np.asarray(profile.datetime),
        np.asarray(profile.targets),
        mean,
        scenarios,
    )
