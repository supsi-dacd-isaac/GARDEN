"""Backend-neutral full-profile prediction."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from neural_building_emulator.columns import DISTURBANCE_INPUT_COLUMNS
from neural_building_emulator.data import BuildingProfile, ClosedLoopProfile
from neural_building_emulator.scaling import inverse_target
from neural_building_emulator.ventilation import (
    EPlusVentilationRollout,
    build_eplus_ventilation_schedule,
)
from neural_building_emulator.train import (
    full_profile_arrays,
    predict_closed_loop_full_profile,
    predict_full_profile,
    predict_probabilistic_closed_loop_full_profile,
    sample_full_profile_scenarios,
    sample_probabilistic_closed_loop_full_profile_scenarios,
)

from .artifacts import LoadedArtifact
from .models import (
    AutoregressiveLSTM,
    ProbabilisticAutoregressiveLSTM,
    VanillaProbabilisticAutoregressiveLSTM,
)
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
    *,
    ventilation_rollout: EPlusVentilationRollout | None = None,
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
    normalized = model(
        jnp.asarray(metadata), jnp.asarray(inputs_scaled), initial,
        ventilation_rollout=ventilation_rollout,
    )
    prediction = inverse_target(np.asarray(normalized), artifact.scalers)
    return PredictionResult(
        profile_id=profile.profile_id,
        datetime=np.asarray(datetime),
        target=np.asarray(target),
        mean=prediction,
        scenarios=None,
    )


def _predict_probabilistic_lstm(
    artifact: LoadedArtifact,
    profile: BuildingProfile | ClosedLoopProfile,
    *,
    key: jax.Array,
    num_scenarios: int,
    num_eval_particles: int | None,
    hp_scenario_mode: str,
    ventilation_rollout: EPlusVentilationRollout | None = None,
) -> PredictionResult:
    model = artifact.model
    if not isinstance(
        model,
        (ProbabilisticAutoregressiveLSTM, VanillaProbabilisticAutoregressiveLSTM),
    ):
        raise TypeError(
            f"Expected a probabilistic autoregressive LSTM, got {type(model)!r}"
        )
    if not isinstance(profile, ClosedLoopProfile):
        raise TypeError("probabilistic closed-loop LSTM requires ClosedLoopProfile")
    metadata = artifact.scalers.metadata.transform(profile.metadata)
    inputs_scaled = artifact.scalers.inputs.transform(profile.inputs)
    initial = _lstm_initial_output(
        artifact, np.asarray(profile.initial_temperature[0])
    )
    settings = artifact.metadata["probabilistic_lstm_config"]
    particle_count = (
        num_scenarios
        if num_scenarios > 0
        else (
            int(num_eval_particles)
            if num_eval_particles is not None
            else int(settings["eval_particles"])
        )
    )
    scenario_mode = hp_scenario_mode if num_scenarios > 0 else "expected"
    if ventilation_rollout is not None and not isinstance(
        model, VanillaProbabilisticAutoregressiveLSTM
    ):
        raise ValueError("EnergyPlus-rule ventilation requires the vanilla probabilistic LSTM")
    normalized = model.sample(
        jnp.asarray(metadata),
        jnp.asarray(inputs_scaled),
        initial,
        key=key,
        num_particles=particle_count,
        sample_process_noise=True,
        hp_scenario_mode=scenario_mode,
        **({"ventilation_rollout": ventilation_rollout} if ventilation_rollout is not None else {}),
    )
    particles = np.asarray(normalized) * artifact.scalers.target.scale + artifact.scalers.target.mean
    scenarios = particles if num_scenarios > 0 else None
    return PredictionResult(
        profile_id=profile.profile_id,
        datetime=np.asarray(profile.datetime),
        target=np.asarray(profile.targets),
        mean=np.mean(particles, axis=0),
        scenarios=scenarios,
    )


def _lstm_ventilation_rollout(
    artifact: LoadedArtifact,
    profile: ClosedLoopProfile,
) -> EPlusVentilationRollout:
    metadata_columns = tuple(artifact.metadata["metadata_columns"])
    input_columns = tuple(artifact.metadata["input_columns"])
    if len(metadata_columns) != profile.metadata.shape[-1]:
        raise ValueError("Artifact metadata_columns do not match the profile")
    schedule = build_eplus_ventilation_schedule(
        profile.datetime, profile.metadata, metadata_columns,
    )
    return EPlusVentilationRollout(
        schedule=schedule,
        outdoor_input_index=input_columns.index(DISTURBANCE_INPUT_COLUMNS[0]),
        ventilation_input_index=input_columns.index(DISTURBANCE_INPUT_COLUMNS[2]),
        input_mean=tuple(float(value) for value in artifact.scalers.inputs.mean),
        input_scale=tuple(float(value) for value in artifact.scalers.inputs.scale),
        temperature_mean_c=float(np.asarray(artifact.scalers.target.mean).reshape(-1)[0]),
        temperature_scale_c=float(np.asarray(artifact.scalers.target.scale).reshape(-1)[0]),
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
    thermal_qroom_reference: np.ndarray | None = None,
    hp_generation_availability: np.ndarray | None = None,
) -> PredictionResult:
    """Predict a complete autonomous trajectory from any registered artifact."""
    if artifact.backend == "lstm":
        if hp_generation_availability is not None:
            raise ValueError(
                "The external HP generation gate is intentionally unavailable for "
                "unstructured LSTM artifacts"
            )
        rng_key = jax.random.PRNGKey(0) if key is None else key
        if ventilation_rollout_mode not in ("recorded", "eplus_rule"):
            raise ValueError("ventilation_rollout_mode must be 'recorded' or 'eplus_rule'")
        ventilation_rollout = None
        if ventilation_rollout_mode == "eplus_rule":
            if not isinstance(profile, ClosedLoopProfile):
                raise ValueError("EnergyPlus-rule ventilation requires a closed-loop profile")
            ventilation_rollout = _lstm_ventilation_rollout(artifact, profile)
        if artifact.spec.probabilistic:
            return _predict_probabilistic_lstm(
                artifact,
                profile,
                key=rng_key,
                num_scenarios=num_scenarios,
                num_eval_particles=num_eval_particles,
                hp_scenario_mode=hp_scenario_mode,
                ventilation_rollout=ventilation_rollout,
            )
        if num_scenarios > 0:
            raise ValueError("The deterministic LSTM baselines have no scenarios")
        return _predict_lstm(artifact, profile, ventilation_rollout=ventilation_rollout)

    assert artifact.legacy_artifact is not None
    legacy_metadata = artifact.legacy_artifact.metadata
    rng_key = jax.random.PRNGKey(0) if key is None else key
    if artifact.spec.task == "q_to_t":
        if hp_generation_availability is not None:
            raise ValueError("HP generation gating requires a closed-loop HP artifact")
        if thermal_qroom_reference is not None:
            raise ValueError("The Qroom thermal oracle requires a contracting closed-loop HP model")
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
    if thermal_qroom_reference is not None and (not artifact.spec.probabilistic or num_scenarios <= 0):
        raise ValueError("The Qroom thermal oracle requires probabilistic closed-loop scenarios")
    if not artifact.spec.probabilistic:
        if hp_generation_availability is not None:
            raise ValueError(
                "HP generation gating is currently implemented for the probabilistic "
                "contracting closed-loop artifact"
            )
        if num_scenarios > 0:
            raise ValueError("A deterministic closed-loop artifact has no scenarios")
        mean = predict_closed_loop_full_profile(
            artifact.model,
            profile,
            artifact.scalers,
            ventilation_rollout_mode=ventilation_rollout_mode,  # type: ignore[arg-type]
            metadata_columns=tuple(legacy_metadata.get("metadata_columns", ())) or None,
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
            thermal_qroom_reference=thermal_qroom_reference,
            hp_generation_availability=hp_generation_availability,
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
            hp_generation_availability=hp_generation_availability,
        )
    return PredictionResult(
        profile.profile_id,
        np.asarray(profile.datetime),
        np.asarray(profile.targets),
        mean,
        scenarios,
    )
