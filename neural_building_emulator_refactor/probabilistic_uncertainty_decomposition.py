"""Factorial uncertainty-source ablation for a fitted closed-loop HP emulator.

The analysis evaluates every combination of persistent ``xi``, thermal process
noise and stochastic HP emission on the same annual profile. It is diagnostic:
the source effects need not be additive because all three channels interact
inside the closed loop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from neural_building_emulator.model_io import _build_model_skeleton, load_training_artifact
from neural_building_emulator.models import ProbabilisticContractingClosedLoopHPEmulator
from neural_building_emulator.scaling import WindowScalers, inverse_target
from neural_building_emulator.train import _probabilistic_ventilation_rollout_kwargs

from .single_building_exact_probabilistic_reproduction import (
    _load_profile,
    _scenario_metrics,
)


class Ablation(NamedTuple):
    name: str
    xi: bool
    process: bool
    hp: bool


ABLATIONS = (
    Ablation("deterministic_center", False, False, False),
    Ablation("xi_only", True, False, False),
    Ablation("process_only", False, True, False),
    Ablation("hp_only", False, False, True),
    Ablation("xi_process", True, True, False),
    Ablation("xi_hp", True, False, True),
    Ablation("process_hp", False, True, True),
    Ablation("full", True, True, True),
)


def _load_local_model(
    path: Path,
    reference_metadata: dict,
    scalers: WindowScalers,
) -> ProbabilisticContractingClosedLoopHPEmulator:
    skeleton = _build_model_skeleton(reference_metadata, scalers)
    model = eqx.tree_deserialise_leaves(path, skeleton)
    if not isinstance(model, ProbabilisticContractingClosedLoopHPEmulator):
        raise TypeError(f"{path} is not a probabilistic contracting closed-loop model")
    return model


def _controlled_scenarios(
    model: ProbabilisticContractingClosedLoopHPEmulator,
    profile,
    scalers: WindowScalers,
    metadata_columns: tuple[str, ...],
    *,
    ablation: Ablation,
    num_scenarios: int,
    seed: int,
    return_hp_aux: bool = False,
) -> np.ndarray | tuple[np.ndarray, tuple[np.ndarray, ...]]:
    metadata = jnp.asarray(scalers.metadata.transform(profile.metadata))
    inputs = jnp.asarray(scalers.inputs.transform(profile.inputs))
    target_mean = np.asarray(scalers.target.mean, dtype=np.float32).reshape(-1)
    target_scale = np.asarray(scalers.target.scale, dtype=np.float32).reshape(-1)
    initial_temperature = jnp.asarray(
        ((profile.initial_temperature[0] - target_mean[:1]) / target_scale[:1]).astype(np.float32)
    )
    horizon = inputs.shape[0]
    key = jax.random.PRNGKey(seed)
    xi_key, process_key, mode_key, power_key, hydronic_key = jax.random.split(key, 5)
    xi_draws = jax.random.normal(xi_key, (num_scenarios, model.latent_dim))
    xi = xi_draws if ablation.xi else jnp.zeros_like(xi_draws)
    zero_xi = jnp.zeros_like(xi_draws)
    enabled_xi_pathways = set(model.persistent_xi_pathways)
    initial_xi = xi if "initial" in enabled_xi_pathways else zero_xi
    thermal_xi = xi if "thermal" in enabled_xi_pathways else zero_xi
    q_response_xi = xi if "q_to_t" in enabled_xi_pathways else zero_xi
    hp_xi = xi if "hp" in enabled_xi_pathways else zero_xi
    process_draws = jax.random.normal(
        process_key,
        (num_scenarios, horizon, model.state_dim),
    )
    process_noise = (
        process_draws
        if ablation.process and model.process_noise_mode != "none"
        else jnp.zeros_like(process_draws)
    )
    hp_mode_uniform = jax.random.uniform(mode_key, (num_scenarios, horizon))
    qroom_mode_uniform = jax.random.uniform(
        jax.random.fold_in(mode_key, 71),
        (num_scenarios, horizon),
    )
    hp_power_noise = jax.random.normal(power_key, (num_scenarios, horizon))
    hp_latent = jnp.zeros((num_scenarios, 0), dtype=inputs.dtype)
    controller_noise = jnp.zeros(
        (num_scenarios, horizon, model.controller_state_dim),
        dtype=inputs.dtype,
    )
    hydronic_innovation = jnp.zeros((num_scenarios, horizon), dtype=inputs.dtype)
    ventilation_kwargs = _probabilistic_ventilation_rollout_kwargs(
        profile,
        mode="eplus_rule",
        metadata_columns=metadata_columns,
    )
    hp_mode = "bernoulli" if ablation.hp else "expected"

    def rollout(
        particle_xi,
        particle_initial_xi,
        particle_thermal_xi,
        particle_q_response_xi,
        particle_hp_xi,
        particle_process,
        particle_mode,
        particle_qroom_mode,
        particle_power,
        particle_hp_latent,
        particle_controller,
        particle_hydronic,
    ):
        predictions, aux = model.rollout_particle(
            metadata,
            inputs,
            initial_temperature,
            particle_xi,
            particle_process,
            particle_mode,
            particle_power,
            particle_hp_latent,
            particle_controller,
            qroom_mode_uniform=particle_qroom_mode,
            hydronic_innovation=particle_hydronic,
            hp_scenario_mode=hp_mode,
            initial_xi=particle_initial_xi,
            thermal_xi=particle_thermal_xi,
            q_response_xi=particle_q_response_xi,
            hp_xi=particle_hp_xi,
            **ventilation_kwargs,
        )
        if return_hp_aux:
            # Keep only scalar HP/buffer diagnostics. The later latent-state
            # arrays are large and are not needed for source attribution.
            return predictions, aux[:10]
        return predictions

    outputs = jax.vmap(rollout)(
        xi,
        initial_xi,
        thermal_xi,
        q_response_xi,
        hp_xi,
        process_noise,
        hp_mode_uniform,
        qroom_mode_uniform,
        hp_power_noise,
        hp_latent,
        controller_noise,
        hydronic_innovation,
    )
    if return_hp_aux:
        predictions, hp_aux = outputs
        return inverse_target(np.asarray(predictions), scalers), tuple(
            np.asarray(values) for values in hp_aux
        )
    return inverse_target(np.asarray(outputs), scalers)


def _write_dashboard(
    results: pd.DataFrame,
    scenarios: dict[tuple[str, str], np.ndarray],
    profile,
    output_dir: Path,
) -> tuple[Path, Path]:
    colors = {"global": "#1565c0", "local": "#c62828"}
    metric_specs = (
        ("temperature_band_mean_width", "Tin mean 95% width [C]"),
        ("winter_temperature_band_mean_width_c", "Winter Tin mean 95% width [C]"),
        ("qroom_band_mean_width", "Qroom mean 95% width [W/m2]"),
        ("pel_band_mean_width", "Pel mean 95% width [W/m2]"),
        ("temperature_coverage95", "Tin empirical 95% coverage"),
        ("temperature_mean_rmse", "Tin ensemble-mean RMSE [C]"),
    )
    figure = make_subplots(rows=2, cols=3, subplot_titles=tuple(label for _, label in metric_specs))
    for panel, (metric, _) in enumerate(metric_specs):
        row, col = divmod(panel, 3)
        for model_name in ("global", "local"):
            selected = results[results["model"] == model_name]
            figure.add_trace(
                go.Bar(
                    x=selected["ablation"],
                    y=selected[metric],
                    name=model_name,
                    legendgroup=model_name,
                    marker_color=colors[model_name],
                    showlegend=panel == 0,
                    hovertemplate=(
                        f"model={model_name}<br>ablation=%{{x}}<br>"
                        f"{metric}=%{{y:.4f}}<extra></extra>"
                    ),
                ),
                row=row + 1,
                col=col + 1,
            )
    figure.update_layout(
        title="Annual uncertainty-source decomposition",
        barmode="group",
        height=900,
        width=1600,
        hoverlabel={"namelength": -1},
    )
    dashboard_path = output_dir / "uncertainty_source_decomposition.html"
    figure.write_html(dashboard_path, include_plotlyjs="cdn")

    selected_names = ("deterministic_center", "xi_only", "process_only", "hp_only", "full")
    traces = make_subplots(
        rows=2,
        cols=len(selected_names),
        shared_xaxes=True,
        subplot_titles=tuple(
            f"{model}: {name}" for model in ("global", "local") for name in selected_names
        ),
        vertical_spacing=0.08,
    )
    for row, model_name in enumerate(("global", "local"), start=1):
        for col, ablation_name in enumerate(selected_names, start=1):
            values = scenarios[(model_name, ablation_name)][:, :, 0]
            lower = np.quantile(values, 0.025, axis=0)
            upper = np.quantile(values, 0.975, axis=0)
            mean = np.mean(values, axis=0)
            traces.add_trace(
                go.Scatter(x=profile.datetime, y=lower, line={"width": 0}, showlegend=False, hoverinfo="skip"),
                row=row,
                col=col,
            )
            traces.add_trace(
                go.Scatter(
                    x=profile.datetime,
                    y=upper,
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor="rgba(198,40,40,0.18)",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            traces.add_trace(
                go.Scattergl(
                    x=profile.datetime,
                    y=profile.targets[:, 0],
                    line={"color": "#1565c0", "width": 1},
                    name="EnergyPlus",
                    showlegend=row == 1 and col == 1,
                ),
                row=row,
                col=col,
            )
            traces.add_trace(
                go.Scattergl(
                    x=profile.datetime,
                    y=mean,
                    line={"color": "#c62828", "width": 1},
                    name="Scenario mean",
                    showlegend=row == 1 and col == 1,
                ),
                row=row,
                col=col,
            )
    traces.update_yaxes(title_text="Tin [C]", col=1)
    traces.update_layout(
        title="Temperature bands by active uncertainty source",
        height=850,
        width=1900,
        hovermode="x unified",
        hoverlabel={"namelength": -1},
    )
    traces_path = output_dir / "temperature_uncertainty_source_traces.html"
    traces.write_html(traces_path, include_plotlyjs="cdn")
    return dashboard_path, traces_path


def run(args: argparse.Namespace) -> None:
    artifact = load_training_artifact(args.artifact_dir)
    global_model = artifact.model
    if not isinstance(global_model, ProbabilisticContractingClosedLoopHPEmulator):
        raise TypeError("Artifact is not a probabilistic contracting closed-loop model")
    if global_model.qroom_activation_model != "continuous":
        raise ValueError("This three-source decomposition expects continuous Qroom activation")
    if global_model.hydronic_innovation_enabled:
        raise ValueError("Disable hydronic innovation or add it as a fourth factorial source")
    metadata_columns = tuple(artifact.metadata["metadata_columns"])
    profile = _load_profile(
        args.force_off_dataset,
        args.profile_id,
        policy="force_off",
        metadata_columns=metadata_columns,
    ).profile
    models = {"global": global_model}
    if args.local_model is not None:
        models["local"] = _load_local_model(
            args.local_model,
            artifact.metadata,
            artifact.scalers,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenario_sets: dict[tuple[str, str], np.ndarray] = {}
    rows: list[dict[str, float | str | bool]] = []
    for model_index, (model_name, model) in enumerate(models.items()):
        for ablation_index, ablation in enumerate(ABLATIONS):
            values = _controlled_scenarios(
                model,
                profile,
                artifact.scalers,
                metadata_columns,
                ablation=ablation,
                num_scenarios=args.num_scenarios,
                seed=args.seed + 10_000 * model_index,
            )
            scenario_sets[(model_name, ablation.name)] = values
            metrics = _scenario_metrics(values, profile.targets, profile.datetime)
            rows.append(
                {
                    "model": model_name,
                    "ablation": ablation.name,
                    "xi": ablation.xi,
                    "process": ablation.process,
                    "hp": ablation.hp,
                    **metrics,
                }
            )
            print(
                f"model={model_name} ablation={ablation.name} "
                f"Tin_width={metrics['temperature_band_mean_width']:.3f} "
                f"winter_width={metrics['winter_temperature_band_mean_width_c']:.3f} "
                f"Tin_rmse={metrics['temperature_mean_rmse']:.3f}",
                flush=True,
            )
    results = pd.DataFrame(rows)
    csv_path = args.output_dir / "uncertainty_source_decomposition.csv"
    json_path = args.output_dir / "uncertainty_source_decomposition.json"
    results.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2))
    dashboard_path, traces_path = _write_dashboard(
        results,
        scenario_sets,
        profile,
        args.output_dir,
    )
    print(f"saved_csv={csv_path}")
    print(f"saved_dashboard={dashboard_path}")
    print(f"saved_traces={traces_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(
            "output/neural_building_emulator_refactor/"
            "calibrated_four_model_benchmark_force_off_debt_1000/models/"
            "probabilistic_structured/artifacts/"
            "closed_loop_hp_contracting_probabilistic/selected"
        ),
    )
    parser.add_argument(
        "--local-model",
        type=Path,
        default=Path(
            "output/neural_building_emulator_refactor/exact_local_reproduction_11140124/"
            "exact_local_model.eqx"
        ),
    )
    parser.add_argument("--force-off-dataset", type=Path, default=Path("data/control=force_off"))
    parser.add_argument("--profile-id", type=int, default=11140124)
    parser.add_argument("--num-scenarios", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20015)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "output/neural_building_emulator_refactor/exact_local_reproduction_11140124/"
            "uncertainty_decomposition"
        ),
    )
    args = parser.parse_args()
    for name in ("artifact_dir", "force_off_dataset"):
        if not getattr(args, name).exists():
            parser.error(f"--{name.replace('_', '-')} does not exist: {getattr(args, name)}")
    if args.local_model is not None and not args.local_model.exists():
        parser.error(f"--local-model does not exist: {args.local_model}")
    if args.num_scenarios < 2:
        parser.error("--num-scenarios must be at least 2")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
