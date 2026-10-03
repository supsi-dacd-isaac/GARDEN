"""Decompose persistent-xi uncertainty across model pathways.

All transient noise and hard HP sampling are disabled. The same sampled latent
draw is selectively routed through initial-state, thermal-transition, positive
Q-to-T, and HP/controller/buffer pathways. The complete 2^4 factorial exposes
nonlinear interactions between these pathways.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from neural_building_emulator.model_io import load_training_artifact
from neural_building_emulator.models import ProbabilisticContractingClosedLoopHPEmulator
from neural_building_emulator.scaling import WindowScalers, inverse_target
from neural_building_emulator.train import _probabilistic_ventilation_rollout_kwargs

from .conditional_force_off_uncertainty import _conditional_metrics, _regimes
from .probabilistic_uncertainty_decomposition import _load_local_model
from .single_building_exact_probabilistic_reproduction import _load_profile


class Pathways(NamedTuple):
    initial: bool
    thermal: bool
    q_to_t: bool
    hp: bool

    @property
    def name(self) -> str:
        names = [
            name
            for name, active in zip(
                ("initial", "thermal", "q_to_t", "hp"),
                self,
            )
            if active
        ]
        return "+".join(names) if names else "none"

    @property
    def count(self) -> int:
        return sum(self)


PATHWAY_CONFIGS = tuple(
    sorted(
        (Pathways(*values) for values in itertools.product((False, True), repeat=4)),
        key=lambda value: (value.count, value.name),
    )
)


def _pathway_scenarios(
    model: ProbabilisticContractingClosedLoopHPEmulator,
    profile,
    scalers: WindowScalers,
    metadata_columns: tuple[str, ...],
    *,
    pathways: Pathways,
    num_scenarios: int,
    seed: int,
) -> np.ndarray:
    metadata = jnp.asarray(scalers.metadata.transform(profile.metadata))
    inputs = jnp.asarray(scalers.inputs.transform(profile.inputs))
    target_mean = np.asarray(scalers.target.mean, dtype=np.float32).reshape(-1)
    target_scale = np.asarray(scalers.target.scale, dtype=np.float32).reshape(-1)
    initial_temperature = jnp.asarray(
        ((profile.initial_temperature[0] - target_mean[:1]) / target_scale[:1]).astype(np.float32)
    )
    horizon = inputs.shape[0]
    key = jax.random.PRNGKey(seed)
    xi_key, mode_key = jax.random.split(key)
    xi = jax.random.normal(xi_key, (num_scenarios, model.latent_dim))
    zero_xi = jnp.zeros_like(xi)
    process_noise = jnp.zeros(
        (num_scenarios, horizon, model.state_dim), dtype=inputs.dtype
    )
    hp_mode_uniform = jax.random.uniform(mode_key, (num_scenarios, horizon))
    qroom_mode_uniform = jnp.mod(hp_mode_uniform + 0.5, 1.0)
    hp_power_noise = jnp.zeros((num_scenarios, horizon), dtype=inputs.dtype)
    hp_latent = jnp.zeros((num_scenarios, 0), dtype=inputs.dtype)
    controller_noise = jnp.zeros(
        (num_scenarios, horizon, model.controller_state_dim), dtype=inputs.dtype
    )
    hydronic_innovation = jnp.zeros((num_scenarios, horizon), dtype=inputs.dtype)
    ventilation_kwargs = _probabilistic_ventilation_rollout_kwargs(
        profile,
        mode="eplus_rule",
        metadata_columns=metadata_columns,
    )

    initial_xi = xi if pathways.initial else zero_xi
    thermal_xi = xi if pathways.thermal else zero_xi
    q_response_xi = xi if pathways.q_to_t else zero_xi
    hp_xi = xi if pathways.hp else zero_xi
    predictions = jax.vmap(
        lambda base_xi, particle_initial_xi, particle_thermal_xi, particle_q_xi, particle_hp_xi, particle_process, particle_mode, particle_qroom_mode, particle_power, particle_hp_latent, particle_controller, particle_hydronic: model.rollout_particle(
            metadata,
            inputs,
            initial_temperature,
            base_xi,
            particle_process,
            particle_mode,
            particle_power,
            particle_hp_latent,
            particle_controller,
            qroom_mode_uniform=particle_qroom_mode,
            hydronic_innovation=particle_hydronic,
            hp_scenario_mode="expected",
            initial_xi=particle_initial_xi,
            thermal_xi=particle_thermal_xi,
            q_response_xi=particle_q_xi,
            hp_xi=particle_hp_xi,
            **ventilation_kwargs,
        )[0]
    )(
        zero_xi,
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
    return inverse_target(np.asarray(predictions), scalers)


def _assert_shared_xi_equivalence(
    model: ProbabilisticContractingClosedLoopHPEmulator,
    profile,
    scalers: WindowScalers,
) -> float:
    horizon = min(96, len(profile.inputs))
    metadata = jnp.asarray(scalers.metadata.transform(profile.metadata))
    inputs = jnp.asarray(scalers.inputs.transform(profile.inputs[:horizon]))
    target_mean = np.asarray(scalers.target.mean, dtype=np.float32).reshape(-1)
    target_scale = np.asarray(scalers.target.scale, dtype=np.float32).reshape(-1)
    initial_temperature = jnp.asarray(
        ((profile.initial_temperature[0] - target_mean[:1]) / target_scale[:1]).astype(np.float32)
    )
    xi = jax.random.normal(jax.random.PRNGKey(991), (model.latent_dim,))
    process = jnp.zeros((horizon, model.state_dim), dtype=inputs.dtype)
    mode = jnp.linspace(0.01, 0.99, horizon, dtype=inputs.dtype)
    power = jnp.zeros((horizon,), dtype=inputs.dtype)
    hp_latent = jnp.zeros((0,), dtype=inputs.dtype)
    controller = jnp.zeros((horizon, model.controller_state_dim), dtype=inputs.dtype)
    baseline = model.rollout_particle(
        metadata,
        inputs,
        initial_temperature,
        xi,
        process,
        mode,
        power,
        hp_latent,
        controller,
        hp_scenario_mode="expected",
    )[0]
    explicit = model.rollout_particle(
        metadata,
        inputs,
        initial_temperature,
        xi,
        process,
        mode,
        power,
        hp_latent,
        controller,
        hp_scenario_mode="expected",
        initial_xi=xi,
        thermal_xi=xi,
        q_response_xi=xi,
        hp_xi=xi,
    )[0]
    max_difference = float(jnp.max(jnp.abs(baseline - explicit)))
    if max_difference > 1e-7:
        raise AssertionError(
            f"Pathway hooks changed the default rollout: max difference {max_difference}"
        )
    return max_difference


def _write_dashboard(results: pd.DataFrame, output_dir: Path) -> Path:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Global: winter temperature width",
            "Local: winter temperature width",
            "Global: winter ensemble-mean RMSE",
            "Full-xi policy and seasonal comparison",
        ),
        vertical_spacing=0.14,
    )
    colors = {"setpoint": "#1565c0", "force_off": "#c62828"}
    for col, model_name in enumerate(("global", "local"), start=1):
        selected = results[
            (results["model"] == model_name) & (results["regime"] == "winter")
        ]
        for policy in ("setpoint", "force_off"):
            values = selected[selected["policy"] == policy]
            figure.add_trace(
                go.Bar(
                    x=values["pathways"],
                    y=values["temperature_width_mean"],
                    name=policy,
                    legendgroup=policy,
                    marker_color=colors[policy],
                    showlegend=col == 1,
                    hovertemplate=(
                        f"model={model_name}<br>policy={policy}<br>pathways=%{{x}}"
                        "<br>winter width=%{y:.3f} C<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
    global_winter = results[
        (results["model"] == "global") & (results["regime"] == "winter")
    ]
    for policy in ("setpoint", "force_off"):
        values = global_winter[global_winter["policy"] == policy]
        figure.add_trace(
            go.Bar(
                x=values["pathways"],
                y=values["temperature_mean_rmse"],
                name=f"{policy} RMSE",
                marker_color=colors[policy],
                showlegend=False,
                hovertemplate=(
                    f"policy={policy}<br>pathways=%{{x}}"
                    "<br>winter RMSE=%{y:.3f} C<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
    complete = results[
        (results["model"] == "global")
        & (results["pathways"] == "initial+thermal+q_to_t+hp")
        & (results["regime"].isin(("winter", "non_winter")))
    ]
    for policy in ("setpoint", "force_off"):
        values = complete[complete["policy"] == policy]
        figure.add_trace(
            go.Bar(
                x=values["regime"],
                y=values["temperature_width_mean"],
                name=f"{policy} all xi",
                marker_color=colors[policy],
                showlegend=False,
                hovertemplate=(
                    f"policy={policy}<br>regime=%{{x}}"
                    "<br>width=%{y:.3f} C<extra></extra>"
                ),
            ),
            row=2,
            col=2,
        )
    figure.update_yaxes(title_text="Mean 95% width [C]", row=1)
    figure.update_yaxes(title_text="RMSE [C]", row=2, col=1)
    figure.update_yaxes(title_text="Mean 95% width [C]", row=2, col=2)
    figure.update_layout(
        title="Persistent latent pathway decomposition",
        barmode="group",
        height=1050,
        width=1900,
        hoverlabel={"namelength": -1},
    )
    path = output_dir / "xi_pathway_decomposition.html"
    figure.write_html(path, include_plotlyjs="cdn")
    return path


def run(args: argparse.Namespace) -> None:
    artifact = load_training_artifact(args.artifact_dir)
    if not isinstance(artifact.model, ProbabilisticContractingClosedLoopHPEmulator):
        raise TypeError("Artifact is not a probabilistic contracting closed-loop model")
    metadata_columns = tuple(artifact.metadata["metadata_columns"])
    input_columns = tuple(artifact.metadata["input_columns"])
    profiles = {
        "setpoint": _load_profile(
            args.setpoint_dataset,
            args.profile_id,
            policy="setpoint",
            metadata_columns=metadata_columns,
        ).profile,
        "force_off": _load_profile(
            args.force_off_dataset,
            args.profile_id,
            policy="force_off",
            metadata_columns=metadata_columns,
        ).profile,
    }
    models = {"global": artifact.model}
    if args.local_model is not None:
        models["local"] = _load_local_model(
            args.local_model,
            artifact.metadata,
            artifact.scalers,
        )
    equivalence = {
        name: _assert_shared_xi_equivalence(model, profiles["force_off"], artifact.scalers)
        for name, model in models.items()
    }
    print(f"default_pathway_equivalence={equivalence}", flush=True)

    rows: list[dict] = []
    for model_index, (model_name, model) in enumerate(models.items()):
        for policy_index, (policy, profile) in enumerate(profiles.items()):
            regimes = _regimes(profile, input_columns)
            for pathways in PATHWAY_CONFIGS:
                scenarios = _pathway_scenarios(
                    model,
                    profile,
                    artifact.scalers,
                    metadata_columns,
                    pathways=pathways,
                    num_scenarios=args.num_scenarios,
                    seed=args.seed + model_index * 10_000 + policy_index * 1_000,
                )
                for regime in ("all", "winter", "non_winter", "heating_available"):
                    rows.append(
                        {
                            "model": model_name,
                            "policy": policy,
                            "pathways": pathways.name,
                            "pathway_count": pathways.count,
                            "initial": pathways.initial,
                            "thermal": pathways.thermal,
                            "q_to_t": pathways.q_to_t,
                            "hp": pathways.hp,
                            "regime": regime,
                            **_conditional_metrics(
                                scenarios,
                                profile.targets,
                                regimes[regime],
                            ),
                        }
                    )
                winter = rows[-3]
                print(
                    f"model={model_name} policy={policy} pathways={pathways.name} "
                    f"winter_width={winter['temperature_width_mean']:.3f} "
                    f"winter_rmse={winter['temperature_mean_rmse']:.3f}",
                    flush=True,
                )
    results = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "xi_pathway_decomposition.csv"
    json_path = args.output_dir / "xi_pathway_decomposition.json"
    results.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "profile_id": args.profile_id,
                "num_scenarios": args.num_scenarios,
                "default_pathway_equivalence_max_abs": equivalence,
                "results": results.to_dict(orient="records"),
            },
            indent=2,
        )
    )
    dashboard_path = _write_dashboard(results, args.output_dir)
    print(f"saved_results={csv_path}")
    print(f"saved_dashboard={dashboard_path}")


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
    parser.add_argument(
        "--setpoint-dataset",
        type=Path,
        default=Path("data/control=setpoint_identification_dhw_fixed"),
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
            "xi_pathway_decomposition"
        ),
    )
    args = parser.parse_args()
    for name in ("artifact_dir", "setpoint_dataset", "force_off_dataset"):
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
