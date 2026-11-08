"""Autonomous annual tank traces and native interval calibration diagnostics."""

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from neural_building_emulator.columns import TANK_TEMPERATURE_COLUMNS
from neural_building_emulator.dataset_adapter import read_entity_dataset
from neural_building_emulator.train import _probabilistic_ventilation_rollout_kwargs
from .artifacts import load_artifact
from .profiles import load_profiles


def plot(path, dates, samples, truth, building, policy):
    labels = ("Indoor temperature (C)", "Delivered room heat (W/m2)", "HP SH power (W/m2)",
              "SH tank temperature (C)", "DHW tank temperature (C)")
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=.05, subplot_titles=labels)
    lower, upper = np.quantile(samples, [.025, .975], axis=0)
    mean = samples.mean(axis=0)
    for c in range(5):
        for y, name, color, fill in [(lower[:, c], "Lower", "rgba(200,50,50,0)", None),
            (upper[:, c], "95% ensemble interval", "rgba(200,50,50,0)", "tonexty"),
            (truth[:, c], "EnergyPlus", "#176dad", None), (mean[:, c], "Emulator mean", "#cc3636", None)]:
            fig.add_trace(go.Scatter(x=dates, y=y, name=name, line=dict(color=color, width=1.5),
                fill=fill, fillcolor="rgba(200,50,50,.15)", showlegend=c == 0 and name != "Lower",
                legendgroup=name, hoverinfo="skip" if name == "Lower" else "x+y+name"), row=c+1, col=1)
    fig.update_layout(title=dict(text=f"Building {building}<br><sup>{policy.replace('_', ' ')}</sup>", font_size=16,
                                y=.99, yanchor="top", x=.03), height=1300,
        template="plotly_white", margin=dict(l=65, r=20, t=160, b=45),
        legend=dict(orientation="h", y=1.01, yanchor="bottom", x=0), hovermode="x unified")
    fig.write_html(path, include_plotlyjs="directory", config=dict(responsive=True, displaylogo=False))


def metrics(samples, truth, mask):
    x = samples[:, mask]
    y = truth[mask]
    count = x.shape[0]
    low, high = np.quantile(x, [.025, .975], axis=0)
    ordered = np.sort(x, axis=0)
    correction = np.sum((2*np.arange(count)-count+1)[:, None, None] * ordered, axis=0) / count**2
    crps = np.mean(np.abs(x-y[None]), axis=0) - correction
    return dict(rmse=np.sqrt(np.mean((x.mean(axis=0)-y)**2, axis=0)),
        bias=np.mean(x.mean(axis=0)-y, axis=0), coverage_95=np.mean((y>=low)&(y<=high), axis=0),
        width_95=np.mean(high-low, axis=0), crps=np.mean(crps, axis=0))


def run(args, recipe, split, artifact_path):
    artifact = load_artifact(artifact_path)
    if artifact.model.tank_temperatures is None:
        raise ValueError("Tank diagnostics require the two-state model")
    destination = args.output_dir / "variants" / args.variant / "tank_diagnostics" / args.policy
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "complete.json").exists():
        return
    dataset = args.campaign_root.resolve() / "data" / ("control="+args.policy)
    metadata_cols = artifact.legacy_artifact.metadata["metadata_columns"]
    mean = np.asarray(artifact.scalers.target.mean).reshape(-1)
    scale = np.asarray(artifact.scalers.target.scale).reshape(-1)

    @eqx.filter_jit
    def sample(model, metadata, inputs, initial, key, ventilation, count):
        output, aux = model.sample_with_aux(metadata, inputs, initial, key=key, num_particles=count,
            sample_process_noise=True, hp_scenario_mode="bernoulli", **ventilation)
        return output, aux[-1]

    rows = []
    for building in split.test_ids:
        profile, = load_profiles(artifact, dataset_path=dataset, profile_ids=[building])
        observed = read_entity_dataset(dataset, columns=["datetime", *TANK_TEMPERATURE_COLUMNS], profile_ids=[building])
        observed = observed.sort_values("datetime").iloc[1:]
        np.testing.assert_array_equal(observed.datetime.to_numpy(), profile.datetime)
        truth = np.column_stack([profile.targets, observed[TANK_TEMPERATURE_COLUMNS].to_numpy()])
        u = jnp.asarray(artifact.scalers.inputs.transform(profile.inputs))
        m = jnp.asarray(artifact.scalers.metadata.transform(profile.metadata))
        initial = jnp.asarray((profile.initial_temperature[0] - mean[:1]) / scale[:1])
        ventilation = _probabilistic_ventilation_rollout_kwargs(profile, mode="eplus_rule", metadata_columns=metadata_cols)
        particles = recipe["evaluation"]["num_scenarios"]
        batches = []
        seed = recipe["seed"] + (2000 if "force_off" in args.policy else 1000)
        key = jax.random.fold_in(jax.random.PRNGKey(seed), int(building))
        for batch, start in enumerate(range(0, particles, 25)):
            pred, tanks = sample(artifact.model, m, u, initial, jax.random.fold_in(key, batch),
                                 ventilation, min(25, particles-start))
            batches.append(np.concatenate([np.asarray(pred)*scale+mean, np.asarray(tanks)], axis=-1))
        scenarios = np.concatenate(batches)
        if not np.isfinite(scenarios).all():
            raise ValueError(f"Nonfinite annual tank rollout: {building}")
        input_columns = artifact.legacy_artifact.metadata["input_columns"]
        heating = profile.inputs[:, input_columns.index("space_heating_available")] > .5
        for regime, mask in (("all", np.ones(len(u), bool)), ("heating", heating)):
            scores = metrics(scenarios, truth, mask)
            for c, channel in enumerate(("Tin", "Qroom", "Pel_SH", "T_SH", "T_DHW")):
                rows.append(dict(building=building, policy=args.policy, regime=regime, channel=channel,
                                 **{k:float(v[c]) for k,v in scores.items()},
                                 fraction_near_tank_bound=(float(np.mean((scenarios[:, mask, c] <= 1.01)
                                     | (scenarios[:, mask, c] >= 89.99))) if c >= 3 else np.nan)))
        if building in split.test_ids[:3]:
            plot(destination / f"profile_{building}.html", profile.datetime, scenarios, truth, building, args.policy)
        print(f"Tank diagnostics: {args.policy} building={building}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "per_building.csv", index=False)
    frame.groupby(["regime", "channel"])[["rmse", "bias", "coverage_95", "width_95", "crps"]].mean().to_csv(destination / "summary.csv")
    (destination / "complete.json").write_text(json.dumps(dict(artifact=str(artifact_path), policy=args.policy,
        profile_ids=split.test_ids, particles=recipe["evaluation"]["num_scenarios"],
        note="New matched-key Monte Carlo diagnostic draw; only initial Tin observed, both tank states predicted. Interval-mean tank observation model."), indent=2))
