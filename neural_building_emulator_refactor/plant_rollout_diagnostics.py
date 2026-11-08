"""Native annual energy and cooling-event diagnostics for the plant ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .buffer_energy_diagnostics import audit_bucket, source_signals
from .buffer_recharge_diagnostics import DT, DHW, PEL, REQUEST, SETPOINT, TIN, drop_features
from .buffer_recharge_report import summaries


def empirical_crps(samples, truth):
    """Empirical CRPS with particle axis first, without a quadratic pair matrix."""
    samples = np.asarray(samples, dtype=np.float64)
    count = samples.shape[0]
    if count == 0 or not np.all(np.isfinite(samples)) or not np.all(np.isfinite(truth)):
        raise ValueError("CRPS requires finite, nonempty samples and finite truth")
    weights = (2 * np.arange(1, count + 1) - count - 1).reshape((count,) + (1,) * (samples.ndim - 1))
    return np.abs(samples - truth).mean(axis=0) - (weights * np.sort(samples, axis=0)).sum(axis=0) / count**2


def distribution_metrics(samples, truth):
    samples = np.asarray(samples)
    lower, upper = np.quantile(samples, [.05, .95], axis=0)
    return {
        "crps": float(np.mean(empirical_crps(samples, truth))),
        "bias": float(np.mean(samples.mean(axis=0) - truth)),
        "coverage90": float(np.mean((lower <= truth) & (truth <= upper))),
        "width90": float(np.mean(upper - lower)),
        "pit_mid": float(np.mean((samples < truth).mean(axis=0) + .5 * (samples == truth).mean(axis=0))),
    }


def energy_scores(predicted, observed):
    """Score complete 4/24-hour blocks and full-year energy in Wh/m2."""
    rows = {}
    for hours in (4, 24):
        steps = int(hours / DT)
        blocks = len(observed) // steps
        n = blocks * steps
        samples = predicted[:, :n].reshape(predicted.shape[0], blocks, steps).sum(axis=2) * DT
        truth = observed[:n].reshape(blocks, steps).sum(axis=1) * DT
        rows.update({f"{hours}h_{key}": value for key, value in distribution_metrics(samples, truth).items()})
    rows.update({f"annual_{key}": value for key, value in distribution_metrics(
        predicted.sum(axis=1) * DT, observed.sum() * DT).items()})
    return rows


def make_rollout(particles):
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    @eqx.filter_jit
    def rollout(model, metadata, inputs, initial, key, gate, ventilation):
        prediction, aux = model.sample_with_aux(
            metadata, inputs, initial, key=key, num_particles=particles,
            sample_process_noise=True, hp_scenario_mode="bernoulli",
            hp_generation_availability=gate, **ventilation)
        ratios = jax.vmap(lambda xi: model.zone_heat_ratio(
            metadata, model.routed_persistent_xi(xi, "hp")
            if "hp" in model.persistent_xi_pathways else jnp.zeros_like(xi)))(aux[13])
        result = dict(predictions=prediction, cop=aux[1], energy=aux[2], available=aux[3],
                      requested=aux[4], hp_on=aux[5], zone_heat_ratio=ratios,
                      capacity=model.buffer_capacity_wh_m2(metadata))
        if model.dhw_recovery is not None:
            result.update(dhw_probability=aux[-3], dhw_active=aux[-2], dhw_debt=aux[-1])
        return result

    return rollout


def event_metrics(group, result, accounting):
    paths, energy = result["predictions"], result["energy"]
    rows = []
    for event in group.itertuples():
        start, release, end = (int(getattr(event, name)) - 1 for name in ("start", "release", "end"))
        features = drop_features(paths[:, :, 0], start, release, end)
        drops = features["drop_c"]
        row = {
            "building": int(event.building), "event": int(event.event), "split": event.split,
            "observed_drop_c": event.drop_c,
            "mean_trace_drop_c": float(drop_features(paths[:, :, 0].mean(axis=0), start, release, end)["drop_c"]),
            "mean_particle_drop_c": float(drops.mean()),
            "drop_lower90": float(np.quantile(drops, .05)), "drop_upper90": float(np.quantile(drops, .95)),
            **{f"drop_{key}": value for key, value in distribution_metrics(drops, event.drop_c).items()},
            "pre_energy_wh_m2": float(energy[:, start].mean()),
            "minimum_energy_wh_m2": float(energy[:, start:end].min(axis=1).mean()),
            "fraction_particles_energy_below_1": float((energy[:, start:end].min(axis=1) < 1).mean()),
            "storage_limited_fraction": float(accounting["limited"][:, start:end].mean()),
            "dhw_first_hour": event.dhw_first_hour,
            "initially_heating_near_setpoint": event.initially_heating_near_setpoint,
            "observed_minimum_hour_from_release": event.minimum_hour_from_release,
            "predicted_minimum_hour_from_release": float(features["minimum_hour_from_release"].mean()),
        }
        for threshold in (.5, 1.):
            probability = float((drops > threshold).mean())
            row[f"p_drop_gt_{threshold}"] = probability
            row[f"brier_gt_{threshold}"] = (probability - (event.drop_c > threshold))**2
        if "dhw_active" in result:
            active = result["dhw_active"][:, release:release + 4]
            probability = float((active.max(axis=1) > .5).mean())
            row.update(dhw_first_hour_probability=probability,
                       dhw_first_hour_brier=(probability - event.dhw_first_hour)**2,
                       dhw_debt_at_release_wh_m2=float(result["dhw_debt"][:, release].mean()))
        rows.append(row)
    return pd.DataFrame(rows)


def annual_metrics(building, result, signals, accounting):
    paths = result["predictions"]
    generated = result["cop"] * paths[:, :, 2]
    delivered = paths[:, :, 1] / result["zone_heat_ratio"][:, None]
    reference = signals.iloc[1:]
    row = dict(building=int(building), particles=paths.shape[0], capacity_wh_m2=float(result["capacity"]),
               zone_heat_ratio=float(result["zone_heat_ratio"].mean()),
               hp_on_fraction=float(result["hp_on"].mean()),
               overflow_wh_m2=float(accounting["overflow"].sum(axis=1).mean()),
               standing_loss_wh_m2=float(accounting["loss"].sum(axis=1).mean()),
               storage_limited_fraction=float(accounting["limited"].mean()),
               max_transition_residual_wh_m2=float(np.abs(accounting["transition_residual"]).max()),
               max_balance_residual_wh_m2=float(np.abs(accounting["residual"]).max()))
    for name, predicted, observed in (
        ("source", generated, reference.charge_w_m2.to_numpy()),
        ("building_heat", delivered, reference.q_total_w_m2.to_numpy()),
        ("electricity", paths[:, :, 2], reference.pel_sh_w_m2.to_numpy()),
    ):
        row[f"{name}_predicted_wh_m2"] = float(predicted.sum(axis=1).mean() * DT)
        row[f"{name}_observed_wh_m2"] = float(observed.sum() * DT)
        row.update({f"{name}_{key}": value for key, value in energy_scores(predicted, observed).items()})
    if "dhw_active" in result:
        probability = result["dhw_active"].mean(axis=0)
        active_dhw = (reference[DHW].to_numpy() > .5) & (reference[PEL].to_numpy() > 10)
        row.update(dhw_on_fraction=float(probability.mean()),
                   dhw_mode_brier=float(np.mean((probability - active_dhw)**2)))
    return row


def write_trace(path, result, profile, signals, events):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    dates = profile.datetime
    paths = result["predictions"]
    reference = signals.iloc[1:]
    channels = [paths[:, :, 0], paths[:, :, 1], paths[:, :, 2], result["cop"] * paths[:, :, 2], result["energy"]]
    observations = [reference[TIN], profile.targets[:, 1], reference.pel_sh_w_m2, reference.charge_w_m2, None]
    titles = ["Indoor temperature [C]", "Zone heat [W/m2 zone]", "SH electricity [W/m2 building]",
              "SH charging [W/m2 building]", "Usable SH inventory [Wh/m2 building]", "Operating states"]
    figure = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=.035, subplot_titles=titles)
    for row, (samples, truth) in enumerate(zip(channels, observations), start=1):
        lower, upper = np.quantile(samples, [.05, .95], axis=0)
        for values, options in ((lower, {}), (upper, dict(fill="tonexty", fillcolor="rgba(198,40,40,.16)"))):
            figure.add_trace(go.Scatter(x=dates, y=values, line=dict(width=0), showlegend=False,
                                        hoverinfo="skip", **options), row=row, col=1)
        figure.add_trace(go.Scatter(x=dates, y=samples.mean(axis=0), name="Emulator mean / native 90% bands",
                                    line=dict(color="#c62828", width=1), showlegend=row == 1), row=row, col=1)
        if truth is not None:
            figure.add_trace(go.Scatter(x=dates, y=truth, name="EnergyPlus",
                                        line=dict(color="#1565c0", width=1.2), showlegend=row == 1), row=row, col=1)
    figure.add_trace(go.Scatter(x=dates, y=reference[SETPOINT], name="Setpoint",
                                line=dict(color="#666", dash="dot")), row=1, col=1)
    active_dhw = (reference[DHW] > .5) & (reference[PEL] > 10)
    for values, name, color in ((reference[REQUEST], "Requested lockout", "#455a64"),
                                (active_dhw, "Observed active DHW (diagnostic only)", "#ef6c00")):
        figure.add_trace(go.Scatter(x=dates, y=values, name=name, line=dict(color=color)), row=6, col=1)
    if "dhw_active" in result:
        figure.add_trace(go.Scatter(x=dates, y=result["dhw_active"].mean(axis=0), name="Predicted DHW probability",
                                    line=dict(color="#00897b")), row=6, col=1)
    axes = ["xaxis" + (str(i) if i > 1 else "") for i in range(1, 7)]
    buttons = [dict(label="Annual", method="relayout", args=[{f"{axis}.autorange": True for axis in axes}])]
    for event in events.nlargest(3, "drop_c").itertuples():
        bounds = [str(dates[max(0, event.start - 25)]), str(dates[min(len(dates) - 1, event.end + 23)])]
        buttons.append(dict(label=str(event.datetime)[:16], method="relayout",
                            args=[{f"{axis}.range": bounds for axis in axes}]))
    figure.update_layout(title=f"EGID {int(events.building.iloc[0])}: native annual plant diagnostics",
                         height=1450, template="plotly_white", hovermode="x unified",
                         legend=dict(orientation="h", y=-.08), margin=dict(t=110, b=160),
                         updatemenus=[dict(buttons=buttons, x=1, y=1.045, xanchor="right")])
    figure.write_html(path, include_plotlyjs="directory")


def run(args):
    if args.particles < 2:
        raise ValueError("At least two annual particles are required")
    args.output.mkdir(parents=True, exist_ok=True)
    fingerprints = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(args.artifact.iterdir()) if p.suffix in (".eqx", ".npz", ".json")}
    request = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    request.update(artifact_hashes=fingerprints, events_hash=hashlib.sha256(args.events.read_bytes()).hexdigest())
    record = args.output / "request.json"
    if record.exists() and json.loads(record.read_text()) != request:
        raise ValueError("Diagnostic inputs changed; use a fresh output directory")
    record.write_text(json.dumps(request, indent=2, sort_keys=True))

    import jax
    import jax.numpy as jnp
    from neural_building_emulator.train import _probabilistic_ventilation_rollout_kwargs
    from .artifacts import load_artifact
    from .profiles import load_profiles

    artifact = load_artifact(args.artifact)
    events = pd.read_csv(args.events)
    events = events[events.split == "test"]
    if args.profile_ids:
        if not set(args.profile_ids) <= set(events.building):
            raise ValueError("Diagnostic profile IDs must belong to the frozen test event set")
        events = events[events.building.isin(args.profile_ids)]
    cache = args.output / "buildings"
    cache.mkdir(exist_ok=True)
    rollout = make_rollout(args.particles)
    annual, event_frames = [], []
    for building, group in events.groupby("building", sort=True):
        annual_path, event_path = cache / f"{building}_annual.json", cache / f"{building}_events.csv"
        trace_path = args.output / f"profile_{building}.html"
        if annual_path.exists() and event_path.exists() and (building not in args.plot_ids or trace_path.exists()):
            annual.append(json.loads(annual_path.read_text()))
            event_frames.append(pd.read_csv(event_path))
            continue
        signals, _ = source_signals(args.dataset, int(building))
        profile = load_profiles(artifact, dataset_path=args.dataset, profile_ids=[int(building)])[0]
        np.testing.assert_array_equal(profile.datetime.astype("datetime64[ns]"), signals.index.to_numpy()[1:])
        scalers = artifact.scalers
        ventilation = _probabilistic_ventilation_rollout_kwargs(
            profile, mode="eplus_rule", metadata_columns=artifact.legacy_artifact.metadata["metadata_columns"])
        result = {key: np.asarray(value) for key, value in rollout(
            artifact.model, jnp.asarray(scalers.metadata.transform(profile.metadata)),
            jnp.asarray(scalers.inputs.transform(profile.inputs)),
            jnp.asarray((profile.initial_temperature[0] - scalers.target.mean[:1]) / scalers.target.scale[:1]),
            jax.random.fold_in(jax.random.PRNGKey(args.seed), int(building)),
            jnp.asarray(1 - signals[REQUEST].to_numpy(np.float32)[1:]), ventilation).items()}
        result["predictions"] = result["predictions"] * scalers.target.scale + scalers.target.mean
        accounting = audit_bucket(
            result["energy"], result["cop"] * result["predictions"][:, :, 2], result["requested"],
            result["predictions"][:, :, 1], result["available"], float(result["capacity"]),
            zone_heat_ratio=result["zone_heat_ratio"],
            cap_before_delivery=artifact.model.hp_buffer_capacity_mode == "global")
        if np.max(np.abs(accounting["transition_residual"])) > .003:
            raise ValueError(f"Buffer transition accounting failed for {building}")
        row = annual_metrics(building, result, signals, accounting)
        local = event_metrics(group, result, accounting)
        if building in args.plot_ids:
            write_trace(trace_path, result, profile, signals, group)
        annual_path.write_text(json.dumps(row, indent=2, sort_keys=True))
        local.to_csv(event_path, index=False)
        annual.append(row)
        event_frames.append(local)
        pd.DataFrame(annual).to_csv(args.output / "annual_energy.csv", index=False)
        pd.concat(event_frames, ignore_index=True).to_csv(args.output / "ensemble_events.csv", index=False)
        print(f"Audited {building}: source bias {row['source_annual_bias'] / 1000:.3f} kWh/m2; "
              f"event CRPS {local.drop_crps.mean():.4f}", flush=True)
    pd.DataFrame(annual).to_csv(args.output / "annual_energy.csv", index=False)
    all_events = pd.concat(event_frames, ignore_index=True)
    all_events.to_csv(args.output / "ensemble_events.csv", index=False)
    pd.DataFrame([{**row, "subset": name} for name, frame in (
        ("all_eligible", all_events), ("initially_heating", all_events[all_events.initially_heating_near_setpoint]))
        for row in summaries(frame)]).to_csv(args.output / "event_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/control=force_off"))
    parser.add_argument("--events", type=Path, default=Path("output/neural_building_emulator_refactor/buffer_energy_diagnostics_200/events.csv"))
    parser.add_argument("--particles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2013)
    parser.add_argument("--profile-ids", type=int, nargs="+")
    parser.add_argument("--plot-ids", type=int, nargs="+", default=[11152319, 11230131, 191952443])
    run(parser.parse_args())


if __name__ == "__main__":
    main()
