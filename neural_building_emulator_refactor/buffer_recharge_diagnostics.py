"""Event-level evidence for buffer depletion and delayed SH recharging.

Observed DHW modes are diagnostic labels only, never emulator inputs. Events
are selected from the requested policy, independently of their temperature drop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DT = 0.25
TIN = "Zone Air Temperature"
QROOM = "zone_thermal_heating_power"
SETPOINT = "Zone Thermostat Heating Setpoint Temperature"
VENT = "Zone Ventilation Standard Density Volume Flow Rate"
REQUEST = "hp_force_off_requested"
DHW = "hp_mode_is_dhw"
PEL = "heat_pump_electric_power"


def intervals(active: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open intervals, including censored boundary intervals."""
    edges = np.diff(np.r_[False, np.asarray(active, dtype=bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def load_signals(root: Path, building: int) -> tuple[pd.DataFrame, dict]:
    folder = root / f"egid={building}"
    data = pd.read_parquet(folder / "timeseries.parquet")
    metadata = json.loads((folder / "simulation_metadata.json").read_text())
    zone = data[data.zone.str.lower() == "fl0_thz0"].set_index("datetime").sort_index()
    plant = data[data.zone == "technical_room"].set_index("datetime").sort_index()
    building_data = data[data.zone == "building_total"].set_index("datetime").sort_index()
    if zone.empty or not zone.index.is_unique or not zone.index.equals(plant.index):
        raise ValueError(f"Invalid zone/plant alignment for {building}")
    if not np.all(np.diff(zone.index.asi8) == int(DT * 3600 * 1e9)):
        raise ValueError(f"Nonuniform timestamps for {building}")
    values = metadata["gardenCompatibility"]["values"]
    result = zone[[TIN, QROOM, SETPOINT, VENT]].copy()
    result[QROOM] /= float(values["floor_area"])
    for column in (REQUEST, DHW, PEL, "hp_force_off_effective"):
        result[column] = plant[column]
    result["tout"] = building_data["Environment, Site Outdoor Air Drybulb Temperature"]
    demand = data[data.zone.str.lower().str.startswith("fl")].groupby("datetime")[
        "input_dhw_mixed_water_l"
    ].sum(min_count=1)
    result["dhw_l"] = demand
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError(f"Missing event signals for {building}")
    return result, metadata


def drop_features(temperature: np.ndarray, start: int, release: int, end: int) -> dict:
    """Accept either [time] or [particle,time], retaining the leading axes."""
    baseline = np.mean(temperature[..., start - 4:start], axis=-1)
    segment = temperature[..., start:end]
    return {
        "drop_c": np.maximum(baseline - np.min(segment, axis=-1), 0),
        "lockout_drop_c": np.maximum(
            baseline - np.min(temperature[..., start:release], axis=-1), 0
        ),
        "post_release_extra_drop_c": np.maximum(
            temperature[..., release - 1]
            - np.min(temperature[..., release:end], axis=-1), 0
        ),
        "minimum_hour_from_release": (np.argmin(segment, axis=-1) + start - release) * DT,
    }


def event_rows(signals: pd.DataFrame, building: int, split: str) -> list[dict]:
    rows = []
    events = intervals(signals[REQUEST].to_numpy() > 0.5)
    temperature = signals[TIN].to_numpy()
    qroom = signals[QROOM].to_numpy()
    setpoint = signals[SETPOINT].to_numpy()
    mode = signals[DHW].to_numpy() > 0.5
    pel = signals[PEL].to_numpy()
    for event, (start, release) in enumerate(events):
        next_start = events[event + 1][0] if event + 1 < len(events) else len(signals)
        end = min(release + 8, next_start, len(signals))
        if start < 96 or release >= end:
            continue
        pre = slice(start - 8, start)
        after = slice(release, end)
        active_dhw = mode & (pel > 10)
        sh_on = (~mode) & (pel > 10)
        sh_recovery = np.flatnonzero(sh_on[after])
        dhw_prefix = 0
        for value in active_dhw[after]:
            if not value:
                break
            dhw_prefix += 1
        row = {
            "building": building, "split": split, "event": event,
            "start": start, "release": release, "end": end,
            "datetime": signals.index[start].isoformat(),
            "month": signals.index[start].month,
            "duration_h": (release - start) * DT,
            "post_window_h": (end - release) * DT,
            "winter": signals.index[start].month in (11, 12, 1, 2, 3),
            "constant_setpoint": np.ptp(setpoint[start - 8:end]) < 0.05,
            "vent_active_fraction": float(np.mean(signals[VENT].iloc[start:end] > 1e-8)),
            "pre_qroom_w_m2": float(np.mean(qroom[pre])),
            "pre_heat_24h_wh_m2": float(np.sum(qroom[start - 96:start]) * DT),
            "pre_hp_on_fraction": float(np.mean(sh_on[pre])),
            "pre_dhw_l_3h": float(signals.dhw_l.iloc[start - 12:start].sum()),
            "lockout_dhw_l": float(signals.dhw_l.iloc[start:release].sum()),
            "tout_c": float(signals.tout.iloc[pre].mean()),
            "dhw_at_release": bool(active_dhw[release]),
            "dhw_first_hour": bool(np.any(active_dhw[release:min(release + 4, end)])),
            "dhw_post_hours": float(np.sum(active_dhw[after]) * DT),
            "dhw_initial_recovery_h": dhw_prefix * DT,
            "sh_restart_delay_h": float(sh_recovery[0] * DT) if len(sh_recovery) else np.nan,
            "sh_restart_censored": not len(sh_recovery),
            "qroom_min_fraction": float(np.min(qroom[start:end]) / max(np.mean(qroom[pre]), 1)),
            "heat_deficit_wh_m2": float(np.sum(np.maximum(np.mean(qroom[pre]) - qroom[start:end], 0)) * DT),
        }
        row.update({key: float(value) for key, value in drop_features(temperature, start, release, end).items()})
        row["drop_gt_0p5c"] = row["drop_c"] > 0.5
        row["drop_gt_1c"] = row["drop_c"] > 1.0
        rows.append(row)
    return rows


def eligible(events: pd.DataFrame) -> pd.DataFrame:
    return events[events.winter & events.constant_setpoint & (events.post_window_h == 2)].copy()


def summarize(events: pd.DataFrame, output: Path) -> None:
    selected = eligible(events)
    aggregations = {
        "events": ("event", "size"), "buildings": ("building", "nunique"),
        "drop_mean_c": ("drop_c", "mean"), "drop_gt_0p5c": ("drop_gt_0p5c", "mean"),
        "drop_gt_1c": ("drop_gt_1c", "mean"),
        "post_extra_drop_c": ("post_release_extra_drop_c", "mean"),
        "pre_qroom_w_m2": ("pre_qroom_w_m2", "mean"),
        "duration_h": ("duration_h", "mean"),
        "sh_restart_delay_h": ("sh_restart_delay_h", "mean"),
    }
    selected.groupby(["split", "dhw_first_hour"]).agg(**aggregations).to_csv(output / "conditional_summary.csv")
    selected.groupby(["split", "building", "dhw_first_hour"]).agg(**aggregations).to_csv(output / "building_summary.csv")
    # Compare within building, month, duration and preceding-load bin, then
    # bootstrap buildings, not the many correlated events within each building.
    selected["load_bin"] = selected.groupby("building").pre_qroom_w_m2.transform(
        lambda x: pd.qcut(x.rank(method="average"), 3, labels=False, duplicates="drop")
    )
    keys = ["split", "building", "month", "duration_h", "load_bin"]
    differences = []
    for key, group in selected.groupby(keys):
        if group.dhw_first_hour.nunique() != 2:
            continue
        row = dict(zip(keys, key))
        row["events"] = len(group)
        for metric in ("drop_c", "drop_gt_0p5c", "post_release_extra_drop_c"):
            row[metric] = group.loc[group.dhw_first_hour, metric].mean() - group.loc[~group.dhw_first_hour, metric].mean()
        differences.append(row)
    matched = pd.DataFrame(differences)
    matched.to_csv(output / "matched_strata.csv", index=False)
    boot = []
    rng = np.random.default_rng(13)
    if not matched.empty:
        for split, frame in matched.groupby("split"):
            for metric in ("drop_c", "drop_gt_0p5c", "post_release_extra_drop_c"):
                values = frame.groupby("building")[metric].mean().to_numpy()
                draws = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
                boot.append({"split": split, "metric": metric, "buildings": len(values),
                             "difference": values.mean(), "lower95": np.quantile(draws, .025),
                             "upper95": np.quantile(draws, .975)})
    pd.DataFrame(boot).to_csv(output / "matched_building_bootstrap.csv", index=False)


def sample_profile(artifact, profile, key, particles: int, gate: np.ndarray, *, qroom_oracle: bool = False):
    import equinox as eqx
    import jax.numpy as jnp
    from neural_building_emulator.train import _probabilistic_ventilation_rollout_kwargs

    scalers = artifact.scalers
    kwargs = _probabilistic_ventilation_rollout_kwargs(
        profile, mode="eplus_rule", metadata_columns=artifact.legacy_artifact.metadata["metadata_columns"]
    )
    if qroom_oracle:
        kwargs["thermal_qroom_reference"] = jnp.asarray(profile.targets[:, 1])

    @eqx.filter_jit
    def rollout(model, metadata, inputs, initial, rng, availability, ventilation):
        prediction, aux = model.sample_with_aux(
            metadata, inputs, initial, key=rng, num_particles=particles,
            sample_process_noise=True, hp_scenario_mode="bernoulli",
            hp_generation_availability=availability, **ventilation,
        )
        return prediction, aux[2]

    normalized, energy = rollout(
        artifact.model, jnp.asarray(scalers.metadata.transform(profile.metadata)),
        jnp.asarray(scalers.inputs.transform(profile.inputs)),
        jnp.asarray((profile.initial_temperature[0] - scalers.target.mean[:1]) / scalers.target.scale[:1]),
        key, jnp.asarray(gate), kwargs,
    )
    return np.asarray(normalized) * scalers.target.scale + scalers.target.mean, np.asarray(energy)


def ensemble_events(args, events: pd.DataFrame, output: Path) -> None:
    import jax
    from .artifacts import load_artifact
    from .profiles import load_profiles

    artifact = load_artifact(args.artifact)
    rows = []
    selected = eligible(events)
    ids = sorted(selected.loc[selected.split.isin(args.ensemble_splits), "building"].unique())
    if args.max_ensemble_buildings:
        ids = ids[:args.max_ensemble_buildings]
    cache = output / "ensemble_cache"
    cache.mkdir(exist_ok=True)
    for building in ids:
        cached = cache / f"{building}.npz"
        if cached.exists():
            with np.load(cached) as saved:
                predictions, energy = saved["predictions"], saved["energy"]
        else:
            profile = load_profiles(artifact, dataset_path=args.dataset, profile_ids=[int(building)])[0]
            signals, _ = load_signals(args.dataset, int(building))
            # The production loader consumes the first observation as the
            # initial condition; predictions begin at the second raw row.
            if not np.array_equal(profile.datetime.astype("datetime64[ns]"), signals.index.to_numpy()[1:]):
                raise ValueError("Profile and raw timestamps differ")
            predictions, energy = sample_profile(
                artifact, profile, jax.random.fold_in(jax.random.PRNGKey(args.seed), int(building)),
                args.particles,
                (1 - signals[REQUEST].to_numpy(dtype=np.float32)[1:])
                * ((1 - signals[DHW].to_numpy(dtype=np.float32)[1:]) if args.oracle_dhw else 1),
                qroom_oracle=args.oracle_qroom,
            )
            np.savez_compressed(cached, predictions=predictions, energy=energy)
        for _, event in selected[selected.building == building].iterrows():
            start, release, end = (int(event[name]) - 1 for name in ("start", "release", "end"))
            drops = drop_features(predictions[:, :, 0], start, release, end)["drop_c"]
            mean_drop = drop_features(predictions[:, :, 0].mean(axis=0), start, release, end)["drop_c"]
            y = event.drop_c
            crps = np.mean(np.abs(drops - y)) - .5 * np.mean(np.abs(drops[:, None] - drops[None, :]))
            row = {"building": building, "event": event.event, "split": event.split,
                   "observed_drop_c": y, "mean_trace_drop_c": float(mean_drop),
                   "mean_particle_drop_c": drops.mean(), "drop_crps": crps,
                   "drop_lower90": np.quantile(drops, .05), "drop_upper90": np.quantile(drops, .95),
                   "drop_coverage90": np.quantile(drops, .05) <= y <= np.quantile(drops, .95),
                   "drop_pit_mid": np.mean(drops < y) + .5 * np.mean(drops == y),
                   "pre_energy_wh_m2": energy[:, start].mean(),
                   "minimum_energy_wh_m2": energy[:, start:end].min(axis=1).mean(),
                   "fraction_particles_energy_below_1": np.mean(energy[:, start:end].min(axis=1) < 1),
                   "dhw_first_hour": event.dhw_first_hour}
            for threshold in (.5, 1.):
                probability = np.mean(drops > threshold)
                row[f"p_drop_gt_{threshold}"] = probability
                row[f"brier_gt_{threshold}"] = (probability - (y > threshold)) ** 2
            rows.append(row)
        pd.DataFrame(rows).to_csv(output / "ensemble_events.csv", index=False)
        print(f"Ensemble diagnostics {building}: {len(rows)} events", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/control=force_off"))
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--ensemble-splits", nargs="+", default=["test"])
    parser.add_argument("--max-ensemble-buildings", type=int, default=0)
    parser.add_argument("--particles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2013)
    parser.add_argument("--oracle-dhw", action="store_true", help="Diagnostic only: block generation using observed DHW mode")
    parser.add_argument("--oracle-qroom", action="store_true", help="Diagnostic only: supply observed Qroom to the thermal model")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    request = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    request_path = args.output / "request.json"
    if request_path.exists():
        previous = json.loads(request_path.read_text())
        previous.setdefault("oracle_dhw", False)
        previous.setdefault("oracle_qroom", False)
        if previous != request:
            raise ValueError("Use a fresh output directory for changed diagnostic settings")
    request_path.write_text(json.dumps(request, indent=2))
    manifest = json.loads(args.split_manifest.read_text())
    rows = []
    events_path = args.output / "events.csv"
    if events_path.exists():
        events = pd.read_csv(events_path)
    else:
        for split in ("train", "validation", "calibration", "test"):
            for building in manifest[f"{split}_ids"]:
                signals, _ = load_signals(args.dataset, building)
                rows.extend(event_rows(signals, building, split))
            print(f"Read {split}: {len(rows)} cumulative events", flush=True)
        events = pd.DataFrame(rows)
        events.to_csv(events_path, index=False)
    summarize(events, args.output)
    if args.artifact:
        ensemble_events(args, events, args.output)


if __name__ == "__main__":
    main()
