"""Audit buffer accounting, observed withdrawal, and predictable cooling risk.

This module does not train or mutate the production emulator. Counterfactual
bucket replays hold its requested delivery fixed, so are not closed-loop forecasts.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .buffer_recharge_diagnostics import DT, DHW, PEL, QROOM, REQUEST, SETPOINT, TIN, VENT, eligible, load_signals


def audit_bucket(energy, generated, requested, delivered, available, capacity, *,
                 zone_heat_ratio=1., cap_before_delivery=True):
    """Audit either charging order, converting zone powers to building area."""
    ratio = np.asarray(zone_heat_ratio)
    if np.any(~np.isfinite(ratio)) or np.any(ratio <= 0):
        raise ValueError("Zone/building heat ratios must be finite and positive")
    if np.ndim(capacity) != 0 or not np.isfinite(capacity) or capacity <= 0:
        raise ValueError("A positive scalar capacity is required")
    if ratio.ndim == 1:
        ratio = ratio[:, None]
    available_whole = available / ratio
    delivered_whole = delivered / ratio
    supplied = energy + DT * generated
    if np.any(supplied < -1e-5):
        raise ValueError("This production audit expects nonnegative available energy")
    charged = np.minimum(supplied, capacity) if cap_before_delivery else supplied
    remaining = np.maximum(DT * (available_whole - delivered_whole), 0)
    overflow = (np.maximum(supplied - capacity, 0) if cap_before_delivery
                else np.maximum(remaining - capacity, 0))
    loss = charged - DT * available_whole
    next_energy = np.minimum(remaining, capacity)
    residual = supplied - overflow - loss - DT * delivered_whole - next_energy
    transition_residual = next_energy[:, :-1] - energy[:, 1:]
    return {
        "overflow": overflow, "loss": loss,
        "limited": available < requested - 1e-4,
        "residual": residual, "transition_residual": transition_residual,
        "charged": charged,
    }


def replay_bucket(generated, requested, initial, retention, capacity, *, cap_before_delivery=True):
    """Fixed-demand replay in Wh/m2; optionally permit same-interval throughput."""
    energy = np.asarray(initial, dtype=np.float64).copy()
    history = np.empty_like(requested, dtype=np.float32)
    delivery = np.empty_like(requested, dtype=np.float32)
    for t in range(requested.shape[1]):
        history[:, t] = energy
        charged = np.maximum(energy + DT * generated[:, t], 0)
        available = (np.minimum(charged, capacity) if cap_before_delivery else charged) * retention
        delivery[:, t] = np.minimum(requested[:, t], available / DT)
        energy = np.minimum(np.maximum(available - DT * delivery[:, t], 0), capacity)
    return history, delivery


def source_signals(root: Path, building: int):
    signals, metadata = load_signals(root, building)
    raw = pd.read_parquet(root / f"egid={building}" / "timeseries.parquet")
    plant = raw[raw.zone == "technical_room"].set_index("datetime").sort_index()
    values = metadata["gardenCompatibility"]["values"]
    area = float(values["floor_area"] * values["totalFloors"])
    signals["q_total_w_m2"] = raw[raw.zone.str.startswith("fl")].groupby("datetime")[QROOM].sum() / area
    signals["charge_w_m2"] = plant.thermal_storage_charge_sh_kw * 1000 / area
    signals["reported_delivery_w_m2"] = plant.thermal_delivered_sh_kw * 1000 / area
    signals["pel_sh_w_m2"] = signals[PEL] * (1 - signals[DHW]) / area
    return signals, metadata


def withdrawal_before_drop(temperature, q_total, generation, start, release, end, threshold=.2):
    """Count heat withdrawn in a no-charge episode; failures remain diagnostic.

    Delivery integrated to a threshold crossing is not a measured tank capacity:
    pump heat, emitter/distribution storage and nonuniform tank temperature matter.
    """
    beginning = start
    while beginning > 0 and start - beginning < 96 and generation[beginning - 1] < .1:
        beginning -= 1
    baseline = float(np.mean(temperature[start - 4:start]))
    restart = np.flatnonzero(generation[start:end] >= .1)
    stop = start + int(restart[0]) if len(restart) else end
    crossings = np.flatnonzero(temperature[start:stop] < baseline - threshold)
    crossing = start + int(crossings[0]) if len(crossings) else stop
    # Tin is end-of-interval, so reaching a threshold includes the heat
    # delivered during that interval. Censoring at recharge excludes it.
    withdrawal_end = crossing + 1 if len(crossings) else stop
    return {
        "withdrawal_start": beginning,
        "no_charge_age_at_onset_h": (start - beginning) * DT,
        "observed_no_charge_duration_h": (stop - beginning) * DT,
        "cooling_before_recharge": bool(len(crossings)),
        "time_to_cooling_h": (crossing - start) * DT if len(crossings) else np.nan,
        "withdrawal_to_cooling_or_censor_wh_m2": float(np.sum(q_total[beginning:withdrawal_end]) * DT),
        "withdrawal_before_interruption_wh_m2": float(np.sum(q_total[beginning:start]) * DT),
        "recorded_charge_during_lockout_wh_m2": float(np.sum(generation[start:release]) * DT),
    }


def enrich_events(args, events):
    rows, inventory = [], []
    for building, group in events.groupby("building", sort=True):
        signals, metadata = source_signals(args.dataset, int(building))
        values = metadata["gardenCompatibility"]["values"]
        plant = metadata["resolved_plant_design"]
        area = float(values["floor_area"] * values["totalFloors"])
        sh_volume = float(plant["sh_buffer_volume_m3"]) / area
        dhw_volume = float(plant["dhw_tank_volume_m3"])
        common = {
            "water_capacity_wh_m2_k": 1163 * sh_volume,
            "sh_volume_l_m2": sh_volume * 1000,
            "dhw_volume_l": dhw_volume * 1000,
            "dhw_recovery_w_m2": float(values["DHW_recovery_W"]) / area,
            "hp_capacity_w_m2": float(values["hp_ref_capacity_W"]) / area,
            "design_sh_w_m2": float(values["SH_design_cap_W"]) / area,
            "floors": float(values["totalFloors"]),
            "construction_period": float(values["constructionPeriod"]),
        }
        for _, event in group.iterrows():
            start, release, end = (int(event[key]) for key in ("start", "release", "end"))
            row = event.to_dict() | common
            row["pre_temperature_gap_c"] = float((signals[TIN] - signals[SETPOINT]).iloc[start-4:start].mean())
            row["initially_heating_near_setpoint"] = bool(
                event.pre_qroom_w_m2 >= 5 and abs(row["pre_temperature_gap_c"]) <= .3
                and signals[VENT].iloc[start-4:start].max() <= 1e-8)
            row.update(withdrawal_before_drop(
                signals[TIN].to_numpy(), signals.q_total_w_m2.to_numpy(),
                signals.charge_w_m2.to_numpy(), start, release, end,
            ))
            row["withdrawal_equivalent_water_delta_k"] = row["withdrawal_to_cooling_or_censor_wh_m2"] / common["water_capacity_wh_m2_k"]
            for hours in (1, 3, 6, 12, 24):
                row[f"known_pre_dhw_{hours}h_tank_volumes"] = signals.dhw_l.iloc[start - int(hours / DT):start].sum() / (1000 * dhw_volume)
            row["known_lockout_dhw_tank_volumes"] = signals.dhw_l.iloc[start:release].sum() / (1000 * dhw_volume)
            row["known_post_dhw_tank_volumes"] = signals.dhw_l.iloc[release:end].sum() / (1000 * dhw_volume)
            row["hour_sin"] = np.sin(2 * np.pi * signals.index[start].hour / 24)
            row["hour_cos"] = np.cos(2 * np.pi * signals.index[start].hour / 24)
            row["observed_pre_charge_6h_wh_m2"] = signals.charge_w_m2.iloc[start - 24:start].sum() * DT
            row["observed_pre_heat_6h_wh_m2"] = signals.q_total_w_m2.iloc[start - 24:start].sum() * DT
            row["observed_zone_to_building_heat_ratio"] = signals[QROOM].iloc[start:end].sum() / max(signals.q_total_w_m2.iloc[start:end].sum(), 1e-8)
            rows.append(row)
        schema = metadata["timeseriesColumns"]
        charge = signals.charge_w_m2.to_numpy()
        pel = signals.pel_sh_w_m2.to_numpy()
        off = signals.hp_force_off_effective > .5
        inventory.append({
            "building": building, "split": group.split.iloc[0], **common,
            "tank_model": plant.get("sh_tank_model"),
            "tank_temperature_retained": "sh_tank_measured_temperature_c" in schema,
            "raw_retention_policy": metadata["energyplusArtifactRetention"].get("policy"),
            "charge_equals_reported_delivery": np.array_equal(charge, signals.reported_delivery_w_m2),
            "charge_wh_m2": charge.sum() * DT,
            "zone_heat_wh_m2": signals[QROOM].sum() * DT,
            "building_heat_wh_m2": signals.q_total_w_m2.sum() * DT,
            "pel_sh_wh_m2": pel.sum() * DT,
            "recorded_charge_to_pel_ratio": charge.sum() / max(pel.sum(), 1e-8),
            "room_heat_during_off_wh_m2": signals.q_total_w_m2[off].sum() * DT,
            "charge_during_effective_off_wh_m2": signals.charge_w_m2[off].sum() * DT,
            "requested_effective_mismatch_fraction": np.mean((signals[REQUEST] > .5) != off),
        })
        if len(inventory) % 20 == 0:
            print(f"Enriched {len(inventory)} buildings", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(args.output / "events.csv", index=False)
    pd.DataFrame(inventory).to_csv(args.output / "data_inventory.csv", index=False)
    return result


def fit_event_probes(events, output):
    """Forecast-event probes, not trajectory calibration or surrogate outputs."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    base = ["duration_h", "tout_c", "month", "hour_sin", "hour_cos", "hp_capacity_w_m2",
            "design_sh_w_m2", "construction_period", "floors"]
    storage = ["sh_volume_l_m2", "dhw_volume_l", "dhw_recovery_w_m2"]
    history = [f"known_pre_dhw_{h}h_tank_volumes" for h in (1, 3, 6, 12, 24)]
    known_future = ["known_lockout_dhw_tank_volumes", "known_post_dhw_tank_volumes"]
    settings = {
        "weather_duration_calendar": base,
        "known_dhw_history_and_storage": base + storage + history,
        "known_annual_dhw_schedule": base + storage + history + known_future,
        "privileged_observed_state": base + storage + history + known_future + [
            "observed_pre_charge_6h_wh_m2", "observed_pre_heat_6h_wh_m2",
            "no_charge_age_at_onset_h", "pre_hp_on_fraction", "dhw_first_hour",
        ],
    }
    train, validation, test = (events[events.split == split] for split in ("train", "validation", "test"))
    predictions, scores, specifications = [], [], {}
    for name, columns in settings.items():
        candidates = []
        for iterations in (50, 100, 200):
            model = HistGradientBoostingClassifier(
                max_iter=iterations, max_leaf_nodes=7, min_samples_leaf=100,
                learning_rate=.05, l2_regularization=10, early_stopping=False, random_state=13,
            ).fit(train[columns], train.drop_gt_0p5c.astype(int))
            probability = model.predict_proba(validation[columns])[:, 1]
            candidates.append((brier_score_loss(validation.drop_gt_0p5c, probability), iterations, model))
        _, iterations, model = min(candidates, key=lambda value: value[0])
        specifications[name] = {"features": columns, "iterations_selected_on_validation": iterations,
                                "train_buildings": sorted(train.building.unique().tolist())}
        for split, frame in (("validation", validation), ("test", test)):
            probability = model.predict_proba(frame[columns])[:, 1]
            truth = frame.drop_gt_0p5c.astype(int).to_numpy()
            scores.append({"probe": name, "split": split, "events": len(frame),
                           "brier": brier_score_loss(truth, probability),
                           "log_loss": log_loss(truth, probability), "auc": roc_auc_score(truth, probability),
                           "mean_probability": probability.mean(), "observed_rate": truth.mean()})
            for building, event, y, p in zip(frame.building, frame.event, truth, probability):
                predictions.append({"probe": name, "split": split, "building": building,
                                    "event": event, "truth": y, "probability": p, "brier": (p-y)**2})
        print(f"Fitted diagnostic probe {name} ({iterations} iterations)", flush=True)
    pd.DataFrame(scores).to_csv(output / "event_probe_scores.csv", index=False)
    predictions = pd.DataFrame(predictions)
    predictions.to_csv(output / "event_probe_predictions.csv", index=False)
    (output / "event_probe_specifications.json").write_text(json.dumps(specifications, indent=2))
    base_scores = predictions[(predictions.probe == "weather_duration_calendar") & (predictions.split == "test")]
    rows = []
    rng = np.random.default_rng(13)
    for name in settings:
        candidate = predictions[(predictions.probe == name) & (predictions.split == "test")]
        joined = candidate.merge(base_scores, on=["building", "event"], suffixes=("_candidate", "_base"), validate="one_to_one")
        differences = (joined.brier_candidate - joined.brier_base).groupby(joined.building).mean().to_numpy()
        draws = rng.choice(differences, size=(2000, len(differences)), replace=True).mean(axis=1)
        rows.append({"probe": name, "brier_difference": differences.mean(), "lower95": np.quantile(draws, .025),
                     "upper95": np.quantile(draws, .975), "buildings": len(differences)})
    pd.DataFrame(rows).to_csv(output / "event_probe_bootstrap.csv", index=False)


def sample_audit(artifact, profile, gate, key, particles):
    import equinox as eqx
    import jax.numpy as jnp
    from neural_building_emulator.train import _probabilistic_ventilation_rollout_kwargs

    scalers = artifact.scalers
    ventilation = _probabilistic_ventilation_rollout_kwargs(
        profile, mode="eplus_rule", metadata_columns=artifact.legacy_artifact.metadata["metadata_columns"])

    @eqx.filter_jit
    def rollout(model, metadata, inputs, initial, rng, availability, vent):
        prediction, aux = model.sample_with_aux(
            metadata, inputs, initial, key=rng, num_particles=particles, sample_process_noise=True,
            hp_scenario_mode="bernoulli", hp_generation_availability=availability, **vent)
        return prediction, (aux[1], aux[2], aux[3], aux[4])

    prediction, aux = rollout(
        artifact.model, jnp.asarray(scalers.metadata.transform(profile.metadata)),
        jnp.asarray(scalers.inputs.transform(profile.inputs)),
        jnp.asarray((profile.initial_temperature[0] - scalers.target.mean[:1]) / scalers.target.scale[:1]),
        key, jnp.asarray(gate), ventilation)
    return np.asarray(prediction) * scalers.target.scale + scalers.target.mean, tuple(np.asarray(x) for x in aux)


def model_audit(args, events):
    import jax
    from .artifacts import load_artifact
    from .profiles import load_profiles

    request = json.loads((args.previous / "request.json").read_text())
    artifact = load_artifact(Path(request["artifact"]))
    if artifact.model.hp_buffer_capacity_mode != "global":
        raise ValueError("This accounting diagnostic is specific to the frozen global-cap checkpoint")
    capacity = artifact.model.hp_energy_cap_wh_m2
    annual_rows, event_rows = [], []
    cache = args.output / "mean_audit_traces"
    cache.mkdir(exist_ok=True)
    for building, group in events[events.split == "test"].groupby("building", sort=True):
        annual_file = cache / f"{building}_annual.json"
        event_file = cache / f"{building}_events.csv"
        if annual_file.exists() and event_file.exists():
            annual_rows.append(json.loads(annual_file.read_text()))
            event_rows.extend(pd.read_csv(event_file).to_dict("records"))
            continue
        signals, metadata = source_signals(args.dataset, int(building))
        profile = load_profiles(artifact, dataset_path=args.dataset, profile_ids=[int(building)])[0]
        if not np.array_equal(profile.datetime.astype("datetime64[ns]"), signals.index.to_numpy()[1:]):
            raise ValueError("Raw and production profile timestamps differ")
        predictions, (cop, energy, available, requested) = sample_audit(
            artifact, profile, 1 - signals[REQUEST].to_numpy(np.float32)[1:],
            jax.random.fold_in(jax.random.PRNGKey(request["seed"]), int(building)), request["particles"])
        with np.load(args.previous / "ensemble_cache" / f"{building}.npz") as previous:
            np.testing.assert_allclose(predictions, previous["predictions"], atol=2e-4, rtol=2e-4)
            np.testing.assert_allclose(energy, previous["energy"], atol=2e-4, rtol=2e-4)
        generated = cop * predictions[:, :, 2]
        accounting = audit_bucket(energy, generated, requested, predictions[:, :, 1], available, capacity)
        if np.max(np.abs(accounting["transition_residual"])) > .002:
            raise ValueError("Buffer transition accounting failed")
        # Some particles can be exactly empty: estimate retention using the
        # nonempty steps only, not by treating zeros as a different loss rate.
        retention = np.nanmedian(np.divide(DT * available, accounting["charged"],
                                          out=np.full_like(available, np.nan), where=accounting["charged"] > 1e-4), axis=1)
        actual_pel = signals.pel_sh_w_m2.to_numpy()[1:]
        actual_charge = signals.charge_w_m2.to_numpy()[1:]
        replays = {}
        for name, charge in (("observed_pel_model_cop", cop * actual_pel[None, :]),
                             ("recorded_charge_channel", np.broadcast_to(actual_charge, generated.shape))):
            replays[name] = replay_bucket(charge, requested, energy[:, 0], retention, capacity)
        row = {
            "building": int(building), "particles": request["particles"], "capacity_wh_m2": float(capacity),
            "generated_wh_m2": float(np.mean(np.sum(generated, axis=1) * DT)),
            "delivered_wh_m2": float(np.mean(np.sum(predictions[:, :, 1], axis=1) * DT)),
            "pel_wh_m2": float(np.mean(np.sum(predictions[:, :, 2], axis=1) * DT)),
            "overflow_wh_m2": float(np.mean(np.sum(accounting["overflow"], axis=1))),
            "standing_loss_wh_m2": float(np.mean(np.sum(accounting["loss"], axis=1))),
            "effective_cop_energy_ratio": float(generated.sum() / max(predictions[:, :, 2].sum(), 1e-8)),
            "capacity_binding_fraction": float(np.mean(accounting["overflow"] > 1e-4)),
            "storage_limited_fraction": float(accounting["limited"].mean()),
            "max_transition_residual_wh_m2": float(np.max(np.abs(accounting["transition_residual"]))),
            "loss_rate_per_h": float(np.mean(-np.log(retention) / DT)),
            "recorded_charge_wh_m2": float(actual_charge.sum() * DT),
            "actual_pel_wh_m2": float(actual_pel.sum() * DT),
            "actual_zone_heat_wh_m2": float(signals[QROOM].iloc[1:].sum() * DT),
            "actual_building_heat_wh_m2": float(signals.q_total_w_m2.iloc[1:].sum() * DT),
        }
        annual_rows.append(row)
        local_rows = []
        for _, event in group.iterrows():
            s, r, e = (int(event[k]) - 1 for k in ("start", "release", "end"))
            local = {"building": int(building), "event": int(event.event), "dhw_first_hour": bool(event.dhw_first_hour),
                     "observed_drop_gt_0p5c": bool(event.drop_gt_0p5c), "observed_drop_c": event.drop_c,
                     "energy_start_wh_m2": energy[:, s].mean(), "energy_min_wh_m2": energy[:, s:e].min(axis=1).mean(),
                     "predicted_withdrawal_wh_m2": predictions[:, s:e, 1].sum(axis=1).mean() * DT,
                     "observed_withdrawal_wh_m2": signals.q_total_w_m2.iloc[s+1:e+1].sum() * DT,
                     "requested_qroom_w_m2": requested[:, s:e].mean(),
                     "delivered_qroom_w_m2": predictions[:, s:e, 1].mean(),
                     "storage_limited_fraction": accounting["limited"][:, s:e].mean(),
                     "overflow_pre_24h_wh_m2": accounting["overflow"][:, s-96:s].sum(axis=1).mean(),
                     "reserve_hours_at_pre_load": energy[:, s].mean() / max(event.pre_qroom_w_m2, 1),
                     "inventory_equivalent_water_delta_k": energy[:, s].mean() / event.water_capacity_wh_m2_k}
            for name, (replay_energy, replay_q) in replays.items():
                local[f"{name}_energy_start_wh_m2"] = replay_energy[:, s].mean()
                local[f"{name}_energy_min_wh_m2"] = replay_energy[:, s:e].min(axis=1).mean()
                local[f"{name}_limited_fraction"] = np.mean(replay_q[:, s:e] < requested[:, s:e] - 1e-4)
            local_rows.append(local)
        event_rows.extend(local_rows)
        np.savez_compressed(cache / f"{building}.npz", datetime=profile.datetime,
                            energy=energy.mean(axis=0), requested=requested.mean(axis=0),
                            generated=generated.mean(axis=0), available=available.mean(axis=0),
                            overflow=accounting["overflow"].mean(axis=0), loss=accounting["loss"].mean(axis=0),
                            **{f"{name}_energy": values[0].mean(axis=0) for name, values in replays.items()})
        annual_file.write_text(json.dumps(row, indent=2))
        pd.DataFrame(local_rows).to_csv(event_file, index=False)
        pd.DataFrame(annual_rows).to_csv(args.output / "model_annual_energy.csv", index=False)
        pd.DataFrame(event_rows).to_csv(args.output / "model_event_energy.csv", index=False)
        print(f"Audited frozen checkpoint {building}", flush=True)
    pd.DataFrame(annual_rows).to_csv(args.output / "model_annual_energy.csv", index=False)
    pd.DataFrame(event_rows).to_csv(args.output / "model_event_energy.csv", index=False)


def write_event_plots(args, events):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    examples = []
    for building in (11152319, 11230131, 191952443):
        selection = events[(events.building == building) & events.initially_heating_near_setpoint]
        if selection.empty:
            continue
        failure = selection.loc[selection.drop_c.idxmax()]
        examples.append(failure)
        # A successful event from the same building with similar duration/load.
        success = selection[selection.drop_c < .1].copy()
        if not success.empty:
            distance = ((success.duration_h - failure.duration_h).abs()
                        + (success.pre_qroom_w_m2 - failure.pre_qroom_w_m2).abs() / 10)
            examples.append(success.loc[distance.idxmin()])
    links = []
    for number, event in enumerate(examples):
        building = int(event.building)
        signals, _ = source_signals(args.dataset, building)
        left, right = max(1, int(event.start) - 24), min(len(signals), int(event.end) + 16)
        part = signals.iloc[left:right]
        dates = part.index
        sl = slice(left-1, right-1)
        with np.load(args.previous / "ensemble_cache" / f"{building}.npz") as cache:
            paths, energy = cache["predictions"][:, sl], cache["energy"][:, sl]
        with np.load(args.output / "mean_audit_traces" / f"{building}.npz") as cache:
            audit = {key: cache[key][sl] for key in ("requested", "generated", "recorded_charge_channel_energy", "overflow")}
        mean = paths.mean(axis=0)
        lower, upper = np.quantile(paths[:, :, 0], [.05, .95], axis=0)
        curves = [
            [(part[TIN].to_numpy(), "EnergyPlus Tin", "#1565c0"), (mean[:, 0], "Emulator mean Tin", "#c62828")],
            [(part[QROOM].to_numpy(), "EnergyPlus zone heat", "#1565c0"),
             (part.q_total_w_m2.to_numpy(), "EnergyPlus building-average heat", "#455a64"),
             (mean[:, 1], "Emulator delivered heat", "#c62828"),
             (audit["requested"], "Emulator requested heat", "#ef6c00")],
            [(energy.mean(axis=0), "Emulator inventory", "#c62828"),
             (audit["recorded_charge_channel_energy"], "Observed-charge replay inventory (not forecast)", "#7b1fa2")],
            [(part.charge_w_m2.to_numpy(), "Recorded SH charging", "#1565c0"),
             (audit["generated"], "Emulator generated heat", "#c62828"),
             (audit["overflow"] / DT, "Emulator cap spill", "#ef6c00")],
        ]
        titles = ("Indoor temperature [C]", "Room heat [W/m2]", "Buffer inventory [Wh/m2]", "Charging and discarded heat [W/m2]")
        figure = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.06, subplot_titles=titles)
        figure.add_trace(go.Scatter(x=dates, y=lower, mode="lines", line=dict(width=0), showlegend=False), row=1, col=1)
        figure.add_trace(go.Scatter(x=dates, y=upper, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(198,40,40,.15)", name="Raw 90% Tin interval"), row=1, col=1)
        for row, items in enumerate(curves, start=1):
            for values, name, color in items:
                figure.add_trace(go.Scatter(x=dates, y=values, name=name, line=dict(color=color)), row=row, col=1)
        onset, release = signals.index[int(event.start)], signals.index[int(event.release)]
        for row in range(1, 5):
            figure.add_vrect(x0=onset.isoformat(), x1=release.isoformat(), fillcolor="gray", opacity=.12, line_width=0, row=row, col=1)
        figure.update_layout(title=f"EGID {building}, {event.datetime}: drop {event.drop_c:.2f} C; gray = requested lockout",
                             height=1100, template="plotly_white", hovermode="x unified", margin=dict(t=80),
                             legend=dict(orientation="h", y=-.08))
        filename = f"event_{building}_{int(event.event)}.html"
        figure.write_html(args.output / filename, include_plotlyjs="directory")
        links.append(f'<a href="{filename}">EGID {building}, {event.datetime}, drop {event.drop_c:.2f} C</a>')
        if number == 0:
            fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True, layout="constrained")
            axes[0].fill_between(dates, lower, upper, color="#c62828", alpha=.12)
            for axis, items, title in zip(axes, curves, titles):
                for values, name, color in items:
                    axis.plot(dates, values, label=name, color=color, lw=1.4)
                axis.axvspan(onset, release, color="gray", alpha=.12)
                axis.set_ylabel(title)
                axis.legend(fontsize=8, loc="upper left")
                axis.grid(alpha=.2)
            fig.suptitle(f"EGID {building}: frozen buffer diagnosis; shaded interval = lockout")
            fig.savefig(args.output / "event_preview.png", dpi=140)
            plt.close(fig)
    return links


def write_report(args):
    annual = pd.read_csv(args.output / "model_annual_energy.csv")
    events = pd.read_csv(args.output / "events.csv")
    state = pd.read_csv(args.output / "model_event_energy.csv")
    inventory = pd.read_csv(args.output / "data_inventory.csv")
    score = pd.read_csv(args.output / "event_probe_scores.csv")
    boot = pd.read_csv(args.output / "event_probe_bootstrap.csv")
    conditional = state.groupby(["dhw_first_hour", "observed_drop_gt_0p5c"]).mean(numeric_only=True).drop(columns=["building", "event"])
    conditional.to_csv(args.output / "conditional_energy_summary.csv")
    observed = events[events.split == "test"].groupby(["dhw_first_hour", "drop_gt_0p5c"]).agg(
        events=("event", "size"), no_charge_age_h=("no_charge_age_at_onset_h", "mean"),
        withdrawal_wh_m2=("withdrawal_to_cooling_or_censor_wh_m2", "mean"),
        equivalent_water_delta_k=("withdrawal_equivalent_water_delta_k", "mean"),
        cooling_before_recharge=("cooling_before_recharge", "mean"),
        zone_to_building_ratio=("observed_zone_to_building_heat_ratio", "mean"))
    observed.to_csv(args.output / "observed_withdrawal_summary.csv")
    near_setpoint = events[(events.split == "test") & events.initially_heating_near_setpoint]
    subset = near_setpoint.merge(state, on=["building", "event"], suffixes=("", "_model"), validate="one_to_one")
    subset_summary = subset.groupby(["dhw_first_hour", "drop_gt_0p5c"]).agg(
        events=("event", "size"), drop_c=("drop_c", "mean"),
        withdrawal_wh_m2=("withdrawal_to_cooling_or_censor_wh_m2", "mean"),
        energy_start_wh_m2=("energy_start_wh_m2", "mean"), energy_min_wh_m2=("energy_min_wh_m2", "mean"),
        limited_fraction=("storage_limited_fraction", "mean"))
    subset_summary.to_csv(args.output / "initially_heating_subset.csv")
    aggregate = {
        "generated_excess_vs_recorded_charge_fraction": annual.generated_wh_m2.sum() / annual.recorded_charge_wh_m2.sum() - 1,
        "electric_energy_bias_fraction": annual.pel_wh_m2.sum() / annual.actual_pel_wh_m2.sum() - 1,
        "discarded_fraction_of_generated_heat": annual.overflow_wh_m2.sum() / annual.generated_wh_m2.sum(),
        "standing_loss_fraction_of_generated_heat": annual.standing_loss_wh_m2.sum() / annual.generated_wh_m2.sum(),
        "zone_target_vs_building_heat_fraction": annual.actual_zone_heat_wh_m2.sum() / annual.actual_building_heat_wh_m2.sum(),
        "maximum_buffer_balance_residual_wh_m2": annual.max_transition_residual_wh_m2.max(),
        "buildings_with_excess_generation": int((annual.generated_wh_m2 > annual.recorded_charge_wh_m2).sum()),
        "initially_heating_events": len(near_setpoint),
        "initially_heating_large_drops": int(near_setpoint.drop_gt_0p5c.sum()),
        "initially_heating_large_drops_with_dhw": int((near_setpoint.drop_gt_0p5c & near_setpoint.dhw_first_hour).sum()),
    }
    excess = (annual.generated_wh_m2 - annual.recorded_charge_wh_m2).to_numpy() / 1000
    bootstrap = np.random.default_rng(13).choice(excess, size=(2000, len(excess)), replace=True).mean(axis=1)
    aggregate["mean_generation_excess_kwh_m2"] = float(excess.mean())
    aggregate["generation_excess_kwh_m2_lower95"] = float(np.quantile(bootstrap, .025))
    aggregate["generation_excess_kwh_m2_upper95"] = float(np.quantile(bootstrap, .975))
    (args.output / "findings.json").write_text(json.dumps(aggregate, indent=2))
    links = write_event_plots(args, events)
    table = lambda df: df.to_html(float_format=lambda x: f"{x:.4f}")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Buffer energy diagnosis</title>
    <style>body{{font:15px system-ui;margin:32px}}table{{border-collapse:collapse;font-size:12px}}td,th{{padding:6px;border:1px solid #ddd}}h2{{margin-top:32px}}</style></head><body>
    <h1>Buffer energy diagnosis</h1><p>Frozen production checkpoint; no model changes or ex-post calibration.</p>
    <p>Same 200-building split. Accounting replays use the same 100 annual particles in 20 held-out buildings as the preceding diagnosis.
    Auxiliary states are checked against those cached trajectories. Replay candidates use observed SH electricity or a recorded charging
    channel but keep requested room heat fixed: they are open-loop diagnostics, not forecasts or calibration methods.</p>
    <h2>Main findings</h2><p>Relative to the recorded SH charging channel, modeled generation is
    {100*aggregate['generated_excess_vs_recorded_charge_fraction']:.1f}% higher, while SH electric energy is
    {100*aggregate['electric_energy_bias_fraction']:.1f}% different. The current buffer discards
    {100*aggregate['discarded_fraction_of_generated_heat']:.1f}% of generated heat at its cap and assigns
    {100*aggregate['standing_loss_fraction_of_generated_heat']:.1f}% to standing losses.
    Its algebraic balance closes: this is parameter/representation diagnosis, not an arithmetic energy-creation bug.</p>
    <p>The first-floor zone target is {100*aggregate['zone_target_vs_building_heat_fraction']:.1f}% of whole-building average heat delivery.
    This spatial boundary mismatch must be reconciled before interpreting the global latent inventory as a physical tank.</p>
    <h2>Focused confidence-band traces</h2><ul>{''.join('<li>'+link+'</li>' for link in links)}</ul>
    <h2>Initially heated, near-setpoint subset</h2><p>Pre-event heat >=5 W/m2, mean temperature within 0.3 C of setpoint,
    and no pre-event ventilation. This sensitivity check avoids conflating ordinary passive cooling with a heat-delivery failure.
    {aggregate['initially_heating_large_drops_with_dhw']} of {aggregate['initially_heating_large_drops']} drops above 0.5 C
    coincide with active DHW in the first post-release hour. This is an association, not proof of the exact tank-control mechanism.</p>{table(subset_summary)}
    <h2>Observed no-charge episodes</h2><p>Heat withdrawn to a 0.2 C cooling threshold, or right-censoring at recharge/end of observation.
    These are not measured tank capacities. Pump heat, distribution/emitter storage and heat-delivery constraints are not separated.</p>{table(observed)}
    <h2>Conditional emulator energy</h2>{table(conditional)}
    <h2>Annual emulator accounting</h2>{table(annual)}
    <h2>Predictability from known inputs</h2><p>Small diagnostic gradient-boosted classifiers: 140 training buildings, iteration count chosen
    on 20 validation buildings, evaluated on the same 20 held-out buildings. Calibration buildings are unused.
    The annual-DHW-schedule probe assumes the full deterministic DHW input schedule is available, as in the emulator task.
    The privileged probe uses simulator state/outcomes and is diagnostic only. Probabilities are not applied to emulator paths.</p>{table(score)}
    <h2>Paired test Brier differences versus weather/calendar baseline</h2>{table(boot)}
    <h2>Data limitations</h2><p>{int(inventory.tank_temperature_retained.sum())}/{len(inventory)} buildings retain SH tank temperatures.
    Charging and named delivered-SH columns are identical in {int(inventory.charge_equals_reported_delivery.sum())}/{len(inventory)} buildings;
    the latter cannot be treated as room delivery. Raw EnergyPlus artifact retention policies: {html.escape(str(inventory.raw_retention_policy.value_counts().to_dict()))}.
    The currently installed neighboring idf-builder template uses a mixed SH tank, whereas these exports specify stratified SH tanks.
    Its current EMS rules are therefore not assumed authoritative for this dataset.</p>
    <h2>Implications for a controlled next experiment</h2><p>First align the storage energy boundary with whole-building heat delivery
    and validate the recorded source-side charging channel. Then supervise the effective electricity-to-heat conversion directly,
    so it cannot hide an inflated heat input in cap spill or learned standing losses. Compare usable-storage/delivery changes with
    DHW recovery state using identical conditions. The observed-charge replay still often retains substantial inventory, so correcting
    the heat input alone is not established as sufficient. Tank-node or outlet-temperature measurements are needed to distinguish
    stratification from delivery thresholds conclusively; none are required as future emulator inference inputs.</p>
    <h2>Provenance</h2><pre>{html.escape((args.output / 'request.json').read_text())}</pre></body></html>"""
    (args.output / "index.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/control=force_off"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    previous = json.loads((args.previous / "request.json").read_text())
    artifact = Path(previous["artifact"])
    request = {**{key: str(value) for key, value in vars(args).items()},
               "checkpoint_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in artifact.glob("*.eqx")},
               "previous_request": previous}
    request_path = args.output / "request.json"
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError("Use a fresh directory when changing audit provenance")
    request_path.write_text(json.dumps(request, indent=2))
    event_path = args.output / "events.csv"
    events = pd.read_csv(event_path) if event_path.exists() else None
    if events is None or "initially_heating_near_setpoint" not in events:
        events = enrich_events(args, eligible(pd.read_csv(args.previous / "events.csv")))
    if not (args.output / "event_probe_bootstrap.csv").exists():
        fit_event_probes(events, args.output)
    model_audit(args, events)
    write_report(args)


if __name__ == "__main__":
    main()
