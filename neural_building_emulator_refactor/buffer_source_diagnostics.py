"""Validate retained charging labels and isolate buffer accounting confounders.

Source regressions are diagnostic probes trained on training buildings only.
Bucket replays use frozen ensemble-mean forcing and some observed signals;
they are neither closed-loop forecasts nor recalibrated emulator trajectories.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .buffer_energy_diagnostics import replay_bucket, source_signals
from .buffer_recharge_diagnostics import DHW, DT, QROOM, REQUEST


def charging_checks(charge, electric, off, dhw):
    arrays = [np.asarray(a, dtype=float) for a in (charge, electric, off, dhw)]
    if not all(a.ndim == 1 and a.shape == arrays[0].shape for a in arrays):
        raise ValueError("Charging signals must be aligned one-dimensional arrays")
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Charging signals contain nonfinite values")
    charge, electric, off, dhw = arrays
    active = electric > 1
    ratios = charge[active] / electric[active]
    return {
        "total_charge_wh_m2": float(charge.sum() * DT),
        "negative_charge_wh_m2": float(np.minimum(charge, 0).sum() * DT),
        "minimum_charge_w_m2": float(charge.min()),
        "negative_charge_steps": int(np.sum(charge < -1e-6)),
        "charge_during_off_wh_m2": float(charge[off > .5].sum() * DT),
        "charge_during_dhw_wh_m2": float(charge[dhw > .5].sum() * DT),
        "charge_without_sh_electric_wh_m2": float(charge[electric <= 1e-6].sum() * DT),
        "charge_to_electric_ratio": float(charge.sum() / max(electric.sum(), 1e-8)),
        "active_ratio_q01": float(np.quantile(ratios, .01)) if len(ratios) else np.nan,
        "active_ratio_q99": float(np.quantile(ratios, .99)) if len(ratios) else np.nan,
    }


def conversion_window_checks(charge, electric, steps):
    charge, electric = np.asarray(charge, dtype=float), np.asarray(electric, dtype=float)
    count = len(charge) // steps
    heat = charge[:count * steps].reshape(count, steps).sum(axis=1)
    power = electric[:count * steps].reshape(count, steps).sum(axis=1)
    active = power / steps > 1
    ratios = heat[active] / power[active]
    outside = active & ((heat < power) | (heat > 8 * power))
    return {"block_hours": steps * DT, "active_blocks": int(active.sum()),
            "ratio_q01": float(np.quantile(ratios, .01)), "ratio_q99": float(np.quantile(ratios, .99)),
            "outside_1_8_fraction": float(outside.sum() / max(active.sum(), 1)),
            "outside_1_8_charge_fraction": float(np.maximum(heat[outside], 0).sum() / max(np.maximum(heat[active], 0).sum(), 1e-8))}


def source_features(outdoor, metadata, mode):
    """Only deterministic weather and known design metadata, never tank state."""
    outdoor = np.asarray(outdoor, dtype=float)
    base = np.column_stack([np.ones_like(outdoor), outdoor / 10])
    names = ["intercept", "outdoor_per_10c"]
    if mode == "weather":
        return base, names
    if mode != "weather_design":
        raise ValueError(f"Unknown feature mode {mode}")
    supply = float(metadata["sizing"]["sh_supply_C"]) / 50
    reference = float(metadata["gardenCompatibility"]["values"]["hp_ref_cop"]) / 4
    extra = np.column_stack([
        np.full_like(outdoor, supply), np.full_like(outdoor, reference),
        supply * outdoor / 10, (outdoor / 10) ** 2,
    ])
    return np.column_stack([base, extra]), names + [
        "design_supply_per_50c", "reference_cop_per_4",
        "design_supply_outdoor_interaction", "outdoor_squared",
    ]


def hourly_source_rows(signals, metadata, mode):
    """Regress block charge on sum(Pel * COP features), avoiding tiny-Pel ratios."""
    x, names = source_features(signals.tout.to_numpy(), metadata, mode)
    electric = signals.pel_sh_w_m2.to_numpy()
    charge = signals.charge_w_m2.to_numpy()
    count = len(signals) // 4
    design = (x[:count * 4] * electric[:count * 4, None]).reshape(count, 4, -1).mean(axis=1)
    target = charge[:count * 4].reshape(count, 4).mean(axis=1)
    active = design[:, 0] > 1
    # Equal building-policy influence; relative charge error within each record.
    norm = max(float(np.mean(design[active, 0] ** 2)), 1) * max(int(active.sum()), 1)
    return design[active], target[active], norm, names


def fit_source_probe(records, ridge):
    dimension = records[0][0].shape[1]
    gram, rhs = np.zeros((dimension, dimension)), np.zeros(dimension)
    for design, target, norm, _ in records:
        gram += design.T @ design / norm
        rhs += design.T @ target / norm
    penalty = np.eye(dimension) * ridge * len(records)
    penalty[0, 0] = 0
    return np.linalg.solve(gram + penalty, rhs)


def block_source_error(record, coefficients):
    design, target, _, _ = record
    if not len(target):
        raise ValueError("No active source blocks")
    electric = design[:, 0]
    # Physical bound is applied to the block-effective conversion, not charge.
    prediction = electric * np.clip((design @ coefficients) / electric, 1, 8)
    return {
        "relative_rmse": float(np.sqrt(np.mean((prediction - target) ** 2)) / max(target.mean(), 1e-8)),
        "energy_bias_fraction": float(prediction.sum() / max(target.sum(), 1e-8) - 1),
        "rmse_w_m2": float(np.sqrt(np.mean((prediction - target) ** 2))),
    }


def inventory_and_probes(args, inventory):
    checks, window_checks, learned, all_records = [], [], {}, {}
    policies = {"setpoint": args.setpoint_dataset, "force_off": args.force_off_dataset}
    for policy, root in policies.items():
        for i, row in enumerate(inventory.itertuples()):
            signals, metadata = source_signals(root, int(row.building))
            if not np.isfinite(signals.to_numpy()).all():
                raise ValueError(f"Nonfinite source data: {policy}/{row.building}")
            check = {"building": row.building, "split": row.split, "policy": policy}
            check.update(charging_checks(signals.charge_w_m2, signals.pel_sh_w_m2,
                                         signals[REQUEST], signals[DHW]))
            check["charge_alias_equal"] = bool(np.array_equal(signals.charge_w_m2, signals.reported_delivery_w_m2))
            check["annual_building_heat_to_charge_ratio"] = float(signals.q_total_w_m2.sum() / signals.charge_w_m2.sum())
            checks.append(check)
            for steps in (1, 4, 16, 96):
                window_checks.append({"building": row.building, "split": row.split, "policy": policy,
                                      **conversion_window_checks(signals.charge_w_m2, signals.pel_sh_w_m2, steps)})
            for mode in ("weather", "weather_design"):
                all_records[(policy, row.building, row.split, mode)] = hourly_source_rows(signals, metadata, mode)
            if (i + 1) % 20 == 0:
                print(f"Validated {policy}: {i + 1}/{len(inventory)} building records", flush=True)
    rows = []
    for mode in ("weather", "weather_design"):
        train = [record for key, record in all_records.items() if key[2:] == ("train", mode)]
        validation = [record for key, record in all_records.items() if key[2:] == ("validation", mode)]
        candidates = []
        for ridge in (1e-6, 1e-4, .01, .1, 1):
            coefficients = fit_source_probe(train, ridge)
            error = np.mean([block_source_error(record, coefficients)["relative_rmse"] for record in validation])
            candidates.append((error, ridge, coefficients))
        error, ridge, coefficients = min(candidates, key=lambda item: item[0])
        learned[mode] = {"ridge": ridge, "coefficients": coefficients.tolist(), "features": train[0][3],
                         "validation_relative_rmse": float(error), "fit_splits": ["train"],
                         "selection_split": "validation", "policies": list(policies)}
        for (policy, building, split, feature_mode), record in all_records.items():
            if feature_mode == mode:
                rows.append({"model": mode, "policy": policy, "building": building,
                             "split": split, **block_source_error(record, coefficients)})
    pd.DataFrame(checks).to_csv(args.output / "source_checks.csv", index=False)
    pd.DataFrame(window_checks).to_csv(args.output / "source_window_checks.csv", index=False)
    pd.DataFrame(rows).to_csv(args.output / "source_probe_scores.csv", index=False)
    (args.output / "source_probes.json").write_text(json.dumps(learned, indent=2))
    return learned


def accounting_replays(args, inventory, learned):
    events = pd.read_csv(args.previous / "events.csv")
    annual = pd.read_csv(args.previous / "model_annual_energy.csv").set_index("building")
    source_rows, event_rows = [], []
    for row in inventory[inventory.split == "test"].itertuples():
        building = int(row.building)
        signals, metadata = source_signals(args.force_off_dataset, building)
        with np.load(args.previous / "mean_audit_traces" / f"{building}.npz") as data:
            cache = {key: data[key] for key in data.files}
        if not np.array_equal(cache["datetime"].astype("datetime64[ns]"), signals.index.to_numpy()[1:]):
            raise ValueError("Audit and raw timestamps differ")
        with np.load(args.ensemble / "ensemble_cache" / f"{building}.npz") as data:
            electric = data["predictions"][:, :, 2].mean(axis=0)
        features, _ = source_features(signals.tout.to_numpy()[1:], metadata, "weather_design")
        conversion = np.clip(features @ learned["weather_design"]["coefficients"], 1, 8)
        inferred_charge = electric * conversion
        actual = signals.charge_w_m2.to_numpy()[1:]
        source_rows.append({"building": building, "recorded_charge_wh_m2": actual.sum() * DT,
                            "raw_generated_wh_m2": cache["generated"].sum() * DT,
                            "probe_generated_wh_m2": inferred_charge.sum() * DT})
        # This observed annual ratio is an oracle accounting intervention, not
        # an inference feature. It isolates spatial normalization from source bias.
        ratio = signals.q_total_w_m2.sum() / signals[QROOM].sum()
        water_capacity = 1163 * metadata["resolved_plant_design"]["sh_buffer_volume_m3"] / (
            metadata["gardenCompatibility"]["values"]["floor_area"]
            * metadata["gardenCompatibility"]["values"]["totalFloors"])
        capacity = float(annual.loc[building, "capacity_wh_m2"])
        retention = np.array([np.exp(-DT * annual.loc[building, "loss_rate_per_h"])])
        arms = {
            "mean_forcing_baseline": (cache["generated"], 1, capacity, retention, True),
            "recorded_source": (actual, 1, capacity, retention, True),
            "building_boundary": (cache["generated"], ratio, capacity, retention, True),
            "source_and_boundary": (actual, ratio, capacity, retention, True),
            "source_boundary_no_loss": (actual, ratio, capacity, np.ones(1), True),
            "source_boundary_aftercap": (actual, ratio, capacity, retention, False),
            "source_boundary_no_loss_aftercap": (actual, ratio, capacity, np.ones(1), False),
        }
        for delta in (5, 15, 25):
            arms[f"source_boundary_{delta}k"] = (actual, ratio, water_capacity * delta, retention, True)
            arms[f"source_boundary_{delta}k_no_loss_aftercap"] = (actual, ratio, water_capacity * delta, np.ones(1), False)
        for name, (charge, demand_ratio, cap, keep, before) in arms.items():
            requested = cache["requested"][None, :] * demand_ratio
            energy, delivered = replay_bucket(charge[None, :], requested,
                                               np.array([min(cache["energy"][0], cap)]), keep, cap,
                                               cap_before_delivery=before)
            for event in events[events.building == building].itertuples():
                start, end = int(event.start) - 1, int(event.end) - 1
                segment = slice(start, end)
                deficit = np.maximum(requested[0, segment] - delivered[0, segment], 0)
                demand_energy = requested[0, segment].sum() * DT
                event_rows.append({
                    "building": building, "event": event.event, "arm": name,
                    "initially_heating_near_setpoint": event.initially_heating_near_setpoint,
                    "observed_large_drop": event.drop_gt_0p5c, "observed_drop_c": event.drop_c,
                    "dhw_first_hour": event.dhw_first_hour,
                    "inventory_min_wh_m2": energy[0, segment].min(),
                    "shortage_energy_fraction": deficit.sum() * DT / max(demand_energy, 1e-8),
                    "has_shortage": bool(np.any(deficit > 1)),
                    "shortage_above_10pct": bool(deficit.sum() * DT > .1 * max(demand_energy, 1e-8)),
                })
        print(f"Replayed accounting interventions: {building}", flush=True)
    pd.DataFrame(source_rows).to_csv(args.output / "known_input_source_replay.csv", index=False)
    frame = pd.DataFrame(event_rows)
    frame.to_csv(args.output / "accounting_event_replays.csv", index=False)
    summary = frame[frame.initially_heating_near_setpoint].groupby(["arm", "observed_large_drop"]).agg(
        events=("event", "size"), minimum_inventory_wh_m2=("inventory_min_wh_m2", "mean"),
        shortage_fraction=("has_shortage", "mean"), shortage_above_10pct=("shortage_above_10pct", "mean"),
        mean_shortage_energy_fraction=("shortage_energy_fraction", "mean"))
    summary.to_csv(args.output / "accounting_replay_summary.csv")


def write_report(args):
    checks = pd.read_csv(args.output / "source_checks.csv")
    scores = pd.read_csv(args.output / "source_probe_scores.csv")
    known = pd.read_csv(args.output / "known_input_source_replay.csv")
    replays = pd.read_csv(args.output / "accounting_replay_summary.csv")
    windows = pd.read_csv(args.output / "source_window_checks.csv")
    window_summary = windows[windows.split == "test"].groupby(["policy", "block_hours"])[[
        "ratio_q01", "ratio_q99", "outside_1_8_fraction", "outside_1_8_charge_fraction"]].mean().reset_index()
    test = scores[scores.split == "test"]
    summary = scores.groupby(["split", "policy", "model"])[["relative_rmse", "energy_bias_fraction"]].mean().reset_index()
    contrasts = []
    rng = np.random.default_rng(13)
    for policy, frame in test.groupby("policy"):
        pivot = frame.pivot(index="building", columns="model", values="relative_rmse")
        differences = (pivot.weather_design - pivot.weather).to_numpy()
        bootstrap = rng.choice(differences, (2000, len(differences)), replace=True).mean(axis=1)
        contrasts.append({"policy": policy, "relative_rmse_difference": differences.mean(),
                          "lower95": np.quantile(bootstrap, .025), "upper95": np.quantile(bootstrap, .975)})
    contrasts = pd.DataFrame(contrasts)
    contrasts.to_csv(args.output / "source_probe_bootstrap.csv", index=False)
    findings = {
        "building_policy_records": len(checks),
        "negative_charge_steps": int(checks.negative_charge_steps.sum()),
        "negative_charge_energy_fraction": float(-checks.negative_charge_wh_m2.sum() / checks.total_charge_wh_m2.sum()),
        "charge_during_off_wh_m2_sum": float(checks.charge_during_off_wh_m2.sum()),
        "charge_during_dhw_wh_m2_sum": float(checks.charge_during_dhw_wh_m2.sum()),
        "charge_without_sh_electric_wh_m2_sum": float(checks.charge_without_sh_electric_wh_m2.sum()),
        "charge_without_sh_electric_energy_fraction": float(checks.charge_without_sh_electric_wh_m2.sum() / checks.total_charge_wh_m2.sum()),
        "charge_during_dhw_energy_fraction": float(checks.charge_during_dhw_wh_m2.sum() / checks.total_charge_wh_m2.sum()),
        "charge_alias_equal_records": int(checks.charge_alias_equal.sum()),
        "building_heat_to_charge_min": float(checks.annual_building_heat_to_charge_ratio.min()),
        "building_heat_to_charge_max": float(checks.annual_building_heat_to_charge_ratio.max()),
        "raw_generation_bias_fraction": float(known.raw_generated_wh_m2.sum() / known.recorded_charge_wh_m2.sum() - 1),
        "known_input_probe_generation_bias_fraction": float(known.probe_generated_wh_m2.sum() / known.recorded_charge_wh_m2.sum() - 1),
    }
    (args.output / "findings.json").write_text(json.dumps(findings, indent=2))
    table = lambda frame: frame.to_html(index=False, float_format=lambda value: f"{value:.4f}", border=0)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Source and buffer accounting diagnostics</title>
    <style>body{{font:15px system-ui;margin:32px;max-width:1250px;color:#222}}table{{border-collapse:collapse;font-size:13px}}
    td,th{{padding:7px;text-align:right;border-bottom:1px solid #ddd}}th{{text-align:left}}pre{{white-space:pre-wrap}}</style></head><body>
    <h1>Source and buffer accounting diagnostics</h1>
    <p>No emulator weights, paths or production configuration were changed. The source regressions are training-only diagnostic fits;
    ridge strength was selected on validation buildings. Calibration buildings were not used for fitting or selection.</p>
    <h2>Retained charging channel</h2><pre>{html.escape(json.dumps(findings, indent=2))}</pre>
    <p>Charging is zero throughout requested lockout. However, small signed exchanges, transfer during recorded DHW mode,
    and transfer without concurrent SH electricity mean this is not an exact instantaneous COP-times-power target.
    It can constrain longer-block source energy, subject to unresolved sensor semantics and time aggregation.
    The named delivered-SH alias is not independent evidence. Building heat can exceed recorded charging annually:
    pump/distribution heat, initial/final stored energy and export definitions are unresolved. Do not enforce exact equality of these
    measured annual totals or call their ratio a verified compressor COP.</p>
    <h3>Aggregation sensitivity on held-out buildings</h3>{table(window_summary)}
    <p>Ratios exclude blocks with mean SH electricity below 1 W/m2. The 1-8 range is the current emulator's conversion bound,
    not a claim that each physical source-side transfer ratio must lie within it. Longer blocks test whether transient mismatch washes out.</p>
    <h2>Predicting effective conversion from known inputs</h2>
    <p>Hourly source-heat regression. Weather-only uses outdoor temperature; weather/design additionally uses design supply temperature,
    reference COP, their weather interaction, and weather curvature. These additions are tested jointly, not separately attributed.
    Each building/policy has equal fitting weight. Relative RMSE is normalized by mean active-hour source heat.</p>{table(summary)}
    <h3>Paired building bootstrap: design minus weather-only test error</h3>{table(contrasts)}
    <p>Applying the learned conversion to the frozen emulator's mean SH electric power changes aggregate source-energy bias from
    {100*findings['raw_generation_bias_fraction']:.1f}% to {100*findings['known_input_probe_generation_bias_fraction']:.1f}%.
    This is a diagnostic substitution, not a retrained model or a claim about KPI calibration.</p>
    <h2>Accounting interventions</h2>
    <p>All replays use the same frozen ensemble-mean heat request and annual start. The baseline is itself a mean-forcing
    bucket replay: nonlinear filtering of the mean is not the mean of particle rollouts. Source interventions use observed charging;
    boundary interventions use the observed annual building/zone heat ratio, held constant within each building.
    Both are privileged diagnostics, not deployable forecasts. The no_loss arms disable standing losses; aftercap arms cap storage
    after delivery to permit within-interval throughput. All other arms retain the original learned loss and pre-delivery capping.
    These are not the previously trained physical-volume models, and fixing their heat requests prevents thermal/controller feedback.</p>{table(replays)}
    <p>The event subset is initially heated and near setpoint, including successful events. Shortage means at least one interval more
    than 1 W/m2 below frozen requested heat; the second criterion requires over 10% event-integrated unmet heat.
    These are accounting indicators, not forecasts of temperature drops.</p>
    <h2>Provenance</h2><pre>{html.escape((args.output / 'request.json').read_text())}</pre></body></html>"""
    (args.output / "index.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force-off-dataset", type=Path, default=Path("data/control=force_off"))
    parser.add_argument("--setpoint-dataset", type=Path, default=Path("data/control=setpoint_identification_dhw_fixed"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    request = {key: str(value) for key, value in vars(args).items()}
    request["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    request["bucket_diagnostic_sha256"] = hashlib.sha256(Path(__file__).with_name("buffer_energy_diagnostics.py").read_bytes()).hexdigest()
    request["previous_request"] = json.loads((args.previous / "request.json").read_text())
    request_path = args.output / "request.json"
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError("Use a fresh output directory when changing diagnostic provenance")
    request_path.write_text(json.dumps(request, indent=2))
    inventory = pd.read_csv(args.previous / "data_inventory.csv")
    probe_path = args.output / "source_probes.json"
    learned = json.loads(probe_path.read_text()) if probe_path.exists() else inventory_and_probes(args, inventory)
    if not (args.output / "accounting_replay_summary.csv").exists():
        accounting_replays(args, inventory, learned)
    write_report(args)


if __name__ == "__main__":
    main()
