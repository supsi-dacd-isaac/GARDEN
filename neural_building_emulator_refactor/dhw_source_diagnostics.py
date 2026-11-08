"""Read-only audit of nominal DHW recovery power versus retained charging data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .plant_accounting_experiment import BASE, OUTPUT


def summarize_building(data, metadata):
    plant = data[data.zone == "technical_room"].sort_values("datetime")
    if plant.datetime.duplicated().any() or not np.all(
        np.diff(plant.datetime.to_numpy(dtype="datetime64[ns]")) == np.timedelta64(15, "m")
    ):
        raise ValueError("Expected unique quarter-hour plant observations")
    water = data[data.zone.str.lower().str.startswith("fl")].groupby("datetime")[
        "input_dhw_mixed_water_l"
    ].sum(min_count=1)
    np.testing.assert_array_equal(water.index.to_numpy(), plant.datetime.to_numpy())
    plant, water = plant.iloc[1:], water.iloc[1:]
    nominal = float(metadata["gardenCompatibility"]["values"]["DHW_recovery_W"])
    charge = plant.thermal_storage_charge_dhw_kw.to_numpy(float) * 1000
    active = ((plant.hp_mode_is_dhw.to_numpy() > .5)
              & (plant.heat_pump_electric_power.to_numpy() > 10))
    if nominal <= 0 or not active.any() or not np.isfinite(charge).all() or not np.isfinite(water).all():
        raise ValueError("DHW audit requires finite signals, positive nominal power and active observations")
    annual_wh = float(charge.sum() * .25)
    litres = float(water.sum())
    if litres <= 0:
        raise ValueError("DHW audit requires positive annual mixed-water draw")
    ratio = charge[active] / nominal
    return dict(
        steps=len(plant), observed_active_fraction=active.mean(),
        nominal_recovery_w=nominal,
        resolved_nominal_recovery_w=float(metadata["resolved_plant_design"]["dhw_tank_source_capacity_w"]),
        resistance_capacity_w=float(metadata["resolved_plant_design"].get("dhw_resistance_capacity_w") or 0),
        mean_active_charge_w=charge[active].mean(), mean_active_charge_over_nominal=ratio.mean(),
        q95_active_charge_over_nominal=np.quantile(ratio, .95),
        active_fraction_above_surrogate_max=np.mean(ratio > 1.5),
        annual_charge_wh=annual_wh, annual_mixed_water_l=litres,
        hp_source_equivalent_draw_delta_k=annual_wh / (1.163 * litres),
        charge_energy_without_active_fraction=charge[~active].sum() / charge.sum(),
    )


def run(dataset, split_path, output):
    split = json.loads(split_path.read_text())
    columns = ["zone", "datetime", "hp_mode_is_dhw", "heat_pump_electric_power",
               "thermal_storage_charge_dhw_kw", "input_dhw_mixed_water_l"]
    rows = []
    for fold in ("train", "test"):
        for building in split[f"{fold}_ids"]:
            folder = dataset / f"egid={building}"
            metadata = json.loads((folder / "simulation_metadata.json").read_text())
            data = pd.read_parquet(folder / "timeseries.parquet", columns=columns)
            rows.append(dict(building=building, split=fold, **summarize_building(data, metadata)))
    frame = pd.DataFrame(rows)
    summary = frame.groupby("split").agg(
        buildings=("building", "size"),
        median_active_charge_over_nominal=("mean_active_charge_over_nominal", "median"),
        buildings_mean_charge_above_surrogate_max=("mean_active_charge_over_nominal", lambda x: int((x > 1.5).sum())),
        mean_active_fraction_above_surrogate_max=("active_fraction_above_surrogate_max", "mean"),
        median_hp_source_equivalent_draw_delta_k=("hp_source_equivalent_draw_delta_k", "median"),
        mean_observed_active_fraction=("observed_active_fraction", "mean"),
    ).reset_index()
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "buildings.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "request.json").write_text(json.dumps(dict(
        dataset=str(dataset), split_path=str(split_path), split=split,
        note="Diagnostic only; no fitting or changes to the running DHW experiment. "
             "Charging power is a retained plant channel, not a direct measurement of compressor COP.",
    ), indent=2))
    print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/control=force_off"))
    parser.add_argument("--split", type=Path, default=BASE / "split_manifest.json")
    parser.add_argument("--output", type=Path, default=OUTPUT / "dhw_source_diagnostics")
    args = parser.parse_args()
    run(args.dataset, args.split, args.output)


if __name__ == "__main__":
    main()
