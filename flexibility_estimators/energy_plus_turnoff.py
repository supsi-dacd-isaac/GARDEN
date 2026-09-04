"""Turn-off flexibility analysis for EnergyPlus force-off exports.

Because the analysis uses HP electric power only, a force-off zeroes that
signal, so relative reductions are identically 1. This script therefore targets
absolute shed:

    DeltaP = P_base - P_off   [kW]

Expected layout under --data-dir (default: data/control=force_off):

    egid=<id>/
      timeseries.parquet
      simulation_metadata.json

Each timeseries is zone-major and must include at least:
  datetime, zone, heat_pump_electric_power, hp_force_off_effective,
  Environment, Site Outdoor Air Drybulb Temperature
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "control=force_off"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "energy_plus_turnoff_plots"

TS_COLUMNS = [
    "datetime",
    "zone",
    "heat_pump_electric_power",
    "hp_force_off_effective",
    "Environment, Site Outdoor Air Drybulb Temperature",
]
OUTDOOR_COLUMN = "Environment, Site Outdoor Air Drybulb Temperature"
DT_HOURS = 0.25  # 15-minute EnergyPlus export


def _egid_dirs(data_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in data_dir.iterdir()
        if path.is_dir() and path.name.startswith("egid=")
    )


def _load_metadata(building_dir: Path) -> dict:
    meta_path = building_dir / "simulation_metadata.json"
    payload = json.loads(meta_path.read_text())
    values = dict(payload.get("gardenCompatibility", {}).get("values", {}))
    if "egid" not in values:
        values["egid"] = int(building_dir.name.split("=", 1)[1])
    meter = payload.get("meterSummary", {})
    values["annual_house_load_kwh"] = meter.get("total_house_load (kWh)")
    infiltration = payload.get("resolved_infiltration", {})
    values["resolved_ach_h_1"] = infiltration.get("resolved_ach_h_1")
    return values


def _is_space_heating_hp(meta: dict) -> bool:
    binding = str(meta.get("hp_size_binding", "")).strip().upper()
    capacity = meta.get("hp_ref_capacity_W")
    try:
        capacity_ok = capacity is not None and float(capacity) > 0.0
    except (TypeError, ValueError):
        capacity_ok = False
    return binding == "SH" and capacity_ok


def load_building_panel(
    data_dir: Path,
    *,
    max_buildings: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load HP buildings into wide panels indexed by datetime.

    Returns
    -------
    meta
        One row per egid with gardenCompatibility fields plus annual HP energy.
    power_kw
        Heat-pump electric power [kW], columns = egid.
    force_off
        Effective force-off flag in {0, 1}, columns = egid.
    outdoor_c
        Outdoor dry-bulb temperature [°C], columns = egid.
    """
    dirs = _egid_dirs(data_dir)
    if max_buildings is not None:
        dirs = dirs[: max(0, int(max_buildings))]

    meta_rows: list[dict] = []
    power_cols: dict[int, pd.Series] = {}
    force_cols: dict[int, pd.Series] = {}
    outdoor_cols: dict[int, pd.Series] = {}

    for index, building_dir in enumerate(dirs, start=1):
        meta = _load_metadata(building_dir)
        if not _is_space_heating_hp(meta):
            continue

        egid = int(meta["egid"])
        frame = pd.read_parquet(building_dir / "timeseries.parquet", columns=TS_COLUMNS)
        frame["datetime"] = pd.to_datetime(frame["datetime"])

        technical = (
            frame.loc[frame["zone"] == "technical_room", ["datetime", "heat_pump_electric_power"]]
            .drop_duplicates("datetime")
            .sort_values("datetime")
            .set_index("datetime")
        )
        building = (
            frame.loc[
                frame["zone"] == "building_total",
                ["datetime", "hp_force_off_effective", OUTDOOR_COLUMN],
            ]
            .drop_duplicates("datetime")
            .sort_values("datetime")
            .set_index("datetime")
        )

        power_w = technical["heat_pump_electric_power"].astype(float)
        force = building["hp_force_off_effective"].astype(float)
        outdoor = building[OUTDOOR_COLUMN].astype(float)

        # Align on the building clock; keep only overlapping timestamps.
        aligned = pd.concat(
            {
                "power_w": power_w,
                "force_off": force,
                "outdoor_c": outdoor,
            },
            axis=1,
        ).sort_index()
        aligned = aligned.dropna(subset=["power_w", "force_off"])
        if aligned.empty:
            continue

        power_kw = aligned["power_w"] / 1000.0
        annual_hp_kwh = float((aligned["power_w"] * DT_HOURS / 1000.0).sum())
        floor_area = float(meta.get("floor_area", np.nan))
        total_floors = float(meta.get("totalFloors", np.nan))
        heated_area = floor_area * total_floors

        meta_rows.append(
            {
                **meta,
                "egid": egid,
                "annual_consumption": annual_hp_kwh,
                "m2": heated_area,
                "U": float(meta.get("surface_to_volume_ratio", np.nan)),
                "resolved_ach_h_1": meta.get("resolved_ach_h_1"),
            }
        )
        power_cols[egid] = power_kw.rename(egid)
        force_cols[egid] = aligned["force_off"].rename(egid)
        outdoor_cols[egid] = aligned["outdoor_c"].rename(egid)

        if index % 50 == 0 or index == len(dirs):
            print(f"loaded {index}/{len(dirs)} folders; kept {len(meta_rows)} HP buildings")

    if not meta_rows:
        raise ValueError(f"No space-heating HP buildings found under {data_dir}")

    meta = pd.DataFrame(meta_rows).set_index("egid").sort_index()
    power_kw = pd.concat(power_cols, axis=1).sort_index()
    force_off = pd.concat(force_cols, axis=1).sort_index()
    outdoor_c = pd.concat(outdoor_cols, axis=1).sort_index()

    # Keep only buildings present in all panels.
    common = meta.index.intersection(power_kw.columns).intersection(force_off.columns)
    meta = meta.loc[common]
    power_kw = power_kw.loc[:, common]
    force_off = force_off.loc[:, common]
    outdoor_c = outdoor_c.loc[:, common]
    return meta, power_kw, force_off, outdoor_c


def _save_or_show(fig: plt.Figure, output_dir: Path | None, name: str) -> None:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / name, dpi=150, bbox_inches="tight")
    plt.show()


def mean_power_by_force_off(
    power_kw: pd.DataFrame,
    force_off: pd.DataFrame,
) -> pd.DataFrame:
    """Return [2 x n_buildings] mean power for force_off in {0, 1}."""
    baseline = power_kw.where(force_off <= 0.5).mean(axis=0)
    forced = power_kw.where(force_off > 0.5).mean(axis=0)
    return pd.DataFrame({0.0: baseline, 1.0: forced}).T


def absolute_flexibility(mean_by_state: pd.DataFrame) -> pd.Series:
    """DeltaP = P_base - P_off [kW]. Positive means load shed under force-off."""
    baseline = mean_by_state.loc[0.0]
    forced = mean_by_state.loc[1.0]
    return baseline - forced


def time_temp_plot(
    power_kw: pd.Series,
    outdoor_c: pd.Series,
    force_off: pd.Series,
    *,
    temp_bins: pd.Series,
    title: str,
    output_dir: Path | None,
    filename: str,
) -> None:
    frame = pd.DataFrame(
        {
            "value": power_kw.to_numpy(dtype=float),
            "mean_T": temp_bins.to_numpy(),
            "force_off_long": (force_off.to_numpy(dtype=float) > 0.5),
        },
        index=pd.to_datetime(power_kw.index),
    )
    frame["hour"] = frame.index.hour
    grouped = (
        frame.groupby(["hour", "mean_T", "force_off_long"], observed=False)["value"]
        .mean()
        .unstack("mean_T")
    )
    if False not in grouped.index.get_level_values("force_off_long"):
        print(f"skip {title}: no baseline force-off samples")
        return
    if True not in grouped.index.get_level_values("force_off_long"):
        print(f"skip {title}: no active force-off samples")
        return

    base = grouped.xs(False, level="force_off_long")
    flex = grouped.xs(True, level="force_off_long")
    # Absolute shed [kW]: baseline minus force-off.
    delta = base - flex
    delta.index.name = "Hour of day"
    delta.columns.name = "Temperature bins (°C)"

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), layout="constrained")
    sns.heatmap(delta, ax=ax)
    ax.set_xlabel("Temperature bins (°C)")
    ax.set_ylabel("Hour of day")
    ax.set_title(title)
    _save_or_show(fig, output_dir, filename)


def run_analysis(
    data_dir: Path,
    *,
    max_buildings: int | None = None,
    output_dir: Path | None = DEFAULT_OUTPUT_DIR,
    train_ratio: float = 0.8,
    seed: int = 0,
) -> None:
    meta, power_kw, force_off, outdoor_c = load_building_panel(
        data_dir,
        max_buildings=max_buildings,
    )
    print(f"buildings={len(meta)} timesteps={len(power_kw)}")

    # ---------------------------------------------------------------------------
    # Metadata diagnostic: annual HP consumption vs heated area, colored by S/V
    # (proxy for the ODIS U-value scatter; EnergyPlus exports do not store U).
    # ---------------------------------------------------------------------------
    meta_hp = meta.copy()
    meta_small = (
        meta_hp.sample(n=min(1000, len(meta_hp)), random_state=seed)
        if len(meta_hp) > 1000
        else meta_hp
    )
    fig, ax = plt.subplots(1, 1, layout="constrained")
    scatter = ax.scatter(
        meta_small["annual_consumption"],
        meta_small["m2"],
        c=meta_small["U"],
        cmap="viridis",
        alpha=0.8,
    )
    fig.colorbar(scatter, ax=ax, label="surface_to_volume_ratio [1/m]")
    ax.set_xlabel("Annual HP Consumption (kWh)")
    ax.set_ylabel("Heated Area (m²)")
    ax.set_title("Annual HP Consumption vs Heated Area colored by S/V")
    _save_or_show(fig, output_dir, "01_annual_consumption_vs_area.png")

    # ---------------------------------------------------------------------------
    # Mean power distributions for force_off = 0 and force_off = 1
    # ---------------------------------------------------------------------------
    mean_by_state = mean_power_by_force_off(power_kw, force_off)
    fig, ax = plt.subplots(1, 1, layout="constrained")
    mean_by_state.T.plot.hist(alpha=0.5, bins=50, ax=ax)
    ax.set_xlabel("Mean HP electric power (kW) by force-off state")
    ax.set_title("Distribution of mean power when force_off is 0 vs 1")
    _save_or_show(fig, output_dir, "02_mean_power_by_force_off.png")

    # ---------------------------------------------------------------------------
    # Absolute flexibility histogram: DeltaP = P_base - P_off [kW]
    # ---------------------------------------------------------------------------
    p_abs_change = absolute_flexibility(mean_by_state)
    fig, ax = plt.subplots(1, 1, layout="constrained")
    p_abs_change.plot.hist(alpha=0.5, bins=50, ax=ax)
    ax.set_xlabel("Absolute HP power shed between force_off states (kW)")
    ax.set_title(r"$\Delta P = P_{base}-P_{off}$")
    ax.spines[["top", "right"]].set_visible(False)
    _save_or_show(fig, output_dir, "03_absolute_flexibility.png")
    print(
        "absolute flexibility [kW]: "
        f"mean={p_abs_change.mean():.4f} "
        f"std={p_abs_change.std():.4f} "
        f"min={p_abs_change.min():.4f} "
        f"max={p_abs_change.max():.4f}"
    )

    # ---------------------------------------------------------------------------
    # LightGBM: explain absolute shed from static metadata
    # ---------------------------------------------------------------------------
    # Note: annual_consumption is intentionally excluded — with P_off≈0 it is
    # almost identical to DeltaP and would make the regression tautological.
    feature_cols = [
        "buildingType",
        "totalFloors",
        "constructionPeriod",
        "wallsRenovationPeriod",
        "floorsRenovationPeriod",
        "roofRenovationPeriod",
        "windowsRenovationPeriod",
        "dwellingNumber",
        "shSetpoint",
        "floor_area",
        "volume",
        "envelope_area",
        "window_area",
        "surface_to_volume_ratio",
        "thermal_mass_class",
        "hp_ref_capacity_W",
        "hp_ref_cop",
        "SH_design_cap_W",
        "DHW_recovery_W",
        "dhwVolume_m3",
        "shVolume_m3",
        "m2",
        "resolved_ach_h_1",
    ]
    available = [column for column in feature_cols if column in meta_hp.columns]
    meta_features = meta_hp.loc[:, available].apply(pd.to_numeric, errors="coerce")
    target = p_abs_change.loc[meta_features.index].astype(float)
    valid = meta_features.notna().all(axis=1) & target.notna()
    meta_features = meta_features.loc[valid]
    target = target.loc[valid]

    # Drop constant columns (e.g. fixed shSetpoint / thermal_mass_class).
    nunique = meta_features.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        print(f"dropping constant metadata columns: {constant_cols}")
        meta_features = meta_features.drop(columns=constant_cols)

    if len(target) < 4:
        print("skip LightGBM/SHAP: need at least 4 buildings")
    else:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(target))
        n_train = max(2, int(train_ratio * len(target)))
        n_train = min(n_train, len(target) - 1)
        train_idx = order[:n_train]
        test_idx = order[n_train:]
        meta_tr = meta_features.iloc[train_idx]
        meta_te = meta_features.iloc[test_idx]
        y_tr = target.iloc[train_idx]
        y_te = target.iloc[test_idx]

        # Default LightGBM min_child_samples=20 makes tiny cohorts predict a constant.
        min_child_samples = max(1, min(20, n_train // 5))
        model = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=min_child_samples,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            verbosity=-1,
        ).fit(meta_tr, y_tr)
        pred = model.predict(meta_te)
        print(
            "LightGBM: "
            f"n_train={len(meta_tr)} n_test={len(meta_te)} "
            f"min_child_samples={min_child_samples} "
            f"pred_std={float(np.std(pred)):.4f} "
            f"y_test_std={float(y_te.std()):.4f}"
        )
        if float(np.std(pred)) < 1e-6:
            print(
                "warning: LightGBM predictions are nearly constant; "
                "increase --max-buildings or check metadata variance"
            )

        fig, ax = plt.subplots(1, 1, layout="constrained")
        ax.scatter(y_te.to_numpy(), pred, alpha=0.8)
        lims = [
            float(np.nanmin([y_te.min(), pred.min(), 0.0])),
            float(np.nanmax([y_te.max(), pred.max()])),
        ]
        ax.plot(lims, lims, c="k", ls="--")
        ax.set_xlabel("True absolute shed (kW)")
        ax.set_ylabel("Predicted absolute shed (kW)")
        ax.set_title("LightGBM hold-out predictions of absolute HP shed")
        _save_or_show(fig, output_dir, "04_lightgbm_predictions.png")

        try:
            import shap

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(meta_te)
            plt.close("all")
            shap.summary_plot(shap_values, meta_te, show=False)
            fig = plt.gcf()
            fig.tight_layout()
            _save_or_show(fig, output_dir, "05_shap_summary.png")
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001 - keep the rest of the script usable
            print(f"skip SHAP summary: {exc}")

    # ---------------------------------------------------------------------------
    # Absolute shed conditional on outdoor temperature
    # ---------------------------------------------------------------------------
    # Stack buildings into long form using each building's own force-off schedule
    # and outdoor temperature (schedules/weather are not shared across egids).
    long_parts = []
    for egid in power_kw.columns:
        part = pd.DataFrame(
            {
                "power_kw": power_kw[egid],
                "force_off_long": force_off[egid] > 0.5,
                "outdoor_c": outdoor_c[egid],
                "egid": egid,
            }
        ).dropna()
        long_parts.append(part)
    long_df = pd.concat(long_parts, axis=0)

    temps = pd.qcut(long_df["outdoor_c"], np.linspace(0, 1, 6), duplicates="drop")
    long_df = long_df.assign(mean_T=temps)
    flex_bins = (
        long_df.pivot_table(
            index=["mean_T", "force_off_long"],
            columns="egid",
            values="power_kw",
            aggfunc="mean",
            observed=False,
        )
        .sort_index()
    )
    baseline = flex_bins.xs(False, level="force_off_long")
    forced = flex_bins.xs(True, level="force_off_long")
    flex_bins_abs_ch = baseline - forced

    fig, ax = plt.subplots(1, 1, layout="constrained", figsize=(4, 10))
    ordered = flex_bins_abs_ch.T.sort_values(flex_bins_abs_ch.index[0])
    sns.heatmap(ordered, ax=ax)
    ax.set_xlabel("Temperature bins (°C)")
    ax.set_ylabel("egid")
    ax.set_title("Absolute HP shed [kW] by outdoor-temperature bin")
    _save_or_show(fig, output_dir, "06_flexibility_by_temperature.png")

    # ---------------------------------------------------------------------------
    # Hour × temperature heatmaps for two example buildings
    # ---------------------------------------------------------------------------
    # Prefer buildings with both states and non-trivial baseline/absolute shed.
    candidates = p_abs_change.sort_values(ascending=False).index.tolist()
    example_ids = candidates[:2]
    if len(example_ids) < 2 and len(power_kw.columns) >= 2:
        example_ids = list(power_kw.columns[:2])

    # Shared temperature bin edges from the pooled outdoor distribution.
    shared_bins = pd.qcut(
        outdoor_c.stack(),
        np.linspace(0, 1, 6),
        duplicates="drop",
        retbins=True,
    )[1]

    for egid in example_ids:
        building_temps = pd.cut(
            outdoor_c[egid],
            bins=shared_bins,
            include_lowest=True,
            duplicates="drop",
        )
        time_temp_plot(
            power_kw[egid],
            outdoor_c[egid],
            force_off[egid],
            temp_bins=building_temps,
            title=f"Absolute HP shed during force-off [kW] — egid={egid}",
            output_dir=output_dir,
            filename=f"07_time_temp_egid_{egid}.png",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Folder containing egid=*/timeseries.parquet exports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PNG figures are written.",
    )
    parser.add_argument(
        "--max-buildings",
        type=int,
        default=None,
        help="Optional cap on the number of egid folders to load.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Only show figures; do not write PNGs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = None if args.no_save else args.output_dir
    run_analysis(
        args.data_dir,
        max_buildings=args.max_buildings,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
