"""Compare linear and GAM nuisance corrections for the 3 h flexibility KPI.

Both models are fitted jointly over all buildings. They share building-specific
intercepts and upward/downward setpoint slopes; only the common exogenous
nuisance model differs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import sparse
from scipy.sparse.linalg import lsmr
from sklearn.preprocessing import SplineTransformer, StandardScaler

from neural_building_emulator.columns import closed_loop_required_columns
from neural_building_emulator.data import read_dataset_frame, to_closed_loop_profiles

from .flexibility_weather_heatmap import (
    DEFAULT_DATASET,
    WeatherHeatmapConfig,
    _select_space_heating_profiles,
    extract_event_responses,
)


DEFAULT_OUTPUT_DIR = Path(
    "output/neural_building_emulator_refactor/flexibility_joint_gam_study"
)
SMOOTH_FEATURES = (
    "pre_outdoor_temperature_c",
    "delta_outdoor_temperature_c",
    "pre_irradiance_w_m2",
    "delta_irradiance_w_m2",
)
LINEAR_FEATURES = (
    "delta_internal_gain_w_m2",
    "delta_ventilation_m3_s",
)
NUISANCE_FEATURES = (*SMOOTH_FEATURES, *LINEAR_FEATURES)
FEATURE_LABELS = {
    "pre_outdoor_temperature_c": "Pre-event outdoor temperature [C]",
    "delta_outdoor_temperature_c": "Post-pre outdoor temperature [C]",
    "pre_irradiance_w_m2": "Pre-event irradiance [W/m2]",
    "delta_irradiance_w_m2": "Post-pre irradiance [W/m2]",
    "delta_internal_gain_w_m2": "Post-pre internal gains [W/m2]",
    "delta_ventilation_m3_s": "Post-pre ventilation [m3/s]",
}


@dataclass
class NuisanceTransformer:
    kind: str
    n_knots: int
    spline_transformers: list[SplineTransformer]
    linear_scaler: StandardScaler | None
    all_linear_scaler: StandardScaler | None
    spline_slices: list[slice]
    output_dim: int

    @classmethod
    def fit(cls, frame: pd.DataFrame, *, kind: str, n_knots: int) -> "NuisanceTransformer":
        if kind == "linear":
            scaler = StandardScaler().fit(frame.loc[:, NUISANCE_FEATURES])
            return cls(kind, n_knots, [], None, scaler, [], len(NUISANCE_FEATURES))
        if kind != "gam":
            raise ValueError("kind must be 'linear' or 'gam'")
        transformers: list[SplineTransformer] = []
        slices: list[slice] = []
        offset = 0
        for feature in SMOOTH_FEATURES:
            transformer = SplineTransformer(
                n_knots=n_knots,
                degree=3,
                include_bias=False,
                extrapolation="linear",
            ).fit(frame.loc[:, [feature]])
            width = transformer.n_features_out_
            transformers.append(transformer)
            slices.append(slice(offset, offset + width))
            offset += width
        linear_scaler = StandardScaler().fit(frame.loc[:, LINEAR_FEATURES])
        return cls(
            kind,
            n_knots,
            transformers,
            linear_scaler,
            None,
            slices,
            offset + len(LINEAR_FEATURES),
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.kind == "linear":
            assert self.all_linear_scaler is not None
            return self.all_linear_scaler.transform(
                frame.loc[:, NUISANCE_FEATURES]
            ).astype(np.float64)
        assert self.linear_scaler is not None
        smooth = [
            transformer.transform(frame.loc[:, [feature]])
            for transformer, feature in zip(self.spline_transformers, SMOOTH_FEATURES)
        ]
        linear = self.linear_scaler.transform(frame.loc[:, LINEAR_FEATURES])
        return np.column_stack([*smooth, linear]).astype(np.float64)


@dataclass
class JointFlexibilityModel:
    kind: str
    profile_ids: np.ndarray
    transformer: NuisanceTransformer
    coefficients: np.ndarray
    smoothing: float
    solver_iterations: int
    solver_condition: float

    @property
    def n_buildings(self) -> int:
        return len(self.profile_ids)

    @property
    def nuisance_coefficients(self) -> np.ndarray:
        return self.coefficients[3 * self.n_buildings :]

    def building_coefficients(self) -> pd.DataFrame:
        n = self.n_buildings
        return pd.DataFrame(
            {
                "profile_id": self.profile_ids.astype(int),
                "intercept_w_m2": self.coefficients[:n],
                "beta_up_w_m2_k": self.coefficients[n : 2 * n],
                "beta_down_w_m2_k": self.coefficients[2 * n : 3 * n],
            }
        )


def _profile_indices(frame: pd.DataFrame, profile_ids: np.ndarray) -> np.ndarray:
    mapping = {int(profile_id): index for index, profile_id in enumerate(profile_ids)}
    try:
        return np.asarray(
            [mapping[int(profile_id)] for profile_id in frame["profile_id"]],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ValueError(f"Unknown profile in prediction frame: {exc.args[0]}") from exc


def _design_matrix(
    frame: pd.DataFrame,
    profile_ids: np.ndarray,
    transformer: NuisanceTransformer,
) -> sparse.csr_matrix:
    n_rows = len(frame)
    n_buildings = len(profile_ids)
    rows = np.arange(n_rows, dtype=np.int64)
    building = _profile_indices(frame, profile_ids)
    delta = frame["delta_tset_c"].to_numpy(dtype=np.float64)
    upward = np.maximum(delta, 0.0)
    downward = np.maximum(-delta, 0.0)
    intercept = sparse.csr_matrix(
        (np.ones(n_rows), (rows, building)), shape=(n_rows, n_buildings)
    )
    upward_design = sparse.csr_matrix(
        (upward, (rows, building)), shape=(n_rows, n_buildings)
    )
    # The minus sign gives beta_down the same positive interpretation as beta_up.
    downward_design = sparse.csr_matrix(
        (-downward, (rows, building)), shape=(n_rows, n_buildings)
    )
    nuisance = sparse.csr_matrix(transformer.transform(frame))
    return sparse.hstack(
        [intercept, upward_design, downward_design, nuisance], format="csr"
    )


def _smoothness_penalty(
    n_columns: int,
    n_buildings: int,
    transformer: NuisanceTransformer,
    smoothing: float,
) -> sparse.csr_matrix:
    if transformer.kind != "gam" or smoothing <= 0.0:
        return sparse.csr_matrix((0, n_columns))
    penalties: list[sparse.csr_matrix] = []
    nuisance_offset = 3 * n_buildings
    for block in transformer.spline_slices:
        width = block.stop - block.start
        second_difference = np.diff(np.eye(width), n=2, axis=0)
        left = sparse.csr_matrix(
            (len(second_difference), nuisance_offset + block.start)
        )
        right = sparse.csr_matrix(
            (
                len(second_difference),
                n_columns - nuisance_offset - block.stop,
            )
        )
        penalties.append(
            sparse.hstack(
                [left, sparse.csr_matrix(second_difference), right],
                format="csr",
            )
        )
    return np.sqrt(smoothing) * sparse.vstack(penalties, format="csr")


def fit_joint_model(
    frame: pd.DataFrame,
    *,
    kind: str,
    n_knots: int,
    smoothing: float,
    max_iterations: int,
) -> JointFlexibilityModel:
    profile_ids = np.asarray(sorted(frame["profile_id"].astype(int).unique()))
    transformer = NuisanceTransformer.fit(frame, kind=kind, n_knots=n_knots)
    design = _design_matrix(frame, profile_ids, transformer)
    counts = frame.groupby("profile_id")["profile_id"].transform("size").to_numpy(float)
    row_weights = np.sqrt(1.0 / counts)
    weighted_design = design.multiply(row_weights[:, None]).tocsr()
    target = frame["delta_pel_w_m2"].to_numpy(dtype=np.float64)
    weighted_target = target * row_weights
    penalty = _smoothness_penalty(
        design.shape[1], len(profile_ids), transformer, smoothing
    )
    if penalty.shape[0]:
        weighted_design = sparse.vstack([weighted_design, penalty], format="csr")
        weighted_target = np.concatenate([weighted_target, np.zeros(penalty.shape[0])])
    solution = lsmr(
        weighted_design,
        weighted_target,
        atol=1e-8,
        btol=1e-8,
        maxiter=max_iterations,
    )
    return JointFlexibilityModel(
        kind=kind,
        profile_ids=profile_ids,
        transformer=transformer,
        coefficients=np.asarray(solution[0]),
        smoothing=float(smoothing),
        solver_iterations=int(solution[2]),
        solver_condition=float(solution[6]),
    )


def predict(model: JointFlexibilityModel, frame: pd.DataFrame) -> np.ndarray:
    design = _design_matrix(frame, model.profile_ids, model.transformer)
    return np.asarray(design @ model.coefficients).reshape(-1)


def prediction_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    target = frame["delta_pel_w_m2"].to_numpy(dtype=np.float64)
    error = prediction - target
    values = frame.loc[:, ["profile_id"]].copy()
    values["squared_error"] = error**2
    values["absolute_error"] = np.abs(error)
    by_building = values.groupby("profile_id").agg(
        mse=("squared_error", "mean"),
        mae=("absolute_error", "mean"),
    )
    total = float(np.sum((target - np.mean(target)) ** 2))
    return {
        "rmse_w_m2": float(np.sqrt(np.mean(error**2))),
        "mae_w_m2": float(np.mean(np.abs(error))),
        "r2": 1.0 - float(np.sum(error**2)) / total if total > 1e-12 else np.nan,
        "building_balanced_rmse_w_m2": float(np.sqrt(by_building["mse"].mean())),
        "mean_building_rmse_w_m2": float(np.sqrt(by_building["mse"]).mean()),
        "median_building_rmse_w_m2": float(np.sqrt(by_building["mse"]).median()),
        "mean_building_mae_w_m2": float(by_building["mae"].mean()),
    }


def _per_building_metrics(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    prefix: str,
) -> pd.DataFrame:
    values = frame.loc[:, ["profile_id"]].copy()
    error = prediction - frame["delta_pel_w_m2"].to_numpy(dtype=np.float64)
    values["squared_error"] = error**2
    values["absolute_error"] = np.abs(error)
    result = values.groupby("profile_id").agg(
        mse=("squared_error", "mean"),
        mae=("absolute_error", "mean"),
        event_count=("squared_error", "size"),
    )
    result[f"{prefix}_rmse_w_m2"] = np.sqrt(result.pop("mse"))
    result[f"{prefix}_mae_w_m2"] = result.pop("mae")
    return result.reset_index()


def _assign_time_block_split(
    events: pd.DataFrame,
    *,
    block_days: int,
    seed: int,
) -> pd.DataFrame:
    values = events.copy()
    times = pd.to_datetime(values["event_time"])
    origin = times.min().normalize()
    block = ((times - origin).dt.total_seconds() // (block_days * 86400)).astype(int)
    fold = (block + seed) % 5
    values["split"] = np.where(fold.eq(0), "test", np.where(fold.eq(1), "validation", "train"))
    return values


def _load_events(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, int]]:
    dataset = Path(args.dataset)
    selected_ids, selection = _select_space_heating_profiles(
        dataset,
        max_profiles=args.max_profiles,
        selection=args.profile_selection,
        seed=args.seed,
    )
    config = WeatherHeatmapConfig(
        horizons_hours=(args.horizon_hours,),
        min_setpoint_change_c=args.min_setpoint_change_c,
        weather_window="pre",
        min_buildings_per_bin=1,
        line_dashboard_horizon_hours=args.horizon_hours,
    )
    frames: list[pd.DataFrame] = []
    columns = closed_loop_required_columns()
    for start in range(0, len(selected_ids), args.read_batch_size):
        batch = selected_ids[start : start + args.read_batch_size]
        raw = read_dataset_frame(dataset, columns=columns, profile_ids=batch)
        profiles = to_closed_loop_profiles(
            raw,
            include_space_heating_availability=True,
            include_internal_gains=True,
            hp_power_area_normalization="building_heated_area",
        )
        for profile in profiles:
            events, _ = extract_event_responses(profile, config)
            if not events.empty:
                frames.append(events)
        print(
            f"processed_profiles={min(start + len(batch), len(selected_ids))}/"
            f"{len(selected_ids)}",
            flush=True,
        )
    if not frames:
        raise ValueError("No valid setpoint events were found")
    events = pd.concat(frames, ignore_index=True)
    required = ["delta_pel_w_m2", "delta_tset_c", *NUISANCE_FEATURES]
    finite = np.all(np.isfinite(events.loc[:, required]), axis=1)
    events = events.loc[finite].reset_index(drop=True)
    return events, selection


def _binned_residual(
    values: np.ndarray,
    residual: np.ndarray,
    *,
    bins: int = 20,
) -> pd.DataFrame:
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
    labels = np.digitize(values, edges[1:-1])
    frame = pd.DataFrame({"value": values, "residual": residual, "bin": labels})
    return frame.groupby("bin").agg(
        value=("value", "mean"),
        residual=("residual", "mean"),
        residual_std=("residual", "std"),
        count=("residual", "size"),
    ).reset_index()


def _comparison_dashboard(
    test: pd.DataFrame,
    linear_prediction: np.ndarray,
    gam_prediction: np.ndarray,
    per_building: pd.DataFrame,
    coefficients: pd.DataFrame,
    path: Path,
    seed: int,
) -> None:
    figure = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Observed versus predicted",
            "Residual over pre-event Tout",
            "Residual over change in Tout",
            "Per-building held-out RMSE",
            "Upward beta",
            "Downward beta",
        ),
        horizontal_spacing=0.09,
        vertical_spacing=0.16,
    )
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(test), size=min(25000, len(test)), replace=False)
    target = test["delta_pel_w_m2"].to_numpy(dtype=float)
    for name, prediction_values, color in (
        ("Linear nuisance", linear_prediction, "#2878b5"),
        ("Shared GAM", gam_prediction, "#d84a3a"),
    ):
        figure.add_trace(
            go.Scattergl(
                x=target[sample],
                y=prediction_values[sample],
                mode="markers",
                marker={"size": 4, "opacity": 0.18, "color": color},
                name=name,
                legendgroup=name,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        residual = prediction_values - target
        for column, feature in ((2, SMOOTH_FEATURES[0]), (3, SMOOTH_FEATURES[1])):
            summary = _binned_residual(
                test[feature].to_numpy(dtype=float), residual
            )
            figure.add_trace(
                go.Scatter(
                    x=summary["value"],
                    y=summary["residual"],
                    mode="lines+markers",
                    line={"width": 3, "color": color},
                    name=name,
                    legendgroup=name,
                    showlegend=False,
                ),
                row=1,
                col=column,
            )
    limits = np.quantile(np.abs(np.concatenate([target, linear_prediction, gam_prediction])), 0.995)
    figure.add_shape(
        type="line", x0=-limits, y0=-limits, x1=limits, y1=limits,
        line={"color": "#5d6570", "dash": "dot"}, row=1, col=1,
    )
    figure.add_hline(y=0.0, line_dash="dot", line_color="#5d6570", row=1, col=2)
    figure.add_hline(y=0.0, line_dash="dot", line_color="#5d6570", row=1, col=3)

    figure.add_trace(
        go.Scattergl(
            x=per_building["linear_rmse_w_m2"],
            y=per_building["gam_rmse_w_m2"],
            mode="markers",
            marker={"size": 6, "opacity": 0.55, "color": "#2a9d8f"},
            name="Buildings",
            showlegend=False,
            text=per_building["profile_id"],
            hovertemplate="building=%{text}<br>linear=%{x:.3f}<br>GAM=%{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    rmse_limit = float(
        np.quantile(
            np.concatenate(
                [per_building["linear_rmse_w_m2"], per_building["gam_rmse_w_m2"]]
            ),
            0.995,
        )
    )
    figure.add_shape(
        type="line", x0=0, y0=0, x1=rmse_limit, y1=rmse_limit,
        line={"color": "#5d6570", "dash": "dot"}, row=2, col=1,
    )
    for column, coefficient in ((2, "beta_up_w_m2_k"), (3, "beta_down_w_m2_k")):
        x = coefficients[f"linear_{coefficient}"]
        y = coefficients[f"gam_{coefficient}"]
        figure.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                marker={"size": 6, "opacity": 0.55, "color": "#e76f51"},
                name="Buildings",
                showlegend=False,
                text=coefficients["profile_id"],
                hovertemplate="building=%{text}<br>linear=%{x:.3f}<br>GAM=%{y:.3f}<extra></extra>",
            ),
            row=2,
            col=column,
        )
        coefficient_limit = float(np.quantile(np.abs(np.concatenate([x, y])), 0.995))
        figure.add_shape(
            type="line", x0=-coefficient_limit, y0=-coefficient_limit,
            x1=coefficient_limit, y1=coefficient_limit,
            line={"color": "#5d6570", "dash": "dot"}, row=2, col=column,
        )
    figure.update_xaxes(title_text="Observed delta Pel [W/m2]", row=1, col=1)
    figure.update_yaxes(title_text="Predicted delta Pel [W/m2]", row=1, col=1)
    figure.update_xaxes(title_text=FEATURE_LABELS[SMOOTH_FEATURES[0]], row=1, col=2)
    figure.update_xaxes(title_text=FEATURE_LABELS[SMOOTH_FEATURES[1]], row=1, col=3)
    figure.update_yaxes(title_text="Mean residual [W/m2]", row=1, col=2)
    figure.update_yaxes(title_text="Mean residual [W/m2]", row=1, col=3)
    figure.update_xaxes(title_text="Linear RMSE [W/m2]", row=2, col=1)
    figure.update_yaxes(title_text="GAM RMSE [W/m2]", row=2, col=1)
    figure.update_xaxes(title_text="Linear beta [W/(m2 K)]", row=2, col=2)
    figure.update_yaxes(title_text="GAM beta [W/(m2 K)]", row=2, col=2)
    figure.update_xaxes(title_text="Linear beta [W/(m2 K)]", row=2, col=3)
    figure.update_yaxes(title_text="GAM beta [W/(m2 K)]", row=2, col=3)
    figure.update_layout(
        title={
            "text": "Joint flexibility regression: linear nuisance versus shared GAM",
            "x": 0.5,
        },
        template="plotly_white",
        width=1750,
        height=1000,
        font={"size": 14},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": 1.03},
        margin={"l": 90, "r": 50, "t": 120, "b": 80},
    )
    figure.update_annotations(font={"size": 18})
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path)


def _partial_effect_dashboard(
    linear: JointFlexibilityModel,
    gam: JointFlexibilityModel,
    fit_frame: pd.DataFrame,
    path: Path,
) -> None:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=tuple(FEATURE_LABELS[feature] for feature in SMOOTH_FEATURES),
        horizontal_spacing=0.1,
        vertical_spacing=0.16,
    )
    reference = {feature: float(fit_frame[feature].median()) for feature in NUISANCE_FEATURES}
    for index, feature in enumerate(SMOOTH_FEATURES):
        row, column = divmod(index, 2)
        grid = np.linspace(
            fit_frame[feature].quantile(0.01),
            fit_frame[feature].quantile(0.99),
            150,
        )
        synthetic = pd.DataFrame(
            {name: np.full(len(grid), value) for name, value in reference.items()}
        )
        synthetic[feature] = grid
        for model, name, color in (
            (linear, "Linear nuisance", "#2878b5"),
            (gam, "Shared GAM", "#d84a3a"),
        ):
            effect = model.transformer.transform(synthetic) @ model.nuisance_coefficients
            midpoint = model.transformer.transform(
                pd.DataFrame({key: [value] for key, value in reference.items()})
            ) @ model.nuisance_coefficients
            figure.add_trace(
                go.Scatter(
                    x=grid,
                    y=effect - float(midpoint[0]),
                    mode="lines",
                    line={"width": 3, "color": color},
                    name=name,
                    legendgroup=name,
                    showlegend=index == 0,
                ),
                row=row + 1,
                col=column + 1,
            )
        figure.update_xaxes(
            title_text=FEATURE_LABELS[feature], row=row + 1, col=column + 1
        )
        figure.update_yaxes(
            title_text="Centered nuisance effect [W/m2]",
            row=row + 1,
            col=column + 1,
        )
    figure.update_layout(
        title={"text": "Learned common exogenous nuisance functions", "x": 0.5},
        template="plotly_white",
        width=1450,
        height=900,
        font={"size": 14},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": 1.04},
        margin={"l": 100, "r": 50, "t": 120, "b": 80},
    )
    figure.update_annotations(font={"size": 18})
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path)


def run_study(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events, selection = _load_events(args)
    events = _assign_time_block_split(
        events, block_days=args.holdout_block_days, seed=args.seed
    )
    events.to_parquet(output_dir / "event_regression_data.parquet", index=False)
    train = events[events["split"].eq("train")].reset_index(drop=True)
    validation = events[events["split"].eq("validation")].reset_index(drop=True)
    test = events[events["split"].eq("test")].reset_index(drop=True)
    print(
        f"events train={len(train)} validation={len(validation)} test={len(test)} "
        f"buildings={events['profile_id'].nunique()}"
    )

    validation_rows: list[dict[str, float | int]] = []
    for smoothing in args.gam_smoothing_grid:
        candidate = fit_joint_model(
            train,
            kind="gam",
            n_knots=args.gam_n_knots,
            smoothing=smoothing,
            max_iterations=args.solver_max_iterations,
        )
        metrics = prediction_metrics(validation, predict(candidate, validation))
        validation_rows.append({"smoothing": smoothing, **metrics})
        print(
            f"gam_smoothing={smoothing:g} "
            f"validation_building_balanced_rmse="
            f"{metrics['building_balanced_rmse_w_m2']:.6f}"
        )
    validation_table = pd.DataFrame(validation_rows)
    validation_table.to_csv(output_dir / "gam_smoothing_validation.csv", index=False)
    best_smoothing = float(
        validation_table.loc[
            validation_table["building_balanced_rmse_w_m2"].idxmin(), "smoothing"
        ]
    )
    fit_frame = pd.concat([train, validation], ignore_index=True)
    linear = fit_joint_model(
        fit_frame,
        kind="linear",
        n_knots=args.gam_n_knots,
        smoothing=0.0,
        max_iterations=args.solver_max_iterations,
    )
    gam = fit_joint_model(
        fit_frame,
        kind="gam",
        n_knots=args.gam_n_knots,
        smoothing=best_smoothing,
        max_iterations=args.solver_max_iterations,
    )
    linear_prediction = predict(linear, test)
    gam_prediction = predict(gam, test)
    linear_metrics = prediction_metrics(test, linear_prediction)
    gam_metrics = prediction_metrics(test, gam_prediction)

    per_building = _per_building_metrics(test, linear_prediction, "linear").merge(
        _per_building_metrics(test, gam_prediction, "gam"),
        on=["profile_id", "event_count"],
        how="inner",
    )
    per_building["rmse_improvement_w_m2"] = (
        per_building["linear_rmse_w_m2"] - per_building["gam_rmse_w_m2"]
    )
    per_building.to_csv(output_dir / "per_building_test_metrics.csv", index=False)
    coefficients = linear.building_coefficients().merge(
        gam.building_coefficients(),
        on="profile_id",
        suffixes=("_linear", "_gam"),
    )
    coefficients = coefficients.rename(
        columns={
            "intercept_w_m2_linear": "linear_intercept_w_m2",
            "beta_up_w_m2_k_linear": "linear_beta_up_w_m2_k",
            "beta_down_w_m2_k_linear": "linear_beta_down_w_m2_k",
            "intercept_w_m2_gam": "gam_intercept_w_m2",
            "beta_up_w_m2_k_gam": "gam_beta_up_w_m2_k",
            "beta_down_w_m2_k_gam": "gam_beta_down_w_m2_k",
        }
    )
    coefficients.to_csv(output_dir / "building_flexibility_coefficients.csv", index=False)
    _comparison_dashboard(
        test,
        linear_prediction,
        gam_prediction,
        per_building,
        coefficients,
        output_dir / "linear_vs_gam_dashboard.html",
        args.seed,
    )
    _partial_effect_dashboard(
        linear,
        gam,
        fit_frame,
        output_dir / "nuisance_partial_effects.html",
    )
    summary: dict[str, object] = {
        "dataset": str(args.dataset),
        "horizon_hours": args.horizon_hours,
        "selection": selection,
        "event_counts": events["split"].value_counts().to_dict(),
        "holdout": {
            "method": "calendar blocks assigned to train/validation/test by block modulo 5",
            "block_days": args.holdout_block_days,
            "seed_offset": args.seed,
        },
        "nuisance_features": {
            "smooth_in_gam": list(SMOOTH_FEATURES),
            "linear_in_gam": list(LINEAR_FEATURES),
            "all_linear_baseline": list(NUISANCE_FEATURES),
        },
        "gam_n_knots": args.gam_n_knots,
        "selected_gam_smoothing": best_smoothing,
        "linear_test": linear_metrics,
        "gam_test": gam_metrics,
        "gam_relative_improvement": {
            name: (
                (linear_metrics[name] - gam_metrics[name]) / linear_metrics[name]
                if linear_metrics[name] != 0.0
                else np.nan
            )
            for name in (
                "rmse_w_m2",
                "mae_w_m2",
                "building_balanced_rmse_w_m2",
                "mean_building_rmse_w_m2",
            )
        },
        "fraction_buildings_with_lower_gam_rmse": float(
            np.mean(per_building["rmse_improvement_w_m2"] > 0.0)
        ),
        "solver": {
            "linear_iterations": linear.solver_iterations,
            "linear_condition": linear.solver_condition,
            "gam_iterations": gam.solver_iterations,
            "gam_condition": gam.solver_condition,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True)
    )
    print(json.dumps({"linear_test": linear_metrics, "gam_test": gam_metrics}, indent=2))
    print(f"saved_dashboard={output_dir / 'linear_vs_gam_dashboard.html'}")
    print(f"saved_summary={output_dir / 'summary.json'}")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jointly compare linear and GAM nuisance models for flexibility."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-profiles", type=int, default=None)
    parser.add_argument("--profile-selection", choices=("first", "random"), default="first")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--read-batch-size", type=int, default=8)
    parser.add_argument("--horizon-hours", type=float, default=3.0)
    parser.add_argument("--min-setpoint-change-c", type=float, default=0.05)
    parser.add_argument("--holdout-block-days", type=int, default=14)
    parser.add_argument("--gam-n-knots", type=int, default=7)
    parser.add_argument(
        "--gam-smoothing-grid",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0, 10.0, 100.0],
    )
    parser.add_argument("--solver-max-iterations", type=int, default=3000)
    args = parser.parse_args(argv)
    if args.holdout_block_days < 1:
        parser.error("--holdout-block-days must be positive")
    if args.gam_n_knots < 4:
        parser.error("--gam-n-knots must be at least 4")
    return args


def main() -> None:
    run_study(parse_args())


if __name__ == "__main__":
    main()
