"""Summarize deterministic thermal-variant Q-inversion diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


KINDS = {
    "bounded_mlp": "Constant alpha",
    "bounded_mlp_feedback": "Constant alpha + detached Tin feedback",
    "bounded_mlp_dual_alpha": "Seasonal dual alpha",
    "bounded_mlp_physical_alpha": "RC-rate alpha",
    "passive_memory_positive_q": "Passive memory + positive Q",
}
COLORS = {
    "Constant alpha": "#68757d",
    "Constant alpha + detached Tin feedback": "#8f5d9f",
    "Seasonal dual alpha": "#e76f51",
    "RC-rate alpha": "#2a9d8f",
    "Passive memory + positive Q": "#355ea8",
    "Passive memory + positive Q + observed-Q auxiliary": "#7b2cbf",
    "Constant alpha + observed-Q auxiliary": "#f4a261",
    "Seasonal dual alpha + observed-Q auxiliary": "#9c6644",
    "RC-rate alpha + observed-Q auxiliary": "#577590",
}
METRICS = (
    ("closed_tin_rmse_c", "Closed-loop Tin RMSE [C]"),
    ("force_off_response_rmse_c", "Force-off Tin response RMSE [C]"),
    ("observed_q_autonomous_rmse_c", "Observed-Q autonomous Tin RMSE [C]"),
    ("oracle_q_autonomous_rmse_c", "Oracle-Q autonomous Tin RMSE [C]"),
    ("oracle_q_teacher_forced_rmse_c", "Oracle-Q one-step Tin RMSE [C]"),
    ("oracle_q_below_reachable_fraction", "Target below reachable Tin fraction"),
)


def _rows(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("*/oracle_q_inversion_metrics.json")):
        data = json.loads(path.read_text())
        kind = data["thermal_kind"]
        if kind not in KINDS:
            continue
        observed_q_weight = float(
            data.get("observed_q_thermal_loss_weight", 0.0)
        )
        controller_training_feedback = data.get(
            "controller_training_feedback", "closed_loop"
        )
        model = KINDS[kind]
        if observed_q_weight > 0.0:
            model = f"{model} + observed-Q auxiliary"
        if controller_training_feedback != "closed_loop":
            model = f"{model} + trained with {controller_training_feedback}"
        closed = data["legacy_closed_loop"]
        oracle = data["oracle_q_capped"]
        teacher = data["teacher_forced_inverse"]
        training = data["training"]
        forced_temperature = data.get(
            "controller_teacher_forced_temperature", {}
        )
        forced_history = data.get(
            "controller_teacher_forced_all_history", {}
        )
        rows.append(
            {
                "profile_id": int(data["profile_id"]),
                "model": model,
                "thermal_kind": kind,
                "observed_q_thermal_loss_weight": observed_q_weight,
                "controller_training_feedback": controller_training_feedback,
                "controller_calendar_mode": data.get(
                    "controller_calendar_mode", "all"
                ),
                "closed_tin_rmse_c": closed["temperature_rmse_c"],
                "closed_qroom_rmse_w_m2": closed["qroom_rmse_w_m2"],
                "closed_pel_rmse_w_m2": closed["pel_rmse_w_m2"],
                "temperature_forced_qroom_rmse_w_m2": (
                    forced_temperature.get("qroom_rmse_w_m2")
                ),
                "temperature_forced_pel_rmse_w_m2": (
                    forced_temperature.get("pel_rmse_w_m2")
                ),
                "all_history_forced_qroom_rmse_w_m2": forced_history.get(
                    "qroom_rmse_w_m2"
                ),
                "all_history_forced_pel_rmse_w_m2": forced_history.get(
                    "pel_rmse_w_m2"
                ),
                "force_off_response_rmse_c": closed[
                    "force_off_temperature_response_rmse"
                ],
                "observed_q_autonomous_rmse_c": data["observed_q"][
                    "temperature_rmse_c"
                ],
                "observed_q_teacher_forced_rmse_c": data[
                    "observed_q_teacher_forced"
                ]["temperature_rmse_c"],
                "oracle_q_autonomous_rmse_c": oracle["temperature_rmse_c"],
                "oracle_q_teacher_forced_rmse_c": teacher[
                    "temperature_rmse_c"
                ],
                "oracle_q_below_reachable_fraction": oracle[
                    "target_below_reachable_min_fraction"
                ],
                "oracle_q_energy_ratio": oracle["qroom_energy_ratio"],
                "alpha_min": data.get("thermal_alpha_min"),
                "alpha_mean": data.get("thermal_alpha_mean"),
                "alpha_max": data.get("thermal_alpha_max"),
                "q_gain_mean_c_per_w_m2": data.get(
                    "thermal_q_gain_mean_c_per_w_m2"
                ),
                "q_gain_min_c_per_w_m2": data.get(
                    "thermal_q_gain_min_c_per_w_m2"
                ),
                "q_gain_max_c_per_w_m2": data.get(
                    "thermal_q_gain_max_c_per_w_m2"
                ),
                "passive_delta_mean_c": data.get(
                    "thermal_passive_delta_mean_c"
                ),
                "passive_delta_std_c": data.get(
                    "thermal_passive_delta_std_c"
                ),
                "alpha_free": training["thermal_alpha_free"],
                "alpha_heating": training["thermal_alpha_heating"],
                "rate_base_per_hour": training[
                    "thermal_rate_base_per_hour"
                ],
                "rate_vent_per_hour": training[
                    "thermal_rate_vent_per_hour"
                ],
                "rate_stack_per_hour": training[
                    "thermal_rate_stack_per_hour"
                ],
                "max_grad_norm": training["max_grad_norm"],
                "train_observed_q_thermal_loss": training.get(
                    "train_observed_q_thermal_loss"
                ),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError(f"No thermal-variant metrics found under {root}")
    return result


def run(root: Path) -> Path:
    data = _rows(root)
    data.to_csv(root / "alpha_variant_comparison_metrics.csv", index=False)
    figure = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=tuple(label for _, label in METRICS),
        vertical_spacing=0.11,
        horizontal_spacing=0.10,
    )
    profiles = sorted(data.profile_id.unique())
    for panel, (metric, _) in enumerate(METRICS):
        row = panel // 2 + 1
        col = panel % 2 + 1
        for model in data.model.drop_duplicates():
            selected = data[data.model == model].set_index("profile_id")
            figure.add_trace(
                go.Bar(
                    x=[str(profile) for profile in profiles],
                    y=[selected.loc[profile, metric] for profile in profiles],
                    name=model,
                    legendgroup=model,
                    showlegend=panel == 0,
                    marker_color=COLORS.get(model, "#6a4c93"),
                    customdata=[model for _ in profiles],
                    hovertemplate=(
                        "model=%{customdata}<br>profile=%{x}<br>"
                        "value=%{y:.5f}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
    figure.update_layout(
        title="Deterministic thermal variants with post-fit Q inversion",
        barmode="group",
        height=1150,
        template="plotly_white",
        legend={"orientation": "h", "y": 1.05},
    )
    output = root / "alpha_variant_comparison.html"
    figure.write_html(output, include_plotlyjs="cdn")
    print(data.round(5).to_string(index=False), flush=True)
    print(f"saved_alpha_variant_comparison={output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "output/neural_building_emulator_refactor/"
            "deterministic_alpha_variant_oracle_q_comparison"
        ),
    )
    args = parser.parse_args()
    run(args.root)


if __name__ == "__main__":
    main()
