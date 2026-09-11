"""Command-line entry point for reloaded full-profile scoring."""

from __future__ import annotations

import argparse
from pathlib import Path

from .scoring import score_artifact


def _horizons(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item <= 0.0 for item in result):
        raise argparse.ArgumentTypeError("horizons must be comma-separated positive hours")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a saved refactored emulator artifact.")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--profile-source", choices=("train", "test", "selected"), default="test")
    parser.add_argument(
        "--profile-id",
        type=int,
        action="append",
        dest="profile_ids",
        help="Score one explicit profile ID; repeat the option to select multiple profiles.",
    )
    parser.add_argument("--max-profiles", type=int)
    parser.add_argument("--kpi-mode", choices=("mean", "scenario_average"), default="mean")
    parser.add_argument("--num-scenarios", type=int, default=100)
    parser.add_argument("--hp-scenario-mode", choices=("bernoulli", "expected"), default="bernoulli")
    parser.add_argument(
        "--ventilation-rollout-mode",
        choices=("recorded", "eplus_rule"),
        default="recorded",
        help=(
            "Use the recorded EnergyPlus ventilation input or regenerate it sequentially "
            "from each scenario's predicted indoor temperature."
        ),
    )
    parser.add_argument("--horizons-hours", type=_horizons, default=(0.5, 1.0, 2.0, 3.0))
    parser.add_argument("--dt-hours", type=float, default=0.25)
    parser.add_argument("--min-setpoint-change", type=float, default=0.05)
    parser.add_argument("--min-events", type=int, default=20)
    parser.add_argument(
        "--kpi-estimator",
        choices=("direct_ratio", "regression"),
        default="direct_ratio",
    )
    parser.add_argument("--controls", choices=("none", "weather", "full"), default="full")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num-profile-plots", type=int, default=5)
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix inserted before every generated file extension, for example _f.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_artifact(
        args.artifact_dir,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        profile_source=args.profile_source,
        profile_ids=args.profile_ids,
        max_profiles=args.max_profiles,
        kpi_mode=args.kpi_mode,
        num_scenarios=args.num_scenarios,
        hp_scenario_mode=args.hp_scenario_mode,
        ventilation_rollout_mode=args.ventilation_rollout_mode,
        horizons_hours=args.horizons_hours,
        dt_hours=args.dt_hours,
        min_setpoint_change=args.min_setpoint_change,
        min_events=args.min_events,
        kpi_estimator=args.kpi_estimator,
        controls=args.controls,
        seed=args.seed,
        num_profile_plots=args.num_profile_plots,
        output_suffix=args.output_suffix,
    )


if __name__ == "__main__":
    main()
