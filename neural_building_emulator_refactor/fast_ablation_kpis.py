"""CLI for the lightweight autonomous-rollout ablation scorecard."""

from __future__ import annotations

import argparse
from pathlib import Path

from .fast_kpis import FastKpiConfig, evaluate_fast_kpis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute cheap trajectory, temporal, and raw flexibility KPIs."
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--profile-source", choices=("train", "test", "selected"), default="test")
    parser.add_argument("--max-profiles", type=int, default=100)
    parser.add_argument("--num-eval-particles", type=int, default=4)
    parser.add_argument("--dt-hours", type=float, default=0.25)
    parser.add_argument("--event-horizon-hours", type=float, default=3.0)
    parser.add_argument("--min-setpoint-change-c", type=float, default=0.05)
    parser.add_argument("--acf-lags", type=int, nargs="+", default=(1, 2, 4, 8, 12, 24, 48, 96))
    parser.add_argument("--relative-power-floor-w-m2", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_profiles <= 0:
        raise ValueError("--max-profiles must be positive")
    if args.num_eval_particles <= 0:
        raise ValueError("--num-eval-particles must be positive")
    evaluate_fast_kpis(
        args.artifact_dir,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        profile_source=args.profile_source,
        max_profiles=args.max_profiles,
        num_eval_particles=args.num_eval_particles,
        config=FastKpiConfig(
            dt_hours=args.dt_hours,
            event_horizon_hours=args.event_horizon_hours,
            min_setpoint_change_c=args.min_setpoint_change_c,
            acf_lags=tuple(args.acf_lags),
            relative_power_floor_w_m2=args.relative_power_floor_w_m2,
        ),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
