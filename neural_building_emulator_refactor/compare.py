"""Score multiple saved artifacts and collect one comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .artifacts import load_artifact
from .scoring import score_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple saved emulator artifacts.")
    parser.add_argument("--artifact-dir", type=Path, action="append", required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-source", choices=("train", "test", "selected"), default="test")
    parser.add_argument("--max-profiles", type=int)
    parser.add_argument("--num-scenarios", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num-profile-plots", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for index, artifact_dir in enumerate(args.artifact_dir, start=1):
        artifact = load_artifact(artifact_dir)
        run_dir = args.output_dir / f"{index:02d}_{artifact.spec.name}"
        score_artifact(
            artifact_dir,
            dataset_path=args.dataset,
            output_dir=run_dir,
            profile_source=args.profile_source,
            max_profiles=args.max_profiles,
            kpi_mode="scenario_average" if artifact.spec.probabilistic else "mean",
            num_scenarios=args.num_scenarios,
            seed=args.seed,
            num_profile_plots=args.num_profile_plots,
        )
        summary = json.loads((run_dir / "summary.json").read_text())
        rows.append(
            {
                "artifact_dir": str(artifact_dir),
                "model_name": artifact.spec.name,
                "task": artifact.spec.task,
                "probabilistic": artifact.spec.probabilistic,
                **summary["metrics"],
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.output_dir / "model_comparison.csv", index=False)
    print(comparison.to_string(index=False))
    print(f"saved_comparison={args.output_dir / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
