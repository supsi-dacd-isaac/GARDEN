"""Audit whether event controls leave independent up/down setpoint variation.

No simulator or emulator is fitted. Full-design rank alone is insufficient:
constant nuisance columns can be harmless, whereas confounded directional
columns invalidate a unique interpretation of the reported coefficients.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SOURCE = Path("output/neural_building_emulator_refactor/current_best_cap0p5_lstm_markov_1000_e15")


def directional_identifiability(design, relative_tolerance=1e-8):
    """Check up/down identification after nuisance controls using only NumPy."""
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be finite and positive")
    design = np.asarray(design, dtype=float)
    if design.ndim != 2 or design.shape[1] < 3 or len(design) < 3:
        raise ValueError("Expected event rows and intercept/up/down design columns")
    if not np.isfinite(design).all():
        raise ValueError("Nonfinite regression design")
    directions = design[:, 1:3]
    nuisance = design[:, [0, *range(3, design.shape[1])]]
    residual = directions - nuisance @ np.linalg.lstsq(nuisance, directions, rcond=None)[0]
    singular = np.linalg.svd(residual, compute_uv=False)
    reference = max(float(np.linalg.norm(directions, ord=2)), 1e-12)
    rank = int(np.sum(singular > relative_tolerance * reference))
    fractions = np.linalg.norm(residual, axis=0) / np.maximum(np.linalg.norm(directions, axis=0), 1e-12)
    return dict(design_rank=int(np.linalg.matrix_rank(design)), columns=design.shape[1],
                directional_rank_after_controls=rank,
                directional_singular_max=float(singular[0]),
                directional_singular_min=float(singular[-1]),
                up_residual_fraction=float(fractions[0]), down_residual_fraction=float(fractions[1]),
                relative_tolerance=relative_tolerance)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--target", type=Path, default=Path("data/control=setpoint"))
    parser.add_argument("--output", type=Path, default=SOURCE / "tariff_policy_transfer")
    args = parser.parse_args()
    # The reusable rank check needs neither a model checkpoint nor the trainer.
    import pandas as pd

    from neural_building_emulator.flexibility_event_study_kpis import (
        EventStudyConfig, regression_data_for_trace,
    )
    from .artifacts import load_artifact
    from .profiles import load_profiles

    recipe = read_json(args.source / "comparison_request.json")["recipe"]
    split = read_json(args.source / "split_manifest.json")
    artifact_info = read_json(args.source / "variants/structured/artifact.json")
    artifact = load_artifact(Path(artifact_info["artifact_dir"]))
    evaluation = recipe["evaluation"]
    horizons = tuple(evaluation["horizons_hours"])
    steps = tuple(int(round(h / .25)) for h in horizons)
    rows = []
    for dataset_name, dataset in (("source", Path(recipe["dataset_path"])), ("tariff", args.target)):
        for fold in ("calibration", "test"):
            ids = split[fold + "_ids"]
            for start in range(0, len(ids), 16):
                profiles = load_profiles(artifact, dataset_path=dataset, profile_ids=ids[start:start + 16])
                for profile in profiles:
                    for controls in ("none", "full"):
                        config = EventStudyConfig(horizons, steps, .25, evaluation["min_setpoint_change"],
                                                  evaluation["min_events"], controls, "regression")
                        for horizon, step in zip(horizons, steps):
                            data = regression_data_for_trace(profile, profile.targets,
                                                             horizon_steps=step, config=config)
                            rows.append(dict(dataset=dataset_name, fold=fold, profile_id=profile.profile_id,
                                             controls=controls, horizon_hours=horizon,
                                             events=len(data.events), delta_min=float(data.deltas.min()),
                                             delta_max=float(data.deltas.max()),
                                             unique_deltas=len(np.unique(data.deltas)),
                                             **directional_identifiability(data.design)))
                print(f"identifiability {dataset_name}:{fold} {min(start + 16, len(ids))}/{len(ids)}", flush=True)
    frame = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "kpi_identifiability.csv", index=False)
    summary = frame.groupby(["dataset", "fold", "controls"]).agg(
        buildings=("profile_id", "nunique"), cells=("profile_id", "size"),
        min_directional_rank=("directional_rank_after_controls", "min"),
        max_directional_rank=("directional_rank_after_controls", "max"),
        unique_deltas_min=("unique_deltas", "min"), unique_deltas_max=("unique_deltas", "max"),
        min_events=("events", "min"), median_events=("events", "median"),
    ).reset_index()
    summary.to_csv(args.output / "kpi_identifiability_summary.csv", index=False)
    write_json(args.output / "kpi_identifiability_request.json", dict(
        source=str(args.source.resolve()), target=str(args.target.resolve()),
        implementation_sha256=digest(Path(__file__)),
        model_used="Only for consistent input schema/loading; observed EnergyPlus traces only",
        interpretation="Rank 2 identifies both directional coefficients after nuisance controls; "
                       "rank 1 identifies only one combination; rank 0 identifies neither. "
                       "This diagnostic does not change the requested benchmark estimator.",
    ))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
