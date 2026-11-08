"""Paired structured baseline/two-tank experiments on Ticino v9."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .calibrated_model_benchmark import BenchmarkSplit, _emulator_distribution, _force_off_emulator_distribution
from .campaign_benchmark import POLICIES
from .fair_probabilistic_lstm_comparison import _experiment_configs
from .generic_response_report import native_scores
from .plant_accounting_report import paired_crps_bootstrap
from .routed_latent_targeted_loss_ablation import _train_variant


CAMPAIGN = Path("data/garden-ticino-five-cities-1000-v9-final")
BASE = Path("output/neural_building_emulator_refactor/ticino_v5_cap0p5_lstm_markov_999_e15")
OUTPUT = Path("output/neural_building_emulator_refactor/ticino_v9_two_tanks_200_e15")
VARIANTS = ("baseline", "two_tanks")
FOLDS = ("train_ids", "validation_ids", "calibration_ids", "test_ids")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def freeze(path, value):
    value = json.loads(json.dumps(value, default=str))
    if path.exists() and read(path) != value:
        raise ValueError(f"Frozen request changed: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def subset(source, seed=13, max_profiles=200):
    all_ids = [int(x) for key in FOLDS for x in source[key]]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Source folds overlap")
    if max_profiles == 1000:
        if len(all_ids) != 1000 or set(all_ids) != set(source["selected_ids"]):
            raise ValueError("Full campaign must contain exactly the original 1000 buildings")
        return {key: tuple(sorted(int(x) for x in source[key])) for key in FOLDS}, sorted(all_ids)
    if max_profiles != 200:
        raise ValueError("Supported cohort sizes are 200 and 1000")
    rng = np.random.default_rng(seed)
    folds = {key: tuple(sorted(int(x) for x in rng.choice(source[key], count, replace=False)))
             for key, count in zip(FOLDS, (140, 20, 20, 20))}
    selected = sorted(x for values in folds.values() for x in values)
    if len(set(selected)) != 200:
        raise ValueError("Source folds overlap")
    return folds, selected


def variant_recipe(base, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    recipe = copy.deepcopy(base)
    recipe["probabilistic_structured"].update(
        hp_dhw_metadata=True, prob_tank_temperature_state=variant == "two_tanks",
        prob_tank_temperature_loss_weight=1.0 if variant == "two_tanks" else 0.0,
    )
    return recipe


def prepare(args):
    import pyarrow.parquet as pq

    source = read(args.campaign_root / "split_manifest.json")
    folds, selected = subset(source, max_profiles=args.max_profiles)
    root = args.campaign_root.resolve()
    base = read(args.base_root / "comparison_request.json")["recipe"]
    artifact = Path(read(args.base_root / "variants/structured/artifact.json")["artifact_dir"])
    config = read(artifact / "metadata_closed_loop_hp_contracting_probabilistic.json")["train_config"]
    base["probabilistic_structured"] = config
    paths = [root / "data" / ("control=" + policy) for policy in POLICIES]
    base.update(dataset_path=str(paths[0]), additional_dataset_paths=[str(paths[1])], max_profiles=len(selected))
    protocol = None
    if args.validation_epoch_gate:
        from .tank_temperature_monitor import epoch_protocol
        protocol = epoch_protocol(folds, initial_epochs=args.epochs)
    base["optimizer"]["epochs"] = protocol["maximum_epochs"] if protocol else args.epochs
    base["evaluation"].update(num_scenarios=args.num_scenarios,
                              num_profile_plots=5 if len(selected) == 1000 else 3)
    base["force_off_evaluation"] = dict(dataset_path=str(paths[1]), raw_only=True)
    base["campaign"] = dict(root=str(root), train_policies=list(POLICIES[:2]),
        transfer_datasets=[str(p) for p in paths[2:]], source_fold_preservation=True,
        calibration_scope="Reserved calibration fold, unused in this native comparison")
    split_data = dict(dataset_path=str(paths[0]), additional_dataset_paths=[str(paths[1])], seed=13,
        selected_ids=selected, requested_max_profiles=len(selected), eligible_count=1000, **folds)
    split_data["fingerprint"] = hashlib.sha256(json.dumps(split_data, sort_keys=True).encode()).hexdigest()[:16]
    split = BenchmarkSplit.from_json(split_data)
    checks, references = {}, {}
    for policy in paths:
        for building in selected:
            folder = policy / f"egid={building}"
            marker = read(folder / "complete.json")
            meta = read(folder / "simulation_metadata.json")
            if digest(folder / "simulation_metadata.json") != marker["simulationMetadataSha256"]:
                raise ValueError(f"Metadata hash mismatch: {folder}")
            values = meta["gardenCompatibility"]["values"]
            for key in ("shVolume_m3", "dhwVolume_m3", "DHW_recovery_W"):
                if not np.isfinite(values[key]) or values[key] <= 0:
                    raise ValueError(f"Invalid {key} in {folder}")
            for key, resolved_key in (("shVolume_m3", "sh_buffer_volume_m3"), ("dhwVolume_m3", "dhw_tank_volume_m3")):
                np.testing.assert_allclose(values[key], meta["resolved_plant_design"][resolved_key], rtol=1e-6)
            if building in references and references[building] != values:
                raise ValueError(f"Policy-specific metadata for {building}")
            references[building] = values
            table = pq.read_table(folder / "timeseries.parquet", columns=["datetime", "zone", "sh_tank_measured_temperature_c", "dhw_tank_mean_temperature_c"]).to_pandas()
            plant = table[table.zone.eq("technical_room")].sort_values("datetime")
            if len(plant) != 35040 or not np.isfinite(plant.iloc[:, 2:].to_numpy()).all():
                raise ValueError(f"Incomplete tank labels: {folder}")
            np.testing.assert_array_equal(plant.datetime, pd.date_range("2023-01-01", periods=35040, freq="15min"))
            checks[str(folder / "complete.json")] = digest(folder / "complete.json")
            if len(checks) % 200 == 0:
                print(f"Validated annual partitions: {len(checks)}/{4*len(selected)}", flush=True)
    freeze(args.output_dir / "split_manifest.json", asdict(split))
    freeze(args.output_dir / "comparison_request.json", dict(recipe=base, variants=list(VARIANTS),
        variant_recipes={v: variant_recipe(base, v) for v in VARIANTS},
        source_artifact=str(artifact.resolve()), source_artifact_hashes={str(p): digest(p) for p in artifact.iterdir() if p.is_file()},
        dataset_manifest_sha256=digest(root / "split_manifest.json"), validated_partitions=len(checks),
        epoch_protocol=protocol,
        scope="Two architecture arms, both with the same two DHW metadata inputs. Native ensembles only. "
              "Candidate replaces hidden buffer energy with two supervised effective temperatures; "
              "no future observed plant states at inference. Single-seed screen, not a multi-seed conclusion."))
    freeze(args.output_dir / "data_hashes.json", checks)
    sources = sorted({*Path("neural_building_emulator").rglob("*.py"), *Path("neural_building_emulator_refactor").rglob("*.py")})
    freeze(args.output_dir / "implementation_hashes.json", {str(p): digest(p) for p in sources})
    print(f"Prepared {len(selected)} buildings: { {k:len(v) for k,v in folds.items()} }; {len(checks)} validated annual partitions", flush=True)


def context(args):
    for path, expected in read(args.output_dir / "implementation_hashes.json").items():
        if digest(path) != expected:
            raise ValueError(f"Scientific source changed: {path}; use a fresh experiment directory")
    for path, expected in read(args.output_dir / "data_hashes.json").items():
        if digest(path) != expected:
            raise ValueError(f"Dataset completion record changed: {path}")
    request = read(args.output_dir / "comparison_request.json")
    return request["variant_recipes"][args.variant], BenchmarkSplit.from_json(read(args.output_dir / "split_manifest.json"))


def checkpoint(args):
    folder = args.output_dir / "variants" / args.variant
    artifact = Path(read(folder / "artifact.json")["artifact_dir"])
    freeze(folder / "checkpoint_hashes.json", {str(p): digest(p) for p in artifact.iterdir() if p.is_file()})
    return artifact


def train_variant(args, recipe, split):
    args.num_profile_plots = 3
    args.resume = True
    args.epochs = recipe["optimizer"]["epochs"]
    protocol = read(args.output_dir / "comparison_request.json").get("epoch_protocol")
    if protocol:
        from .tank_temperature_monitor import train_monitored
        train_monitored(args, recipe, split, protocol)
    else:
        _train_variant(args, recipe, split, args.variant)
    checkpoint(args)


def smoke(args, recipe, split):
    from .trainer import train
    from .calibrated_model_benchmark import _eligible_hp_ids
    from .artifacts import load_artifact
    from .profiles import load_profiles
    from .prediction import predict_profile
    import jax

    recipe = copy.deepcopy(recipe)
    recipe.update(sequence_length=96, stride=35039)
    recipe["optimizer"].update(batch_size=2, train_eval_max_windows=2)
    recipe["probabilistic_structured"].update(state_dim=4, hidden_dim=8, depth=1,
        prob_particles=2, prob_eval_particles=2, prob_plot_particles=2,
        prob_flex_kpi_crps_weight=0., prob_flex_kpi_crps_windows_per_profile=1)
    retained = {*split.train_ids[:2], *split.validation_ids[:1]}
    destination = args.output_dir / "smoke" / args.variant
    configs = _experiment_configs(recipe, destination / "models", max_profiles=3, epochs=1,
        fixed_test_profile_ids=split.validation_ids[:1],
        excluded_profile_ids=tuple(sorted(set(_eligible_hp_ids(Path(recipe["dataset_path"]))) - retained)))
    artifact = load_artifact(Path(train(configs["probabilistic_structured"])))
    profile, = load_profiles(artifact, dataset_path=Path(recipe["dataset_path"]), profile_ids=split.validation_ids[:1])
    result = predict_profile(artifact, profile, key=jax.random.PRNGKey(101), num_scenarios=2,
                            hp_scenario_mode="bernoulli", ventilation_rollout_mode="eplus_rule")
    if result.scenarios.shape != (2, 35039, 3) or not np.isfinite(result.scenarios).all():
        raise ValueError("Annual autonomous inference failed after smoke reload")
    freeze(destination / "smoke_complete.json", dict(artifact=str(artifact.artifact_dir), finite_annual_shape=list(result.scenarios.shape)))
    print("Smoke training, checkpoint reload and annual inference passed", flush=True)


def score(args, recipe, split):
    artifact = checkpoint(args)
    output = args.output_dir / "variants" / args.variant / "evaluation" / args.policy
    score_checkpoint(args, recipe, artifact, output, split.test_ids,
                     num_scenarios=recipe["evaluation"]["num_scenarios"],
                     num_profile_plots=recipe["evaluation"]["num_profile_plots"])


def score_checkpoint(args, recipe, artifact, output, profile_ids, *, num_scenarios, num_profile_plots):
    from .force_off_policy_evaluation import evaluate_force_off_policy
    from .scoring import score_artifact

    request = dict(artifact=str(artifact), ids=list(profile_ids), policy=args.policy,
                   particles=num_scenarios, num_profile_plots=num_profile_plots,
                   artifact_hashes={str(p): digest(p) for p in Path(artifact).iterdir() if p.is_file()})
    if (output / "complete.json").exists():
        if read(output / "complete.json") != request:
            raise ValueError(f"Evaluation request changed: {output}")
        return
    evaluation = recipe["evaluation"]
    off = "force_off" in args.policy
    common = dict(dataset_path=args.campaign_root.resolve() / "data" / ("control=" + args.policy),
        output_dir=output, profile_ids=profile_ids, num_scenarios=num_scenarios,
        hp_scenario_mode="bernoulli", ventilation_rollout_mode="eplus_rule", num_profile_plots=num_profile_plots,
        seed=recipe["seed"] + (2000 if off else 1000))
    if off:
        evaluate_force_off_policy(artifact, **common, evaluate_ungated=False,
            num_eval_particles=recipe["probabilistic_structured"]["prob_eval_particles"],
            kpi_horizons_hours=evaluation["horizons_hours"], kpi_min_events=evaluation["min_events"])
    else:
        score_artifact(artifact, **common, kpi_mode="scenario_average", kpi_estimator="regression",
            horizons_hours=evaluation["horizons_hours"], min_setpoint_change=evaluation["min_setpoint_change"],
            min_events=evaluation["min_events"], controls=evaluation["controls"], evaluate_direct_f=True)
    freeze(output / "complete.json", request)


def validate_report_artifacts(artifacts):
    """Validate both explicit and fixed-three-output legacy checkpoint schemas."""
    from neural_building_emulator.columns import CLOSED_LOOP_TARGET_COLUMNS

    before, after = [a.legacy_artifact.metadata for a in artifacts]
    for key in ("train_ids", "test_ids", "metadata_columns", "input_columns",
                "model_kind", "target_alignment", "target_mode"):
        if before[key] != after[key]:
            raise ValueError(f"Unmatched trained schema/split: {key}")
    for artifact, metadata in zip(artifacts, (before, after)):
        columns = metadata.get("target_columns")
        if columns is None:
            if metadata["model_kind"] != "closed_loop_hp_contracting_probabilistic":
                raise ValueError("Cannot infer targets for an unsupported legacy model")
            columns = CLOSED_LOOP_TARGET_COLUMNS
        if tuple(columns) != tuple(CLOSED_LOOP_TARGET_COLUMNS):
            raise ValueError("Unexpected target columns or channel order")
        if any(np.asarray(getattr(artifact.scalers.target, field)).size != len(columns)
               for field in ("mean", "scale")):
            raise ValueError("Target scaler dimensions disagree with the output schema")
        if tuple(getattr(artifact.model, key) for key in
                 ("temperature_target_index", "qroom_target_index", "pel_target_index")) != (0, 1, 2):
            raise ValueError("Model output indices disagree with the output schema")
    for group in ("metadata", "inputs", "target"):
        for field in ("mean", "scale"):
            np.testing.assert_array_equal(getattr(getattr(artifacts[0].scalers, group), field),
                                          getattr(getattr(artifacts[1].scalers, group), field))


def report(args):
    from .artifacts import load_artifact

    artifacts = [load_artifact(Path(read(args.output_dir / "variants" / v / "artifact.json")["artifact_dir"])) for v in VARIANTS]
    validate_report_artifacts(artifacts)
    frames = []
    for variant in VARIANTS:
        for policy in POLICIES:
            folder = args.output_dir / "variants" / variant / "evaluation" / policy
            if not (folder / "complete.json").exists():
                raise ValueError(f"Incomplete evaluation: {folder}")
            for kind in ("regression", "f"):
                if "force_off" in policy:
                    frame = _force_off_emulator_distribution(folder, method=variant,
                        source_model="structured_gated", estimator="regression" if kind == "regression" else "direct_ratio",
                        kpi_kind=kind)
                else:
                    frame = _emulator_distribution(folder, method=variant, kpi_kind=kind)
                frame["kpi_kind"] = policy + "_" + kind
                frames.append(frame)
    distribution = pd.concat(frames, ignore_index=True)
    summary, metrics, reliability, buildings, pit = native_scores(distribution)
    expected = read(args.output_dir / "comparison_request.json")["recipe"]["evaluation"]["num_scenarios"]
    if not pit.particles.eq(expected).all():
        raise ValueError("Incomplete native ensembles in some KPI cells")
    paired = paired_crps_bootstrap(buildings)
    # Report incomplete cells explicitly rather than hiding selective ensemble failures.
    if not (paired.matched_cells.eq(paired.baseline_cells) & paired.matched_cells.eq(paired.candidate_cells)).all():
        raise ValueError("Unmatched KPI cells: comparison would not be fair")
    output = args.output_dir / "reports"
    output.mkdir(exist_ok=True)
    for name, frame in [("summary", summary), ("metrics", metrics), ("reliability", reliability),
                        ("building_metrics", buildings), ("pit", pit), ("paired_crps", paired)]:
        frame.to_csv(output / (name + ".csv"), index=False)
    links = "".join(f'<li><a href="../{html.escape(str(p.relative_to(args.output_dir)))}">{html.escape(str(p.relative_to(args.output_dir)))}</a></li>'
                    for p in sorted((args.output_dir / "variants").rglob("*.html")))
    page = '<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Two-tank comparison</title><style>body{font:15px system-ui;margin:24px;color:#222}table{display:block;overflow:auto;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd}h1{font-size:24px}</style>'
    cohort_size = len(read(args.output_dir / "split_manifest.json")["selected_ids"])
    page += f'<h1>Structured baseline versus two tank temperatures</h1><p>Same {cohort_size}-building cohort, split, original-target losses and training-budget rule. Native annual ensembles; no ex-post recalibration. Both models receive DHW volume/recovery metadata. Bootstrap intervals resample buildings; single training seed.</p>'
    page += '<h2>Native KPI scores</h2>' + summary.round(4).to_html(index=False)
    page += '<h2>Paired CRPS differences</h2><p>Negative differences favour the two-tank model. Cell counts expose invalid or missing KPI definitions.</p>' + paired.round(4).to_html(index=False)
    trace_rows = []
    for variant in VARIANTS:
        for policy in POLICIES:
            folder = args.output_dir / "variants" / variant / "evaluation" / policy
            scores = pd.read_csv(folder / "temporal_trace_probabilistic_scores.csv")
            calibration = pd.read_csv(folder / "temporal_trace_calibration.csv")
            if "model" in scores:
                scores = scores[scores.model.eq("structured_gated")]
                calibration = calibration[calibration.model.eq("structured_gated")]
            for row in scores.to_dict("records"):
                curve = calibration[calibration.signal.eq(row["signal"])]
                low = curve.loc[np.isclose(curve["quantile"], .05), "empirical_coverage"].item()
                high = curve.loc[np.isclose(curve["quantile"], .95), "empirical_coverage"].item()
                trace_rows.append(dict(method=variant, policy=policy, signal=row["signal"],
                    scope=row["scope"], crps=row["mean_crps"], coverage_90=high-low,
                    width_90=row["mean_central_90_width"], width_95=row["mean_central_95_width"],
                    calibration_mae=curve.abs_calibration_error.mean()))
    traces = pd.DataFrame(trace_rows)
    traces.to_csv(output / "trace_comparison.csv", index=False)
    page += '<h2>Shared thermal and power trace distributions</h2><p>Native time-step CRPS, rank-based coverage and interval width; Qroom and Pel are scored while space heating is available.</p>' + traces.round(4).to_html(index=False)
    decisions = []
    for variant in VARIANTS:
        path = args.output_dir / "variants" / variant / "epoch_decision.json"
        if path.exists():
            decisions.append(dict(method=variant, **read(path)))
    if decisions:
        page += '<h2>Validation-only epoch decisions</h2>' + pd.DataFrame(decisions).to_html(index=False)
    tank_summaries = []
    for policy in POLICIES:
        path = args.output_dir / "variants/two_tanks/tank_diagnostics" / policy / "summary.csv"
        if path.exists():
            table = pd.read_csv(path)
            table.insert(0, "policy", policy)
            tank_summaries.append(table)
    if tank_summaries:
        tanks = pd.concat(tank_summaries, ignore_index=True)
        tanks.to_csv(output / "tank_trace_summary.csv", index=False)
        page += '<h2>Candidate trajectory diagnostics</h2>' + tanks.round(4).to_html(index=False)
    page += '<h2>Annual plots</h2><ul>' + links + '</ul></html>'
    (output / "comparison.html").write_text(page)
    print(summary.to_string(index=False), flush=True)


def supervise(args):
    from .tank_temperature_monitor import wait_for_worker, write_status
    jobs = [("train", v, POLICIES[0]) for v in VARIANTS]
    jobs += [("score", v, p) for v in VARIANTS for p in POLICIES]
    jobs += [("tank_diagnostics", "two_tanks", p) for p in POLICIES]
    jobs += [("report", "baseline", POLICIES[0])]
    logs = args.output_dir / "logs"
    logs.mkdir(exist_ok=True)
    for index, (stage, variant, policy) in enumerate(jobs):
        command = [sys.executable, "-u", "-m", __spec__.name, "--output-dir", str(args.output_dir),
                   "--campaign-root", str(args.campaign_root), "--stage", stage,
                   "--variant", variant, "--policy", policy]
        state = dict(state="running", index=index, jobs=len(jobs), stage=stage, variant=variant, policy=policy,
                     started_unix=time.time(), command=command)
        with (logs / f"{index:02d}_{stage}_{variant}_{policy}.log").open("a") as log:
            worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            state["worker_pid"] = worker.pid
            code = wait_for_worker(worker, args.output_dir, state, Path(log.name))
        if code:
            state.update(state="failed", returncode=code)
            write_status(args.output_dir / "status.json", state)
            raise SystemExit(code)
    state.update(state="complete", returncode=0, completed_unix=time.time())
    state.pop("worker_pid", None)
    write_status(args.output_dir / "status.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=CAMPAIGN)
    parser.add_argument("--base-root", type=Path, default=BASE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--stage", choices=("prepare", "smoke", "train", "score", "validation_score", "tank_diagnostics", "report", "supervise"), required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="baseline")
    parser.add_argument("--policy", choices=POLICIES, default=POLICIES[0])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--max-profiles", type=int, choices=(200, 1000), default=200)
    parser.add_argument("--validation-epoch-gate", action="store_true")
    parser.add_argument("--audit-epoch", type=int)
    parser.add_argument("--audit-artifact", type=Path)
    parser.add_argument("--num-scenarios", type=int, default=100)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args)
        return
    recipe, split = context(args)
    if args.stage == "supervise":
        supervise(args)
    elif args.stage == "train":
        train_variant(args, recipe, split)
    elif args.stage == "smoke":
        smoke(args, recipe, split)
    elif args.stage == "score":
        score(args, recipe, split)
    elif args.stage == "validation_score":
        from .tank_temperature_monitor import validation_score
        validation_score(args, recipe, split)
    elif args.stage == "tank_diagnostics":
        from .tank_temperature_diagnostics import run
        run(args, recipe, split, checkpoint(args))
    else:
        report(args)


if __name__ == "__main__":
    main()
