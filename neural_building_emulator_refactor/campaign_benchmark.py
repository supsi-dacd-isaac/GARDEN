"""Matched training and frozen policy transfer for paired four-policy campaigns."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

POLICIES = ("setpoint_excitation_v2", "force_off_excitation_v2",
            "tariff_setpoint_v2", "tariff_force_off_v2")
VARIANTS = ("structured", "continuous", "hurdle_markov")
METHODS = (*VARIANTS, "lightgbm")
FOLDS = ("train_ids", "validation_ids", "calibration_ids", "test_ids")
KINDS = ("regression_based", "f", "force_off_regression", "force_off_f")


def read_json(path):
    return json.loads(Path(path).read_text())


def freeze(path, value):
    payload = json.loads(json.dumps(value))
    if path.exists() and read_json(path) != payload:
        raise ValueError(f"Campaign provenance changed: {path}; use a new output directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def resolve_campaign(base, campaign, output, *, excluded_ids=()):
    """Preserve supplied folds; missing outputs require an explicit exclusion list."""
    from .calibrated_model_benchmark import BenchmarkSplit, _eligible_hp_ids

    campaign, output = Path(campaign).resolve(), Path(output)
    source = read_json(campaign / "split_manifest.json")
    selected = set(source["selected_ids"])
    if len(selected) != len(source["selected_ids"]):
        raise ValueError("Duplicate selected building IDs")
    seen = set()
    for name in FOLDS:
        ids = set(source[name])
        if len(ids) != len(source[name]) or seen & ids:
            raise ValueError(f"Duplicate/overlapping building folds: {name}")
        seen.update(ids)
    if seen != selected:
        raise ValueError("Supplied folds do not partition selected building IDs")
    if base["max_profiles"] != source["requested_max_profiles"]:
        raise ValueError("Requested budget must match the supplied campaign manifest")
    paths = [campaign / "data" / ("control=" + policy) for policy in POLICIES]
    complete_sets, marker_hashes = [], {}
    for path in paths:
        complete = set()
        for building in sorted(selected):
            folder = path / f"egid={building}"
            marker = folder / "complete.json"
            if all((folder / name).is_file() for name in
                   ("complete.json", "simulation_metadata.json", "timeseries.parquet")):
                complete.add(building)
                marker_hashes[str(marker.relative_to(campaign))] = digest(marker)
        complete_sets.append(complete)
    usable = set.intersection(*complete_sets)
    missing = selected - usable
    if missing != set(excluded_ids):
        raise ValueError(f"Missing paired buildings {sorted(missing)} differ from explicit exclusions")
    if any(values != usable for values in complete_sets):
        raise ValueError("Asymmetric completion across policies; inspect before excluding buildings")
    for path in paths:
        if set(_eligible_hp_ids(path)) != usable:
            raise ValueError(f"Eligibility differs from completed cohort: {path}")
    payload = dict(dataset_path=str(paths[0]), additional_dataset_paths=[str(paths[1])],
                   seed=source["seed"], selected_ids=sorted(usable),
                   **{name: sorted(set(source[name]) & usable) for name in FOLDS})
    if any(not payload[name] for name in FOLDS):
        raise ValueError("A campaign fold is empty")
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    split = BenchmarkSplit.from_json(dict(payload, fingerprint=fingerprint,
                                         requested_max_profiles=source["requested_max_profiles"],
                                         eligible_count=len(usable)))
    recipe = json.loads(json.dumps(base))
    recipe.update(dataset_path=str(paths[0]), additional_dataset_paths=[str(paths[1])])
    recipe["force_off_evaluation"] = dict(dataset_path=str(paths[1]), raw_only=False)
    recipe["poor_excitation_evaluation"] = {"enabled": False}
    # The unused deterministic recipe may contain resolved paths and IDs from an
    # older experiment. Retain it only because the shared config factory needs it.
    for key in ("dataset_path", "additional_dataset_paths", "excluded_profile_ids",
                "fixed_test_profile_ids", "output_dir", "model_checkpoint_dir"):
        recipe.get("deterministic_structured", {}).pop(key, None)
    recipe["campaign"] = dict(root=str(campaign), transfer_datasets=[str(p) for p in paths[2:]],
                              excluded_missing_ids=sorted(missing), split_seed=source["seed"],
                              training_seed=recipe["seed"], train_policies=list(POLICIES[:2]),
                              calibration_scope="Excitation calibration buildings only; no tariff fitting",
                              trace_bands="Native ensemble, never RNS-transformed")
    freeze(output / "split_manifest.json", asdict(split))
    freeze(output / "campaign_provenance.json", dict(
        campaign=recipe["campaign"], source_manifest_sha256=digest(campaign / "split_manifest.json"),
        campaign_manifest_sha256=digest(campaign / "campaign_manifest.json"),
        completion_marker_hashes=marker_hashes, split=asdict(split),
        excluded_reason="Missing complete paired simulator outputs; no replacement or reshuffling",
        dhw_mode_convention="Keep historical >0.5 mask; fractional mode is not an energy fraction",
    ))
    return recipe, split


def context(args):
    from .calibrated_model_benchmark import BenchmarkSplit
    from .lstm_hurdle_ablation import _fingerprints
    from .hp_flex_loss_ablation import _load_artifact_record

    request = read_json(args.output_dir / "comparison_request.json")
    split = BenchmarkSplit.from_json(read_json(args.output_dir / "split_manifest.json"))
    for path, expected in read_json(args.output_dir / "implementation_hashes.json").items():
        if digest(path) != expected:
            raise ValueError(f"Scientific implementation changed: {path}")
    artifacts = {}
    for variant in request.get("variants", VARIANTS):
        artifact = _load_artifact_record(args.output_dir, variant)
        if _fingerprints(artifact) != read_json(args.output_dir / "variants" / variant / "artifact.json")["artifact_hashes"]:
            raise ValueError(f"Checkpoint changed: {variant}")
        artifacts[variant] = artifact
    return request["recipe"], split, artifacts


def policy_root(root, policy):
    return root if policy in POLICIES[:2] else root / "transfer" / policy


def score_request(args, recipe, split, artifacts):
    policy, fold, variant = args.policy, args.fold, args.variant
    off = "force_off" in policy
    dataset = args.campaign_root.resolve() / "data" / ("control=" + policy)
    root = policy_root(args.output_dir, policy)
    destination = root / "variants" / variant / ("force_off" if off else "scoring") / fold
    ids = getattr(split, fold + "_ids")
    evaluation = recipe["evaluation"]
    seed = recipe["seed"] + (2000 if off else 1000) + (0 if fold == "test" else -1000)
    request = dict(dataset=str(dataset), profile_ids=list(ids), evaluation=evaluation, seed=seed,
                   artifact=str(artifacts[variant]), fold=fold, policy=policy,
                   checkpoint=read_json(args.output_dir / "variants" / variant / "artifact.json")["artifact_hashes"])
    return destination, request


def score_policy(args, recipe, split, artifacts):
    from .scoring import score_artifact
    from .force_off_policy_evaluation import evaluate_force_off_policy

    destination, request = score_request(args, recipe, split, artifacts)
    fold, variant = args.fold, args.variant
    off = "force_off" in args.policy
    evaluation = recipe["evaluation"]
    marker = destination / "campaign_complete.json"
    if marker.exists():
        freeze(marker, request)
        return
    destination.mkdir(parents=True, exist_ok=True)
    common = dict(dataset_path=Path(request["dataset"]), output_dir=destination, profile_ids=request["profile_ids"],
                  num_scenarios=evaluation["num_scenarios"], hp_scenario_mode="bernoulli",
                  ventilation_rollout_mode=evaluation["ventilation_rollout_mode"],
                  seed=request["seed"], num_profile_plots=evaluation["num_profile_plots"] if fold == "test" else 0)
    if off:
        evaluate_force_off_policy(artifacts[variant], **common, evaluate_ungated=False,
                                  num_eval_particles=recipe["probabilistic_structured"]["prob_eval_particles"],
                                  kpi_horizons_hours=evaluation["horizons_hours"],
                                  kpi_min_events=evaluation["min_events"])
    else:
        score_artifact(artifacts[variant], **common, kpi_mode="scenario_average",
                       horizons_hours=evaluation["horizons_hours"], kpi_estimator="regression",
                       min_setpoint_change=evaluation["min_setpoint_change"],
                       min_events=evaluation["min_events"], controls=evaluation["controls"],
                       evaluate_direct_f=True)
    freeze(marker, request)


def lightgbm_transfer(args, recipe, split, artifact):
    import lightgbm as lgb
    import pandas as pd
    from .calibrated_model_benchmark import (
        _metadata_target_table, _force_off_metadata_target_table, _normal_distribution_rows,
    )

    policy = args.policy
    root = policy_root(args.output_dir, policy)
    dataset = args.campaign_root.resolve() / "data" / ("control=" + policy)
    settings = json.loads(json.dumps(recipe))
    settings["dataset_path"] = str(dataset)
    settings["force_off_evaluation"] = {"dataset_path": str(dataset)}
    evaluation_split = replace(split, train_ids=(), validation_ids=(), calibration_ids=(),
                               selected_ids=split.test_ids)
    target_args = SimpleNamespace(output_dir=root, resume=True)
    off = "force_off" in policy
    for kind in (KINDS[2:] if off else KINDS[:2]):
        source = args.output_dir / "lightgbm" / kind
        manifest = read_json(source / "model_manifest.json")
        if off:
            table, all_targets, columns = _force_off_metadata_target_table(target_args, settings, evaluation_split, artifact)
            targets = [t for t in all_targets if t.estimator == ("regression" if kind.endswith("regression") else "direct_ratio")]
        else:
            table, targets, columns = _metadata_target_table(target_args, settings, evaluation_split, artifact, kind)
        if list(columns) != manifest["metadata_columns"]:
            raise ValueError("LightGBM transfer feature mismatch")
        rows = []
        for target in targets:
            entry = manifest["models"][target.key]
            predictions = {}
            for quantile in (.16, .5, .84):
                model = lgb.Booster(model_file=str(source / "models" / entry[f"q{quantile:g}"]["model_file"]))
                predictions[quantile] = model.predict(table.loc[:, columns])
            rows.extend(_normal_distribution_rows(
                table, target=target, predictions=predictions, method="lightgbm", kpi_kind=kind,
                sample_count=recipe["evaluation"]["num_scenarios"],
                scale_floor=max(.001, .01 * entry["training_target_std"])))
        destination = root / "lightgbm" / kind
        destination.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(destination / "distribution_test.csv", index=False)
        freeze(destination / "prediction_request.json", dict(dataset=str(dataset), test_ids=list(split.test_ids),
               source_hashes={str(p): digest(p) for p in sorted(source.rglob("*.txt"))},
               fitting="None: frozen source-policy LightGBM models"))


def distributions(root, fold, policies, variants=VARIANTS):
    import pandas as pd
    from .calibrated_model_benchmark import _emulator_distribution, _force_off_emulator_distribution

    frames = []
    for policy in policies:
        base = policy_root(root, policy)
        off = "force_off" in policy
        for kind in (KINDS[2:] if off else KINDS[:2]):
            for variant in variants:
                destination = base / "variants" / variant / ("force_off" if off else "scoring") / fold
                if off:
                    frame = _force_off_emulator_distribution(destination, method=variant,
                        source_model="structured_gated", kpi_kind=kind,
                        estimator="regression" if kind.endswith("regression") else "direct_ratio")
                else:
                    frame = _emulator_distribution(destination, method=variant, kpi_kind=kind)
                frames.append(frame)
            frames.append(pd.read_csv(base / "lightgbm" / kind / f"distribution_{fold}.csv"))
    return pd.concat(frames, ignore_index=True)


def validate_distributions(frame, ids, particles, horizons, methods=METHODS):
    import numpy as np
    import pandas as pd

    keys = ["kpi_kind", "profile_id", "horizon_hours", "direction"]
    expected = {(kind, b, h, direction) for kind in set(frame.kpi_kind) for b in ids for h in horizons
                for direction in (("force_off",) if kind.startswith("force_off") else ("up", "down"))}
    if set(frame.method) != set(methods) or not np.isfinite(frame[["truth", "prediction"]]).all().all():
        raise ValueError("Incomplete model roster or nonfinite KPI values")
    if frame.duplicated(["method", *keys, "sample_index"]).any():
        raise ValueError("Duplicate ensemble particles")
    reference = None
    for method, values in frame.groupby("method"):
        groups = values.groupby(keys)
        cells = groups.agg(truth=("truth", "first"), truths=("truth", "nunique"), count=("prediction", "size"))
        if set(cells.index) != expected or not cells.truths.eq(1).all() or not cells["count"].eq(particles).all():
            raise ValueError(f"Incomplete matched KPI cells: {method}")
        if not values.sample_index.isin(range(particles)).all():
            raise ValueError("Unexpected particle labels")
        truth = cells.truth.sort_index()
        if reference is not None:
            pd.testing.assert_index_equal(truth.index, reference.index)
            np.testing.assert_allclose(truth, reference, rtol=1e-6, atol=1e-6)
        reference = truth


def report(args, recipe, split):
    import pandas as pd
    from .generic_response_report import native_scores
    from .plant_accounting_report import paired_crps_bootstrap
    from .probabilistic_calibration import calibrate_distribution_table, ecc_reorder, summarize_joint_distribution_tables

    transfer = args.policy in POLICIES[2:]
    policies = (args.policy,) if transfer else POLICIES[:2]
    root = policy_root(args.output_dir, args.policy) if transfer else args.output_dir
    destination = root / "reports"
    destination.mkdir(parents=True, exist_ok=True)
    variants = tuple(recipe.get("benchmark_variants", VARIANTS))
    reference = "baseline" if "baseline" in variants else "structured"
    calibration = distributions(args.output_dir, "calibration", POLICIES[:2], variants)
    test = distributions(args.output_dir, "test", policies, variants)
    calibration = calibration[calibration.kpi_kind.isin(test.kpi_kind.unique())]
    for frame, ids in ((calibration, split.calibration_ids), (test, split.test_ids)):
        validate_distributions(frame, ids, recipe["evaluation"]["num_scenarios"], recipe["evaluation"]["horizons_hours"], (*variants, "lightgbm"))
    rns, metrics, reliability, buildings = calibrate_distribution_table(calibration, test, **recipe["calibration"])
    ecc = ecc_reorder(test, rns)
    for name, frame in (("raw_calibration_distributions", calibration), ("raw_test_distributions", test),
                        ("rns_test_distributions", rns), ("rns_ecc_test_distributions", ecc)):
        frame.to_csv(destination / (name + ".csv"), index=False)
    for name, frame in (("distribution_metrics", metrics), ("distribution_reliability", reliability),
                        ("distribution_by_building", buildings)):
        pd.concat([frame, frame[frame.calibration.eq("rns")].assign(calibration="rns_ecc")]).to_csv(destination / (name + ".csv"), index=False)
    summary, cells, native_reliability, native_buildings, pit = native_scores(test)
    paired = paired_crps_bootstrap(native_buildings, reference=reference)
    for name, frame in (("native_summary", summary), ("native_cells", cells), ("native_pit", pit),
                        ("native_reliability", native_reliability), ("native_buildings", native_buildings), ("paired_crps", paired)):
        frame.to_csv(destination / (name + ".csv"), index=False)
    joint = []
    # Setpoint and force-off ensembles come from separate annual rollouts. Do
    # not invent a joint copula by pairing their unrelated particle indices.
    for kinds in (KINDS[:2], KINDS[2:]):
        if set(kinds) <= set(test.kpi_kind):
            values = summarize_joint_distribution_tables(calibration, {"raw": test, "rns": rns, "rns_ecc": ecc}, kpi_kinds=kinds)
            values["policy_family"] = "force_off" if kinds[0].startswith("force_off") else "setpoint"
            joint.append(values)
    pd.concat(joint).to_csv(destination / "joint_distribution_metrics.csv", index=False)
    links = "".join(f'<li><a href="{html.escape(str(p.resolve()))}">{html.escape(str(p.relative_to(root)))}</a></li>'
                    for p in sorted((root / "variants").rglob("*.html")) if "profile_" in p.name)
    scope = ("Frozen tariff-policy transfer; calibration fitted on excitation policies only." if transfer
             else "Excitation policies; independent calibration buildings. Models share campaign folds.")
    (destination / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Paired campaign comparison</title>'
        '<style>body{font:15px system-ui;margin:30px}td,th{padding:6px}table{border-collapse:collapse}</style>'
        f'<h1>{len(variants)+1}-model campaign comparison</h1><p>{html.escape(scope)}</p>'
        '<p>RNS/ECC transform KPI ensembles only. Annual bands remain native. '
        'LightGBM is a metadata-only KPI benchmark; its synthetic marginal samples do not establish a learned joint copula. '
        'Single training seed; no pristine-holdout claim.</p><h2>Native KPIs</h2>'
        + summary.to_html(index=False) + f'<h2>Paired CRPS vs {reference}</h2>' + paired.to_html(index=False)
        + '<p><a href="distribution_metrics.csv">Raw/RNS/ECC marginal metrics</a> | '
        '<a href="joint_distribution_metrics.csv">Within-policy joint metrics</a></p>'
        + '<h2>Annual traces</h2><ul>' + links + '</ul>')


def benchmark_command(args, stages, variant=None):
    command = [sys.executable, "-u", "-m", "neural_building_emulator_refactor.lstm_hurdle_ablation",
               "--current-best-root", str(args.current_best_root), "--campaign-root", str(args.campaign_root),
               "--output-dir", str(args.output_dir), "--max-profiles", "1000", "--epochs", str(args.epochs),
               "--num-scenarios", str(args.num_scenarios), "--num-profile-plots", str(args.num_profile_plots),
               "--variants", *VARIANTS, "--stages", *stages]
    if args.exclude_missing_profile_ids:
        command += ["--exclude-missing-profile-ids", *map(str, args.exclude_missing_profile_ids)]
    if variant:
        command += ["--worker-variant", variant]
    return command


def job_plan(args):
    shared = [sys.executable, "-u", "-m", "neural_building_emulator_refactor.campaign_benchmark", "--campaign-root", str(args.campaign_root),
              "--output-dir", str(args.output_dir)]
    jobs = [("prepare", benchmark_command(args, ["split"]))]
    jobs += [("train_" + v, benchmark_command(args, ["train"], v)) for v in VARIANTS]
    jobs += [("lightgbm", [*shared, "--stage", "lightgbm"])]
    for policy in POLICIES:
        for fold in (("calibration", "test") if policy in POLICIES[:2] else ("test",)):
            for variant in VARIANTS:
                jobs.append((f"score_{policy}_{fold}_{variant}", [*shared, "--stage", "score", "--policy", policy,
                             "--fold", fold, "--variant", variant]))
        if policy in POLICIES[2:]:
            jobs.append((f"lightgbm_{policy}", [*shared, "--stage", "transfer_lightgbm", "--policy", policy]))
        if policy != POLICIES[0]:
            jobs.append((f"report_{policy}", [*shared, "--stage", "report", "--policy", policy]))
    return jobs


def supervise(args):
    import fcntl

    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    with (root / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = job_plan(args)
        freeze(root / "job_plan.json", jobs)
        logs = root / "logs"
        logs.mkdir(exist_ok=True)
        for index, (name, command) in enumerate(jobs):
            status = dict(state="running", job=name, index=index, jobs=len(jobs), supervisor_pid=os.getpid(),
                          started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), command=command)
            with (logs / (name + ".log")).open("a", buffering=1) as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                status["worker_pid"] = process.pid
                (root / "status.json").write_text(json.dumps(status, indent=2) + "\n")
                print(f"campaign job={name} pid={process.pid} log={log.name}", flush=True)
                result = process.wait()
            status.update(state="failed" if result else "job_complete", returncode=result)
            (root / "status.json").write_text(json.dumps(status, indent=2) + "\n")
            if result:
                raise RuntimeError(f"Campaign stopped at {name}; inspect its log")
        status["state"] = "complete"
        (root / "status.json").write_text(json.dumps(status, indent=2) + "\n")


def detach(args):
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["MPLCONFIGDIR"] = str(root / "matplotlib_cache")
    environment.setdefault("OMP_NUM_THREADS", "4")
    environment.setdefault("OPENBLAS_NUM_THREADS", "4")
    command = ["/usr/bin/caffeinate", "-i", "/usr/bin/nice", "-n", "10",
               sys.executable, "-u", "-m", "neural_building_emulator_refactor.campaign_benchmark",
               *[value for value in sys.argv[1:] if value != "--detach"]]
    with (root / "supervisor.log").open("a", buffering=1) as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True, env=environment)
    record = dict(pid=process.pid, command=command, cwd=str(Path.cwd()),
                  started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), log=str(root / "supervisor.log"))
    (root / "launcher.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


def main():
    if "--completed-structured-run" in sys.argv:
        from .complete_campaign_benchmark import main as complete_main
        complete_main()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--current-best-root", type=Path, default=Path(
        "output/neural_building_emulator_refactor/buffer_capacity_sweep_current_best_200/cap_0p5h"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--num-scenarios", type=int, default=100)
    parser.add_argument("--num-profile-plots", type=int, default=5)
    parser.add_argument("--exclude-missing-profile-ids", type=int, nargs="*", default=())
    parser.add_argument("--stage", choices=("all", "prepare", "score", "lightgbm", "transfer_lightgbm", "report"), default="all")
    parser.add_argument("--variant", choices=(*VARIANTS, "baseline", "two_tanks"))
    parser.add_argument("--policy", choices=POLICIES)
    parser.add_argument("--fold", choices=("calibration", "test"))
    parser.add_argument("--detach", action="store_true", help="Launch the serial campaign with caffeinate and persistent logs.")
    args = parser.parse_args()
    if args.detach:
        if args.stage != "all":
            parser.error("--detach requires --stage all")
        detach(args)
    elif args.stage == "all":
        supervise(args)
    elif args.stage == "prepare":
        subprocess.run(benchmark_command(args, ["split"]), check=True)
    else:
        recipe, split, artifacts = context(args)
        if args.stage == "score":
            if not args.variant or not args.policy or not args.fold:
                parser.error("Scoring requires --variant, --policy and --fold")
            score_policy(args, recipe, split, artifacts)
        elif args.stage == "lightgbm":
            from .calibrated_model_benchmark import _fit_lightgbm
            _fit_lightgbm(SimpleNamespace(output_dir=args.output_dir, resume=True), recipe, split, artifacts.get("baseline", artifacts.get("structured")))
        elif args.stage == "transfer_lightgbm":
            if args.policy not in POLICIES[2:]:
                parser.error("Transfer requires a tariff policy")
            lightgbm_transfer(args, recipe, split, artifacts.get("baseline", artifacts.get("structured")))
        else:
            report(args, recipe, split)


if __name__ == "__main__":
    main()
