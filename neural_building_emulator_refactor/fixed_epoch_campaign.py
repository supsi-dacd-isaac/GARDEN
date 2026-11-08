"""Give both structured comparators a full 15-epoch budget and rescore them.

The earlier weights-only checkpoints cannot restore Adam's moments. Replay the
unchanged seeded training recipe, verify epochs 1 and 10, and continue with the
live optimizer through epoch 15. Reuse the completed LSTMs and LightGBM.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .campaign_benchmark import POLICIES, digest, freeze, read_json, policy_root

STRUCTURED = ("baseline", "two_tanks")
REUSED = ("continuous", "hurdle_markov")
VARIANTS = (*STRUCTURED, *REUSED)
MODEL_KIND = "closed_loop_hp_contracting_probabilistic"
PATH_FIELDS = {"output_dir", "model_checkpoint_dir"}


def assert_same_recipe(reference, candidate):
    """Changing the destination must be the only recipe change in a replay."""
    different = {key for key in set(reference) | set(candidate)
                 if key not in PATH_FIELDS and reference.get(key) != candidate.get(key)}
    if different:
        raise ValueError(f"Replay changes training settings: {sorted(different)}")
    if candidate["epochs"] != 15 or candidate["early_stopping_patience"] is not None:
        raise ValueError("Fixed comparison requires 15 epochs without early stopping")


def training_config(request, split, root, variant):
    from .fair_probabilistic_lstm_comparison import _experiment_configs
    from .legacy import build_legacy_config
    from .registry import get_model_spec
    from .artifacts import json_safe

    retained = {*split.train_ids, *split.validation_ids}
    config = _experiment_configs(request["variant_recipes"][variant], root / "variants" / variant / "models",
        max_profiles=len(retained), epochs=15, fixed_test_profile_ids=split.validation_ids,
        excluded_profile_ids=tuple(sorted(set(split.selected_ids)-retained)))["probabilistic_structured"]
    legacy = build_legacy_config(config, get_model_spec(config.model_name))
    assert_same_recipe(request["replay"][variant]["train_config"], json_safe(asdict(legacy)))
    return config, legacy


def verify_replay_checkpoint(reference, candidate):
    import equinox as eqx
    import jax
    import numpy as np
    from .artifacts import load_artifact

    before = read_json(reference / f"metadata_{MODEL_KIND}.json")
    after = read_json(candidate / f"metadata_{MODEL_KIND}.json")
    assert_same_recipe(before["train_config"], after["train_config"])
    for key in ("checkpoint_epoch", "train_ids", "test_ids", "selected_ids", "input_columns", "metadata_columns"):
        if before[key] != after[key]:
            raise ValueError(f"Replay checkpoint mismatch: {key}")
    filenames = (f"model_{MODEL_KIND}.eqx", f"scalers_{MODEL_KIND}.npz")
    exact = all(digest(reference / name) == digest(candidate / name) for name in filenames)
    maximum = 0.
    if not exact:
        a, b = load_artifact(reference), load_artifact(candidate)
        left = jax.tree_util.tree_leaves(eqx.filter(a.model, eqx.is_array))
        right = jax.tree_util.tree_leaves(eqx.filter(b.model, eqx.is_array))
        if len(left) != len(right):
            raise ValueError("Replay model structure differs")
        for x, y in zip(left, right):
            np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-7, err_msg="Replay weights diverged")
            if np.size(x):
                maximum = max(maximum, float(np.max(np.abs(np.asarray(x)-np.asarray(y)))))
        for channel in ("metadata", "inputs", "target"):
            for field in ("mean", "scale"):
                np.testing.assert_array_equal(getattr(getattr(a.scalers, channel), field),
                                              getattr(getattr(b.scalers, channel), field))
    return dict(epoch=after["checkpoint_epoch"], byte_identical=exact,
        maximum_parameter_difference=maximum, rtol=1e-6, atol=1e-7,
        reference=str(reference), candidate=str(candidate))


def link_directory(source, destination):
    """Reuse completed outputs only where this job plan never writes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() and destination.resolve() == source.resolve():
        return
    if destination.exists():
        raise ValueError(f"Refusing to replace existing results: {destination}")
    destination.symlink_to(source.resolve(), target_is_directory=True)


def prepare(args):
    from .calibrated_model_benchmark import BenchmarkSplit
    from .lstm_hurdle_ablation import _fingerprints
    from .artifacts import json_safe

    source, root = args.source_run.resolve(), args.output_dir.resolve()
    if source == root or read_json(source / "status.json")["state"] != "complete":
        raise ValueError("Use a new destination and a completed source comparison")
    old = read_json(source / "comparison_request.json")
    if args.campaign_root.resolve() != Path(old["recipe"]["campaign"]["root"]).resolve():
        raise ValueError("Requested dataset differs from the completed comparison")
    split_data = read_json(source / "split_manifest.json")
    split = BenchmarkSplit.from_json(split_data)
    data_hashes = read_json(source / "data_hashes.json")
    for path, expected in data_hashes.items():
        if digest(path) != expected:
            raise ValueError(f"Dataset completion marker changed: {path}")
    # The full benchmark preserved the implementation used for structured training.
    for path, expected in old["source_implementation_hashes"].items():
        p = Path(path)
        if p.parts[0] == "neural_building_emulator" and not p.name.startswith("test_") and digest(p) != expected:
            raise ValueError(f"Structured training implementation changed: {path}")
    replay = {}
    for variant in STRUCTURED:
        record = read_json(source / "variants" / variant / "artifact.json")
        selected = Path(record["artifact_dir"])
        if _fingerprints(selected) != record["artifact_hashes"]:
            raise ValueError(f"Source selected checkpoint changed: {variant}")
        checkpoints = {str(epoch): str(selected.parent / f"epoch_{epoch:03d}") for epoch in (1, 10)}
        metadata = read_json(Path(checkpoints["10"]) / f"metadata_{MODEL_KIND}.json")
        if metadata["train_ids"] != list(split.train_ids) or metadata["test_ids"] != list(split.validation_ids):
            raise ValueError("Source checkpoint split mismatch")
        replay[variant] = dict(checkpoints=checkpoints, train_config=metadata["train_config"],
            checkpoint_hashes={path: _fingerprints(Path(path)) for path in checkpoints.values()})
    request = dict(recipe=old["recipe"], variant_recipes=old["variant_recipes"], variants=list(VARIANTS),
        source_run=str(source), source_request_sha256=digest(source / "comparison_request.json"),
        replay=replay, reused_variants=list(REUSED),
        epoch_protocol=dict(initial_epochs=15, maximum_epochs=15, audit_epochs=[],
            profile_ids=list(split.validation_ids), particles=50, fixed_budget=True,
            decision="All neural models receive 15 training epochs; no epoch-10 stopping gate.",
            checkpoint_selection="Minimum three-target validation NRMSE across all 15 completed epochs"),
        scope="Replay unchanged structured recipes to recover Adam history, verify saved epochs 1 and 10, "
              "continue through 15, and recompute every structured native/calibrated test and transfer score. "
              "Reuse completed 15-epoch LSTMs and frozen LightGBM; calibration uses excitation policies only.")
    for variant in STRUCTURED:
        config, legacy = training_config(request, split, root, variant)
        freeze(root / "variants" / variant / "training_request.json", json_safe(asdict(config)))
        freeze(root / "variants" / variant / "legacy_training_request.json", json_safe(asdict(legacy)))
    freeze(root / "comparison_request.json", request)
    freeze(root / "split_manifest.json", split_data)
    freeze(root / "data_hashes.json", data_hashes)
    (root / "logs").mkdir(exist_ok=True)
    for variant in REUSED:
        prior = source / "variants" / variant
        record = read_json(prior / "artifact.json")
        artifact = Path(record["artifact_dir"])
        if _fingerprints(artifact) != record["artifact_hashes"]:
            raise ValueError(f"Reused checkpoint changed: {variant}")
        if read_json(artifact / "refactor_manifest.json")["experiment_config"]["optimizer"]["epochs"] != 15:
            raise ValueError("Reused LSTM must have a 15-epoch training budget")
        if not (artifact.parent / "epoch_015").is_dir():
            raise ValueError("Reused LSTM did not complete 15 epochs")
        destination = root / "variants" / variant
        freeze(destination / "artifact.json", dict(record, reused_current_best=True, source_run=str(source)))
        for name in ("training_status.json", "epoch_decision.json"):
            shutil.copy2(prior / name, destination / name)
        for name in ("models", "validation", "scoring", "force_off"):
            link_directory(prior / name, destination / name)
        shutil.copy2(source / "logs" / f"train_{variant}_campaign.log", root / "logs" / f"train_{variant}_reused.log")
    for policy in (None, *POLICIES[2:]):
        prior = source if policy is None else policy_root(source, policy)
        destination = root if policy is None else policy_root(root, policy)
        link_directory(prior / "lightgbm", destination / "lightgbm")
        if policy:
            for variant in REUSED:
                link_directory(prior / "variants" / variant, destination / "variants" / variant)
    paths = sorted({*Path("neural_building_emulator").rglob("*.py"),
                    *Path("neural_building_emulator_refactor").rglob("*.py")})
    freeze(root / "implementation_hashes.json", {str(p): digest(p) for p in paths})
    for path in paths:
        target = root / "source_snapshot" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    print("Prepared full 15-epoch budget; 2 structured replays, 2 reused LSTMs, reused LightGBM", flush=True)


def context(args):
    from .calibrated_model_benchmark import BenchmarkSplit
    from .lstm_hurdle_ablation import _fingerprints

    root = args.output_dir
    request = read_json(root / "comparison_request.json")
    for filename in ("implementation_hashes.json", "data_hashes.json"):
        for path, expected in read_json(root / filename).items():
            if digest(path) != expected:
                raise ValueError(f"Frozen provenance changed: {path}")
    for info in request["replay"].values():
        for path, expected in info["checkpoint_hashes"].items():
            if _fingerprints(Path(path)) != expected:
                raise ValueError(f"Replay reference changed: {path}")
    for variant in VARIANTS:
        record_path = root / "variants" / variant / "artifact.json"
        if record_path.exists():
            record = read_json(record_path)
            if _fingerprints(Path(record["artifact_dir"])) != record["artifact_hashes"]:
                raise ValueError(f"Checkpoint changed: {variant}")
    return request, BenchmarkSplit.from_json(read_json(root / "split_manifest.json"))


def train_fixed(args, request, split):
    from neural_building_emulator.train import run_closed_loop_training
    from .registry import get_model_spec
    from .artifacts import load_artifact, write_legacy_manifest
    from .profiles import saved_profile_ids
    from .lstm_hurdle_ablation import _fingerprints
    from .tank_temperature_monitor import write_status

    root = args.output_dir / "variants" / args.variant
    if (root / "artifact.json").exists():
        print(f"Reusing completed 15-epoch run: {args.variant}", flush=True)
        return
    config, legacy = training_config(request, split, args.output_dir, args.variant)
    references = request["replay"][args.variant]["checkpoints"]
    def observe(epoch, artifact):
        if str(epoch) in references:
            verification = verify_replay_checkpoint(Path(references[str(epoch)]), artifact)
            freeze(root / f"replay_verification_epoch_{epoch:03d}.json", verification)
            print(f"replay_verified {json.dumps(verification)}", flush=True)
        write_status(root / "training_status.json", dict(stage="training", completed_epoch=epoch,
            phase="reconstructing_optimizer" if epoch < 10 else "continuing_to_epoch_15", updated_unix=time.time()))
        return False
    started = time.perf_counter()
    run_closed_loop_training(legacy, epoch_observer=observe)
    selected = Path(legacy.model_checkpoint_dir) / legacy.model_kind / "selected"
    if not (selected.parent / "epoch_015").is_dir():
        raise ValueError("Structured training did not reach epoch 15")
    write_legacy_manifest(selected, spec=get_model_spec(config.model_name), experiment_config=config)
    artifact = load_artifact(selected)
    if saved_profile_ids(artifact, "train") != split.train_ids or saved_profile_ids(artifact, "test") != split.validation_ids:
        raise ValueError("Training/validation split changed")
    freeze(root / "artifact.json", dict(artifact_dir=str(selected.resolve()), variant=args.variant,
        training_minutes=(time.perf_counter()-started)/60, artifact_hashes=_fingerprints(selected),
        completed_epochs=15, training_mode="verified seeded replay plus continuation with live Adam"))
    write_status(root / "training_status.json", dict(stage="complete", completed_epoch=15,
        selected_epoch=artifact.legacy_artifact.metadata["checkpoint_epoch"], updated_unix=time.time()))


def job_plan(args):
    local = [sys.executable, "-u", "-m", "neural_building_emulator_refactor.fixed_epoch_campaign",
             "--output-dir", str(args.output_dir), "--campaign-root", str(args.campaign_root)]
    campaign = [sys.executable, "-u", "-m", "neural_building_emulator_refactor.campaign_benchmark",
                "--output-dir", str(args.output_dir), "--campaign-root", str(args.campaign_root)]
    jobs = [("train_"+v, [*local, "--stage", "train", "--variant", v]) for v in STRUCTURED]
    for policy in POLICIES:
        for fold in (("calibration", "test") if policy in POLICIES[:2] else ("test",)):
            for variant in STRUCTURED:
                jobs.append((f"score_{policy}_{fold}_{variant}", [*campaign, "--stage", "score",
                    "--policy", policy, "--fold", fold, "--variant", variant]))
        if policy != POLICIES[0]:
            jobs.append((f"report_{policy}", [*campaign, "--stage", "report", "--policy", policy]))
    for policy in POLICIES:
        jobs.append((f"tank_diagnostics_{policy}", [*local, "--stage", "tank_diagnostics",
            "--policy", policy, "--variant", "two_tanks"]))
    jobs.append(("compare_epoch_budgets", [*local, "--stage", "compare"]))
    return jobs


def compare_budgets(args, request):
    import numpy as np
    import pandas as pd
    source = Path(request["source_run"])
    tables = []
    for policy in (None, *POLICIES[2:]):
        old = source if policy is None else policy_root(source, policy)
        new = args.output_dir if policy is None else policy_root(args.output_dir, policy)
        before, after = [pd.read_csv(p / "reports/distribution_metrics.csv") for p in (old, new)]
        keys = ["method", "kpi_kind", "horizon_hours", "direction", "calibration"]
        merged = before.merge(after, on=keys, how="outer", suffixes=("_before", "_after"), indicator=True, validate="one_to_one")
        if not merged._merge.eq("both").all():
            raise ValueError("Epoch-budget comparisons contain different KPI cells")
        np.testing.assert_array_equal(merged["count_before"], merged["count_after"])
        merged["policy"] = policy or "excitation"
        tables.append(merged.drop(columns="_merge"))
    cells = pd.concat(tables, ignore_index=True)
    columns = ["crps_before", "crps_after", "coverage_90_before", "coverage_90_after", "width_90_before", "width_90_after"]
    summary = cells.groupby(["policy", "method", "kpi_kind", "calibration"])[columns].mean().reset_index()
    summary["crps_change_pct"] = 100*(summary.crps_after/summary.crps_before-1)
    root = args.output_dir / "reports"
    cells.to_csv(root / "epoch_budget_cells.csv", index=False)
    summary.to_csv(root / "epoch_budget_comparison.csv", index=False)
    (root / "epoch_budget_comparison.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Equal 15-epoch training budget</title>'
        '<style>body{font:15px system-ui;margin:30px}td,th{padding:6px}table{border-collapse:collapse}</style>'
        '<h1>Structured models: 10 versus 15 training epochs</h1>'
        '<p>Same building folds, recipes, seeds and evaluation settings. Best validation checkpoint '
        'within each budget. RNS/ECC refitted on excitation calibration buildings only.</p>'
        '<p><a href="index.html">Five-model excitation comparison</a> · '
        '<a href="../transfer/tariff_setpoint_v2/reports/index.html">Tariff setpoints</a> · '
        '<a href="../transfer/tariff_force_off_v2/reports/index.html">Tariff force-off</a></p>'
        + summary[summary.method.isin(STRUCTURED)].round(4).to_html(index=False))


def supervise(args):
    from .tank_temperature_monitor import write_status, wait_for_worker
    root = args.output_dir
    with (root / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = job_plan(args)
        freeze(root / "job_plan.json", jobs)
        for index, (name, command) in enumerate(jobs):
            stage = command[command.index("--stage")+1]
            variant = command[command.index("--variant")+1] if "--variant" in command else "baseline"
            log_path = root / "logs" / f"{name}_campaign.log"
            state = dict(state="running", job=name, stage=stage, variant=variant, index=index,
                         jobs=len(jobs), supervisor_pid=os.getpid(), started_unix=time.time(), command=command)
            with log_path.open("a", buffering=1) as log:
                worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                state["worker_pid"] = worker.pid
                print(f"fixed_epoch job={name} pid={worker.pid} log={log_path}", flush=True)
                code = wait_for_worker(worker, root, state, log_path)
            if code:
                state.update(state="failed", returncode=code)
                write_status(root / "status.json", state)
                raise SystemExit(code)
        state.update(state="complete", returncode=0, completed_unix=time.time())
        state.pop("worker_pid", None)
        write_status(root / "status.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--stage", choices=("prepare", "all", "train", "tank_diagnostics", "compare"), required=True)
    parser.add_argument("--variant", choices=STRUCTURED)
    parser.add_argument("--policy", choices=POLICIES)
    args = parser.parse_args()
    args.output_dir, args.campaign_root = args.output_dir.resolve(), args.campaign_root.resolve()
    if args.stage == "prepare":
        if args.source_run is None:
            parser.error("Preparation requires --source-run")
        prepare(args)
        return
    request, split = context(args)
    if args.stage == "all":
        supervise(args)
    elif args.stage == "train":
        if args.variant not in STRUCTURED:
            parser.error("Training requires --variant")
        train_fixed(args, request, split)
    elif args.stage == "tank_diagnostics":
        from .tank_temperature_diagnostics import run
        artifact = Path(read_json(args.output_dir / "variants" / args.variant / "artifact.json")["artifact_dir"])
        run(args, request["variant_recipes"][args.variant], split, artifact)
    else:
        compare_budgets(args, request)


if __name__ == "__main__":
    main()
