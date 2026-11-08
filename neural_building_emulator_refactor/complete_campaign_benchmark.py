"""Complete the serial campaign using already trained structured comparators.

Entry point: campaign_benchmark --completed-structured-run PATH. The completed
experiment is read-only; new checkpoints, calibration and reports live separately.
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
from types import SimpleNamespace

from .campaign_benchmark import POLICIES, FOLDS, digest, freeze, read_json, score_request

VARIANTS = ("baseline", "two_tanks", "continuous", "hurdle_markov")
LSTMS = VARIANTS[2:]


def prepare(args):
    from .artifacts import load_artifact
    from .calibrated_model_benchmark import BenchmarkSplit
    from .lstm_hurdle_ablation import _fingerprints, _variant_recipe
    from .profiles import saved_profile_ids

    source, root = args.completed_structured_run.resolve(), args.output_dir.resolve()
    if source == root:
        raise ValueError("Use a new output directory; completed results are read-only")
    old = read_json(source / "comparison_request.json")
    if read_json(source / "status.json")["state"] != "complete":
        raise ValueError("Structured experiment must have completed successfully")
    split = BenchmarkSplit.from_json(read_json(source / "split_manifest.json"))
    campaign_manifest = args.campaign_root.resolve() / "split_manifest.json"
    manifest = read_json(campaign_manifest)
    if digest(campaign_manifest) != old["dataset_manifest_sha256"]:
        raise ValueError("Campaign manifest changed since structured training")
    for fold in (*FOLDS, "selected_ids"):
        if list(getattr(split, fold)) != sorted(manifest[fold]):
            raise ValueError(f"Completed run does not preserve full campaign fold: {fold}")
    data_hashes = read_json(source / "data_hashes.json")
    for path, expected in data_hashes.items():
        if digest(path) != expected:
            raise ValueError(f"Dataset completion record changed: {path}")
    protocol = old["epoch_protocol"]
    if not protocol or protocol["profile_ids"] != list(split.validation_ids):
        raise ValueError("Missing matched validation-only epoch protocol")
    recipe = json.loads(json.dumps(old["variant_recipes"]["baseline"]))
    if recipe.get("comparison_protocol") != "current_best_shared_losses":
        raise ValueError("Shared LSTM losses must already be specified")
    if not recipe["probabilistic_structured"]["hp_dhw_metadata"]:
        raise ValueError("Expected structured DHW metadata inputs")
    recipe["hp_dhw_metadata"] = True
    recipe["benchmark_variants"] = list(VARIANTS)
    recipe["force_off_evaluation"]["raw_only"] = False
    recipe["campaign"]["calibration_scope"] = "Excitation calibration buildings only; no tariff fitting"
    if Path(recipe["dataset_path"]).resolve() != args.campaign_root.resolve() / "data/control=setpoint_excitation_v2":
        raise ValueError("Campaign root differs from completed training")
    recipes = {v: _variant_recipe(recipe, v) for v in LSTMS}
    recipes.update({v: old["variant_recipes"][v] for v in VARIANTS[:2]})
    freeze(root / "split_manifest.json", asdict(split))
    freeze(root / "comparison_request.json", dict(recipe=recipe, variants=list(VARIANTS),
        variant_recipes=recipes, epoch_protocol=protocol, completed_structured_run=str(source),
        source_request_sha256=digest(source / "comparison_request.json"),
        source_implementation_hashes=read_json(source / "implementation_hashes.json"),
        scope="Reuse both completed structured checkpoints and native test scores; train vanilla and "
              "asymmetric-Markov LSTMs on matched inputs, shared losses and validation epoch gate. "
              "Two-tank auxiliary supervision remains specific to that architecture. Fit LightGBM "
              "on excitation training/validation only, and RNS/ECC on reserved excitation calibration only."))
    freeze(root / "data_hashes.json", data_hashes)
    (root / "logs").mkdir(exist_ok=True)
    artifacts = {}
    reused = {}
    for variant in VARIANTS[:2]:
        prior = source / "variants" / variant
        record = read_json(prior / "artifact.json")
        artifact = Path(record["artifact_dir"])
        expected = read_json(prior / "checkpoint_hashes.json")
        if {str(p): digest(p) for p in artifact.iterdir() if p.is_file()} != expected:
            raise ValueError(f"Structured checkpoint changed: {variant}")
        loaded = load_artifact(artifact)
        if saved_profile_ids(loaded, "train") != split.train_ids or saved_profile_ids(loaded, "test") != split.validation_ids:
            raise ValueError(f"Checkpoint fold mismatch: {variant}")
        artifacts[variant] = artifact
        target = root / "variants" / variant
        freeze(target / "artifact.json", dict(record, artifact_hashes=_fingerprints(artifact),
            reused_current_best=True, source_run=str(source)))
        for filename in ("epoch_decision.json", "training_status.json"):
            shutil.copy2(prior / filename, target / filename)
        for path in (prior / "models").rglob("epoch_diagnostics_*.csv"):
            dest = target / "models" / path.relative_to(prior / "models")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        for path in (prior / "validation").glob("epoch_*/gate_scores.csv"):
            dest = target / "validation" / path.relative_to(prior / "validation")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        for path in (source / "logs").glob(f"*train_{variant}_*.log"):
            shutil.copy2(path, root / "logs" / path.name)
        for policy in POLICIES:
            origin = prior / "evaluation" / policy
            marker = read_json(origin / "complete.json")
            if (marker["ids"] != list(split.test_ids)
                    or marker["particles"] != recipe["evaluation"]["num_scenarios"]
                    or marker["num_profile_plots"] != recipe["evaluation"]["num_profile_plots"]
                    or marker["artifact_hashes"] != expected or marker["policy"] != policy):
                raise ValueError(f"Cannot reuse unmatched test evaluation: {origin}")
            job = SimpleNamespace(output_dir=root, campaign_root=args.campaign_root,
                                  variant=variant, policy=policy, fold="test")
            destination, request = score_request(job, recipe, split, artifacts)
            # Separate copies keep future writes from mutating the original run.
            shutil.copytree(origin, destination, dirs_exist_ok=True)
            freeze(destination / "campaign_complete.json", request)
            reused[f"{variant}/{policy}"] = dict(source=str(origin), destination=str(destination),
                files={p.name: digest(p) for p in origin.iterdir() if p.is_file()})
    freeze(root / "reused_test_evaluations.json", reused)
    paths = sorted({*Path("neural_building_emulator").rglob("*.py"),
                    *Path("neural_building_emulator_refactor").rglob("*.py")})
    freeze(root / "implementation_hashes.json", {str(p): digest(p) for p in paths})
    for path in paths:
        dest = root / "source_snapshot" / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    print(f"Prepared {len(split.selected_ids)} buildings; reused 2 checkpoints and 8 test evaluations", flush=True)


def context(args):
    from .calibrated_model_benchmark import BenchmarkSplit
    from .lstm_hurdle_ablation import _fingerprints

    root = args.output_dir
    for name in ("implementation_hashes.json", "data_hashes.json"):
        for path, expected in read_json(root / name).items():
            if digest(path) != expected:
                raise ValueError(f"Frozen provenance changed: {path}")
    request = read_json(root / "comparison_request.json")
    for variant in VARIANTS:
        path = root / "variants" / variant / "artifact.json"
        if path.exists():
            record = read_json(path)
            if _fingerprints(Path(record["artifact_dir"])) != record["artifact_hashes"]:
                raise ValueError(f"Checkpoint changed: {variant}")
    return request, BenchmarkSplit.from_json(read_json(root / "split_manifest.json"))


def job_plan(args):
    shared = [sys.executable, "-u", "-m", "neural_building_emulator_refactor.campaign_benchmark",
              "--campaign-root", str(args.campaign_root), "--output-dir", str(args.output_dir)]
    jobs = [("train_" + v, [*shared, "--completed-structured-run", str(args.completed_structured_run),
                            "--stage", "train", "--variant", v]) for v in LSTMS]
    jobs.append(("lightgbm", [*shared, "--stage", "lightgbm"]))
    for policy in POLICIES:
        for fold in (("calibration", "test") if policy in POLICIES[:2] else ("test",)):
            for variant in VARIANTS:
                jobs.append((f"score_{policy}_{fold}_{variant}", [*shared, "--stage", "score",
                             "--policy", policy, "--fold", fold, "--variant", variant]))
        if policy in POLICIES[2:]:
            jobs.append((f"lightgbm_{policy}", [*shared, "--stage", "transfer_lightgbm", "--policy", policy]))
        if policy != POLICIES[0]:
            jobs.append((f"report_{policy}", [*shared, "--stage", "report", "--policy", policy]))
    return jobs


def supervise(args):
    from .tank_temperature_monitor import wait_for_worker, write_status

    root = args.output_dir
    with (root / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        context(args)
        jobs = job_plan(args)
        freeze(root / "job_plan.json", jobs)
        for index, (name, command) in enumerate(jobs):
            variant = command[command.index("--variant")+1] if "--variant" in command else "baseline"
            stage = command[command.index("--stage")+1]
            # Match the existing live dashboard's batch-log naming convention.
            path = root / "logs" / (name + "_campaign.log")
            state = dict(state="running", job=name, stage=stage, variant=variant, index=index,
                jobs=len(jobs), supervisor_pid=os.getpid(), started_unix=time.time(), command=command)
            with path.open("a", buffering=1) as log:
                worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                state["worker_pid"] = worker.pid
                print(f"campaign job={name} pid={worker.pid} log={path}", flush=True)
                code = wait_for_worker(worker, root, state, path)
            if code:
                state.update(state="failed", returncode=code)
                write_status(root / "status.json", state)
                raise SystemExit(code)
        state.update(state="complete", returncode=0, completed_unix=time.time())
        state.pop("worker_pid", None)
        write_status(root / "status.json", state)


def detach(args):
    root = args.output_dir.resolve()
    environment = dict(os.environ, MPLCONFIGDIR=str(root / "matplotlib_cache"))
    environment.setdefault("OMP_NUM_THREADS", "4")
    environment.setdefault("OPENBLAS_NUM_THREADS", "4")
    command = ["/usr/bin/caffeinate", "-i", "/usr/bin/nice", "-n", "10", sys.executable,
               "-u", "-m", "neural_building_emulator_refactor.campaign_benchmark",
               *[v for v in sys.argv[1:] if v != "--detach"]]
    # Hold the lock across the duplicate-launch check and process creation.
    with (root / "launch.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launch = root / "launch.json"
        if launch.exists():
            prior = read_json(launch)
            try:
                os.kill(prior["supervisor_pid"], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError("Campaign launcher is already alive")
        with (root / "supervisor.log").open("a") as log:
            worker = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, env=environment)
        launch.write_text(json.dumps(dict(supervisor_pid=worker.pid, command=command,
            started_unix=time.time(), cwd=str(Path.cwd())), indent=2) + "\n")
    print(f"Launched serial benchmark pid={worker.pid}; {root / 'status.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed-structured-run", type=Path)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "prepare", "train", "validation_score"), default="all")
    parser.add_argument("--variant", choices=LSTMS)
    parser.add_argument("--policy", choices=POLICIES[:2])
    parser.add_argument("--audit-epoch", type=int)
    parser.add_argument("--audit-artifact", type=Path)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.campaign_root = args.campaign_root.resolve()
    if args.stage == "prepare":
        prepare(args)
        return
    request, split = context(args)
    if args.completed_structured_run is None:
        args.completed_structured_run = Path(request["completed_structured_run"])
    if args.stage == "all":
        detach(args) if args.detach else supervise(args)
    elif args.stage == "train":
        if args.variant not in LSTMS:
            parser.error("--stage train requires an LSTM --variant")
        from .tank_temperature_monitor import train_monitored
        train_monitored(args, request["variant_recipes"][args.variant], split, request["epoch_protocol"],
                        validation_module="neural_building_emulator_refactor.complete_campaign_benchmark")
    else:
        from .tank_temperature_monitor import validation_score
        validation_score(args, request["variant_recipes"][args.variant], split)


if __name__ == "__main__":
    main()
