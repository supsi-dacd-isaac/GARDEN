"""Validation-only epoch budgeting and local monitoring for the tank comparison."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd


def write_status(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def epoch_protocol(folds, *, initial_epochs=10):
    if initial_epochs != 10:
        raise ValueError("The frozen validation gate is defined at epochs 8 and 10")
    return dict(initial_epochs=10, maximum_epochs=15, audit_epochs=[8, 10],
        profile_ids=list(folds["validation_ids"]), particles=50,
        minimum_relative_improvement=.01, maximum_group_regression=.01,
        decision="Equal weight to mean relative trace CRPS and mean relative KPI CRPS. "
                 "Continue from 10 to 15 only for >=1% combined improvement from 8 to 10 "
                 "and neither group worsens by >1%. All validation buildings, both excitation "
                 "policies; matched seeds and particles. No tariff or test data in stopping.",
        checkpoint_selection="Minimum original three-target validation total NRMSE across completed epochs",
        window_rotation="Fixed 15-epoch rotation schedule for both arms, including runs stopped at 10")


def extension_decision(before, after, protocol):
    keys = ["group", "policy", "metric"]
    if before.duplicated(keys).any() or after.duplicated(keys).any():
        raise ValueError("Duplicated validation score cells")
    paired = before.merge(after, on=keys, how="outer", suffixes=("_8", "_10"),
                          indicator=True, validate="one_to_one")
    if not paired._merge.eq("both").all() or set(paired.group) != {"trace", "kpi"}:
        raise ValueError("Validation score cells must match across epochs")
    x, y = paired.crps_8.to_numpy(float), paired.crps_10.to_numpy(float)
    if not np.isfinite(np.r_[x, y]).all() or (x <= 0).any() or (y < 0).any():
        raise ValueError("Invalid validation CRPS")
    paired["ratio"] = y / x
    ratios = paired.groupby("group").ratio.mean()
    improvement = float(1 - ratios.mean())
    extend = (improvement >= protocol["minimum_relative_improvement"]
              and float(ratios.max()) <= 1 + protocol["maximum_group_regression"])
    return dict(extend_to_15=bool(extend), stop_epoch=15 if extend else 10,
                combined_relative_improvement=improvement,
                trace_relative_improvement=float(1-ratios["trace"]),
                kpi_relative_improvement=float(1-ratios["kpi"]),
                rule=protocol["decision"])


def validation_score(args, recipe, split):
    from .tank_temperature_experiment import read, score_checkpoint, POLICIES
    from .artifacts import load_artifact

    protocol = read(args.output_dir / "comparison_request.json")["epoch_protocol"]
    if args.audit_epoch not in protocol["audit_epochs"] or args.policy not in POLICIES[:2]:
        raise ValueError("Only declared epochs and excitation policies may inform the gate")
    if tuple(protocol["profile_ids"]) != split.validation_ids:
        raise ValueError("Epoch gate must use exactly the validation fold")
    artifact = load_artifact(args.audit_artifact)
    metadata = artifact.legacy_artifact.metadata if artifact.backend == "legacy" else artifact.metadata
    if (tuple(metadata["test_ids"]) != split.validation_ids
            or tuple(metadata["train_ids"]) != split.train_ids
            or metadata["checkpoint_epoch"] != args.audit_epoch):
        raise ValueError("Epoch audit artifact/splits disagree with the frozen request")
    output = args.output_dir / "variants" / args.variant / "validation" / f"epoch_{args.audit_epoch:03d}" / args.policy
    score_checkpoint(args, recipe, args.audit_artifact, output, split.validation_ids,
                     num_scenarios=protocol["particles"], num_profile_plots=0)


def validation_summary(folder, protocol):
    from .tank_temperature_experiment import POLICIES, read, freeze
    from .calibrated_model_benchmark import _emulator_distribution, _force_off_emulator_distribution
    from .generic_response_report import native_scores

    rows, distributions = [], []
    for policy in POLICIES[:2]:
        path = folder / policy
        marker = read(path / "complete.json")
        if marker["ids"] != protocol["profile_ids"] or marker["particles"] != protocol["particles"]:
            raise ValueError("Incomplete or unmatched validation evaluation")
        traces = pd.read_csv(path / "temporal_trace_probabilistic_scores.csv")
        if "model" in traces:
            traces = traces[traces.model.eq("structured_gated")]
        if set(traces.signal) != {"Tin", "Qroom", "Pel_SH"} or len(traces) != 3:
            raise ValueError("Missing shared trace scores")
        for row in traces.itertuples():
            if (row.profile_count != len(protocol["profile_ids"])
                    or row.min_scenarios_per_profile != protocol["particles"]
                    or row.max_scenarios_per_profile != protocol["particles"]):
                raise ValueError("Incomplete trace ensembles in validation")
            rows.append(dict(group="trace", policy=policy, metric=row.signal, crps=row.mean_crps))
        for kind in ("regression", "f"):
            if "force_off" in policy:
                frame = _force_off_emulator_distribution(path, method="audit", source_model="structured_gated",
                    estimator="regression" if kind == "regression" else "direct_ratio", kpi_kind=kind)
            else:
                frame = _emulator_distribution(path, method="audit", kpi_kind=kind)
            frame["kpi_kind"] = policy + "_" + kind
            distributions.append(frame)
    summary, _, _, buildings, pit = native_scores(pd.concat(distributions, ignore_index=True))
    if not pit.particles.eq(protocol["particles"]).all():
        raise ValueError("Incomplete KPI ensembles in validation")
    for row in summary.itertuples():
        policy, kind = row.kpi_kind.rsplit("_", 1)
        rows.append(dict(group="kpi", policy=policy, metric=kind, crps=row.crps))
    buildings.to_csv(folder / "building_kpis.csv", index=False)
    scores = pd.DataFrame(rows)
    scores.to_csv(folder / "gate_scores.csv", index=False)
    freeze(folder / "summary_complete.json", dict(cells=len(buildings), protocol=protocol))
    return scores


def train_monitored(args, recipe, split, protocol, *,
                    validation_module="neural_building_emulator_refactor.tank_temperature_experiment"):
    from dataclasses import asdict, replace
    from .tank_temperature_experiment import read, freeze, POLICIES
    from .calibrated_model_benchmark import _eligible_hp_ids
    from .fair_probabilistic_lstm_comparison import _experiment_configs
    from .legacy import build_legacy_config
    from .registry import get_model_spec
    from .artifacts import write_legacy_manifest, load_artifact, json_safe
    from .profiles import saved_profile_ids
    from neural_building_emulator.train import run_closed_loop_training

    root = args.output_dir / "variants" / args.variant
    record = root / "artifact.json"
    if record.exists():
        return Path(read(record)["artifact_dir"])
    retained = {*split.train_ids, *split.validation_ids}
    eligible = set(_eligible_hp_ids(Path(recipe["dataset_path"])))
    is_lstm = args.variant in ("continuous", "hurdle_markov")
    config = _experiment_configs(recipe, root / "models", max_profiles=len(retained),
        epochs=protocol["maximum_epochs"], fixed_test_profile_ids=split.validation_ids,
        excluded_profile_ids=tuple(sorted(eligible-retained)))["probabilistic_lstm" if is_lstm else "probabilistic_structured"]
    if is_lstm:
        from .lstm_hurdle_ablation import VARIANT_MODELS
        config = replace(config, model_name=VARIANT_MODELS[args.variant])
        freeze(root / "training_request.json", json_safe(asdict(config)))
    spec = get_model_spec(config.model_name)
    legacy = None if is_lstm else build_legacy_config(config, spec)
    if (config.optimizer.early_stopping_patience if is_lstm else legacy.early_stopping_patience) is not None:
        raise ValueError("The validation gate must be the only early stopping rule")

    def observe(epoch, artifact):
        state_path = root / "training_status.json"
        write_status(state_path, dict(stage="training", completed_epoch=epoch, updated_unix=time.time()))
        if epoch not in protocol["audit_epochs"]:
            return False
        for policy in POLICIES[:2]:
            command = [sys.executable, "-u", "-m", validation_module,
                "--stage", "validation_score", "--output-dir", str(args.output_dir),
                "--campaign-root", str(args.campaign_root), "--variant", args.variant,
                "--policy", policy, "--audit-epoch", str(epoch), "--audit-artifact", str(artifact)]
            log_path = args.output_dir / "logs" / f"validation_{args.variant}_{epoch:03d}_{policy}.log"
            with log_path.open("a") as log:
                worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                write_status(state_path, dict(stage="annual_validation", completed_epoch=epoch,
                    policy=policy, worker_pid=worker.pid, log=str(log_path), updated_unix=time.time()))
                code = worker.wait()
            if code:
                raise RuntimeError(f"Annual validation failed ({code}): {log_path}")
        audit_root = root / "validation"
        scores = validation_summary(audit_root / f"epoch_{epoch:03d}", protocol)
        stop = False
        if epoch == protocol["initial_epochs"]:
            before = pd.read_csv(audit_root / "epoch_008/gate_scores.csv")
            a = pd.read_csv(audit_root / "epoch_008/building_kpis.csv")
            b = pd.read_csv(audit_root / "epoch_010/building_kpis.csv")
            keys = ["kpi_kind", "profile_id", "horizon_hours", "direction"]
            paired = a.merge(b, on=keys, how="outer", indicator=True, validate="one_to_one")
            if not paired._merge.eq("both").all():
                raise ValueError("KPI cells changed between validation epochs")
            np.testing.assert_allclose(paired.truth_x, paired.truth_y)
            decision = extension_decision(before, scores, protocol)
            freeze(root / "epoch_decision.json", decision)
            print(f"validation_epoch_decision {json.dumps(decision)}", flush=True)
            stop = not decision["extend_to_15"]
        write_status(state_path, dict(stage="finalizing" if stop else "training",
            completed_epoch=epoch, updated_unix=time.time()))
        return stop

    started = time.perf_counter()
    if is_lstm:
        from .probabilistic_lstm_trainer import train_probabilistic_lstm
        artifact_path = train_probabilistic_lstm(config, spec, epoch_observer=observe)
    else:
        run_closed_loop_training(legacy, epoch_observer=observe)
        artifact_path = Path(legacy.model_checkpoint_dir) / legacy.model_kind / "selected"
        write_legacy_manifest(artifact_path, spec=spec, experiment_config=config)
    artifact = load_artifact(artifact_path)
    if saved_profile_ids(artifact, "train") != split.train_ids or saved_profile_ids(artifact, "test") != split.validation_ids:
        raise ValueError("Training or validation split changed")
    values = dict(artifact_dir=str(artifact_path.resolve()), variant=args.variant,
                  training_minutes=(time.perf_counter()-started)/60, reused_current_best=False)
    if is_lstm:
        from .lstm_hurdle_ablation import _fingerprints
        values["artifact_hashes"] = _fingerprints(artifact_path)
    freeze(record, values)
    metadata = artifact.legacy_artifact.metadata if artifact.backend == "legacy" else artifact.metadata
    write_status(root / "training_status.json", dict(stage="complete",
        selected_epoch=metadata["checkpoint_epoch"], updated_unix=time.time()))
    return artifact_path


def wait_for_worker(worker, root, state, log_path):
    """Persist a heartbeat, epoch progress and nested validation-worker status."""
    root = Path(root)
    variant = root / "variants" / state["variant"]
    state["log"] = str(log_path)
    while True:
        state["heartbeat_unix"] = time.time()
        state["elapsed_minutes"] = (time.time()-state["started_unix"])/60
        if log_path.exists():
            info = log_path.stat()
            state.update(log_bytes=info.st_size, seconds_since_log_update=time.time()-info.st_mtime)
            with log_path.open("rb") as handle:
                handle.seek(max(0, info.st_size-32768))
                lines = handle.read().decode(errors="replace").splitlines()
            progress = [line for line in lines if line.startswith("batch_progress ")]
            if progress:
                state["batch_progress"] = progress[-1]
        paths = sorted((variant / "models").rglob("epoch_diagnostics_*.csv"), key=lambda p:p.stat().st_mtime)
        if paths:
            with paths[-1].open() as handle:
                rows = list(csv.DictReader(handle))
            if rows:
                state["training_progress"] = {key:rows[-1].get(key) for key in
                    ("epoch", "test_total_nrmse", "train_loss", "epoch_elapsed_min", "skipped_update_batches", "nonfinite_train_batches")}
        status = variant / "training_status.json"
        if status.exists():
            state["training_status"] = json.loads(status.read_text())
        code = worker.poll()
        if code is not None:
            state["worker_returncode"] = code
        write_status(root / "status.json", state)
        if code is not None:
            return code
        time.sleep(30)
