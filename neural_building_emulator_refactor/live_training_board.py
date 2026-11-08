"""Publish a local-file dashboard snapshot without importing the training code."""

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import re
import time


VARIANTS = ("baseline", "two_tanks", "continuous", "hurdle_markov")
LABELS = {"baseline": "Baseline", "two_tanks": "Two tanks",
          "continuous": "Vanilla LSTM", "hurdle_markov": "Markov LSTM"}
FIELDS = ("epoch", "train_loss", "train_total_nrmse", "test_total_nrmse",
          "train_rmse_c", "test_rmse_c", "train_qroom_rmse_w_m2", "test_qroom_rmse_w_m2",
          "train_pel_rmse_w_m2", "test_pel_rmse_w_m2", "train_loss_energy",
          "train_loss_flex_kpi_crps", "train_loss_tank_temperature_crps",
          "total_elapsed_min", "epoch_elapsed_min", "skipped_update_batches", "nonfinite_train_batches")


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def complete_lines(path):
    """Writers may be appending a row; only consume newline-terminated rows."""
    text = path.read_text(errors="replace")
    if not text.endswith("\n"):
        text = text.rsplit("\n", 1)[0] + "\n" if "\n" in text else ""
    return text


def model_snapshot(root, variant):
    folder = root / "variants" / variant
    files = list((folder / "models").rglob("epoch_diagnostics_*.csv"))
    epochs = []
    if files:
        source = max(files, key=lambda p: p.stat().st_mtime)
        for row in csv.DictReader(io.StringIO(complete_lines(source))):
            item = {key: number(row.get(key)) for key in FIELDS}
            if all(item[key] is not None for key in ("epoch", "train_loss", "test_total_nrmse", "total_elapsed_min")):
                epochs.append(item)
    # Keep the last complete record per epoch if a log has been appended twice.
    epochs = sorted({r["epoch"]: r for r in epochs}.values(), key=lambda r: r["epoch"])
    completed = {int(r["epoch"]): r for r in epochs}
    batches = {}
    log_modified = None
    for path in sorted((root / "logs").glob(f"*train_{variant}_*.log")):
        log_modified = max(log_modified or 0, path.stat().st_mtime)
        for line in complete_lines(path).splitlines():
            if not line.startswith("batch_progress "):
                continue
            record = dict(re.findall(r"(\w+)=([^ ]+)", line))
            values = {key: number(record.get(key)) for key in ("epoch", "batch", "loss", "elapsed_min", "skipped_updates")}
            if any(v is None for v in values.values()):
                continue
            epoch, batch = int(values["epoch"]), int(values["batch"])
            prior = completed.get(epoch-1)
            if epoch > 1 and prior is None:
                continue
            values["minutes"] = (prior["total_elapsed_min"] if prior else 0) + values["elapsed_min"]
            batches[epoch, batch] = values
    batches = [batches[key] for key in sorted(batches)]
    for i, row in enumerate(batches):
        window = batches[max(0, i-9):i+1]
        row["rolling_loss"] = sum(x["loss"] for x in window)/len(window) if len(window) >= 5 else None
    artifact = read_json(folder / "artifact.json")
    selected_epoch = None
    if artifact:
        metadata = read_json(Path(artifact["artifact_dir"]) / "metadata_closed_loop_hp_contracting_probabilistic.json", {})
        if not metadata:
            metadata = read_json(Path(artifact["artifact_dir"]) / "refactor_manifest.json", {})
        selected_epoch = metadata.get("checkpoint_epoch")
    audit = []
    for path in sorted((folder / "validation").glob("epoch_*/gate_scores.csv")):
        for row in csv.DictReader(io.StringIO(complete_lines(path))):
            audit.append(dict(epoch=int(path.parent.name.split("_")[-1]), group=row["group"],
                              policy=row["policy"], metric=row["metric"], crps=number(row["crps"])))
    return dict(id=variant, label=LABELS[variant], epochs=epochs, batches=batches,
                selected_epoch=selected_epoch, artifact_complete=bool(artifact),
                epoch_decision=read_json(folder / "epoch_decision.json"), audit=audit,
                log_modified=log_modified,
                training_status=read_json(folder / "training_status.json", {}))


def pipeline_snapshot(root, state, request):
    """Observe existing files and the serial supervisor; never alter a job."""
    jobs = []
    source_policies = ("setpoint_excitation_v2", "force_off_excitation_v2")
    kinds = ("regression_based", "f", "force_off_regression", "force_off_f")
    for index, (name, command) in enumerate(read_json(root / "job_plan.json", [])):
        def argument(flag, default=None):
            return command[command.index(flag) + 1] if flag in command else default
        stage = argument("--stage")
        variant = argument("--variant")
        policy, fold = argument("--policy"), argument("--fold")
        destination = root if policy in (*source_policies, None) else root / "transfer" / policy
        reused = stage == "score" and variant in request.get("reused_variants", ("baseline", "two_tanks")) and fold == "test"
        if stage == "train":
            complete = (root / "variants" / variant / "artifact.json").is_file()
            label = "Train " + LABELS.get(variant, variant)
        elif stage == "score":
            scores = destination / "variants" / variant / ("force_off" if "force_off" in policy else "scoring") / fold
            complete = (scores / "campaign_complete.json").is_file()
            label = "Score " + LABELS.get(variant, variant)
        elif stage == "lightgbm":
            complete = all((root / "lightgbm" / kind / filename).is_file()
                           for kind in kinds for filename in
                           ("model_manifest.json", "distribution_calibration.csv", "distribution_test.csv"))
            label = "Fit LightGBM"
        elif stage == "transfer_lightgbm":
            selected = kinds[2:] if "force_off" in policy else kinds[:2]
            complete = all((destination / "lightgbm" / kind / "prediction_request.json").is_file() for kind in selected)
            label = "Transfer LightGBM"
        elif stage == "tank_diagnostics":
            complete = (root / "variants" / variant / "tank_diagnostics" / policy / "complete.json").is_file()
            label = "Tank-state diagnostics"
        elif stage == "compare":
            complete = (root / "reports/epoch_budget_comparison.html").is_file()
            label = "Compare 10- and 15-epoch budgets"
        else:
            complete = (destination / "reports/index.html").is_file()
            label = "Compare native / RNS / RNS–ECC"
        current = state.get("job") == name
        status = ("reused" if reused else "complete") if complete else "queued"
        if current and state.get("state") in ("running", "failed"):
            status = state["state"]
        log = root / "logs" / (name + "_campaign.log")
        jobs.append(dict(index=index, name=name, label=label, stage=stage,
            variant=variant, policy=policy, fold=fold, status=status,
            log_url=log.resolve().as_uri() if log.is_file() else None))
    recipe = request.get("recipe", {})
    horizons = recipe.get("evaluation", {}).get("horizons_hours", [])
    quantiles = recipe.get("lightgbm", {}).get("quantiles", [])
    # Two setpoint KPI families with two directions; two force-off families.
    expected = len(horizons) * 6 * len(quantiles)
    lightgbm = dict(saved_models=sum(len(list((root / "lightgbm" / kind / "models").glob("*.txt"))) for kind in kinds),
                    expected_models=expected,
                    completed_families=sum((root / "lightgbm" / kind / "model_manifest.json").is_file() for kind in kinds))
    return dict(jobs=jobs, lightgbm=lightgbm)


def snapshot(root):
    state = read_json(root / "status.json", {})
    request = read_json(root / "comparison_request.json", {})
    split = read_json(root / "split_manifest.json", {})
    launch = read_json(root / "launch.json", {})
    alive = None
    if launch.get("supervisor_pid"):
        try:
            os.kill(launch["supervisor_pid"], 0)
            alive = True
        except ProcessLookupError:
            alive = False
        except PermissionError:
            pass
    return dict(generated_unix=time.time(), generated_iso=datetime.now(timezone.utc).isoformat(),
        run=root.name, status=state, supervisor_alive=alive,
        protocol=request.get("epoch_protocol", {}),
        pipeline=pipeline_snapshot(root, state, request),
        folds={key: len(split.get(key, [])) for key in ("train_ids", "validation_ids", "calibration_ids", "test_ids")},
        models=[model_snapshot(root, variant) for variant in VARIANTS])


def publish(root, output):
    data = snapshot(root)
    text = json.dumps(data, allow_nan=False, separators=(",", ":"))
    # A local script resource works under file:// without CORS exceptions or a server.
    javascript = "window.gardenTrainingSnapshot=" + text + ";\nwindow.dispatchEvent(new Event('garden-training-update'));\n"
    for name, value in (("data.js", javascript), ("snapshot.json", text+"\n")):
        target = output / name
        temporary = target.with_suffix(target.suffix+".tmp")
        temporary.write_text(value)
        temporary.replace(target)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="Four-model campaign directory to monitor.")
    parser.add_argument("--interval", type=float, default=15)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("interval must be at least one second")
    root = args.run.resolve()
    output = root / "live_training"
    output.mkdir(parents=True, exist_ok=True)
    template = Path(__file__).resolve().parent / "viewer_assets" / "live_training.html"
    (output / "index.html").write_text(template.read_text())
    from plotly.offline import get_plotlyjs
    (output / "plotly.min.js").write_text(get_plotlyjs())
    while True:
        try:
            data = publish(root, output)
            if args.once or data["status"].get("state") in ("complete", "failed"):
                print(f"Final snapshot: {data['status'].get('state')}", flush=True)
                break
        except Exception as exc:
            print(f"Snapshot refresh failed: {type(exc).__name__}: {exc}", flush=True)
            if args.once:
                raise
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
