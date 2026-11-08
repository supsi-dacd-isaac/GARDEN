"""Frozen four-model transfer with source RNS/ECC and separate target adaptation.

Run from the repository root. Each neural fold runs in a fresh process; no model
is fitted, and no target-policy labels enter the source-calibrated predictions.
"""

from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .calibrated_model_benchmark import BenchmarkSplit, KPI_KINDS, _emulator_distribution
from .generic_response_report import native_scores
from .kpi_policy_transfer import _html_table, _predict_lightgbm
from .plant_accounting_report import paired_crps_bootstrap
from .probabilistic_calibration import (
    GROUP_COLUMNS, calibrate_distribution_table, ecc_reorder,
    summarize_joint_distribution_tables,
)
from .scoring import score_artifact


BASE = Path("output/neural_building_emulator_refactor")
SOURCE = BASE / "current_best_cap0p5_lstm_markov_1000_e15"
LIGHTGBM = BASE / "calibrated_four_model_benchmark_current_best_1000_rns_ecc"
METHODS = ("structured", "continuous", "hurdle_markov", "lightgbm")
CELL = ["kpi_kind", "profile_id", "horizon_hours", "direction"]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def freeze(path, value):
    if path.exists() and read_json(path) != value:
        raise ValueError(f"Frozen request changed: {path}; use a new output directory")
    if not path.exists():
        write_json(path, value)


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError(f"Frozen source changed: {path}")


def flatten(value, prefix=""):
    if isinstance(value, dict):
        return {key: item for name, child in value.items()
                for key, item in flatten(child, f"{prefix}.{name}" if prefix else name).items()}
    return {prefix: value}


def audit_domains(source, target, split, output):
    rows = []
    for fold, ids in (("calibration", split.calibration_ids), ("test", split.test_ids)):
        for profile in ids:
            a = read_json(source / f"egid={profile}/simulation_metadata.json")
            b = read_json(target / f"egid={profile}/simulation_metadata.json")
            for section in ("inputs", "resolved_building_model", "resolved_plant_design",
                            "resolved_infiltration", "gardenCompatibility"):
                left, right = flatten(a.get(section), section), flatten(b.get(section), section)
                for key in sorted(left.keys() | right.keys()):
                    if left.get(key) != right.get(key):
                        rows.append(dict(fold=fold, profile_id=profile, field=key,
                                         source=json.dumps(left.get(key)),
                                         target=json.dumps(right.get(key))))
    differences = pd.DataFrame(rows, columns=["fold", "profile_id", "field", "source", "target"])
    differences.to_csv(output / "domain_differences.csv", index=False)
    summary = differences.groupby(["fold", "field"]).agg(
        buildings=("profile_id", "nunique"), source_example=("source", "first"),
        target_example=("target", "first")).reset_index()
    summary.to_csv(output / "domain_difference_summary.csv", index=False)
    return summary


def context(args):
    recipe = read_json(args.source / "comparison_request.json")["recipe"]
    split = BenchmarkSplit.from_json(read_json(args.source / "split_manifest.json"))
    old_split = BenchmarkSplit.from_json(read_json(args.lightgbm / "resolved_recipe.json")["split"])
    for fold in ("train_ids", "validation_ids", "calibration_ids", "test_ids"):
        if getattr(split, fold) != getattr(old_split, fold):
            raise ValueError(f"LightGBM/neural split mismatch: {fold}")
    if set(split.calibration_ids) & set(split.test_ids):
        raise ValueError("Calibration/test leakage")
    artifacts = {}
    hashes = {}
    for method in METHODS[:-1]:
        selected = read_json(args.source / f"variants/{method}/artifact.json")
        artifacts[method] = Path(selected["artifact_dir"]).resolve()
        hashes.update({str(artifacts[method] / name): value
                       for name, value in selected["artifact_hashes"].items()})
    verify_hashes(hashes)
    verify_hashes(read_json(args.source / "implementation_hashes.json"))
    verify_hashes(read_json(args.source / "calibration_comparison/input_hashes.json"))
    return recipe, split, artifacts, hashes


def prepare(args, recipe, split, artifacts, hashes):
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [args.source / "comparison_request.json", args.source / "split_manifest.json",
             args.source / "implementation_hashes.json", Path(__file__).resolve()]
    paths += [args.source / f"calibration_comparison/raw_{fold}_distributions.csv"
              for fold in ("calibration", "test")]
    paths += list((args.lightgbm / "lightgbm").glob("*/model_manifest.json"))
    paths += list((args.lightgbm / "lightgbm").glob("*/models/*.txt"))
    for profile in (*split.calibration_ids, *split.test_ids):
        paths += [args.dataset / f"egid={profile}" / name
                  for name in ("simulation_metadata.json", "timeseries.parquet")]
    hashes.update({str(path.resolve()): digest(path) for path in paths})
    freeze(args.output / "request.json", dict(
        source=str(args.source.resolve()), lightgbm=str(args.lightgbm.resolve()),
        dataset=str(args.dataset.resolve()), split=read_json(args.source / "split_manifest.json"),
        artifacts={key: str(value) for key, value in artifacts.items()},
        evaluation=recipe["evaluation"], seed=recipe["seed"], hashes=hashes,
        calibration=recipe["calibration"], joint_normalization="source calibration truth std",
        zero_shot="Frozen predictors and source-policy calibration; no target-label fitting",
        adaptation="Frozen predictors, RNS fitted only on target calibration buildings",
    ))
    audit_domains(Path(recipe["dataset_path"]), args.dataset, split, args.output)


def score_fold(args, recipe, split, artifacts):
    destination = args.output / "scoring" / args.fold / args.method
    marker = destination / "complete.json"
    if marker.exists():
        verify_hashes(read_json(marker)["hashes"])
        print(f"reuse {args.method}:{args.fold}", flush=True)
        return
    ids = getattr(split, f"{args.fold}_ids")
    evaluation = recipe["evaluation"]
    plots = int(evaluation["num_profile_plots"]) if args.fold == "test" else 0
    start = time.monotonic()
    print(f"score {args.method}:{args.fold} buildings={len(ids)} "
          f"particles={evaluation['num_scenarios']}", flush=True)
    score_artifact(
        artifacts[args.method], dataset_path=args.dataset, output_dir=destination,
        profile_ids=ids, kpi_mode="scenario_average", num_scenarios=int(evaluation["num_scenarios"]),
        hp_scenario_mode="bernoulli", ventilation_rollout_mode=evaluation["ventilation_rollout_mode"],
        horizons_hours=tuple(evaluation["horizons_hours"]),
        min_setpoint_change=evaluation["min_setpoint_change"], min_events=evaluation["min_events"],
        kpi_estimator="regression", controls=evaluation["controls"],
        seed=int(recipe["seed"]) + (1000 if args.fold == "test" else 0),
        num_profile_plots=plots, profile_plot_ids=ids[:plots] or None, evaluate_direct_f=True,
    )
    summary = read_json(destination / "summary.json")
    if summary["profile_count"] != len(ids) or summary["num_scenarios"] != evaluation["num_scenarios"]:
        raise ValueError("Incomplete annual scoring")
    write_json(marker, dict(minutes=(time.monotonic() - start) / 60,
                           hashes={str(p.resolve()): digest(p) for p in destination.iterdir()
                                   if p.is_file() and p != marker}))


def load_source(args, fold):
    frame = pd.read_csv(args.source / f"calibration_comparison/raw_{fold}_distributions.csv")
    frame = frame[frame.method.isin((*METHODS[:-1], "historical_lightgbm"))
                  & frame.kpi_kind.isin(KPI_KINDS)].copy()
    frame["method"] = frame.method.replace({"historical_lightgbm": "lightgbm"})
    return frame


def load_target(args, fold):
    frames = [_emulator_distribution(args.output / "scoring" / fold / method,
                                    method=method, kpi_kind=kind)
              for method in METHODS[:-1] for kind in KPI_KINDS]
    frames += [pd.read_csv(args.output / f"lightgbm/{kind}/distribution_{fold}.csv")
               for kind in KPI_KINDS]
    return pd.concat(frames, ignore_index=True)


def validate_distribution(frame, ids, particles, horizons=(.5, 1., 2., 3.)):
    """Fail closed instead of silently comparing different buildings or KPI cells."""
    if set(frame.method) != set(METHODS):
        raise ValueError("Incomplete model roster")
    if not np.isfinite(frame[["prediction", "truth"]].to_numpy(float)).all():
        raise ValueError("Nonfinite KPI predictions or truth")
    if frame.duplicated(["method", *CELL, "sample_index"]).any():
        raise ValueError("Duplicate particles")
    expected = pd.MultiIndex.from_product(
        [KPI_KINDS, sorted(ids), horizons, ("up", "down")], names=CELL)
    reference = None
    rows = []
    for method, group in frame.groupby("method"):
        cells = group.groupby(CELL).agg(truth=("truth", "first"), truth_count=("truth", "nunique"),
                                       particles=("sample_index", "size"))
        if len(expected.difference(cells.index)) or len(cells.index.difference(expected)):
            raise ValueError(f"{method}: missing/extra KPI cells; inspect event coverage before scoring")
        if not cells.truth_count.eq(1).all() or not cells.particles.eq(particles).all():
            raise ValueError(f"{method}: inconsistent truth/particle count")
        if any(set(g.sample_index) != set(range(particles)) for _, g in group.groupby(CELL)):
            raise ValueError(f"{method}: mismatched particle labels")
        truth = cells.reindex(expected).truth.to_numpy()
        if reference is not None:
            np.testing.assert_allclose(truth, reference, rtol=1e-6, atol=1e-6)
        reference = truth
        rows.append(dict(method=method, buildings=len(ids), cells=len(cells), particles=particles))
    return pd.DataFrame(rows)


def calibrated_branches(source, target_calibration, target_test, settings):
    if set(source.profile_id) & set(target_test.profile_id):
        raise ValueError("Source calibration overlaps target test")
    if set(target_calibration.profile_id) & set(target_test.profile_id):
        raise ValueError("Target calibration overlaps target test")
    result = {"raw": target_test}
    required = set(map(tuple, target_test[list(GROUP_COLUMNS)].drop_duplicates().to_numpy()))
    for label, calibration in (("source_rns", source), ("target_rns", target_calibration)):
        available = set(map(tuple, calibration[list(GROUP_COLUMNS)].drop_duplicates().to_numpy()))
        if required != available:
            raise ValueError("Calibration/test KPI group mismatch")
        rns, _, _, _ = calibrate_distribution_table(calibration, target_test, **settings)
        result[label] = rns
        result[label + "_ecc"] = ecc_reorder(target_test, rns)
    return result


def report(args, recipe, split):
    reports = args.output / "reports"
    reports.mkdir(exist_ok=True)
    particles = int(recipe["evaluation"]["num_scenarios"])
    source_cal, source_test = (load_source(args, fold) for fold in ("calibration", "test"))
    target_cal, target_test = (load_target(args, fold) for fold in ("calibration", "test"))
    audits = []
    for label, frame, ids in (("source_calibration", source_cal, split.calibration_ids),
                              ("source_test", source_test, split.test_ids),
                              ("target_calibration", target_cal, split.calibration_ids),
                              ("target_test", target_test, split.test_ids)):
        frame.to_csv(reports / f"{label}_distributions.csv", index=False)
        audits.append(validate_distribution(frame, ids, particles).assign(dataset=label))
    audit = pd.concat(audits, ignore_index=True)
    audit.to_csv(reports / "coverage_audit.csv", index=False)
    branches = calibrated_branches(source_cal, target_cal, target_test, recipe["calibration"])
    collections = {name: [] for name in ("macro_metrics", "cell_metrics", "reliability", "buildings", "pit")}
    paired = []
    for name, frame in branches.items():
        validate_distribution(frame, split.test_ids, particles)
        frame.to_csv(reports / f"{name}_test_distributions.csv", index=False)
        scores = native_scores(frame)
        for key, table in zip(collections, scores):
            collections[key].append(table.assign(calibration=name))
        paired.append(paired_crps_bootstrap(scores[3], reference="structured", seed=recipe["seed"])
                      .assign(calibration=name))
    combined = {key: pd.concat(tables, ignore_index=True) for key, tables in collections.items()}
    for name, table in combined.items():
        table.to_csv(reports / f"{name}.csv", index=False)
    pd.concat(paired, ignore_index=True).to_csv(reports / "paired_crps.csv", index=False)
    joint = summarize_joint_distribution_tables(source_cal, branches, kpi_kinds=KPI_KINDS)
    if not joint.profile_count.eq(len(split.test_ids)).all() or not joint.dimension.eq(16).all():
        raise ValueError("Incomplete joint KPI evaluation")
    joint.to_csv(reports / "joint_metrics.csv", index=False)
    source_rns, _, _, _ = calibrate_distribution_table(source_cal, source_test, **recipe["calibration"])
    source_summaries = pd.concat([native_scores(frame)[0].assign(calibration=name)
                                 for name, frame in (("raw", source_test), ("source_rns", source_rns))])
    degradation = combined["macro_metrics"].merge(source_summaries,
        on=["method", "kpi_kind", "calibration"], suffixes=("_target", "_source"), validate="one_to_one")
    degradation["crps_ratio_target_to_source"] = degradation.crps_target / degradation.crps_source
    degradation.to_csv(reports / "source_to_target.csv", index=False)
    annual = []
    for method in METHODS[:-1]:
        destination = args.output / "scoring/test" / method
        annual.append(dict(method=method, **read_json(destination / "summary.json")["metrics"]))
    pd.DataFrame(annual).to_csv(reports / "annual_signal_metrics.csv", index=False)
    links = "".join(f'<li>{escape(method)}: <a href="../../scoring/test/{method}/{p.name}">{p.stem}</a></li>'
                    for method in METHODS[:-1]
                    for p in sorted((args.output / "scoring/test" / method).glob("profile_*.html")))
    # The report is one level below output, so its trace links go up once.
    links = links.replace("../../scoring/", "../scoring/")
    macro = combined["macro_metrics"]
    zero = macro[~macro.calibration.str.startswith("target")]
    adaptation = macro[macro.calibration.str.startswith("target")]
    domain = pd.read_csv(args.output / "domain_difference_summary.csv")
    domain = domain[~domain.field.str.startswith("inputs.controlProfile")]
    (reports / "index.html").write_text(f'''<!doctype html><meta charset="utf-8">
    <title>Frozen tariff-policy transfer</title><style>body{{font:15px system-ui;margin:28px;color:#222}}
    table{{border-collapse:collapse;font-size:12px}}td,th{{padding:7px;border-bottom:1px solid #ddd}}</style>
    <h1>Frozen tariff-policy transfer</h1><p>Same {len(split.test_ids)} test buildings,
    {len(split.calibration_ids)} calibration buildings; {particles} continuous annual particles.
    Structured 0.5h, vanilla LSTM, Markov LSTM and frozen metadata LightGBM. No retraining.</p>
    <p><strong>Not a pure policy intervention:</strong> the older tariff dataset also changes DHW
    configuration. The full field-level audit is in ../domain_differences.csv.</p>
    <h2>Zero-shot transfer: raw and source-policy calibration</h2>{_html_table(zero)}
    <h2>Target-policy adaptation: not zero-shot</h2>{_html_table(adaptation)}
    <p>Target RNS uses only the target calibration fold, never target test labels.</p>
    <h2>Joint 16-KPI scores</h2>{_html_table(joint)}
    <p>All joint scores use source-calibration truth scales. RNS and ECC have identical marginals;
    ECC retains raw particle ranks. LightGBM uses metadata only, with Gaussian marginal ensembles
    sharing quantile indices, not a learned joint copula. Neither RNS nor ECC calibrates traces.</p>
    <h2>Source-to-target degradation</h2>{_html_table(degradation[["method", "kpi_kind", "calibration",
        "crps_source", "crps_target", "crps_ratio_target_to_source"]])}
    <h2>Native annual traces and confidence bands</h2><ul>{links}</ul>
    <h2>Non-control metadata differences</h2>{_html_table(domain)}
    <p>Paired building-bootstrap CRPS intervals: paired_crps.csv. Complete cell/particle audit:
    coverage_audit.csv. Per-building PIT: pit.csv. This is one training seed and a previously
    inspected holdout, not a new pristine final test.</p>''')
    write_json(args.output / "completion.json", dict(
        test_buildings=len(split.test_ids), calibration_buildings=len(split.calibration_ids),
        methods=list(METHODS), particles=particles, branches=list(branches),
        report=str((reports / "index.html").resolve()),
        hashes={str(p.resolve()): digest(p) for p in reports.iterdir() if p.is_file()},
    ))
    print(macro.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--lightgbm", type=Path, default=LIGHTGBM)
    parser.add_argument("--dataset", type=Path, default=Path("data/control=setpoint"))
    parser.add_argument("--output", type=Path, default=SOURCE / "tariff_policy_transfer")
    parser.add_argument("--stage", choices=("prepare", "score", "lightgbm", "report", "all"), default="all")
    parser.add_argument("--method", choices=METHODS[:-1])
    parser.add_argument("--fold", choices=("calibration", "test"))
    args = parser.parse_args()
    recipe, split, artifacts, hashes = context(args)
    if args.stage in ("prepare", "all"):
        prepare(args, recipe, split, artifacts, hashes)
    else:
        request = read_json(args.output / "request.json")
        verify_hashes(request["hashes"])
        for name in ("source", "lightgbm", "dataset"):
            if request[name] != str(getattr(args, name).resolve()):
                raise ValueError(f"Frozen {name} mismatch")
    if args.stage == "score":
        if args.method is None or args.fold is None:
            parser.error("score requires --method and --fold")
        score_fold(args, recipe, split, artifacts)
    if args.stage in ("lightgbm", "all"):
        _predict_lightgbm(benchmark_dir=args.lightgbm, dataset=args.dataset, output_dir=args.output,
                         recipe=recipe, split=split, source_artifact=artifacts["structured"],
                         num_scenarios=recipe["evaluation"]["num_scenarios"], resume=True)
    if args.stage == "all":
        for method in METHODS[:-1]:
            for fold in ("test", "calibration"):
                command = [sys.executable, "-u", "-m", __spec__.name, "--stage", "score",
                           "--method", method, "--fold", fold]
                for name in ("source", "lightgbm", "dataset", "output"):
                    command += ["--" + name, str(getattr(args, name))]
                with (args.output / f"{method}_{fold}.log").open("a") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                                   env={**os.environ, "PYTHONUNBUFFERED": "1"})
                print(f"completed {method}:{fold}", flush=True)
    if args.stage in ("report", "all"):
        report(args, recipe, split)


if __name__ == "__main__":
    main()
