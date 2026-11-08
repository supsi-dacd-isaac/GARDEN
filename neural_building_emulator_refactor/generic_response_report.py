"""Matched native annual KPI and event audit for the generic response-loss screen."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .buffer_recharge_report import paired_bootstrap
from .calibrated_model_benchmark import (
    _emulator_distribution, _force_off_emulator_distribution, _summarize_raw_distribution_table,
)
from .generic_response_experiment import BASE, OUTPUT, VARIANT, fingerprints
from .plant_accounting_report import paired_crps_bootstrap
from .routed_latent_targeted_loss_ablation import _load_context
from neural_building_emulator.train import TrainConfig


BASE_DIAGNOSTICS = Path(
    "output/neural_building_emulator_refactor/plant_accounting_current_best_200/diagnostics/baseline"
)


def controlled_config_changes(baseline, candidate):
    for key in ("train_ids", "test_ids", "metadata_columns", "input_columns"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"Uncontrolled change in {key}")
    defaults = json.loads(json.dumps(asdict(TrainConfig()), default=str))
    before = {**defaults, **baseline["train_config"]}
    after = {**defaults, **candidate["train_config"]}
    rows = []
    for key in sorted(before.keys() | after.keys()):
        if before.get(key) == after.get(key) or key in ("output_dir", "model_checkpoint_dir"):
            continue
        if not key.startswith(("prob_event_energy_", "prob_block_energy_")):
            raise ValueError(f"Uncontrolled non-objective change: {key}")
        rows.append({"setting": key, "baseline": json.dumps(before.get(key)),
                     "candidate": json.dumps(after.get(key))})
    return pd.DataFrame(rows)


def native_scores(distribution):
    metrics, reliability, buildings = _summarize_raw_distribution_table(distribution)
    rows = []
    keys = ["method", "kpi_kind", "profile_id", "horizon_hours", "direction"]
    for key, group in distribution.groupby(keys):
        if group.truth.nunique() != 1:
            raise ValueError("A KPI cell must have one observed truth")
        samples = group.prediction.to_numpy(float)
        if not np.isfinite(samples).all() or len(samples) < 2:
            raise ValueError("Native scoring requires finite ensembles with at least two particles")
        truth = float(group.truth.iloc[0])
        rows.append(dict(zip(keys, key), particles=len(samples),
                         pit=float(np.mean(samples < truth) + .5 * np.mean(samples == truth)),
                         sample_variance=float(np.var(samples, ddof=1))))
    pit = pd.DataFrame(rows)
    summary = metrics.groupby(["method", "kpi_kind"])[
        ["crps", "bias", "mae", "coverage_90", "width_90", "calibration_mae"]
    ].mean().reset_index()
    summary = summary.merge(pit.groupby(["method", "kpi_kind"]).agg(
        valid_cells=("profile_id", "size"), minimum_particles=("particles", "min"),
        mean_pit=("pit", "mean")), on=["method", "kpi_kind"], validate="one_to_one")
    return summary, metrics, reliability, buildings, pit


def verify_matched_events(baseline, candidate):
    keys = ["building", "event"]
    if baseline.duplicated(keys).any() or candidate.duplicated(keys).any():
        raise ValueError("Duplicate event keys")
    joined = baseline.merge(candidate, on=keys, how="outer",
                            suffixes=("_base", "_candidate"), indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("Incomplete event matching")
    np.testing.assert_allclose(joined.observed_drop_c_base, joined.observed_drop_c_candidate)
    for column in ("initially_heating_near_setpoint", "dhw_first_hour"):
        np.testing.assert_array_equal(joined[column + "_base"], joined[column + "_candidate"])


def run(base_root, output, baseline_diagnostics):
    _, split, baseline_artifact = _load_context(base_root)
    candidate_root = output / "variants" / VARIANT
    request = json.loads((candidate_root / "experiment_request.json").read_text())
    if request["baseline_hashes"] != fingerprints(baseline_artifact):
        raise ValueError("Baseline checkpoint changed after the experiment was specified")
    candidate_artifact = Path(json.loads((candidate_root / "artifact.json").read_text())["artifact_dir"])
    if json.loads((candidate_root / "scoring_checkpoint.json").read_text()) != fingerprints(candidate_artifact):
        raise ValueError("Candidate checkpoint changed after scoring")
    metadata_name = "metadata_closed_loop_hp_contracting_probabilistic.json"
    configuration = controlled_config_changes(
        json.loads((baseline_artifact / metadata_name).read_text()),
        json.loads((candidate_artifact / metadata_name).read_text()),
    )
    frames, traces = [], []
    for method, root in (("baseline", base_root), (VARIANT, candidate_root)):
        sp = root / "scoring" / "test"
        if method == "baseline":
            sp /= "probabilistic_structured"
        off = root / "force_off_policy" / "test"
        for path in (sp / "summary.json", off / "force_off_policy_summary.json"):
            if not path.exists():
                raise FileNotFoundError(f"Incomplete comparison: {path}")
        for kind in ("regression", "f"):
            frame = _emulator_distribution(sp, method=method, kpi_kind=kind)
            frame["kpi_kind"] = f"setpoint_{kind}"
            frames.append(frame)
            frames.append(_force_off_emulator_distribution(
                off, method=method, source_model="structured_gated",
                estimator="regression" if kind == "regression" else "direct_ratio",
                kpi_kind=f"force_off_{kind}",
            ))
        traces.extend((method, p) for p in sorted(sp.glob("profile_*.html")))
        traces.extend((method, p) for p in sorted((off / "probabilistic_traces").glob("*structured_gated.html")))
    distribution = pd.concat(frames, ignore_index=True)
    summary, metrics, reliability, buildings, pit = native_scores(distribution)
    paired = paired_crps_bootstrap(buildings)
    if (len(paired) != 4 or not (paired.matched_cells == paired.baseline_cells).all()
            or not (paired.matched_cells == paired.candidate_cells).all()
            or not (paired.matched_buildings == len(split.test_ids)).all()):
        raise ValueError("Incomplete paired KPI comparison; inspect missing cells")
    expected_particles = int(request["recipe"]["evaluation"]["num_scenarios"])
    if not pit.particles.eq(expected_particles).all():
        raise ValueError("Unequal or incomplete annual particle counts")
    annual, summaries, events = [], [], {}
    for method, root in (("baseline", baseline_diagnostics),
                         (VARIANT, output / "diagnostics" / VARIANT)):
        audit = json.loads((root / "request.json").read_text())
        observed = Path(audit["events"])
        if hashlib.sha256(observed.read_bytes()).hexdigest() != audit["events_hash"]:
            raise ValueError("Observed diagnostic event source changed")
        frame = pd.read_csv(root / "annual_energy.csv")
        if set(frame.building) != set(split.test_ids):
            raise ValueError("Incomplete annual diagnostics")
        frame["method"] = method
        annual.append(frame)
        frame = pd.read_csv(root / "event_summary.csv")
        frame["method"] = method
        summaries.append(frame)
        events[method] = pd.read_csv(root / "ensemble_events.csv")
        traces.extend((method, p) for p in sorted(root.glob("profile_*.html")))
    verify_matched_events(events["baseline"], events[VARIANT])
    event_contrasts = []
    for subset in ("all_eligible", "initially_heating"):
        a, b = events["baseline"], events[VARIANT]
        if subset == "initially_heating":
            a, b = a[a.initially_heating_near_setpoint], b[b.initially_heating_near_setpoint]
        event_contrasts.extend(dict(row, subset=subset) for row in paired_bootstrap(a, b))
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    event_summary = pd.concat(summaries, ignore_index=True)
    annual = pd.concat(annual, ignore_index=True)
    contrasts = pd.DataFrame(event_contrasts)
    for name, frame in (
        ("summary", summary), ("kpi_cells", metrics), ("reliability", reliability),
        ("building_kpis", buildings), ("pit", pit), ("paired_crps", paired),
        ("annual_diagnostics", annual), ("cooling_events", event_summary),
        ("paired_cooling_events", contrasts),
        ("configuration_changes", configuration),
    ):
        frame.to_csv(reports / f"{name}.csv", index=False)
    table = lambda frame: frame.to_html(index=False, border=0, float_format=lambda x: f"{x:.4f}")
    review_html = ""
    review_path = output / "final_review.json"
    if review_path.exists():
        review = json.loads(review_path.read_text())
        if review["summary_sha256"] != hashlib.sha256((reports / "summary.csv").read_bytes()).hexdigest():
            raise ValueError("Final review is stale relative to the scored comparison")
        review_html = "<h2>Conclusion</h2>" + "".join(
            f"<p>{html.escape(paragraph)}</p>" for paragraph in review["paragraphs"])
    links = "".join(f'<li>{html.escape(method)}: <a href="{html.escape(str(p.resolve()))}">'
                    f'{html.escape(p.name)}</a></li>' for method, p in traces)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Generic response comparison</title>
    <style>body{{font:15px system-ui;margin:32px;color:#222}}table{{border-collapse:collapse;font-size:13px}}
    td,th{{padding:7px;border-bottom:1px solid #ddd}}th{{text-align:left}}</style></head><body>
    <h1>Generic emulator: loss-only comparison</h1>
    <p>Frozen effective-storage baseline versus the same architecture with joint T/Q/P response scores at
    setpoint changes and force-off command onset/release, plus 6/24/72/240-hour block-average scores.
    Both added terms have weight 1. This comparison bundles two objective terms; it cannot isolate them.
    No new dynamic states, tank rules, observed plant labels, latent dimensions or capacity constraints.</p>
    <p>Same 140 training and 20 validation buildings, ten epochs, validation-NRMSE checkpoint selection,
    and 20 research-held-out test buildings. One hundred native continuous annual particles per policy,
    matched seeds. No RNS/ECC/reweighting. These are not pristine final-test buildings.
    Training still uses ten-day windows and truncated backpropagation; block scoring is not an annual
    training objective and does not restore gradients across detached state boundaries.</p>
    {review_html}
    <h2>Verified configuration differences</h2>{table(configuration)}
    <h2>Native annual KPI scores</h2><p>Lower CRPS is better. Check coverage together with width and bias.
    Do not pool groups with different units. PIT is summarized by its mean here; full per-cell PIT and
    reliability curves are saved in CSV. A mean PIT of 0.5 alone does not establish calibration.</p>{table(summary)}
    <h2>Paired building-bootstrap differences</h2><p>Candidate minus baseline. All cells and particles
    matched. Buildings, not individual events or horizons, are resampled.</p>{table(paired)}
    <h2>Cooling-event audit</h2><p>Includes successful interruptions as well as failures.
    Observed DHW labels are diagnostic strata only, never inference inputs.
    Read event-weighted summaries alongside building-bootstrap contrasts.</p>{table(event_summary)}
    <h2>Paired event differences</h2>{table(contrasts)}
    <h2>Annual confidence-band plots</h2><ul>{links}</ul></body></html>"""
    (reports / "index.html").write_text(page)
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, default=BASE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--baseline-diagnostics", type=Path, default=BASE_DIAGNOSTICS)
    args = parser.parse_args()
    run(args.base_root, args.output_dir, args.baseline_diagnostics)


if __name__ == "__main__":
    main()
