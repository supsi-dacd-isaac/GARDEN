"""Native KPI comparison and building-bootstrap contrasts for plant experiments."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .calibrated_model_benchmark import (
    _emulator_distribution, _force_off_emulator_distribution, _summarize_raw_distribution_table,
)
from .plant_accounting_experiment import BASE, OUTPUT
from .buffer_recharge_report import paired_bootstrap as paired_event_bootstrap


def paired_crps_bootstrap(buildings, *, reference="baseline", seed=13):
    keys = ["kpi_kind", "profile_id", "horizon_hours", "direction"]
    baseline = buildings[buildings.method == reference]
    rows = []
    rng = np.random.default_rng(seed)
    for method, candidate in buildings[buildings.method != reference].groupby("method"):
        joined = candidate.merge(baseline, on=keys, suffixes=("_candidate", "_base"), validate="one_to_one")
        np.testing.assert_allclose(joined.truth_candidate, joined.truth_base, atol=1e-6, rtol=1e-6)
        for kind, group in joined.groupby("kpi_kind"):
            differences = (group.crps_candidate - group.crps_base).groupby(group.profile_id).mean().to_numpy()
            draws = rng.choice(differences, (2000, len(differences)), replace=True).mean(axis=1)
            rows.append({"method": method, "reference": reference, "kpi_kind": kind, "matched_buildings": len(differences),
                         "matched_cells": len(group), "baseline_cells": int((baseline.kpi_kind == kind).sum()),
                         "candidate_cells": int((candidate.kpi_kind == kind).sum()),
                         "crps_difference": differences.mean(),
                         "lower95": np.quantile(draws, .025), "upper95": np.quantile(draws, .975)})
    return pd.DataFrame(rows)


def recovery_event_diagnostics(events, training_events=None):
    """Check first-hour recovery selectivity using observed diagnostic labels."""
    if "dhw_first_hour_probability" not in events:
        return pd.DataFrame()
    rows = []
    for subset, frame in (
        ("all_eligible", events),
        ("initially_heating", events[events.initially_heating_near_setpoint]),
    ):
        prior = None
        if training_events is not None:
            training = (training_events if subset == "all_eligible" else
                        training_events[training_events.initially_heating_near_setpoint])
            if not training.empty:
                prior = float(training.dhw_first_hour.mean())
        for regime, group in (
            ("all", frame), ("dhw", frame[frame.dhw_first_hour]),
            ("no_dhw", frame[~frame.dhw_first_hour]),
        ):
            if group.empty:
                continue
            probability = group.dhw_first_hour_probability.to_numpy(float)
            truth = group.dhw_first_hour.to_numpy(float)
            if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
                raise ValueError("DHW event probabilities must be finite and in [0, 1]")
            rows.append(dict(
                subset=subset, regime=regime, events=len(group), buildings=group.building.nunique(),
                observed_recovery_frequency=truth.mean(), predicted_recovery_probability=probability.mean(),
                recovery_brier=np.mean((probability - truth)**2),
                observed_drop_frequency=float((group.observed_drop_c > .5).mean()),
                predicted_drop_probability=group["p_drop_gt_0.5"].mean(),
            ))
            if prior is not None:
                reference_brier = float(np.mean((prior - truth)**2))
                rows[-1].update(
                    training_recovery_frequency=prior, training_climatology_brier=reference_brier,
                    recovery_brier_skill=(1 - rows[-1]["recovery_brier"] / reference_brier
                                          if reference_brier > 0 else np.nan),
                )
    return pd.DataFrame(rows)


def storage_support_diagnostics(annual, observed_events):
    """Compare modeled capacity with observed no-charge heat withdrawals."""
    events = observed_events[observed_events.initially_heating_near_setpoint].merge(
        annual[["building", "capacity_wh_m2"]], on="building", validate="many_to_one",
    )
    rows = []
    for subset, group in (
        ("initially_heating", events),
        ("no_dhw_no_cooling", events[~events.dhw_first_hour & ~events.cooling_before_recharge]),
        ("dhw_drop_gt_0p5c", events[events.dhw_first_hour & events.drop_gt_0p5c]),
    ):
        if group.empty:
            continue
        capacity = group.capacity_wh_m2.to_numpy(float)
        withdrawal = group.withdrawal_to_cooling_or_censor_wh_m2.to_numpy(float)
        water_capacity = group.water_capacity_wh_m2_k.to_numpy(float)
        if (not np.isfinite(capacity).all() or (capacity <= 0).any()
                or not np.isfinite(water_capacity).all() or (water_capacity <= 0).any()
                or not np.isfinite(withdrawal).all()):
            raise ValueError("Storage consistency requires finite withdrawals and positive finite capacities")
        ratio = withdrawal / capacity
        rows.append(dict(
            subset=subset, events=len(group), buildings=group.building.nunique(),
            mean_capacity_wh_m2=capacity.mean(), median_observed_withdrawal_wh_m2=np.median(withdrawal),
            fraction_withdrawal_above_capacity=np.mean(ratio > 1),
            median_withdrawal_capacity_ratio=np.median(ratio), q95_withdrawal_capacity_ratio=np.quantile(ratio, .95),
            median_capacity_equivalent_water_band_k=np.median(capacity / water_capacity),
        ))
    return pd.DataFrame(rows)


def diagnostic_comparisons(output):
    energy, events, contrasts, links, recovery, storage = [], [], [], [], [], []
    frames = {}
    for method in ("baseline", "accounting", "physical", "physical_dhw"):
        root = output / "diagnostics" / method
        if not (root / "event_summary.csv").exists():
            continue
        annual = pd.read_csv(root / "annual_energy.csv")
        row = dict(method=method, buildings=len(annual),
                   source_bias_fraction=annual.source_predicted_wh_m2.sum() / annual.source_observed_wh_m2.sum() - 1,
                   delivery_bias_fraction=annual.building_heat_predicted_wh_m2.sum() / annual.building_heat_observed_wh_m2.sum() - 1,
                   electricity_bias_fraction=annual.electricity_predicted_wh_m2.sum() / annual.electricity_observed_wh_m2.sum() - 1,
                   spill_fraction=annual.overflow_wh_m2.sum() / annual.source_predicted_wh_m2.sum(),
                   loss_fraction=annual.standing_loss_wh_m2.sum() / annual.source_predicted_wh_m2.sum(),
                   source_24h_crps=annual.source_24h_crps.mean(),
                   source_annual_crps=annual.source_annual_crps.mean(),
                   source_annual_coverage90=annual.source_annual_coverage90.mean(),
                   source_annual_width90=annual.source_annual_width90.mean(),
                   delivery_24h_crps=annual.building_heat_24h_crps.mean(),
                   capacity_wh_m2=annual.capacity_wh_m2.mean(),
                   zone_heat_ratio=annual.zone_heat_ratio.mean(),
                   storage_limited_fraction=annual.storage_limited_fraction.mean(),
                   max_balance_error_wh_m2=annual.max_transition_residual_wh_m2.max())
        if "dhw_mode_brier" in annual:
            row.update(dhw_mode_brier=annual.dhw_mode_brier.mean(), dhw_on_fraction=annual.dhw_on_fraction.mean())
        energy.append(row)
        request = json.loads((root / "request.json").read_text())
        observed_path = Path(request["events"])
        if hashlib.sha256(observed_path.read_bytes()).hexdigest() != request["events_hash"]:
            raise ValueError("Observed event source changed after the diagnostic rollout")
        observed = pd.read_csv(observed_path)
        storage_summary = storage_support_diagnostics(annual, observed[observed.split == "test"])
        storage_summary["method"] = method
        storage.append(storage_summary)
        frames[method] = pd.read_csv(root / "ensemble_events.csv")
        recovery_summary = recovery_event_diagnostics(frames[method], observed[observed.split == "train"])
        if not recovery_summary.empty:
            recovery_summary["method"] = method
            recovery.append(recovery_summary)
        summary = pd.read_csv(root / "event_summary.csv")
        summary["method"] = method
        events.append(summary)
        links.extend((method, path.resolve()) for path in sorted(root.glob("profile_*.html")))
    for reference, candidate in (("baseline", "accounting"), ("baseline", "physical"),
                                 ("baseline", "physical_dhw"), ("physical", "physical_dhw")):
        if not {reference, candidate} <= frames.keys():
            continue
        for subset in ("all_eligible", "initially_heating"):
            a, b = frames[reference], frames[candidate]
            if subset == "initially_heating":
                a, b = a[a.initially_heating_near_setpoint], b[b.initially_heating_near_setpoint]
            matched = a.merge(b, on=["building", "event"], suffixes=("_a", "_b"), validate="one_to_one")
            np.testing.assert_allclose(matched.observed_drop_c_a, matched.observed_drop_c_b)
            contrasts.extend(dict(row, method=candidate, reference=reference, subset=subset)
                             for row in paired_event_bootstrap(a, b))
    return (pd.DataFrame(energy), pd.concat(events, ignore_index=True) if events else pd.DataFrame(),
            pd.DataFrame(contrasts), links,
            pd.concat(recovery, ignore_index=True) if recovery else pd.DataFrame(),
            pd.concat(storage, ignore_index=True) if storage else pd.DataFrame())


def comparison_conclusions(summary, contrasts, completion):
    """Draw only conclusions supported by a complete matched KPI comparison."""
    kinds = {"setpoint_f", "setpoint_regression", "force_off_f", "force_off_regression"}
    rows = []
    for status in completion:
        method = status["method"]
        if method == "baseline":
            continue
        row = dict(method=method, decision="incomplete",
                   conclusion="Full matched annual KPI comparison is not yet available.")
        if status["setpoint_scored"] and status["force_off_scored"] and not contrasts.empty:
            paired = contrasts[(contrasts.method == method) & (contrasts.reference == "baseline")]
            scores = summary[summary.method == method]
            matched = (set(paired.kpi_kind) == kinds and len(paired) == len(kinds)
                       and set(scores.kpi_kind) == kinds
                       and (paired.matched_cells == paired.baseline_cells).all()
                       and (paired.matched_cells == paired.candidate_cells).all()
                       and (paired.matched_buildings == status["expected_test_buildings"]).all())
            if matched:
                if (paired.lower95 > 0).all():
                    row.update(decision="do_not_promote", conclusion=(
                        "CRPS is worse in all four KPI groups; every paired building-bootstrap interval "
                        "excludes zero. Retain the frozen baseline."))
                else:
                    row.update(decision="review", conclusion=(
                        "Matched KPI scores are available. Inspect coverage, bias, interval width and "
                        "energy/cooling diagnostics before deciding whether to replace the baseline."))
        if not status["diagnostics_complete"]:
            row["conclusion"] += " The energy/cooling-event audit remains incomplete."
        rows.append(row)
    return pd.DataFrame(rows)


def run(base_root, output):
    frames, status, trace_links = [], [], []
    expected_ids = set(json.loads((base_root / "split_manifest.json").read_text())["test_ids"])
    for method in ("baseline", "accounting", "physical", "physical_dhw"):
        root = base_root if method == "baseline" else output / "variants" / method
        sp = root / "scoring" / "test"
        if method == "baseline":
            sp /= "probabilistic_structured"
        off = root / "force_off_policy" / "test"
        diagnostics = output / "diagnostics" / method
        annual_path = diagnostics / "annual_energy.csv"
        diagnostic_ids = set(pd.read_csv(annual_path).building) if annual_path.exists() else set()
        row = {"method": method, "setpoint_scored": False, "force_off_scored": False,
               "expected_test_buildings": len(expected_ids), "diagnosed_buildings": len(diagnostic_ids),
               "diagnostics_complete": (diagnostic_ids == expected_ids
                                        and (diagnostics / "event_summary.csv").exists())}
        if (sp / "summary.json").exists():
            for kind in ("regression", "f"):
                data = _emulator_distribution(sp, method=method, kpi_kind=kind)
                data["kpi_kind"] = f"setpoint_{kind}"
                frames.append(data)
            row["setpoint_scored"] = True
            trace_links.extend((method, "setpoint", path.resolve()) for path in sorted(sp.glob("profile_*.html")))
        if (off / "force_off_policy_summary.json").exists():
            for kind, estimator in (("regression", "regression"), ("f", "direct_ratio")):
                frames.append(_force_off_emulator_distribution(
                    off, method=method, source_model="structured_gated", estimator=estimator, kpi_kind=f"force_off_{kind}"))
            row["force_off_scored"] = True
            trace_links.extend((method, "force_off", path.resolve()) for path in sorted((off / "probabilistic_traces").glob("*structured_gated.html")))
        record = root / "artifact.json"
        artifact = None
        if record.exists():
            artifact = Path(json.loads(record.read_text())["artifact_dir"])
        elif method == "baseline" and (root / "artifact_paths.json").exists():
            artifact = Path(json.loads((root / "artifact_paths.json").read_text())["probabilistic_structured"])
        if artifact is not None:
            metadata = json.loads((artifact / "metadata_closed_loop_hp_contracting_probabilistic.json").read_text())
            row.update(checkpoint_epoch=metadata["checkpoint_epoch"],
                       validation_nrmse=metadata["checkpoint_metric_value"],
                       metadata_dimensions=metadata["metadata_dim"])
        status.append(row)
    if not frames:
        raise ValueError("No completed scoring outputs")
    distribution = pd.concat(frames, ignore_index=True)
    metrics, reliability, buildings = _summarize_raw_distribution_table(distribution)
    rows = []
    for key, group in distribution.groupby(["method", "kpi_kind", "profile_id", "horizon_hours", "direction"]):
        truth, samples = float(group.truth.iloc[0]), group.prediction.to_numpy(float)
        samples = samples[np.isfinite(samples)]
        rows.append(dict(zip(["method", "kpi_kind", "profile_id", "horizon_hours", "direction"], key),
                         particles=len(samples), pit_mid=float(np.mean(samples < truth) + .5 * np.mean(samples == truth)),
                         sample_variance=float(np.var(samples))))
    pit = pd.DataFrame(rows)
    summary = metrics.groupby(["method", "kpi_kind"])[[
        "crps", "bias", "mae", "coverage_90", "width_90", "calibration_mae"]].mean().reset_index()
    summary = summary.merge(pit.groupby(["method", "kpi_kind"]).agg(
        valid_cells=("profile_id", "size"), minimum_particles=("particles", "min"), mean_pit=("pit_mid", "mean")),
        on=["method", "kpi_kind"], validate="one_to_one")
    contrasts = paired_crps_bootstrap(buildings)
    if {"physical", "physical_dhw"} <= set(buildings.method):
        contrasts = pd.concat([contrasts, paired_crps_bootstrap(
            buildings[buildings.method.isin(["physical", "physical_dhw"])], reference="physical")], ignore_index=True)
    energy, event_summary, event_contrasts, diagnostic_links, recovery, storage = diagnostic_comparisons(output)
    dhw_source_path = output / "dhw_source_diagnostics" / "summary.csv"
    dhw_source = pd.read_csv(dhw_source_path) if dhw_source_path.exists() else pd.DataFrame()
    conclusions = comparison_conclusions(summary, contrasts, status)
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    for name, frame in (("summary", summary), ("kpi_cells", metrics), ("reliability", reliability),
                        ("building_kpis", buildings), ("pit", pit), ("paired_crps", contrasts),
                        ("completion", pd.DataFrame(status)), ("energy_accounting", energy),
                        ("cooling_events", event_summary), ("paired_cooling_events", event_contrasts),
                        ("recovery_events", recovery),
                        ("storage_support", storage),
                        ("dhw_source", dhw_source),
                        ("conclusions", conclusions)):
        frame.to_csv(reports / f"{name}.csv", index=False)
    table = lambda frame: frame.to_html(index=False, float_format=lambda x: f"{x:.4f}", border=0)
    review_path = output / "final_review.json"
    review_html = ""
    if review_path.exists():
        review = json.loads(review_path.read_text())
        if review["summary_sha256"] != hashlib.sha256((reports / "summary.csv").read_bytes()).hexdigest():
            raise ValueError("Final review no longer matches the scored comparison")
        review_html = "<h2>Final review</h2>" + "".join(
            f"<p>{html.escape(paragraph)}</p>" for paragraph in review["paragraphs"])
    links = "".join(f'<li>{html.escape(method)} / {policy}: <a href="{html.escape(str(path))}">{html.escape(path.name)}</a></li>'
                    for method, policy, path in trace_links)
    links += "".join(f'<li>{html.escape(method)} / plant diagnostics: <a href="{html.escape(str(path))}">{html.escape(path.name)}</a></li>'
                     for method, path in diagnostic_links)
    complete = all(row["setpoint_scored"] and row["force_off_scored"] and row["diagnostics_complete"] for row in status)
    compute_path = output / "compute_state.json"
    compute = json.loads(compute_path.read_text()) if compute_path.exists() else {}
    compute_notice = (f'<p><strong>Compute status: {html.escape(compute["status"])}</strong><br>'
                      f'{html.escape(compute.get("reason", ""))}</p>' if compute else "")
    comparison_scope = ("All planned comparisons are present; use the physical and DHW contrasts below to assess "
                        "their effects." if complete else
                        "Unfinished physical-buffer or DHW contrasts cannot yet establish whether those changes help.")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Plant accounting comparison</title>
    <style>body{{font:15px system-ui;margin:32px;color:#222}}table{{border-collapse:collapse;font-size:13px}}
    td,th{{padding:7px;border-bottom:1px solid #ddd}}th{{text-align:left}}</style></head><body>
    <h1>Plant accounting: native-ensemble comparison</h1>
    <p><strong>{'All scoring and diagnostic stages present' if complete else 'Incomplete: missing scores and diagnostic stages are not results'}</strong></p>
    {compute_notice}
    <p>Controlled experiment settings: the existing 140/20 training/validation building split, ten epochs and the original validation-NRMSE
    checkpoint selection. Twenty held-out buildings are scored using 100 continuous annual particles and the same seeds.
    No RNS, ECC or trajectory reweighting. Initial indoor temperature is supplied exactly as in the baseline; subsequent simulator
    temperatures, heating powers and plant states are not inference inputs. These test buildings have been used in prior research,
    so this is a controlled research comparison, not a pristine final holdout.</p>
    <h2>Score-based screening</h2><p>These automatic rules use matched KPI contrasts. The final review below also considers
    energy accounting, cooling events and recovery selectivity.</p>{table(conclusions)}
    {review_html}
    <p>A negative result for accounting-only does not invalidate the measured source/area mismatch. It shows that these bundled
    corrections and loss weights do not improve the emulator by themselves. Compensating errors are a plausible explanation,
    not an identified mechanism. {comparison_scope}</p>
    <h2>Completion</h2>{table(pd.DataFrame(status))}
    <h2>Native flexibility scores</h2><p>Lower CRPS is better. Coverage must be read with interval width and location bias.
    Values average KPI cells; the four groups have different definitions/units and are not pooled into a single score.</p>{table(summary)}
    <h2>Paired CRPS differences against baseline</h2><p>Candidate minus baseline, matched building/horizon/direction cells.
    Bootstrap resamples buildings, retaining all their KPI coordinates together. Missing cells are reported, not counted as improvements.</p>{table(contrasts)}
    <h2>What each arm changes</h2><ul><li>Baseline: frozen best 0.5h-cap checkpoint.</li>
    <li>Accounting: learned zone/building heat ratio, whole-building withdrawals, and 4/24-hour source/delivery CRPS supervision.</li>
    <li>Physical: additionally uses volume-based capacity, post-delivery storage capping, and known DHW/design-supply metadata.</li>
    <li>Physical + DHW: same physical representation and metadata, plus causal DHW deficit/recovery and mode supervision.</li></ul>
    <p>The physical versus accounting contrast bundles capacity, throughput and metadata; it cannot identify their separate effects.
    The physical + DHW contrast isolates the additional recovery mechanism and its supervision. The DHW controller is a surrogate
    hypothesis, not a claimed reconstruction of unretained EnergyPlus tank-node logic.</p>
    <h2>Energy accounting</h2><p>Continuous annual native rollouts. Powers and stored energy use whole-building area except the
    explicitly labeled zone output. Source is compared with the retained signed SH charging channel; it is not assumed to be a
    precise compressor-COP measurement. Recorded building heat can exceed this charging channel; unretained boundary flows such as
    pump heat may contribute. The two supervision targets are therefore soft constraints. CRPS and interval widths are in Wh/m2.
    No observed heat or operating mode feeds the rollout. Active DHW labels require the mode flag and HP electricity above 10 W,
    since the exported mode flag alone stays latched during some forced-off intervals.</p>{table(energy)}
    <h2>Capacity consistency check</h2><p>Observed whole-building heat delivered during the no-charge episode, starting before
    the interruption when applicable and ending at a 0.2 C cooling threshold or recharge/censoring. All rows require initial heating.
    The no-DHW/no-cooling subset has no active DHW in the first hour after release and no 0.2 C crossing before recharge.
    Means and quantiles weight events. Withdrawal above modeled capacity warns that the available-heat representation may be
    too restrictive, but does not measure tank capacity: emitter/distribution storage, pump heat and other unrecorded boundary
    flows can contribute. The equivalent water-temperature band is an energy conversion, not an observed tank temperature.
    These observed quantities are diagnostic only and do not feed inference.</p>{table(storage)}
    <h2>DHW power-anchor check</h2><p>Read-only source-channel audit on the original training and test splits, under force-off
    operation. The recovery surrogate bounds its effective power at 0.5 to 1.5 times metadata nominal DHW recovery power.
    Observed charging can exceed that range. This does not alone prove predictive failure because the draw-energy conversion
    is also learned, but it prevents interpreting those fitted parameters as uniquely identified tank physics. Charging-channel
    power is not a direct compressor-COP measurement. This audit does not alter the experiment settings.</p>{table(dhw_source)}
    <h2>Cooling events</h2><p>Same complete winter interruption events, retaining both successes and failures. The initially-heating
    subset requires pre-event zone heat at least 5 W/m2, temperature within 0.3 C of setpoint and no pre-event ventilation.
    DHW regime labels are observed outcomes used only for diagnosis. Summary means weight events; paired contrasts weight buildings equally.</p>{table(event_summary)}
    <h2>Paired cooling-event differences</h2><p>Candidate minus reference; bootstrap resamples buildings. Negative CRPS/Brier differences
    are better. Coverage differences are not automatically improvements and must be read with width.</p>{table(event_contrasts)}
    <h2>Recovery-event selectivity</h2><p>Probability of any active DHW recovery in the first hour after release. This table is
    available only for completed models with an explicit recovery state. Means weight events; the observed DHW/no-DHW split
    is diagnostic only, not an inference input. Drop probabilities refer to cooling above 0.5 C. Matching the aggregate recovery
    frequency alone is insufficient if recovery is assigned to the wrong events. Brier skill compares with a constant recovery
    probability estimated from matching training events (all eligible or initially heating), never from test outcomes.</p>{table(recovery)}
    <h2>Annual confidence-band traces</h2><ul>{links}</ul></body></html>"""
    (reports / "index.html").write_text(page)
    print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, default=BASE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.base_root, args.output_dir)


if __name__ == "__main__":
    main()
